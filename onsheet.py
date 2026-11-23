"""hecate-onsheet: run hecate (frozen) and the on-sheet fine-tune S on one window of a public PHerc. Paris 4 segment,
rendered with VC3D's vc_render_tifxyz, and score both 2D ink maps against the organisers' 20260918 ink labels.

Everything is read from public sources: the CT, the segment mesh and the ink labels from the Vesuvius Challenge
open-data bucket, hecate from Hugging Face. The S checkpoint is a file you give with --s-weights.

  python onsheet.py prep   --out DIR                 mesh + labels -> DIR/seg (9.6 um tifxyz), DIR/labels.npz
  python onsheet.py fetch  --out DIR                 the CT chunks the window can touch -> DIR/ct.zarr (level 2 only)
  python onsheet.py render --out DIR                 vc_render_tifxyz, 16 layers -> DIR/render.zarr
  python onsheet.py infer  --out DIR --model frozen|S [--s-weights PATH] [--device cuda]
  python onsheet.py score  --out DIR                 2D AUC of both maps -> DIR/results.json (+ check vs reference/;
                                                     exit code 1 if the reference window DIFFERS)

run_window.sh runs all five in order. Default window: segment 20231210121321, rows 1200-2800, cols 2800-4400 of
the 9.6 um canvas (15.36 mm square), the window of reference/window.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = json.loads((HERE / 'reference' / 'window.json').read_text())

BUCKET = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'
SCROLL = 'PHercParis4'
VOL_ID = '20260411134726'
CT = f'{BUCKET}/{SCROLL}/volumes/{VOL_ID}-2.400um-0.2m-78keV-masked.zarr'
LEVEL, F = 2, 4                     # pyramid level 2 = 9.6 um; 4 level-0 voxels per level-2 voxel
LABELS = 'ink-labels/2.4um-volume-20260411134726/20260918'
HF = 'https://huggingface.co/scrollprize/hecate/resolve/{rev}/{file}'
HECATE_REV = '9cb86e500e944b11a06a7020403cde5dffb5bcb2'
HECATE_SHA = {'hecate.py': 'c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59',
              'hecate_9.6um.pth': '809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d'}
S_SHA = '4b84fdf2f90cb369b254f04415d474bac8ab087a47022badd2d20fa32205b4f2'
STEP = 20.0                         # canvas px per grid node at 9.6 um (tifxyz scale 0.05, every 4th 2.4 um node)
CH = 128                            # CT chunk edge (level 2)
NSL = 16                            # rendered layers, 1 voxel apart
VC_RENDER_BIN = os.environ.get('VC_RENDER_BIN')                    # a local vc_render_tifxyz binary
VC3D_IMAGE = os.environ.get('VC3D_IMAGE')                          # or a docker image with /opt/vc3d/bin/vc_render_tifxyz
NO_RENDERER = '''no VC3D renderer given. This tree does not ship one and has no default. Set and export one of:
  export VC_RENDER_BIN=/path/to/vc_render_tifxyz    your own VC3D build
  export VC3D_IMAGE=<docker image>                  an image that holds /opt/vc3d/bin/vc_render_tifxyz
vc_render_tifxyz is part of VC3D (github.com/ScrollPrize/villa, folder volume-cartographer/; tested at 47111d1a).'''


def log(*a):
    print(*a, flush=True)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def get(url, tries=6):
    for a in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(2 ** a)
        except Exception:
            time.sleep(2 ** a)
    raise RuntimeError(f'download failed: {url}')


def list_keys(prefix):
    """All object keys under prefix in the public bucket (S3 ListObjectsV2, no auth)."""
    keys, token = [], None
    ns = '{http://s3.amazonaws.com/doc/2006-03-01/}'
    while True:
        q = f'{BUCKET}/?list-type=2&prefix={urllib.request.quote(prefix)}'
        if token:
            q += f'&continuation-token={urllib.request.quote(token)}'
        root = ET.fromstring(get(q))
        keys += [c.find(ns + 'Key').text for c in root.findall(ns + 'Contents')]
        if root.find(ns + 'IsTruncated').text != 'true':
            return keys
        token = root.find(ns + 'NextContinuationToken').text


def download_prefix(prefix, dst):
    for k in list_keys(prefix):
        p = Path(dst) / k[len(prefix):]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(get(f'{BUCKET}/{k}'))


# ---------------------------------------------------------------- geometry (9.6 um tifxyz grid)
def load_grid(d):
    import tifffile
    g = np.stack([tifffile.imread(Path(d) / f'{c}.tif').astype(np.float64) for c in 'xyz'])
    v = np.all(np.isfinite(g), 0) & ~np.all(g == -1, 0) & (g[2] >= 0)
    return g, v


def grid_normals(g, v):
    """Unit normals at nodes from central differences; NaN where undefined."""
    gg = np.where(v[None], g, np.nan)
    dr = np.gradient(gg, axis=1)
    dc = np.gradient(gg, axis=2)
    n = np.cross(np.moveaxis(dr, 0, -1), np.moveaxis(dc, 0, -1))
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.where(ln > 0, ln, np.nan)


def pixel_valid(v, shape):
    """Canvas pixels whose 4 surrounding nodes are all valid (pixel p <-> node (p + 0.5) / STEP)."""
    vv = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    r = np.floor((np.arange(shape[0]) + 0.5) / STEP).astype(int)
    c = np.floor((np.arange(shape[1]) + 0.5) / STEP).astype(int)
    ok_r, ok_c = r < vv.shape[0], c < vv.shape[1]
    out = np.zeros(shape, bool)
    out[np.ix_(ok_r, ok_c)] = vv[np.ix_(r[ok_r], c[ok_c])]
    return out


def at_pixels(g, nrm, rr, cc):
    from scipy import ndimage
    u, w = (rr + 0.5) / STEP, (cc + 0.5) / STEP
    P = np.stack([ndimage.map_coordinates(g[k], [u, w], order=1) for k in range(3)], 1)
    Nn = np.stack([ndimage.map_coordinates(np.nan_to_num(nrm[..., k]), [u, w], order=1) for k in range(3)], 1)
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    return P, Nn


def window(args):
    return args.seg, args.r0, args.c0, args.size


def win_mask(out, args):
    """Scored px of the window: canvas px with 4 valid nodes and a finite position and normal."""
    seg, r0, c0, n = window(args)
    g, v = load_grid(out / 'seg' / seg)
    lab = np.load(out / 'labels.npz')
    valid = pixel_valid(v, lab['pos'].shape)
    rr, cc = np.mgrid[r0:r0 + n, c0:c0 + n]
    P, Nn = at_pixels(g, grid_normals(g, v), rr.ravel(), cc.ravel())
    ok = valid[r0:r0 + n, c0:c0 + n].ravel() & np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
    return ok.reshape(n, n), lab['pos'][r0:r0 + n, c0:c0 + n], lab['neg'][r0:r0 + n, c0:c0 + n]


# ---------------------------------------------------------------- steps
def prep(out, args):
    """Public 2.4 um mesh -> every 4th node at level-2 voxel coordinates (p2 = (p0 - 1.5) / 4, holes stay -1);
    labels: inklabels.zarr/2 and supervision.zarr/2; pos = ink & supervision, neg = supervision & ~ink."""
    import tifffile
    import zarr
    seg = args.seg
    src = out / 'src'
    mesh = f'{SCROLL}/segments/{seg}/mesh/{seg}-on-{VOL_ID}-2.4um.tifxyz/'
    download_prefix(mesh, src / 'mesh')
    meta = json.loads((src / 'mesh' / 'meta.json').read_text())
    assert meta['scale'][0] == 0.05000000074505806, meta['scale']
    g0 = np.stack([tifffile.imread(src / 'mesh' / f'{c}.tif') for c in 'xyz'])
    g = g0[:, ::F, ::F].astype(np.float64)
    hole = np.all(g == -1, 0) | ~np.all(np.isfinite(g), 0)
    g2 = ((g - (F - 1) / 2) / F)
    g2[:, hole] = -1
    g2 = g2.astype(np.float32)
    sd = out / 'seg' / seg
    sd.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate('xyz'):
        tifffile.imwrite(sd / f'{c}.tif', g2[k])
    (sd / 'meta.json').write_text(json.dumps({'type': 'seg', 'uuid': seg, 'format': 'tifxyz',
                                              'scale': [0.05000000074505806, 0.05000000074505806]}))
    a = {}
    for z in ('inklabels', 'supervision'):
        pre = f'{SCROLL}/segments/{seg}/{LABELS}/{z}.zarr/'
        d = src / f'{z}.zarr'
        for k in list_keys(pre):
            rel = k[len(pre):]
            if rel == 'zarr.json' or rel.startswith(f'{LEVEL}/'):
                p = d / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(get(f'{BUCKET}/{k}'))
        a[z] = np.asarray(zarr.open_array(str(d / str(LEVEL)), mode='r')[:]) > 0
    pos, neg = a['inklabels'] & a['supervision'], a['supervision'] & ~a['inklabels']
    np.savez_compressed(out / 'labels.npz', pos=pos, neg=neg)
    got = {c: sha_arr(g2[k]) for k, c in enumerate('xyz')}
    got.update(pos=sha_arr(pos), neg=sha_arr(neg))
    ref = REF['inputs'] if seg == REF['segment'] else {}
    info = dict(segment=seg, grid=list(g2.shape[1:]), canvas=list(pos.shape), pos_px=int(pos.sum()),
                neg_px=int(neg.sum()), array_sha256=got,
                equal_to_reference={k: (got[k] == ref[k]['sha256']) if k in ref else None for k in got})
    (out / 'prep.json').write_text(json.dumps(info, indent=1))
    log(json.dumps(info, indent=1))
    shutil.rmtree(src)


def touched(out, args):
    """Level-2 chunk keys (z, y, x) any sample of the window can read (+-9 vx along the normal, trilinear)."""
    seg, r0, c0, n = window(args)
    g, v = load_grid(out / 'seg' / seg)
    rr, cc = np.mgrid[r0:r0 + n:4, c0:c0 + n:4]
    P, N = at_pixels(g, grid_normals(g, v), rr.ravel(), cc.ravel())
    ok = np.all(np.isfinite(P), 1) & np.all(np.isfinite(N), 1)
    keys = set()
    for t in np.arange(-9.0, 9.01, 0.5):
        q = P[ok] + t * N[ok]
        for f in (np.floor, np.ceil):
            for dd in (-2, 2):
                k = (f(q) + dd).astype(int) // CH
                keys |= set(map(tuple, k[:, ::-1]))           # xyz -> zyx
    return keys


def fetch(out, args):
    """Level 2 of the public CT, re-rooted so that level 0 of DIR/ct.zarr = source level 2 (VC3D then reads the
    9.6 um tifxyz in its own frame); only the chunks the window touches."""
    root = out / 'ct.zarr'
    (root / '0').mkdir(parents=True, exist_ok=True)
    meta = json.loads(get(CT + '/.zattrs'))
    ms = meta['multiscales'][0]
    d2 = dict(ms['datasets'][LEVEL]); d2['path'] = '0'
    ms['datasets'] = [d2]
    ms['name'] = f'{SCROLL} {VOL_ID} re-rooted: 0 = source level {LEVEL} (9.6 um), touched chunks only'
    (root / '.zattrs').write_text(json.dumps(meta, indent=1))
    (root / '.zgroup').write_bytes(get(CT + '/.zgroup'))
    za = get(f'{CT}/{LEVEL}/.zarray')
    (root / '0' / '.zarray').write_bytes(za)
    shp = json.loads(za)['shape']
    keys = sorted(k for k in touched(out, args) if all(0 <= k[i] < -(-shp[i] // CH) for i in range(3)))

    def one(k):
        dst = root / '0' / f'{k[0]}/{k[1]}/{k[2]}'
        if dst.exists():
            return 0, True
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            b = get(f'{CT}/{LEVEL}/{k[0]}/{k[1]}/{k[2]}')
        except urllib.error.HTTPError:                     # 404: empty chunk, zarr reads the fill value 0
            return 0, False
        tmp = dst.with_suffix('.part')
        tmp.write_bytes(b)
        tmp.replace(dst)
        return len(b), True

    T0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, keys))
    info = dict(n_chunks=len(keys), n_missing_remote=sum(not r[1] for r in res),
                gb=round(sum(r[0] for r in res) / 1e9, 2), seconds=round(time.time() - T0, 1))
    (out / 'fetch.json').write_text(json.dumps(info))
    log(json.dumps(info))


def render(out, args):
    """vc_render_tifxyz, 16 layers 1 voxel apart, crop = the window. VC3D layer j = depth +7.5 - j along the mesh
    normal (VC3D steps along -normal), so the stack is flipped before inference (see README)."""
    if not (VC_RENDER_BIN or VC3D_IMAGE):
        sys.exit('onsheet.py render: ' + NO_RENDERER)
    seg, r0, c0, n = window(args)
    dst = out / 'render.zarr'
    assert not dst.exists(), f'exists: {dst}'
    (out / 'vc3dcfg').mkdir(exist_ok=True)
    vargs = ['--volume', str(out / 'ct.zarr'), '--group-idx', '0', '--scale', '1', '--segmentation',
             str(out / 'seg' / seg), '--num-slices', str(NSL), '--slice-step', '1', '--cache-gb', '4',
             '--pyramid', 'false', '--crop-x', str(c0), '--crop-y', str(r0), '--crop-width', str(n),
             '--crop-height', str(n), '--zarr-output', str(out / 'render.zarr.part')]
    if VC_RENDER_BIN:
        cmd = [VC_RENDER_BIN] + vargs
        env = dict(os.environ, VC3D_CONFIG_DIR=str(out / 'vc3dcfg'), OMP_NUM_THREADS='4')
    else:
        cmd = ['docker', 'run', '--rm', '--memory', '8g', '--cpus', '4', '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', f'{out}:{out}', '-e', f'HOME={out}', '-e', f'VC3D_CONFIG_DIR={out}/vc3dcfg',
               '-e', 'OMP_NUM_THREADS=4', '--entrypoint', '/opt/vc3d/bin/vc_render_tifxyz', VC3D_IMAGE] + vargs
        env = None
    with open(out / 'render.log', 'w') as lf:
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, env=env)
    (out / 'render.zarr.part').rename(dst)
    log('render done', dst)


def hecate_file(out, name):
    d = out / 'hecate'
    d.mkdir(exist_ok=True)
    p = d / name
    if not p.exists():
        urllib.request.urlretrieve(HF.format(rev=HECATE_REV, file=name), p)
    assert sha_file(p) == HECATE_SHA[name], f'{name}: sha256 differs from hecate @ {HECATE_REV[:8]}'
    return p


def load_hecate(out):
    import importlib.util
    spec = importlib.util.spec_from_file_location('hecate', hecate_file(out, 'hecate.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['hecate'] = mod
    spec.loader.exec_module(mod)
    return mod


def infer(out, args):
    import torch
    import zarr
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hec = load_hecate(out)
    if args.model == 'frozen':
        ck = hecate_file(out, 'hecate_9.6um.pth')
    else:
        assert args.s_weights, '--s-weights PATH (or S_WEIGHTS) is needed for --model S'
        ck = Path(args.s_weights)
        assert sha_file(ck) == S_SHA, 'S checkpoint sha256 differs from the released one'
    model = hec.load_model(str(ck), args.device)
    vol = np.asarray(zarr.open_group(str(out / 'render.zarr'), mode='r')['0'][:])
    vol = np.ascontiguousarray(vol[::-1])                  # VC3D layer j -> plane 15 - j (outer face first)
    o2 = np.zeros(vol.shape[1:], np.uint8)
    T0 = time.time()
    hec.predict(model, vol, o2, reverse=False, batch_size=4)
    np.save(out / f'map_{args.model}.npy', o2)
    try:
        from PIL import Image
        Image.fromarray(o2).save(out / f'map_{args.model}.png')
    except ImportError:
        pass
    log(args.model, 'map done', round(time.time() - T0, 1), 's')


def auc(pos, neg):
    from scipy import stats
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score(out, args):
    ok, pos, neg = win_mask(out, args)
    tp, tn = pos & ok, neg & ok
    res = dict(segment=args.seg, window=dict(r0=args.r0, c0=args.c0, size=args.size),
               scan=f'{SCROLL} {VOL_ID} level 2 (9.6 um), 78 keV', labels=f'{SCROLL} {LABELS}',
               n_ink_px=int(tp.sum()), n_noink_px=int(tn.sum()), n_scored_px=int(ok.sum()), auc_2d={}, map_sha256={})
    for m in ('frozen', 'S'):
        p = out / f'map_{m}.npy'
        if p.exists():
            mp = np.load(p)
            res['map_sha256'][m] = sha_arr(mp)
            mp = mp.astype(np.float32)
            res['auc_2d'][m] = auc(mp[tp], mp[tn])
    is_ref = (args.seg, args.r0, args.c0, args.size) == (REF['segment'], REF['r0'], REF['c0'], REF['size'])
    if is_ref:
        tol = REF['tolerance']
        chk = {}
        for m, a in res['auc_2d'].items():
            r = REF['stored_auc_2d'][m]
            chk[m] = dict(auc=a, stored=r, diff=a - r, within_tolerance=bool(abs(a - r) <= tol),
                          map_equal_to_stored=res['map_sha256'][m] == REF['stored_map_sha256'][m])
        prep_info = json.loads((out / 'prep.json').read_text())
        res['reference_check'] = dict(tolerance=tol, models=chk,
                                      inputs_equal_to_reference=prep_info['equal_to_reference'],
                                      verdict='REPEATS' if len(chk) == 2 and all(
                                          c['within_tolerance'] for c in chk.values()) else 'DIFFERS')
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))
    if is_ref and res['reference_check']['verdict'] != 'REPEATS':          # the one command must fail on a mismatch
        sys.exit('onsheet.py score: DIFFERS. The reference window needs both maps (frozen, S) within '
                 f'{REF["tolerance"]} AUC of the stored numbers; see reference_check in {out / "results.json"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['prep', 'fetch', 'render', 'infer', 'score'])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seg', default=REF['segment'])
    ap.add_argument('--r0', type=int, default=REF['r0'])
    ap.add_argument('--c0', type=int, default=REF['c0'])
    ap.add_argument('--size', type=int, default=REF['size'])
    ap.add_argument('--model', choices=['frozen', 'S'])
    ap.add_argument('--s-weights', default=os.environ.get('S_WEIGHTS'))
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    {'prep': prep, 'fetch': fetch, 'render': render, 'infer': infer, 'score': score}[a.step](out, a)


if __name__ == '__main__':
    main()
