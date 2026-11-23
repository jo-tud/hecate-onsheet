"""VES-3067: fine-tune hecate (9.6 um) from its frozen checkpoint with the label-J 3D target and +-6 vx render-vs-mesh
frame jitter (README "The change"). Everything not named here is F-517's ft.py (imported, not copied).

Input: F-457 training blocks DATA/train.npz on the 4 F-517 train segments (32 planes, mesh t = -15.5 .. 15.5, raw
uint8; 0 outside the valid mask). Per patch a jitter j (integer, uniform -6 .. 6, own rng [3067, seed, 3]) picks the
16-plane hecate window vol[Z0 + j : Z0 + j + 16]; its mesh t is T16 + j. The labels are written in mesh t, so they
move with the CT, not with the window (F-399 `common.masks`):
  supervised voxels = sup px (labelled, F-457 mask, sites excluded) & valid px & |T16 + j| <= 7.5
  arm J (label J):  y = pos px & |T16 + j| <= 1.5   (ink on the shell, 0 on the rest of the slab)
  arm U (label D):  y = pos px on every supervised plane (no sheet constraint; the matched null)
Loss: 3D BCE over the supervised voxels only (mean). The 2D head is not run in training (README "Which heads"):
its parameters get no gradient and AdamW leaves them untouched (asserted). All other parameters train, lr 1e-5,
weight decay 1e-4, BatchNorm in eval mode (frozen statistics), batch 8, 64 x 64 px patches, rot90 / flip in-plane
only, seed 0, 8 000 steps.

  python ft_j.py --check                      synthetic jitter check (no model, no data)
  python ft_j.py --arm J|U [--steps 8000] [--tag ''] [--device cuda]
Writes OUT/<arm><tag>/{hecate_ft.pth (hecate.load_model payload), train.json}.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F517 = HERE.parent / '2026-11-04-ves3034-hecate-ft'
OUT = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves3067'
JIT = 6
SHELL, SLAB = 1.5, 7.5
LR, WD = 1e-5, 1e-4
STEPS = 8000
NO_GRAD_2D = ('canonical.decoder.depth_collapse.attn_conv', 'depth_coordinate_scale')   # 2D-only tensors (F-517 attempt 3)


def load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


FT = load('ves3034_ft', F517 / 'ft.py')
D, Z0, T16, PATCH, BATCH = FT.D, FT.Z0, FT.T16, FT.PATCH, FT.BATCH
assert FT.NPL_IN == 32 and D == 16 and Z0 == 8 and Z0 - JIT >= 0 and Z0 + JIT + D <= FT.NPL_IN


def masks(arm, j):
    """(supervised planes, label planes) of the 16-plane window whose mesh t is T16 + j."""
    t = T16 + j
    sup = np.abs(t) <= SLAB
    tgt = np.abs(t) <= SHELL if arm == 'J' else np.ones(D, bool)
    return sup, tgt


def batch(rng, rngj, arm, X, POS, SUP, OK, centres):
    """BATCH jittered patches centred on supervised px. Returns x (B,1,16,P,P), y3, w3 (float32) and the js.
    The patch rng stream is ft.py's (same draws per step); the jitter draws use rngj only."""
    xs, ys, ws, js = [], [], [], []
    h = PATCH // 2
    for _ in range(BATCH):
        b, r, c = centres[rng.integers(len(centres))]
        r = int(np.clip(r - h, 0, X.shape[2] - PATCH)); c = int(np.clip(c - h, 0, X.shape[3] - PATCH))
        j = int(rngj.integers(-JIT, JIT + 1))
        x = X[b, Z0 + j:Z0 + j + D, r:r + PATCH, c:c + PATCH]
        p2, s2, o2 = (q[b, r:r + PATCH, c:c + PATCH] for q in (POS, SUP, OK))
        sp, tp = masks(arm, j)
        y = tp[:, None, None] & p2[None]
        w = sp[:, None, None] & (s2 & o2)[None]
        a = [x, np.broadcast_to(y, x.shape), np.broadcast_to(w, x.shape)]
        k, fl = int(rng.integers(4)), bool(rng.integers(2))
        a = [np.rot90(q, k, axes=(-2, -1)) for q in a]
        if fl:
            a = [q[..., ::-1] for q in a]
        for lst, q in zip((xs, ys, ws), a):
            lst.append(np.ascontiguousarray(q))
        js.append(j)
    f = lambda L: np.stack(L).astype(np.float32)[:, None]
    return f(xs), f(ys), f(ws), js


def check():
    """F-399 synthetic check: on a stack whose value is its mesh t, every label / supervised voxel of 64 jittered
    batches sits exactly at |t| <= 1.5 / <= 7.5 (arm J), arm U labels every supervised voxel, j covers -6 .. 6, and
    the window planes carry no label information (each window plane is a label plane in some batch and not in
    another)."""
    tm = (np.arange(FT.NPL_IN) - 15.5).astype(np.float32)
    X = np.broadcast_to(tm[None, :, None, None], (2, FT.NPL_IN, 96, 96)).copy() + 100   # value = mesh t + 100
    POS = np.ones((2, 96, 96), bool); SUP = np.ones_like(POS); OK = np.ones_like(POS)
    centres = np.argwhere(SUP)
    ok, js_all, lab_plane = {}, [], np.zeros((2, D), int)
    for arm in ('J', 'U'):
        rng, rngj = np.random.default_rng(0), np.random.default_rng(1)
        good = True
        for _ in range(64):
            x, y, w, js = batch(rng, rngj, arm, X, POS, SUP, OK, centres)
            t = x[:, 0] - 100
            good &= bool(np.all(np.abs(t[w[:, 0] > 0]) <= SLAB) and np.all(np.abs(t[w[:, 0] == 0]) > SLAB))
            lab = (y[:, 0] > 0) & (w[:, 0] > 0)
            if arm == 'J':
                good &= bool(np.all(np.abs(t[lab]) <= SHELL) and np.all(np.abs(t[(w[:, 0] > 0) & ~lab]) > SHELL))
                for bi in range(len(js)):
                    pl = lab[bi].any((1, 2))
                    lab_plane[0][pl] += 1; lab_plane[1][~pl] += 1
            else:
                good &= bool(np.array_equal(lab, w[:, 0] > 0))
            js_all += js
        ok[f'masks_{arm}'] = good
    ok['jitter_range'] = sorted(set(js_all)) == list(range(-JIT, JIT + 1))
    ok['no_plane_prior'] = bool((lab_plane[0] > 0).all() and (lab_plane[1] > 0).all())
    print(json.dumps(ok))
    assert all(ok.values()), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('J', 'U'))
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check or a.arm is None:
        return check()
    import torch
    import torch.nn.functional as F
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    T0 = time.time()
    hec = FT.load_hecate()
    meta = json.loads((FT.DATA / 'prep.json').read_text())
    bw = np.array([blk['w'] for blk in meta['blocks']])
    keep = np.isin(bw, FT.TRAIN_SEGS)
    assert not np.isin(bw[keep], FT.HELD_SEGS).any() and keep.sum() == 4 * 48, keep.sum()
    tr = np.load(FT.DATA / 'train.npz')
    idx = np.flatnonzero(keep)
    vol, ok, pos, sup = (tr[k][idx] for k in ('vol', 'ok', 'pos', 'sup'))
    X = vol * ok[:, None]                                            # raw uint8, 0 outside the mask (ft.py)
    print(f'{len(idx)} train blocks, {time.time() - T0:.0f} s', flush=True)
    stats = dict(sup_px=int(sup.sum()), pos_px=int((pos & sup).sum()), pos_frac_sup=float((pos & sup).sum() / sup.sum()))

    model = hec.load_model(FT.CKPT, 'cpu').to(dev)
    model.requires_grad_(True)
    model.train()
    for mod in model.modules():                                     # frozen BN statistics (F-517)
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    centres = np.argwhere(sup)
    rng = np.random.default_rng([3034, a.seed])                     # ft.py's patch stream
    rngj = np.random.default_rng([3067, a.seed, 3])                 # jitter stream
    log, jcount, ypos = [], np.zeros(2 * JIT + 1, int), []
    for step in range(a.steps):
        x, y3, w3, js = batch(rng, rngj, a.arm, X, pos, sup, ok, centres)
        for j in js:
            jcount[j + JIT] += 1
        x, y3, w3 = (torch.from_numpy(q).to(dev) for q in (x, y3, w3))
        lo3 = model.forward_3d(x / model.divisor)
        l3 = (F.binary_cross_entropy_with_logits(lo3, y3, reduction='none') * w3).sum() / w3.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        l3.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            ypos.append(float((y3 * w3).sum() / w3.sum().clamp_min(1)))
            log.append([step, float(l3), round(time.time() - T0, 1)])
            print(f'{a.arm}{a.tag} step {step} l3 {float(l3):.4f} {time.time() - T0:.0f} s', flush=True)
    od = OUT / f'{a.arm}{a.tag}'
    od.mkdir(parents=True, exist_ok=True)
    payload = torch.load(FT.CKPT, map_location='cpu', weights_only=True)
    ref = payload['ema_model']
    new = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    moved = [k for k in ref if not torch.equal(ref[k], new[k])]
    still_2d = [k for k in ref if k.startswith(NO_GRAD_2D)]
    assert still_2d and not any(k in moved for k in still_2d), [k for k in still_2d if k in moved]
    payload['ema_model'] = new
    payload['ves3067'] = dict(arm=a.arm, steps=a.steps, seed=a.seed, jitter=JIT)
    torch.save(payload, od / 'hecate_ft.pth')
    info = dict(arm=a.arm, tag=a.tag, seed=a.seed, steps=a.steps, batch=BATCH, patch=PATCH, lr=LR, wd=WD, jitter=JIT,
                shell=SHELL, slab=SLAB, n_moved=len(moved), n_tensors=len(ref), unchanged_2d=still_2d,
                jitter_counts=jcount.tolist(), label_frac_logged=ypos, train_segs=FT.TRAIN_SEGS, n_blocks=int(len(idx)),
                stats=stats, loss_log=log, seconds=round(time.time() - T0, 1),
                sec_per_step=round((log[-1][2] - log[0][2]) / max(1, log[-1][0] - log[0][0]), 3) if len(log) > 1 else None,
                device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
                max_mem_gb=round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == 'cuda' else None)
    (od / 'train.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ('loss_log', 'label_frac_logged')}), flush=True)


if __name__ == '__main__':
    main()
