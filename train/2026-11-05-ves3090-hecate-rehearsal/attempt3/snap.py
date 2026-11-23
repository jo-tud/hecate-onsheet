"""VES-3227 (F-524 attempt 3): the registered sheet-snap rule (README "Snap rule S") and the label-set check.

Snap rule S, per 32-plane training block (raw uint8 x valid mask, mesh t = -15.5 .. 15.5 on planes 0 .. 31):
  1. in-plane Gaussian, sigma SIG_XY = 16 px, normalised over the valid px (ok); then along t, sigma SIG_T = 1 plane;
  2. gradient between adjacent planes, at integer mesh t (-15 .. 15);
  3. e(r, c) = the integer t in -EMAX .. EMAX (EMAX = 5) with the most negative gradient = the outer (+t) falling edge
     of the sheet. Both streams have +t outward (falling edge at +t in the mean profile, README);
  4. per stream (Paris 4 train blocks; 0139 rehearsal blocks): m = round(median of e over the supervised valid px);
     s = clip(e - m, -SMAX, SMAX), SMAX = 4, int8. The global mesh-vs-edge offset m is removed, so only the local
     mesh error moves the label (Paris 4's mesh sits on the edge, 0139's at the sheet centre).
Label J_snap: y = pos px & |mesh t - s(r, c)| <= 1.5 (4 planes, as label J; s = 0 gives label J exactly).

  python snap.py --check       synthetic checks (planted offsets; no data)
  python snap.py labelset      label-set check on the real training blocks -> labelset.json, OUT/snap_*.npz
                               exit 4 = gate S2 or S3 fails (stop, no GPU)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves3227'
NPL = 32
TM = np.arange(NPL) - 15.5                       # mesh t of the 32 block planes
TB = np.arange(-15, 16)                          # mesh t of the 31 between-plane gradients
SIG_XY, SIG_T, EMAX, SMAX = 16.0, 1.0, 5, 4
WIN = np.abs(TB) <= EMAX
S2_MIN_SD = 1.0                                  # gate S2: split-half reliable SD of s on Paris 4 ink px (vx)
S3_MIN_AGREE = 0.80                              # gate S3: shifted-block recovery share (k = -2, +2)


def edge_map(xb, okb):
    """e (int8, H x W) of one block xb (32, H, W) with valid px okb (H, W); 0 where no valid px nearby."""
    w = okb.astype(np.float32)
    den = gaussian_filter(w, SIG_XY)
    sm = np.stack([gaussian_filter(xb[k].astype(np.float32) * w, SIG_XY) for k in range(NPL)])
    sm = sm / np.maximum(den, 1e-6)[None]
    sm = gaussian_filter1d(sm, SIG_T, axis=0, mode='nearest')
    g = np.diff(sm, axis=0)[WIN]
    e = TB[WIN][np.argmin(g, 0)].astype(np.int8)
    e[den < 1e-3] = 0
    return e


def snap_stream(X, SUP, OK, okmask=None):
    """s (n, H, W) int8, m, e for a stream of blocks X (n, 32, H, W) = vol * ok. okmask overrides OK for the edge
    smoothing only (split-half check); m always comes from the supervised valid px."""
    om = OK if okmask is None else okmask
    e = np.stack([edge_map(X[b], om[b]) for b in range(len(X))])
    m = int(np.round(np.median(e[SUP & OK])))
    s = np.clip(e.astype(np.int16) - m, -SMAX, SMAX).astype(np.int8)
    return s, m, e


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def synth(offsets, rng, H=128, noise=6.0):
    """Blocks with a sheet whose outer falling edge sits at mesh t = 1 + offsets[b] (+ a smooth ramp in block 0)."""
    X = np.zeros((len(offsets), NPL, H, H), np.uint8)
    yy, xx = np.mgrid[0:H, 0:H]
    for b, o in enumerate(offsets):
        edge = np.full((H, H), 1.0 + o)
        prof = 70 + 25 / (1 + np.exp((TM[:, None, None] - edge[None]) / 1.2)) \
            - 25 / (1 + np.exp((TM[:, None, None] - edge[None] + 7) / 1.2))
        X[b] = np.clip(prof + rng.normal(0, noise, prof.shape), 1, 255).astype(np.uint8)
    return X


def check():
    ok = {}
    rng = np.random.default_rng(3227)
    offs = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 0, 0]          # median 0 -> m = round(median(1 + o)) = 1
    X = synth(offs, rng)
    OK = np.ones((len(offs),) + X.shape[2:], bool); OK[:, :, :8] = False   # an invalid strip
    X = X * OK[:, None]
    SUP = OK.copy()
    s, m, e = snap_stream(X, SUP, OK)
    ok['global_offset'] = bool(m == 1)
    rec = np.mean([np.mean(s[b][OK[b]] == offs[b]) for b in range(len(offs))])
    ok['planted_offsets_recovered'] = bool(rec >= 0.99)
    # clip: a planted offset of +6 is clipped to SMAX
    X6 = synth([6], rng) * OK[:1, None]
    e6 = edge_map(X6[0], OK[0]); ok['edge_clip_at_emax'] = bool(np.mean(e6[OK[0]] == EMAX) >= 0.99)
    # sign: shifting a block by +k planes moves e by +k
    Xs = np.roll(X[4], 2, axis=0); es = edge_map(Xs, OK[4])
    ok['shift_sign'] = bool(np.mean(es[OK[4]] == e[4][OK[4]] + 2) >= 0.99)
    print(json.dumps(dict(ok, recovered=float(rec))))
    assert all(ok.values()), ok


def split_masks(OK):
    """4 x 4 px tile checkerboard halves of OK (disjoint px sets for the split-half reliability)."""
    H, W = OK.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W]
    par = ((yy // 4 + xx // 4) % 2 == 0)[None]
    return OK & par, OK & ~par


def stream_stats(name, X, POS, SUP, OK):
    s, m, e = snap_stream(X, SUP, OK)
    okA, okB = split_masks(OK)
    sA = np.clip(np.stack([edge_map(X[b], okA[b]) for b in range(len(X))]).astype(np.int16) - m, -SMAX, SMAX)
    sB = np.clip(np.stack([edge_map(X[b], okB[b]) for b in range(len(X))]).astype(np.int16) - m, -SMAX, SMAX)
    sup = SUP & OK
    ink = sup & POS
    r = {}
    for nm, mk in (('sup', sup), ('ink', ink)):
        a, b_ = sA[mk].astype(float), sB[mk].astype(float)
        cov = float(np.mean((a - a.mean()) * (b_ - b_.mean())))
        v = s[mk].astype(float)
        bm = np.array([np.median(s[i][mk[i]]) for i in range(len(s)) if mk[i].any()])
        r[nm] = dict(n_px=int(mk.sum()), mean=float(v.mean()), sd=float(v.std()),
                     split_cov=cov, reliable_sd=float(np.sqrt(max(cov, 0.0))),
                     split_corr=float(np.corrcoef(a, b_)[0, 1]),
                     split_agree_le1=float(np.mean(np.abs(a - b_) <= 1)),
                     share_abs_ge1=float(np.mean(np.abs(v) >= 1)), share_abs_ge2=float(np.mean(np.abs(v) >= 2)),
                     share_clipped=float(np.mean(np.abs(v) >= SMAX)),
                     share_edge_at_emax=float(np.mean(np.abs(e[mk]) >= EMAX)),
                     block_median_sd=float(bm.std()), n_blocks=int(len(bm)),
                     hist={int(k): float(c) for k, c in zip(*np.unique(v, return_counts=True))})
    # S3 (real-data positive control): shift a block by k planes, e must move by k (px with |e + k| <= EMAX - 1)
    agree = {}
    for k in (-2, 2):
        hit = tot = 0
        for b in range(0, len(X), 4):
            ek = edge_map(np.roll(X[b], k, axis=0), OK[b])
            mk = sup[b] & (np.abs(e[b].astype(int) + k) <= EMAX - 1)
            hit += int(np.sum(ek[mk] == e[b][mk] + k)); tot += int(mk.sum())
        agree[str(k)] = hit / max(tot, 1)
    # mean CT profile of ink px aligned on mesh t vs on the snapped sheet (reading)
    prof_mesh = [float(X[:, k][ink].mean()) for k in range(NPL)]
    al = np.zeros(NPL); cnt = np.zeros(NPL)
    for b in range(len(X)):
        mk = ink[b]
        if not mk.any():
            continue
        sv = s[b][mk].astype(int)
        for k in range(NPL):
            kk = k + sv                                      # plane k in snapped t = mesh plane k + s
            g = (kk >= 0) & (kk < NPL)
            al[k] += X[b, kk[g], np.nonzero(mk)[0][g], np.nonzero(mk)[1][g]].astype(float).sum(); cnt[k] += g.sum()
    r['shift_recovery'] = agree
    r['m'] = m
    r['profile_ink_mesh_t'] = prof_mesh
    r['profile_ink_snapped_t'] = (al / np.maximum(cnt, 1)).tolist()
    r['sha_s'] = sha(s)
    np.savez_compressed(OUT / f'snap_{name}.npz', s=s, e=e, m=m)
    return r, s


def labelset():
    sys.path.insert(0, str(HERE.parent))
    from r3090 import load
    FR = load('ves3090_ft_r', HERE.parent / 'ft_r.py')
    FT = FR.FT
    meta = json.loads((FT.DATA / 'prep.json').read_text())
    bw = np.array([blk['w'] for blk in meta['blocks']])
    idx = np.flatnonzero(np.isin(bw, FT.TRAIN_SEGS))
    assert not np.isin(bw[idx], FT.HELD_SEGS).any() and len(idx) == 4 * 48
    tr = np.load(FT.DATA / 'train.npz')
    vol, okp, posp, supp = (tr[k][idx] for k in ('vol', 'ok', 'pos', 'sup'))
    X0, pos0, sup0, ok0, segs0 = FR.load_0139()
    res = dict(rule=dict(SIG_XY=SIG_XY, SIG_T=SIG_T, EMAX=EMAX, SMAX=SMAX), bars=dict(S2_MIN_SD=S2_MIN_SD,
                                                                                    S3_MIN_AGREE=S3_MIN_AGREE))
    OUT.mkdir(parents=True, exist_ok=True)
    res['paris4'], sp = stream_stats('paris4', vol * okp[:, None], posp, supp, okp)
    res['p0139'], s0 = stream_stats('p0139', X0, pos0, sup0, ok0)
    g = dict(S2=res['paris4']['ink']['reliable_sd'] >= S2_MIN_SD,
             S3=min(res['paris4']['shift_recovery'].values()) >= S3_MIN_AGREE)
    res['gates'] = g
    res['verdict'] = 'GO' if all(g.values()) else 'STOP'
    (HERE / 'labelset.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(gates=g, verdict=res['verdict'], p4_ink={k: res['paris4']['ink'][k] for k in
                          ('sd', 'reliable_sd', 'split_corr', 'share_abs_ge2', 'share_clipped')},
                          p4_shift=res['paris4']['shift_recovery'], m=(res['paris4']['m'], res['p0139']['m']),
                          p0139_ink={k: res['p0139']['ink'][k] for k in ('sd', 'reliable_sd', 'split_corr')})))
    return 0 if all(g.values()) else 4


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'labelset':
        sys.exit(labelset())
    check()
