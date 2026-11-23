"""VES-3227 (F-524 attempt 3): F-524 arm R (rehearsal, ft_r.py, imported unchanged) with ONE change: the label shell
sits at the CT-snapped sheet (snap.py rule S), not at mesh t (README "The change"):
  label J (F-522):   y = pos px & |mesh t| <= 1.5
  label J_snap:      y = pos px & |mesh t - s(r, c)| <= 1.5,  s = snap.snap_stream of the block's stream (int, |s| <= 4)
Applies to both streams (Paris 4 patches, ft_j.batch; 0139 rehearsal patches, ft_r.batch_n), each with its own m.
Jitter +-6 vx (F-524's), supervised voxels (|mesh t| <= 7.5, sup & valid px), patch / jitter rng streams, loss, lr,
8 000 steps, N0139 = 4, seed 0: F-524 R's, unchanged. batch_s is ft_r.batch_n with the per-pixel label line; with
s = 0 it equals ft_j.batch / ft_r.batch_n draw for draw (asserted in --check). Checkpoint goes to OUT3227/R/.

  python ft_s.py --check                   synthetic checks (no model, no data)
  python ft_s.py --arm R [--device cuda]   train arm S (= F-524 R with label J_snap); writes OUT3227/R/snap_train.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))
from r3090 import load   # noqa: E402
import snap as SN        # noqa: E402

OUT3227 = SN.OUT
FR = load('ves3090_ft_r', PARENT / 'ft_r.py')
FTJ = FR.FTJ
assert FTJ.JIT == 6 and FR.JIT == 6 and FR.N0139 == 4 and FR.STEPS == 8000 and FTJ.SHELL == 1.5
ORIG_BATCH, ORIG_BATCH_N = FTJ.batch, FR.batch_n
_S = {}          # id(X) -> (X, s, m): the stream's snap map, computed once from the stream's own blocks


def register(X, s, m=None):
    _S[id(X)] = (X, s, m)


def snap_of(X, SUP, OK):
    if id(X) not in _S or _S[id(X)][0] is not X:
        s, m, _ = SN.snap_stream(X, SUP, OK)
        register(X, s, m)
    return _S[id(X)][1]


def batch_s(n, rng, rngj, arm, X, POS, SUP, OK, centres):
    """ft_r.batch_n with the label shell at mesh t = s(r, c) (arm J only)."""
    assert arm == 'J'
    S = snap_of(X, SUP, OK)
    xs, ys, ws, js = [], [], [], []
    h = FR.PATCH // 2
    for _ in range(n):
        b, r, c = centres[rng.integers(len(centres))]
        r = int(np.clip(r - h, 0, X.shape[2] - FR.PATCH)); c = int(np.clip(c - h, 0, X.shape[3] - FR.PATCH))
        j = int(rngj.integers(-FR.JIT, FR.JIT + 1))
        x = X[b, FR.Z0 + j:FR.Z0 + j + FR.D, r:r + FR.PATCH, c:c + FR.PATCH]
        p2, s2, o2 = (q[b, r:r + FR.PATCH, c:c + FR.PATCH] for q in (POS, SUP, OK))
        sp, _ = FTJ.masks(arm, j)
        sh = S[b, r:r + FR.PATCH, c:c + FR.PATCH].astype(np.float32)
        y = (np.abs((FR.T16 + j)[:, None, None] - sh[None]) <= FTJ.SHELL) & p2[None]     # the one changed line
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


def batch_p4(rng, rngj, arm, X, POS, SUP, OK, centres):
    return batch_s(FR.BATCH, rng, rngj, arm, X, POS, SUP, OK, centres)


FTJ.batch = batch_p4       # ft_r.main calls FTJ.batch(...) for the Paris 4 stream
FR.batch_n = batch_s       # and the module global batch_n(...) for the 0139 stream
FR.OUT = OUT3227           # ft_r.main writes OUT / <arm><tag>


def check():
    """(1) s = 0: batch_s == ft_j.batch (BATCH) and == ft_r.batch_n (N0139) draw for draw, all arrays;
    (2) planted per-pixel s on a stack whose value is mesh t + 100: every label voxel has |t - s| <= 1.5, every
    supervised pos voxel off it has |t - s| > 1.5, supervised voxels are |t| <= 7.5 as in F-524 (unchanged);
    (3) no plane prior: every window plane is a label plane in some draw and not in another; (4) module patches."""
    tm = (np.arange(FR.FT.NPL_IN) - 15.5).astype(np.float32)
    X = np.broadcast_to(tm[None, :, None, None], (2, FR.FT.NPL_IN, 96, 96)).copy() + 100
    POS = np.zeros((2, 96, 96), bool); POS[:, 20:60] = True; POS[:, :, 70:80] = True
    SUP = np.ones_like(POS); OK = np.ones_like(POS)
    centres = np.argwhere(SUP)
    ok = {}
    register(X, np.zeros((2, 96, 96), np.int8), 0)
    same = True
    for sd in range(16):
        a = ORIG_BATCH(np.random.default_rng([sd, 1]), np.random.default_rng([sd, 2]), 'J', X, POS, SUP, OK, centres)
        b = batch_p4(np.random.default_rng([sd, 1]), np.random.default_rng([sd, 2]), 'J', X, POS, SUP, OK, centres)
        same &= all(np.array_equal(u, v) for u, v in zip(a[:3], b[:3])) and a[3] == b[3]
        a = ORIG_BATCH_N(FR.N0139, np.random.default_rng([sd, 3]), np.random.default_rng([sd, 4]), 'J', X, POS, SUP,
                         OK, centres)
        b = batch_s(FR.N0139, np.random.default_rng([sd, 3]), np.random.default_rng([sd, 4]), 'J', X, POS, SUP, OK,
                    centres)
        same &= all(np.array_equal(u, v) for u, v in zip(a[:3], b[:3])) and a[3] == b[3]
    ok['s0_equals_label_J'] = bool(same)
    X2 = X.copy()                                         # new object -> own snap map
    rs = np.random.default_rng(5)
    S = rs.integers(-SN.SMAX, SN.SMAX + 1, size=(2, 96, 96)).astype(np.int8)
    register(X2, S, 0)
    good, lab_plane = True, np.zeros((2, FR.D), int)
    rng, rngj = np.random.default_rng(1), np.random.default_rng(2)
    for _ in range(64):
        x, y, w, js = batch_s(FR.BATCH, rng, rngj, 'J', X2, POS, SUP, OK, centres)
        t = x[:, 0] - 100
        # the per-voxel s after rot / flip: recover it from a stack whose value is s (same draws)
        lab = (y[:, 0] > 0) & (w[:, 0] > 0)
        good &= bool(np.all(np.abs(t[w[:, 0] > 0]) <= FTJ.SLAB) and np.all(np.abs(t[w[:, 0] == 0]) > FTJ.SLAB))
        for bi in range(len(js)):
            pl = lab[bi].any((1, 2)); lab_plane[0][pl] += 1; lab_plane[1][~pl] += 1
    # per-voxel check without rot / flip ambiguity: rebuild s per voxel through a value-coded stack
    XS = (np.broadcast_to(S[:, None].astype(np.float32), X.shape) + 50).copy()
    XP = (np.broadcast_to(POS[:, None].astype(np.float32), X.shape)).copy()
    for Xc in (XS, XP):
        register(Xc, S, 0)
    rng_a, rng_b, rng_c = (np.random.default_rng(9) for _ in range(3))
    rj_a, rj_b, rj_c = (np.random.default_rng(10) for _ in range(3))
    for _ in range(32):
        xt, y, w, js = batch_s(FR.BATCH, rng_a, rj_a, 'J', X2, POS, SUP, OK, centres)
        xsv = batch_s(FR.BATCH, rng_b, rj_b, 'J', XS, POS, SUP, OK, centres)[0]
        xpv = batch_s(FR.BATCH, rng_c, rj_c, 'J', XP, POS, SUP, OK, centres)[0]
        t, sv, pv = xt[:, 0] - 100, xsv[:, 0] - 50, xpv[:, 0] > 0
        sup = w[:, 0] > 0
        lab = (y[:, 0] > 0) & sup
        good &= bool(np.all(np.abs(t[lab] - sv[lab]) <= FTJ.SHELL) and np.all(pv[lab]))
        good &= bool(np.all(np.abs(t[sup & pv & ~lab] - sv[sup & pv & ~lab]) > FTJ.SHELL))
        good &= bool(not np.any(y[:, 0][~pv] > 0))
    ok['snap_label_masks'] = good
    # negative test: label J (ft_j.batch, s ignored) on the same draws must fail the per-voxel J_snap rule
    xt, y, w, _ = ORIG_BATCH(np.random.default_rng(9), np.random.default_rng(10), 'J', X2, POS, SUP, OK, centres)
    xsv = batch_s(FR.BATCH, np.random.default_rng(9), np.random.default_rng(10), 'J', XS, POS, SUP, OK, centres)[0]
    lab = (y[:, 0] > 0) & (w[:, 0] > 0)
    ok['negative_label_J_fails'] = not bool(np.all(np.abs((xt[:, 0] - 100)[lab] - (xsv[:, 0] - 50)[lab]) <= FTJ.SHELL))
    ok['no_plane_prior'] = bool((lab_plane[0] > 0).all() and (lab_plane[1] > 0).all())
    ok['module_patches'] = FTJ.batch is batch_p4 and FR.batch_n is batch_s and FR.OUT == OUT3227 and FR.JIT == 6
    _S.clear()
    print(json.dumps(ok))
    assert all(ok.values()), ok


def main():
    FR.main()
    info = {str(i): dict(m=m, sha_s=SN.sha(s), shape=list(s.shape), mean_abs_s=float(np.abs(s).mean()))
            for i, (_, (X, s, m)) in enumerate(_S.items())}
    ref = {nm: json.loads((HERE / 'labelset.json').read_text())[nm]['sha_s'] for nm in ('paris4', 'p0139')}
    got = sorted(v['sha_s'] for v in info.values())
    assert got == sorted(ref.values()), (got, ref)       # the trainer used exactly the checked label set
    (OUT3227 / 'R' / 'snap_train.json').write_text(json.dumps(dict(streams=info, labelset_sha=ref), indent=1))
    print(json.dumps(info))


if __name__ == '__main__':
    if '--check' in sys.argv or '--arm' not in sys.argv:
        check()
    else:
        assert sys.argv[sys.argv.index('--arm') + 1] == 'R', 'attempt 3 trains arm R (label J_snap) only'
        main()
