"""VES-3090: F-522's label-J hecate fine-tune (ft_j.py, arm J, imported) with one anti-forgetting change per arm
(README "Arms"). The Paris 4 stream is F-522 arm J's, draw for draw (same rngs, same 8 patches per step).

  arm R (rehearsal): + N0139 = 4 PHerc0139 patches per step from the ves2241 dev blocks of REHEARSE (w040, w044),
         label J and +-6 vx jitter by ft_j.masks / the ft_j.batch rule, own rngs [3090, seed, 4] (patches) and
         [3090, seed, 5] (jitter). Loss = mean BCE over the Paris 4 supervised voxels + mean BCE over the 0139
         supervised voxels (each normalised by its own count, weight 1). One forward of the 12 patches (BN frozen).
  arm A (L2-SP):     loss = Paris 4 BCE + (ALPHA / 2) * sum over trained tensors of ||theta - theta_0||^2, theta_0 =
         the frozen checkpoint. The anchor gradient ALPHA * (theta - theta_0) is added to p.grad of every parameter
         that received a BCE gradient this step, so tensors outside the 3D graph (2D head, aux heads) stay untouched
         exactly as in F-522 (asserted). ALPHA = 2 * l_J / D_J (README): l_J = 0.172, F-522 J's mean BCE over its
         last 20 logged steps; D_J = 1003.48, ||theta_J - theta_0||^2 of F-522's J checkpoint. -> 3.4e-4.
Everything else as F-522 J: lr 1e-5, AdamW wd 1e-4, BN frozen, batch 8 Paris 4 patches 64 x 64, rot90 / flip,
seed 0, 8 000 steps, 3D head only.

  python ft_r.py --check                     synthetic checks (no model, no data)
  python ft_r.py --arm R|A [--steps 8000] [--tag ''] [--device cuda]
Writes OUT/<arm><tag>/{hecate_ft.pth, train.json}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r3090 import load, OUT, DATA0139, REHEARSE, GUARD   # noqa: E402

FTJ = load('ves3067_ft_j', HERE.parent / '2026-11-05-ves3067-hecate-labelj' / 'ft_j.py')
FT = FTJ.FT
D, Z0, T16, PATCH, BATCH, JIT = FTJ.D, FTJ.Z0, FTJ.T16, FTJ.PATCH, FTJ.BATCH, FTJ.JIT
N0139 = 4                     # arm R hyper-parameter: 0139 patches added per step
L_J, D_J = 0.172, 1003.48     # F-522 J: mean BCE last 20 logged steps; ||theta_J - theta_0||^2 (README)
ALPHA = 3.4e-4                # arm A hyper-parameter: 2 * L_J / D_J = 3.43e-4, 2 significant digits
STEPS = FTJ.STEPS
assert abs(ALPHA - 2 * L_J / D_J) < 5e-6


def batch_n(n, rng, rngj, arm, X, POS, SUP, OK, centres):
    """ft_j.batch with n patches instead of BATCH (same draws, same order; checked equal for n = BATCH)."""
    xs, ys, ws, js = [], [], [], []
    h = PATCH // 2
    for _ in range(n):
        b, r, c = centres[rng.integers(len(centres))]
        r = int(np.clip(r - h, 0, X.shape[2] - PATCH)); c = int(np.clip(c - h, 0, X.shape[3] - PATCH))
        j = int(rngj.integers(-JIT, JIT + 1))
        x = X[b, Z0 + j:Z0 + j + D, r:r + PATCH, c:c + PATCH]
        p2, s2, o2 = (q[b, r:r + PATCH, c:c + PATCH] for q in (POS, SUP, OK))
        sp, tp = FTJ.masks(arm, j)
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


def load_0139():
    """Raw uint8 32-plane blocks of the REHEARSE segments (ves2241 prep.py; dev-five sites already cut out +-48 px)."""
    meta = json.loads((DATA0139 / 'prep.json').read_text())
    bw = np.array([b['w'] for b in meta['blocks']])
    keep = np.isin(bw, REHEARSE)
    assert not np.isin(bw[keep], GUARD).any() and keep.sum() == 79, keep.sum()
    z = np.load(DATA0139 / 'train.npz')
    idx = np.flatnonzero(keep)
    vol, ok, pos, sup = (z[k][idx] for k in ('vol', 'ok', 'pos', 'sup'))
    assert vol.dtype == np.uint8 and vol.shape[1] == FT.NPL_IN
    return vol * ok[:, None], pos, sup, ok, sorted(set(bw[keep].tolist()))


def check():
    """(1) batch_n(BATCH) == ft_j.batch draw for draw (so arm R / A see F-522 J's Paris 4 patches); (2) on the F-399
    synthetic stack, 0139 patches from batch_n(N0139, arm J) obey the label-J masks (ft_j.check's rules)."""
    tm = (np.arange(FT.NPL_IN) - 15.5).astype(np.float32)
    X = np.broadcast_to(tm[None, :, None, None], (2, FT.NPL_IN, 96, 96)).copy() + 100
    POS = np.zeros((2, 96, 96), bool); POS[:, 20:60] = True
    SUP = np.ones_like(POS); OK = np.ones_like(POS)
    centres = np.argwhere(SUP)
    ok = {}
    a = FTJ.batch(np.random.default_rng(7), np.random.default_rng(8), 'J', X, POS, SUP, OK, centres)
    b = batch_n(BATCH, np.random.default_rng(7), np.random.default_rng(8), 'J', X, POS, SUP, OK, centres)
    ok['same_as_ft_j'] = all(np.array_equal(u, v) for u, v in zip(a[:3], b[:3])) and a[3] == b[3]
    good = True
    rng, rngj = np.random.default_rng(1), np.random.default_rng(2)
    for _ in range(64):
        x, y, w, js = batch_n(N0139, rng, rngj, 'J', X, POS, SUP, OK, centres)
        t = x[:, 0] - 100
        lab = (y[:, 0] > 0) & (w[:, 0] > 0)
        good &= len(js) == N0139
        good &= bool(np.all(np.abs(t[w[:, 0] > 0]) <= FTJ.SLAB) and np.all(np.abs(t[lab]) <= FTJ.SHELL))
    ok['rehearsal_masks'] = good
    ok['alpha_rule'] = abs(ALPHA - 2 * L_J / D_J) < 5e-6
    print(json.dumps(ok))
    assert all(ok.values()), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('R', 'A'))
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
    X = vol * ok[:, None]
    centres = np.argwhere(sup)
    if a.arm == 'R':
        X0, pos0, sup0, ok0, segs0 = load_0139()
        centres0 = np.argwhere(sup0)
        rng0, rngj0 = np.random.default_rng([3090, a.seed, 4]), np.random.default_rng([3090, a.seed, 5])
    print(f'{len(idx)} Paris 4 train blocks{"" if a.arm != "R" else f", {len(X0)} 0139 blocks {segs0}"}, '
          f'{time.time() - T0:.0f} s', flush=True)

    model = hec.load_model(FT.CKPT, 'cpu').to(dev)
    model.requires_grad_(True)
    model.train()
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()
    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=FTJ.LR, weight_decay=FTJ.WD)
    rng = np.random.default_rng([3034, a.seed])                     # F-522 J's patch stream
    rngj = np.random.default_rng([3067, a.seed, 3])                 # F-522 J's jitter stream

    def bce(lo, y, w):
        return (F.binary_cross_entropy_with_logits(lo, y, reduction='none') * w).sum() / w.sum().clamp_min(1)

    log, anchored = [], set()
    for step in range(a.steps):
        x, y3, w3, js = FTJ.batch(rng, rngj, 'J', X, pos, sup, ok, centres)
        if a.arm == 'R':
            x0, y0, w0, _ = batch_n(N0139, rng0, rngj0, 'J', X0, pos0, sup0, ok0, centres0)
            x = np.concatenate([x, x0])
        xt = torch.from_numpy(x).to(dev)
        lo = model.forward_3d(xt / model.divisor)
        lp = bce(lo[:BATCH], *(torch.from_numpy(q).to(dev) for q in (y3, w3)))
        l0 = bce(lo[BATCH:], *(torch.from_numpy(q).to(dev) for q in (y0, w0))) if a.arm == 'R' else torch.zeros((), device=dev)
        opt.zero_grad(set_to_none=True)
        (lp + l0).backward()
        pen = 0.0
        if a.arm == 'A':
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        d = p - theta0[n]
                        p.grad.add_(d, alpha=ALPHA)
                        pen += float((d * d).sum())
                        anchored.add(n)
            pen *= ALPHA / 2
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            log.append([step, float(lp), float(l0), pen, round(time.time() - T0, 1)])
            print(f'{a.arm}{a.tag} step {step} lP4 {float(lp):.4f} l0139 {float(l0):.4f} pen {pen:.4f} '
                  f'{time.time() - T0:.0f} s', flush=True)
    od = OUT / f'{a.arm}{a.tag}'
    od.mkdir(parents=True, exist_ok=True)
    payload = torch.load(FT.CKPT, map_location='cpu', weights_only=True)
    ref = payload['ema_model']
    new = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    moved = [k for k in ref if not torch.equal(ref[k], new[k])]
    still_2d = [k for k in ref if k.startswith(FTJ.NO_GRAD_2D)]
    assert still_2d and not any(k in moved for k in still_2d), [k for k in still_2d if k in moved]
    drift = float(sum(((new[k].float() - ref[k].float()) ** 2).sum() for k in ref
                      if ref[k].is_floating_point() and 'running' not in k))
    payload['ema_model'] = new
    payload['ves3090'] = dict(arm=a.arm, steps=a.steps, seed=a.seed, n0139=N0139 if a.arm == 'R' else 0,
                              alpha=ALPHA if a.arm == 'A' else 0.0)
    torch.save(payload, od / 'hecate_ft.pth')
    info = dict(arm=a.arm, tag=a.tag, seed=a.seed, steps=a.steps, batch=BATCH, n0139=N0139 if a.arm == 'R' else 0,
                alpha=ALPHA if a.arm == 'A' else 0.0, rehearse=list(REHEARSE) if a.arm == 'R' else [],
                patch=PATCH, lr=FTJ.LR, wd=FTJ.WD, jitter=JIT, n_moved=len(moved), n_tensors=len(ref),
                n_anchored=len(anchored), drift_sq=drift, drift_sq_F522_J=D_J, unchanged_2d=still_2d,
                loss_log_cols=['step', 'l_paris4', 'l_0139', 'penalty', 'seconds'], loss_log=log,
                seconds=round(time.time() - T0, 1),
                sec_per_step=round((log[-1][4] - log[0][4]) / max(1, log[-1][0] - log[0][0]), 3) if len(log) > 1 else None,
                device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
                max_mem_gb=round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == 'cuda' else None)
    (od / 'train.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ('loss_log', 'unchanged_2d')}), flush=True)


if __name__ == '__main__':
    main()
