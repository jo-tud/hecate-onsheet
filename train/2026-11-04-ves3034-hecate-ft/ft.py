"""VES-3034: fine-tune hecate (9.6 um) from its frozen checkpoint on the Paris 4 train segments (README "Data and split").

Two heads, one loss (README "Training code"): 2D = BCE + soft Dice of forward() against the human 2D label inside the
supervision mask; 3D = BCE of forward_3d() against the 3D target over all valid render voxels; total = L2D + L3D.
3D target (README "The 3D target"), built once per block from frozen hecate's own tiled 3D output:
  y = [p3D >= 0.5] & not air (render 0 or outside the valid mask) & not dark (render < Otsu T of the segment)
  arm S: also y = 0 where |t| >= 2 (only planes t = -1.5 .. 1.5 can be 1); arm U: no sheet clip (null arm).
Inputs: F-457 training blocks DATA/train.npz (32 planes t = -15.5 .. 15.5, 256 x 256, raw uint8, sites excluded by
EXCL); hecate reads the central 16 planes (t = -7.5 .. 7.5) in forward order, as hecate.predict(reverse=False).

  python ft.py --arm S|U [--steps 4000] [--device cuda] [--tag pilot] [--attempt3]
--attempt3 (README "Attempt 3", VES-3061): loss = L3D only, lr 1e-4; trainable = the 3D decoder
(canonical.decoder.channel_reduce, decoder_blocks, logit); frozen = canonical.normalization, canonical.backbone (the
ResNet-152 encoder), the 2D-only parameters (decoder.depth_collapse.attn_conv, depth_coordinate_scale) and the unused
aux heads. Without the flag the attempt 1 / 2 loop runs unchanged.
Writes OUT/<arm><tag>/{hecate_ft.pth (load_model-compatible payload), train.json, target_stats.json} and caches the
frozen 3D targets in OUT/target_p3d.npz (shared by both arms).
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HECATE_DIR = Path(os.environ.get('HECATE_DIR', 'hecate'))     # hecate.py + hecate_9.6um.pth from huggingface.co/scrollprize/hecate
CKPT = HECATE_DIR / 'hecate_9.6um.pth'
HEC_PY = HECATE_DIR / 'hecate.py'
DATA = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves2658/run'            # F-457 run root (prep.json, train.npz)
OUT = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves3034'
TRAIN_SEGS = ('20230702185753', '20231007101619', '20231016151002', '20231031143852')   # even positions (README)
HELD_SEGS = ('20230929220926', '20231012184424', '20231106155351', '20231210121321')    # odd positions, never read
NPL_IN, D = 32, 16
Z0 = NPL_IN // 2 - D // 2                       # 8: hecate.predict's forward window of a 32-plane render
T16 = np.arange(D) - 7.5                         # plane offsets of the 16 hecate planes, level-2 vx
CLIP = np.abs(T16) < 2                           # arm S: planes that may carry 1 (t in -1.5 .. 1.5)
PATCH, BATCH, LR, WD = 64, 8, 1e-5, 1e-4
STEPS = 4000
LR3 = 1e-4                                       # attempt 3: 3D-only loss, decoder lr
TRAIN3 = ('canonical.decoder.channel_reduce.', 'canonical.decoder.decoder_blocks.', 'canonical.decoder.logit.')


def load_hecate():
    spec = importlib.util.spec_from_file_location('hecate_ref', HEC_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def otsu(v):
    """Otsu threshold of uint8 values (256-bin histogram, numpy)."""
    h = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mt = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        sb = (mt * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = -1
    return int(np.argmax(sb)) + 1               # dark = value < T, i.e. value <= argmax bin


def frozen_targets(hec, dev, vol, ok):
    """Frozen hecate tiled 3D output (uint8 0..255) on the 16 forward planes of each block; cached on disk."""
    f = OUT / 'target_p3d.npz'
    if f.exists():
        z = np.load(f)
        return z['p3d']
    model = hec.load_model(CKPT, dev)
    p3d = np.zeros((len(vol), D, vol.shape[2], vol.shape[3]), np.uint8)
    for b in range(len(vol)):
        o = np.zeros(vol.shape[1:], np.uint8)
        v = vol[b] * ok[b][None]                # outside the valid mask = air (render convention: 0)
        hec.predict(model, v, o, reverse=False, batch_size=8, three_d=True)
        p3d[b] = o[Z0:Z0 + D]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, p3d=p3d)
    del model
    return p3d


def soft_dice(logit, y, m):
    p = torch.sigmoid(logit) * m
    y = y * m
    return 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)


def batch(rng, X, Y3, Y2, M2, OK, centres):
    """BATCH random 64 x 64 patches centred on supervised px; in-plane rot90 / flip only (never along t)."""
    xs, y3s, y2s, m2s, oks = [], [], [], [], []
    h = PATCH // 2
    for _ in range(BATCH):
        b, r, c = centres[rng.integers(len(centres))]
        r = int(np.clip(r - h, 0, X.shape[2] - PATCH)); c = int(np.clip(c - h, 0, X.shape[3] - PATCH))
        a = [X[b, :, r:r + PATCH, c:c + PATCH], Y3[b, :, r:r + PATCH, c:c + PATCH],
             Y2[b, r:r + PATCH, c:c + PATCH], M2[b, r:r + PATCH, c:c + PATCH], OK[b, :, r:r + PATCH, c:c + PATCH]]
        k, fl = int(rng.integers(4)), bool(rng.integers(2))
        a = [np.rot90(q, k, axes=(-2, -1)) for q in a]
        if fl:
            a = [q[..., ::-1] for q in a]
        for lst, q in zip((xs, y3s, y2s, m2s, oks), a):
            lst.append(np.ascontiguousarray(q))
    f = lambda L: torch.from_numpy(np.stack(L).astype(np.float32))[:, None]
    return f(xs), f(y3s), f(y2s), f(m2s), f(oks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('S', 'U'), required=True)
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--attempt3', action='store_true')
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    T0 = time.time()
    hec = load_hecate()
    meta = json.loads((DATA / 'prep.json').read_text())
    bw = np.array([blk['w'] for blk in meta['blocks']])
    keep = np.isin(bw, TRAIN_SEGS)
    assert not np.isin(bw[keep], HELD_SEGS).any() and keep.sum() == 4 * 48, keep.sum()
    tr = np.load(DATA / 'train.npz')
    idx = np.flatnonzero(keep)
    vol, ok, pos, sup = (tr[k][idx] for k in ('vol', 'ok', 'pos', 'sup'))
    print(f'{len(idx)} train blocks loaded, {time.time() - T0:.0f} s', flush=True)

    p3d = frozen_targets(hec, dev, vol, ok)                         # (n, 16, 256, 256) uint8
    print(f'frozen 3D targets ready, {time.time() - T0:.0f} s', flush=True)
    x16 = vol[:, Z0:Z0 + D] * ok[:, None]                           # forward order, raw uint8, 0 outside the mask
    air = (x16 == 0) | ~ok[:, None]
    T = {w: otsu(x16[bw[idx] == w][(x16[bw[idx] == w] > 0) & ok[bw[idx] == w][:, None]]) for w in TRAIN_SEGS}
    dark = np.stack([x16[i] < T[bw[b]] for i, b in enumerate(idx)])
    y3 = (p3d >= 128) & ~air & ~dark                                # rint(255 p) >= 128  <=>  p >= 0.5 (to 1/510)
    if a.arm == 'S':
        y3 &= CLIP[None, :, None, None]
    valid3 = np.broadcast_to(ok[:, None], y3.shape)
    stats = dict(otsu=T, pos_frac_valid=float(y3[valid3].mean()),
                 pos_frac_by_plane=[float(y3[:, k][ok].mean()) for k in range(D)],
                 frozen_pos_frac_by_plane=[float((p3d[:, k] >= 128)[ok].mean()) for k in range(D)])
    print(json.dumps(stats), flush=True)

    model = hec.load_model(CKPT, 'cpu').to(dev)
    model.requires_grad_(True)
    lr = LR
    if a.attempt3:
        lr = LR3
        for k, q in model.named_parameters():
            q.requires_grad_(k.startswith(TRAIN3))
    model.train()
    for mod in model.modules():                                     # frozen BN statistics (README)
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()
    trainable = [k for k, q in model.named_parameters() if q.requires_grad]
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f'trainable tensors {len(trainable)}, params {n_train}, lr {lr}', flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WD)
    centres = np.argwhere(sup)
    rng = np.random.default_rng([3034, a.seed])
    Xn = x16                                                        # normalised per batch (/ model.divisor)
    log = []
    for step in range(a.steps):
        x, t3, t2, m2, okp = batch(rng, Xn, y3, pos, sup, np.broadcast_to(ok[:, None], y3.shape), centres)
        x, t3, t2, m2, okp = (q.to(dev) for q in (x, t3, t2, m2, okp))
        img = x / model.divisor
        valid = ((x > 0) & (okp > 0)).any(2, keepdim=True).expand_as(x)  # predict(): support = raw.any over depth
        lo3 = model.forward_3d(img)
        l3 = (F.binary_cross_entropy_with_logits(lo3, t3, reduction='none') * okp).sum() / okp.sum().clamp_min(1)
        if a.attempt3:                                              # 3D-only loss; 2D head not run in training
            l2 = torch.zeros((), device=dev)
            loss = l3
        else:
            lo2 = model(img, valid)
            l2 = ((F.binary_cross_entropy_with_logits(lo2, t2, reduction='none') * m2).sum()
                  / m2.sum().clamp_min(1) + soft_dice(lo2, t2, m2))
            loss = l2 + l3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            log.append([step, float(l2), float(l3), round(time.time() - T0, 1)])
            print(f'{a.arm}{a.tag} step {step} l2 {float(l2):.4f} l3 {float(l3):.4f} {time.time() - T0:.0f} s', flush=True)
    od = OUT / f'{a.arm}{a.tag}'
    od.mkdir(parents=True, exist_ok=True)
    payload = torch.load(CKPT, map_location='cpu', weights_only=True)
    payload['ema_model'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload['ves3034'] = dict(arm=a.arm, steps=a.steps, seed=a.seed, attempt3=a.attempt3)
    if a.attempt3:                                                  # frozen tensors must be bit-identical
        ref = torch.load(CKPT, map_location='cpu', weights_only=True)['ema_model']
        same = [k for k in ref if torch.equal(ref[k], payload['ema_model'][k])]
        moved = [k for k in ref if k not in same]
        assert all(k.startswith(TRAIN3) for k in moved), [k for k in moved if not k.startswith(TRAIN3)][:5]
        print(f'attempt3: {len(moved)} tensors moved (all in the decoder), {len(same)} unchanged', flush=True)
    torch.save(payload, od / 'hecate_ft.pth')
    info = dict(arm=a.arm, tag=a.tag, seed=a.seed, steps=a.steps, batch=BATCH, patch=PATCH, lr=lr, wd=WD, attempt3=a.attempt3,
                trainable=trainable, n_trainable=n_train,
                train_segs=TRAIN_SEGS, n_blocks=int(len(idx)), loss_log=log, seconds=round(time.time() - T0, 1),
                sec_per_step=round((log[-1][3] - log[0][3]) / max(1, log[-1][0] - log[0][0]), 3) if len(log) > 1 else None,
                device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
                max_mem_gb=round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == 'cuda' else None)
    (od / 'train.json').write_text(json.dumps(info, indent=1))
    (od / 'target_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'loss_log'}), flush=True)


if __name__ == '__main__':
    main()
