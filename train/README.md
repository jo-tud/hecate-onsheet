# Training code of S (reference copy)

These are the files that trained the S checkpoint on 2026-11-06 (8 000 steps, seed 0, 56 min on one RTX PRO 6000,
23.8 GB peak), copied from the author's working repository at commit `288c59dbb`. The folder names are the
working-repository names. The only changes are the data paths (now `$ONSHEET_DATA/<name>` and `$HECATE_DIR`);
`PROVENANCE.md` one level up lists them. **Not runnable from this tree alone:** the training blocks they read
(`$ONSHEET_DATA/ves2658/run/train.npz`, Paris 4; `$ONSHEET_DATA/ves2241/train.npz`, PHerc. 0139) are 32-plane
renders of the labelled segments made by internal prep scripts that are not in this release.

Chain: `attempt3/ft_s.py` → `ft_r.py` (0139 rehearsal) → `ft_j.py` (shell label + depth jitter) → `ft.py`
(the training loop around hecate). Commands that made S:

```
python 2026-11-05-ves3090-hecate-rehearsal/attempt3/snap.py --check
python 2026-11-05-ves3090-hecate-rehearsal/attempt3/ft_s.py --check
python 2026-11-05-ves3090-hecate-rehearsal/attempt3/ft_s.py --arm R --steps 8000 --device cuda
```

## The recipe, in words

Start from `hecate_9.6um.pth`. Change four things:

1. **Label on the sheet, not through the slab.** For every labelled ink pixel, ink is a thin shell of 4 planes
   (±1.5 voxels) at the sheet; every other voxel within ±7.5 voxels of the surface is "no ink". Unlabelled pixels
   are not trained.
2. **The shell follows the CT, not the mesh.** Per pixel, find the sheet's outer edge in the CT (steepest fall of
   the smoothed intensity along the normal within ±5 voxels; in-plane smoothing σ 16 px), subtract the segment
   median, clip to ±4 voxels, and move the shell there (`snap.py`).
3. **Random depth shift** of ±6 voxels per training crop, so the net cannot learn "the sheet is always in plane 7".
4. **Rehearsal of a second scroll:** every step mixes 8 Paris 4 crops with 4 crops of PHerc. 0139 w040 and w044
   under the same label rule; the two losses are averaged with equal weight.

Loss: 3D binary cross-entropy over the trained voxels only. AdamW, lr 1e-5, weight decay 1e-4, batch-norm
statistics frozen, 64 × 64 px crops with rotations and flips, 8 000 steps, one seed. The 2D head's own weights are
not changed (its input trunk is). Paris 4 training segments: 20230702185753, 20231007101619, 20231016151002,
20231031143852; held out: 20230929220926, 20231012184424, 20231106155351, 20231210121321.
