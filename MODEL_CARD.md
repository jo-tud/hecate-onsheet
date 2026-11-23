---
license: cc-by-nc-4.0
base_model: scrollprize/hecate
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - vesuvius-challenge
  - ink-detection
  - hecate
  - ink-sdk
inference: false
---

# hecate-onsheet (S): hecate 9.6 µm fine-tuned to keep its 3D ink on the papyrus sheet

S is a fine-tune of the public 9.6 µm 3D ink model [hecate](https://huggingface.co/scrollprize/hecate). hecate
predicts ink in 3D around a segment, but on PHerc. Paris 4 part of that ink sits above and below the sheet. S was
trained so that its 3D ink sits on the sheet, with the sheet position taken from the CT, not from the mesh. It loads
and runs like `hecate_9.6um.pth`: same `hecate.py`, same `predict`, strict load; only `ema_model` differs.

**For whom:** people who run hecate 9.6 µm on PHerc. Paris 4, for example through
[ink-sdk](https://github.com/LimeGS/ink-sdk), and use its 3D output or want less ink off the sheet.

| file | size | sha256 |
|---|---|---|
| `hecate_onsheet_S.pth` | 522 573 195 bytes | `4b84fdf2f90cb369b254f04415d474bac8ab087a47022badd2d20fa32205b4f2` |
| `hecate-9.6um-onsheet-s.toml` | ink-sdk profile | (the same file as in ink-sdk once it ships the profile) |
| `results/*.json` | the source of every number on this page | |

## Run it in ink-sdk

Input: a 16-layer surface volume rendered with VC3D `vc_render_tifxyz` at 9.6 µm (layers 1 voxel apart), as a uint8
`.npy` of shape 16 × H × W.

If your ink-sdk lists `hecate@9.6um-onsheet-s` (`ink-sdk models`):
```
ink-sdk fetch -p hecate@9.6um-onsheet-s        # downloads this file at the pinned revision, checks size and sha256
ink-sdk doctor                                 # [verified] hecate@9.6um-onsheet-s
ink-sdk infer -p hecate@9.6um-onsheet-s --surface-volume render16.npy --pixel-um 9.6 --out runs/s
```
If it does not (ink-sdk `2df3ab50`, 11-11), add the profile from this page first:
```
pip install -e "./ink-sdk[hecate,fetch]"       # git clone https://github.com/LimeGS/ink-sdk
mkdir -p profiles && hf download jo-tud/hecate-onsheet hecate-9.6um-onsheet-s.toml --local-dir profiles
export INK_SDK_PROFILES=$PWD/profiles          # remove this file once ink-sdk ships the profile: it refuses a duplicate id
```
then the three lines above.

**The S map is `runs/s/probability_primary.npy`** (the `reverse` direction; also in `probability_reverse.npy`).
S was trained on `vc_render_tifxyz` stacks read in ink-sdk's `reverse` order. The profile runs both directions and
makes `reverse` the primary map. The forward map (`probability.npy`) reads the other face of the sheet: Pearson r 0.134
with the S map. On our reference window the ink-sdk primary map equals the map of our own runner: r 0.999999, every
pixel within one 8-bit step (1/255), 98.9 % of the pixels byte-equal (`results/f4015_ink_sdk_profile.json`).

Tested: ink-sdk `2df3ab50` + the profile, fp32, CUDA, with torch 2.12.1+cu130 (the build behind the stored maps);
the match with our own runner above is on one window. Fresh installs of 2026-11-13 got torch 2.15.0+cu132 and gave
the map of torch 2.12.1+cu130 within one 8-bit step on two windows (the reference window and one on segment
20231012184424): 7 108 and 4 929 of 2 560 000 pixels (0.28 % and 0.19 %) differ by 1/255, none by more. The maps are
not byte-identical across torch builds (`results/f666_fresh_install_map.json`). Not tested: `ink-sdk render` /
`--tifxyz` input, bf16, windows other than these two.

## Results

All on PHerc. Paris 4, scan 20260411134726 (2.4 µm, 78 keV) read at 9.6 µm (level 2), labels release 20260918.
S was fine-tuned on 4 of the 8 labelled segments; every Paris 4 number below is on the other 4 (**held out** from the
fine-tune). Frozen hecate is the baseline. Its card says it was developed on the Scroll Prize ink dataset with 3D
supervision on Paris 4 and names no held-out segments, so we treat it as **in-sample** on these segments; if that is
right, the comparison is tilted in its favour.

| what | S | frozen hecate | other baseline | data | results file |
|---|---|---|---|---|---|
| off-sheet share: of the 3D ink (p ≥ 0.5) at labelled ink spots, the share more than half a sheet spacing from the sheet (sheet found in the CT: steepest intensity fall along the normal); lower is better | **0.150** [0.113, 0.192] | 0.214 [0.174, 0.256] | 0.774 / 0.692: S's own map moved 6–8 vx outward / inward | 224 ink spots on 2 held-out segments; paired S − frozen −0.064 [−0.088, −0.041] | `results/f561_on_sheet_paris4.json` |
| 2D ink AUC on the VC3D render, 4 windows of 15.4 mm | **0.940** | 0.918 | chance 0.5 | one window per held-out segment, pooled px | `results/f561_villa_render_4windows.json` |
| 2D ink AUC on the reference window | **0.915** | 0.867 | chance 0.5 | segment 20231210121321, rows 1200–2800, cols 2800–4400 | `results/f581_fresh_clone_one_window.json` |
| guard: 2D / 3D ink AUC on PHerc. 0139 (9.362 µm, 113 keV), w035, w039, w041 (not used in the fine-tune) | 0.799 / 0.829 | 0.789 / 0.837 | bar set before the run: frozen − 0.03 | 119 ink / 113 no-ink spots; in-sample for frozen, dev for S | `results/f524a3_training_run_and_0139_guard.json` |

95 % intervals: bootstrap over spots. In `f524a3_…` the key `R` is S; in `f561_…` and `f555_…` the key `S` is S and
`R` is another fine-tune that is not released.

## Limits and named fails

- **One scroll.** PHerc. Paris 4 only, one seed, one checkpoint; the off-sheet spots lie on 2 of the 4 held-out
  segments.
- **No gain on PHerc. 1667.** The same recipe on 4 PHerc. 1667 segments, scored on 2 held-out ones: off-sheet share
  0.107 vs frozen 0.119, paired difference [−0.036, +0.012] (frozen hecate was already on the sheet there). On
  PHerc. 0139 w045 + three PHerc. 0841 segments that model moved ink towards the sheet (−0.049 [−0.081, −0.016]) but
  its 2D AUC fell from 0.896 to 0.854 (`results/f564_fail_1667.json`, `results/f564a2_fail_public_arena_0139_0841.json`).
- **The first bar failed.** In the mesh frame (sheet = mesh) the off-sheet share of S stayed above the bar we had set
  (`results/f524a3_…`, verdict FAIL), because the mesh lies a few voxels off the papyrus. The CT-frame result above
  is a bar set after that fail and tested on two fresh spot draws (0.144 and 0.150; `results/f555_…`).
- **The gain is on one side, and the peak is wider.** S removes ink that lies outward of the sheet. Inward of the
  sheet it has more ink off the sheet than frozen hecate (0.062 against 0.042 of the ink mass), and its peak on the
  sheet is wider (3.9 against 2.8 voxels at half maximum; `results/f561_depth_profile.json`). These are point
  readings with no interval.
- **An outside author's mesh.** On Forrest's public PHerc. Paris 4 ufsm wraps (`ufsm-c2f-20261030`) the gain is
  measured only where that mesh lies within 3 vx of an organiser mesh, on patches resampled to 1 px = 1 vx:
  off-sheet share 0.221 against 0.258 for frozen hecate, paired −0.037 (95 % CI −0.055 … −0.017), 225 labelled
  spots on 17 wraps (`results/f647_forrest_wraps.json`). It is not measured away from organiser meshes. Inside the
  sheet S's ink is not narrower than hecate's.
- **2D AUC is pixel-level** (labelled ink vs labelled no-ink), not a letter-reading test. No new text is claimed.
- **Direction.** Read in the other order, S gives a map of the other face (r 0.134). Check your stack order.

## What still needs the GitHub repo

Nothing, to run S: this page and ink-sdk are enough. [jo-tud/hecate-onsheet](https://github.com/jo-tud/hecate-onsheet)
(MIT) is needed only to **repeat our numbers**: `run_window.sh` builds the reference window from the open data
(9.6 µm segment from the public 2.4 µm mesh, CT chunks, VC3D render), runs frozen hecate and S, and checks both 2D
AUC values against the stored ones (±0.005); `train/` holds the training code. The off-sheet share was scored with
our internal scorer, which is in neither place.

## Licence

These weights are a modified copy of `hecate_9.6um.pth` (MIT, notice below), trained on Vesuvius Challenge open data
(CT, meshes and ink labels of PHerc. Paris 4 and PHerc. 0139, CC BY-NC 4.0). They are released under **CC BY-NC 4.0**.
PHerc. Paris 4 results may appear in prize entries and public repositories; they may not go into journal or
conference publications before 2027-06-25 (organisers, 2026-11-09). VC3D `vc_render_tifxyz` (ScrollPrize/villa,
`volume-cartographer/`, GPL-3.0) is called as a separate program and not shipped here. ink-sdk: MIT, LimeGS.

hecate licence notice:

```
MIT License

Copyright (c) 2026 Vesuvius Challenge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
