# Third-party material and the licences of the inputs

## Code and weights this release builds on

| what | source | licence | how it is used here |
|---|---|---|---|
| hecate (model code `hecate.py`, checkpoint `hecate_9.6um.pth`) | https://huggingface.co/scrollprize/hecate, revision `9cb86e50` | MIT (Copyright 2026 Vesuvius Challenge), text below | not copied into this tree: `onsheet.py` downloads both files at that revision and checks their sha256. S is a fine-tune of `hecate_9.6um.pth`. |
| canonical 2 µm ink detector (hecate's base model) | https://huggingface.co/scrollprize/ink_canonical_2um, revision `075855bc` | MIT | not used directly; named because hecate starts from it |
| ink-sdk (optional second way to run S) | https://github.com/LimeGS/ink-sdk (commit `40a647a4` of `main`; first tested at `2df3ab50`) | MIT (Copyright 2026 LimeGS) | not copied and not needed by `run_window.sh`; the README shows how to run S in it |
| VC3D `vc_render_tifxyz` | https://github.com/ScrollPrize/villa, folder `volume-cartographer/` (commit `47111d1a`; source `volume-cartographer/apps/src/vc_render_tifxyz.cpp`) | **GPL-3.0** (`volume-cartographer/LICENSE`; its `NOTICE`: version 3 or later, Copyright 2023 EduceLab). Only villa's root `LICENSE` is MIT. | called as a separate program to render the CT onto the segment. Not copied, not linked and not shipped here: you bring your own build or image. The code in this tree stays MIT. |

## Data

| what | source | terms | how it is used here |
|---|---|---|---|
| PHerc. Paris 4 CT, scan `20260411134726` (2.4 µm, 78 keV), pyramid level 2 | `s3://vesuvius-challenge-open-data/PHercParis4/volumes/` | Vesuvius Challenge open data, CC BY-NC 4.0 | read over https; not in this tree |
| PHerc. Paris 4 segment meshes `<seg>-on-20260411134726-2.4um.tifxyz` | `s3://vesuvius-challenge-open-data/PHercParis4/segments/<seg>/mesh/` | as above | read over https; not in this tree |
| PHerc. Paris 4 ink labels, release `20260918` (`inklabels.zarr`, `supervision.zarr`) | `.../segments/<seg>/ink-labels/2.4um-volume-20260411134726/20260918/` | as above | S was trained on 4 of the 8 labelled segments; scored on the other 4 |
| PHerc. 0139 scan `20250728140407` (9.362 µm, 113 keV), segments w035–w044 and their `20260918` ink labels | `s3://vesuvius-challenge-open-data/PHerc0139/` | as above | S rehearsed on w040 and w044 during training; w035, w039, w041 are the guard |
| Forrest's PHerc. Paris 4 ufsm wraps `ufsm-c2f-20261030` (`surfaces/wraps/wNNN.sfc`) | `dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/segments/ufsm-c2f-20261030/` | community upload by its author; not redistributed here | only scored: the outside-mesh row of the README limits (`results/f647_forrest_wraps.json`); not in this tree, not used in training |

PHerc. Paris 4: the organisers said on 2026-11-09 that results and images may appear in progress-prize entries and
public repositories; academic publication of Paris 4 results in journals or conferences is not allowed before
2027-06-25.

## The S checkpoint

The S weights are a modified copy of `hecate_9.6um.pth` (MIT) trained on the CC BY-NC 4.0 labels above. We release
them under **CC BY-NC 4.0** and keep hecate's MIT notice with them. The code in this tree is MIT (`LICENSE`).

| what | where | size | sha256 |
|---|---|---|---|
| `hecate_onsheet_S.pth` | https://huggingface.co/jo-tud/hecate-onsheet | 522 573 195 bytes | `4b84fdf2f90cb369b254f04415d474bac8ab087a47022badd2d20fa32205b4f2` |

`MODEL_CARD.md` of this tree is the README of that model page; it carries the licence and hecate's MIT notice.
The code is at https://github.com/jo-tud/hecate-onsheet.

## hecate licence text

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
