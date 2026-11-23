# Provenance

| file | source in the author's working repository | change |
|---|---|---|
| `onsheet.py` | new for this release (2026-11-13). Geometry (`load_grid`, `grid_normals`, `pixel_valid`, `at_pixels`), the chunk list and `auc` are copied from the scorer of the stored results; the CT fetch and the VC3D call follow `sheets3.py` / `render3_vc.sh` of the 4-window result | new |
| `run_window.sh`, `requirements.txt`, `figure/make_figure.py`, `reference/window.json` | new | new |
| `train/2026-11-04-ves3034-hecate-ft/ft.py` | same path, commit `288c59dbb` | 6 lines: data and hecate paths → `$ONSHEET_DATA`, `$HECATE_DIR`; `import os` |
| `train/2026-11-05-ves3067-hecate-labelj/ft_j.py` | same path, `288c59dbb` | 2 lines: output path, `import os` |
| `train/2026-11-05-ves3090-hecate-rehearsal/r3090.py` | same path, `288c59dbb` | 5 lines: four data paths, `import os` |
| `train/2026-11-05-ves3090-hecate-rehearsal/ft_r.py` | same path, `288c59dbb` | identical |
| `train/2026-11-05-ves3090-hecate-rehearsal/attempt3/ft_s.py` | same path, `288c59dbb` | identical |
| `train/2026-11-05-ves3090-hecate-rehearsal/attempt3/snap.py` | same path, `288c59dbb` | 2 lines: output path, `import os` |
| `train/2026-11-05-ves3090-hecate-rehearsal/attempt3/labelset.json` | same path, `288c59dbb` | identical |
| `results/f555_on_sheet_paris4_first_draw.json` | `experiments/2026-11-07-ves3320-ctframe-leak/attempt2/results.json` | identical |
| `results/f561_on_sheet_paris4.json` | `experiments/2026-11-07-ves3343-s-second-fresh/results.json` | identical |
| `results/f561_villa_render_4windows.json` | `experiments/2026-11-07-ves3343-s-second-fresh/results3.json` | identical |
| `results/f581_fresh_clone_one_window.json` | `experiments/2026-11-08-ves3490-s-fresh-clone/results.json` | identical |
| `results/f636_fresh_clone_release_tree.json` | `experiments/2026-11-13-ves3950-s-release/results.json` | identical |
| `results/f636_fetch.json` | `experiments/2026-11-13-ves3950-s-release/fetch.json` | identical |
| `results/f639_ink_sdk_same_map.json` | `experiments/2026-11-13-ves3957-inksdk-releases/results.json` and `profiles.json` (commit `15daf34b`) | extract, all values unchanged: keys `ink_sdk`, `criteria`, `S_ckpt_only_fwd_equals_S_both_fwd`; the rows `PC2`, `S`, `S_ckpt_only` of `pairs`, `loads` and `runs` (of `runs` only `profile`, `returncode`, `seconds`, `files`). Renamed: `reference_window_support_px` (source key `F636_support_px`), `profile_changed_lines` (source `profiles.json` `profiles`, the 3 profiles of these rows), `input.render_stack` (source `profiles.json` `inputs`, the entry `F636_render.npy`). New text: `about`. Rows about other models and another piece are left out |
| `results/f636_figure_display.json`, `figure.png` | written by `figure/make_figure.py` on the output folder of the fresh-clone run behind `results/f636_fresh_clone_release_tree.json` (no model run); the JSON holds the grey-level table of the matched panel, its 2D AUC and the grey-level shares | new (2026-11-13); `figure.png` replaces the four-panel figure |
| `results/f561_depth_profile.json` | `experiments/2026-11-13-ves4083-s-handover/depth_profile.json`, written by `depth_profile.py` there from the stored outputs of the run behind `results/f561_on_sheet_paris4.json` (no new model run); its control: the off-sheet share summed from the bins equals the scored `L_snap` of both models to 1e-9 (`results.json` there, `K1`) | identical |
| `figure_depth.png`, `figure/make_depth_panel.py` | new (2026-11-13); the script draws the PNG from `results/f561_depth_profile.json` and `results/f561_on_sheet_paris4.json` | new |
| `results/f4015_ink_sdk_profile.json` | `experiments/2026-11-13-ves4015-s-where-users/results.json` (commit `a2e34f4e`) | all values unchanged except `C1.doctor_line`: the local file path after the profile id is cut |
| `results/f666_fresh_install_map.json` | `experiments/2026-11-13-ves4121-s-outside-check-a3/results_d3.json` and `sides.json` (commit `f83c4f4d`; `sides.json` one commit later), written by `make_results_file.py` there | extract, all values unchanged: of `results_d3.json` the readings `n_px`, `n_differ`, `max_step`, `byte_equal_share`, `pearson_r` of `D3`, `positive_control_F636` and `null_other_direction`, the three bar verdicts of both pairs, the four map sha256 and `r_bar`; of `sides.json` the torch and Python builds, device, precision, run and venv times, the ink-sdk commit and the weights sha256. Renamed: `reference_window` (source `positive_control_F636`), `second_window` (source `D3`), `baseline_other_reading_direction` (source `null_other_direction`), the bar names (source `D3a`, `D3b`, `D3c`). New text: `about`, `first_bar`, `numeric_bar`, the window positions, `stored_map_repeated.note`. Local paths and the checks of the comparator itself are left out |
| `results/f647_forrest_wraps.json` | `experiments/2026-11-13-ves3959-s-forrest-wraps/results.json` (commit `af465b94`) | identical |
| `results/f524a3_training_run_and_0139_guard.json` | `experiments/2026-11-05-ves3090-hecate-rehearsal/attempt3/results.json` | identical |
| `results/f564_fail_1667.json` | `experiments/2026-11-07-ves3369-s1667-public/results.json` | identical |
| `results/f564a2_fail_public_arena_0139_0841.json` | `experiments/2026-11-07-ves3369-s1667-public/attempt2/results.json` | identical |

The result files keep internal set and task names (for example `paris4-ink-h4fresh5-ves3343`); they hold no paths.
The S checkpoint: `hecate_ft.pth` in the working repository, published unchanged as `hecate_onsheet_S.pth` on
https://huggingface.co/jo-tud/hecate-onsheet; sha256 `4b84fdf2f90cb369b254f04415d474bac8ab087a47022badd2d20fa32205b4f2`.
`MODEL_CARD.md`: the README of the model page. Source: `experiments/2026-11-13-ves4015-s-where-users/drafts/hf-model-card/README.md`
(commit `a2e34f4e`), plus two bullets under "Limits and named fails" (one-sided gain; an outside author's mesh). Its
numbers are those of `README.md`; the ink-sdk numbers are in `results/f4015_ink_sdk_profile.json`. The fresh-install
numbers under "Tested" stand only on the model page; they are in `results/f666_fresh_install_map.json`.
