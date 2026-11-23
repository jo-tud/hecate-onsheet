"""Figure of one run_window.sh output dir: CT render, organisers' ink labels, frozen hecate, S (same window).

  python figure/make_figure.py OUT_DIR OUT_DIR/figure.png      (also writes OUT_DIR/figure.json)

Rows: CT | labels;  frozen hecate | S with its grey levels matched to hecate's;  S as the model gives it | caption.
CT: mean of the 4 middle layers of the VC3D render (planes 6-9 = depth -1.5 .. +1.5 vx), contrast stretched to its
1st-99th percentile. Labels: white = ink, grey = checked no ink, black = not labelled. Maps: the uint8 2D ink
probability of each model on the same linear grey scale (0 black, 1 white), no stretch.
Matched panel (display only): hecate's map is whiter than S's, so at a glance it looks clearer. match_levels()
re-maps the grey levels of the S map so that it has hecate's distribution of grey levels; every pixel keeps its
rank, so the two maps then differ only in where the ink is. The script stops if the re-mapping is not monotone,
and it writes the 2D AUC of the re-mapped map (same pixels as onsheet.py score) into the title and figure.json.
Needs numpy, scipy, zarr, pillow; a TrueType font (DejaVu Sans) if present, else PIL's default.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT, DST = Path(sys.argv[1]), Path(sys.argv[2])
PANEL = 600                       # px per panel; the full figure is 1260 px wide (2 panels + 3 gaps of 20 px)
GAP, HEAD = 20, 44
BG, INK, MUTED = (16, 16, 16), (235, 235, 235), (170, 170, 170)


def font(size, bold=False):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
              'DejaVuSans%s.ttf' % ('-Bold' if bold else '')):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def match_levels(src, ref):
    """Re-map the grey levels of uint8 map `src` to the distribution of uint8 map `ref` (same size), keeping ranks.
    Level v -> the smallest ref level whose cumulative count is >= (number of src px below v) + half the px at v.
    Returns (re-mapped map, the 256-entry table, largest gap of the two cumulative distributions afterwards)."""
    assert src.dtype == ref.dtype == np.uint8 and src.size == ref.size
    cnt = np.bincount(src.ravel(), minlength=256)
    cr = np.cumsum(np.bincount(ref.ravel(), minlength=256))
    lut = np.minimum(np.searchsorted(cr, np.cumsum(cnt) - cnt / 2, side='left'), 255).astype(np.uint8)
    assert (np.diff(lut.astype(int)) >= 0).all(), 'the grey-level table must never swap two levels'
    out = lut[src]
    gap = float(np.abs(np.cumsum(np.bincount(out.ravel(), minlength=256)) - cr).max() / src.size)
    return out, lut, gap


def wrap(d, text, f, width):
    lines, cur = [], ''
    for w in text.split():
        t = (cur + ' ' + w).strip()
        if cur and d.textlength(t, font=f) > width:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur]


def main():
    import argparse
    import zarr
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import onsheet
    res = json.loads((OUT / 'results.json').read_text())
    w = res['window']
    r0, c0, n = w['r0'], w['c0'], w['size']
    vol = np.asarray(zarr.open_group(str(OUT / 'render.zarr'), mode='r')['0'][:])[::-1]   # our plane order
    ct = vol[6:10].astype(np.float32).mean(0)
    lo, hi = np.percentile(ct, [1, 99])
    ct = np.clip((ct - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    lab = np.load(OUT / 'labels.npz')
    pos, neg = lab['pos'][r0:r0 + n, c0:c0 + n], lab['neg'][r0:r0 + n, c0:c0 + n]
    truth = np.zeros((n, n), np.uint8)
    truth[neg] = 90
    truth[pos] = 255
    a = res['auc_2d']
    mf, ms = np.load(OUT / 'map_frozen.npy'), np.load(OUT / 'map_S.npy')
    mm, lut, gap = match_levels(ms, mf)
    ok, wpos, wneg = onsheet.win_mask(OUT, argparse.Namespace(seg=res['segment'], r0=r0, c0=c0, size=n))
    a_m = onsheet.auc(mm.astype(np.float32)[wpos & ok], mm.astype(np.float32)[wneg & ok])
    three = (('frozen', mf), ('S', ms), ('S_matched', mm))
    bright = {k: float((m >= 128).mean()) for k, m in three}
    white = {k: float((m >= 230).mean()) for k, m in three}
    cgap = np.abs(np.cumsum(np.bincount(mm.ravel(), minlength=256)) - np.cumsum(np.bincount(mf.ravel(), minlength=256))) / mm.size
    gap10 = float(cgap[10:].max())                           # the same gap from grey level 10 up
    info = dict(what='display step of figure/make_figure.py: S map with its grey levels matched to frozen hecate',
                segment=res['segment'], window=w, auc_2d=dict(frozen=a['frozen'], S=a['S'], S_matched=a_m),
                auc_change_by_matching=a_m - a['S'], table_monotone=True, levels_of_S_merged=int(256 - len(set(lut[np.unique(ms)]))),
                largest_gap_of_cumulative_distributions=gap, level_of_largest_gap=int(cgap.argmax()),
                largest_gap_from_level_10_up=gap10, share_of_px_at_least_mid_grey=bright,
                share_of_px_near_white_230_or_more=white, mean_grey_level={k: float(m.mean()) for k, m in three},
                grey_level_table=[int(x) for x in lut])
    DST.with_suffix('.json').write_text(json.dumps(info, indent=1) + '\n')
    panels = [('CT (VC3D render, sheet layers)', ct),
              ('Ink labels (organisers, 20260918)', truth),
              (f'hecate, frozen: AUC {a["frozen"]:.3f}', mf),
              (f"S, hecate's grey levels (display): AUC {a_m:.3f}", mm),
              (f'S as the model gives it: AUC {a["S"]:.3f}', ms)]
    cap = [f'PHerc. Paris 4, scan 20260411134726 at 9.6 um (level 2), 78 keV. Segment {res["segment"]}, window '
           f'{n} x {n} px = {n * 0.0096:.1f} mm, rows {r0}-{r0 + n}, cols {c0}-{c0 + n}. This segment was held out '
           'from the S fine-tune.',
           f'2D ink AUC: labelled ink px ({res["n_ink_px"]:,}) vs checked no-ink px ({res["n_noink_px"]:,}); chance 0.5.',
           'Above right: the S map with its grey levels re-mapped to those of hecate. Every pixel keeps its rank; only '
           'the contrast changes (display only). Both maps then have the same share of dark, grey and white pixels '
           f'(largest gap of the two cumulative curves {gap:.3f}, at grey level {int(cgap.argmax())}; {gap10:.3f} from '
           'level 10 up) and differ only in where the ink is.',
           'Left: the S map as the model gives it. hecate and this map: probability 0-1 on the same linear grey '
           f'scale, no stretch. Near-white pixels (230 of 255 or more): hecate {white["frozen"]:.1%}, S {white["S"]:.1%}; '
           f'mean grey level {mf.mean():.0f} and {ms.mean():.0f}.']
    W = 2 * PANEL + 3 * GAP
    H = 3 * (PANEL + HEAD) + 4 * GAP
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    ft, fc = font(24, True), font(21)
    for i, (title, arr) in enumerate(panels):
        x = GAP + (i % 2) * (PANEL + GAP)
        y = GAP + (i // 2) * (PANEL + HEAD + GAP)
        assert d.textlength(title, font=ft) <= PANEL, title
        d.text((x, y + 6), title, fill=INK, font=ft)
        im = Image.fromarray(arr).resize((PANEL, PANEL), Image.Resampling.LANCZOS)
        img.paste(im.convert('RGB'), (x, y + HEAD))
    x, y = GAP + PANEL + GAP, GAP + 2 * (PANEL + HEAD + GAP) + HEAD        # the sixth slot holds the caption
    for par in cap:
        for line in wrap(d, par, fc, PANEL):
            d.text((x, y), line, fill=MUTED, font=fc)
            y += 30
        y += 14
    assert y <= H - GAP, 'caption does not fit its slot'
    img.save(DST)
    print(DST, img.size, f'matched AUC {a_m:.4f} (S {a["S"]:.4f}), largest gap {gap:.4f}')


if __name__ == '__main__':
    main()
