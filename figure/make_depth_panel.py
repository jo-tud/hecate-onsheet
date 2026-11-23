"""Depth panel: where the 3D ink of frozen hecate and of S sits along the sheet normal (the stored on-sheet result).

  python figure/make_depth_panel.py OUT.png

Draws from two files of this tree and runs no model:
  results/f561_depth_profile.json   ink mass per depth bin at the 224 labelled ink spots (2 held-out segments)
  results/f561_on_sheet_paris4.json the scored result (off-sheet share, intervals, baselines)
x = distance from the sheet found in the CT, in units of the local sheet spacing (0.1 per bin; the two end bins
also hold the mass beyond +-2). "Off the sheet" = more than half a spacing away, the shaded zones. y = share of the
model's own 3D ink mass (p >= 0.5) per bin; each curve sums to 1. The script sums the bins of the shaded zones and
stops if that sum differs from the scored off-sheet share, or if a zone's sum differs from its label.
Needs pillow; a TrueType font (DejaVu Sans) if present, else PIL's default.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DST = Path(sys.argv[1])
K = 2                                   # drawn at 2x and reduced, for smooth lines
W, GAP = 1260, 20                       # the same width and margin as figure.png
BG, ZONE, GRID = (16, 16, 16), (34, 34, 34), (52, 52, 52)
INK, MUTED = (235, 235, 235), (170, 170, 170)
COL = {'frozen': (57, 135, 229), 'S': (217, 89, 38)}        # blue, orange (checked for colour-blind separation)
NAME = {'frozen': 'hecate, frozen', 'S': 'S (on-sheet fine-tune)'}
YMAX, YT = 0.25, (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
XT = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def font(size, bold=False):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
              'DejaVuSans%s.ttf' % ('-Bold' if bold else '')):
        try:
            return ImageFont.truetype(p, size * K)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    prof = json.loads((ROOT / 'results/f561_depth_profile.json').read_text())
    res = json.loads((ROOT / 'results/f561_on_sheet_paris4.json').read_text())
    maps = res['readings']['maps']
    edges, n = prof['bin_edges'], len(prof['bin_edges']) - 1
    lo, hi = edges.index(-0.5), edges.index(0.5)             # bins [0, lo) and [hi, n) are the shaded zones
    for m in COL:                       # the picture and the scored number must be the same thing
        b = prof['models'][m]['share_per_bin']
        assert abs(sum(b[:lo]) + sum(b[hi:]) - maps[m]['L_snap']) < 1e-9, m          # the drawn bins, summed
        assert abs(sum(b[:lo]) - prof['models'][m]['inward_off']) < 1e-9, m          # the two zone labels
        assert abs(sum(b[hi:]) - prof['models'][m]['outward_off']) < 1e-9, m
        assert abs(sum(b) - 1) < 1e-9 and max(b) < YMAX, m
    nlab = res['criteria']['F0']['power']['n_label']
    assert nlab == prof['n_label_sites']
    sp = prof['sheet_spacing_vx']['median']
    P = {m: prof['models'][m] for m in COL}
    bars = res['criteria']['F1']['bars']
    cap = [f'PHerc. Paris 4, scan 20260411134726 at 9.6 um (level 2), 78 keV. {nlab} labelled ink spots on 2 segments held out',
           'from the S fine-tune. Ink mass = 3D probability where p >= 0.5, summed per depth; each curve sums to 1.',
           f'0 = the sheet found in the CT; shaded = more than half a sheet spacing away (median spacing {sp:.1f} vx = {sp * 9.6:.0f} um).',
           'Off the sheet: S {:.3f} [{:.3f}, {:.3f}], hecate {:.3f} [{:.3f}, {:.3f}]. Baselines: S\'s own map moved off the'.format(
               maps['S']['L_snap'], *maps['S']['L_snap_ci'], maps['frozen']['L_snap'], *maps['frozen']['L_snap_ci']),
           f'sheet {bars["out"]:.3f} / {bars["in"]:.3f}; ink spread evenly over the window {maps["S"]["L0_snap"]:.3f}. The end bins hold all mass beyond +-2.',
           'The gain is on the outward side (+ along the segment normal). Inward S has more ink off the sheet than',
           'hecate, and its peak is wider ({:.1f} vs {:.1f} vx at half maximum); side values are point readings, no interval.'.format(
               P['S']['fwhm_vx'], P['frozen']['fwhm_vx'])]
    HEAD, LEG, PLOT_H, AX = 44, 72, 400, 76
    H = GAP + HEAD + LEG + PLOT_H + AX + 30 * len(cap) + 10 + GAP
    img = Image.new('RGB', (W * K, H * K), BG)
    d = ImageDraw.Draw(img)
    ft, fl, fs, fc = font(24, True), font(20), font(17), font(21)

    def text(x, y, s, f, fill, anchor='la'):
        d.text((x * K, y * K), s, fill=fill, font=f, anchor=anchor)

    text(GAP, GAP + 6, 'Where the 3D ink sits along the sheet normal', ft, INK)
    y = GAP + HEAD
    x = GAP
    for m in ('frozen', 'S'):           # legend: coloured line, text in the text colour
        d.line([(x * K, (y + 14) * K), ((x + 34) * K, (y + 14) * K)], fill=COL[m], width=3 * K)
        s = f'{NAME[m]}: {maps[m]["L_snap"]:.3f} of its ink off the sheet'
        text(x + 44, y + 2, s, fl, INK)
        x += 44 + d.textlength(s, font=fl) / K + 40
    x0, x1 = GAP + 70, W - GAP - 10     # plot area
    y0, y1 = y + LEG, y + LEG + PLOT_H

    def px(u):
        return x0 + (u - edges[0]) / (edges[-1] - edges[0]) * (x1 - x0)

    def py(v):
        return y1 - v / YMAX * (y1 - y0)

    for a, b in ((edges[0], -0.5), (0.5, edges[-1])):         # the off-sheet zones
        d.rectangle([px(a) * K, y0 * K, px(b) * K, y1 * K], fill=ZONE)
    for v in YT:
        d.line([(x0 * K, py(v) * K), (x1 * K, py(v) * K)], fill=GRID, width=K)
        text(x0 - 10, py(v), f'{v:.2f}', fs, MUTED, 'rm')
    for u in XT:
        d.line([(px(u) * K, y1 * K), (px(u) * K, (y1 + 6) * K)], fill=MUTED, width=K)
        text(px(u), y1 + 10, ('%+.1f' % u) if u else '0', fs, MUTED, 'ma')
    for u in (-0.5, 0.5):
        d.line([(px(u) * K, y0 * K), (px(u) * K, y1 * K)], fill=MUTED, width=K)
    for m in ('frozen', 'S'):           # step lines, one step per bin
        v = P[m]['share_per_bin']
        pts = []
        for i in range(n):
            pts += [(px(edges[i]) * K, py(v[i]) * K), (px(edges[i + 1]) * K, py(v[i]) * K)]
        d.line(pts, fill=COL[m], width=2 * K, joint='curve')
    text(px(0), y0 + 8, 'on the sheet', fs, MUTED, 'ma')
    for u, side, key in ((-1.25, 'inward, off the sheet', 'inward_off'), (1.25, 'outward, off the sheet', 'outward_off')):
        text(px(u), y0 + 8, side, fs, MUTED, 'ma')
        yy = y0 + 40
        for m in ('frozen', 'S'):       # direct labels: the share of each model's ink in this zone
            s = f'{"hecate" if m == "frozen" else "S"} {P[m][key]:.3f}'
            wt = d.textlength(s, font=fl) / K
            xs = px(u) - (wt + 24) / 2
            d.line([(xs * K, (yy + 12) * K), ((xs + 16) * K, (yy + 12) * K)], fill=COL[m], width=3 * K)
            text(xs + 24, yy, s, fl, INK)
            yy += 28
    text(x0, y1 + 40, 'inward', fs, MUTED, 'la')
    text(x1, y1 + 40, 'outward', fs, MUTED, 'ra')
    text((x0 + x1) / 2, y1 + 40, 'distance from the sheet found in the CT, in sheet spacings', fs, INK, 'ma')
    text(GAP, y0 - 26, 'share of ink mass per 0.1 spacing', fs, MUTED)
    yc = y1 + AX
    for line in cap:
        text(GAP, yc, line, fc, MUTED)
        yc += 30
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    DST.parent.mkdir(parents=True, exist_ok=True)
    img.save(DST)
    print(DST, img.size)


if __name__ == '__main__':
    main()
