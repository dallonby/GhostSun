"""Local-contrast crops at matched disc-relative positions."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_stacks import read_fits_f32, write_png_gray8, disc_mask

base = os.path.dirname(os.path.abspath(__file__)) + r'\realstack'
out = base

def crop_local(img, cx, cy, r, fx, fy, size, path):
    """Crop centered at disc-relative (fx, fy) in units of r, local stretch."""
    x0 = int(cx + fx * r - size // 2)
    y0 = int(cy + fy * r - size // 2)
    x0 = max(0, min(x0, img.shape[1] - size))
    y0 = max(0, min(y0, img.shape[0] - size))
    c = img[y0:y0 + size, x0:x0 + size]
    lo, hi = np.percentile(c, [2, 99.5])
    g = np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1)
    write_png_gray8(path, (g * 255).astype(np.uint8))

for fn, label in [('native8.fits', 'new'), ('oldstack.fits', 'old'), ('native8-scan01.fits', 'single')]:
    p = os.path.join(base, fn)
    img = read_fits_f32(p)
    _, (cx, cy, r) = disc_mask(img)
    # active-region area: the plage complex sits up-left of center in these
    crop_local(img, cx, cy, r, -0.15, -0.35, 700, os.path.join(out, f'cmp_{label}_ar.png'))
    # filament zone lower center
    crop_local(img, cx, cy, r, -0.05, 0.35, 700, os.path.join(out, f'cmp_{label}_fil.png'))
    print(label, 'ok', f'c=({cx:.0f},{cy:.0f}) r={r:.0f}')
