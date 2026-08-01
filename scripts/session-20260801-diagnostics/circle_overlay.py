"""Overlay a best-fit circle on the stacked disc + report r(theta) profile."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_stacks import read_fits_f32, write_png_gray8

base = os.path.dirname(os.path.abspath(__file__)) + r'\realstack'
img = read_fits_f32(os.path.join(base, 'native8.fits'))
h, w = img.shape

# limb radius vs angle by steepest gradient
thr = np.percentile(img, 75) * 0.4
ys, xs = np.where(img > thr)
cx, cy = xs.mean(), ys.mean()
r0 = np.sqrt((img > thr).sum() / np.pi)
angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
rs = np.linspace(0.85*r0, 1.15*r0, 260)
radii = np.full(len(angles), np.nan)
for i, a in enumerate(angles):
    px = cx + rs*np.cos(a); py = cy + rs*np.sin(a)
    ok = (px > 1) & (px < w-2) & (py > 1) & (py < h-2)
    if ok.sum() < 60: continue
    vals = img[py[ok].astype(int), px[ok].astype(int)]
    g = np.gradient(vals)
    radii[i] = rs[ok][np.argmin(g)]

okm = ~np.isnan(radii)
# refit center allowing decenter terms
A = np.c_[np.ones(okm.sum()), np.cos(angles[okm]), np.sin(angles[okm])]
coef, *_ = np.linalg.lstsq(A, radii[okm], rcond=None)
rbar = coef[0]
cx2 = cx + coef[1]; cy2 = cy + coef[2]
print(f"center=({cx2:.1f},{cy2:.1f}) rbar={rbar:.1f}")
for name, sl in [('E (x+)', 0), ('S (y+)', 90), ('W (x-)', 180), ('N (y-)', 270)]:
    i = sl
    seg = radii[(np.arange(i-8, i+9)) % 360]
    print(f"  {name}: r = {np.nanmean(seg):7.1f}  ({(np.nanmean(seg)-rbar)/rbar*100:+.2f}%)")

# render with sqrt stretch and draw circle + measured limb
lo, hi = np.percentile(img, [1, 99.8])
g8 = (np.sqrt(np.clip((img-lo)/(hi-lo), 0, 1)) * 255).astype(np.uint8)
tt = np.linspace(0, 2*np.pi, 6000)
for rr, val in [(rbar, 255)]:
    px = (cx2 + rr*np.cos(tt)).astype(int)
    py = (cy2 + rr*np.sin(tt)).astype(int)
    ok = (px >= 1) & (px < w-1) & (py >= 1) & (py < h-1)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            g8[py[ok]+dy, px[ok]+dx] = val
# measured limb dots (every 2 deg) in black for contrast
for i in range(0, 360, 2):
    if np.isnan(radii[i]): continue
    a = angles[i]
    px = int(cx + radii[i]*np.cos(a)); py = int(cy + radii[i]*np.sin(a))
    if 3 < px < w-4 and 3 < py < h-4:
        g8[py-3:py+4, px-3:px+4] = 0
small = g8[::2, ::2]
write_png_gray8(os.path.join(base, 'circle_overlay.png'), small)
print('overlay written')
