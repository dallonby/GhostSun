"""Measure disc ellipticity of stacked FITS via limb gradient at many angles."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_stacks import read_fits_f32

def measure(path, label):
    img = read_fits_f32(path)
    h, w = img.shape
    thr = np.percentile(img, 75) * 0.4
    ys, xs = np.where(img > thr)
    cx, cy = xs.mean(), ys.mean()
    r0 = np.sqrt((img > thr).sum() / np.pi)
    # radial limb position at many angles: max |gradient| along ray
    angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
    radii = []
    rs = np.linspace(0.85*r0, 1.15*r0, 220)
    for a in angles:
        px = cx + rs*np.cos(a)
        py = cy + rs*np.sin(a)
        ok = (px > 1) & (px < w-2) & (py > 1) & (py < h-2)
        if ok.sum() < 50:
            radii.append(np.nan); continue
        vals = img[py[ok].astype(int), px[ok].astype(int)]
        g = np.gradient(vals)
        radii.append(rs[ok][np.argmin(g)])  # steepest falloff
    radii = np.array(radii)
    # fit r(theta) = r0 + a cos2t + b sin2t  (ellipticity term)
    okm = ~np.isnan(radii)
    A = np.c_[np.ones(okm.sum()), np.cos(2*angles[okm]), np.sin(2*angles[okm]),
              np.cos(angles[okm]), np.sin(angles[okm])]
    coef, *_ = np.linalg.lstsq(A, radii[okm], rcond=None)
    rbar, c2, s2 = coef[0], coef[1], coef[2]
    ell_amp = np.hypot(c2, s2)
    ang = 0.5*np.degrees(np.arctan2(s2, c2))
    # rx/ry equivalent for axis-aligned part
    rx = rbar + c2
    ry = rbar - c2
    print(f"{label:<24} r={rbar:7.1f}  ellipticity={2*ell_amp/rbar*100:5.2f}% "
          f"(axis {ang:+6.1f} deg)  rx/ry={rx/ry:.4f}")

base = os.path.dirname(os.path.abspath(__file__)) + r'\realstack'
measure(os.path.join(base, 'native8.fits'), 'native stack (fixed)')
measure(os.path.join(base, 'oldstack.fits'), 'old-path stack')
measure(os.path.join(base, 'native8-scan01.fits'), 'single scan (conic geom)')
