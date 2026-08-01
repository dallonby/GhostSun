"""Separate linear shear from residual wobble in the limb traces."""
import os, glob
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

print(f"{'scan':<10} {'shear px/fr':>11} {'tilt_deg':>8} {'wobRMS':>7} {'wobPP':>7} {'hfRMS':>6}  notes")
for p in sorted(glob.glob(os.path.join(OUT, 'resid_scan-*.npy'))):
    name = os.path.basename(p).replace('resid_', '').replace('.npy', '')
    d = np.load(p)          # cols: frame, left-cx, right-cx, center
    fr, centers = d[:, 0], d[:, 3]

    # linear shear fit to chord centers
    A = np.c_[fr, np.ones(len(fr))]
    (slope, icpt), *_ = np.linalg.lstsq(A, centers, rcond=None)
    resid = centers - (slope * fr + icpt)

    # low-frequency wobble (mount) vs high-frequency (seeing) split at ~50-frame smoothing
    k = 51
    lo = np.convolve(resid, np.ones(k)/k, mode='same')
    lo[:k] = lo[k]; lo[-k:] = lo[-k-1]
    hi = resid - lo
    tilt = np.degrees(np.arctan(slope / 0.91))  # approx px/frame scale
    print(f"{name:<10} {slope:>11.4f} {tilt:>8.2f} {np.std(lo):>7.2f} {lo.max()-lo.min():>7.1f} {np.std(hi):>6.2f}")
    np.save(os.path.join(OUT, f'wobble_{name}.npy'), np.c_[fr, resid, lo, hi])
