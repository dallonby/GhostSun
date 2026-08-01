"""Within-scan geometry: fit circle to limb edge traces, measure residual wobble."""
import os, glob
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

def fit_circle(x, y):
    """Algebraic circle fit (Kasa). Returns cx, cy, R."""
    A = np.c_[2*x, 2*y, np.ones(len(x))]
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    R = np.sqrt(sol[2] + cx**2 + cy**2)
    return cx, cy, R

print(f"{'scan':<12} {'pxPerFrame':>10} {'R_px':>8} {'edgeRMS':>8} {'edgeP95':>8} {'ctrDriftPP':>10}")
for p in sorted(glob.glob(os.path.join(OUT, 'edges_scan-*.ser.npy'))):
    name = os.path.basename(p).replace('edges_', '').replace('.npy', '')
    edges = np.load(p)
    n = len(edges)
    valid = ~np.isnan(edges[:, 0]) & ~np.isnan(edges[:, 1])
    idx = np.where(valid)[0]
    # exclude extreme polar caps where the edge is tangent-smeared (keep middle 90% of chord range)
    centers = (edges[:, 0] + edges[:, 1]) / 2
    chords = edges[:, 1] - edges[:, 0]
    maxc = np.nanmax(chords)
    good = valid & (chords > 0.30 * maxc)
    gi = np.where(good)[0]

    # build point sets from both edges; frame axis scaled by unknown s (px/frame)
    # solve for s by requiring circle fit residual minimal — search coarse then fine
    best = None
    for s in np.linspace(0.80, 1.00, 41):
        x = np.r_[edges[gi, 0], edges[gi, 1]]
        y = np.r_[gi, gi] * s
        cx, cy, R = fit_circle(x, y)
        res = np.sqrt((x - cx)**2 + (y - cy)**2) - R
        rms = np.sqrt(np.mean(res**2))
        if best is None or rms < best[1]:
            best = (s, rms)
    s0 = best[0]
    for s in np.linspace(s0 - 0.01, s0 + 0.01, 41):
        x = np.r_[edges[gi, 0], edges[gi, 1]]
        y = np.r_[gi, gi] * s
        cx, cy, R = fit_circle(x, y)
        res = np.sqrt((x - cx)**2 + (y - cy)**2) - R
        rms = np.sqrt(np.mean(res**2))
        if rms < best[1]:
            best = (s, rms)
    s = best[0]
    x = np.r_[edges[gi, 0], edges[gi, 1]]
    y = np.r_[gi, gi] * s
    cx, cy, R = fit_circle(x, y)
    res = np.sqrt((x - cx)**2 + (y - cy)**2) - R
    rms = np.sqrt(np.mean(res**2))
    p95 = np.percentile(np.abs(res), 95)

    # center drift: smooth the center trace, peak-to-peak of low-freq component
    c = centers[gi]
    k = 201
    if len(c) > k:
        sm = np.convolve(c, np.ones(k)/k, mode='valid')
        drift_pp = sm.max() - sm.min()
    else:
        drift_pp = np.nan
    print(f"{name:<12} {s:>10.4f} {R:>8.1f} {rms:>8.2f} {p95:>8.2f} {drift_pp:>10.1f}")
    np.save(os.path.join(OUT, f'resid_{name}.npy'),
            np.c_[gi, edges[gi,0]-cx, edges[gi,1]-cx, centers[gi]])
