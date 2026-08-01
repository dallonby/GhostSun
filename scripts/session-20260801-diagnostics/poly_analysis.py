"""Is the center-trace residual smooth polynomial geometry or random wobble?"""
import os, glob
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
print(f"{'scan':<10} {'lin RMS':>8} {'quad RMS':>8} {'cubic RMS':>9} {'quintic':>8} {'sm-resid':>8}")
for p in sorted(glob.glob(os.path.join(OUT, 'resid_scan-*.npy'))):
    name = os.path.basename(p).replace('resid_', '').replace('.npy', '')
    d = np.load(p)
    fr, centers = d[:, 0], d[:, 3]
    f = (fr - fr.mean()) / fr.std()
    rms = {}
    for deg in (1, 2, 3, 5):
        c = np.polyfit(f, centers, deg)
        rms[deg] = np.std(centers - np.polyval(c, f))
    # residual after heavy smoothing removed (i.e. HF noise floor around smooth trend)
    c5 = np.polyval(np.polyfit(f, centers, 5), f)
    r = centers - c5
    k = 101
    lo = np.convolve(r, np.ones(k)/k, mode='same')
    hf = np.std((r - lo)[k:-k])
    print(f"{name:<10} {rms[1]:>8.1f} {rms[2]:>8.1f} {rms[3]:>9.1f} {rms[5]:>8.1f} {hf:>8.2f}")

# also: how consistent is the quintic trend between scans (after direction flip)?
trends = []
for p in sorted(glob.glob(os.path.join(OUT, 'resid_scan-*.npy'))):
    d = np.load(p)
    fr, centers = d[:, 0], d[:, 3]
    # normalize frame axis 0..1
    t = (fr - fr.min()) / (fr.max() - fr.min())
    c = np.polyfit(t, centers, 5)
    tt = np.linspace(0, 1, 200)
    trends.append(np.polyval(c, tt))
trends = np.array(trends)
# flip odd scans (bidirectional) to sky orientation, remove mean
sky = trends.copy()
for i in range(len(sky)):
    if i % 2 == 1:
        sky[i] = sky[i][::-1]
sky -= sky.mean(axis=1, keepdims=True)
mean_curve = sky.mean(axis=0)
spread = sky.std(axis=0)
print(f"\nsky-oriented trend curves: mean PP={mean_curve.max()-mean_curve.min():.1f}px  "
      f"scan-to-scan spread RMS={spread.mean():.1f}px  (PP of individual: "
      + ", ".join(f"{s.max()-s.min():.0f}" for s in sky) + ")")
