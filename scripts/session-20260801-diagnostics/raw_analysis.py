"""Dirty reconstruction + geometry/dither analysis straight from raw SER files."""
import struct, os, glob, sys
import numpy as np

BASE = r'C:\Users\djall\Documents\Codex\2026-07-23\i\outputs\GhostSun-ha-lpf-runway-20260731\GhostSun-Windows-x64\GhostSun Captures\scan-1785599603'
OUT = os.path.dirname(os.path.abspath(__file__))
W, H, HDR = 3840, 120, 178
FRAME_BYTES = W * H * 2

def mean_spectrum(path, frame_idx):
    """Return mean spectrum (H values) of one frame, averaged along slit."""
    with open(path, 'rb') as f:
        f.seek(HDR + frame_idx * FRAME_BYTES)
        fr = np.frombuffer(f.read(FRAME_BYTES), dtype='<u2').reshape(H, W)
    return fr.astype(np.float64).mean(axis=1), fr

def extract_rows(path, rows, n_frames):
    """Extract given spectral rows from every frame. Returns dict row -> (n_frames, W) array."""
    out = {r: np.empty((n_frames, W), dtype=np.uint16) for r in rows}
    row_bytes = W * 2
    with open(path, 'rb') as f:
        for i in range(n_frames):
            base = HDR + i * FRAME_BYTES
            for r in rows:
                f.seek(base + r * row_bytes)
                out[r][i] = np.frombuffer(f.read(row_bytes), dtype='<u2')
    return out

def frames_in(path):
    with open(path, 'rb') as f:
        h = f.read(178)
    return struct.unpack_from('<i', h, 38)[0]

paths = sorted(glob.glob(os.path.join(BASE, 'scan-*.ser')))

# --- locate Ha core row from a mid-scan frame of scan-01 ---
n0 = frames_in(paths[0])
spec, frame0 = mean_spectrum(paths[0], n0 // 2)
core = int(np.argmin(spec))
# continuum reference: brightest row reasonably far from core
cont_candidates = [r for r in range(H) if abs(r - core) > 25]
cont = int(max(cont_candidates, key=lambda r: spec[r]))
print(f"spectral rows: core(min)={core}  continuum(ref)={cont}")
print(f"mean spectrum min/max: {spec.min():.0f}/{spec.max():.0f}")
np.save(os.path.join(OUT, 'mean_spectrum.npy'), spec)
np.save(os.path.join(OUT, 'sample_frame.npy'), frame0[:, ::4])

results = []
for p in paths:
    n = frames_in(p)
    rows = extract_rows(p, [core, cont], n)
    ha = rows[core].astype(np.float32)     # (n, 3840) dirty Ha reconstruction
    co = rows[cont].astype(np.float32)

    # per-frame integrated flux (transparency/banding diagnostic), disc frames only
    flux = ha.sum(axis=1)
    thresh = flux.max() * 0.25
    on_disc = flux > thresh
    idx = np.where(on_disc)[0]
    first, last = idx[0], idx[-1]

    # per-frame chord edges at half-max (sub-pixel by linear interp)
    prof = ha
    edges = np.full((n, 2), np.nan)
    for i in range(first, last + 1):
        row = prof[i]
        mx = row.max()
        if mx < 200: continue
        half = mx * 0.5
        above = row > half
        w = np.where(above)[0]
        if len(w) < 50: continue
        l, r = w[0], w[-1]
        if l > 0:
            edges[i, 0] = l - 1 + (half - row[l-1]) / (row[l] - row[l-1] + 1e-9)
        if r < W - 1:
            edges[i, 1] = r + (half - row[r]) / (row[r+1] - row[r] + 1e-9)

    valid = ~np.isnan(edges[:, 0])
    centers = (edges[:, 0] + edges[:, 1]) / 2
    chords = edges[:, 1] - edges[:, 0]
    vi = np.where(valid)[0]
    # fit circle to edge trace: chord(t) = 2*sqrt(R^2-(t-t0)^2 * s^2)  -> fit in (frame, edge) space
    # simpler: disc height in frames vs width in px
    disc_frames = last - first + 1
    max_chord = np.nanmax(chords)
    cx = np.nanmean(centers[vi])

    # transparency variation among disc frames (normalized flux vs smooth trend)
    f_disc = flux[first:last+1]
    trend = np.convolve(f_disc, np.ones(101)/101, mode='same')
    ripple = (f_disc / (trend + 1e-9))
    ripple_std = np.nanstd(ripple[50:-50])

    results.append(dict(name=os.path.basename(p), n=n, first=int(first), last=int(last),
                        disc_frames=int(disc_frames), max_chord=float(max_chord),
                        center_x=float(cx), ripple=float(ripple_std)))
    np.save(os.path.join(OUT, f"dirty_{os.path.basename(p)}.npy"), ha[:, ::2][::1])
    np.save(os.path.join(OUT, f"edges_{os.path.basename(p)}.npy"), edges)
    np.save(os.path.join(OUT, f"flux_{os.path.basename(p)}.npy"), flux)
    print(f"{os.path.basename(p)}: frames={n} disc=[{first}..{last}] ({disc_frames} fr) "
          f"maxchord={max_chord:.1f}px centerx={cx:.1f} fluxripple={ripple_std*100:.2f}%")

# aspect ratio: px per frame implied if disc circular
for r in results:
    print(f"{r['name']}: px/frame if circular = {r['max_chord']/r['disc_frames']:.4f}")
