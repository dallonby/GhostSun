# Session diagnostics — 2026-08-01 stacking deep-dive

Rough, single-session analysis scripts from the native-domain stacking
redesign (branch `native-stacking`). Kept for reference while the stacking
work is iterated on; this folder is disposable and may be removed once the
findings are absorbed into the pipeline proper.

**Caveat: paths are hardcoded** to the capture session
`GhostSun-ha-lpf-runway-20260731\GhostSun Captures\scan-1785599603` and to a
local scratch directory. Edit the `BASE`/`base` constants (or pass an argv
where supported) before running elsewhere. Python 3 + numpy only.

## Raw-SER analysis (reads the .ser scans directly, no pipeline)

- `ser_probe.py` — dump SER headers (dims, bit depth, frame counts,
  timestamp-trailer presence).
- `raw_analysis.py` — dirty per-scan reconstruction from the Ha core row;
  measures per-scan disc extent, chord edges, scan-rate spread, per-frame
  flux ripple (transparency/clouds). Writes .npy intermediates the scripts
  below consume.
- `wobble_analysis.py` — circle fit to the limb-edge traces; scan-rate
  (px/frame) and edge residuals per scan.
- `shear_analysis.py` — separates the linear shear (bidirectional-scan
  drift, alternating sign) from residual center-trace wobble.
- `poly_analysis.py` — is the center-trace residual smooth (calibratable
  geometry) or noise; cross-scan repeatability of the trend curves.
- `render_dirty.py` — renders the dirty reconstructions to PNG with the
  measured centerline overlaid (found the polar-cap S-hooks).

## Output/stack analysis (reads the FITS products)

- `compare_stacks.py` — sharpness (HF energy, gradient), banding residual
  and disc geometry for stack vs stack vs single; writes crops.
- `roundness.py` — limb ellipticity via steepest-gradient radius at 180
  position angles, r(θ) = r0 + cos2θ/sin2θ fit.
- `circle_overlay.py` — draws the best-fit circle + measured limb points on
  the stacked disc (the "is it round" acceptance test); prints cardinal
  radii. Start here when judging shape.
- `crops2.py` — matched disc-relative crops with local contrast stretch.

Key findings these produced (details in the `F5.5` commit message): per-scan
conic geometry scatters ±10% in sx; scans are bidirectional with ±3° drift
shear; scan rate varies 10% between scans; the instrument has a nonlinear
plate-scale stretch near the slit ends; scan-07 of the session is
cloud-damaged (3.4% flux ripple).
