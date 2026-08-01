# Hand-off: native-domain stacking work (branch `native-stacking`)

You are picking up an in-progress redesign of GhostSun's multi-scan
stacking on a new machine. This file is the working context; read it with
the folder README (script inventory) and the `F5.5` commit message.

## Where things stand

All of the following is implemented, tested (28 unit tests green) and
validated on real data, in `crates/ghostsun-core/src/stack.rs` unless
noted:

- `stack_native()` — native-domain stacking: inter-scan registration
  (similarity + robust residual affine + optional evolution flow) composed
  with each scan's geometric warp; the corrected native disk is
  Lanczos-resampled exactly once per scan. Entry points: app
  `process_scans` (ghostsun-app/src/acquire.rs), CLI `stack-ser`
  (ghostsun-cli/src/main.rs), bench `Stack-native` rows.
- `photometric_geom()` — whole-disc flux-profile geometry (half-max
  crossings + midchord line) replacing the limb-conic fit for stacking.
  Motivation: the conic fits (`ellipse::fit_robust`) scatter ±10% in sx and
  hundreds of px in center on real full-disc scans, so every scan warps
  differently and nothing registers. Radius pinned to session median.
- Self-calibration on a rendered probe scan: `measure_y_distortion()`
  (cubic slit-end plate-scale distortion, folded into the same resample
  via `apply_ydist`) then `limb_axis_ratio()` (residual rx/ry folded into
  sx).
- Consensus-gated reference selection (top-3 sharpness candidates,
  majority vote) — a cloud-wrecked scan can carry the highest HF energy in
  its artifacts and, as reference, veto every healthy scan.
- Per-column quality weights (burst severity × transparency gains) carried
  through each scan's transform; MAD-clipped weighted mean; a second pass
  re-registers everything against the first stack.
- `ReconOptions::keep_native` exposes the corrected pre-warp disk +
  geometry + burst severities in `ReconReport`.
- CLI `stack-ser` has a `.recon-cache` sidecar cache (native + warped FITS
  + meta.txt per scan, keyed by SER path hash, invalidated on mtime/
  options/version): iterations cost ~3 min instead of ~20.

Validation session: 8 Ha scans, `GhostSun-ha-lpf-runway-20260731\
GhostSun Captures\scan-1785599603` (~29 GB, copied separately from the old
machine). Result: 6/8 scans stack (old path: 3/8 including a FALSELY
MIRRORED scan blended in), cardinal limb radii agree to ~0.8% (old path
15.6% elliptical), banding metric halved vs a single scan.

Reproduce:

    ghostsun stack-ser scan-01.ser ... scan-08.ser --out-dir out \
      --name native8 --reverse "1,3,5,7" --dispersion vertical

`GS_STACK_DEBUG=1` prints per-registration diagnostics. `--dispersion
vertical` is REQUIRED for these 3840x120 hardware-ROI SERs (auto-detect
fails on the short spectral axis). `--reverse` lists 0-based indices of
reverse-direction scans (session is bidirectional, odd indices here).

## THE TOP OPEN ITEM — trust the user's eye, not the summary metrics

The user judges the stacked disc as still visibly NOT disc-like, despite
cardinal radii agreeing to ~0.8%. Their visual judgment has been right
twice already when summary metrics said "fine" (they caught the 4% pumpkin
and the south-limb pear bulge). Do not argue from the cardinal numbers.
First action on this machine: plot the FULL 360° r(θ) curve
(`circle_overlay.py` computes it; plot it rather than printing 4 points)
and look for the mode the cardinal points average away — cos3θ/cos4θ
residual, a flat at the north cap (measured −1.8%), or an asymmetric
limb-darkening halo that reads as shape even when the hard limb is round.
Overlay the circle on a hard-stretched rendering too: the perceived shape
may live in the halo, not the limb.

## Known facts about the data/instrument (measured, do not re-derive)

- Disc width along slit stable to 0.1% (3238±3 px at half-max); scan rate
  varies 10% BETWEEN scans; scans alternate direction with ±3° drift
  shear (sign flips with direction — physical, not error).
- SERs carry NO per-frame timestamps (trailer=0) — worth fixing in
  acquisition.
- Nonlinear plate-scale stretch near the slit ends (the pear bulge no
  affine can fix); slit ends also defocus (user's earlier polar-blur
  question — same optical zone).
- scan-07 is cloud-damaged (3.4% frame flux ripple; wrecked recon with an
  artifact band that wins HF-energy rankings). scan-08 ran ~10% slower
  than scans 1–6 (real, sx ≈ 1.22 vs 1.13 cluster).
- The per-scan reconstruction FITS from the OLD pipeline are untrusted
  (user's explicit instruction) — work from raw SERs.

## Open queue, in rough priority

1. The shape complaint above.
2. scan-04 and scan-08 refuse registration just under the NCC_MIN = 0.85
   gate on pass 0 (scan-04 recovers in pass 1). Consider gating on
   post-residual-affine NCC, or a slightly lower gate + verification.
3. Port photometric geometry into single-scan `reconstruct` — individual
   reconstructions are still ~11% elliptical eggs (the conic fit). This
   also fixes the app's per-scan products.
4. Stack-level deconvolution (SNR now supports it) to recover the softness
   vs the sharpest single scan.
5. Phase 1: forward drizzle onto a 2× grid (px/frame 0.84–0.93 native
   oversampling + real inter-scan dither of ~40 px are still uncashed).
   TODO.md F4/F12/F13 describe the intended design.
6. The warped-domain `stack::stack` still has the HF-energy reference flaw
   — port the consensus gate or deprecate that path.
7. `stack-ser` cache: consider hashing ReconOptions into the meta so tune
   changes invalidate correctly (currently version/flip/dispersion/mtime).

## Repo state notes

- Branch `native-stacking`: commit 1 = user's unrelated WIP focus/camera
  work (carried for transfer, unreviewed); commit 2 = the stacking work;
  commit 3 = these scripts. `acquire.rs` contains both workstreams.
- This diagnostics folder is disposable by design (user's call) — delete
  once absorbed.
