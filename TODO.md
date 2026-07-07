# GhostSun — Advanced Features Roadmap & Implementation TODO

> **STATUS 2026-07-07: Phase 0 and F1–F8 are implemented and benchmarked**
> (see CLAUDE.md §2 for the acceptance numbers and deviations). This document
> remains the specification: consult it when tuning, extending, or debugging
> any of these stages. Known deviations from the gates: F1 met its goal via
> band-4 SNR (+1.3 dB) and SSIM rather than raw PSNR; F6 met the resolution
> gate (limbσ −32%) but not the PSNR gate and stays opt-in; F8 is the
> lightweight block-coordinate variant (registration/gain blocks only).

Audience: a junior developer implementing these without the original author.
Read `CLAUDE.md` first — especially §2 (the measurement discipline) and §4
(hard-won lessons). Nothing below gets merged without passing the harness
gates defined for it, and every stage keeps the project invariants:

- **f32 end-to-end, quantize once at output.**
- **Every new correction/enhancement is a flag-gated stage** with an
  ablation row in `bench`, a **clean-injection audit** (must add ~nothing on
  `synth --clean`), a **deadband/gate** wherever an estimator's noise could
  be injected, and a **robustness sweep** (≥3 seeds + one harsh case) before
  it's declared done.
- **Never touch the `--baseline` (INTI-faithful) path** — it exists to stay
  faithfully bad for comparison.
- Extraction-side features must add their degradation to `synth.rs` FIRST,
  with ground truth recorded in `SynthTruth`, so the harness can score the
  fix. If the synth doesn't model the problem, the feature cannot be tuned.

Dependency order (later items assume earlier ones):

```
Phase 0  Harness upgrades (everything below depends on this)
Phase 1  F1 Profile-model extraction  →  F2 Doppler compensation  →  F3 Flexure tracking
Phase 2  F4 Drizzle reconstruction
Phase 3  F5 Multi-scan stacking
Phase 4  F6 PSF estimation + deconvolution
Phase 5  F7 Post-hoc denoising (Anscombe+wavelet; optional Noise2Void)
Phase 6  F8 Global MAP joint inversion (research track)
```

---

## Phase 0 — Harness upgrades (do this first, in full)

The current harness scores intensity fidelity only. The new features need new
degradations, new truth channels, new metrics, and a parameter-sweep driver.

### 0.1 Synth: new physical effects (`src/synth.rs`)

Add to `SynthParams` (all default OFF or matching current behavior so
existing reference numbers stay comparable):

- `doppler: bool` — a velocity field over the disk: solar rotation
  (linear in x, ~±2 km/s at limbs → convert to px via a new
  `dispersion_kms_per_px` param, default ≈ 5 km/s/px so rotation ≈ ±0.4 px)
  plus small-scale turbulence (reuse `fbm`, amplitude ~0.15 px, wavelength
  ~20 px). Shift the absorption-line center by `v(x,y)` px in the frame
  renderer. **Record the truth velocity map** rendered on the ground-truth
  grid (write `ground_truth_velocity.png`, scaled ±1 px ↔ 0..65535, and note
  the scale in a sidecar `truth.txt`).
- `flexure_px: f64` — slow spectral drift of the whole line over the scan:
  `flex(t) = flexure_px * (0.6*sin(2πt/N*1.3) + 0.4*t/N)` added to the line
  center of every row of frame t. Record `flex: Vec<f64>` in `SynthTruth`.
  Default 0.15 px (realistic for a 90 s scan), 0 when `--clean`.
- `psf_seeing_px: f64` — blur the *sun* before sampling: apply a Gaussian
  blur of this sigma to the sampled sun coordinates... practical approach:
  when evaluating `core_intensity`/`continuum_intensity` per (frame,row),
  average over a small fixed set of offsets (e.g. 5×5 Gauss-Hermite-ish
  stencil scaled by sigma). Also model the **slit boxcar in X**: average the
  sun over `slit_px` (default = scan_step, i.e. contiguous coverage) in the
  scan direction. Record both in `SynthTruth`. NOTE: the ground-truth image
  stays UNBLURRED — deconvolution (F6) is scored by how far back toward the
  unblurred truth it gets; all pre-F6 features should be scored against a
  blurred copy of the truth (add `--truth-blurred` output) so they aren't
  penalized for honest optics. `eval` gains a `--truth` selector.
- `n_scans: usize` — emit K sequential SER files of the *same* sun with:
  per-scan fresh jitter/transparency/noise seeds, a small global pointing
  offset (±5 px), a small evolution term (advect the fbm texture phase by a
  few px per scan, rotate prominence brightness ±10%) to make naive stacking
  fail and flow-compensated stacking win. Files `synth_scan{k}.ser`.
- `oversample`: already controllable via `scan_step`; add a preset comment.

### 0.2 Eval: new metrics (`src/metrics.rs`)

- **Velocity RMS** (for F2): `eval-velocity recon_velocity.png truth_velocity.png`
  → RMS error in px (and km/s given dispersion), disk interior only.
- **Resolution metric** (for F4/F6): fit the erf edge width across the limb
  of the aligned reconstruction (reuse `limb.rs` centroid machinery, but fit
  sigma); report `limb_sigma_px`. Lower = sharper. Guards deconvolution
  against "sharpened noise, blurred signal" outcomes when read together with
  PSNR.
- **Power-spectrum fidelity** (for F1/F6/F7 anti-hallucination): radially
  averaged power spectrum of (recon − truth) residual vs of truth, reported
  in 4 spatial-frequency bands. A denoiser that erases real fibrils shows up
  as band-3/4 residual power ≈ truth power (it deleted the signal); a
  hallucinating one shows excess. Print as `band_snr = [b1,b2,b3,b4]` dB.
- **Photometric flatness** (for F3): after intensity match, the mean of
  (recon/truth) in 16 radial×azimuthal sectors; report max sector deviation
  in %. Catches slow photometric waves that PSNR dilutes.

### 0.3 Bench: sweep driver + regression ledger (`src/main.rs`)

- `bench --sweep "stage.param=v1,v2,v3"` — run GhostSun-full with one tuning
  parameter overridden across values, print one table row per value. Wire via
  a `TuneParams` struct (all the magic numbers listed per-feature below move
  into it, with current values as defaults) threaded through `ReconOptions`.
  This is THE tool for optimizing every feature; do not hand-tune by editing
  constants.
- `bench --json out.json` — dump the results table machine-readably; append
  a dated entry to `bench_history.jsonl` (gitignored) so regressions are
  visible over time.
- Add the new metrics columns to the bench table: `limbσ`, `vRMS` (when
  doppler on), `flat%`, `band4`.

### 0.4 Acceptance for Phase 0

- All current reference numbers reproduce unchanged with new params at
  defaults (CLAUDE.md §2 table).
- Each new degradation, enabled alone, measurably hurts GhostSun-full (write
  the numbers into CLAUDE.md as the new "unsolved" baselines each feature
  must then recover).

---

## F1 — Profile-model spectral extraction (low-rank denoising)

**Goal:** replace single-wavelength B-spline sampling with extraction from a
fitted/denoised line-profile model per (row, frame). Expected +2–4 dB
PSNR-disk on the noisy bench at unchanged bandpass purity.

**Physics/why:** each of the ~500k spectra in the cube is the same absorption
profile with 4–5 slowly varying parameters. ~20 wavelength samples inform
each output pixel; we currently use ~1. Projecting onto a low-rank profile
basis averages noise down by ~√(effective samples) without mixing
off-core wavelengths (the model *knows* the core; INTI's 7-tap kernel does
not).

**Design (two stages, do A then B):**

- **A. Parametric fit:** for each (row, frame), fit
  `S(λ) = C·(1 − d·φ((λ−μ)/σ))` with φ Gaussian; reuse
  `mathutil::fit_inverted_gaussian` but with these constraints: μ seeded from
  the smile polynomial ±1.5 px; σ shared per-row (fit σ on the mean image
  per row once, hold fixed per frame — per-spectrum σ is too noisy); so per
  spectrum only (C, d, μ) — a 3-param fit, 2 of them linear given μ. Solve μ
  by 1-D Newton on the profile-correlation peak, then (C,d) linearly.
  Core intensity = `C·(1 − d)` — that is the new output pixel value.
  Windows: ±`w_fit` px around the smile position (default 8; tune 6..12).
- **B. PCA residual basis (the "cutting-edge" half):** collect residuals
  (observed − parametric model) over a large sample (every 4th row/frame),
  compute the top-K eigenprofiles (K default 3, tune 0..6) via simple
  covariance + our `eig3`-style solver generalized (or power iteration —
  fine at 20-dim), then refit each spectrum as parametric + Σ aₖ·PCₖ with
  the aₖ linear. Handles real-world profile asymmetries the Gaussian misses
  (Hα wings). K=0 must exactly reproduce stage A.
- Off-disk pixels (C below threshold): fall back to current B-spline
  sampling (prominences are emission — the absorption model is wrong there).
  Gate by continuum level, taper the blend over ~2 px of threshold.

**Files:** new `src/profile.rs`; `extract.rs` gains
`ExtractMode::{BSpline, Profile}`; `ReconOptions.extract_mode` (default
Profile once accepted); baseline path untouched.

**Tuning params (into `TuneParams`):** `w_fit`, `pca_k`, `sigma_row_smooth`,
off-disk gate threshold + taper width.

**Harness gates:**
- Noisy bench: PSNR-disk ≥ current +2 dB, SSIM up, `band_snr` band 3–4 must
  IMPROVE (proves it's denoising, not smoothing).
- `synth --clean`: PSNR within 1 dB of B-spline extraction (model bias check;
  the Gaussian-on-Gaussian synth makes this easy — real profiles come later).
- Prominences: visually confirm off-limb emission survives; add an
  `eval` mask metric if regression suspected (mean absolute error over
  truth>0 off-disk pixels).
- Sweep `pca_k` 0..6 and `w_fit` 6..12; record the table in the PR notes.

**Pitfalls:** don't let the μ fit wander to telluric/adjacent lines (clamp to
±1.5 px of smile); don't fit σ per spectrum (noise blowup — this WILL look
fine on clean data and destroy noisy data); watch for bias at the extraction
window edges near the image border (clamp like current code).

---

## F2 — Doppler-compensated extraction + Dopplergram product

**Goal:** sample each spectrum at its *local* line center instead of the
smile average, eliminating velocity→intensity crosstalk; emit the velocity
map as a science product. Requires F1 (μ per spectrum is its byproduct).

**Design:**
- Velocity map `v(x,y) = μ(x,y) − smile(y) − flex(x)` (flex from F3 when
  present, else 0). μ raw is noisy: denoise with **edge-preserving smoothing**
  — robust LOESS in 2-D or simply our robust_trend applied separably at
  window ~7 px, Tukey-clipped (real velocity structure is smooth at 5–10 px
  scale; single-pixel μ outliers are noise).
- Core image = profile model evaluated at μ_local (F1 stage A/B already
  yields `C·(1−d)` which IS the value at the local center — so the intensity
  side of F2 is free with F1; this feature is mostly about *validating* that
  and producing the Dopplergram).
- Output: `recon_velocity.fits` (f32, px units) + `_velocity.png`
  (symmetric scale, annotate scale in the log); CLI `--velocity` flag.

**Harness gates (needs Phase 0 `doppler: true` synth):**
- With doppler enabled and F2 off: record the PSNR penalty (expect ~1–2 dB).
  With F2 on: recover ≥80% of it.
- `eval-velocity` RMS ≤ 0.05 px on default noise, interior.
- Clean+doppler scan: velocity RMS ≤ 0.02 px (bias check).

**Pitfalls:** near the limb the line weakens and μ noise explodes — weight
the velocity smoothing by fitted depth d, and taper the compensation to the
smile default below d < ~0.15 (another gate+taper, per CLAUDE.md §4). Do NOT
smooth the *intensity* with the velocity smoother.

---

## F3 — Per-frame spectral flexure tracking

**Goal:** track slow drift of the line position over the scan (thermal/
flexure), currently assumed static. Real-data photometric waves disappear;
synth (Phase 0 `flexure_px`) makes it measurable.

**Design:**
- Per frame t, estimate global line offset `flex_raw(t)` = depth-weighted
  robust mean over rows of (μ(y,t) − smile(y)) — with F1 this is a cheap
  reduction of already-fitted μ; without F1, fit on ~40 evenly spaced rows.
- `flex(t) = robust LOESS(flex_raw, window ~101 frames)` — flexure is slow;
  everything faster is Doppler/noise and must NOT go into flex (or it will
  eat the Dopplergram's mean). Deadband 0.02 px.
- Apply as additive shift to sampling positions in extraction (both modes).
- Ordering: flex must be estimated from a Doppler-balanced statistic —
  the disk-mean velocity ≈ 0 by symmetry, so a robust mean over the full
  slit is safe mid-disk; near scan edges (partial chords) freeze flex to its
  nearest trusted value (chord gate at 45% like drift correction).

**Harness gates:** flexure synth on, F3 off → record penalty (expect ~0.5–1.5
dB + `flat%` blowup); F3 on → recover ≥80%, `flat%` back under 1.5×
no-flexure level; clean scan → injection ≤0.2 dB. Diag: print
`corr(flex_est, flex_true)` and residual RMS in the bench diagnostics block
(extend `SynthTruth` consumption in `main.rs::diagnostics`).

---

## F4 — Drizzle reconstruction (jitter as dither)

**Goal:** exploit scan-direction oversampling: build the disk on a finer or
same-size grid by footprint-weighted accumulation of every frame's sample,
using measured per-frame offsets. SNR ↑ where coverage overlaps; genuine
super-resolution in X when `scan_step < 1`.

**Design:**
- This REPLACES the "one column per frame" assignment in `extract.rs` for
  the final image, and it must run AFTER jitter/drift/flexure estimation
  (needs their per-frame offsets) — so restructure: estimation stages
  compute offsets from the quick column disk (current code path), then a
  final `drizzle_reconstruct()` re-extracts every frame with F1 and
  deposits into the output grid with:
  - x position: `(t − t_c)·scan_step_est + shear·(y−yc)` … BUT note the
    geometric warp already handles shear/scale. Simplest correct
    factorization: drizzle in the *raw disk* domain onto a grid `pixfrac`
    finer in X only (default 2× when scan_step<0.75), each frame depositing
    a boxcar footprint of width `slit_px/scan_step` output pixels at
    `x = t + jitter-informed offsets`, y shifted by the per-frame vertical
    correction. Then the existing single warp maps drizzle-grid → final
    (fold the 2× into `sx`).
  - Inverse-variance weights from the F1 fit covariance (or 1/signal).
- `scan_step_est` comes from the ellipse `sx` — iterate once: coarse recon →
  geometry → drizzle → final geometry.
- Accumulate `sum(w·v)` and `sum(w)` planes; divide at the end; keep a
  coverage map for diagnostics.

**Tuning params:** `pixfrac` (drop footprint shrink, 0.6..1.0), x-upsample
factor (1 or 2), weight floor.

**Harness gates:**
- `scan_step 0.5` synth: PSNR and `limb_sigma_px` must both beat the
  interpolation path; power-spectrum band 3–4 up.
- `scan_step 1.0` (no oversampling): must NOT regress vs current path
  (drizzle degenerates to shift-and-add; if it regresses, gate on
  measured oversampling).
- Clean scan: identical-to-current within 0.5 dB.

**Pitfalls:** normalization at coverage seams (columns where a frame was
rejected) — check the coverage map has no holes before dividing; uneven
weights create visible column stripes — that's what the transparency
deadband lesson looked like, same medicine (deadband the per-frame weight
variation).

---

## F5 — Multi-scan registration & stacking

**Goal:** √N SNR from K sequential scans, robust to solar evolution between
scans. The biggest real-world lever; INTI has nothing here.

**Design:**
- New subcommand: `ghostsun stack out1.fits out2.fits ... --out stacked`
  operating on our own f32 FITS reconstructions (each already circularized,
  P-rotated, centered — so registration is small residual only).
- Global registration: disk fit (reuse `metrics::fit_disk`) → scale +
  translation; refine translation by the existing NCC coarse-to-fine search.
- **Evolution compensation:** coarse optical flow between each scan and the
  running reference: block matching (32×32 blocks, ±3 px search, NCC, subpixel
  parabola — all machinery exists in `jitter.rs`, generalize to 2-D) →
  smooth the flow field with robust LOESS → warp each scan by its flow
  before averaging. Flow magnitude deadband 0.1 px.
- **Quality weighting:** per-scan per-block weight = local high-frequency
  energy ratio vs the sharpest scan (lucky-imaging style); floor at 0.2 so
  no region is starved. Robust mean (Tukey) across scans per pixel to kill
  leftover transients.
- Reference building: start from the sharpest scan (global HF energy),
  iterate stack→re-register once.

**Harness gates (needs Phase 0 `n_scans`):**
- K=4 synth scans: stacked PSNR ≥ single-scan + 4.5 dB (ideal 6; flow errors
  eat some), SSIM up, `limb_sigma_px` not worse than the sharpest input.
- Evolution on, flow OFF: record the smearing penalty; flow ON must recover
  ≥70% of it.
- K=1: byte-identical passthrough.

**Pitfalls:** flow must be *stiff* (heavy smoothing) or it will "correct"
noise into the reference (hallucinated sharpness — the power-spectrum metric
catches this); prominences evolve fastest — consider flow-freezing off-disk
with a radial gate.

---

## F6 — PSF estimation + regularized deconvolution

**Goal:** measure the effective PSF from the data itself and deconvolve.
Contrast/resolution feature — judged by `limb_sigma_px` + band SNR + PSNR
against the UNBLURRED truth (Phase 0 gives `psf_seeing_px`).

**Design:**
- **PSF estimate:** the limb is a step edge convolved with the PSF. Extend
  `limb.rs` to fit erf width σ_edge at every accepted limb point; robust-fit
  σ_edge vs position angle → anisotropic Gaussian PSF: σ_x (includes slit
  boxcar + seeing-in-X), σ_y (seeing along slit). Subtract the known limb-
  darkening slope bias: fit the erf jointly with a linear ramp (5-param fit;
  do it on the mean of ~30 aligned radial cuts per PA bin to get SNR).
- **Deconvolution:** Richardson–Lucy with Poisson likelihood, anisotropic
  Gaussian kernel (separable), **Total-Variation regularization** (RL-TV,
  Dey et al. 2006) or simply early stopping; iterations default 15, tune
  5..40. Run AFTER the warp on the final image; off-disk emission included
  but background pedestal must be subtracted first and re-added (RL assumes
  zero background).
- CLI: `--deconv [iters]`, off by default. Log the fitted PSF.

**Harness gates:**
- `psf_seeing_px 1.2` synth: PSNR vs unblurred truth improves ≥1.5 dB over
  no-deconv; `limb_sigma_px` down ≥30%; band 3–4 residual improves; **band 4
  must not show excess power above truth** (ringing/hallucination guard).
- PSF estimate diag: fitted σ vs synth truth within 15%.
- Clean, unblurred scan with deconv accidentally on: ≤0.5 dB damage
  (the PSF estimator should report σ ≈ sampling floor and the stage should
  auto-skip below σ_edge < 0.8 px — gate it).

**Pitfalls:** deconvolving before geometry would be more principled (PSF is
defined in raw coords) but the anisotropy after warp is still Gaussian under
affine — transform the kernel by the same affine instead of resampling
twice. Don't RL the 16-bit PNG; operate on the f32 image with the photon
gain estimated from off-disk variance (variance/mean of sky).

---

## F7 — Post-hoc denoising (variance-stabilized classical; optional N2V)

**Goal:** optional cosmetic-but-honest final denoise for low-SNR scans.

**Design (classical first, it's 90% of the value):**
- Estimate gain/read-noise from the data (off-disk regions + photon transfer:
  local variance vs mean over flat disk patches, robust line fit).
- **Generalized Anscombe transform** → noise ≈ unit Gaussian →
  **cycle-spinning undecimated wavelet shrinkage** (B3-spline à trous, 4
  levels, BayesShrink or fixed-k soft threshold, k default 1.0 tune
  0.5..2.0) → exact unbiased inverse Anscombe (Makitalo & Foi closed form).
  All implementable in ~300 lines in `mathutil.rs`/new `denoise.rs`; no new
  dependencies.
- Optional research follow-up: **Noise2Void** blind-spot denoiser — only if
  the wavelet version proves insufficient; it means hand-rolling a tiny conv
  net + SGD in Rust or adding a `candle` dependency. Keep it a separate
  branch until judged.
- CLI: `--denoise [strength]`, off by default; NEVER default-on (science
  data policy).

**Harness gates:** low-SNR synth (add `--exposure 0.25` scale param to synth
signal level): PSNR ≥ +2 dB, SSIM up, band-4 residual must not fall below
0.5× truth power (over-smoothing guard = deleted fibrils), prominences
preserved (off-disk MAE metric from F1). Standard-SNR scan: improvement may
be ~0; must never hurt by >0.3 dB. Clean scan: ≤0.1 dB injection.

---

## F8 — Global MAP joint inversion (research track — do last)

**Goal:** replace the sequential pipeline with one forward model solved
jointly; each estimator then sees the others' residuals. Expected gain:
recovers most of the remaining gap to the clean ceiling; also THE platform
for publication-grade claims.

**Forward model (all parameters below are unknowns):**
```
cube(y,λ,t) ≈ T(t) · g(y) · Profile[ I(warp⁻¹(t,y)),  λ − smile(y) − flex(t) − v(t,y) ; d, σ ]
warp: x = sx·(u + k·v_y) + jitter_x?(t), y = v_y + J(t)
noise: Poisson(gain) + read σ_r
```
Unknowns: image I (the product), per-frame J(t), T(t), flex(t); per-row g(y);
global sx, k, smile coeffs; per-pixel v, d (from F1/F2 machinery).

**Approach:** block-coordinate descent — each block is exactly one of the
existing estimators generalized to "fit residuals given current model", so
the sequential pipeline IS the initializer and the first iteration. 3–5
outer iterations. Gauss-Newton per block; the image block is the drizzle
deposit (F4) with current parameters. Regularizers: smoothness priors on
J/T/flex/g matching the current LOESS windows (they become explicit priors —
document the equivalence).

**Prereqs:** F1–F4 complete (they are the blocks). Budget: this is weeks,
not days; keep it behind `--map-iterations N` (default 0 = current
pipeline).

**Harness gates:** must beat the sequential pipeline on EVERY bench metric
simultaneously on ≥3 seeds + harsh case, and converge (metric monotone over
outer iterations — print per-iteration table). If any block diverges, its
prior is too weak — tighten to match the sequential stage's effective
constraint and re-derive.

---

## F9 — Per-frame 2-D registration + limb-constrained fusion — **IMPLEMENTED 2026-07-07** (anchored-drift + x-registration variant; the fully-fused single LS solve remains open)

**Motivation (2026-07-07 real scans):** residual limb jaggedness and feature
"combing" are per-frame *seeing displacement* — and only the slit-direction
(y) component is currently modeled. Seeing also displaces each frame ALONG
the scan (x): the slit samples a sun strip up to ~1 px away from its nominal
position, which scrambles column ordering and shreds features/limb in a way
no y-correction can fix. The 3 px→6 px drift-clamp saturation on real data
also suggests the y-model needs an absolute reference, not just relative
texture NCC.

**Design:**
1. **Fused y-solve:** one weighted least-squares per-frame y-offset combining
   (a) the existing multi-lag texture-NCC constraints, (b) per-column
   top-limb and bottom-limb residuals vs the fitted ellipse as ABSOLUTE
   anchors (weight by edge strength; prominence-robust via Tukey), and (c) a
   smoothness prior equivalent to the current high-pass. Replaces the
   sequential fast-jitter → midchord-drift stages; removes the drift clamp
   entirely (anchored solutions don't run away).
2. **x-registration:** estimate per-frame x-offset dx(t) from the asymmetry
   of NCC similarity vs frame lag (a frame displaced +dx is more similar to
   its +lag neighbors), solved on the same constraint-graph machinery; then
   deposit columns at their corrected fractional x via the drizzle path
   (F4's filtered warp already samples fractional x — feed it
   per-column x positions instead of uniform spacing).
3. **Split-half Wiener shrinkage** (already in the y fast-jitter pass since
   2026-07-07) must wrap both axes: attenuate by measured split-half
   reliability so estimator noise never becomes injected jitter.

**Harness:** add per-frame x-jitter to synth (mirror of y-jitter, recorded in
`SynthTruth`); gate = x-jitter residual RMS < 0.5·uncorrected, limb jaggedness
(new metric: RMS of polar limb radius residual at high angular frequency)
reduced, no clean-scan injection, and the drift diag no longer saturates any
clamp on real scans.

## F10+ — Novel-concept queue (from the 2026-07-07 real-data session)

Specced in conversation, in recommended order (details in CLAUDE.md session
notes / chat log):
- **F10 (DONE 2026-07-07): telluric-anchored wavelength calibration** — see
  `profile::estimate_flexure_telluric`.
- **F11: spectrograph as 200 fps seeing monitor** — per-frame profile-fit
  residual/contrast as a free seeing-quality metric; use for seeing-adaptive
  drizzle footprints/weights and worst-percentile frame rejection.
- **F12: lucky-column multi-scan mosaicing** — per-column quality selection
  across scans (each scan crossed each longitude under different seeing).
- **F13: slit-width MAP super-resolution in x** — forward model
  (boxcar x seeing x sampling) on a 2x grid, TV-regularized; F9.2 provides
  the measured dither; needs a fine-grid synth truth to score.
- **F14: per-frame 1-D destretch along the slit** — generalize split-half to
  N bands, apply a smooth low-order y-warp per frame (anisoplanatism).
- **F15: cloud-model regularized extraction** — Beckers cloud-model profile
  family instead of PCA; yields optical-depth and absorbing-material
  velocity maps.

## Phase M — Apple Silicon / Metal port (started 2026-07-07)

- **M0 (DONE):** workspace split — `ghostsun-core` (lib), `ghostsun` (CLI),
  `ghostsun-app` (desktop). Core gained a `progress` callback on
  ReconOptions and a split colorize API (`render::prepare` +
  `render::render_with`) for live re-rendering.
- **M1 (DONE):** desktop app (`crates/ghostsun-app`): eframe/egui on the
  wgpu backend — native **Metal** on Apple Silicon. Dark solar theme,
  open/drag-drop .ser/.fits/.png, background reconstruction with live log,
  pan/zoom viewer (scroll = zoom-at-cursor, drag = pan, double-click = fit),
  Grayscale / Hα Color / Doppler views, live prominence-boost and gamma
  sliders (full-res re-render ~100 ms via the prep split), save colorized
  PNG, status readout (zoom, cursor px, intensity, r/R).
- **M2 (DONE 2026-07-07, first increment):** `ghostsun-core::gpu` — wgpu
  (Metal) compute kernels for temporal NLM and the composed Lanczos warp,
  with `ghostsun gpucheck` equivalence gate (NLM 3.9e-7, warp 1.3e-4 max
  relative diff; end-to-end real-scan GPU-vs-CPU 1.6e-5). Pipeline uses GPU
  by default with silent CPU fallback (`--no-gpu`). Timing truth: the warp
  is 8x faster on Metal; NLM is transfer-bound (CPU already ~1 s); the
  biggest M2 win was CPU-side — parallelizing the mean-image pass
  (11.4 s -> 0.6 s). Stage timings now print as `[t]` lines.
- **M2.5 (NEXT):** GPU profile-model extraction — the dominant remaining
  stage (~7 s): one thread per (frame, row): load 120-px spectrum, IIR
  B-spline prefilter in-thread, 11-candidate mu scan + linear (C,D) solve.
  Upload raw 8-bit frames (1 B/px) and scale in-shader; chunk ~256 frames
  per dispatch. PCA stays CPU on a subsample; pass the basis into a second
  dispatch for the projection add-back. Also port robust_loess (transversalium
  + transparency trends, ~3.5 s) if profiling still shows it hot.
- **M3:** app polish — histogram panel, tune-parameter editor, multi-scan
  stacking UI, batch queue, .app bundle + icon (cargo-bundle).

## Process checklist for every feature (copy into each PR)

- [ ] Synth models the degradation; truth recorded; penalty measured with
      feature off (numbers in PR).
- [ ] Feature flag-gated; default per spec above; baseline path untouched.
- [ ] Noisy bench: target met (feature-specific gate above); ablation row
      shows the feature earning its place.
- [ ] Clean-injection audit passed (≤ stated dB).
- [ ] Estimator-vs-truth diagnostic printed in bench (corr/RMS) where a new
      estimator exists.
- [ ] `bench --sweep` run over the feature's tuning params; chosen defaults
      justified by the table (attach it).
- [ ] Seeds {7, 42, 1001} + harsh case (`--jitter 1.2 --tilt 4`) pass.
- [ ] Edge behavior checked explicitly (disk x-limbs, short chords, partial
      disk) — this is where everything has broken so far.
- [ ] CLAUDE.md updated: reference-numbers table, new lessons if any.
- [ ] Zero compiler warnings.
