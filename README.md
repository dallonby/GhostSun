# GhostSun

High-fidelity solar disk reconstruction from spectroheliograph (Sol'Ex / SHG) SER
scans, in native Rust. A ground-up redesign of the algorithms in
[INTI](https://github.com/Vdesnoux/inti), validated against a physically-modeled
synthetic scan harness with known ground truth.

GhostSun includes a native desktop app for macOS and Windows 11. It uses Metal
on macOS and Direct3D 12/Vulkan on Windows through the same `wgpu` rendering and
compute layer.

## Results

Synthetic benchmark (identical input SER, registered PSNR/SSIM vs ground truth):

| variant | PSNR disk | SSIM | PSNR limb | limb width |
|---|---|---|---|---|
| INTI-faithful baseline | 30.5 dB | 0.878 | 18.1 dB | 2.67 px |
| **GhostSun** | **35.6 dB** | **0.979** | **28.8 dB** | **1.21 px** |

Consistent across seeds and observing conditions; the gap widens in bad
seeing. On a degradation-free scan the GhostSun core pipeline reaches 56 dB —
the INTI processing chain itself caps out at ~37 dB from quantization and
repeated bilinear resampling.

Real data (9,100-frame 3840x120 Sol'Ex-class Hα scans): reconstructed disk
circular to **0.34%** (INTI baseline: 7.2% elliptical), sub-pixel line fit at
0.024 px RMS, ~28 s per scan, plus a full-disk Dopplergram showing solar
rotation — a product INTI does not have.

Reproduce with:

```
cargo run --release -- bench --dir testdata --ablations
```

## What INTI gets wrong, and what GhostSun does instead

1. **uint16 quantization at every stage.** INTI rounds the image to 16-bit
   integers after extraction, flat-fielding, shear, scale, and rotation.
   GhostSun is f32 end-to-end; the only quantization is in optional PNG export
   (FITS output is float).

2. **Three sequential bilinear resamplings.** INTI corrects tilt, X-scale and
   P-angle as three separate `order=1` warps, each low-pass filtering the
   image. GhostSun composes shear + anisotropic scale + rotation + flips into
   **one** affine transform applied once with Lanczos-3.

3. **Whole-pixel spectral line fit.** INTI fits the smile polynomial to
   per-row `argmin` (integer!) positions; its sub-pixel Gaussian code exists
   but is disabled. GhostSun fits a Gauss-Newton inverted Gaussian per row
   (sub-pixel, depth-weighted) and a Tukey-IRLS robust polynomial.

4. **Catmull-Rom + fixed 7-tap smoothing kernel for extraction.** The kernel
   `[-2,3,6,7,6,3,-2]/21` irreversibly mixes off-core wavelengths into the
   image (~4 dB loss on the benchmark). GhostSun samples with exact
   prefiltered cubic B-splines; a Gaussian spectral window is available but
   optional (`--window-sigma`).

5. **No scan-jitter correction.** Seeing shifts every frame along the slit;
   INTI stacks raw columns (ragged limb, shredded fine structure). GhostSun
   estimates per-frame shifts by NCC of *derivative* column profiles over the
   disk interior (limb rows excluded — their edges move opposite ways as the
   chord changes), builds a multi-baseline shift graph (lags 1–16) solved by
   weighted least squares (no random-walk accumulation), high-passes it so
   real geometry survives, and resamples with B-splines. A second estimator
   removes slow drift using the fact that vertical chord midpoints of a
   sheared circle are exactly collinear.

6. **No transparency correction.** Passing haze/clouds become vertical bands
   (INTI's "bad lines" corrector is disabled in the source). GhostSun
   measures per-frame flux on a **continuum** extraction (transparency is
   common-mode across wavelength; the photosphere is smooth so real
   chromospheric structure can't masquerade as transparency), detrends with
   robust local quadratic regression, and applies tapered, dead-banded gains.

7. **Non-robust transversalium flat.** INTI divides by a Savitzky-Golay
   smoothed profile — filaments and plage bias it, and its median/quantile
   trends are biased on curved profiles. GhostSun uses robust LOESS
   (quadratic, Tukey-weighted) row-gain estimation at two scales.

8. **Edge points replaced by a 6th-degree polynomial before ellipse fitting.**
   GhostSun keeps the measured sub-pixel edges (gradient-centroid refinement,
   erf-limb equivalent) and fits the ellipse with Halir-Flusser direct least
   squares inside RANSAC, refined by Tukey IRLS on Sampson distances —
   prominences and active regions are rejected as outliers, not smoothed into
   the geometry.

9. **Geometry model.** The fitted ellipse is converted in closed form to the
   physical scan model (X-scale + slit shear of a true circle), which is what
   the single output warp inverts.

## Advanced features (beyond INTI)

- **Profile-model extraction** (default): every spectrum is fitted with a
  constrained absorption model + residual-PCA denoising; the core is read at
  the *local* line center, making extraction inherently immune to Doppler
  shifts and spectrograph flexure.
- **Dopplergrams** (`--velocity`): full-disk line-of-sight velocity maps fall
  out of the profile fit (FITS + PNG).
- **Flexure tracking**: slow spectral drift over the scan is measured and fed
  to the continuum extraction (nonlinear part only — the linear part is
  degenerate with solar rotation and stays in the velocity map).
- **Footprint-filtered warp** (default): oversampled scans are downsampled
  with a properly scaled kernel — anti-aliased and noise-averaging
  (drizzle-equivalent).
- **Multi-scan stacking** (`stack a.fits b.fits ...`): global registration +
  stiff optical-flow evolution compensation + per-scan quadratic gain
  surfaces + sharpness-weighted robust mean.
- **PSF estimation + deconvolution** (`--deconv`): anisotropic PSF measured
  from the limb transition itself; Richardson-Lucy with total-variation
  regularization; auto-skips when seeing is good.
- **Variance-stabilized denoising** (`--denoise`): generalized Anscombe +
  undecimated B3 wavelet shrinkage with noise parameters estimated from the
  image (photon-transfer regression).
- **Block-coordinate refinement** (`--map-iterations N`): re-estimates
  registration/gain corrections against the corrected disk.
- **Parameter sweeps** (`bench --sweep name=v1,v2,...`) and a JSON results
  ledger (`bench --json ledger.jsonl`) for tuning everything measurably.

## Usage

### Desktop app

Open or drag a `.ser`, `.fits`, `.fit`, or `.png` file into GhostSun. A `.ser`
scan is reconstructed in the background; FITS and PNG files open in view-only
mode. The viewer provides grayscale, Hα color, and (when reconstructed)
Doppler views.

On macOS, run `cargo run --release --package ghostsun-app`. For Windows 11,
download the `GhostSun-Windows-x64` build artifact or follow the
[Windows build and usage guide](docs/windows.md).

### Command line

```
# reconstruct a real scan
ghostsun recon scan.ser --out-dir out            # writes FITS (f32) + PNG16

# INTI-faithful mode for comparison
ghostsun recon scan.ser --out-dir out --baseline --name inti

# wavelength shift (px from line core), e.g. continuum or Doppler wing
ghostsun recon scan.ser --shift 15

# colorized presentation (black background, prominences preserved)
ghostsun colorize out/recon.fits                  # writes recon_color.png
ghostsun recon scan.ser --colorize                # or directly from recon

# synthetic data with ground truth
ghostsun synth --out-dir testdata [--clean] [--jitter 1.0] [--tilt 4]
ghostsun eval out/recon_linear.png testdata/ground_truth.png
ghostsun bench --dir testdata --ablations
```

Stage toggles: `--no-jitter`, `--no-transparency`, `--no-transversalium`,
`--window-sigma S`, `--rotation P`, `--flip-x`, `--flip-y`.

## Layout

- `src/ser.rs` — memory-mapped SER v3 reader/writer
- `src/linefit.rs` — sub-pixel line geometry (Gauss-Newton + robust poly)
- `src/extract.rs` — B-spline disk extraction (+ INTI baseline mode)
- `src/jitter.rs` — fast jitter (multi-baseline LS) + slow drift (midchord)
- `src/flatfield.rs` — continuum-referenced transparency + LOESS transversalium
- `src/limb.rs`, `src/ellipse.rs` — sub-pixel limb + RANSAC/IRLS ellipse
- `src/warp.rs` — single composed Lanczos-3 warp (+ INTI 3-pass baseline)
- `src/profile.rs` — profile-model extraction, velocity + flexure (F1-F3)
- `src/stack.rs` — multi-scan registration and stacking (F5)
- `src/deconv.rs` — limb-measured PSF + RL-TV deconvolution (F6)
- `src/denoise.rs` — Anscombe + wavelet denoising (F7)
- `src/render.rs` — Hα false-color rendering with sky extraction
- `src/synth.rs` — physically-modeled synthetic scan generator
- `src/metrics.rs` — registered PSNR/SSIM/limb-width/band-SNR evaluation
- `inti-reference/` — upstream INTI clone (analysis reference only)
