//! F1/F2: profile-model spectral extraction.
//!
//! Each (row, frame) spectrum is fitted with a constrained absorption model
//! S(x) = C - D * G((x - mu)/sigma_row), with sigma fixed per row (fit on the
//! mean image — per-spectrum sigma is too noisy and WILL blow up on real
//! noise even though it looks fine on clean data). (C, D) are linear given
//! mu; mu is found by scanning +-mu_range around the smile position and
//! refining the SSE minimum with a parabola. Core intensity = C - D — the
//! model value at the exact local line center, which makes the extraction
//! inherently immune to Doppler shifts and spectral flexure (F2/F3 read the
//! fitted mu instead of correcting the intensity).
//!
//! A residual PCA stage (top-K eigenprofiles over mu-centered residuals)
//! captures real profile asymmetries the Gaussian misses while rejecting
//! most noise: only the projection onto K components is added back.
//!
//! Off-disk (weak/absent absorption: prominences are emission) falls back to
//! plain B-spline sampling at the smile position, blended with a taper.

use crate::image2d::Image;
use crate::linefit::LineGeometry;
use crate::mathutil::{
    bspline_eval, bspline_prefilter, fit_inverted_gaussian, gaussian_smooth, pca_topk, polyval,
};
use crate::ser::SerReader;
use rayon::prelude::*;

pub struct ProfileMaps {
    pub core: Image,  // extracted line-core intensity
    pub mu: Image,    // fitted line-center position (absolute spectral px)
    pub depth: Image, // fitted relative depth D/C (0 off-disk)
    /// de-smiled, continuum-weighted mean spectrum per frame (for telluric
    /// anchoring); offsets are relative to the smile position
    pub frame_spec: Vec<Vec<f32>>,
    pub spec_offsets: Vec<f64>,
    /// fraction of fitted slit rows that can reach each offset (the smile
    /// puts the far wings on the detector for some rows and not others)
    pub spec_coverage: Vec<f32>,
    /// illuminated rows that contributed to each frame's spectrum — the
    /// signal gate, which the row-normalised spectrum can no longer supply
    pub frame_spec_rows: Vec<f32>,
}

pub struct ProfileTune {
    pub w_fit: usize,    // half-window in px (default 8)
    pub pca_k: usize,    // residual PCA components (default 3; 0 = parametric only)
    pub mu_range: f64,   // mu search range around smile (default 1.5)
    pub depth_gate: f64, // below this depth fall back to B-spline (default 0.10)
    /// F17 spectral-subspace rank. 0 disables it and leaves the parametric
    /// core + residual PCA in charge; >0 replaces both with a projection of
    /// the whole mu-centred profile onto a rank-(kl_k+1) subspace learned
    /// from this scan.
    pub kl_k: usize,
    /// Half-window for that projection, in dispersion px. 0 = auto, i.e. the
    /// widest window the smile leaves in common to every fitted row.
    pub w_kl: usize,
}

impl Default for ProfileTune {
    fn default() -> Self {
        ProfileTune { w_fit: 8, pca_k: 3, mu_range: 3.0, depth_gate: 0.10, kl_k: 3, w_kl: 0 }
    }
}

/// A learned spectral matched filter: the row of the orthogonal projector
/// that reads the line centre.
///
/// The subspace is spanned by the scan's own mean normalised profile plus the
/// top eigenprofiles of the variation about it. Because only the value AT THE
/// LINE CENTRE is wanted, the whole projection collapses to one weight vector
/// (see `mathutil::projector_row`) and applying it is a single dot product
/// over the window — no per-pixel least squares, and the GPU path needs
/// nothing but the weights.
#[derive(Clone, Debug)]
pub struct KlFilter {
    /// samples kept left and right of the fitted centre; `weights` has
    /// w_lo + w_hi + 1 entries and the centre sits at index w_lo
    pub w_lo: usize,
    pub w_hi: usize,
    pub weights: Vec<f64>,
    /// sum of squared weights: the variance multiplier this estimator applies
    /// to white noise, against 1.0 for reading the centre sample alone
    pub noise_gain: f64,
    /// basis vectors that survived orthonormalisation
    pub rank: usize,
    /// mean normalised profile, kept for diagnostics
    pub mean: Vec<f64>,
}

/// What `fit_frame` should do about the spectral subspace.
pub(crate) enum KlMode<'a> {
    Off,
    /// collect mu-centred, continuum-normalised profiles over (left, right)
    Learn(usize, usize),
    Apply(&'a KlFilter),
}

/// Per-row line width from the mean image, smoothed over rows.
pub(crate) fn fit_sigma_rows(mean_img: &Image, geom: &LineGeometry) -> Vec<f64> {
    let h = mean_img.h;
    let mut sig = vec![f64::NAN; h];
    for y in geom.y1..=geom.y2.min(h - 1) {
        let row: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
        let mu0 = polyval(&geom.coeffs, y as f64);
        let a = (mu0 as isize - 10).max(1) as usize;
        let b = ((mu0 as isize) + 11).min(row.len() as isize - 1) as usize;
        if b <= a + 6 {
            continue;
        }
        let xs: Vec<f64> = (a..b).map(|x| x as f64).collect();
        if let Some((_, s, _, _)) = fit_inverted_gaussian(&xs, &row[a..b], mu0, 2.5) {
            if (0.8..8.0).contains(&s) {
                sig[y] = s;
            }
        }
    }
    // fill + smooth
    let valid: Vec<f64> = sig.iter().cloned().filter(|v| v.is_finite()).collect();
    let med = if valid.is_empty() {
        2.5
    } else {
        let mut v = valid.clone();
        crate::mathutil::median_inplace(&mut v)
    };
    let filled: Vec<f64> = sig.iter().map(|v| if v.is_finite() { *v } else { med }).collect();
    gaussian_smooth(&filled, 15.0)
}

pub(crate) struct ColumnFit {
    pub(crate) core: Vec<f32>,
    pub(crate) mu: Vec<f32>,
    pub(crate) depth: Vec<f32>,
    /// mu-centered residual vectors (2w+1 per row), for PCA
    pub(crate) resid: Vec<f32>,
    /// model scale C per row (to normalize residuals)
    pub(crate) cscale: Vec<f32>,
    /// continuum-weighted de-smiled mean spectrum of this frame
    pub(crate) spec: Vec<f64>,
    /// per-offset accumulated weight: rows differ in which offsets they reach
    pub(crate) spec_w: Vec<f64>,
    pub(crate) spec_rows: f64,
    /// KlMode::Learn only: mu-centred, continuum-normalised profiles
    /// (2*w_kl+1 per row), the training set for the subspace
    pub(crate) prof: Vec<f32>,
}

/// Fit one frame (columns of the output disk). Returns per-row results.
pub(crate) fn fit_frame(
    frame: &Image,
    smile: &[f64],
    sigma_row: &[f64],
    shift: f64,
    tune: &ProfileTune,
    spec_offsets: &[f64],
    kl: &KlMode<'_>,
) -> ColumnFit {
    let h = frame.h;
    let w = frame.w;
    let wf = tune.w_fit as isize;
    let nwin = (2 * wf + 1) as usize;
    // Only one of the two residual buffers is ever populated: the narrow
    // resid feeds the old PCA add-back, the wide prof trains the subspace.
    let (kl_lo, kl_hi) = match kl {
        KlMode::Learn(a, b) => (*a, *b),
        _ => (0, 0),
    };
    let nkl = kl_lo + kl_hi + 1;
    let mut out = ColumnFit {
        core: vec![0.0; h],
        mu: vec![f32::NAN; h],
        depth: vec![0.0; h],
        resid: if matches!(kl, KlMode::Off) { vec![0.0; h * nwin] } else { Vec::new() },
        cscale: vec![0.0; h],
        spec: vec![0.0; spec_offsets.len()],
        spec_w: vec![0.0; spec_offsets.len()],
        spec_rows: 0.0,
        prof: if kl_lo + kl_hi > 0 { vec![0.0; h * nkl] } else { Vec::new() },
    };
    let mut coef = vec![0.0f64; w];
    for y in 0..h {
        let row = frame.row(y);
        for (i, &v) in row.iter().enumerate() {
            coef[i] = v as f64;
        }
        bspline_prefilter(&mut coef);
        let center = smile[y] + shift;
        let sig = sigma_row[y];

        // samples at fixed positions around the smile center
        let xs: Vec<f64> = (-wf..=wf)
            .map(|i| (center + i as f64).clamp(1.0, (w - 2) as f64))
            .collect();
        let ss: Vec<f64> = xs.iter().map(|&x| bspline_eval(&coef, x)).collect();

        // scan mu candidates; (C, D) linear per candidate
        let mut best = (f64::MAX, center, 0.0f64, 0.0f64); // sse, mu, c, d
        let mut sses: Vec<(f64, f64)> = Vec::with_capacity(11);
        let steps = 21;
        for k in 0..steps {
            let mu = center - tune.mu_range + 2.0 * tune.mu_range * k as f64 / (steps - 1) as f64;
            let (mut n, mut sg, mut sgg, mut sv, mut svg) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for (i, &x) in xs.iter().enumerate() {
                let g = (-((x - mu) * (x - mu)) / (2.0 * sig * sig)).exp();
                n += 1.0;
                sg += g;
                sgg += g * g;
                sv += ss[i];
                svg += ss[i] * g;
            }
            let det = n * sgg - sg * sg;
            if det.abs() < 1e-9 {
                continue;
            }
            // S = C - D*G  =>  minimize; normal equations
            let d = (sv * sg - n * svg) / det;
            let c = (sv + d * sg) / n;
            let mut sse = 0.0;
            for (i, &x) in xs.iter().enumerate() {
                let g = (-((x - mu) * (x - mu)) / (2.0 * sig * sig)).exp();
                let r = ss[i] - (c - d * g);
                sse += r * r;
            }
            sses.push((mu, sse));
            if sse < best.0 {
                best = (sse, mu, c, d);
            }
        }
        // parabolic refinement of mu on the SSE samples
        let bi = sses.iter().position(|&(m, _)| m == best.1).unwrap_or(0);
        let mut mu = best.1;
        if bi > 0 && bi + 1 < sses.len() {
            let (vm, v0, vp) = (sses[bi - 1].1, sses[bi].1, sses[bi + 1].1);
            let den = vm - 2.0 * v0 + vp;
            if den > 1e-12 {
                let step = sses[1].0 - sses[0].0;
                mu += step * (0.5 * (vm - vp) / den).clamp(-0.6, 0.6);
            }
        }
        // final (C, D) at refined mu
        let (mut n, mut sg, mut sgg, mut sv, mut svg) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for (i, &x) in xs.iter().enumerate() {
            let g = (-((x - mu) * (x - mu)) / (2.0 * sig * sig)).exp();
            n += 1.0;
            sg += g;
            sgg += g * g;
            sv += ss[i];
            svg += ss[i] * g;
        }
        let det = n * sgg - sg * sg;
        let (c, d) = if det.abs() > 1e-9 {
            let d = (sv * sg - n * svg) / det;
            ((sv + d * sg) / n, d)
        } else {
            (sv / n, 0.0)
        };

        let depth = if c > 1e-6 { (d / c).clamp(-1.0, 1.0) } else { 0.0 };
        let mut core_model = c - d;
        // off-disk fallback: sample at the smile center
        let bspl = bspline_eval(&coef, center.clamp(1.0, (w - 2) as f64));
        let t = ((depth - tune.depth_gate + 0.03) / 0.06).clamp(0.0, 1.0);

        // F17: read the core off the subspace projection of the whole
        // profile instead of off the two-parameter Gaussian. Same quantity,
        // same mu, but estimated from every sample in the window at once.
        if let KlMode::Apply(f) = kl {
            if t > 0.5 {
                let wl = f.w_lo as isize;
                let mut acc = 0.0;
                for (k, &wt) in f.weights.iter().enumerate() {
                    let x = (mu + (k as isize - wl) as f64).clamp(1.0, (w - 2) as f64);
                    acc += wt * bspline_eval(&coef, x);
                }
                core_model = acc;
            }
        }
        out.core[y] = (t * core_model + (1.0 - t) * bspl).max(0.0) as f32;
        out.mu[y] = if t > 0.5 { mu as f32 } else { f32::NAN };
        out.depth[y] = (depth.max(0.0) * t) as f32;
        out.cscale[y] = c.max(1.0) as f32;

        // De-smiled spectrum accumulation. Each row contributes only at the
        // offsets it actually recorded — clamping instead would fold the
        // detector edge pixel into the far wings and invent a feature there.
        //
        // Values stay in ABSOLUTE units. This spectrum has two consumers with
        // opposite needs: the anchor/flexure estimator wants a shape and
        // divides by its own robust continuum anyway, while the transparency
        // stage reads one offset of it as a per-frame FLUX. Normalising each
        // row by its fitted continuum here serves the first and silently
        // destroys the second — it did exactly that, and only on the GPU
        // path, which is the only caller that uses it for transparency.
        if c > 1.0 {
            for (k, &o) in spec_offsets.iter().enumerate() {
                let x = smile[y] + o;
                if x < 4.0 || x > (w - 5) as f64 {
                    continue;
                }
                out.spec[k] += bspline_eval(&coef, x);
                out.spec_w[k] += 1.0;
            }
            out.spec_rows += 1.0;
        }

        // mu-centered residuals for PCA (normalized by C)
        if matches!(kl, KlMode::Off) && t > 0.5 && c > 1.0 {
            for i in -wf..=wf {
                let x = (mu + i as f64).clamp(1.0, (w - 2) as f64);
                let s = bspline_eval(&coef, x);
                let g = (-((i * i) as f64) / (2.0 * sig * sig)).exp();
                let model = c - d * g;
                out.resid[y * nwin + (i + wf) as usize] = ((s - model) / c) as f32;
            }
        }

        // Training sample for the subspace: the mu-centred profile itself,
        // scaled by the fitted continuum so profiles from bright and dim
        // parts of the disc are directly comparable. Rows whose window would
        // run off the detector are left zero and skipped by the caller — a
        // clamped sample repeats an edge pixel and would teach the basis a
        // feature that is not in the spectrum.
        if let KlMode::Learn(a, b) = kl {
            let (wl, wr) = (*a as isize, *b as isize);
            if t > 0.5 && c > 1.0 && mu - wl as f64 >= 1.0 && mu + wr as f64 <= (w - 2) as f64 {
                for i in -wl..=wr {
                    let sv = bspline_eval(&coef, mu + i as f64);
                    out.prof[y * nkl + (i + wl) as usize] = (sv / c) as f32;
                }
            }
        }
    }
    out
}

/// Widest half-window the smile leaves in common to every fitted row.
///
/// The smile moves the line by tens of px across the slit, so a window that
/// fits at the middle row runs off the detector at the ends. Taking the
/// common window keeps one basis valid for the whole slit and avoids masking
/// inside the projection.
pub fn auto_kl_window(
    geom: &LineGeometry,
    mean_img: &Image,
    slit_h: usize,
    tune: &ProfileTune,
    core_sigma: f64,
) -> (usize, usize) {
    let iw = mean_img.w as f64;
    let (mut cmin, mut cmax) = (f64::MAX, f64::MIN);
    for y in geom.y1..=geom.y2.min(slit_h.saturating_sub(1)) {
        let c = polyval(&geom.coeffs, y as f64);
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    if !cmin.is_finite() || !cmax.is_finite() {
        return (tune.w_fit, tune.w_fit);
    }
    // The smile moves the line across the sensor, so the two sides run out of
    // detector at different offsets. Taking the symmetric intersection throws
    // away the wider side for nothing; keep them separate.
    let margin = tune.mu_range + 2.0;
    let mut lo = (cmin - margin - 1.0).floor();
    let mut hi = (iw - 2.0 - cmax - margin).floor();

    // Stop short of any other absorption feature. A telluric sits at a fixed
    // WAVELENGTH while the window is centred on the SOLAR line, so it drifts
    // through the window with Doppler and flexure and contributes variance
    // that has nothing to do with the chromosphere; inside the subspace it
    // would both inflate the rank and leak into the core estimate.
    let (offs, spec) = desmiled_mean_spectrum(mean_img, geom);
    if !spec.is_empty() {
        for a in detect_anchor_offsets(&offs, &spec, core_sigma) {
            // leave a 3 px guard so the line's own flank stays out too
            if a < 0.0 {
                lo = lo.min(-a - 3.0);
            } else {
                hi = hi.min(a - 3.0);
            }
        }
    }
    let wf = tune.w_fit as f64;
    // 64 px is well past the point where a Halpha profile has anything left
    // to say, and it bounds the per-pixel cost.
    let clamp = |v: f64| -> usize {
        if !v.is_finite() || v < wf {
            tune.w_fit
        } else {
            (v as usize).min(64)
        }
    };
    (clamp(lo), clamp(hi))
}

/// The de-smiled offset grid, and how many of the fitted rows can actually
/// reach each offset.
///
/// The obvious grid is the INTERSECTION over rows -- every offset every row can
/// see -- and that is what this pipeline used to build. It is badly wrong when
/// the smile is large: the line centre on a 100 px detector here runs from 40 to
/// 71 px, so intersecting discards 40 of 100 columns, and any spectral feature
/// living in the discarded 40 becomes invisible even though EVERY row records
/// one. Both telluric anchors on the 2026-08-23 optics sit there, which silently
/// disabled telluric flexure anchoring and the dispersion measurement with it.
///
/// So take the UNION instead and carry the coverage. Offsets seen by too few
/// rows are dropped: they are noisy, and worse, they are contributed by one end
/// of the slit only, so they sample a different part of the Sun than the rest of
/// the grid.
/// Minimum row coverage an offset needs to enter the de-smiled grid.
/// `GS_SPEC_COV=1.0` restores the old intersection behaviour for bisecting.
pub fn spec_min_coverage() -> f64 {
    std::env::var("GS_SPEC_COV").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15)
}

pub fn desmiled_offset_grid(
    mean_img: &Image,
    geom: &LineGeometry,
    min_coverage: f64,
) -> (Vec<f64>, Vec<f64>) {
    let w = mean_img.w as f64;
    let y2 = geom.y2.min(mean_img.h.saturating_sub(1));
    if y2 <= geom.y1 {
        return (Vec::new(), Vec::new());
    }
    let centres: Vec<f64> =
        (geom.y1..=y2).map(|y| polyval(&geom.coeffs, y as f64)).collect();
    let cmin = centres.iter().cloned().fold(f64::MAX, f64::min);
    let cmax = centres.iter().cloned().fold(f64::MIN, f64::max);
    if !cmin.is_finite() || !cmax.is_finite() {
        return (Vec::new(), Vec::new());
    }
    let lo = (4.0 - cmax).ceil();
    let hi = (w - 5.0 - cmin).floor();
    let mut offsets = Vec::new();
    let mut coverage = Vec::new();
    let n = centres.len() as f64;
    let mut o = lo;
    while o <= hi {
        let c = centres.iter().filter(|&&cy| cy + o >= 4.0 && cy + o <= w - 5.0).count() as f64
            / n;
        if c >= min_coverage {
            offsets.push(o);
            coverage.push(c);
        }
        o += 1.0;
    }
    (offsets, coverage)
}

/// De-smiled mean spectrum of a scan's mean image: offsets relative to the
/// line centre, and the row-flux-weighted mean intensity at each.
///
/// Shared by `pipeline::mean_spectrum`, the anchor detector and the subspace
/// window chooser, so all three describe the same spectrum.
pub fn desmiled_mean_spectrum(mean_img: &Image, geom: &LineGeometry) -> (Vec<f64>, Vec<f64>) {
    let w = mean_img.w as f64;
    let (offsets, _cov) = desmiled_offset_grid(mean_img, geom, spec_min_coverage());
    if offsets.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut prof = vec![0.0f64; offsets.len()];
    let mut wsum = vec![0.0f64; offsets.len()];
    let margin = ((geom.y2 - geom.y1) / 20).max(10);
    for y in geom.y1 + margin..geom.y2.saturating_sub(margin).min(mean_img.h - 1) {
        let mut coef: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
        bspline_prefilter(&mut coef);
        let c = polyval(&geom.coeffs, y as f64);
        let rw = mean_img.row(y).iter().map(|&v| v as f64).sum::<f64>();
        if rw <= 1e-9 {
            continue;
        }
        for (k, &o) in offsets.iter().enumerate() {
            let x = c + o;
            if x < 4.0 || x > w - 5.0 {
                continue; // this row never recorded that wavelength
            }
            prof[k] += rw * bspline_eval(&coef, x);
            wsum[k] += rw;
        }
    }
    for (v, n) in prof.iter_mut().zip(&wsum) {
        *v /= n.max(1e-9);
    }
    (offsets, prof)
}

/// Offsets, in px from the line core, of absorption features that are NOT the
/// target line — tellurics and blended weak solar lines.
///
/// The user cannot be asked where their H2O line sits: it moves with every
/// change of camera lens, grating angle and order, so a pixel constant here
/// would be the same latent optics-change bug as the old fixed wing offset.
/// The rule is the one the flexure anchoring already uses — a local minimum
/// at least 1.5% below the robust local continuum, far enough from the core
/// not to be part of it — so the two can never disagree about what is a line.
pub fn detect_anchor_offsets(offsets: &[f64], spectrum: &[f64], core_sigma: f64) -> Vec<f64> {
    let m = spectrum.len();
    if m < 30 || offsets.len() != m {
        return Vec::new();
    }
    let cont = crate::mathutil::robust_loess_quadratic(spectrum, 25, 3);
    let ratio: Vec<f64> = spectrum
        .iter()
        .zip(&cont)
        .map(|(v, c)| if *c > 1e-9 { v / c } else { 1.0 })
        .collect();
    let core_excl = (4.0 * core_sigma).max(8.0);
    let mut out = Vec::new();
    for k in 2..m - 2 {
        if offsets[k].abs() < core_excl {
            continue;
        }
        if ratio[k] < ratio[k - 1]
            && ratio[k] < ratio[k + 1]
            && ratio[k] < ratio[k - 2]
            && ratio[k] < ratio[k + 2]
            && ratio[k] < 0.985
        {
            out.push(offsets[k]);
        }
    }
    out
}

/// Mu-centred, continuum-normalised profiles from a subsample of the scan.
///
/// Shared by the subspace filter and the `specrank` diagnostic so the two can
/// never disagree about what they are describing.
pub(crate) fn collect_kl_samples(
    reader: &SerReader,
    geom: &LineGeometry,
    smile: &[f64],
    sigma_row: &[f64],
    slit_h: usize,
    transpose: bool,
    shift: f64,
    tune: &ProfileTune,
    spatial_offsets: Option<&[f64]>,
    w_lo: usize,
    w_hi: usize,
) -> Vec<Vec<f64>> {
    let n = reader.header.frame_count;
    let nkl = w_lo + w_hi + 1;
    let stride = (n / 48).max(1);
    let frames: Vec<usize> = (0..n).step_by(stride).collect();
    let per_frame: Vec<Vec<Vec<f64>>> = frames
        .par_iter()
        .map(|&t| {
            let mut frame = reader.frame(t);
            if transpose {
                frame = frame.transpose();
            }
            let offset = spatial_offsets.and_then(|v| v.get(t)).copied().unwrap_or(0.0);
            let fit = if offset.abs() >= 1e-6 {
                frame = shift_spatial_cubic(&frame, offset);
                let sm = shift_series_linear(smile, offset);
                let sg = shift_series_linear(sigma_row, offset);
                fit_frame(&frame, &sm, &sg, shift, tune, &[], &KlMode::Learn(w_lo, w_hi))
            } else {
                fit_frame(&frame, smile, sigma_row, shift, tune, &[], &KlMode::Learn(w_lo, w_hi))
            };
            let mut out = Vec::new();
            for y in (geom.y1..=geom.y2.min(slit_h - 1)).step_by(4) {
                let v: Vec<f64> = (0..nkl).map(|i| fit.prof[y * nkl + i] as f64).collect();
                if v.iter().any(|x| *x > 1e-9) {
                    out.push(v);
                }
            }
            out
        })
        .collect();
    per_frame.into_iter().flatten().collect()
}

/// Learn the spectral subspace from a subsample of the scan.
///
/// Basis = the mean normalised profile plus the top `kl_k` eigenprofiles of
/// the variation about it. The mean is included as a basis vector rather than
/// subtracted off, so the fit has a free amplitude and a pixel whose
/// continuum differs from the scan average is not forced back toward it.
pub(crate) fn learn_kl_filter(
    reader: &SerReader,
    geom: &LineGeometry,
    smile: &[f64],
    sigma_row: &[f64],
    slit_h: usize,
    transpose: bool,
    shift: f64,
    tune: &ProfileTune,
    spatial_offsets: Option<&[f64]>,
    w_lo: usize,
    w_hi: usize,
) -> Option<KlFilter> {
    let samples = collect_kl_samples(
        reader, geom, smile, sigma_row, slit_h, transpose, shift, tune, spatial_offsets, w_lo,
        w_hi,
    );
    let nkl = w_lo + w_hi + 1;
    if samples.len() < 500 {
        return None;
    }
    let (comps, mean) = pca_topk(&samples, tune.kl_k, 60);
    if comps.is_empty() {
        return None;
    }
    let mut basis = vec![mean.clone()];
    basis.extend(comps);
    let weights = crate::mathutil::projector_row(&basis, w_lo);
    if weights.len() != nkl {
        return None;
    }
    let noise_gain: f64 = weights.iter().map(|x| x * x).sum();
    if std::env::var("GS_DEBUG").is_ok() {
        let mut acc: Vec<String> = Vec::new();
        for r in 1..=basis.len() {
            let w = crate::mathutil::projector_row(&basis[..r], w_lo);
            let g: f64 = w.iter().map(|x| x * x).sum();
            acc.push(format!("{r}:{:.2}x", 1.0 / g.max(1e-12).sqrt()));
        }
        eprintln!(
            "  kl: {} samples, window {} px, per-rank noise reduction {}",
            samples.len(),
            nkl,
            acc.join("  ")
        );
    }
    // A projector row cannot have a noise gain above 1 (that is the identity,
    // i.e. reading the centre sample); anything at or above it means the
    // subspace is as wide as the data and there is nothing to gain.
    if !(noise_gain.is_finite() && noise_gain > 0.0 && noise_gain < 0.95) {
        return None;
    }
    Some(KlFilter { w_lo, w_hi, weights, noise_gain, rank: basis.len(), mean })
}

/// Full profile-model extraction of the disk (plus mu/depth maps).
/// Tries the GPU kernel first when allowed; CPU is fallback + reference.
pub fn extract_profile_auto(
    reader: &SerReader,
    geom: &LineGeometry,
    mean_img: &Image,
    transpose: bool,
    shift: f64,
    tune: &ProfileTune,
    use_gpu: bool,
    spatial_offsets: Option<&[f64]>,
) -> (ProfileMaps, bool) {
    if use_gpu {
        if let Some(maps) =
            crate::gpu_extract::extract_profile_gpu(
                reader,
                geom,
                mean_img,
                transpose,
                shift,
                tune,
                spatial_offsets,
            )
        {
            return (maps, true);
        }
    }
    (
        extract_profile(
            reader,
            geom,
            mean_img,
            transpose,
            shift,
            tune,
            spatial_offsets,
        ),
        false,
    )
}

pub(crate) fn shift_spatial_cubic(frame: &Image, offset: f64) -> Image {
    if offset.abs() < 1e-6 {
        return frame.clone();
    }
    let mut out = Image::new(frame.w, frame.h);
    for y in 0..frame.h {
        let src = (y as f64 + offset).clamp(0.0, (frame.h - 1) as f64);
        let y1 = src.floor() as isize;
        let f = (src - y1 as f64) as f32;
        for x in 0..frame.w {
            let p0 = frame.at_clamped(x as isize, y1 - 1);
            let p1 = frame.at_clamped(x as isize, y1);
            let p2 = frame.at_clamped(x as isize, y1 + 1);
            let p3 = frame.at_clamped(x as isize, y1 + 2);
            let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
            let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
            let c = -0.5 * p0 + 0.5 * p2;
            out.set(x, y, ((a * f + b) * f + c) * f + p1);
        }
    }
    out
}

pub(crate) fn shift_series_linear(values: &[f64], offset: f64) -> Vec<f64> {
    if offset.abs() < 1e-6 {
        return values.to_vec();
    }
    (0..values.len())
        .map(|y| {
            let src = (y as f64 + offset).clamp(0.0, (values.len() - 1) as f64);
            let y0 = src.floor() as usize;
            let y1 = (y0 + 1).min(values.len() - 1);
            let f = src - y0 as f64;
            values[y0] * (1.0 - f) + values[y1] * f
        })
        .collect()
}

/// CPU reference implementation.
pub fn extract_profile(
    reader: &SerReader,
    geom: &LineGeometry,
    mean_img: &Image,
    transpose: bool,
    shift: f64,
    tune: &ProfileTune,
    spatial_offsets: Option<&[f64]>,
) -> ProfileMaps {
    let n = reader.header.frame_count;
    let slit_h = if transpose { reader.header.width } else { reader.header.height };
    let smile: Vec<f64> = (0..slit_h).map(|y| polyval(&geom.coeffs, y as f64)).collect();
    let sigma_row = fit_sigma_rows(mean_img, geom);
    let nwin = 2 * tune.w_fit + 1;

    // Spectral grid: the UNION of what the rows can reach, not the
    // intersection. See desmiled_offset_grid — the intersection hides
    // telluric anchors whenever the smile is large.
    let (spec_offsets, spec_cov) = desmiled_offset_grid(mean_img, geom, spec_min_coverage());

    // F17: learn the spectral subspace before the main pass, so the filter
    // can be applied inline and no per-frame profile buffer is ever kept.
    let kl_filter = if tune.kl_k > 0 {
        let mut med: Vec<f64> = sigma_row
            .iter()
            .skip(geom.y1)
            .take(geom.y2.saturating_sub(geom.y1) + 1)
            .cloned()
            .collect();
        let core_sigma =
            if med.is_empty() { 2.5 } else { crate::mathutil::median_inplace(&mut med) };
        let (wl, wr) = if tune.w_kl > 0 {
            (tune.w_kl, tune.w_kl)
        } else {
            auto_kl_window(geom, mean_img, slit_h, tune, core_sigma)
        };
        if wl + wr > 2 * tune.w_fit {
            learn_kl_filter(
                reader, geom, &smile, &sigma_row, slit_h, transpose, shift, tune,
                spatial_offsets, wl, wr,
            )
        } else {
            None
        }
    } else {
        None
    };
    if let Some(f) = &kl_filter {
        if std::env::var("GS_DEBUG").is_ok() {
            eprintln!(
                "  kl: window -{}..+{} px, rank {}, noise gain {:.4} ({:.2}x less noise than the centre sample)",
                f.w_lo,
                f.w_hi,
                f.rank,
                f.noise_gain,
                1.0 / f.noise_gain.sqrt()
            );
        }
    }
    let kl_mode = match &kl_filter {
        Some(f) => KlMode::Apply(f),
        None => KlMode::Off,
    };

    let fits: Vec<ColumnFit> = (0..n)
        .into_par_iter()
        .map(|t| {
            let mut frame = reader.frame(t);
            if transpose {
                frame = frame.transpose();
            }
            let offset = spatial_offsets
                .and_then(|v| v.get(t))
                .copied()
                .unwrap_or(0.0);
            if offset.abs() >= 1e-6 {
                frame = shift_spatial_cubic(&frame, offset);
                let shifted_smile = shift_series_linear(&smile, offset);
                let shifted_sigma = shift_series_linear(&sigma_row, offset);
                fit_frame(
                    &frame,
                    &shifted_smile,
                    &shifted_sigma,
                    shift,
                    tune,
                    &spec_offsets,
                    &kl_mode,
                )
            } else {
                fit_frame(&frame, &smile, &sigma_row, shift, tune, &spec_offsets, &kl_mode)
            }
        })
        .collect();

    let mut core = Image::new(n, slit_h);
    let mut mu = Image::new(n, slit_h);
    let mut depth = Image::new(n, slit_h);
    for (t, f) in fits.iter().enumerate() {
        for y in 0..slit_h {
            core.set(t, y, f.core[y]);
            mu.set(t, y, f.mu[y]);
            depth.set(t, y, f.depth[y]);
        }
    }

    // ---- residual PCA denoising (stage B) ----
    // Skipped when the subspace filter is in charge: it already reads the
    // core off the whole profile, and adding a residual model on top would
    // double-count the same wing information.
    if tune.pca_k > 0 && kl_filter.is_none() {
        // subsample residual vectors from fitted pixels
        let mut samples: Vec<Vec<f64>> = Vec::new();
        for (t, f) in fits.iter().enumerate() {
            if t % 3 != 0 {
                continue;
            }
            for y in (0..slit_h).step_by(4) {
                if f.depth[y] > 0.05 && f.cscale[y] > 1.0 {
                    let v: Vec<f64> = (0..nwin).map(|i| f.resid[y * nwin + i] as f64).collect();
                    if v.iter().any(|x| x.abs() > 1e-9) {
                        samples.push(v);
                    }
                }
            }
        }
        if samples.len() > 500 {
            let (comps, mean) = pca_topk(&samples, tune.pca_k, 60);
            // project every fitted pixel's residual; add reconstruction at center
            let wc = tune.w_fit; // center index
            for (t, f) in fits.iter().enumerate() {
                for y in 0..slit_h {
                    if f.depth[y] > 0.05 && f.cscale[y] > 1.0 {
                        let mut add = mean[wc];
                        for comp in &comps {
                            let mut a = 0.0;
                            for i in 0..nwin {
                                a += (f.resid[y * nwin + i] as f64 - mean[i]) * comp[i];
                            }
                            add += a * comp[wc];
                        }
                        let v = core.at(t, y) as f64 + add * f.cscale[y] as f64;
                        core.set(t, y, v.max(0.0) as f32);
                    }
                }
            }
        }
    }

    let frame_spec: Vec<Vec<f32>> = fits
        .iter()
        .map(|f| {
            f.spec
                .iter()
                .zip(&f.spec_w)
                .map(|(&v, &n)| if n > 0.0 { (v / n) as f32 } else { 0.0 })
                .collect()
        })
        .collect();
    let frame_spec_rows: Vec<f32> = fits.iter().map(|f| f.spec_rows as f32).collect();

    ProfileMaps {
        core,
        mu,
        depth,
        frame_spec,
        spec_offsets,
        spec_coverage: spec_cov.iter().map(|&v| v as f32).collect(),
        frame_spec_rows,
    }
}

// ---------------------------------------------------------------------------
// Telluric-anchored flexure (concept #1).
//
// Telluric absorption lines are imprinted by Earth's atmosphere: they sit at
// fixed wavelengths in the spectrograph frame, shifting with instrument
// flexure but NOT with solar Doppler. Anchoring the per-frame wavelength
// zero-point on them breaks the flexure/solar-rotation degeneracy that the
// solar-line estimator has to resolve by assumption (linear part -> rotation).
//
// Weak solar photospheric lines can appear among the anchors; they carry a
// rotation ramp (linear in scan position). Lines whose per-frame shift slope
// disagrees with the anchor median are rejected before combining.
// ---------------------------------------------------------------------------

pub struct TelluricFlex {
    pub flex: Vec<f64>,
    pub n_lines: usize,
    pub line_offsets: Vec<f64>,
    /// Spectral dispersion (A/px). `None` when it could not be established:
    /// a SINGLE anchor cannot determine it, because some dispersion always
    /// maps one line onto some catalog entry exactly (a real case: the anchor
    /// at +39 px sits on H2O 6568.81 at 0.154 A/px and on 6564.21 at 0.036,
    /// both perfect fits). Only the SEPARATION between two or more anchors
    /// constrains the scale.
    pub dispersion: Option<f64>,
}

pub fn estimate_flexure_telluric(
    maps: &ProfileMaps,
    _smile: &[f64],
    core_sigma: f64,
    dispersion_hint: Option<f64>,
) -> Option<TelluricFlex> {
    let n = maps.frame_spec.len();
    let m = maps.spec_offsets.len();
    if n < 100 || m < 30 {
        return None;
    }
    // global mean spectrum over frames with signal
    // Gate on ILLUMINATED ROWS, not on the spectrum's sum: the spectrum is
    // now row-normalised, so a frame with two lit rows sums the same as a
    // frame with two thousand and the sum no longer separates them.
    let weights: Vec<f64> = maps.frame_spec_rows.iter().map(|&v| v as f64).collect();
    let wmax = weights.iter().cloned().fold(f64::MIN, f64::max);
    let good: Vec<usize> = (0..n).filter(|&t| weights[t] > 0.3 * wmax).collect();
    if good.len() < 100 {
        return None;
    }
    let mut mean = vec![0.0f64; m];
    let mut mw = vec![0.0f64; m];
    for &t in &good {
        for k in 0..m {
            let v = maps.frame_spec[t][k] as f64;
            if v > 0.0 {
                mean[k] += v;
                mw[k] += 1.0;
            }
        }
    }
    for (v, n) in mean.iter_mut().zip(&mw) {
        *v /= n.max(1e-9);
    }
    // local continuum for depth measurement
    let cont = crate::mathutil::robust_loess_quadratic(&mean, 25, 3);
    let ratio: Vec<f64> = mean
        .iter()
        .zip(&cont)
        .map(|(v, c)| if *c > 1e-9 { v / c } else { 1.0 })
        .collect();
    // detect anchor lines: local minima, >=1.5% deep, away from the core
    let core_excl = (4.0 * core_sigma).max(8.0);
    let mut anchors: Vec<usize> = Vec::new();
    for k in 2..m - 2 {
        let o = maps.spec_offsets[k];
        if o.abs() < core_excl {
            continue;
        }
        if ratio[k] < ratio[k - 1]
            && ratio[k] < ratio[k + 1]
            && ratio[k] < ratio[k - 2]
            && ratio[k] < ratio[k + 2]
            && ratio[k] < 0.985
        {
            anchors.push(k);
        }
    }
    if anchors.is_empty() {
        return None;
    }
    // per frame, per anchor: parabola sub-pixel minimum on the frame spectrum
    let mut shifts: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; anchors.len()];
    for (a, &k0) in anchors.iter().enumerate() {
        for &t in &good {
            let sp = &maps.frame_spec[t];
            let lo = k0.saturating_sub(7).max(1);
            let hi = (k0 + 8).min(m - 1);
            if hi <= lo + 9 {
                continue;
            }
            let wx: Vec<f64> = (lo..hi).map(|k| maps.spec_offsets[k]).collect();
            let wv: Vec<f64> = (lo..hi).map(|k| sp[k] as f64).collect();
            if let Some(pos) = baseline_corrected_dip(&wx, &wv, maps.spec_offsets[k0], 2.5) {
                shifts[a][t] = pos - maps.spec_offsets[k0];
            }
        }
        // center each line's series on its own median
        let mut valid: Vec<f64> = shifts[a].iter().cloned().filter(|v| v.is_finite()).collect();
        if valid.len() < 50 {
            for v in shifts[a].iter_mut() {
                *v = f64::NAN;
            }
            continue;
        }
        let med = crate::mathutil::median_inplace(&mut valid);
        for v in shifts[a].iter_mut() {
            if v.is_finite() {
                *v -= med;
            }
        }
    }
    // Solar-vs-telluric classification by WAVELENGTH MATCHING. Slope-based
    // voting is degenerate whenever the scan runs along the rotation axis
    // (rotation then lives along the slit, not across frames — observed on
    // real N-S scans where every anchor shares the flexure slope). Instead
    // fit a dispersion that matches the anchor offsets against the H2O
    // telluric catalog around Halpha; anchors landing on H2O lines are
    // telluric.
    let h2o: [f64; 11] = [
        6543.91, 6548.62, 6552.63, 6557.17, 6558.15, 6560.50, 6561.10,
        6564.21, 6565.53, 6568.81, 6572.08,
    ];
    let solar_lines: [f64; 4] = [6546.24, 6551.68, 6559.58, 6569.21];
    let halpha = 6562.801;
    let offs: Vec<f64> = anchors.iter().map(|&k| maps.spec_offsets[k]).collect();
    // Establishing the dispersion needs either a caller-supplied scale or at
    // least TWO anchors. With one anchor the catalog match is degenerate --
    // scanning the dispersion always finds a value that lands it exactly on
    // some line -- so a fitted figure there is not a measurement, it is an
    // accident of which catalog entry the scan reached first.
    let disp = match dispersion_hint {
        Some(d) if d.is_finite() && d > 0.0 => Some(d),
        _ if offs.len() >= 2 => {
            // Provisional: only kept if >= 2 anchors then MATCH the catalog.
            let mut best = (f64::MAX, 0.0f64);
            let mut d = 0.03;
            while d <= 0.25 {
                let mut tot = 0.0;
                for &o in &offs {
                    let lam = halpha + o * d;
                    let d1 = h2o.iter().map(|l| (l - lam).abs()).fold(f64::MAX, f64::min);
                    let d2 = solar_lines.iter().map(|l| (l - lam).abs()).fold(f64::MAX, f64::min);
                    tot += d1.min(d2);
                }
                if tot < best.0 {
                    best = (tot, d);
                }
                d += 0.0005;
            }
            Some(best.1)
        }
        _ => None,
    };
    // Without a scale the anchors cannot be classified as telluric rather than
    // solar, and a solar line carries the rotation ramp this estimator exists
    // to avoid. Decline, so the caller falls back to the solar-line path.
    let Some(disp) = disp else {
        if std::env::var("GS_DEBUG").is_ok() {
            eprintln!(
                "[telluric] {} anchor(s) at {:?} px and no dispersion hint -- \
                 cannot establish the scale from one line; declining",
                offs.len(),
                offs.iter().map(|o| *o as i64).collect::<Vec<_>>()
            );
        }
        return None;
    };
    let keep: Vec<usize> = (0..anchors.len())
        .filter(|&a| {
            let lam = halpha + offs[a] * disp;
            let d_h2o = h2o.iter().map(|l| (l - lam).abs()).fold(f64::MAX, f64::min);
            let d_sol = solar_lines.iter().map(|l| (l - lam).abs()).fold(f64::MAX, f64::min);
            d_h2o < 0.15 && d_h2o < d_sol
        })
        .collect();
    if std::env::var("GS_DEBUG").is_ok() {
        eprintln!(
            "[telluric] dispersion {:.4} A/px, anchors {:?} -> telluric {:?}",
            disp,
            offs.iter().map(|o| *o as i64).collect::<Vec<_>>(),
            keep.iter().map(|&a| offs[a] as i64).collect::<Vec<_>>()
        );
    }
    // A dispersion fitted from fewer than two MATCHED anchors is not a
    // measurement. One anchor can be placed exactly on some catalog line at
    // many different scales (the +39 px anchor lands on H2O 6568.81 at
    // 0.154 A/px and on 6564.21 at 0.036, both perfectly), so the classifier
    // cannot tell telluric from solar -- and a solar line carries the very
    // rotation ramp this estimator exists to avoid. Decline unless the caller
    // supplied the scale, in which case one matched anchor is enough.
    if keep.is_empty() || (dispersion_hint.is_none() && keep.len() < 2) {
        if std::env::var("GS_DEBUG").is_ok() {
            eprintln!(
                "[telluric] only {} matched anchor(s) and no --a-per-px scale; \
                 declining rather than guessing the dispersion",
                keep.len()
            );
        }
        return None;
    }
    if std::env::var("GS_DUMP").is_ok() {
        // per-frame series: Halpha median shift + each anchor's shift
        let ha: Vec<f64> = (0..n)
            .map(|t| {
                let mut devs: Vec<f64> = (0..maps.mu.h)
                    .filter(|&y| maps.depth.at(t, y) > 0.15 && maps.mu.at(t, y).is_finite())
                    .map(|y| maps.mu.at(t, y) as f64 - _smile[y])
                    .collect();
                if devs.len() > 50 {
                    crate::mathutil::median_inplace(&mut devs)
                } else {
                    f64::NAN
                }
            })
            .collect();
        let mut out = String::from("t,ha");
        for &k in anchors.iter() {
            out.push_str(&format!(",a{}", maps.spec_offsets[k] as i64));
        }
        out.push('\n');
        for t in 0..n {
            out.push_str(&format!("{},{:.4}", t, ha[t]));
            for a in 0..anchors.len() {
                out.push_str(&format!(",{:.4}", shifts[a][t]));
            }
            out.push('\n');
        }
        let _ = std::fs::write(std::env::temp_dir().join("anchor_series.csv"), out);
    }
    // combine: median over kept lines per frame, fill, light smoothing
    let mut flex = vec![f64::NAN; n];
    for t in 0..n {
        let mut vals: Vec<f64> = keep
            .iter()
            .filter_map(|&a| {
                let v = shifts[a][t];
                if v.is_finite() { Some(v) } else { None }
            })
            .collect();
        if !vals.is_empty() {
            flex[t] = crate::mathutil::median_inplace(&mut vals);
        }
    }
    let valid_idx: Vec<usize> = (0..n).filter(|&t| flex[t].is_finite()).collect();
    if valid_idx.len() < n / 3 {
        return None;
    }
    let mut filled = flex.clone();
    for t in 0..n {
        if !filled[t].is_finite() {
            let nearest = valid_idx.iter().min_by_key(|&&v| v.abs_diff(t)).unwrap();
            filled[t] = flex[*nearest];
        }
    }
    let smooth = crate::mathutil::gaussian_smooth(&filled, 2.0);
    let mean_f = smooth.iter().sum::<f64>() / smooth.len() as f64;
    let out: Vec<f64> = smooth
        .iter()
        .map(|v| {
            let f = v - mean_f;
            if f.abs() < 0.005 { 0.0 } else { f }
        })
        .collect();
    Some(TelluricFlex {
        flex: out,
        n_lines: keep.len(),
        line_offsets: keep.iter().map(|&a| maps.spec_offsets[anchors[a]]).collect(),
        dispersion: Some(disp),
    })
}

/// Where to sample the line wings for a Dopplergram, chosen from the line
/// profile rather than fixed in pixels.
#[derive(Clone, Copy, Debug)]
pub struct WingOffset {
    /// Offset to use, in px from the line core.
    pub px: f64,
    /// Best offset found on each flank (blue negative, red positive).
    pub blue_px: f64,
    pub red_px: f64,
    /// Line half-width at half depth, px — the floor the offsets respect.
    pub hwhm_px: f64,
}

/// Pick the Dopplergram wing offset that maximises velocity sensitivity per
/// unit photon noise.
///
/// A Doppler shift is read off the wing as an intensity change, so the signal
/// scales with the profile slope `|dI/dl|` while the noise scales with
/// `sqrt(I)`. Their ratio is the quantity to maximise, and it has a genuine
/// interior optimum: at the core the slope vanishes (which is why a core image
/// shows almost no Doppler structure at all), and far out in the wings the
/// slope vanishes again while the photon count is highest.
///
/// Both flanks are evaluated separately — Ha is not symmetric, and a blend or
/// telluric on one side must not drag the other. Returns `None` when the line
/// is too shallow for a well-defined flank, leaving the caller on its default.
pub fn optimal_wing_offset(
    mean_img: &Image,
    smile: &[f64],
    y1: usize,
    y2: usize,
) -> Option<WingOffset> {
    let w = mean_img.w;
    if y2 <= y1 || w < 32 {
        return None;
    }
    // How far the smile lets us sample on both sides without leaving the frame.
    let (mut cmin, mut cmax) = (f64::MAX, f64::MIN);
    for y in y1..=y2.min(smile.len().saturating_sub(1)) {
        cmin = cmin.min(smile[y]);
        cmax = cmax.max(smile[y]);
    }
    if !cmin.is_finite() || !cmax.is_finite() {
        return None;
    }
    let reach = ((cmin - 3.0).min(w as f64 - 4.0 - cmax)).floor() as i64;
    if reach < 8 {
        return None;
    }
    let reach = reach.min(60) as usize;

    // Flux-weighted de-smiled mean profile: every row resampled about its own
    // fitted line centre, so the smile does not smear the flanks we measure.
    let margin = ((y2 - y1) / 20).max(10);
    let mut prof = vec![0.0f64; 2 * reach + 1];
    let mut wsum = 0.0;
    for y in (y1 + margin)..y2.saturating_sub(margin) {
        let row = mean_img.row(y);
        let mut coef: Vec<f64> = row.iter().map(|&v| v as f64).collect();
        crate::mathutil::bspline_prefilter(&mut coef);
        let c = smile[y];
        let rw: f64 = row.iter().map(|&v| v as f64).sum();
        if rw <= 0.0 {
            continue;
        }
        for (k, slot) in prof.iter_mut().enumerate() {
            let o = k as f64 - reach as f64;
            *slot += rw * crate::mathutil::bspline_eval(&coef, c + o);
        }
        wsum += rw;
    }
    if wsum <= 0.0 {
        return None;
    }
    for v in prof.iter_mut() {
        *v /= wsum;
    }
    let sm = crate::mathutil::gaussian_smooth(&prof, 1.2);
    let core = sm[reach];
    let cont = sm.iter().cloned().fold(f64::MIN, f64::max);
    if !(cont > 0.0) || (cont - core) / cont < 0.15 {
        return None; // too shallow to have a usable flank
    }
    // Half depth defines the floor: sampling inside it means both wings draw
    // on the same photons and the difference stops being a shift measurement.
    let half = core + 0.5 * (cont - core);
    let mut hwhm = 1.0f64;
    for k in reach..sm.len() {
        if sm[k] >= half {
            hwhm = (k - reach) as f64;
            break;
        }
    }
    // The two wings sit at -o and +o, so they are 2*o apart; requiring one
    // HWHM of separation means each is half a HWHM from the core, NOT a whole
    // one. Getting this wrong made the lower bound equal to hwhm, the search
    // range [hwhm, 2*hwhm], and every "optimum" simply the clamped lower bound.
    let floor = (0.5 * hwhm).max(2.0);

    // Search only where a Doppler wing can physically sit. For a line of this
    // shape |dI/dl| peaks near one HWHM out and dividing by sqrt(I) pulls the
    // optimum slightly inward, so the answer lives inside roughly half to two
    // HWHM. Without that bound the metric happily picks a telluric or a blend
    // far out in the wing, which is exactly what it did on the June scans
    // (red flank chose +28 px, 2.4 A off core).
    let (lo_o, hi_o) = (floor, 2.0 * hwhm);
    let sens = |o: f64| -> Option<f64> {
        let k = (o + reach as f64).round() as i64;
        if k < 2 || k as usize >= sm.len() - 2 {
            return None;
        }
        let k = k as usize;
        let slope = (sm[k + 1] - sm[k - 1]).abs() / 2.0;
        Some(slope / sm[k].max(1e-9).sqrt())
    };
    // Fold the two flanks together before choosing. A Dopplergram samples the
    // same distance either side anyway, and averaging means a contaminated
    // flank is diluted by the clean one instead of setting the answer alone.
    let mut best: Option<(f64, f64)> = None;
    let mut o = lo_o;
    while o <= hi_o {
        if let (Some(sb), Some(sr)) = (sens(-o), sens(o)) {
            let sc = 0.5 * (sb + sr);
            if best.map_or(true, |(_, bs)| sc > bs) {
                best = Some((o, sc));
            }
        }
        o += 0.5;
    }
    let (pick, _) = best?;
    // Per-flank peaks are reported for diagnosis only; a large disagreement
    // between them is the signature of a blend on one side.
    let peak_on = |sign: f64| -> f64 {
        let mut b = (0.0f64, f64::MIN);
        let mut o = lo_o;
        while o <= hi_o {
            if let Some(sv) = sens(sign * o) {
                if sv > b.1 {
                    b = (sign * o, sv);
                }
            }
            o += 0.5;
        }
        b.0
    };
    Some(WingOffset {
        px: pick,
        blue_px: peak_on(-1.0),
        red_px: peak_on(1.0),
        hwhm_px: hwhm,
    })
}

/// F3: global per-frame spectral flexure from the mu map.
/// Returns flex(t) (px), robust, slow, dead-banded.
pub fn estimate_flexure(maps: &ProfileMaps, smile: &[f64]) -> Vec<f64> {
    let w = maps.mu.w;
    let h = maps.mu.h;
    // chord gate like other estimators
    let chord: Vec<usize> = (0..w)
        .map(|t| (0..h).filter(|&y| maps.depth.at(t, y) > 0.15).count())
        .collect();
    let max_chord = *chord.iter().max().unwrap_or(&0);
    let mut raw = vec![f64::NAN; w];
    for t in 0..w {
        if max_chord == 0 || chord[t] < (max_chord as f64 * 0.45) as usize || chord[t] < 60 {
            continue;
        }
        // depth-weighted robust mean of mu - smile (disk-mean Doppler ~ 0)
        let mut devs: Vec<f64> = (0..h)
            .filter(|&y| maps.depth.at(t, y) > 0.15 && maps.mu.at(t, y).is_finite())
            .map(|y| maps.mu.at(t, y) as f64 - smile[y])
            .collect();
        if devs.len() > 50 {
            raw[t] = crate::mathutil::median_inplace(&mut devs);
        }
    }
    let valid: Vec<usize> = (0..w).filter(|&t| raw[t].is_finite()).collect();
    if valid.len() < 30 {
        return vec![0.0; w];
    }
    let mut filled = raw.clone();
    for t in 0..w {
        if !filled[t].is_finite() {
            let nearest = valid.iter().min_by_key(|&&v| v.abs_diff(t)).unwrap();
            filled[t] = raw[*nearest];
        }
    }
    // Slow trend only. The constant belongs to the smile polynomial and the
    // LINEAR-in-t component is degenerate with solar rotation (rotation
    // Doppler is linear in scan position) — both are removed by a robust
    // line fit, so flexure keeps only the nonlinear drift and the velocity
    // map keeps the rotation gradient.
    let trend = crate::mathutil::robust_loess_quadratic(&filled, 101, 3);
    let xs: Vec<f64> = (0..w).map(|i| i as f64).collect();
    let ws2 = vec![1.0; w];
    let line = crate::mathutil::polyfit_robust(&xs, &trend, &ws2, 1, 3).unwrap_or(vec![0.0, 0.0]);
    trend
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let f = v - crate::mathutil::polyval(&line, i as f64);
            if f.abs() < 0.02 {
                0.0
            } else {
                f - 0.02 * f.signum()
            }
        })
        .collect()
}

/// F2: velocity map (px) = mu - smile - flex + v_row, depth-weighted robust
/// smoothing. `v_row` restores slit-direction solar velocity absorbed by
/// the smile fit (see slit_velocity_from_telluric).
pub fn velocity_map(maps: &ProfileMaps, smile: &[f64], flex: &[f64], v_row: Option<&[f64]>) -> Image {
    let w = maps.mu.w;
    let h = maps.mu.h;
    // Velocity is only meaningful where there is real absorption signal.
    // Depth alone is NOT enough: an absorption fit on sky NOISE routinely
    // fakes >15% depth, filling the background with +-3 px garbage that
    // then wrecks display normalization. Gate on continuum intensity too.
    let ithresh = crate::mathutil::percentile_f32(&maps.core.data, 80.0) * 0.25;
    let mut v = Image::new(w, h);
    for t in 0..w {
        for y in 0..h {
            let m = maps.mu.at(t, y);
            if m.is_finite() && maps.depth.at(t, y) > 0.15 && maps.core.at(t, y) > ithresh {
                let add = v_row.map(|r| r[y]).unwrap_or(0.0);
                v.set(t, y, (m as f64 - smile[y] - flex[t] + add) as f32);
            } else {
                v.set(t, y, f32::NAN);
            }
        }
    }
    // edge-preserving-ish smoothing: Tukey-clip vs local 5x5 median, then
    // small Gaussian; NaN-aware
    let mut sm = Image::new(w, h);
    for y in 0..h {
        for t in 0..w {
            let c = v.at(t, y);
            if !c.is_finite() {
                sm.set(t, y, 0.0);
                continue;
            }
            let mut acc = 0.0;
            let mut cnt = 0.0;
            for dy in -2i64..=2 {
                for dt in -2i64..=2 {
                    let tt = t as i64 + dt;
                    let yy = y as i64 + dy;
                    if tt < 0 || yy < 0 || tt >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    let n = v.at(tt as usize, yy as usize);
                    if n.is_finite() && (n - c).abs() < 0.6 {
                        acc += n as f64;
                        cnt += 1.0;
                    }
                }
            }
            sm.set(t, y, if cnt > 0.0 { (acc / cnt) as f32 } else { 0.0 });
        }
    }
    sm
}


/// Sub-pixel minimum of a weak dip on a sloping/curved background: fit a
/// robust quadratic BASELINE to the flank samples (|dx| > core), divide it
/// out, then parabola on the corrected dip. Without this, the background
/// slope (the Halpha wing under every telluric anchor) drags the minimum —
/// the measured anchor then partially TRACKS solar Doppler shifts and the
/// flexure subtraction cancels real rotation.
fn baseline_corrected_dip(xs: &[f64], vs: &[f64], x0: f64, core_hw: f64) -> Option<f64> {
    let flank_x: Vec<f64> = xs
        .iter()
        .zip(vs)
        .filter(|(x, _)| (**x - x0).abs() > core_hw)
        .map(|(x, _)| *x)
        .collect();
    let flank_v: Vec<f64> = xs
        .iter()
        .zip(vs)
        .filter(|(x, _)| (**x - x0).abs() > core_hw)
        .map(|(_, v)| *v)
        .collect();
    if flank_x.len() < 5 {
        return None;
    }
    let ws = vec![1.0; flank_x.len()];
    let base = crate::mathutil::polyfit_robust(&flank_x, &flank_v, &ws, 2, 3)?;
    // corrected ratio over the full window
    let ratio: Vec<f64> = xs
        .iter()
        .zip(vs)
        .map(|(x, v)| {
            let b = crate::mathutil::polyval(&base, *x);
            if b > 1e-9 { v / b } else { 1.0 }
        })
        .collect();
    // discrete min within the core region
    let mut kmin = None;
    let mut vmin = f64::MAX;
    for (k, x) in xs.iter().enumerate() {
        if (x - x0).abs() <= core_hw + 1.0 && ratio[k] < vmin {
            vmin = ratio[k];
            kmin = Some(k);
        }
    }
    let k = kmin?;
    if k == 0 || k + 1 >= ratio.len() {
        return None;
    }
    let (vm, v0, vp) = (ratio[k - 1], ratio[k], ratio[k + 1]);
    let den = vm - 2.0 * v0 + vp;
    if den <= 1e-12 {
        return None;
    }
    let step = xs[1] - xs[0];
    Some(xs[k] + step * (0.5 * (vm - vp) / den).clamp(-0.8, 0.8))
}

/// Solar velocity along the SLIT, recovered via the telluric reference.
///
/// The smile polynomial is fitted to the Halpha trace, so any static solar
/// velocity structure along the slit (e.g. rotation when the scan runs
/// north-south) is absorbed into it and vanishes from the Dopplergram. A
/// telluric line's per-row curve traces pure instrument curvature; the
/// difference smile(y) - telluric_curve(y), mean-removed, is the missing
/// solar term. Measured on the mean image (thousands of frames of SNR).
pub fn slit_velocity_from_telluric(
    mean_img: &Image,
    smile: &[f64],
    y1: usize,
    y2: usize,
    tell_offsets: &[f64],
) -> Option<Vec<f64>> {
    let h = mean_img.h;
    let margin = ((y2 - y1) / 20).clamp(5, 40);
    let (ya, yb) = (y1 + margin, y2.saturating_sub(margin));
    if yb <= ya + 60 || tell_offsets.is_empty() {
        return None;
    }
    let mut vy_acc = vec![0.0f64; h];
    let mut n_used = 0usize;
    for &off in tell_offsets {
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        for y in ya..yb {
            let row = mean_img.row(y);
            let x0 = smile[y] + off;
            let lo = (x0 - 7.0).max(1.0) as usize;
            let hi = ((x0 + 8.0) as usize).min(row.len() - 2);
            if hi <= lo + 9 {
                continue;
            }
            let wx: Vec<f64> = (lo..hi).map(|k| k as f64).collect();
            let wv: Vec<f64> = (lo..hi).map(|k| row[k] as f64).collect();
            if let Some(pos) = baseline_corrected_dip(&wx, &wv, x0, 2.5) {
                xs.push(y as f64);
                ys.push(pos);
            }
        }
        if xs.len() < 60 {
            continue;
        }
        let ws = vec![1.0; xs.len()];
        let Some(curve) = crate::mathutil::polyfit_robust(&xs, &ys, &ws, 2, 4) else {
            continue;
        };
        // Only the LINEAR-in-y component is attributable to solar rotation:
        // the quadratic difference between the Halpha smile and a telluric
        // curve is wavelength-dependent instrument curvature (measured at
        // +-2 px on real data — 4x larger than rotation!), and the constant
        // is the wavelength separation. Averaging anchors that BRACKET
        // Halpha cancels instrumental keystone (linear-in-lambda tilt) to
        // first order while rotation, common to both, survives.
        let diff: Vec<f64> = (ya..yb)
            .map(|y| smile[y] - (crate::mathutil::polyval(&curve, y as f64) - off))
            .collect();
        let dy: Vec<f64> = (ya..yb).map(|y| y as f64).collect();
        let dws = vec![1.0; diff.len()];
        if let Some(lin) = crate::mathutil::polyfit_robust(&dy, &diff, &dws, 1, 3) {
            for y in 0..h {
                vy_acc[y] += lin[1] * (y as f64 - (ya + yb) as f64 / 2.0);
            }
            n_used += 1;
        }
    }
    if n_used == 0 {
        return None;
    }
    for v in vy_acc.iter_mut() {
        *v /= n_used as f64;
    }
    Some(vy_acc)
}

#[cfg(test)]
mod tests {
    use super::{shift_series_linear, shift_spatial_cubic};
    use crate::image2d::Image;

    #[test]
    fn spatial_cubic_shift_is_identity_at_zero() {
        let mut image = Image::new(5, 9);
        for y in 0..image.h {
            for x in 0..image.w {
                image.set(x, y, (100 * y + x) as f32);
            }
        }

        let shifted = shift_spatial_cubic(&image, 0.0);
        assert_eq!(shifted.data, image.data);
    }

    #[test]
    fn spatial_cubic_shift_matches_integer_source_rows() {
        let mut image = Image::new(4, 12);
        for y in 0..image.h {
            for x in 0..image.w {
                image.set(x, y, (10 * y + x) as f32);
            }
        }

        let shifted = shift_spatial_cubic(&image, 2.0);
        for y in 0..image.h - 2 {
            for x in 0..image.w {
                assert_eq!(shifted.at(x, y), image.at(x, y + 2));
            }
        }
    }

    #[test]
    fn spatial_interpolators_preserve_linear_motion() {
        let values: Vec<f64> = (0..16).map(|y| 3.0 * y as f64 + 7.0).collect();
        let shifted_values = shift_series_linear(&values, 0.35);
        for (y, &value) in shifted_values.iter().enumerate().take(14).skip(1) {
            let expected = 3.0 * (y as f64 + 0.35) + 7.0;
            assert!((value - expected).abs() < 1e-12);
        }

        let mut image = Image::new(3, 16);
        for y in 0..image.h {
            for x in 0..image.w {
                image.set(x, y, (3.0 * y as f64 + 7.0 + x as f64) as f32);
            }
        }
        let shifted = shift_spatial_cubic(&image, 0.35);
        for y in 1..14 {
            for x in 0..image.w {
                let expected = (3.0 * (y as f64 + 0.35) + 7.0 + x as f64) as f32;
                assert!((shifted.at(x, y) - expected).abs() < 1e-4);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// specrank: how many numbers does one spectrum actually need?
//
// Three candidate bases are compared on the same mu-centred profiles:
//
//   DFT  — the obvious "just FFT it" answer. A Doppler shift is a phase ramp,
//          which is elegant, but the window's two ends sit at different
//          continuum levels and the transform's implicit periodicity turns
//          that step into leakage across every frequency.
//   DCT  — the standard fix for exactly that: an even extension has no step,
//          so the same low-pass keeps more of the profile per coefficient.
//   KL   — the scan's own eigenprofiles. Optimal by construction for a given
//          coefficient count, at the cost of being learned rather than fixed.
//
// The figure of merit is not raw reconstruction error, because part of what a
// truncation discards is NOISE, and discarding that is the point. Each
// orthonormal coefficient carries sigma^2 of noise, so keeping m of n leaves
// (n - m) * sigma^2 of noise behind; subtracting it isolates the SIGNAL a
// truncation actually destroys. Reported in units of the per-sample noise:
// below 1.0 the reduction costs less than the noise already present.
// ---------------------------------------------------------------------------

/// One basis's truncation curve.
pub struct BasisCurve {
    pub name: &'static str,
    /// (coefficients kept, signal RMS lost in units of the per-sample noise)
    pub loss: Vec<(usize, f64)>,
    /// smallest coefficient count whose signal loss stays under the noise
    pub free_at: Option<usize>,
    /// (rank, noise variance multiplier when the CENTRE value is read through
    /// this truncated basis) — the number that decides whether a wider window
    /// is worth having, since a basis concentrated on the core gains nothing
    /// from extra continuum samples
    pub centre_gain: Vec<(usize, f64)>,
}

pub struct SpectralRankReport {
    pub w_lo: usize,
    pub w_hi: usize,
    /// fitted line half-width in px, the scale the core exclusion uses
    pub core_sigma: f64,
    /// offsets of detected non-target absorption features (tellurics, blends)
    pub anchors: Vec<f64>,
    pub n_window: usize,
    pub n_samples: usize,
    /// per-sample noise sigma in continuum-normalised units
    pub sigma: f64,
    /// mean DFT power per frequency, normalised to the noise plateau
    pub power_over_noise: Vec<f64>,
    /// highest frequency index carrying more than 3x the noise plateau
    pub k_cut: usize,
    /// KL variance fractions, largest first
    pub eigen: Vec<f64>,
    pub curves: Vec<BasisCurve>,
}

fn orthonormalize(mut basis: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = basis.first().map(|b| b.len()).unwrap_or(0);
    let mut q: Vec<Vec<f64>> = Vec::new();
    for v in basis.drain(..) {
        let mut v = v;
        if v.len() != n {
            continue;
        }
        for _ in 0..2 {
            for u in q.iter() {
                let d: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for (vi, ui) in v.iter_mut().zip(u.iter()) {
                    *vi -= d * ui;
                }
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm < 1e-9 {
            continue;
        }
        for vi in v.iter_mut() {
            *vi /= nrm;
        }
        q.push(v);
    }
    q
}

/// Mean captured energy per orthonormal basis vector, in order.
fn captured(samples: &[Vec<f64>], basis: &[Vec<f64>]) -> Vec<f64> {
    let inv = 1.0 / samples.len() as f64;
    basis
        .par_iter()
        .map(|q| {
            samples
                .iter()
                .map(|s| {
                    let d: f64 = q.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
                    d * d
                })
                .sum::<f64>()
                * inv
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn curve(
    name: &'static str,
    samples: &[Vec<f64>],
    basis: Vec<Vec<f64>>,
    total: f64,
    sigma: f64,
    n: usize,
    centre: usize,
    step: usize,
) -> BasisCurve {
    let q = orthonormalize(basis);
    let cap = captured(samples, &q);
    let mut centre_gain = Vec::new();
    {
        let mut acc = 0.0;
        for (m, u) in q.iter().enumerate() {
            acc += u[centre] * u[centre];
            if m + 1 <= 10 {
                centre_gain.push((m + 1, acc));
            }
        }
    }
    let mut loss = Vec::new();
    let mut free_at = None;
    let mut acc = 0.0;
    for (m, c) in cap.iter().enumerate() {
        acc += c;
        let kept = m + 1;
        // residual energy, minus the noise that residual is entitled to
        let resid = (total - acc).max(0.0);
        let noise_left = (n - kept.min(n)) as f64 * sigma * sigma;
        let sig = ((resid - noise_left).max(0.0) / n as f64).sqrt();
        let rel = sig / sigma;
        if free_at.is_none() && rel < 1.0 {
            free_at = Some(kept);
        }
        if kept % step == 0 || kept <= 8 || free_at == Some(kept) {
            loss.push((kept, rel));
        }
    }
    BasisCurve { name, loss, free_at, centre_gain }
}

/// Compare DFT, DCT and KL truncation on one scan's spectra.
pub fn spectral_rank_report(
    reader: &SerReader,
    geom: &LineGeometry,
    mean_img: &Image,
    transpose: bool,
    w_half_req: usize,
) -> Option<SpectralRankReport> {
    let tune = ProfileTune::default();
    let slit_h = if transpose { reader.header.width } else { reader.header.height };
    let smile: Vec<f64> = (0..slit_h).map(|y| polyval(&geom.coeffs, y as f64)).collect();
    let sigma_row = fit_sigma_rows(mean_img, geom);
    let mut med: Vec<f64> = sigma_row
        .iter()
        .skip(geom.y1)
        .take(geom.y2.saturating_sub(geom.y1) + 1)
        .cloned()
        .collect();
    let core_sigma = if med.is_empty() { 2.5 } else { crate::mathutil::median_inplace(&mut med) };
    let (offs, spec) = desmiled_mean_spectrum(mean_img, geom);
    let anchors = detect_anchor_offsets(&offs, &spec, core_sigma);
    let (w_lo, w_hi) = if w_half_req > 0 {
        (w_half_req, w_half_req)
    } else {
        auto_kl_window(geom, mean_img, slit_h, &tune, core_sigma)
    };
    let n = w_lo + w_hi + 1;
    let samples = collect_kl_samples(
        reader, geom, &smile, &sigma_row, slit_h, transpose, 0.0, &tune, None, w_lo, w_hi,
    );
    if samples.len() < 200 {
        return None;
    }

    // --- noise floor and optical cutoff, from the mean DFT power ---
    let mut power = vec![0.0f64; n / 2 + 1];
    for s in &samples {
        let m: f64 = s.iter().sum::<f64>() / n as f64;
        for (k, p) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, &v) in s.iter().enumerate() {
                let th = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                re += (v - m) * th.cos();
                im += (v - m) * th.sin();
            }
            *p += re * re + im * im;
        }
    }
    for p in power.iter_mut() {
        *p /= samples.len() as f64;
    }
    let tail: Vec<f64> = power[power.len() * 3 / 4..].to_vec();
    let mut t2 = tail.clone();
    let plateau = crate::mathutil::median_inplace(&mut t2).max(1e-30);
    let sigma = (plateau / n as f64).sqrt();
    let k_cut = power.iter().rposition(|&p| p > 3.0 * plateau).unwrap_or(0);
    let power_over_noise: Vec<f64> = power.iter().map(|&p| p / plateau).collect();

    // --- total energy per sample ---
    let total: f64 =
        samples.iter().map(|s| s.iter().map(|x| x * x).sum::<f64>()).sum::<f64>()
            / samples.len() as f64;

    // --- the three bases, each in its natural "keep the first m" order ---
    let pi = std::f64::consts::PI;
    let mut dft: Vec<Vec<f64>> = vec![vec![1.0; n]];
    for k in 1..=n / 2 {
        dft.push((0..n).map(|i| (2.0 * pi * (k * i) as f64 / n as f64).cos()).collect());
        if k * 2 != n {
            dft.push((0..n).map(|i| (2.0 * pi * (k * i) as f64 / n as f64).sin()).collect());
        }
    }
    let dct: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            (0..n).map(|i| (pi * (i as f64 + 0.5) * k as f64 / n as f64).cos()).collect()
        })
        .collect();
    let kmax = 16.min(n - 1);
    let (comps, mean) = pca_topk(&samples, kmax, 80);
    let eigen_raw: Vec<f64> = {
        let mut acc = vec![0.0; comps.len()];
        for s in &samples {
            for (j, c) in comps.iter().enumerate() {
                let d: f64 =
                    c.iter().zip(s.iter()).zip(mean.iter()).map(|((a, b), m)| a * (b - m)).sum();
                acc[j] += d * d;
            }
        }
        let var: f64 = samples
            .iter()
            .map(|s| s.iter().zip(mean.iter()).map(|(a, m)| (a - m) * (a - m)).sum::<f64>())
            .sum();
        acc.iter().map(|v| v / var.max(1e-30)).collect()
    };
    let mut kl: Vec<Vec<f64>> = vec![mean.clone()];
    kl.extend(comps);

    let step = (n / 12).max(1);
    let curves = vec![
        curve("DFT", &samples, dft, total, sigma, n, w_lo, step),
        curve("DCT", &samples, dct, total, sigma, n, w_lo, step),
        curve("KL ", &samples, kl, total, sigma, n, w_lo, 1),
    ];

    Some(SpectralRankReport {
        w_lo,
        w_hi,
        core_sigma,
        anchors,
        n_window: n,
        n_samples: samples.len(),
        sigma,
        power_over_noise,
        k_cut,
        eigen: eigen_raw,
        curves,
    })
}
