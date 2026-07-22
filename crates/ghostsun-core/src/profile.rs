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
}

pub struct ProfileTune {
    pub w_fit: usize,    // half-window in px (default 8)
    pub pca_k: usize,    // residual PCA components (default 3; 0 = parametric only)
    pub mu_range: f64,   // mu search range around smile (default 1.5)
    pub depth_gate: f64, // below this depth fall back to B-spline (default 0.10)
}

impl Default for ProfileTune {
    fn default() -> Self {
        ProfileTune { w_fit: 8, pca_k: 3, mu_range: 3.0, depth_gate: 0.10 }
    }
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
    pub(crate) spec_w: f64,
}

/// Fit one frame (columns of the output disk). Returns per-row results.
pub(crate) fn fit_frame(
    frame: &Image,
    smile: &[f64],
    sigma_row: &[f64],
    shift: f64,
    tune: &ProfileTune,
    spec_offsets: &[f64],
) -> ColumnFit {
    let h = frame.h;
    let w = frame.w;
    let wf = tune.w_fit as isize;
    let nwin = (2 * wf + 1) as usize;
    let mut out = ColumnFit {
        core: vec![0.0; h],
        mu: vec![f32::NAN; h],
        depth: vec![0.0; h],
        resid: vec![0.0; h * nwin],
        cscale: vec![0.0; h],
        spec: vec![0.0; spec_offsets.len()],
        spec_w: 0.0,
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
        let core_model = c - d;
        // off-disk fallback: sample at the smile center
        let bspl = bspline_eval(&coef, center.clamp(1.0, (w - 2) as f64));
        let t = ((depth - tune.depth_gate + 0.03) / 0.06).clamp(0.0, 1.0);
        out.core[y] = (t * core_model + (1.0 - t) * bspl).max(0.0) as f32;
        out.mu[y] = if t > 0.5 { mu as f32 } else { f32::NAN };
        out.depth[y] = (depth.max(0.0) * t) as f32;
        out.cscale[y] = c.max(1.0) as f32;

        // de-smiled spectrum accumulation, weighted by continuum level
        if c > 1.0 {
            for (k, &o) in spec_offsets.iter().enumerate() {
                let x = (smile[y] + o).clamp(1.0, (w - 2) as f64);
                out.spec[k] += bspline_eval(&coef, x);
            }
            out.spec_w += 1.0;
        }

        // mu-centered residuals for PCA (normalized by C)
        if t > 0.5 && c > 1.0 {
            for i in -wf..=wf {
                let x = (mu + i as f64).clamp(1.0, (w - 2) as f64);
                let s = bspline_eval(&coef, x);
                let g = (-((i * i) as f64) / (2.0 * sig * sig)).exp();
                let model = c - d * g;
                out.resid[y * nwin + (i + wf) as usize] = ((s - model) / c) as f32;
            }
        }
    }
    out
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
) -> (ProfileMaps, bool) {
    if use_gpu {
        if let Some(maps) =
            crate::gpu_extract::extract_profile_gpu(reader, geom, mean_img, transpose, shift, tune)
        {
            return (maps, true);
        }
    }
    (extract_profile(reader, geom, mean_img, transpose, shift, tune), false)
}

/// CPU reference implementation.
pub fn extract_profile(
    reader: &SerReader,
    geom: &LineGeometry,
    mean_img: &Image,
    transpose: bool,
    shift: f64,
    tune: &ProfileTune,
) -> ProfileMaps {
    let n = reader.header.frame_count;
    let slit_h = if transpose { reader.header.width } else { reader.header.height };
    let smile: Vec<f64> = (0..slit_h).map(|y| polyval(&geom.coeffs, y as f64)).collect();
    let sigma_row = fit_sigma_rows(mean_img, geom);
    let nwin = 2 * tune.w_fit + 1;

    // spectral grid (offsets rel. smile) covering the window for all rows
    let iw = mean_img.w as f64;
    let (mut cmin, mut cmax) = (f64::MAX, f64::MIN);
    for y in geom.y1..=geom.y2.min(slit_h - 1) {
        let c = polyval(&geom.coeffs, y as f64);
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    let off_lo = (4.0 - cmin).ceil();
    let off_hi = (iw - 5.0 - cmax).floor();
    let spec_offsets: Vec<f64> = {
        let mut v = Vec::new();
        let mut o = off_lo;
        while o <= off_hi {
            v.push(o);
            o += 1.0;
        }
        v
    };

    let fits: Vec<ColumnFit> = (0..n)
        .into_par_iter()
        .map(|t| {
            let mut frame = reader.frame(t);
            if transpose {
                frame = frame.transpose();
            }
            fit_frame(&frame, &smile, &sigma_row, shift, tune, &spec_offsets)
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
    if tune.pca_k > 0 {
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
            let w = f.spec_w.max(1e-9);
            f.spec.iter().map(|&v| (v / w) as f32).collect()
        })
        .collect();

    ProfileMaps { core, mu, depth, frame_spec, spec_offsets }
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
    /// fitted spectral dispersion (A/px) from the anchor pattern
    pub dispersion: f64,
}

pub fn estimate_flexure_telluric(
    maps: &ProfileMaps,
    _smile: &[f64],
    core_sigma: f64,
) -> Option<TelluricFlex> {
    let n = maps.frame_spec.len();
    let m = maps.spec_offsets.len();
    if n < 100 || m < 30 {
        return None;
    }
    // global mean spectrum over frames with signal
    let weights: Vec<f64> = maps
        .frame_spec
        .iter()
        .map(|sp| sp.iter().map(|&v| v as f64).sum::<f64>())
        .collect();
    let wmax = weights.iter().cloned().fold(f64::MIN, f64::max);
    let good: Vec<usize> = (0..n).filter(|&t| weights[t] > 0.3 * wmax).collect();
    if good.len() < 100 {
        return None;
    }
    let mut mean = vec![0.0f64; m];
    for &t in &good {
        for k in 0..m {
            mean[k] += maps.frame_spec[t][k] as f64;
        }
    }
    for v in mean.iter_mut() {
        *v /= good.len() as f64;
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
    let mut best = (f64::MAX, 0.0f64);
    let mut disp = 0.03;
    while disp <= 0.25 {
        let mut tot = 0.0;
        for &o in &offs {
            let lam = halpha + o * disp;
            let d1 = h2o.iter().map(|l| (l - lam).abs()).fold(f64::MAX, f64::min);
            let d2 = solar_lines.iter().map(|l| (l - lam).abs()).fold(f64::MAX, f64::min);
            tot += d1.min(d2);
        }
        if tot < best.0 {
            best = (tot, disp);
        }
        disp += 0.0005;
    }
    let disp = best.1;
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
    if keep.is_empty() {
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
        dispersion: disp,
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
