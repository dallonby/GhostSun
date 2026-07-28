//! V-curve capture and parabola solving for the three-stage focus procedure.
//!
//! Every stage of the procedure reduces to the same operation: step a
//! micrometer, record a sharpness metric, find the extremum of the resulting
//! curve. Reading that extremum by eye is not good enough. The beam through a
//! spectroheliograph is slow — Rayleigh depth of focus δ ≈ ±2λN², so ±0.3 mm on
//! the camera's slit-length axis at f/15 — which makes the curves broad and
//! flat. Worse, the quantity Stage A actually needs is the *difference* between
//! two such extrema, so eyeballing each one independently compounds the error.
//!
//! Hence: weighted parabola fit, with a leave-one-out jackknife for the
//! uncertainty. Jackknife rather than the analytic covariance because n is
//! small (5–15 samples), it costs nothing at this size, and it stays honest if
//! the user's sampling is lopsided around the vertex.
//!
//! Positions are in whatever units the user reads off their micrometer; nothing
//! here assumes millimetres.

use ghostsun_core::mathutil::{polyfit_weighted, polyval};

/// One captured point on a V-curve.
#[derive(Clone, Copy, Debug)]
pub struct VSample {
    /// Micrometer reading this sample was taken at.
    pub pos: f64,
    /// The metric value (FWHM in px, contrast, …).
    pub value: f64,
    /// Frames averaged into `value`; used as the least-squares weight.
    pub weight: f64,
}

/// Result of solving a V-curve for its extremum.
#[derive(Clone, Copy, Debug)]
pub struct ParabolaFit {
    /// Micrometer position of the extremum — the focus answer.
    pub vertex: f64,
    /// Jackknife 1σ on `vertex`; NaN when n < 4 (too few to resample).
    pub vertex_sigma: f64,
    /// Fitted metric value at the vertex.
    pub extremum: f64,
    /// d²y/dx²: positive for a minimum, negative for a maximum.
    pub curvature: f64,
    /// Residual RMS about the parabola, in metric units.
    pub rms: f64,
    pub n: usize,
    /// Vertex lies inside the sampled span — i.e. it is bracketed, not
    /// extrapolated. An unbracketed vertex is a hint, not an answer.
    pub bracketed: bool,
    /// Curvature has the sign the metric calls for (min for FWHM-like metrics,
    /// max for contrast-like ones). False means the samples do not describe a
    /// focus curve at all — usually too small a range, or pure noise.
    pub shape_ok: bool,
}

impl ParabolaFit {
    /// A fit worth acting on: right shape, bracketed, and the extremum is
    /// resolved better than the sample spacing implies is meaningless.
    pub fn trustworthy(&self) -> bool {
        self.shape_ok && self.bracketed && self.n >= 4
    }
}

/// Least-squares parabola through the samples; `want_min` selects the expected
/// curvature sign. Returns `None` when the samples cannot define a parabola at
/// all (fewer than three distinct positions, or a degenerate/linear fit).
pub fn fit_parabola(samples: &[VSample], want_min: bool) -> Option<ParabolaFit> {
    let (vertex, curvature, extremum) = solve(samples)?;

    let (lo, hi) = span(samples);
    let bracketed = vertex >= lo && vertex <= hi;
    let shape_ok = if want_min {
        curvature > 0.0
    } else {
        curvature < 0.0
    };

    // Residual RMS about the fitted curve, weighted the same way as the fit.
    let coeffs = coeffs(samples)?;
    let mut wsum = 0.0;
    let mut acc = 0.0;
    for s in samples {
        let w = s.weight.max(1e-6);
        let r = s.value - polyval(&coeffs, s.pos);
        acc += w * r * r;
        wsum += w;
    }
    let rms = if wsum > 0.0 { (acc / wsum).sqrt() } else { 0.0 };

    Some(ParabolaFit {
        vertex,
        vertex_sigma: jackknife_sigma(samples),
        extremum,
        curvature,
        rms,
        n: samples.len(),
        bracketed,
        shape_ok,
    })
}

/// Leave-one-out scatter of the vertex. NaN when there are too few samples for
/// the resampling to mean anything.
fn jackknife_sigma(samples: &[VSample]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return f64::NAN;
    }
    let mut vertices = Vec::with_capacity(n);
    let mut subset = Vec::with_capacity(n - 1);
    for skip in 0..n {
        subset.clear();
        subset.extend(
            samples
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != skip)
                .map(|(_, s)| *s),
        );
        // A dropped sample can leave fewer than three distinct positions; those
        // replicates simply do not contribute.
        if let Some((v, _, _)) = solve(&subset) {
            vertices.push(v);
        }
    }
    if vertices.len() < 3 {
        return f64::NAN;
    }
    let m = vertices.len() as f64;
    let mean = vertices.iter().sum::<f64>() / m;
    let ss: f64 = vertices.iter().map(|v| (v - mean) * (v - mean)).sum();
    // Standard jackknife inflation: the leave-one-out replicates under-disperse
    // by (n-1)/n relative to the sampling distribution.
    (ss * (m - 1.0) / m).sqrt()
}

fn coeffs(samples: &[VSample]) -> Option<Vec<f64>> {
    if distinct_positions(samples) < 3 {
        return None;
    }
    let xs: Vec<f64> = samples.iter().map(|s| s.pos).collect();
    let ys: Vec<f64> = samples.iter().map(|s| s.value).collect();
    let ws: Vec<f64> = samples.iter().map(|s| s.weight.max(1e-6)).collect();
    if !xs.iter().chain(ys.iter()).all(|v| v.is_finite()) {
        return None;
    }
    polyfit_weighted(&xs, &ys, &ws, 2)
}

/// (vertex, curvature, value at vertex)
fn solve(samples: &[VSample]) -> Option<(f64, f64, f64)> {
    let c = coeffs(samples)?;
    let a = *c.get(2)?;
    // A near-zero quadratic term means the samples are effectively collinear:
    // the vertex runs off to infinity and any number we report is noise.
    if !a.is_finite() || a.abs() < 1e-12 {
        return None;
    }
    let vertex = -c[1] / (2.0 * a);
    if !vertex.is_finite() {
        return None;
    }
    Some((vertex, 2.0 * a, polyval(&c, vertex)))
}

fn span(samples: &[VSample]) -> (f64, f64) {
    let lo = samples.iter().map(|s| s.pos).fold(f64::MAX, f64::min);
    let hi = samples.iter().map(|s| s.pos).fold(f64::MIN, f64::max);
    (lo, hi)
}

/// Positions differing by more than a whisker count as distinct; capturing
/// twice at the same setting is legitimate averaging, not extra leverage.
fn distinct_positions(samples: &[VSample]) -> usize {
    let mut pos: Vec<f64> = samples.iter().map(|s| s.pos).collect();
    pos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, hi) = (pos.first().copied().unwrap_or(0.0), pos.last().copied().unwrap_or(0.0));
    let tol = ((hi - lo).abs() * 1e-6).max(1e-9);
    let mut n = 0;
    let mut last = f64::NEG_INFINITY;
    for p in pos {
        if (p - last).abs() > tol {
            n += 1;
            last = p;
        }
    }
    n
}

// ---------------------------------------------------------------------------

/// One accumulating V-curve: samples plus the curvature sign its metric wants.
#[derive(Clone)]
pub struct VCurve {
    pub samples: Vec<VSample>,
    /// True for FWHM-like metrics (minimise), false for contrast-like ones.
    pub want_min: bool,
}

impl VCurve {
    pub fn new(want_min: bool) -> VCurve {
        VCurve {
            samples: Vec::new(),
            want_min,
        }
    }

    pub fn push(&mut self, pos: f64, value: f64, weight: f64) {
        if pos.is_finite() && value.is_finite() {
            self.samples.push(VSample { pos, value, weight });
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn undo(&mut self) {
        self.samples.pop();
    }

    pub fn fit(&self) -> Option<ParabolaFit> {
        fit_parabola(&self.samples, self.want_min)
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// ---------------------------------------------------------------------------

/// Stage A's answer: the signed split between the spectral and spatial focus
/// positions, and its uncertainty.
///
/// The split exists because the grating is anamorphic. A beam arriving at the
/// grating with wavefront curvature 1/R leaves with curvature 1/(r²R) in the
/// dispersion plane and 1/R unchanged in the slit-length plane, where
/// r = cos α / cos β ≈ 2–2.6 for a Sol'Ex-type layout at Hα with 2400 l/mm.
/// That is astigmatism, with lever (r² − 1) ≈ 3–6, and it vanishes identically
/// when — and only when — the slit sits at the collimator's focal length. So
/// Δ = 0 is a null test on collimation, and it supplies the infinity reference
/// for the camera lens that cannot be obtained mechanically.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub spectral: ParabolaFit,
    pub spatial: ParabolaFit,
    /// spectral vertex − spatial vertex, in micrometer units.
    pub delta: f64,
    /// Quadrature sum of the two jackknife sigmas; NaN if either is unknown.
    pub delta_sigma: f64,
}

impl Split {
    /// Δ is consistent with zero at 1σ — collimation is closed as far as this
    /// data can tell. Without a usable sigma, fall back on the residual scatter
    /// of the two fits rather than declaring success on a single number.
    pub fn nulled(&self) -> bool {
        if self.delta_sigma.is_finite() {
            self.delta.abs() <= self.delta_sigma
        } else {
            false
        }
    }
}

pub fn split(spectral: &VCurve, spatial: &VCurve) -> Option<Split> {
    let s = spectral.fit()?;
    let k = spatial.fit()?;
    let delta = s.vertex - k.vertex;
    let delta_sigma = if s.vertex_sigma.is_finite() && k.vertex_sigma.is_finite() {
        (s.vertex_sigma * s.vertex_sigma + k.vertex_sigma * k.vertex_sigma).sqrt()
    } else {
        f64::NAN
    };
    Some(Split {
        spectral: s,
        spatial: k,
        delta,
        delta_sigma,
    })
}

// ---------------------------------------------------------------------------

/// A (collimator position, measured Δ) pair from one completed Stage A sweep.
#[derive(Clone, Copy, Debug)]
pub struct NullPoint {
    pub collimator: f64,
    pub delta: f64,
}

/// Where to set the collimator so that Δ = 0, solved empirically.
///
/// This deliberately assumes nothing about the sign or the gain of dΔ/d(collimator).
/// Both depend on the instrument's geometry, and getting either wrong from
/// theory would send the user the wrong way. Instead: measure Δ at two or more
/// collimator settings, fit a straight line, and read off the root. The fit
/// self-calibrates the sign *and* the scale, so the second iteration lands on
/// the answer instead of hunting.
#[derive(Clone, Copy, Debug)]
pub struct NullSolution {
    /// Collimator position predicted to null Δ.
    pub collimator: f64,
    /// dΔ/d(collimator) — the sign convention, measured rather than assumed.
    pub gain: f64,
    pub n: usize,
    /// The root lies inside the sampled collimator range.
    pub bracketed: bool,
}

pub fn solve_null(points: &[NullPoint]) -> Option<NullSolution> {
    if points.len() < 2 {
        return None;
    }
    let xs: Vec<f64> = points.iter().map(|p| p.collimator).collect();
    let ys: Vec<f64> = points.iter().map(|p| p.delta).collect();
    let ws = vec![1.0; points.len()];
    let lo = xs.iter().cloned().fold(f64::MAX, f64::min);
    let hi = xs.iter().cloned().fold(f64::MIN, f64::max);
    if (hi - lo).abs() < 1e-9 {
        return None;
    }
    let c = polyfit_weighted(&xs, &ys, &ws, 1)?;
    let gain = c[1];
    if !gain.is_finite() || gain.abs() < 1e-12 {
        return None;
    }
    let root = -c[0] / gain;
    if !root.is_finite() {
        return None;
    }
    Some(NullSolution {
        collimator: root,
        gain,
        n: points.len(),
        bracketed: root >= lo && root <= hi,
    })
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(vertex: f64, a: f64, positions: &[f64]) -> Vec<VSample> {
        positions
            .iter()
            .map(|&p| VSample {
                pos: p,
                value: a * (p - vertex) * (p - vertex) + 1.0,
                weight: 1.0,
            })
            .collect()
    }

    #[test]
    fn recovers_a_noiseless_minimum() {
        let s = curve(0.37, 4.0, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let f = fit_parabola(&s, true).unwrap();
        assert!((f.vertex - 0.37).abs() < 1e-9, "vertex {}", f.vertex);
        assert!(f.curvature > 0.0);
        assert!(f.shape_ok && f.bracketed);
        assert!(f.rms < 1e-9);
    }

    #[test]
    fn recovers_a_maximum_for_contrast_metrics() {
        let s = curve(0.22, -2.5, &[0.0, 0.1, 0.2, 0.3, 0.4]);
        let f = fit_parabola(&s, false).unwrap();
        assert!((f.vertex - 0.22).abs() < 1e-9);
        assert!(f.curvature < 0.0 && f.shape_ok);
    }

    #[test]
    fn flags_wrong_curvature_instead_of_lying() {
        // A minimum-shaped curve handed to a metric that expects a maximum.
        let s = curve(0.3, 3.0, &[0.1, 0.2, 0.3, 0.4, 0.5]);
        let f = fit_parabola(&s, false).unwrap();
        assert!(!f.shape_ok);
    }

    #[test]
    fn flags_an_unbracketed_vertex() {
        let s = curve(0.9, 3.0, &[0.1, 0.2, 0.3]);
        let f = fit_parabola(&s, true).unwrap();
        assert!(!f.bracketed, "vertex {} should be outside 0.1..0.3", f.vertex);
    }

    #[test]
    fn refuses_collinear_samples() {
        let s: Vec<VSample> = [0.1, 0.2, 0.3, 0.4]
            .iter()
            .map(|&p| VSample {
                pos: p,
                value: 2.0 * p + 1.0,
                weight: 1.0,
            })
            .collect();
        assert!(fit_parabola(&s, true).is_none());
    }

    #[test]
    fn refuses_fewer_than_three_distinct_positions() {
        let s: Vec<VSample> = [0.1, 0.1, 0.2, 0.2]
            .iter()
            .map(|&p| VSample {
                pos: p,
                value: p * p,
                weight: 1.0,
            })
            .collect();
        assert!(fit_parabola(&s, true).is_none());
    }

    #[test]
    fn jackknife_is_small_when_clean_and_grows_with_noise() {
        let clean = curve(0.37, 4.0, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let f_clean = fit_parabola(&clean, true).unwrap();
        assert!(f_clean.vertex_sigma < 1e-6);

        let mut noisy = clean.clone();
        // A deterministic, alternating perturbation — no RNG in tests.
        for (i, s) in noisy.iter_mut().enumerate() {
            s.value += if i % 2 == 0 { 0.02 } else { -0.02 };
        }
        let f_noisy = fit_parabola(&noisy, true).unwrap();
        assert!(f_noisy.vertex_sigma > f_clean.vertex_sigma);
        assert!(f_noisy.vertex_sigma.is_finite());
    }

    #[test]
    fn split_subtracts_the_two_vertices() {
        let mut spec = VCurve::new(true);
        let mut slit = VCurve::new(true);
        for p in [0.1, 0.2, 0.3, 0.4, 0.5] {
            spec.push(p, 4.0 * (p - 0.34) * (p - 0.34) + 1.0, 1.0);
            slit.push(p, 3.0 * (p - 0.28) * (p - 0.28) + 1.0, 1.0);
        }
        let s = split(&spec, &slit).unwrap();
        assert!((s.delta - 0.06).abs() < 1e-9, "delta {}", s.delta);
    }

    #[test]
    fn null_solve_finds_the_root_and_the_sign() {
        // Δ = 0.5·(collimator − 12.4)
        let pts = [
            NullPoint { collimator: 12.0, delta: -0.2 },
            NullPoint { collimator: 12.8, delta: 0.2 },
        ];
        let sol = solve_null(&pts).unwrap();
        assert!((sol.collimator - 12.4).abs() < 1e-9);
        assert!((sol.gain - 0.5).abs() < 1e-9);
        assert!(sol.bracketed);
    }

    #[test]
    fn null_solve_needs_two_distinct_collimator_settings() {
        let pts = [
            NullPoint { collimator: 12.0, delta: -0.2 },
            NullPoint { collimator: 12.0, delta: 0.2 },
        ];
        assert!(solve_null(&pts).is_none());
    }
}
