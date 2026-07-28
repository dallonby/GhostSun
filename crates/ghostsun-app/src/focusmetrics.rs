//! Stage B metrics: telescope focus, measured through the slit.
//!
//! Stage A (collimator + camera) is measured from the two line families the
//! focus tab already fits — spectral absorption lines and slit-jaw dust. Both
//! of those are, by construction, **blind to the telescope**: dust sits
//! physically in the slit plane and the spectral line is the dispersed image of
//! that same plane, so the collimator/camera pair relays them to the sensor no
//! matter where the telescope's focal plane happens to sit. That blindness is
//! what makes the three-unknown problem triangular rather than degenerate, and
//! it is also why Stage A cannot be used to focus the telescope: a third
//! observable is needed, one that responds to solar structure.
//!
//! Two are provided here, because they fail in different ways:
//!
//! * [`limb_edge_width`] — the solar limb crossing the slit, used as a knife
//!   edge. A limb is a disk-to-sky step of order 100%, so slit-jaw dust (a few
//!   percent) is negligible against it: this metric is essentially dust-immune,
//!   and it is the one to trust. It needs the limb on the slit.
//! * [`structure_contrast`] — high-passed along-slit contrast (granulation and
//!   whatever else is on the slit). Always available, but dust contributes a
//!   **constant pedestal** to it. That pedestal does not move with telescope
//!   focus, so the position of the maximum is unaffected and the metric is
//!   still valid for focusing; what it costs is contrast, so the curve is
//!   shallower than it looks like it should be. Do not read its absolute value
//!   as a seeing or resolution figure.
//!
//! Both are computed on **continuum** columns rather than the line core. The
//! core is low-contrast chromosphere, and scattered light flattens its focus
//! curve; the continuum carries granulation and a hard limb. Continuum and line
//! are present in the same frame at different dispersion positions, so this
//! costs nothing.

use std::collections::VecDeque;

use ghostsun_core::mathutil::{fit_inverted_gaussian, gaussian_smooth, robust_trend};

/// Fraction of the local continuum a dispersion position must reach to count as
/// continuum rather than line. 2% down is comfortably outside the wings of
/// anything worth calling a line at these resolutions.
const CONTINUUM_TOL: f64 = 0.98;
/// Progressive relaxations if the strict threshold leaves too little to average.
const CONTINUUM_FALLBACKS: [f64; 3] = [0.95, 0.90, 0.80];
/// Minimum fraction of the dispersion axis that must survive the mask.
const CONTINUUM_MIN_FRAC: f64 = 0.05;
/// Rows below this fraction of the profile peak are unilluminated.
const ILLUM_GATE: f64 = 0.40;
/// High-pass scale for the structure metric, in pixels along the slit.
const STRUCTURE_SIGMA: f64 = 6.0;

/// Dispersion positions that are continuum, i.e. not inside an absorption line.
///
/// The local continuum level is estimated with a median-based trend, which
/// rejects the lines rather than being dragged down by them. A rolling median
/// is biased on a curved profile, which matters a great deal for ratio-based
/// *gains* — but this is only ever used to build a boolean mask, where a
/// sub-percent bias changes nothing.
pub fn continuum_mask(spectrum: &[f64]) -> Vec<bool> {
    let n = spectrum.len();
    if n < 16 {
        return vec![true; n];
    }
    let win = (n / 8).max(11) | 1;
    let envelope = robust_trend(spectrum, win, 2.0);
    let ratio: Vec<f64> = spectrum
        .iter()
        .zip(&envelope)
        .map(|(&s, &e)| if e > 1e-9 { s / e } else { 0.0 })
        .collect();

    let need = ((n as f64) * CONTINUUM_MIN_FRAC).ceil() as usize;
    for tol in std::iter::once(CONTINUUM_TOL).chain(CONTINUUM_FALLBACKS) {
        let mask: Vec<bool> = ratio.iter().map(|&r| r >= tol).collect();
        if mask.iter().filter(|&&m| m).count() >= need.max(4) {
            return mask;
        }
    }
    // Nothing looks like continuum — a saturated or blank frame. Use it all
    // rather than returning an empty profile the caller has to special-case.
    vec![true; n]
}

/// Mean intensity along the slit, averaged over continuum positions only.
///
/// `dispersion_horizontal` matches [`ghostsun_camera::Frame::mean_profile`]:
/// true means dispersion runs along x and the slit along y.
pub fn slit_profile_continuum(
    data: &[u16],
    w: usize,
    h: usize,
    dispersion_horizontal: bool,
    mask: &[bool],
) -> Vec<f64> {
    if w == 0 || h == 0 || data.len() < w * h {
        return Vec::new();
    }
    if dispersion_horizontal {
        // Dispersion along x, slit along y: average the masked columns.
        let cols: Vec<usize> = (0..w).filter(|&x| mask.get(x).copied().unwrap_or(true)).collect();
        if cols.is_empty() {
            return Vec::new();
        }
        let inv = 1.0 / cols.len() as f64;
        (0..h)
            .map(|y| {
                let row = &data[y * w..(y + 1) * w];
                cols.iter().map(|&x| row[x] as f64).sum::<f64>() * inv
            })
            .collect()
    } else {
        // Dispersion along y, slit along x: average the masked rows.
        let rows: Vec<usize> = (0..h).filter(|&y| mask.get(y).copied().unwrap_or(true)).collect();
        if rows.is_empty() {
            return Vec::new();
        }
        let inv = 1.0 / rows.len() as f64;
        (0..w)
            .map(|x| rows.iter().map(|&y| data[y * w + x] as f64).sum::<f64>() * inv)
            .collect()
    }
}

/// The illuminated run of the slit profile, as a half-open index range.
pub fn illuminated_span(profile: &[f64]) -> Option<(usize, usize)> {
    if profile.len() < 8 {
        return None;
    }
    let sm = gaussian_smooth(profile, 2.0);
    let peak = sm.iter().cloned().fold(f64::MIN, f64::max);
    if peak <= 1e-9 {
        return None;
    }
    let thr = ILLUM_GATE * peak;
    let lo = sm.iter().position(|&v| v >= thr)?;
    let hi = sm.iter().rposition(|&v| v >= thr)? + 1;
    if hi.saturating_sub(lo) < 8 {
        None
    } else {
        Some((lo, hi))
    }
}

/// High-passed along-slit contrast over `[lo, hi)`, as a fraction of the mean.
///
/// Maximised at best telescope focus. See the module note on the dust pedestal.
pub fn structure_contrast(profile: &[f64], lo: usize, hi: usize) -> Option<f64> {
    if hi <= lo || hi > profile.len() || hi - lo < 8 {
        return None;
    }
    let seg = &profile[lo..hi];
    let smooth = gaussian_smooth(seg, STRUCTURE_SIGMA);
    let mean = seg.iter().sum::<f64>() / seg.len() as f64;
    if mean <= 1e-9 {
        return None;
    }
    let ss: f64 = seg
        .iter()
        .zip(&smooth)
        .map(|(&v, &s)| (v - s) * (v - s))
        .sum();
    Some((ss / seg.len() as f64).sqrt() / mean)
}

/// Structure contrast over the whole illuminated span and over its outer
/// thirds, for the field-curvature / slit-tilt check.
///
/// At ~1200 mm focal length a 7 mm slit subtends about 20 arcmin — most of the
/// solar diameter — so a refractor's field curvature is not automatically
/// negligible. If `top` and `bottom` peak at different focuser positions, that
/// is tilt or curvature, and no single focus setting fixes it.
#[derive(Clone, Copy, Debug, Default)]
pub struct StructureSplit {
    pub all: Option<f64>,
    pub top: Option<f64>,
    pub bottom: Option<f64>,
}

pub fn structure_split(profile: &[f64]) -> StructureSplit {
    let Some((lo, hi)) = illuminated_span(profile) else {
        return StructureSplit::default();
    };
    let n = hi - lo;
    let third = n / 3;
    StructureSplit {
        all: structure_contrast(profile, lo, hi),
        top: structure_contrast(profile, lo, lo + third),
        bottom: structure_contrast(profile, hi - third, hi),
    }
}

/// FWHM, in pixels along the slit, of the sharpest intensity step in the
/// profile — the solar limb used as a knife edge. Minimised at best focus.
///
/// The gradient of an erf-shaped edge is a Gaussian, so the edge width is
/// recovered by fitting a Gaussian to the derivative peak. Returns `None` when
/// no step is deep enough to be a limb, which is the honest answer when the
/// slit is entirely on the disk or entirely off it.
pub fn limb_edge_width(profile: &[f64]) -> Option<f64> {
    let n = profile.len();
    if n < 24 {
        return None;
    }
    // Light smoothing before differentiating: the derivative of raw photon
    // noise has no maximum worth finding.
    let sm = gaussian_smooth(profile, 1.0);
    let peak = sm.iter().cloned().fold(f64::MIN, f64::max);
    let floor = sm.iter().cloned().fold(f64::MAX, f64::min);
    let range = peak - floor;
    if range <= 1e-9 {
        return None;
    }

    let deriv: Vec<f64> = (0..n)
        .map(|i| {
            let a = sm[i.saturating_sub(1)];
            let b = sm[(i + 1).min(n - 1)];
            (b - a) * 0.5
        })
        .collect();

    let (idx, &gmax) = deriv
        .iter()
        .enumerate()
        .skip(4)
        .take(n.saturating_sub(8))
        .map(|(i, d)| (i, d))
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))?;

    // Gate: a real limb crosses a large fraction of the profile's range within
    // a short run. Without this, the "sharpest edge" of a flat on-disk profile
    // is just the loudest noise spike, and its fitted width is meaningless.
    let half = 12.min(idx).min(n - 1 - idx);
    if half < 6 {
        return None;
    }
    let step = (sm[idx + half] - sm[idx - half]).abs();
    if step < 0.25 * range {
        return None;
    }

    // Fit the derivative peak. `fit_inverted_gaussian` fits a dip, so negate
    // the magnitude of the derivative to present the peak as one.
    let lo = idx - half;
    let hi = idx + half;
    let xs: Vec<f64> = (lo..=hi).map(|i| i as f64).collect();
    let ys: Vec<f64> = (lo..=hi).map(|i| -deriv[i].abs()).collect();
    let sigma0 = 2.0;
    let (_mu, sigma, amp, _off) = fit_inverted_gaussian(&xs, &ys, idx as f64, sigma0)?;
    if !sigma.is_finite() || sigma <= 0.0 || amp <= 0.0 {
        return None;
    }
    // Reject a fit that ran away to the window edge rather than converging.
    if sigma > half as f64 {
        return None;
    }
    let _ = gmax;
    Some(2.3548 * sigma)
}

// ---------------------------------------------------------------------------

/// Rolling buffer of per-frame metric values, read out as a percentile.
///
/// Seeing dominates any single frame, so a mean would measure the atmosphere
/// rather than the focuser. Taking the best decile ("lucky" selection) tracks
/// the instrument's own ceiling, which is what focusing should optimise.
pub struct LuckyBuf {
    values: VecDeque<f64>,
    capacity: usize,
}

impl LuckyBuf {
    pub fn new(capacity: usize) -> LuckyBuf {
        LuckyBuf {
            values: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    pub fn push(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        if self.values.len() >= self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(v);
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// The "lucky" readout: mean of the best decile, for the metric's polarity.
    pub fn lucky(&self, want_min: bool) -> Option<f64> {
        let v: Vec<f64> = self.values.iter().copied().collect();
        lucky_mean(&v, want_min, LUCKY_FRACTION)
    }
}

/// Fraction of frames the lucky readout keeps.
pub const LUCKY_FRACTION: f64 = 0.10;

/// Mean of the best `frac` of the values, for the metric's polarity.
///
/// Deliberately a mean over the tail rather than a percentile *point*. With a
/// bimodal burst — the usual shape under seeing, a mass of poor frames and a
/// few good ones — a point estimate at the decile lands exactly on the
/// boundary, where interpolation drags it onto the bad side and the readout
/// reports roughly the bad mode. Averaging the tail is the standard lucky-
/// imaging selection, is stable at that boundary, and degrades gracefully as
/// the good fraction changes.
pub fn lucky_mean(values: &[f64], want_min: bool, frac: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = ((frac.clamp(0.0, 1.0) * v.len() as f64).ceil() as usize).clamp(1, v.len());
    let tail: &[f64] = if want_min {
        &v[..k]
    } else {
        &v[v.len() - k..]
    };
    Some(tail.iter().sum::<f64>() / tail.len() as f64)
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn erf_edge(n: usize, center: f64, width: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = (i as f64 - center) / width;
                // Disk on the left, sky on the right.
                500.0 * (1.0 - ghostsun_core::mathutil::erf(t)) + 20.0
            })
            .collect()
    }

    #[test]
    fn continuum_mask_excludes_a_line_and_keeps_the_rest() {
        let mut spec = vec![1000.0; 200];
        for i in 95..105 {
            let d = (i as f64 - 100.0) / 3.0;
            spec[i] -= 600.0 * (-d * d).exp();
        }
        let mask = continuum_mask(&spec);
        assert!(!mask[100], "line core should be masked out");
        assert!(mask[10] && mask[190], "far continuum should survive");
        let kept = mask.iter().filter(|&&m| m).count();
        assert!(kept > 150, "kept only {kept} of 200");
    }

    #[test]
    fn continuum_mask_never_returns_empty() {
        let spec = vec![0.0; 64];
        let mask = continuum_mask(&spec);
        assert!(mask.iter().any(|&m| m));
    }

    #[test]
    fn slit_profile_averages_only_masked_columns() {
        // 4 wide, 3 tall; column 1 is "line" and much darker.
        let w = 4;
        let h = 3;
        let data: Vec<u16> = vec![
            100, 10, 100, 100, //
            200, 10, 200, 200, //
            300, 10, 300, 300,
        ];
        let mask = vec![true, false, true, true];
        let prof = slit_profile_continuum(&data, w, h, true, &mask);
        assert_eq!(prof, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn slit_profile_handles_the_transposed_orientation() {
        let w = 3;
        let h = 4;
        // Dispersion along y: rows are wavelengths, row 1 is the line.
        let data: Vec<u16> = vec![
            100, 200, 300, //
            10, 10, 10, //
            100, 200, 300, //
            100, 200, 300,
        ];
        let mask = vec![true, false, true, true];
        let prof = slit_profile_continuum(&data, w, h, false, &mask);
        assert_eq!(prof, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn limb_width_tracks_the_true_edge_width() {
        let sharp = limb_edge_width(&erf_edge(120, 60.0, 1.5)).unwrap();
        let soft = limb_edge_width(&erf_edge(120, 60.0, 4.0)).unwrap();
        assert!(sharp < soft, "sharp {sharp} should beat soft {soft}");
        // Monotone and sane in magnitude: a wider erf must not read narrower.
        assert!(sharp > 0.5 && soft < 40.0, "sharp {sharp} soft {soft}");
    }

    #[test]
    fn limb_width_declines_to_guess_without_a_limb() {
        // Flat on-disk profile with mild noise-like ripple: no step to fit.
        let flat: Vec<f64> = (0..120)
            .map(|i| 500.0 + 3.0 * ((i as f64) * 0.7).sin())
            .collect();
        assert!(limb_edge_width(&flat).is_none());
    }

    #[test]
    fn structure_contrast_rises_with_real_structure() {
        let flat: Vec<f64> = vec![500.0; 120];
        let textured: Vec<f64> = (0..120)
            .map(|i| 500.0 + 25.0 * ((i as f64) * 1.3).sin())
            .collect();
        let (lo, hi) = (0, 120);
        let a = structure_contrast(&flat, lo, hi).unwrap();
        let b = structure_contrast(&textured, lo, hi).unwrap();
        assert!(b > a * 5.0, "flat {a} textured {b}");
    }

    #[test]
    fn structure_contrast_is_scale_invariant() {
        let a: Vec<f64> = (0..120)
            .map(|i| 500.0 + 25.0 * ((i as f64) * 1.3).sin())
            .collect();
        let b: Vec<f64> = a.iter().map(|v| v * 7.0).collect();
        let ca = structure_contrast(&a, 0, 120).unwrap();
        let cb = structure_contrast(&b, 0, 120).unwrap();
        assert!((ca - cb).abs() < 1e-9, "{ca} vs {cb}");
    }

    #[test]
    fn illuminated_span_finds_the_lit_run() {
        let mut p = vec![10.0; 200];
        p[50..150].fill(1000.0);
        let (lo, hi) = illuminated_span(&p).unwrap();
        assert!(lo >= 45 && lo <= 55, "lo {lo}");
        assert!(hi >= 145 && hi <= 155, "hi {hi}");
    }

    #[test]
    fn lucky_picks_the_good_tail_not_the_average() {
        let mut b = LuckyBuf::new(100);
        // Mostly bad seeing (large FWHM), a few good frames.
        for _ in 0..90 {
            b.push(5.0);
        }
        for _ in 0..10 {
            b.push(2.0);
        }
        let lucky = b.lucky(true).unwrap();
        assert!(lucky < 3.0, "lucky {lucky} should follow the good frames");
    }

    #[test]
    fn lucky_is_stable_across_the_decile_boundary() {
        // The failure mode of a percentile *point* estimate: with the good/bad
        // split sitting exactly on the decile, interpolation reports the bad
        // mode. A tail mean must not do that.
        let readout = |good: usize| {
            let mut b = LuckyBuf::new(100);
            for _ in 0..(100 - good) {
                b.push(5.0);
            }
            for _ in 0..good {
                b.push(2.0);
            }
            b.lucky(true).unwrap()
        };
        // The window is 10 frames of 100. Once there are that many good ones it
        // is pure — including exactly at the boundary, which is the case that
        // broke the percentile version.
        for good in [10usize, 11, 20, 50] {
            let v = readout(good);
            assert!((v - 2.0).abs() < 1e-9, "good={good} gave {v}");
        }
        // Below the window size the selection is necessarily diluted — that is
        // inherent to fixed-fraction selection and harmless, because the same
        // dilution applies at every focuser position. What must hold is that
        // the readout never moves the wrong way as conditions improve.
        let mut prev = f64::INFINITY;
        for good in [0usize, 2, 5, 9, 10] {
            let v = readout(good);
            assert!(v <= prev + 1e-9, "good={good}: {v} is worse than {prev}");
            prev = v;
        }
    }

    #[test]
    fn lucky_mean_respects_polarity() {
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // Best decile of ten values is one value.
        assert_eq!(lucky_mean(&v, true, 0.10).unwrap(), 1.0);
        assert_eq!(lucky_mean(&v, false, 0.10).unwrap(), 10.0);
        // A wider fraction averages the tail.
        assert_eq!(lucky_mean(&v, true, 0.30).unwrap(), 2.0);
    }

    #[test]
    fn lucky_mean_ignores_non_finite_values() {
        let v = vec![f64::NAN, 2.0, 3.0, f64::INFINITY];
        assert_eq!(lucky_mean(&v, true, 0.5).unwrap(), 2.0);
    }

    #[test]
    fn lucky_buf_respects_capacity() {
        let mut b = LuckyBuf::new(10);
        for i in 0..50 {
            b.push(i as f64);
        }
        assert_eq!(b.len(), 10);
    }
}
