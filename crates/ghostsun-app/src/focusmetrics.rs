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

/// Sub-pixel H-alpha half-depth edge measurement along the slit.
///
/// `center_rms_px` is coherent displacement of the whole absorption line
/// (Doppler/filament structure); `width_rms_px` is modulation of its half-width
/// (opacity/shape structure). Both are band-limited after removing smile and
/// pixel-scale fitting noise. `focus_score` is deliberately unitless and is
/// maximised: resolved edge structure × edge-gradient SNR.
#[derive(Clone, Debug, Default)]
pub struct HaEdgeMetrics {
    pub focus_score: f64,
    pub jaggedness_rms_px: f64,
    pub center_rms_px: f64,
    pub width_rms_px: f64,
    pub edge_snr: f64,
    pub dance_rms_px: Option<f64>,
    /// Local frame-to-frame edge displacement after removing rigid line motion,
    /// one value per slit pixel. This highlights solar structure moving through
    /// the slit while rejecting fixed smile and detector artefacts.
    pub motion: Vec<f32>,
    pub valid_fraction: f64,
    /// Edge coordinates on the frame's dispersion axis, one per slit pixel.
    pub edge_lo: Vec<f32>,
    pub edge_hi: Vec<f32>,
}

fn median_finite(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.into_iter().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

fn fill_missing(trace: &mut [f64]) -> bool {
    let valid: Vec<usize> = trace
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.is_finite().then_some(i))
        .collect();
    if valid.len() < 8 {
        return false;
    }
    let first = valid[0];
    let last = *valid.last().unwrap();
    let first_value = trace[first];
    let last_value = trace[last];
    trace[..first].fill(first_value);
    trace[last + 1..].fill(last_value);
    for pair in valid.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b == a + 1 {
            continue;
        }
        let (va, vb) = (trace[a], trace[b]);
        for (i, value) in trace.iter_mut().enumerate().take(b).skip(a + 1) {
            let f = (i - a) as f64 / (b - a) as f64;
            *value = va + f * (vb - va);
        }
    }
    true
}

fn trace_detail(trace: &[f64], valid: &[bool]) -> (Vec<f64>, f64, f64) {
    let light = gaussian_smooth(trace, 1.2);
    let broad_sigma = ((trace.len() as f64) / 40.0).clamp(12.0, 64.0);
    let broad = gaussian_smooth(&light, broad_sigma);
    let detail: Vec<f64> = light.iter().zip(&broad).map(|(&a, &b)| a - b).collect();
    let mut detail_ss = 0.0;
    let mut noise_ss = 0.0;
    let mut n = 0usize;
    for i in 0..trace.len() {
        if valid.get(i).copied().unwrap_or(false) {
            detail_ss += detail[i] * detail[i];
            let noise = trace[i] - light[i];
            noise_ss += noise * noise;
            n += 1;
        }
    }
    if n == 0 {
        return (detail, 0.0, 0.0);
    }
    (
        detail,
        (detail_ss / n as f64).sqrt(),
        (noise_ss / n as f64).sqrt(),
    )
}

/// Track both half-depth edges of the selected absorption line along the slit.
pub fn ha_line_edges(
    data: &[u16],
    w: usize,
    h: usize,
    dispersion_horizontal: bool,
    line_center: f64,
) -> Option<HaEdgeMetrics> {
    if w == 0 || h == 0 || data.len() < w * h {
        return None;
    }
    let (spec_len, slit_len) = if dispersion_horizontal { (w, h) } else { (h, w) };
    if spec_len < 24 || slit_len < 32 || !line_center.is_finite() {
        return None;
    }
    let center = line_center.round().clamp(2.0, (spec_len - 3) as f64) as usize;
    let lo = center.saturating_sub(50);
    let hi = (center + 50).min(spec_len - 1);
    if hi - lo < 20 {
        return None;
    }
    let sample = |slit: usize, spec: usize| -> f64 {
        if dispersion_horizontal {
            data[slit * w + spec] as f64
        } else {
            data[spec * w + slit] as f64
        }
    };

    let mut edge_lo = vec![f64::NAN; slit_len];
    let mut edge_hi = vec![f64::NAN; slit_len];
    let mut continua = vec![0.0; slit_len];
    let mut slopes = vec![f64::NAN; slit_len];
    let mut intensity_noise = vec![f64::NAN; slit_len];
    let mut depths = vec![0.0; slit_len];
    let weights = [1.0, 4.0, 6.0, 4.0, 1.0];

    for slit in 0..slit_len {
        let mut profile = vec![0.0; hi - lo + 1];
        for (i, value) in profile.iter_mut().enumerate() {
            let spec = lo + i;
            let mut sum = 0.0;
            for (k, weight) in weights.iter().enumerate() {
                let p = (spec as isize + k as isize - 2).clamp(0, spec_len as isize - 1) as usize;
                sum += weight * sample(slit, p);
            }
            *value = sum / 16.0;
        }
        let wing_n = 8.min(profile.len() / 4).max(2);
        let left = profile[..wing_n].iter().sum::<f64>() / wing_n as f64;
        let right = profile[profile.len() - wing_n..].iter().sum::<f64>() / wing_n as f64;
        let continuum = 0.5 * (left + right);
        continua[slit] = continuum;

        let local_center = center - lo;
        let core_lo = local_center.saturating_sub(14);
        let core_hi = (local_center + 14).min(profile.len() - 1);
        let (core_i, &core) = profile[core_lo..=core_hi]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        let core_i = core_lo + core_i;
        if continuum <= 1.0 || core >= continuum {
            continue;
        }
        depths[slit] = (continuum - core) / continuum;
        let half = 0.5 * (continuum + core);
        let mut lower = None;
        for i in (1..=core_i).rev() {
            if profile[i - 1] >= half && profile[i] < half {
                let denom = profile[i] - profile[i - 1];
                if denom.abs() > 1e-9 {
                    lower = Some(lo as f64 + i as f64 - 1.0 + (half - profile[i - 1]) / denom);
                }
                break;
            }
        }
        let mut upper = None;
        for i in core_i..profile.len() - 1 {
            if profile[i] < half && profile[i + 1] >= half {
                let denom = profile[i + 1] - profile[i];
                if denom.abs() > 1e-9 {
                    upper = Some(lo as f64 + i as f64 + (half - profile[i]) / denom);
                }
                break;
            }
        }
        let (Some(lower), Some(upper)) = (lower, upper) else { continue };
        let width = upper - lower;
        if !(3.0..=70.0).contains(&width) {
            continue;
        }
        edge_lo[slit] = lower;
        edge_hi[slit] = upper;

        let li = (lower - lo as f64).round().clamp(1.0, (profile.len() - 2) as f64) as usize;
        let ui = (upper - lo as f64).round().clamp(1.0, (profile.len() - 2) as f64) as usize;
        slopes[slit] = 0.25
            * ((profile[li + 1] - profile[li - 1]).abs()
                + (profile[ui + 1] - profile[ui - 1]).abs())
            / continuum;
        let mut diff_ss = 0.0;
        let mut diff_n = 0usize;
        for wing in [&profile[..wing_n], &profile[profile.len() - wing_n..]] {
            for pair in wing.windows(2) {
                let d = (pair[1] - pair[0]) / continuum;
                diff_ss += d * d;
                diff_n += 1;
            }
        }
        intensity_noise[slit] = (diff_ss / (2.0 * diff_n.max(1) as f64)).sqrt();
    }

    let continuum_gate = 0.25 * median_finite(continua.iter().copied())?;
    let valid: Vec<bool> = (0..slit_len)
        .map(|i| {
            edge_lo[i].is_finite()
                && edge_hi[i].is_finite()
                && continua[i] >= continuum_gate
                && depths[i] >= 0.05
        })
        .collect();
    let valid_count = valid.iter().filter(|&&v| v).count();
    if valid_count < slit_len / 3 || valid_count < 32 {
        return None;
    }
    for i in 0..slit_len {
        if !valid[i] {
            edge_lo[i] = f64::NAN;
            edge_hi[i] = f64::NAN;
        }
    }
    if !fill_missing(&mut edge_lo) || !fill_missing(&mut edge_hi) {
        return None;
    }

    let center_trace: Vec<f64> = edge_lo
        .iter()
        .zip(&edge_hi)
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect();
    let half_width: Vec<f64> = edge_lo
        .iter()
        .zip(&edge_hi)
        .map(|(&a, &b)| 0.5 * (b - a))
        .collect();
    let (_, center_band, center_noise) = trace_detail(&center_trace, &valid);
    let (_, width_band, width_noise) = trace_detail(&half_width, &valid);
    let center_rms = (center_band * center_band - center_noise * center_noise)
        .max(0.0)
        .sqrt();
    let width_rms = (width_band * width_band - width_noise * width_noise)
        .max(0.0)
        .sqrt();
    let jaggedness = (center_rms * center_rms + width_rms * width_rms).sqrt();
    let slope = median_finite(slopes)?;
    let noise = median_finite(intensity_noise)?.max(1e-6);
    let edge_snr = slope / noise;
    let valid_fraction = valid_count as f64 / slit_len as f64;
    let focus_score = jaggedness * edge_snr.min(25.0) * valid_fraction.sqrt();

    Some(HaEdgeMetrics {
        focus_score,
        jaggedness_rms_px: jaggedness,
        center_rms_px: center_rms,
        width_rms_px: width_rms,
        edge_snr,
        dance_rms_px: None,
        motion: Vec::new(),
        valid_fraction,
        edge_lo: edge_lo.into_iter().map(|v| v as f32).collect(),
        edge_hi: edge_hi.into_iter().map(|v| v as f32).collect(),
    })
}

/// Frame-to-frame change in the edge shapes after removing whole-line motion.
/// Returns both a scalar RMS measurement and a local motion map along the slit.
pub fn ha_edge_motion(
    current: &HaEdgeMetrics,
    previous: &HaEdgeMetrics,
) -> Option<(f64, Vec<f32>)> {
    if current.edge_lo.len() != previous.edge_lo.len() || current.edge_lo.len() < 32 {
        return None;
    }
    let dlo: Vec<f64> = current
        .edge_lo
        .iter()
        .zip(&previous.edge_lo)
        .map(|(&a, &b)| (a - b) as f64)
        .collect();
    let dhi: Vec<f64> = current
        .edge_hi
        .iter()
        .zip(&previous.edge_hi)
        .map(|(&a, &b)| (a - b) as f64)
        .collect();
    let med_lo = median_finite(dlo.iter().copied())?;
    let med_hi = median_finite(dhi.iter().copied())?;
    let motion: Vec<f32> = dlo
        .iter()
        .zip(&dhi)
        .map(|(&a, &b)| {
            let a = a - med_lo;
            let b = b - med_hi;
            (0.5 * (a * a + b * b)).sqrt() as f32
        })
        .collect();
    let ss: f64 = motion.iter().map(|&v| (v as f64).powi(2)).sum();
    Some(((ss / motion.len() as f64).sqrt(), motion))
}

/// Scalar compatibility helper used by focus-metric tests and callers that do
/// not need the local motion map.
#[cfg(test)]
pub fn ha_edge_dance(current: &HaEdgeMetrics, previous: &HaEdgeMetrics) -> Option<f64> {
    ha_edge_motion(current, previous).map(|(rms, _)| rms)
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

    fn synthetic_ha_edges(w: usize, h: usize, phase: f64, wavy: bool) -> Vec<u16> {
        let mut data = vec![0u16; w * h];
        for x in 0..w {
            let xf = x as f64;
            let center = h as f64 / 2.0
                + if wavy {
                    2.0 * (0.12 * xf + phase).sin()
                } else {
                    0.0
                };
            let sigma = 7.0
                + if wavy {
                    0.8 * (0.19 * xf + 0.7 * phase).sin()
                } else {
                    0.0
                };
            for y in 0..h {
                let d = (y as f64 - center) / sigma;
                let value = 50_000.0 - 30_000.0 * (-0.5 * d * d).exp();
                data[y * w + x] = value.round() as u16;
            }
        }
        data
    }

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
    fn ha_edge_metric_detects_resolved_waves_and_temporal_dance() {
        let (w, h) = (640, 120);
        let straight_data = synthetic_ha_edges(w, h, 0.0, false);
        let first_data = synthetic_ha_edges(w, h, 0.0, true);
        let next_data = synthetic_ha_edges(w, h, 0.7, true);
        let straight = ha_line_edges(&straight_data, w, h, false, 60.0).unwrap();
        let first = ha_line_edges(&first_data, w, h, false, 60.0).unwrap();
        let next = ha_line_edges(&next_data, w, h, false, 60.0).unwrap();
        assert!(first.valid_fraction > 0.95, "lock {}", first.valid_fraction);
        assert!(
            first.jaggedness_rms_px > straight.jaggedness_rms_px + 0.25,
            "wavy {} straight {}",
            first.jaggedness_rms_px,
            straight.jaggedness_rms_px
        );
        let (dance, motion) = ha_edge_motion(&next, &first).unwrap();
        assert!(dance > 0.25, "dance {dance}");
        assert_eq!(motion.len(), w);
        assert!(motion.iter().any(|&v| v > 0.25));
        assert!((ha_edge_dance(&next, &first).unwrap() - dance).abs() < 1e-9);
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
