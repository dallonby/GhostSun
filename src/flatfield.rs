//! Photometric corrections on the reconstructed disk:
//!
//! 1. Per-frame (column) transparency normalization — sky transparency and
//!    scintillation multiply whole frames; INTI has no correction at all
//!    (its "bad lines" code is disabled). We estimate robust column flux over
//!    disk rows, divide out the high-frequency part of its trend.
//!
//! 2. Transversalium (row) correction — slit dust makes fixed horizontal
//!    stripes. INTI divides by a Savitzky-Golay smoothed profile (non-robust:
//!    filaments and plage bias it). We use median row statistics over the
//!    disk with a robust two-scale trend.

use crate::image2d::Image;
use crate::mathutil::{percentile_f32, robust_trend};

fn disk_threshold(disk: &Image) -> f32 {
    // robust disk level: 80th percentile is comfortably inside the disk
    percentile_f32(&disk.data, 80.0) * 0.35
}

/// Per-column (frame) transparency normalization.
///
/// Transparency only ever *reduces* flux relative to clear sky, so the
/// reference is a rolling upper-quantile envelope of the column flux (robust
/// to a passing cloud occupying < half the window), not a mean trend.
/// The correction is tapered off near the limbs where the chord flux changes
/// faster than any envelope can follow.
pub fn measure_column_flux(img: &Image) -> Vec<f64> {
    let w = img.w;
    let thresh = disk_threshold(img);
    let mut flux = vec![f64::NAN; w];
    for x in 0..w {
        let col = img.column(x);
        let sel: Vec<f32> = col.into_iter().filter(|&v| v > thresh).collect();
        if sel.len() > 30 {
            flux[x] = crate::mathutil::median_f32(&sel) as f64;
        }
    }
    flux
}

/// Estimate per-column transparency gains from a flux series (ideally
/// measured on a CONTINUUM extraction: transparency is common-mode across
/// wavelength, and the photospheric continuum is far smoother than the
/// chromosphere, so real solar structure cannot masquerade as transparency).
pub fn transparency_gains(flux_in: &[f64], deadband: f64) -> Vec<f64> {
    let w = flux_in.len();
    let flux = flux_in.to_vec();
    let valid: Vec<usize> = (0..w).filter(|&x| flux[x].is_finite()).collect();
    if valid.len() < 20 {
        return vec![1.0; w]; // nothing measurable (no disk?)
    }
    let mut filled = flux.clone();
    for x in 0..w {
        if !filled[x].is_finite() {
            let nearest = valid.iter().min_by_key(|&&v| v.abs_diff(x)).unwrap();
            filled[x] = flux[*nearest];
        }
    }

    // Trend must be slope- AND curvature-neutral on the curved chord-flux
    // profile, and robust so a passing cloud reads as outlier rather than
    // dragging the trend: robust local quadratic regression.
    let trend = crate::mathutil::robust_loess_quadratic(&filled, 121, 3);
    let tmax = trend.iter().cloned().fold(f64::MIN, f64::max).max(1e-9);
    // local flux level for the limb taper
    let local = robust_trend(&filled, 9, 3.0);
    let mut gain = vec![1.0f64; w];
    for x in 0..w {
        if !flux[x].is_finite() || trend[x] <= 1e-9 {
            continue;
        }
        let raw = (filled[x] / trend[x]).clamp(0.7, 1.06);
        // taper: full correction only where the chord flux is a healthy
        // fraction of the disk maximum (interior); zero at the x-limbs
        let tau = ((local[x] / tmax - 0.35) / 0.2).clamp(0.0, 1.0);
        // deadband: don't inject estimator noise for sub-percent deviations
        let dev = tau * (raw - 1.0);
        let dev = if dev.abs() < deadband { 0.0 } else { dev - deadband * dev.signum() };
        gain[x] = 1.0 + dev;
    }
    gain
}

/// Divide each column by its gain.
pub fn apply_column_gains(disk: &mut Image, gain: &[f64]) {
    for x in 0..disk.w.min(gain.len()) {
        if (gain[x] - 1.0).abs() > 1e-4 {
            let g = gain[x] as f32;
            for y in 0..disk.h {
                let v = disk.at(x, y);
                disk.set(x, y, v / g);
            }
        }
    }
}

/// Row-gain (transversalium) correction, two-scale robust version.
///
/// Rows tangent to the disk top/bottom are excluded (chord gate) and the
/// correction tapers to 1 at the ends of the measured range: the row-median
/// profile plunges there faster than any local trend can follow, and a
/// fabricated gain brightens the ENTIRE row — sky included — leaving a
/// bright line hugging the top/bottom limb (observed on real data).
pub fn correct_transversalium(disk: &mut Image, deadband: f64) -> Vec<f64> {
    let h = disk.h;
    let mut total_gain = vec![1.0f64; h];

    for win in [17usize, 61] {
        let thresh = disk_threshold(disk);
        // median over disk columns per row, plus per-row chord length
        let mut prof = vec![f64::NAN; h];
        let mut chord = vec![0usize; h];
        for y in 0..h {
            let sel: Vec<f32> = disk.row(y).iter().cloned().filter(|&v| v > thresh).collect();
            chord[y] = sel.len();
            if sel.len() > 40 {
                prof[y] = crate::mathutil::median_f32(&sel) as f64;
            }
        }
        let max_chord = *chord.iter().max().unwrap_or(&0);
        let min_chord = ((max_chord as f64) * 0.45) as usize;
        // trust only rows with a healthy chord (tangent rows are unmeasurable)
        for y in 0..h {
            if chord[y] < min_chord {
                prof[y] = f64::NAN;
            }
        }
        let valid: Vec<usize> = (0..h).filter(|&y| prof[y].is_finite()).collect();
        if valid.len() < 40 {
            return total_gain;
        }
        let (y_lo, y_hi) = (valid[0], *valid.last().unwrap());
        let mut filled = prof.clone();
        for y in 0..h {
            if !filled[y].is_finite() {
                let nearest = valid.iter().min_by_key(|&&v| v.abs_diff(y)).unwrap();
                filled[y] = prof[*nearest];
            }
        }
        let trend = crate::mathutil::robust_loess_quadratic(&filled, win | 1, 3);
        let taper_w = (win as f64).max(12.0);
        let mut gain = vec![1.0f64; h];
        for y in y_lo..=y_hi {
            if trend[y] > 1e-6 && prof[y].is_finite() {
                let raw = (filled[y] / trend[y]).clamp(0.8, 1.25);
                // taper to 1 at the ends of the measured range
                let edge = (((y - y_lo) as f64) / taper_w)
                    .min(((y_hi - y) as f64) / taper_w)
                    .clamp(0.0, 1.0);
                let dev = edge * (raw - 1.0);
                gain[y] = if dev.abs() < deadband { 1.0 } else { 1.0 + dev - deadband * dev.signum() };
            }
        }
        for y in 0..h {
            if (gain[y] - 1.0).abs() > 1e-4 {
                let g = gain[y] as f32;
                for v in disk.row_mut(y) {
                    *v /= g;
                }
                total_gain[y] *= gain[y];
            }
        }
    }
    total_gain
}
