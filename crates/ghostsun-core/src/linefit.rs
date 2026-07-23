//! Spectral line geometry: sub-pixel measurement of the absorption-line
//! center on the mean spectrum image, followed by a robust polynomial fit.
//!
//! INTI fits np.argmin (whole-pixel!) positions with sigma-clipped polyfit.
//! Here every row gets a Gauss-Newton inverted-Gaussian fit (sub-pixel, with
//! per-row confidence weights from line depth), and the smile polynomial is
//! fit with Tukey-biweight IRLS.

use crate::image2d::Image;
use crate::mathutil::{fit_inverted_gaussian, gaussian_smooth, polyfit_robust, polyval};

pub struct LineGeometry {
    /// polynomial coefficients: x(y) = c[0] + c[1] y + c[2] y^2 (raw y coords)
    pub coeffs: Vec<f64>,
    pub y1: usize,
    pub y2: usize,
    pub rms: f64,
    pub n_rows_used: usize,
}

/// Detect the vertical extent of the spectrum on the mean image
/// (rows where the slit actually saw light).
pub fn detect_spectrum_rows(mean_img: &Image) -> (usize, usize) {
    let h = mean_img.h;
    let mut prof: Vec<f64> = (0..h)
        .map(|y| mean_img.row(y).iter().map(|&v| v as f64).sum::<f64>() / mean_img.w as f64)
        .collect();
    prof = gaussian_smooth(&prof, 5.0);
    let pmax = prof.iter().cloned().fold(f64::MIN, f64::max);
    let pmin = prof.iter().cloned().fold(f64::MAX, f64::min);
    let thresh = pmin + 0.15 * (pmax - pmin);
    let mut y1 = 0;
    let mut y2 = h - 1;
    for (y, &v) in prof.iter().enumerate() {
        if v > thresh {
            y1 = y;
            break;
        }
    }
    for (y, &v) in prof.iter().enumerate().rev() {
        if v > thresh {
            y2 = y;
            break;
        }
    }
    (y1, y2)
}

/// Measure the line center on each row of the mean image and fit the smile
/// polynomial of the given degree.
pub fn fit_line_geometry(mean_img: &Image, deg: usize) -> Option<LineGeometry> {
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let margin = ((y2 - y1) / 20).clamp(5, 40);
    let (ya, yb) = (y1 + margin, y2.saturating_sub(margin));
    if yb <= ya + 10 {
        return None;
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ws = Vec::new();

    for y in ya..yb {
        let row: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
        // light smoothing for the coarse minimum only
        let sm = gaussian_smooth(&row, 1.5);
        // coarse min, excluding image borders
        let lo = 4usize;
        let hi = row.len() - 4;
        let (mut xmin, mut vmin) = (lo, f64::MAX);
        for x in lo..hi {
            if sm[x] < vmin {
                vmin = sm[x];
                xmin = x;
            }
        }
        // window around minimum for the Gaussian fit
        let w = 9usize;
        let a = xmin.saturating_sub(w).max(1);
        let b = (xmin + w + 1).min(row.len() - 1);
        let win_x: Vec<f64> = (a..b).map(|x| x as f64).collect();
        let win_y: Vec<f64> = row[a..b].to_vec();
        let cont = win_y.iter().cloned().fold(f64::MIN, f64::max);
        let depth = (cont - vmin) / cont.max(1e-9);
        if depth < 0.08 || cont < 1.0 {
            continue; // no usable line contrast on this row
        }
        if let Some((mu, sigma, amp, off)) = fit_inverted_gaussian(&win_x, &win_y, xmin as f64, 2.5) {
            if mu > a as f64 && mu < b as f64 && sigma < 20.0 {
                xs.push(y as f64);
                ys.push(mu);
                // weight by relative line depth (SNR proxy)
                ws.push((amp / off.max(1e-9)).clamp(0.0, 1.0));
            }
        }
    }

    if xs.len() < 30 {
        return None;
    }
    let coeffs = polyfit_robust(&xs, &ys, &ws, deg, 4)?;
    let mut ss = 0.0;
    let mut n = 0.0f64;
    for i in 0..xs.len() {
        let r = ys[i] - polyval(&coeffs, xs[i]);
        if r.abs() < 3.0 {
            ss += r * r;
            n += 1.0;
        }
    }
    Some(LineGeometry {
        coeffs,
        y1,
        y2,
        rms: (ss / n.max(1.0)).sqrt(),
        n_rows_used: xs.len(),
    })
}

/// Comparison-baseline variant: whole-pixel argmin + 2-pass sigma-clipped
/// unweighted quadratic fit (mirrors Inti_recon.py).
pub fn fit_line_geometry_baseline(mean_img: &Image) -> Option<LineGeometry> {
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let marge = 30usize;
    let (ya, yb) = (y1 + marge, y2.saturating_sub(marge));
    if yb <= ya + 10 {
        return None;
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for y in ya..yb {
        let row = mean_img.row(y);
        let mut xmin = 0usize;
        let mut vmin = f32::MAX;
        for (x, &v) in row.iter().enumerate() {
            if v < vmin {
                vmin = v;
                xmin = x;
            }
        }
        xs.push(y as f64);
        ys.push(xmin as f64); // integer position: this is the INTI flaw
    }
    let mut mask: Vec<bool> = vec![true; xs.len()];
    let mut coeffs = vec![0.0; 3];
    for _ in 0..2 {
        let fx: Vec<f64> = xs.iter().zip(&mask).filter(|(_, &m)| m).map(|(&x, _)| x).collect();
        let fy: Vec<f64> = ys.iter().zip(&mask).filter(|(_, &m)| m).map(|(&y, _)| y).collect();
        let fw = vec![1.0; fx.len()];
        coeffs = crate::mathutil::polyfit_weighted(&fx, &fy, &fw, 2)?;
        let res: Vec<f64> = xs.iter().zip(&ys).map(|(&x, &y)| y - polyval(&coeffs, x)).collect();
        let std = {
            let sel: Vec<f64> = res.iter().zip(&mask).filter(|(_, &m)| m).map(|(&r, _)| r).collect();
            let mean = sel.iter().sum::<f64>() / sel.len() as f64;
            (sel.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / sel.len() as f64).sqrt()
        };
        for i in 0..mask.len() {
            mask[i] = res[i].abs() < 6.0 * std;
        }
    }
    Some(LineGeometry { coeffs, y1, y2, rms: 0.0, n_rows_used: xs.len() })
}

/// Sub-pixel fit of the deepest absorption line in a single 1-D spectrum.
#[derive(Clone, Copy, Debug)]
pub struct LineFit1d {
    /// Sub-pixel column of the line core.
    pub center: f64,
    /// Gaussian sigma of the line, in pixels.
    pub sigma: f64,
    /// Full width at half maximum, in pixels (2.3548·sigma).
    pub fwhm: f64,
    /// Relative line depth (continuum − core)/continuum, in 0..1.
    pub depth: f64,
    /// Fitted continuum level (flux units).
    pub continuum: f64,
}

/// FWHM / sigma for a Gaussian: 2·sqrt(2·ln 2).
pub const FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949;

/// Find the deepest absorption line in a 1-D spectrum and fit it sub-pixel with
/// the same inverted-Gaussian estimator the pipeline uses per row. This is the
/// single source of truth for line width, shared by live focusing (minimise
/// `fwhm`) and the `spectrum` diagnostic — so the number at the telescope is the
/// number in the reports. `min_depth` gates out noise (relative, e.g. 0.03).
pub fn fit_line_1d(profile: &[f64], min_depth: f64) -> Option<LineFit1d> {
    let n = profile.len();
    if n < 9 {
        return None;
    }
    // Coarse minimum on a lightly smoothed copy (robust to per-sample noise);
    // the Gaussian fit itself runs on the raw samples for sub-pixel accuracy.
    let sm = gaussian_smooth(profile, 1.5);
    let mut sorted = profile.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cont0 = sorted[((0.90 * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    if cont0 <= 1e-9 {
        return None;
    }
    let margin = 3usize;
    let mut xmin = margin;
    let mut vmin = f64::MAX;
    for i in margin..n - margin {
        if sm[i] < vmin {
            vmin = sm[i];
            xmin = i;
        }
    }
    if (cont0 - profile[xmin]) / cont0 < min_depth {
        return None;
    }
    // Symmetric window around the core for the fit (up to ±10 px, edge-limited).
    let half = 10.min(xmin).min(n - 1 - xmin);
    if half < 3 {
        return None;
    }
    let (a, b) = (xmin - half, xmin + half);
    let win_x: Vec<f64> = (a..=b).map(|i| i as f64).collect();
    let win_y: Vec<f64> = (a..=b).map(|i| profile[i]).collect();
    let (mu, sigma, amp, off) = fit_inverted_gaussian(&win_x, &win_y, xmin as f64, 2.5)?;
    // Reject sub-resolution fits: any real line spans several pixels, so a fit
    // that pins near the fitter's minimum-sigma clamp is a single-pixel noise
    // spike, not a line. (Without this a noisy frame yields a spurious
    // FWHM ≈ 0.71 px = 2.3548·0.3 that poisons the min-hold.)
    if !(mu > a as f64 && mu < b as f64) || sigma < 0.7 || sigma > 40.0 || off <= 1e-9 {
        return None;
    }
    Some(LineFit1d {
        center: mu,
        sigma,
        fwhm: FWHM_PER_SIGMA * sigma,
        depth: (amp / off).clamp(0.0, 1.0),
        continuum: off,
    })
}

/// Detect and sub-pixel-fit *every* absorption line in a 1-D spectrum whose
/// relative depth exceeds `min_depth`, deepest-independent. Lets a caller pick
/// the narrowest line (best focus reference) or one nearest a chosen position,
/// instead of only the single deepest dip that [`fit_line_1d`] returns.
pub fn fit_lines_1d(profile: &[f64], min_depth: f64) -> Vec<LineFit1d> {
    let n = profile.len();
    if n < 9 {
        return Vec::new();
    }
    let sm = gaussian_smooth(profile, 1.5);
    let mut sorted = profile.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cont0 = sorted[((0.90 * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    if cont0 <= 1e-9 {
        return Vec::new();
    }
    const NEIGH: usize = 3;
    let margin = 3usize;
    let mut out: Vec<LineFit1d> = Vec::new();
    let mut i = margin;
    while i + margin < n {
        // Local minimum on the smoothed profile, deep enough to be a line.
        let is_min = (1..=NEIGH).all(|k| sm[i] <= sm[i - k])
            && (1..=NEIGH).all(|k| sm[i] <= sm[i + k]);
        if is_min && (cont0 - profile[i]) / cont0 >= min_depth {
            let half = 10.min(i).min(n - 1 - i);
            if half >= 3 {
                let (a, b) = (i - half, i + half);
                let win_x: Vec<f64> = (a..=b).map(|j| j as f64).collect();
                let win_y: Vec<f64> = (a..=b).map(|j| profile[j]).collect();
                if let Some((mu, sigma, amp, off)) =
                    fit_inverted_gaussian(&win_x, &win_y, i as f64, 2.5)
                {
                    if mu > a as f64 && mu < b as f64 && (0.7..=40.0).contains(&sigma) && off > 1e-9 {
                        out.push(LineFit1d {
                            center: mu,
                            sigma,
                            fwhm: FWHM_PER_SIGMA * sigma,
                            depth: (amp / off).clamp(0.0, 1.0),
                            continuum: off,
                        });
                    }
                }
            }
            i += NEIGH; // step past this minimum
        } else {
            i += 1;
        }
    }
    // Merge near-coincident detections (keep the deeper).
    out.sort_by(|a, b| a.center.partial_cmp(&b.center).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<LineFit1d> = Vec::new();
    for f in out {
        if let Some(last) = merged.last_mut() {
            if (f.center - last.center).abs() < 2.0 {
                if f.depth > last.depth {
                    *last = f;
                }
                continue;
            }
        }
        merged.push(f);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_multiple_lines_and_narrowest() {
        // Two lines: a deep broad one (σ4) and a shallow sharp one (σ1.5).
        let prof: Vec<f64> = (0..120)
            .map(|i| {
                let x = i as f64;
                let broad = 800.0 * (-(x - 35.0).powi(2) / (2.0 * 4.0 * 4.0)).exp();
                let sharp = 300.0 * (-(x - 85.0).powi(2) / (2.0 * 1.5 * 1.5)).exp();
                1000.0 - broad - sharp
            })
            .collect();
        let lines = fit_lines_1d(&prof, 0.03);
        assert!(lines.len() >= 2, "found {} lines", lines.len());
        let narrowest = lines.iter().min_by(|a, b| a.fwhm.partial_cmp(&b.fwhm).unwrap()).unwrap();
        assert!((narrowest.center - 85.0).abs() < 0.5, "narrowest at {}", narrowest.center);
        let deepest = lines.iter().max_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap()).unwrap();
        assert!((deepest.center - 35.0).abs() < 0.5, "deepest at {}", deepest.center);
    }

    #[test]
    fn recovers_known_line_width() {
        // Synthetic absorption line: continuum 1000, sigma 3.0 px, centre 40.7.
        let (cont, sigma, mu, amp) = (1000.0_f64, 3.0_f64, 40.7_f64, 700.0_f64);
        let prof: Vec<f64> = (0..80)
            .map(|i| {
                let dx = i as f64 - mu;
                cont - amp * (-dx * dx / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let f = fit_line_1d(&prof, 0.03).expect("line found");
        assert!((f.center - mu).abs() < 0.05, "center {} vs {mu}", f.center);
        assert!((f.sigma - sigma).abs() < 0.05, "sigma {} vs {sigma}", f.sigma);
        assert!((f.fwhm - FWHM_PER_SIGMA * sigma).abs() < 0.15);
    }
}
