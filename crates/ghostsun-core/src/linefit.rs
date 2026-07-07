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

/// INTI-baseline variant: whole-pixel argmin + 2-pass sigma-clipped
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
