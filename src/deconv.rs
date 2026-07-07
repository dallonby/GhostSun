//! F6: PSF estimation from the limb + regularized Richardson-Lucy
//! deconvolution.
//!
//! The true limb is a near-step edge convolved with (seeing x slit x optics),
//! so the radial derivative profile of the limb IS the radial cut of the
//! PSF. We measure the erf-transition width in position-angle bins, fit the
//! anisotropic Gaussian model sigma(theta)^2 = sx^2 cos^2 + sy^2 sin^2, and
//! subtract a fixed intrinsic/sampling floor in quadrature. RL runs with a
//! separable Gaussian kernel and optional total-variation damping, on the
//! background-subtracted f32 image.

use crate::image2d::Image;
use crate::metrics::DiskFit;
use crate::mathutil::gaussian_blur_2d;

/// Below this measured width above the floor, deconvolution auto-skips.
const SIGMA_SKIP_MARGIN: f64 = 0.15;

fn bilinear(img: &Image, x: f64, y: f64) -> f64 {
    let xf = x.floor();
    let yf = y.floor();
    let tx = x - xf;
    let ty = y - yf;
    let xi = xf as isize;
    let yi = yf as isize;
    let v00 = img.at_clamped(xi, yi) as f64;
    let v10 = img.at_clamped(xi + 1, yi) as f64;
    let v01 = img.at_clamped(xi, yi + 1) as f64;
    let v11 = img.at_clamped(xi + 1, yi + 1) as f64;
    v00 * (1.0 - tx) * (1.0 - ty) + v10 * tx * (1.0 - ty) + v01 * (1.0 - tx) * ty + v11 * tx * ty
}

/// Edge-transition sigma in one position-angle bin (second moment of the
/// radial derivative, tail-baseline removed). NaN if unusable.
fn edge_sigma_bin(img: &Image, disk: &DiskFit, pa_lo: f64, pa_hi: f64) -> f64 {
    let n_r = 80usize; // 0.25 px steps over +-10 px
    let r_lo = disk.r - 10.0;
    let mut prof = vec![0.0f64; n_r];
    let mut cnt = vec![0.0f64; n_r];
    let n_a = 40;
    for a in 0..n_a {
        let th = pa_lo + (pa_hi - pa_lo) * (a as f64 + 0.5) / n_a as f64;
        let (ct, st) = (th.cos(), th.sin());
        for i in 0..n_r {
            let r = r_lo + i as f64 * 0.25;
            let x = disk.xc + r * ct;
            let y = disk.yc + r * st;
            if x < 1.0 || y < 1.0 || x > img.w as f64 - 2.0 || y > img.h as f64 - 2.0 {
                continue;
            }
            prof[i] += bilinear(img, x, y);
            cnt[i] += 1.0;
        }
    }
    for i in 0..n_r {
        if cnt[i] > 0.0 {
            prof[i] /= cnt[i];
        }
    }
    let mut d: Vec<f64> = (1..n_r - 1).map(|i| (prof[i - 1] - prof[i + 1]).max(0.0)).collect();
    let base = {
        let mut tails: Vec<f64> = d[..8].to_vec();
        tails.extend_from_slice(&d[d.len() - 8..]);
        crate::mathutil::median_inplace(&mut tails)
    };
    for v in d.iter_mut() {
        *v = (*v - base).max(0.0);
    }
    let sw: f64 = d.iter().sum();
    if sw <= 1e-9 {
        return f64::NAN;
    }
    let mu: f64 = d.iter().enumerate().map(|(i, &v)| v * i as f64).sum::<f64>() / sw;
    let var: f64 = d.iter().enumerate().map(|(i, &v)| v * (i as f64 - mu).powi(2)).sum::<f64>() / sw;
    var.sqrt() * 0.25
}

/// Estimate the anisotropic PSF (sigma_x, sigma_y) from the limb.
pub fn estimate_psf(img: &Image, disk: &DiskFit) -> Option<(f64, f64)> {
    let bins = 8;
    let mut xs = Vec::new(); // cos^2 theta
    let mut ys = Vec::new(); // sigma^2
    for b in 0..bins {
        let lo = b as f64 / bins as f64 * std::f64::consts::TAU;
        let hi = (b + 1) as f64 / bins as f64 * std::f64::consts::TAU;
        let s = edge_sigma_bin(img, disk, lo, hi);
        if s.is_finite() && s > 0.2 && s < 10.0 {
            let th = (lo + hi) / 2.0;
            xs.push(th.cos().powi(2));
            ys.push(s * s);
        }
    }
    if xs.len() < 5 {
        return None;
    }
    // sigma^2(theta) = sy^2 + (sx^2 - sy^2) cos^2(theta): linear LS
    let ws = vec![1.0; xs.len()];
    let c = crate::mathutil::polyfit_weighted(&xs, &ys, &ws, 1)?;
    let sy2 = c[0].max(0.04);
    let sx2 = (c[0] + c[1]).max(0.04);
    Some((sx2.sqrt(), sy2.sqrt()))
}

/// Richardson-Lucy with separable anisotropic Gaussian kernel and optional
/// TV damping. `sigma` is the *measured* limb width; the intrinsic floor is
/// removed in quadrature. Returns None (skip) if the PSF is too small.
pub fn deconvolve(
    img: &Image,
    disk: &DiskFit,
    iters: usize,
    tv_lambda: f64,
    sigma_floor: f64,
) -> Option<(Image, (f64, f64))> {
    let (mx, my) = estimate_psf(img, disk)?;
    if mx.max(my) < sigma_floor + SIGMA_SKIP_MARGIN {
        return None;
    }
    let kx = (mx * mx - sigma_floor * sigma_floor).max(0.0).sqrt();
    let ky = (my * my - sigma_floor * sigma_floor).max(0.0).sqrt();
    if kx.max(ky) < 0.4 {
        return None;
    }

    // background pedestal from off-disk pixels
    let mut off: Vec<f32> = Vec::new();
    for y in (0..img.h).step_by(3) {
        for x in (0..img.w).step_by(3) {
            let dx = x as f64 - disk.xc;
            let dy = y as f64 - disk.yc;
            if (dx * dx + dy * dy).sqrt() > disk.r * 1.15 {
                off.push(img.at(x, y));
            }
        }
    }
    let bg = if off.is_empty() { 0.0 } else { crate::mathutil::percentile_f32(&off, 20.0) } as f64;

    let mut d = img.clone(); // data, background-subtracted
    for v in d.data.iter_mut() {
        *v = (*v - bg as f32).max(0.0);
    }
    let mut est = d.clone();
    let eps = 1e-3f32;
    for _ in 0..iters {
        let blurred = gaussian_blur_2d(&est, kx, ky);
        let mut ratio = Image::new(img.w, img.h);
        for i in 0..ratio.data.len() {
            ratio.data[i] = d.data[i] / blurred.data[i].max(eps);
        }
        let corr = gaussian_blur_2d(&ratio, kx, ky);
        if tv_lambda > 0.0 {
            // RL-TV multiplicative update (Dey et al. 2006)
            let div = tv_divergence(&est);
            for i in 0..est.data.len() {
                let denom = (1.0 - tv_lambda as f32 * div.data[i]).max(0.5);
                est.data[i] = (est.data[i] * corr.data[i] / denom).max(0.0);
            }
        } else {
            for i in 0..est.data.len() {
                est.data[i] = (est.data[i] * corr.data[i]).max(0.0);
            }
        }
    }
    for v in est.data.iter_mut() {
        *v += bg as f32;
    }
    Some((est, (mx, my)))
}

/// div( grad I / |grad I| ) — discrete TV curvature term.
fn tv_divergence(img: &Image) -> Image {
    let (w, h) = (img.w, img.h);
    let eps = 1e-3f64;
    // normalized gradients
    let mut px = vec![0.0f32; w * h];
    let mut py = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let gx = img.at_clamped(x as isize + 1, y as isize) as f64 - img.at(x, y) as f64;
            let gy = img.at_clamped(x as isize, y as isize + 1) as f64 - img.at(x, y) as f64;
            let n = (gx * gx + gy * gy).sqrt().max(eps);
            px[y * w + x] = (gx / n) as f32;
            py[y * w + x] = (gy / n) as f32;
        }
    }
    let mut div = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let dxx = px[y * w + x] - if x > 0 { px[y * w + x - 1] } else { 0.0 };
            let dyy = py[y * w + x] - if y > 0 { py[(y - 1) * w + x] } else { 0.0 };
            div.set(x, y, dxx + dyy);
        }
    }
    div
}
