//! F7: variance-stabilized wavelet denoising.
//!
//! Noise model x = alpha * Poisson(lambda) + N(0, sigma_r): gain and read
//! noise are estimated from the image itself by regressing local variance
//! against local mean (photon-transfer). The generalized Anscombe transform
//! makes the noise ~ unit Gaussian; an undecimated B3-spline a-trous wavelet
//! transform with soft thresholding removes it; the algebraic inverse GAT
//! maps back. Thresholds use the known per-level noise gains of the 2-D B3
//! a-trous transform.

use crate::image2d::Image;
use crate::metrics::DiskFit;

/// Per-level noise std of unit Gaussian noise in 2-D B3 a-trous detail planes.
const LEVEL_SIGMA: [f64; 4] = [0.889, 0.200, 0.086, 0.041];

/// Estimate (alpha, sigma_r^2) by photon transfer: local variance vs mean on
/// small blocks, robust line fit. Variance is measured on the finest wavelet
/// plane (real structure lives at coarser scales) with the appropriate gain.
fn estimate_noise(img: &Image, disk: &DiskFit) -> (f64, f64) {
    let fine = {
        let sm = smooth_b3(img, 1);
        let mut f = img.clone();
        for i in 0..f.data.len() {
            f.data[i] -= sm.data[i];
        }
        f
    };
    let bs = 16usize;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut by = 0;
    while by + bs <= img.h {
        let mut bx = 0;
        while bx + bs <= img.w {
            let cx = bx as f64 + bs as f64 / 2.0 - disk.xc;
            let cy = by as f64 + bs as f64 / 2.0 - disk.yc;
            let r = (cx * cx + cy * cy).sqrt() / disk.r;
            if r < 1.3 {
                let mut m = 0.0f64;
                let mut v = 0.0f64;
                for y in by..by + bs {
                    for x in bx..bx + bs {
                        m += img.at(x, y) as f64;
                        v += (fine.at(x, y) as f64).powi(2);
                    }
                }
                let n = (bs * bs) as f64;
                m /= n;
                // finest-plane variance of unit noise is LEVEL_SIGMA[0]^2
                v = v / n / (LEVEL_SIGMA[0] * LEVEL_SIGMA[0]);
                xs.push(m);
                ys.push(v);
            }
            bx += bs;
        }
        by += bs;
    }
    if xs.len() < 30 {
        return (1.0, 0.0);
    }
    let ws = vec![1.0; xs.len()];
    match crate::mathutil::polyfit_robust(&xs, &ys, &ws, 1, 4) {
        Some(c) => (c[1].max(1e-6), c[0].max(0.0)),
        None => (1.0, 0.0),
    }
}

/// One a-trous B3 smoothing pass at scale 2^(level) (holes).
fn smooth_b3(img: &Image, step: usize) -> Image {
    const K: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let (w, h) = (img.w, img.h);
    let mut tmp = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in K.iter().enumerate() {
                let xx = x as isize + (k as isize - 2) * step as isize;
                acc += kv * img.at_clamped(xx, y as isize) as f64;
            }
            tmp.set(x, y, acc as f32);
        }
    }
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in K.iter().enumerate() {
                let yy = y as isize + (k as isize - 2) * step as isize;
                acc += kv * tmp.at_clamped(x as isize, yy) as f64;
            }
            out.set(x, y, acc as f32);
        }
    }
    out
}

/// Denoise in place. `k` is the soft-threshold multiple (default 1.0).
pub fn denoise(img: &Image, disk: &DiskFit, k: f64) -> Image {
    let (alpha, sr2) = estimate_noise(img, disk);

    // generalized Anscombe forward
    let mut z = img.clone();
    for v in z.data.iter_mut() {
        let x = *v as f64;
        let arg = alpha * x + 0.375 * alpha * alpha + sr2;
        *v = ((2.0 / alpha) * arg.max(0.0).sqrt()) as f32;
    }

    // a-trous decomposition with soft-thresholded details
    let mut c = z.clone();
    let mut recon = Image::new(img.w, img.h); // sum of kept details
    for (level, &ls) in LEVEL_SIGMA.iter().enumerate() {
        let next = smooth_b3(&c, 1 << level);
        let thr = (k * ls) as f32;
        for i in 0..recon.data.len() {
            let d = c.data[i] - next.data[i];
            let kept = if d.abs() <= thr { 0.0 } else { d - thr * d.signum() };
            recon.data[i] += kept;
        }
        c = next;
    }
    for i in 0..recon.data.len() {
        recon.data[i] += c.data[i]; // + smooth residual
    }

    // algebraic inverse GAT
    for v in recon.data.iter_mut() {
        let zz = *v as f64;
        let x = ((zz * alpha / 2.0).powi(2) - 0.375 * alpha * alpha - sr2) / alpha;
        *v = x.max(0.0) as f32;
    }
    recon
}
