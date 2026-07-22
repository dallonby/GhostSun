//! Numerical primitives: robust polynomial fitting, Gaussian line fitting,
//! cubic B-spline interpolation with exact prefiltering, Lanczos kernels,
//! robust statistics and smoothing.

use nalgebra::{DMatrix, DVector};

/// Weighted least-squares polynomial fit. Returns coefficients c[0..=deg]
/// such that p(x) = sum c[k] * x^k.
pub fn polyfit_weighted(xs: &[f64], ys: &[f64], ws: &[f64], deg: usize) -> Option<Vec<f64>> {
    let n = xs.len();
    if n < deg + 1 {
        return None;
    }
    // Normalize x to [-1,1] for conditioning.
    let xmin = xs.iter().cloned().fold(f64::MAX, f64::min);
    let xmax = xs.iter().cloned().fold(f64::MIN, f64::max);
    let span = (xmax - xmin).max(1e-12);
    let norm = |x: f64| 2.0 * (x - xmin) / span - 1.0;

    let mut a = DMatrix::<f64>::zeros(n, deg + 1);
    let mut b = DVector::<f64>::zeros(n);
    for i in 0..n {
        let sw = ws[i].max(0.0).sqrt();
        let xn = norm(xs[i]);
        let mut p = 1.0;
        for k in 0..=deg {
            a[(i, k)] = p * sw;
            p *= xn;
        }
        b[i] = ys[i] * sw;
    }
    let svd = a.svd(true, true);
    let cn = svd.solve(&b, 1e-12).ok()?;

    // Convert coefficients from normalized to raw x by expanding
    // p(norm(x)) where norm(x) = alpha*x + beta.
    let alpha = 2.0 / span;
    let beta = -2.0 * xmin / span - 1.0;
    let mut coeffs = vec![0.0; deg + 1];
    // (alpha x + beta)^k expansion via iterative polynomial multiply
    let mut basis = vec![1.0]; // polynomial "1" in raw x
    for k in 0..=deg {
        for (j, &bj) in basis.iter().enumerate() {
            coeffs[j] += cn[k] * bj;
        }
        // basis *= (alpha x + beta)
        let mut next = vec![0.0; basis.len() + 1];
        for (j, &bj) in basis.iter().enumerate() {
            next[j] += bj * beta;
            next[j + 1] += bj * alpha;
        }
        basis = next;
    }
    Some(coeffs)
}

pub fn polyval(c: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &ck in c.iter().rev() {
        acc = acc * x + ck;
    }
    acc
}

/// Robust polynomial fit with Tukey biweight IRLS.
pub fn polyfit_robust(xs: &[f64], ys: &[f64], w0: &[f64], deg: usize, iters: usize) -> Option<Vec<f64>> {
    let mut w: Vec<f64> = w0.to_vec();
    let mut coeffs = polyfit_weighted(xs, ys, &w, deg)?;
    for _ in 0..iters {
        let res: Vec<f64> = xs.iter().zip(ys).map(|(&x, &y)| y - polyval(&coeffs, x)).collect();
        let mut abs_res: Vec<f64> = res.iter().map(|r| r.abs()).collect();
        let mad = median_inplace(&mut abs_res).max(1e-9);
        let c = 4.685 * 1.4826 * mad;
        for i in 0..w.len() {
            let u = res[i] / c;
            let tw = if u.abs() < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
            w[i] = w0[i] * tw;
        }
        coeffs = polyfit_weighted(xs, ys, &w, deg)?;
    }
    Some(coeffs)
}

pub fn median_inplace(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mid = v.len() / 2;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        0.5 * (v[mid - 1] + v[mid])
    }
}

pub fn median_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut tmp = v.to_vec();
    let mid = tmp.len() / 2;
    tmp.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if tmp.len() % 2 == 1 {
        tmp[mid]
    } else {
        // The lower partition is unordered, so find its maximum rather than
        // sorting the entire sample just to obtain the second middle value.
        let lower = tmp[..mid]
            .iter()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        0.5 * (lower + tmp[mid])
    }
}

pub fn percentile_f32(v: &[f32], p: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut tmp: Vec<f32> = v.to_vec();
    let idx = ((p / 100.0) * (tmp.len() - 1) as f64).round() as usize;
    let idx = idx.min(tmp.len() - 1);
    tmp.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    tmp[idx]
}

/// Error function (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7).
#[allow(dead_code)]
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Fit inverted Gaussian y = offset - amp * exp(-(x-mu)^2/(2 sigma^2))
/// by Gauss-Newton. Returns (mu, sigma, amp, offset) or None.
pub fn fit_inverted_gaussian(
    xs: &[f64],
    ys: &[f64],
    mu0: f64,
    sigma0: f64,
) -> Option<(f64, f64, f64, f64)> {
    let n = xs.len();
    if n < 5 {
        return None;
    }
    let ymax = ys.iter().cloned().fold(f64::MIN, f64::max);
    let ymin = ys.iter().cloned().fold(f64::MAX, f64::min);
    let mut amp = (ymax - ymin).max(1e-6);
    let mut off = ymax;
    let mut mu = mu0;
    let mut sig = sigma0.max(0.5);

    for _ in 0..25 {
        let mut jtj = nalgebra::Matrix4::<f64>::zeros();
        let mut jtr = nalgebra::Vector4::<f64>::zeros();
        for i in 0..n {
            let dx = xs[i] - mu;
            let e = (-dx * dx / (2.0 * sig * sig)).exp();
            let model = off - amp * e;
            let r = ys[i] - model;
            // partials of model
            let d_amp = -e;
            let d_off = 1.0;
            let d_mu = -amp * e * dx / (sig * sig);
            let d_sig = -amp * e * dx * dx / (sig * sig * sig);
            let j = nalgebra::Vector4::new(d_mu, d_sig, d_amp, d_off);
            jtj += j * j.transpose();
            jtr += j * r;
        }
        // Levenberg damping for stability
        for k in 0..4 {
            jtj[(k, k)] *= 1.0 + 1e-3;
            jtj[(k, k)] += 1e-9;
        }
        let delta = jtj.lu().solve(&jtr)?;
        mu += delta[0].clamp(-2.0, 2.0);
        sig = (sig + delta[1].clamp(-2.0, 2.0)).clamp(0.3, 50.0);
        amp += delta[2];
        off += delta[3];
        if delta[0].abs() < 1e-5 {
            break;
        }
    }
    if !mu.is_finite() || amp <= 0.0 {
        return None;
    }
    Some((mu, sig, amp, off))
}

// ---------------------------------------------------------------------------
// Cubic B-spline interpolation with exact prefiltering (Unser 1999).
// Far better frequency response than Catmull-Rom, exact at samples.
// ---------------------------------------------------------------------------

const BSPLINE_POLE: f64 = -0.267949192431123; // sqrt(3) - 2

/// In-place conversion of samples to B-spline coefficients (1D).
pub fn bspline_prefilter(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = BSPLINE_POLE;
    let lambda = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= lambda;
    }
    // causal init (mirror boundary, truncated sum)
    let horizon = n.min(30);
    let mut sum = c[0];
    let mut zn = z;
    for i in 1..horizon {
        sum += zn * c[i];
        zn *= z;
    }
    c[0] = sum;
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    // anticausal init
    c[n - 1] = (z / (z * z - 1.0)) * (z * c[n - 2] + c[n - 1]);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// Evaluate cubic B-spline at fractional position x given coefficient array.
#[inline]
pub fn bspline_eval(c: &[f64], x: f64) -> f64 {
    let n = c.len() as isize;
    let xf = x.floor();
    let t = x - xf;
    let i = xf as isize;
    // basis weights
    let t2 = t * t;
    let t3 = t2 * t;
    let w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    let w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    let w2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    let w3 = t3 / 6.0;
    let get = |j: isize| -> f64 {
        // mirror boundary
        let mut k = j;
        if k < 0 {
            k = -k;
        }
        if k >= n {
            k = 2 * (n - 1) - k;
        }
        c[k.clamp(0, n - 1) as usize]
    };
    w0 * get(i - 1) + w1 * get(i) + w2 * get(i + 1) + w3 * get(i + 2)
}

// ---------------------------------------------------------------------------
// Lanczos-3 kernel
// ---------------------------------------------------------------------------

#[inline]
pub fn lanczos3(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-9 {
        return 1.0;
    }
    if ax >= 3.0 {
        return 0.0;
    }
    let pix = std::f64::consts::PI * x;
    3.0 * (pix.sin() * (pix / 3.0).sin()) / (pix * pix)
}

/// 1D Gaussian smoothing (reflected boundary).
pub fn gaussian_smooth(v: &[f64], sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return v.to_vec();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut kernel = Vec::with_capacity((2 * r + 1) as usize);
    let mut ksum = 0.0;
    for i in -r..=r {
        let k = (-(i as f64).powi(2) / (2.0 * sigma * sigma)).exp();
        kernel.push(k);
        ksum += k;
    }
    for k in kernel.iter_mut() {
        *k /= ksum;
    }
    let n = v.len() as isize;
    let mut out = vec![0.0; v.len()];
    for i in 0..n {
        let mut acc = 0.0;
        for (ki, j) in (-r..=r).enumerate() {
            let mut idx = i + j;
            if idx < 0 {
                idx = -idx;
            }
            if idx >= n {
                idx = 2 * (n - 1) - idx;
            }
            acc += kernel[ki] * v[idx.clamp(0, n - 1) as usize];
        }
        out[i as usize] = acc;
    }
    out
}

/// Robust local trend of a 1D signal: running median followed by Gaussian
/// smoothing. Window sizes in samples.
pub fn robust_trend(v: &[f64], med_win: usize, sigma: f64) -> Vec<f64> {
    let n = v.len();
    let hw = (med_win / 2).max(1);
    let mut med = vec![0.0; n];
    for i in 0..n {
        let lo = i.saturating_sub(hw);
        let hi = (i + hw + 1).min(n);
        let mut win: Vec<f64> = v[lo..hi].to_vec();
        med[i] = median_inplace(&mut win);
    }
    gaussian_smooth(&med, sigma)
}

/// Robust local quadratic regression (LOESS with Tukey IRLS): slope- and
/// curvature-neutral trend estimation that rejects outliers (dust lines,
/// clouds) instead of being dragged by them. This is the correct trend for
/// ratio-based gain estimation — rolling medians/quantiles are biased on
/// curved profiles and fabricate gains.
pub fn robust_loess_quadratic(v: &[f64], win: usize, iters: usize) -> Vec<f64> {
    let n = v.len();
    let hw = (win / 2).max(3);
    let mut out = vec![0.0; n];
    for i in 0..n {
        let lo = i.saturating_sub(hw);
        let hi = (i + hw + 1).min(n);
        let xs: Vec<f64> = (lo..hi).map(|j| j as f64 - i as f64).collect();
        let ys: Vec<f64> = v[lo..hi].to_vec();
        // tricube distance weights
        let ws: Vec<f64> = xs
            .iter()
            .map(|&x| {
                let u = (x / (hw as f64 + 1.0)).abs();
                (1.0 - u * u * u).powi(3).max(1e-6)
            })
            .collect();
        out[i] = match polyfit_robust(&xs, &ys, &ws, 2, iters) {
            Some(c) => c[0],
            None => v[i],
        };
    }
    out
}

/// Savitzky-Golay-like quadratic smoothing via least squares on a moving
/// window (used only by the INTI-baseline mode).
pub fn savgol_quadratic(v: &[f64], window: usize) -> Vec<f64> {
    let n = v.len();
    let hw = (window / 2).max(2);
    let mut out = vec![0.0; n];
    for i in 0..n {
        let lo = i.saturating_sub(hw);
        let hi = (i + hw + 1).min(n);
        let xs: Vec<f64> = (lo..hi).map(|j| j as f64 - i as f64).collect();
        let ys: Vec<f64> = v[lo..hi].to_vec();
        let ws = vec![1.0; xs.len()];
        if let Some(c) = polyfit_weighted(&xs, &ys, &ws, 2) {
            out[i] = c[0];
        } else {
            out[i] = v[i];
        }
    }
    out
}

/// Separable 2-D Gaussian blur on an Image (used by metrics and deconvolution).
pub fn gaussian_blur_2d(img: &crate::image2d::Image, sigma_x: f64, sigma_y: f64) -> crate::image2d::Image {
    let mut out = img.clone();
    if sigma_x > 0.0 {
        for y in 0..img.h {
            let row: Vec<f64> = out.row(y).iter().map(|&v| v as f64).collect();
            let sm = gaussian_smooth(&row, sigma_x);
            for (x, v) in sm.iter().enumerate() {
                out.set(x, y, *v as f32);
            }
        }
    }
    if sigma_y > 0.0 {
        for x in 0..img.w {
            let col: Vec<f64> = (0..img.h).map(|y| out.at(x, y) as f64).collect();
            let sm = gaussian_smooth(&col, sigma_y);
            for (y, v) in sm.iter().enumerate() {
                out.set(x, y, *v as f32);
            }
        }
    }
    out
}

/// Top-k principal components of a set of vectors (rows), via power
/// iteration with deflation. Returns (components, mean).
pub fn pca_topk(samples: &[Vec<f64>], k: usize, iters: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = samples.len();
    if n == 0 || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let d = samples[0].len();
    let mut mean = vec![0.0; d];
    for s in samples {
        for i in 0..d {
            mean[i] += s[i];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    // covariance (d x d, d is small: ~20)
    let mut cov = vec![vec![0.0; d]; d];
    for s in samples {
        for i in 0..d {
            let vi = s[i] - mean[i];
            for j in i..d {
                cov[i][j] += vi * (s[j] - mean[j]);
            }
        }
    }
    for i in 0..d {
        for j in 0..i {
            cov[i][j] = cov[j][i];
        }
    }
    let matvec = |m: &Vec<Vec<f64>>, v: &[f64]| -> Vec<f64> {
        (0..d).map(|i| (0..d).map(|j| m[i][j] * v[j]).sum()).collect()
    };
    let mut comps: Vec<Vec<f64>> = Vec::new();
    let mut work = cov.clone();
    for c in 0..k.min(d) {
        let mut v: Vec<f64> = (0..d).map(|i| if (i + c) % 3 == 0 { 1.0 } else { 0.5 }).collect();
        let mut lambda = 0.0;
        for _ in 0..iters {
            let mv = matvec(&work, &v);
            let norm = mv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
            v = mv.iter().map(|x| x / norm).collect();
            lambda = norm;
        }
        // deflate
        for i in 0..d {
            for j in 0..d {
                work[i][j] -= lambda * v[i] * v[j];
            }
        }
        comps.push(v);
    }
    (comps, mean)
}
