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

/// What the Wiener stage measured and did.
#[derive(Clone, Copy, Debug)]
pub struct WienerReport {
    /// Estimated white-noise power per frequency bin.
    pub noise_floor: f64,
    /// Scale (px) at which signal power falls to the noise floor — the point
    /// beyond which the image carries nothing but noise.
    pub cutoff_px: f64,
    /// Fraction of total power the filter removed.
    pub removed: f64,
}

/// Optimal (Wiener) filtering against the image's own measured power spectrum.
///
/// A power-spectrum comparison of one scan against a nine-scan stack showed
/// real solar power dying out below roughly 20 px while the image is sampled
/// at 1 px — the data is oversampled by more than an order of magnitude, and
/// everything below the crossover is noise occupying dynamic range. The
/// minimum-mean-square estimator for that situation is `H = S/(S+N)`: it
/// passes frequencies where signal dominates untouched and suppresses those
/// where noise does, which is emphatically NOT the same as blurring, because
/// it removes only what carries no information.
///
/// `S` and `N` are measured from this image: the radially averaged spectrum is
/// its own design input. `N` is taken as the high-frequency plateau, which
/// assumes the noise is roughly white — a caveat worth remembering, since an
/// upstream denoiser will already have coloured it, and a coloured floor makes
/// this estimate conservative rather than wrong.
///
/// `strength` scales the assumed noise: 1.0 is the classic Wiener filter,
/// below 1 is gentler, above 1 more aggressive.
pub fn wiener_psd(
    img: &Image,
    strength: f64,
    disk: Option<&DiskFit>,
) -> Option<(Image, WienerReport)> {
    let (w, h) = (img.w, img.h);
    if w < 64 || h < 64 {
        return None;
    }
    let n = w.max(h).next_power_of_two();
    let mut re = vec![0.0f64; n * n];
    let mut im = vec![0.0f64; n * n];
    // The disc sits centred with black margin, so the array edges are already
    // continuous across the wrap and need no window; padding with zeros keeps
    // it that way.
    let (ox, oy) = ((n - w) / 2, (n - h) / 2);
    for y in 0..h {
        for x in 0..w {
            re[(y + oy) * n + (x + ox)] = img.at(x, y) as f64;
        }
    }
    crate::mathutil::fft2_inplace(&mut re, &mut im, n, n, false);

    // Radially averaged power spectrum.
    let nr = n / 2;
    let mut psum = vec![0.0f64; nr + 1];
    let mut pcnt = vec![0.0f64; nr + 1];
    for j in 0..n {
        let fy = if j <= n / 2 { j } else { n - j } as f64;
        for i in 0..n {
            let fx = if i <= n / 2 { i } else { n - i } as f64;
            let k = (fx * fx + fy * fy).sqrt().round() as usize;
            if k > nr {
                continue;
            }
            let idx = j * n + i;
            psum[k] += re[idx] * re[idx] + im[idx] * im[idx];
            pcnt[k] += 1.0;
        }
    }
    let mut p: Vec<f64> = (0..=nr)
        .map(|k| if pcnt[k] > 0.0 { psum[k] / pcnt[k] } else { 0.0 })
        .collect();
    // Smooth the radial profile so the filter is not shaped by bin noise.
    p = crate::mathutil::gaussian_smooth(&p, 2.0);

    // Noise floor: the median of the top quarter of frequencies, where the
    // measurement says no solar signal survives.
    let lo = (nr as f64 * 0.75) as usize;
    let mut tail: Vec<f64> = p[lo..=nr].to_vec();
    if tail.is_empty() {
        return None;
    }
    let noise = crate::mathutil::median_inplace(&mut tail).max(1e-30) * strength;

    // Signal PSD and the transfer function.
    let hfun: Vec<f64> = p
        .iter()
        .map(|&pk| {
            let s = (pk - noise).max(0.0);
            if s + noise > 0.0 { s / (s + noise) } else { 0.0 }
        })
        .collect();
    // Where the filter has fallen to half: the practical resolution limit.
    let cutoff_k = hfun.iter().position(|&v| v < 0.5).unwrap_or(nr).max(1);
    let cutoff_px = n as f64 / cutoff_k as f64;

    let (mut kept, mut total) = (0.0f64, 0.0f64);
    for j in 0..n {
        let fy = if j <= n / 2 { j } else { n - j } as f64;
        for i in 0..n {
            let fx = if i <= n / 2 { i } else { n - i } as f64;
            let k = ((fx * fx + fy * fy).sqrt().round() as usize).min(nr);
            let g = hfun[k];
            let idx = j * n + i;
            let pw = re[idx] * re[idx] + im[idx] * im[idx];
            total += pw;
            kept += pw * g * g;
            re[idx] *= g;
            im[idx] *= g;
        }
    }
    crate::mathutil::fft2_inplace(&mut re, &mut im, n, n, true);
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.set(x, y, re[(y + oy) * n + (x + ox)].max(0.0) as f32);
        }
    }
    // The filter assumes one stationary spectrum, and the limb is where that
    // assumption fails hardest: it is a genuine sharp edge with real power at
    // the frequencies the disc's average spectrum says are noise. Filtering it
    // with the disc's transfer function measurably softens it (PSNRlimb fell
    // 27.2 -> 24.5 dB and limb sigma rose 1.26 -> 1.98 across a strength
    // sweep). So the filter is feathered out across the limb annulus and the
    // original kept beyond it.
    if let Some(d) = disk {
        for y in 0..h {
            let dy = y as f64 - d.yc;
            for x in 0..w {
                let dx = x as f64 - d.xc;
                let r = (dx * dx + dy * dy).sqrt() / d.r.max(1e-9);
                // full strength inside 0.90 R, none beyond 1.00 R
                let t = ((1.00 - r) / 0.10).clamp(0.0, 1.0);
                if t < 1.0 {
                    let a = out.at(x, y) as f64;
                    let b = img.at(x, y) as f64;
                    out.set(x, y, (t * a + (1.0 - t) * b) as f32);
                }
            }
        }
    }
    Some((
        out,
        WienerReport {
            noise_floor: noise,
            cutoff_px,
            removed: if total > 0.0 { 1.0 - kept / total } else { 0.0 },
        },
    ))
}

#[cfg(test)]
mod wiener_tests {
    use super::*;

    #[test]
    fn fft_round_trips_and_matches_a_known_transform() {
        // A single cosine must transform to two symmetric spikes.
        let n = 64usize;
        let mut re: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::TAU * 4.0 * i as f64 / n as f64).cos())
            .collect();
        let orig = re.clone();
        let mut im = vec![0.0f64; n];
        crate::mathutil::fft_inplace(&mut re, &mut im, false);
        let mag: Vec<f64> = (0..n).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()).collect();
        assert!((mag[4] - n as f64 / 2.0).abs() < 1e-6, "spike at k=4: {}", mag[4]);
        assert!((mag[n - 4] - n as f64 / 2.0).abs() < 1e-6);
        for (k, m) in mag.iter().enumerate() {
            if k != 4 && k != n - 4 {
                assert!(*m < 1e-6, "leakage at {k}: {m}");
            }
        }
        crate::mathutil::fft_inplace(&mut re, &mut im, true);
        for i in 0..n {
            assert!((re[i] - orig[i]).abs() < 1e-9, "round trip at {i}");
            assert!(im[i].abs() < 1e-9);
        }
    }

    #[test]
    fn wiener_removes_noise_and_keeps_smooth_structure() {
        // Smooth blob plus white noise: the filter must cut the noise hard
        // while leaving the blob's amplitude essentially intact.
        let (w, h) = (128usize, 128usize);
        let mut clean = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f64 - 64.0) / 22.0;
                let dy = (y as f64 - 64.0) / 22.0;
                clean.set(x, y, (1000.0 * (-(dx * dx + dy * dy) / 2.0).exp()) as f32);
            }
        }
        let mut noisy = clean.clone();
        let mut seed = 12345u64;
        for v in noisy.data.iter_mut() {
            // cheap deterministic LCG noise
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            *v += (u * 60.0) as f32;
        }
        let err = |a: &Image| -> f64 {
            let s: f64 = a
                .data
                .iter()
                .zip(&clean.data)
                .map(|(p, c)| ((*p - *c) as f64).powi(2))
                .sum();
            (s / a.data.len() as f64).sqrt()
        };
        let before = err(&noisy);
        let (filtered, rep) = wiener_psd(&noisy, 1.0, None).expect("wiener ran");
        let after = err(&filtered);
        assert!(
            after < before * 0.6,
            "should cut error substantially: {before:.2} -> {after:.2}"
        );
        assert!(rep.removed > 0.0 && rep.removed < 1.0, "removed {}", rep.removed);
        assert!(rep.cutoff_px > 2.0, "cutoff {} px", rep.cutoff_px);
    }
}
