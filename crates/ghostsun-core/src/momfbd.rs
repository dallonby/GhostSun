//! F18: per-frame seeing measurement, the foundation for multi-frame blind
//! deconvolution of a scanning spectroheliograph.
//!
//! WHY A SCAN IS AN MFBD DATASET AT ALL. Multi-frame blind deconvolution works
//! because many exposures see the same object through DIFFERENT unknown
//! atmospheres, so the ensemble constrains both object and PSFs. A
//! spectroheliograph looks like the opposite — one column of the object per
//! frame — but the PSF is far wider than the scan step, so consecutive frames
//! overlap heavily. Measured on the 2026-08-23 scans: the along-slit signal is
//! still 0.76 correlated 26 frames apart at 686 fps, i.e. ~26 independent looks
//! at each resolution element, ~12 at the 312 fps the tuning rule recommends.
//! And unlike a 2D imager, the displacement between those looks is KNOWN
//! exactly, from the F13 per-frame timestamps, so the registration ambiguity
//! that normally has to be solved jointly is simply absent here.
//!
//! THIS MODULE IS THE FIRST STEP AND ONLY THE FIRST STEP. Before any joint
//! inversion is worth attempting, the per-frame PSF has to be measurable, and
//! that claim has to be checked against truth rather than asserted. So this
//! estimates the per-frame ALONG-SLIT blur and nothing else. It buys lucky
//! weighting on its own, and it is the diagonal of the full problem.
//!
//! THE TRAP THIS MODULE MUST NOT FALL INTO, because this project has already
//! fallen into it once (F15 stacking, where high-frequency energy was used as a
//! sharpness proxy and noise IS high-frequency energy, so noisier scans were
//! weighted UP): a noisy frame carries more power at high spatial frequency
//! than a clean one, and unless the noise floor is subtracted FIRST it reads as
//! a sharper frame. Every step below subtracts an explicitly estimated noise
//! floor before comparing anything.

use crate::image2d::Image;
use crate::mathutil::{fft_inplace, median_inplace};
use rayon::prelude::*;

pub struct BlurEstimate {
    /// per-frame sigma^2 minus the scan median, in px^2 along the slit.
    /// Positive = blurrier than typical. Relative because the absolute scale
    /// needs an object model; lucky weighting only needs the ranking.
    pub dsigma2: Vec<f64>,
    /// frames with enough disc to measure
    pub n_measured: usize,
    /// robust spread of dsigma2, a one-number "how variable was the seeing"
    pub spread: f64,
    /// Autocorrelation of dsigma2 at lags 1..8 frames. THIS IS THE TEST THAT
    /// NEEDS NO GROUND TRUTH: atmospheric seeing is coherent over tens of
    /// milliseconds, so a real measurement decays smoothly over several
    /// frames, while an estimator that is merely reading photon noise gives
    /// white output and drops to ~0 at lag 1. On real data this separates
    /// "measuring seeing" from "measuring nothing" without any truth vector.
    pub acf: Vec<f64>,
}

/// Longest power of two not exceeding `n`.
fn pow2_floor(n: usize) -> usize {
    if n < 2 {
        return 0;
    }
    1usize << (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// Windowed power spectrum of one frame's along-slit profile.
///
/// Returns None when the column has too little disc to say anything.
fn column_power(core: &Image, t: usize, nfft: usize) -> Option<(Vec<f64>, f64)> {
    let h = core.h;
    let col: Vec<f64> = (0..h).map(|y| core.at(t, y) as f64).collect();
    let peak = col.iter().cloned().fold(0.0f64, f64::max);
    if peak <= 0.0 {
        return None;
    }
    // disc interior: the chord, trimmed so the limb ramps stay out. A limb is
    // a genuine sharp edge and would dominate the spectrum it is asked about.
    let thr = 0.5 * peak;
    let y0 = (0..h).position(|y| col[y] > thr)?;
    let y1 = (0..h).rposition(|y| col[y] > thr)?;
    if y1 <= y0 + 32 {
        return None;
    }
    let trim = ((y1 - y0) / 10).max(4);
    let (a, b) = (y0 + trim, y1 - trim);
    if b <= a + 16 {
        return None;
    }
    let m = b - a;
    let mean = col[a..b].iter().sum::<f64>() / m as f64;
    let mut re = vec![0.0f64; nfft];
    let mut im = vec![0.0f64; nfft];
    // Hann window: the chord ends are hard cuts, and their leakage would swamp
    // the high frequencies this estimator reads.
    for (i, r) in re.iter_mut().enumerate().take(m.min(nfft)) {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (m - 1) as f64).cos();
        *r = (col[a + i] - mean) * w;
    }
    fft_inplace(&mut re, &mut im, false);
    let half = nfft / 2;
    let p: Vec<f64> = (0..half).map(|k| re[k] * re[k] + im[k] * im[k]).collect();
    // Noise floor: the top eighth of the band, where a seeing-limited image
    // has no signal left at all.
    let mut tail: Vec<f64> = p[half * 7 / 8..].to_vec();
    let floor = median_inplace(&mut tail);
    Some((p, floor))
}

/// Per-frame along-slit blur relative to the scan median.
///
/// Each frame's noise-subtracted power spectrum is divided by that of its
/// LOCAL neighbours — local because consecutive frames see nearly the same
/// Sun while distant ones do not, so a global reference would report object
/// structure as seeing. For Gaussian blur the ratio is
/// exp(-4 pi^2 (sigma_t^2 - sigma_ref^2) f^2), so a straight line fitted to
/// log(ratio) against f^2 gives the difference of variances directly.
pub fn estimate_frame_blur(core: &Image, half_window: usize) -> BlurEstimate {
    let n = core.w;
    let nfft = pow2_floor(core.h).max(64);
    let half = nfft / 2;
    let spectra: Vec<Option<(Vec<f64>, f64)>> =
        (0..n).into_par_iter().map(|t| column_power(core, t, nfft)).collect();

    let lw = half_window.max(2);
    let dsig: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|t| {
            let (p, nf) = match &spectra[t] {
                Some(v) => v,
                None => return f64::NAN,
            };
            // local reference, this frame excluded
            let lo = t.saturating_sub(lw);
            let hi = (t + lw + 1).min(n);
            let mut r = vec![0.0f64; half];
            let mut rn = 0.0;
            let mut rfloor = 0.0;
            for (k, sp) in spectra.iter().enumerate().take(hi).skip(lo) {
                if k == t {
                    continue;
                }
                if let Some((q, qf)) = sp {
                    for (a, b) in r.iter_mut().zip(q.iter()) {
                        *a += *b;
                    }
                    rfloor += qf;
                    rn += 1.0;
                }
            }
            if rn < 3.0 {
                return f64::NAN;
            }
            for a in r.iter_mut() {
                *a /= rn;
            }
            rfloor /= rn;

            // The fitted BAND is set by the REFERENCE alone, never by the
            // frame under test. A per-frame threshold looks reasonable and is
            // a trap: a noisy frame passes it at fewer frequencies, the fit
            // loses its lever arm in f^2, and the slope collapses toward zero
            // — which reads as "sharper than average" for no reason but noise.
            // That is the F15 stacking bug wearing a different hat, and the
            // unit test below fails without this.
            let mut k_hi = 2;
            for k in 2..half {
                if r[k] - rfloor > 5.0 * rfloor {
                    k_hi = k;
                }
            }
            if k_hi < 8 {
                return f64::NAN;
            }
            let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for k in 2..=k_hi {
                let ps = p[k] - nf;
                let rs = r[k] - rfloor;
                if ps <= 0.0 || rs <= 0.0 {
                    continue;
                }
                let f = k as f64 / nfft as f64;
                let x = f * f;
                let y = (ps / rs).ln();
                // Smooth precision weight instead of a hard cut: the variance
                // of a log ratio goes as 1/SNR^2 in each spectrum, so this is
                // 1/var up to a constant, and it degrades gracefully rather
                // than removing points from one frame and not another.
                let sp = ps / (ps + nf);
                let sr = rs / (rs + rfloor);
                let w = sp * sp * sr * sr;
                sw += w;
                sx += w * x;
                sy += w * y;
                sxx += w * x * x;
                sxy += w * x * y;
            }
            let den = sw * sxx - sx * sx;
            if sw <= 1e-6 || den.abs() < 1e-30 {
                return f64::NAN;
            }
            let slope = (sw * sxy - sx * sy) / den;
            // log ratio = -4 pi^2 (sigma_t^2 - sigma_ref^2) f^2
            -slope / (4.0 * std::f64::consts::PI * std::f64::consts::PI)
        })
        .collect();

    let mut ok: Vec<f64> = dsig.iter().cloned().filter(|v| v.is_finite()).collect();
    let n_measured = ok.len();
    if n_measured == 0 {
        return BlurEstimate {
            dsigma2: vec![0.0; n],
            n_measured: 0,
            spread: 0.0,
            acf: Vec::new(),
        };
    }
    let med = median_inplace(&mut ok);
    let mut ad: Vec<f64> = dsig
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - med).abs())
        .collect();
    let spread = 1.4826 * median_inplace(&mut ad);
    let dsigma2: Vec<f64> =
        dsig.iter().map(|v| if v.is_finite() { v - med } else { 0.0 }).collect();
    let valid: Vec<bool> = dsig.iter().map(|v| v.is_finite()).collect();
    let var: f64 = dsigma2
        .iter()
        .zip(&valid)
        .filter(|(_, &v)| v)
        .map(|(d, _)| d * d)
        .sum::<f64>()
        / n_measured as f64;
    let acf: Vec<f64> = (1..=8)
        .map(|lag| {
            let (mut acc, mut cnt) = (0.0, 0.0);
            for i in 0..n.saturating_sub(lag) {
                if valid[i] && valid[i + lag] {
                    acc += dsigma2[i] * dsigma2[i + lag];
                    cnt += 1.0;
                }
            }
            if cnt < 20.0 || var <= 0.0 { f64::NAN } else { acc / cnt / var }
        })
        .collect();
    BlurEstimate { dsigma2, n_measured, spread, acf }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathutil::gaussian_smooth;

    /// Build a fake raw disk: every column the same random along-slit
    /// structure, blurred by a known per-frame sigma. The estimator must
    /// recover the ORDERING and the differences of sigma^2.
    fn synth_disk(sigmas: &[f64], h: usize, noise: f64, seed: u64) -> Image {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let base: Vec<f64> = (0..h).map(|_| rnd()).collect();
        let base = gaussian_smooth(&base, 1.5);
        let mut img = Image::new(sigmas.len(), h);
        for (t, &s) in sigmas.iter().enumerate() {
            let blurred = if s > 0.0 { gaussian_smooth(&base, s) } else { base.clone() };
            for y in 0..h {
                let disc = if y > h / 8 && y < h * 7 / 8 { 1.0 } else { 0.0 };
                let v = (1000.0 + 300.0 * blurred[y] + noise * 300.0 * rnd()) * disc;
                img.set(t, y, v.max(0.0) as f32);
            }
        }
        img
    }

    #[test]
    fn recovers_the_ordering_of_per_frame_blur() {
        // alternating sharp/blurred blocks, well separated
        let sigmas: Vec<f64> = (0..96).map(|t| if (t / 8) % 2 == 0 { 1.0 } else { 3.0 }).collect();
        let img = synth_disk(&sigmas, 512, 0.05, 7);
        let est = estimate_frame_blur(&img, 12);
        assert!(est.n_measured > 80, "measured {}", est.n_measured);
        let sharp: Vec<f64> = (0..96).filter(|t| (t / 8) % 2 == 0).map(|t| est.dsigma2[t]).collect();
        let blur: Vec<f64> = (0..96).filter(|t| (t / 8) % 2 == 1).map(|t| est.dsigma2[t]).collect();
        let ms = sharp.iter().sum::<f64>() / sharp.len() as f64;
        let mb = blur.iter().sum::<f64>() / blur.len() as f64;
        assert!(mb > ms, "blurred {mb:.3} should exceed sharp {ms:.3}");
        // true difference of variances is 3^2 - 1^2 = 8 px^2; the local
        // reference mixes both populations so the measured split is smaller,
        // but it must be unambiguous
        assert!(mb - ms > 1.0, "separation {:.3} px^2 too small", mb - ms);
    }

    #[test]
    fn a_noisier_frame_is_not_reported_as_sharper() {
        // The F15 stacking bug in miniature: same blur everywhere, but one
        // block far noisier. Noise is high-frequency power, so an estimator
        // that forgets to subtract the floor will call the noisy block sharp.
        let sigmas = vec![2.0f64; 64];
        let mut img = synth_disk(&sigmas, 512, 0.05, 11);
        let noisy = synth_disk(&sigmas, 512, 0.40, 11);
        for t in 32..64 {
            for y in 0..img.h {
                img.set(t, y, noisy.at(t, y));
            }
        }
        let est = estimate_frame_blur(&img, 12);
        let clean: f64 = (4..28).map(|t| est.dsigma2[t]).sum::<f64>() / 24.0;
        let dirty: f64 = (36..60).map(|t| est.dsigma2[t]).sum::<f64>() / 24.0;
        assert!(
            dirty > clean - 0.5,
            "noisy block read as sharper: clean {clean:.3} vs noisy {dirty:.3}"
        );
    }

    #[test]
    fn constant_seeing_produces_no_spurious_variation() {
        let sigmas = vec![2.0f64; 80];
        let img = synth_disk(&sigmas, 512, 0.05, 3);
        let est = estimate_frame_blur(&img, 12);
        assert!(est.spread < 1.0, "spurious spread {:.3} px^2", est.spread);
    }
}
