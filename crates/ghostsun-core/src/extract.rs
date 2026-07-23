//! Disk reconstruction: sample every frame along the fitted line position.
//!
//! Every path starts with cardinal cubic B-spline reconstruction at the
//! requested sub-pixel wavelengths. Point and Gaussian sampling support the
//! main profile pipeline and its continuum measurements. The lightweight
//! comparison pipeline uses an independently derived, noise-adaptive local
//! polynomial estimator. Values remain floating point throughout extraction.

use crate::image2d::Image;
use crate::linefit::LineGeometry;
use crate::mathutil::{bspline_eval, bspline_prefilter, median_inplace, polyval};
use crate::ser::SerReader;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub enum SpectralKernel {
    /// Exact cardinal B-spline sample at the requested wavelength.
    Point,
    /// Normalized Gaussian integration for continuum and wing measurements.
    Gaussian { sigma: f64 },
    /// Noise-reducing local polynomial estimate of the line core.
    LocalPolynomial,
}

pub struct ExtractOptions {
    pub shift: f64,            // wavelength offset in px from line center
    pub transpose_input: bool, // dispersion vertical in SER -> transpose
    pub kernel: SpectralKernel,
    /// Optional per-frame additive spectral offset (F3 flexure), px.
    pub frame_offsets: Option<Vec<f64>>,
}

/// Fast, sparse continuum preview used to solve frame-to-frame motion before
/// the expensive profile extraction. Only three spectral samples are read for
/// each slit row, directly from the memory-mapped SER; no frame allocation or
/// spectral prefilter is required.
pub fn reconstruct_continuum_preview(
    reader: &SerReader,
    geom: &LineGeometry,
    transpose: bool,
    spectral_shift: f64,
) -> Image {
    let n = reader.header.frame_count;
    let slit_h = if transpose {
        reader.header.width
    } else {
        reader.header.height
    };
    let spec_w = if transpose {
        reader.header.height
    } else {
        reader.header.width
    };
    let native_w = reader.header.width;
    let bpp = reader.bytes_per_px();
    // Collapse the three linearly interpolated binomial taps into four
    // native pixels. This is algebraically identical to
    // .25*S(p-1) + .50*S(p) + .25*S(p+1), but avoids two raw loads and the
    // indexing closures in the hot loop.
    let taps: Vec<(usize, [f32; 4])> = (0..slit_h)
        .map(|y| {
            let p = (polyval(&geom.coeffs, y as f64) + spectral_shift)
                .clamp(1.0, (spec_w - 2) as f64);
            let x0 = p.floor() as usize;
            let f = (p - x0 as f64) as f32;
            (
                x0,
                [
                    0.25 * (1.0 - f),
                    0.50 - 0.25 * f,
                    0.25 + 0.25 * f,
                    0.25 * f,
                ],
            )
        })
        .collect();

    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|t| {
            let raw = reader.raw_frames(t, 1);
            let mut col = vec![0.0f32; slit_h];
            if bpp == 2 {
                for y in 0..slit_h {
                    let (x0, weights) = taps[y];
                    let spectral = [
                        x0 - 1,
                        x0,
                        x0 + 1,
                        (x0 + 2).min(spec_w - 1),
                    ];
                    let mut value = 0.0f32;
                    for k in 0..4 {
                        let index = if transpose {
                            spectral[k] * native_w + y
                        } else {
                            y * native_w + spectral[k]
                        };
                        value += weights[k]
                            * u16::from_le_bytes([raw[2 * index], raw[2 * index + 1]]) as f32;
                    }
                    col[y] = value;
                }
            } else {
                for y in 0..slit_h {
                    let (x0, weights) = taps[y];
                    let spectral = [
                        x0 - 1,
                        x0,
                        x0 + 1,
                        (x0 + 2).min(spec_w - 1),
                    ];
                    let mut value = 0.0f32;
                    for k in 0..4 {
                        let index = if transpose {
                            spectral[k] * native_w + y
                        } else {
                            y * native_w + spectral[k]
                        };
                        value += weights[k] * raw[index] as f32;
                    }
                    col[y] = value * 257.0;
                }
            }
            col
        })
        .collect();

    let mut preview = Image::new(n, slit_h);
    for (t, col) in cols.iter().enumerate() {
        preview.set_column(t, col);
    }
    preview
}

/// Reconstruct the disk: output image has width = n_frames, height = slit_h.
pub fn reconstruct_disk(reader: &SerReader, geom: &LineGeometry, opts: &ExtractOptions) -> Image {
    let n = reader.header.frame_count;
    let slit_h = if opts.transpose_input {
        reader.header.width
    } else {
        reader.header.height
    };
    let mut disk = Image::new(n, slit_h);

    // Precompute sampling positions per row
    let pos: Vec<f64> = (0..slit_h)
        .map(|y| polyval(&geom.coeffs, y as f64) + opts.shift)
        .collect();

    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|t| {
            let mut frame = reader.frame(t);
            if opts.transpose_input {
                frame = frame.transpose();
            }
            let off = opts.frame_offsets.as_ref().map(|f| f[t]).unwrap_or(0.0);
            if off == 0.0 {
                extract_column(&frame, &pos, opts.kernel)
            } else {
                let shifted: Vec<f64> = pos.iter().map(|p| p + off).collect();
                extract_column(&frame, &shifted, opts.kernel)
            }
        })
        .collect();

    for (t, col) in cols.iter().enumerate() {
        disk.set_column(t, col);
    }
    disk
}

struct LocalPolynomialKernel {
    offsets: Vec<f64>,
    smooth_weights: Vec<f64>,
    detail_weights: Vec<f64>,
}

static LOCAL_POLYNOMIAL_KERNEL: OnceLock<LocalPolynomialKernel> = OnceLock::new();

// Preserve the quartic model below 0.1% estimated row noise and use the
// lower-variance quadratic model above 0.3%, with a smooth transition to
// avoid hard mode boundaries across the reconstructed disk.
const DETAIL_NOISE_LIMIT: f64 = 0.001;
const SMOOTH_NOISE_LIMIT: f64 = 0.003;

enum PreparedKernel {
    Fixed {
        offsets: Vec<f64>,
        weights: Vec<f64>,
    },
    Adaptive(&'static LocalPolynomialKernel),
}

fn local_polynomial_weights(offsets: &[f64], spatial: &[f64], degree: usize) -> Vec<f64> {
    debug_assert!(degree >= 2 && (degree & 1) == 0);
    debug_assert_eq!(offsets.len(), spatial.len());
    let terms = degree / 2 + 1;

    // Weighted least squares for the even polynomial
    // f(x) = a0 + a2*x^2 + ... . The first row of
    // (X' W X)^-1 X' W maps samples directly to f(0) = a0.
    let mut normal = DMatrix::<f64>::zeros(terms, terms);
    for (&x, &w) in offsets.iter().zip(spatial) {
        for r in 0..terms {
            for c in 0..terms {
                normal[(r, c)] += w * x.powi((2 * (r + c)) as i32);
            }
        }
    }
    let inverse = normal
        .try_inverse()
        .expect("local polynomial kernel is full rank");
    let mut weights: Vec<f64> = offsets
        .iter()
        .zip(spatial)
        .map(|(&x, &w)| {
            let influence = (0..terms)
                .map(|j| inverse[(0, j)] * x.powi((2 * j) as i32))
                .sum::<f64>();
            w * influence
        })
        .collect();

    // Remove round-off drift so constants are preserved exactly enough for
    // long sequences of floating-point processing.
    let sum: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

fn build_local_polynomial_kernel() -> LocalPolynomialKernel {
    const RADIUS: f64 = 2.4;
    const SEGMENTS: usize = 5;
    const BANDWIDTH: f64 = 4.0;

    let actual_step = 2.0 * RADIUS / SEGMENTS as f64;
    let offsets: Vec<f64> = (0..=SEGMENTS)
        .map(|i| -RADIUS + i as f64 * actual_step)
        .collect();
    let spatial: Vec<f64> = offsets
        .iter()
        .map(|&x| (-(x * x) / (2.0 * BANDWIDTH * BANDWIDTH)).exp())
        .collect();
    let smooth_weights = local_polynomial_weights(&offsets, &spatial, 2);
    let detail_weights = local_polynomial_weights(&offsets, &spatial, 4);

    LocalPolynomialKernel {
        offsets,
        smooth_weights,
        detail_weights,
    }
}

fn local_polynomial_kernel() -> &'static LocalPolynomialKernel {
    LOCAL_POLYNOMIAL_KERNEL.get_or_init(build_local_polynomial_kernel)
}

fn prepare_kernel(kernel: SpectralKernel) -> PreparedKernel {
    match kernel {
        SpectralKernel::Point => PreparedKernel::Fixed {
            offsets: vec![0.0],
            weights: vec![1.0],
        },
        SpectralKernel::Gaussian { sigma } if sigma > 0.0 => {
            let radius = (2.5 * sigma).ceil();
            let mut offsets = Vec::new();
            let mut weights = Vec::new();
            let mut offset = -radius;
            while offset <= radius + 1e-9 {
                offsets.push(offset);
                weights.push((-(offset * offset) / (2.0 * sigma * sigma)).exp());
                offset += 0.5;
            }
            let sum: f64 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= sum;
            }
            PreparedKernel::Fixed { offsets, weights }
        }
        SpectralKernel::Gaussian { .. } => PreparedKernel::Fixed {
            offsets: vec![0.0],
            weights: vec![1.0],
        },
        SpectralKernel::LocalPolynomial => PreparedKernel::Adaptive(local_polynomial_kernel()),
    }
}

fn relative_row_noise(row: &[f32], work: &mut Vec<f64>) -> f64 {
    if row.len() < 3 {
        return 0.0;
    }
    work.clear();
    work.extend(
        row.windows(3)
            .map(|v| (v[0] as f64 - 2.0 * v[1] as f64 + v[2] as f64).abs()),
    );

    // For independent Gaussian noise, MAD(second difference) is
    // 0.67449*sqrt(6)*sigma. A broad spectral feature affects fewer than half
    // the samples, so the median remains a useful high-frequency noise probe.
    let sigma = median_inplace(work) / (0.674_489_750_196_081_7 * 6.0_f64.sqrt());
    let scale = row.iter().map(|&v| (v as f64).abs()).sum::<f64>() / row.len() as f64;
    sigma / scale.max(1.0)
}

fn smoothstep(value: f64, low: f64, high: f64) -> f64 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn extract_column(frame: &Image, pos: &[f64], kernel: SpectralKernel) -> Vec<f32> {
    let w = frame.w;
    let mut out = vec![0.0f32; frame.h];
    let kernel = prepare_kernel(kernel);
    let mut coef = vec![0.0f64; w];
    let mut noise_work = Vec::with_capacity(w.saturating_sub(2));
    for y in 0..frame.h {
        let row = frame.row(y);
        for (i, &v) in row.iter().enumerate() {
            coef[i] = v as f64;
        }
        bspline_prefilter(&mut coef);
        let acc = match &kernel {
            PreparedKernel::Fixed { offsets, weights } => offsets
                .iter()
                .zip(weights)
                .map(|(offset, weight)| {
                    let x = (pos[y] + offset).clamp(1.0, (w - 2) as f64);
                    weight * bspline_eval(&coef, x)
                })
                .sum(),
            PreparedKernel::Adaptive(local) => {
                let mut smooth = 0.0;
                let mut detail = 0.0;
                for ((offset, smooth_weight), detail_weight) in local
                    .offsets
                    .iter()
                    .zip(&local.smooth_weights)
                    .zip(&local.detail_weights)
                {
                    let x = (pos[y] + offset).clamp(1.0, (w - 2) as f64);
                    let sample = bspline_eval(&coef, x);
                    smooth += smooth_weight * sample;
                    detail += detail_weight * sample;
                }
                let noisy = smoothstep(
                    relative_row_noise(row, &mut noise_work),
                    DETAIL_NOISE_LIMIT,
                    SMOOTH_NOISE_LIMIT,
                );
                detail + noisy * (smooth - detail)
            }
        };
        out[y] = acc as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moment(offsets: &[f64], weights: &[f64], degree: i32) -> f64 {
        offsets
            .iter()
            .zip(weights)
            .map(|(&x, &weight)| weight * x.powi(degree))
            .sum()
    }

    #[test]
    fn local_polynomial_weights_preserve_the_fitted_models() {
        let kernel = local_polynomial_kernel();

        assert!((moment(&kernel.offsets, &kernel.smooth_weights, 0) - 1.0).abs() < 1e-12);
        assert!(moment(&kernel.offsets, &kernel.smooth_weights, 2).abs() < 1e-12);
        assert!((moment(&kernel.offsets, &kernel.detail_weights, 0) - 1.0).abs() < 1e-12);
        assert!(moment(&kernel.offsets, &kernel.detail_weights, 2).abs() < 1e-12);
        assert!(moment(&kernel.offsets, &kernel.detail_weights, 4).abs() < 1e-11);

        for i in 0..kernel.offsets.len() {
            let opposite = kernel.offsets.len() - 1 - i;
            assert!((kernel.offsets[i] + kernel.offsets[opposite]).abs() < 1e-12);
            assert!((kernel.smooth_weights[i] - kernel.smooth_weights[opposite]).abs() < 1e-12);
            assert!((kernel.detail_weights[i] - kernel.detail_weights[opposite]).abs() < 1e-12);
        }
    }

    #[test]
    fn row_noise_probe_separates_smooth_profiles_from_read_noise() {
        let mut clean = Vec::with_capacity(160);
        let mut noisy = Vec::with_capacity(160);
        for x in 0..160 {
            let dx = x as f64 - 80.2;
            let profile = 30_200.0
                - 24_000.0 * (-(dx * dx) / (2.0 * 2.3_f64.powi(2))).exp()
                - 3_000.0 * (-(dx * dx) / (2.0 * 13.0_f64.powi(2))).exp();
            clean.push(profile as f32);
            let noise = (((x * 37 + 11) % 29) as f64 - 14.0) * 24.0;
            noisy.push((profile + noise) as f32);
        }

        let mut work = Vec::new();
        let clean_level = relative_row_noise(&clean, &mut work);
        let noisy_level = relative_row_noise(&noisy, &mut work);
        assert!(
            clean_level < DETAIL_NOISE_LIMIT,
            "clean level was {clean_level}"
        );
        assert!(
            noisy_level > SMOOTH_NOISE_LIMIT,
            "noisy level was {noisy_level}"
        );
    }

    #[test]
    fn adaptive_kernel_reduces_deterministic_sample_noise() {
        let (w, h) = (96, 96);
        let mut reference = Image::new(w, h);
        let mut noisy = Image::new(w, h);
        let mut positions = Vec::with_capacity(h);

        for y in 0..h {
            let center = 47.2 + 0.23 * (y as f64 * 0.17).sin();
            positions.push(center);
            for x in 0..w {
                let dx = x as f64 - center;
                let profile = 30_200.0 - 24_000.0 * (-(dx * dx) / (2.0 * 2.3_f64.powi(2))).exp();
                // Keep both images in the smoothing regime so this test
                // isolates variance reduction from the model switch itself.
                let trigger = if dx.abs() > 10.0 {
                    (((x * 37 + y * 101 + 11) % 29) as f64 - 14.0) * 24.0
                } else {
                    0.0
                };
                let mut hash = (x as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                hash ^= hash >> 30;
                hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                hash ^= hash >> 27;
                let unit = (hash >> 11) as f64 / (1_u64 << 53) as f64;
                let noise = (2.0 * unit - 1.0) * 300.0;
                reference.set(x, y, (profile + trigger) as f32);
                noisy.set(x, y, (profile + trigger + noise) as f32);
            }
        }

        let point_clean = extract_column(&reference, &positions, SpectralKernel::Point);
        let point_noisy = extract_column(&noisy, &positions, SpectralKernel::Point);
        let local_clean = extract_column(&reference, &positions, SpectralKernel::LocalPolynomial);
        let local_noisy = extract_column(&noisy, &positions, SpectralKernel::LocalPolynomial);
        let rmse = |a: &[f32], b: &[f32]| {
            (a.iter()
                .zip(b)
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum::<f64>()
                / a.len() as f64)
                .sqrt()
        };

        let point_error = rmse(&point_noisy, &point_clean);
        let local_error = rmse(&local_noisy, &local_clean);
        assert!(
            local_error < point_error * 0.75,
            "local {local_error:.3}, point {point_error:.3}"
        );
    }
}
