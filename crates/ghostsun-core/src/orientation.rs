//! Absolute solar orientation by feature registration against a calibrated
//! full-disk reference image.
//!
//! A circular disk alone cannot determine celestial north or image parity.
//! This module normalizes both disks, extracts large-scale-independent
//! chromospheric structure, searches both parities through 360 degrees, and
//! returns the transform which maps the source into the reference pose.

use crate::image2d::Image;
use crate::mathutil::lanczos3;
use crate::metrics::DiskFit;
use rayon::prelude::*;

const MATCH_SIDE: usize = 384;
const MATCH_RADIUS: f64 = 180.0;
const MATCH_INNER_RADIUS: f64 = MATCH_RADIUS * 0.82;

#[derive(Clone, Copy, Debug)]
pub struct OrientationMatch {
    /// Mirror the image horizontally about the disk centre before rotation.
    pub mirrored: bool,
    /// Visual counter-clockwise rotation applied after the optional mirror.
    pub rotation_deg: f64,
    /// Normalized cross-correlation of the registered feature maps.
    pub score: f64,
    /// Best score obtained with the opposite parity.
    pub alternate_parity_score: f64,
    /// Best score at a substantially different rotation with this parity.
    pub alternate_rotation_score: f64,
}

impl OrientationMatch {
    pub fn confidence_margin(&self) -> f64 {
        self.score
            - self
                .alternate_parity_score
                .max(self.alternate_rotation_score)
    }

    /// Reject weak or parity-ambiguous matches instead of silently placing
    /// celestial north in the wrong direction.
    pub fn is_confident(&self) -> bool {
        self.score >= 0.10 && self.confidence_margin() >= 0.025
    }
}

/// Find the mirror/rotation transform which maps `source` into `reference`.
///
/// Disk geometry is supplied independently because GONG references already
/// have calibrated geometry while reconstructed scans have fitted geometry.
pub fn match_to_reference(
    source: &Image,
    source_disk: &DiskFit,
    reference: &Image,
    reference_disk: &DiskFit,
) -> Result<OrientationMatch, String> {
    validate_disk(source, source_disk, "source")?;
    validate_disk(reference, reference_disk, "reference")?;

    let source_features = feature_map(&normalize_disk(source, source_disk));
    let reference_features = feature_map(&normalize_disk(reference, reference_disk));
    let points = match_points(&reference_features);
    if points.len() < 5_000 {
        return Err("not enough valid solar-disk pixels for orientation matching".into());
    }

    let normal = best_for_parity(&source_features, &points, false);
    let mirrored = best_for_parity(&source_features, &points, true);
    let (best, alternate) = if mirrored.score > normal.score {
        (mirrored, normal)
    } else {
        (normal, mirrored)
    };

    Ok(OrientationMatch {
        mirrored: best.mirrored,
        rotation_deg: wrap_degrees(best.angle),
        score: best.score,
        alternate_parity_score: alternate.score,
        alternate_rotation_score: best.alternate_rotation_score,
    })
}

/// Apply the feature-derived pose without changing the canvas size.
///
/// The transform is centred on the fitted solar disk rather than the canvas,
/// preserving the disk location even if the reconstruction is slightly
/// off-centre. Lanczos-3 keeps this final presentation resample sharp.
pub fn apply_orientation(
    image: &Image,
    disk: &DiskFit,
    mirrored: bool,
    rotation_deg: f64,
) -> Image {
    let theta = rotation_deg.to_radians();
    let (ct, st) = (theta.cos(), theta.sin());
    let rows: Vec<Vec<f32>> = (0..image.h)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![0.0f32; image.w];
            let dy = y as f64 - disk.yc;
            for (x, out) in row.iter_mut().enumerate() {
                let dx = x as f64 - disk.xc;

                // Invert the requested output transform. In image coordinates
                // (+y downward), positive angles are visually counter-clockwise.
                let qx = ct * dx - st * dy;
                let qy = st * dx + ct * dy;
                let sx = disk.xc + if mirrored { -qx } else { qx };
                let sy = disk.yc + qy;
                *out = sample_lanczos3(image, sx, sy);
            }
            row
        })
        .collect();

    Image {
        w: image.w,
        h: image.h,
        data: rows.into_iter().flatten().collect(),
    }
}

fn validate_disk(image: &Image, disk: &DiskFit, label: &str) -> Result<(), String> {
    if image.w < 32
        || image.h < 32
        || !disk.xc.is_finite()
        || !disk.yc.is_finite()
        || !disk.r.is_finite()
        || disk.r < 16.0
    {
        return Err(format!("invalid {label} disk geometry"));
    }
    Ok(())
}

fn normalize_disk(image: &Image, disk: &DiskFit) -> Image {
    let mut out = Image::new(MATCH_SIDE, MATCH_SIDE);
    let centre = (MATCH_SIDE as f64 - 1.0) * 0.5;
    let scale = disk.r / MATCH_RADIUS;
    for y in 0..MATCH_SIDE {
        let sy = disk.yc + (y as f64 - centre) * scale;
        for x in 0..MATCH_SIDE {
            let sx = disk.xc + (x as f64 - centre) * scale;
            out.set(x, y, sample_bilinear(image, sx, sy).max(0.0).ln_1p());
        }
    }
    out
}

fn feature_map(normalized: &Image) -> Image {
    // Removing a broad Gaussian illumination model makes limb darkening,
    // exposure, and contrast differences largely irrelevant. The remaining
    // map is dominated by active regions and filaments shared with GONG.
    let background = gaussian_blur(normalized, 9.0);
    let mut residual = Image::new(normalized.w, normalized.h);
    for i in 0..residual.data.len() {
        let bg = background.data[i];
        residual.data[i] = (normalized.data[i] - bg) / bg.abs().max(0.02);
    }
    gaussian_blur(&residual, 1.0)
}

/// Tuples are `(x, y, reference_feature)`, restricted to the disk interior so
/// limb shape and prominences cannot dominate the absolute pose.
fn match_points(reference: &Image) -> Vec<(f64, f64, f64)> {
    let centre = (MATCH_SIDE as f64 - 1.0) * 0.5;
    let r2 = MATCH_INNER_RADIUS * MATCH_INNER_RADIUS;
    let mut points = Vec::new();
    for y in 0..MATCH_SIDE {
        let dy = y as f64 - centre;
        for x in 0..MATCH_SIDE {
            let dx = x as f64 - centre;
            if dx * dx + dy * dy <= r2 {
                points.push((dx, dy, reference.at(x, y) as f64));
            }
        }
    }
    points
}

struct ParityMatch {
    mirrored: bool,
    score: f64,
    angle: f64,
    alternate_rotation_score: f64,
}

fn best_for_parity(source: &Image, points: &[(f64, f64, f64)], mirrored: bool) -> ParityMatch {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_angle = 0.0;
    let mut coarse_scores = Vec::with_capacity(180);
    for step in 0..180 {
        let angle = step as f64 * 2.0;
        let score = score_transform(source, points, mirrored, angle);
        coarse_scores.push((angle, score));
        if score > best_score {
            best_score = score;
            best_angle = angle;
        }
    }

    // Sub-degree refinement is important: at full-disk radius, one degree is
    // several reconstructed pixels and visibly softens fine structure.
    let coarse = best_angle;
    for step in -20..=20 {
        let angle = coarse + step as f64 * 0.1;
        let score = score_transform(source, points, mirrored, angle);
        if score > best_score {
            best_score = score;
            best_angle = angle;
        }
    }
    let alternate_rotation_score = coarse_scores
        .into_iter()
        .filter(|(angle, _)| angular_distance(*angle, best_angle) >= 12.0)
        .map(|(_, score)| score)
        .fold(f64::NEG_INFINITY, f64::max);
    ParityMatch {
        mirrored,
        score: best_score,
        angle: best_angle,
        alternate_rotation_score,
    }
}

fn score_transform(
    source: &Image,
    points: &[(f64, f64, f64)],
    mirrored: bool,
    angle_deg: f64,
) -> f64 {
    let theta = angle_deg.to_radians();
    let (ct, st) = (theta.cos(), theta.sin());
    let centre = (MATCH_SIDE as f64 - 1.0) * 0.5;

    let mut sx_sum = 0.0;
    let mut sy_sum = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for &(dx, dy, target) in points {
        // Destination -> source inverse, matching apply_orientation().
        let qx = ct * dx - st * dy;
        let qy = st * dx + ct * dy;
        let source_x = centre + if mirrored { -qx } else { qx };
        let source_y = centre + qy;
        let value = sample_bilinear(source, source_x, source_y) as f64;
        sx_sum += value;
        sy_sum += target;
        sxx += value * value;
        syy += target * target;
        sxy += value * target;
    }
    let n = points.len() as f64;
    let covariance = sxy - sx_sum * sy_sum / n;
    let source_energy = (sxx - sx_sum * sx_sum / n).max(0.0);
    let target_energy = (syy - sy_sum * sy_sum / n).max(0.0);
    covariance / (source_energy * target_energy).sqrt().max(1e-20)
}

fn gaussian_blur(image: &Image, sigma: f64) -> Image {
    let radius = (sigma * 3.0).ceil() as isize;
    let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut sum = 0.0f64;
    for i in -radius..=radius {
        let weight = (-0.5 * (i as f64 / sigma).powi(2)).exp();
        kernel.push(weight as f32);
        sum += weight;
    }
    for weight in &mut kernel {
        *weight /= sum as f32;
    }

    let mut horizontal = Image::new(image.w, image.h);
    for y in 0..image.h {
        for x in 0..image.w {
            let mut value = 0.0f32;
            for (k, &weight) in kernel.iter().enumerate() {
                let xx = x as isize + k as isize - radius;
                value += image.at_clamped(xx, y as isize) * weight;
            }
            horizontal.set(x, y, value);
        }
    }

    let mut out = Image::new(image.w, image.h);
    for y in 0..image.h {
        for x in 0..image.w {
            let mut value = 0.0f32;
            for (k, &weight) in kernel.iter().enumerate() {
                let yy = y as isize + k as isize - radius;
                value += horizontal.at_clamped(x as isize, yy) * weight;
            }
            out.set(x, y, value);
        }
    }
    out
}

#[inline]
fn sample_bilinear(image: &Image, x: f64, y: f64) -> f32 {
    if x < 0.0 || y < 0.0 || x > (image.w - 1) as f64 || y > (image.h - 1) as f64 {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(image.w - 1);
    let y1 = (y0 + 1).min(image.h - 1);
    let tx = (x - x0 as f64) as f32;
    let ty = (y - y0 as f64) as f32;
    let a = image.at(x0, y0) * (1.0 - tx) + image.at(x1, y0) * tx;
    let b = image.at(x0, y1) * (1.0 - tx) + image.at(x1, y1) * tx;
    a * (1.0 - ty) + b * ty
}

#[inline]
fn sample_lanczos3(image: &Image, x: f64, y: f64) -> f32 {
    if x < -3.0 || y < -3.0 || x > image.w as f64 + 2.0 || y > image.h as f64 + 2.0 {
        return 0.0;
    }
    let xf = x.floor() as isize;
    let yf = y.floor() as isize;
    let mut value = 0.0f64;
    let mut weight_sum = 0.0f64;
    for yy in (yf - 2)..=(yf + 3) {
        let wy = lanczos3(y - yy as f64);
        for xx in (xf - 2)..=(xf + 3) {
            let weight = lanczos3(x - xx as f64) * wy;
            if weight != 0.0 {
                value += image.at_clamped(xx, yy) as f64 * weight;
                weight_sum += weight;
            }
        }
    }
    if weight_sum.abs() < 1e-12 {
        0.0
    } else {
        (value / weight_sum) as f32
    }
}

fn wrap_degrees(angle: f64) -> f64 {
    (angle + 180.0).rem_euclid(360.0) - 180.0
}

fn angular_distance(a: f64, b: f64) -> f64 {
    wrap_degrees(a - b).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asymmetric_sun(side: usize) -> (Image, DiskFit) {
        let c = (side as f64 - 1.0) * 0.5;
        let r = side as f64 * 0.42;
        let disk = DiskFit { xc: c, yc: c, r };
        let mut image = Image::new(side, side);
        let features = [
            (-0.42, -0.28, -0.30, 0.10),
            (0.31, -0.43, 0.42, 0.08),
            (0.18, 0.22, -0.24, 0.13),
            (-0.48, 0.39, 0.30, 0.07),
        ];
        for y in 0..side {
            for x in 0..side {
                let dx = (x as f64 - c) / r;
                let dy = (y as f64 - c) / r;
                let rr = (dx * dx + dy * dy).sqrt();
                if rr <= 1.0 {
                    let mut value = 20_000.0 * (1.0 - 0.35 * rr * rr);
                    for &(fx, fy, amp, sigma) in &features {
                        let d2 = (dx - fx).powi(2) + (dy - fy).powi(2);
                        value *= 1.0 + amp * (-0.5 * d2 / (sigma * sigma)).exp();
                    }
                    value *= 1.0 + 0.035 * (17.0 * dx + 9.0 * dy).sin();
                    image.set(x, y, value as f32);
                }
            }
        }
        (image, disk)
    }

    fn angular_error(a: f64, b: f64) -> f64 {
        (a - b + 180.0).rem_euclid(360.0) - 180.0
    }

    #[test]
    fn recovers_mirror_and_subdegree_rotation() {
        let (reference, disk) = asymmetric_sun(420);
        let expected_angle = 67.4;
        // A mirror+rotation transform is its own inverse.
        let source = apply_orientation(&reference, &disk, true, expected_angle);
        let matched = match_to_reference(&source, &disk, &reference, &disk).unwrap();
        assert!(matched.mirrored);
        assert!(angular_error(matched.rotation_deg, expected_angle).abs() < 0.6);
        assert!(matched.score > 0.8, "score was {}", matched.score);
        assert!(matched.is_confident());
    }

    #[test]
    fn identity_transform_preserves_pixels() {
        let (image, disk) = asymmetric_sun(96);
        let transformed = apply_orientation(&image, &disk, false, 0.0);
        let max_error = image
            .data
            .iter()
            .zip(&transformed.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.01, "maximum identity error was {max_error}");
    }
}
