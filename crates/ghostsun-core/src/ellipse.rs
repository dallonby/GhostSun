//! Robust direct ellipse fitting: Halir-Flusser numerically-stable direct
//! least squares, wrapped in RANSAC for outlier rejection (prominences,
//! filaments touching the limb) and refined with Tukey IRLS on Sampson
//! distances.

use crate::limb::EdgePoint;
use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// General conic A x^2 + B xy + C y^2 + D x + E y + F = 0
#[derive(Clone, Copy, Debug)]
pub struct Conic {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct EllipseGeom {
    pub xc: f64,
    pub yc: f64,
    /// centered, normalized conic: an u^2 + bn uv + cn v^2 = 1
    pub an: f64,
    pub bn: f64,
    pub cn: f64,
    /// derived: x-scale, shear and circle radius of the shear+scale model
    pub sx: f64,
    pub shear: f64,
    pub radius: f64,
}

impl Conic {
    pub fn sampson(&self, x: f64, y: f64) -> f64 {
        let val = self.a * x * x + self.b * x * y + self.c * y * y + self.d * x + self.e * y + self.f;
        let gx = 2.0 * self.a * x + self.b * y + self.d;
        let gy = self.b * x + 2.0 * self.c * y + self.e;
        let g = (gx * gx + gy * gy).sqrt().max(1e-12);
        (val / g).abs()
    }

    pub fn geometry(&self) -> Option<EllipseGeom> {
        let (a, b, c, d, e, f) = (self.a, self.b, self.c, self.d, self.e, self.f);
        let delta = b * b - 4.0 * a * c;
        if delta >= 0.0 {
            return None; // not an ellipse
        }
        let xc = (2.0 * c * d - b * e) / delta;
        let yc = (2.0 * a * e - b * d) / delta;
        let g = -(a * xc * xc + b * xc * yc + c * yc * yc + d * xc + e * yc + f);
        if g <= 0.0 {
            // conic sign flipped; normalize
            let g2 = -g;
            if g2 <= 0.0 {
                return None;
            }
            let (an, bn, cn) = (-a / g2, -b / g2, -c / g2);
            return Self::finish_geom(xc, yc, an, bn, cn);
        }
        let (an, bn, cn) = (a / g, b / g, c / g);
        Self::finish_geom(xc, yc, an, bn, cn)
    }

    fn finish_geom(xc: f64, yc: f64, an: f64, bn: f64, cn: f64) -> Option<EllipseGeom> {
        if an <= 0.0 {
            return None;
        }
        // Physical model: true circle of radius r seen through scan-axis
        // scale sx and slit shear k:  x = sx*(u + k*v), y = v.
        // Substituting into an x^2 + bn xy + cn y^2 = 1 and matching
        // (u^2+v^2)/r^2 gives closed forms:
        let t = cn - bn * bn / (4.0 * an);
        if t <= 0.0 {
            return None;
        }
        let sx2 = t / an;
        let sx = sx2.sqrt();
        let shear = -bn / (2.0 * an * sx);
        let radius = (1.0 / (an * sx2)).sqrt();
        Some(EllipseGeom { xc, yc, an, bn, cn, sx, shear, radius })
    }
}

/// Halir-Flusser direct least-squares ellipse fit (weighted).
pub fn fit_direct(points: &[EdgePoint]) -> Option<Conic> {
    let n = points.len();
    if n < 6 {
        return None;
    }
    // center & scale normalization for conditioning
    let mx = points.iter().map(|p| p.x).sum::<f64>() / n as f64;
    let my = points.iter().map(|p| p.y).sum::<f64>() / n as f64;
    let s = points
        .iter()
        .map(|p| ((p.x - mx).powi(2) + (p.y - my).powi(2)).sqrt())
        .sum::<f64>()
        / n as f64;
    let s = s.max(1e-9);

    let mut s1 = Matrix3::<f64>::zeros();
    let mut s2 = Matrix3::<f64>::zeros();
    let mut s3 = Matrix3::<f64>::zeros();
    for p in points {
        let w = p.weight.max(0.0);
        if w <= 0.0 {
            continue;
        }
        let x = (p.x - mx) / s;
        let y = (p.y - my) / s;
        let d1 = Vector3::new(x * x, x * y, y * y);
        let d2 = Vector3::new(x, y, 1.0);
        s1 += w * d1 * d1.transpose();
        s2 += w * d1 * d2.transpose();
        s3 += w * d2 * d2.transpose();
    }
    let s3_inv = s3.try_inverse()?;
    let t = -s3_inv * s2.transpose();
    let m = s1 + s2 * t;
    // reduced matrix: C1^{-1} * M with C1 = [[0,0,2],[0,-1,0],[2,0,0]]
    let mr = Matrix3::new(
        m[(2, 0)] / 2.0, m[(2, 1)] / 2.0, m[(2, 2)] / 2.0,
        -m[(1, 0)], -m[(1, 1)], -m[(1, 2)],
        m[(0, 0)] / 2.0, m[(0, 1)] / 2.0, m[(0, 2)] / 2.0,
    );
    // eigenvectors of 3x3 general matrix via characteristic cubic
    let eigvecs = eig3(&mr)?;
    let mut a1: Option<Vector3<f64>> = None;
    for v in eigvecs {
        let cond = 4.0 * v[0] * v[2] - v[1] * v[1];
        if cond > 0.0 {
            a1 = Some(v);
            break;
        }
    }
    let a1 = a1?;
    let a2 = t * a1;
    // denormalize conic from (x', y') = ((x-mx)/s, (y-my)/s)
    let (pa, pb, pc, pd, pe, pf) = (a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]);
    let s2f = s * s;
    let a = pa / s2f;
    let b = pb / s2f;
    let c = pc / s2f;
    let d = -2.0 * pa * mx / s2f - pb * my / s2f + pd / s;
    let e = -pb * mx / s2f - 2.0 * pc * my / s2f + pe / s;
    let f = pa * mx * mx / s2f + pb * mx * my / s2f + pc * my * my / s2f - pd * mx / s - pe * my / s + pf;
    Some(Conic { a, b, c, d, e, f })
}

/// All real eigenvectors of a general real 3x3 matrix.
fn eig3(m: &Matrix3<f64>) -> Option<Vec<Vector3<f64>>> {
    // characteristic polynomial: -l^3 + tr l^2 - m2 l + det = 0
    let tr = m.trace();
    let det = m.determinant();
    let m2 = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)]
        + m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)]
        + m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)];
    // l^3 - tr l^2 + m2 l - det = 0
    let roots = solve_cubic(1.0, -tr, m2, -det);
    let mut out = Vec::new();
    for l in roots {
        let a = m - Matrix3::identity() * l;
        // null vector via cross products of rows
        let r0 = Vector3::new(a[(0, 0)], a[(0, 1)], a[(0, 2)]);
        let r1 = Vector3::new(a[(1, 0)], a[(1, 1)], a[(1, 2)]);
        let r2 = Vector3::new(a[(2, 0)], a[(2, 1)], a[(2, 2)]);
        let c0 = r0.cross(&r1);
        let c1 = r0.cross(&r2);
        let c2 = r1.cross(&r2);
        let best = [c0, c1, c2]
            .into_iter()
            .max_by(|u, v| u.norm().partial_cmp(&v.norm()).unwrap())?;
        if best.norm() > 1e-12 {
            out.push(best / best.norm());
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Real roots of a x^3 + b x^2 + c x + d = 0 (trigonometric method).
fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
    let shift = -b / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    if disc > 1e-14 {
        // one real root
        let sq = disc.sqrt();
        let u = (-q / 2.0 + sq).cbrt();
        let v = (-q / 2.0 - sq).cbrt();
        vec![u + v + shift]
    } else {
        // three real roots
        let r = (-p * p * p / 27.0).sqrt().max(1e-300);
        let phi = (-q / (2.0 * r)).clamp(-1.0, 1.0).acos();
        let m = 2.0 * (-p / 3.0).max(0.0).sqrt();
        (0..3)
            .map(|k| m * ((phi + 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos() + shift)
            .collect()
    }
}

#[allow(dead_code)]
pub struct RansacResult {
    pub conic: Conic,
    pub geom: EllipseGeom,
    pub inliers: usize,
    pub total: usize,
    pub residual_rms: f64,
}

/// RANSAC + IRLS ellipse fit.
///
/// Samples are STRATIFIED over y so the 8 points span the disk instead of
/// clustering on one arc (a degenerate conic through a short arc can
/// otherwise win consensus on large images), the inlier tolerance scales
/// with the point-cloud extent (real limbs are spicule-fuzzy at the
/// multi-px scale on big disks), and candidates must be physically
/// plausible (radius/scale/shear gates).
pub fn fit_robust(points: &[EdgePoint], seed: u64) -> Option<RansacResult> {
    let n = points.len();
    if n < 12 {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(seed);

    // extent of the point cloud, for tolerance and plausibility gates
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in points {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }
    let extent = (xmax - xmin).max(ymax - ymin).max(1.0);
    let inlier_tol = (0.0015 * extent).clamp(1.2, 3.0);

    // stratify by y: sort indices, sample one point from each of 8 bands
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| points[a].y.partial_cmp(&points[b].y).unwrap());
    let band = (n / 8).max(1);

    let plausible = |g: &EllipseGeom| -> bool {
        g.radius > 0.10 * extent
            && g.radius < 1.5 * extent
            && g.sx > 0.15
            && g.sx < 8.0
            && g.shear.abs() < 0.35
    };

    let mut best_inliers: Vec<usize> = Vec::new();
    for _ in 0..600 {
        let sample: Vec<EdgePoint> = (0..8)
            .map(|k| {
                let lo = k * band;
                let hi = ((k + 1) * band).min(n);
                points[order[rng.gen_range(lo..hi.max(lo + 1))]]
            })
            .collect();
        let Some(conic) = fit_direct(&sample) else { continue };
        let Some(g) = conic.geometry() else { continue };
        if !plausible(&g) {
            continue;
        }
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| conic.sampson(points[i].x, points[i].y) < inlier_tol)
            .collect();
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }
    if best_inliers.len() < 12 {
        return None;
    }

    // final fit on inliers with Tukey IRLS on Sampson distance
    let mut pts: Vec<EdgePoint> = best_inliers.iter().map(|&i| points[i]).collect();
    let mut conic = fit_direct(&pts)?;
    for _ in 0..4 {
        let mut res: Vec<f64> = pts.iter().map(|p| conic.sampson(p.x, p.y)).collect();
        let mad = crate::mathutil::median_inplace(&mut res.clone()).max(1e-6);
        let cscale = 4.685 * 1.4826 * mad;
        for (p, r) in pts.iter_mut().zip(res.iter_mut()) {
            let u = *r / cscale;
            p.weight = if u.abs() < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
        }
        conic = fit_direct(&pts)?;
    }
    let geom = conic.geometry()?;
    let mut rs: Vec<f64> = pts.iter().map(|p| conic.sampson(p.x, p.y)).collect();
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rms = (rs.iter().map(|r| r * r).sum::<f64>() / rs.len() as f64).sqrt();
    Some(RansacResult {
        conic,
        geom,
        inliers: best_inliers.len(),
        total: n,
        residual_rms: rms,
    })
}
