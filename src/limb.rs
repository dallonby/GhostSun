//! Sub-pixel solar limb detection.
//!
//! For each image row we locate the rising and falling limb crossings from
//! the gradient of a lightly smoothed profile, then refine each edge to
//! sub-pixel accuracy with a gradient-centroid estimator (the gradient of an
//! erf-shaped limb is Gaussian; its centroid is the edge location). Each
//! point carries a confidence weight from edge amplitude.
//!
//! INTI instead takes whole-pixel gradient argmax, filters the points with
//! ad-hoc cascades, and then *replaces them with a 6th-degree polynomial*
//! before ellipse fitting — biasing the geometry.

use crate::image2d::Image;
use crate::mathutil::{gaussian_smooth, percentile_f32};

#[derive(Clone, Copy, Debug)]
pub struct EdgePoint {
    pub x: f64,
    pub y: f64,
    pub weight: f64,
}

/// Refine an edge position by centroid of |gradient| in a +/- w window.
fn refine_edge(grad: &[f64], x0: usize, w: usize, rising: bool) -> Option<(f64, f64)> {
    let n = grad.len();
    let a = x0.saturating_sub(w).max(1);
    let b = (x0 + w + 1).min(n - 1);
    let mut sw = 0.0;
    let mut swx = 0.0;
    let mut amp: f64 = 0.0;
    for x in a..b {
        let g = if rising { grad[x].max(0.0) } else { (-grad[x]).max(0.0) };
        sw += g;
        swx += g * x as f64;
        amp = amp.max(g);
    }
    if sw <= 1e-9 {
        return None;
    }
    Some((swx / sw, amp))
}

/// Detect limb points on both sides for every usable row.
pub fn detect_limb_points(disk: &Image) -> Vec<EdgePoint> {
    let h = disk.h;
    let w = disk.w;
    let mut points = Vec::new();

    // clamp bright plage so it does not dominate gradients (like INTI),
    // but based on robust statistics
    let p95 = percentile_f32(&disk.data, 95.0) as f64;
    let clip = p95 * 1.05;

    let mut chords: Vec<(usize, f64, f64, f64, f64)> = Vec::new(); // y, x1, x2, amp1, amp2

    for y in 0..h {
        let row: Vec<f64> = disk.row(y).iter().map(|&v| (v as f64).min(clip)).collect();
        let sm = gaussian_smooth(&row, 3.0);
        let grad: Vec<f64> = (0..w)
            .map(|x| {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                (sm[xp] - sm[xm]) / 2.0
            })
            .collect();
        // coarse: strongest rising and falling gradient
        let mut imax = 0;
        let mut imin = 0;
        let mut vmax = f64::MIN;
        let mut vmin = f64::MAX;
        for x in 3..w - 3 {
            if grad[x] > vmax {
                vmax = grad[x];
                imax = x;
            }
            if grad[x] < vmin {
                vmin = grad[x];
                imin = x;
            }
        }
        if imin <= imax + 25 {
            continue; // chord too short or edges merged: near-tangent row
        }
        let Some((x1, a1)) = refine_edge(&grad, imax, 6, true) else { continue };
        let Some((x2, a2)) = refine_edge(&grad, imin, 6, false) else { continue };
        chords.push((y, x1, x2, a1, a2));
    }

    if chords.is_empty() {
        return points;
    }
    // normalize weights by median edge amplitude; drop feeble edges
    let mut amps: Vec<f32> = chords.iter().flat_map(|c| [c.3 as f32, c.4 as f32]).collect();
    amps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_amp = amps[amps.len() / 2] as f64;
    for (y, x1, x2, a1, a2) in chords {
        if a1 > 0.2 * med_amp {
            points.push(EdgePoint { x: x1, y: y as f64, weight: (a1 / med_amp).min(2.0) });
        }
        if a2 > 0.2 * med_amp {
            points.push(EdgePoint { x: x2, y: y as f64, weight: (a2 / med_amp).min(2.0) });
        }
    }
    points
}

/// Baseline (INTI-like) edge detection: whole-pixel gradient extrema of a
/// heavily smoothed profile, no sub-pixel refinement.
pub fn detect_limb_points_baseline(disk: &Image) -> Vec<EdgePoint> {
    let h = disk.h;
    let w = disk.w;
    let mut pts = Vec::new();
    let p97 = percentile_f32(&disk.data, 97.0) as f64;
    let bb = p97 * 0.7;
    for y in 0..h {
        let row: Vec<f64> = disk.row(y).iter().map(|&v| (v as f64).min(bb)).collect();
        let sm = gaussian_smooth(&row, 11.0);
        let grad: Vec<f64> = (0..w)
            .map(|x| {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                (sm[xp] - sm[xm]) / 2.0
            })
            .collect();
        let mut imax = 0;
        let mut imin = 0;
        let mut vmax = f64::MIN;
        let mut vmin = f64::MAX;
        for x in 1..w - 1 {
            if grad[x] > vmax {
                vmax = grad[x];
                imax = x;
            }
            if grad[x] < vmin {
                vmin = grad[x];
                imin = x;
            }
        }
        if imin <= imax + 25 || imax == 0 || imin == 0 {
            continue;
        }
        pts.push(EdgePoint { x: imax as f64, y: y as f64, weight: 1.0 });
        pts.push(EdgePoint { x: imin as f64, y: y as f64, weight: 1.0 });
    }
    // INTI: replace measured edges with a 6th-degree polynomial fit per side
    let mut out = Vec::new();
    for side in 0..2 {
        let side_pts: Vec<&EdgePoint> = pts.iter().skip(side).step_by(2).collect();
        if side_pts.len() < 10 {
            continue;
        }
        let xs: Vec<f64> = side_pts.iter().map(|p| p.y).collect();
        let ys: Vec<f64> = side_pts.iter().map(|p| p.x).collect();
        let ws = vec![1.0; xs.len()];
        if let Some(c) = crate::mathutil::polyfit_weighted(&xs, &ys, &ws, 6) {
            for &y in &xs {
                let fitted = crate::mathutil::polyval(&c, y);
                if fitted >= 0.0 {
                    out.push(EdgePoint { x: fitted, y, weight: 1.0 });
                }
            }
        }
    }
    out
}
