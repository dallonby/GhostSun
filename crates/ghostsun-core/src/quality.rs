//! F11: temporal burst detection and repair.
//!
//! Frame index is time. A seeing burst degrades a RUN of consecutive frames
//! coherently (blur + displacement), visible as vertical smudges — worst at
//! the disk's left/right limbs where the registration estimators are gated
//! off. The scan is typically 2-3x oversampled, so each column is nearly
//! reconstructible from its temporal neighbors: a column that DISAGREES
//! with its own look-ahead/look-behind prediction has been hit by seeing,
//! and the prediction itself is the repair.
//!
//! Two passes: (1) predict every column from +-2/+-4 neighbors, flag columns
//! whose disagreement exceeds the local noise floor (with hysteresis so
//! bursts flag as runs); (2) rebuild predictions for flagged columns from
//! the nearest UNFLAGGED frames only (bursts can't contaminate their own
//! repair), and blend by disagreement severity.

use crate::image2d::Image;
use crate::mathutil::percentile_f32;
use rayon::prelude::*;

#[allow(dead_code)]
pub struct BurstRepair {
    pub flags: Vec<bool>,
    #[allow(dead_code)]
    pub severity: Vec<f64>, // normalized disagreement per column
    pub n_flagged: usize,
}

/// Lagrange interpolation of a column at position t from sample columns at
/// given positions (2-4 of them).
fn predict(cols: &[(f64, &[f32])], t: f64, h: usize) -> Vec<f64> {
    let n = cols.len();
    let mut w = vec![1.0f64; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                w[i] *= (t - cols[j].0) / (cols[i].0 - cols[j].0);
            }
        }
    }
    let mut out = vec![0.0f64; h];
    for (wi, (_, col)) in w.iter().zip(cols) {
        for y in 0..h {
            out[y] += wi * col[y] as f64;
        }
    }
    out
}

fn deriv(col: &[f32]) -> Vec<f64> {
    let n = col.len();
    (0..n)
        .map(|y| {
            let a = col[y.saturating_sub(1)] as f64;
            let b = col[(y + 1).min(n - 1)] as f64;
            (b - a) / 2.0
        })
        .collect()
}

/// Detect and repair burst-degraded columns in place (image + optional
/// companion maps like velocity, repaired with the same flags).
pub fn repair_bursts(
    disk: &mut Image,
    companions: &mut [&mut Image],
    burst_thresh: f64,
) -> BurstRepair {
    let w = disk.w;
    let h = disk.h;
    let thresh = percentile_f32(&disk.data, 80.0) * 0.20;
    let cols: Vec<Vec<f32>> = (0..w).map(|x| disk.column(x)).collect();

    // pass 1: CONTRAST COLLAPSE detection. After registration has removed
    // burst displacement, the surviving damage is blur — a drop in
    // derivative energy along the slit. Prediction-disagreement medians are
    // blind to it; the contrast ratio against the temporal neighborhood is
    // not.
    let d_raw: Vec<f64> = (0..w)
        .into_par_iter()
        .map(|x| {
            let col = &cols[x];
            let d = deriv(col);
            let mut vals: Vec<f64> = (0..h)
                .filter(|&y| col[y] > thresh)
                .map(|y| d[y].abs())
                .collect();
            if vals.len() < 24 {
                return f64::NAN;
            }
            crate::mathutil::median_inplace(&mut vals)
        })
        .collect();

    // normalize by the local (rolling median) noise floor
    let valid: Vec<usize> = (0..w).filter(|&x| d_raw[x].is_finite()).collect();
    if valid.len() < 100 {
        return BurstRepair { flags: vec![false; w], severity: vec![0.0; w], n_flagged: 0 };
    }
    let mut filled = d_raw.clone();
    for x in 0..w {
        if !filled[x].is_finite() {
            let nearest = valid.iter().min_by_key(|&&v| v.abs_diff(x)).unwrap();
            filled[x] = d_raw[*nearest];
        }
    }
    let floor = {
        let hw = 50usize;
        let mut out = vec![0.0; w];
        for x in 0..w {
            let lo = x.saturating_sub(hw);
            let hi = (x + hw + 1).min(w);
            let mut win: Vec<f64> = filled[lo..hi].to_vec();
            out[x] = crate::mathutil::median_inplace(&mut win).max(1e-9);
        }
        out
    };
    // severity = inverse contrast ratio (1 = normal, >1 = blurred)
    let sev: Vec<f64> = (0..w).map(|x| (floor[x] / filled[x].max(1e-9)).max(0.0)).collect();

    // flags with hysteresis: strong columns seed, adjacent moderately
    // elevated columns join (temporal persistence of bursts)
    let hi = burst_thresh;
    let lo = 1.0 + 0.6 * (burst_thresh - 1.0);
    let mut flags = vec![false; w];
    for x in 0..w {
        if d_raw[x].is_finite() && sev[x] > hi {
            flags[x] = true;
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for x in 1..w - 1 {
            if !flags[x] && sev[x] > lo && (flags[x - 1] || flags[x + 1]) {
                flags[x] = true;
                changed = true;
            }
        }
    }

    // verification: re-predict each flagged column from its nearest GOOD
    // frames; keep the flag only if the disagreement survives. Kills the
    // persistent limb false-positives (far prediction is intrinsically poor
    // where the chord changes fast; near-good prediction is not).
    let flags_initial = flags.clone();
    let verified: Vec<bool> = (0..w)
        .into_par_iter()
        .map(|x| {
            if !flags_initial[x] {
                return false;
            }
            let mut samples: Vec<(f64, &[f32])> = Vec::new();
            let mut k = 1i64;
            let mut left = 0;
            let mut right = 0;
            while k <= 15 && (left < 2 || right < 2) {
                let xl = x as i64 - k;
                if left < 2 && xl >= 0 && !flags_initial[xl as usize] {
                    samples.push((-(k as f64), &cols[xl as usize][..]));
                    left += 1;
                }
                let xr = x as i64 + k;
                if right < 2 && (xr as usize) < w && !flags_initial[xr as usize] {
                    samples.push((k as f64, &cols[xr as usize][..]));
                    right += 1;
                }
                k += 1;
            }
            if samples.len() < 3 {
                return false;
            }
            let pred = predict(&samples, 0.0, h);
            // verified if this column's contrast is well below what its
            // good-neighbor prediction carries
            let dcol = deriv(&cols[x]);
            let dpred: Vec<f64> = (0..h)
                .map(|y| (pred[(y + 1).min(h - 1)] - pred[y.saturating_sub(1)]) / 2.0)
                .collect();
            let mut cv: Vec<f64> = Vec::new();
            let mut pv: Vec<f64> = Vec::new();
            for y in 0..h {
                if cols[x][y] > thresh {
                    cv.push(dcol[y].abs());
                    pv.push(dpred[y].abs());
                }
            }
            if cv.len() < 24 {
                return false;
            }
            let mc = crate::mathutil::median_inplace(&mut cv);
            let mp = crate::mathutil::median_inplace(&mut pv);
            // prediction averages 2+ frames, which itself lowers contrast a
            // bit; require the column to be clearly below even that
            mp > 1e-9 && mc / mp < 1.0 / (0.75 + 0.25 * burst_thresh)
        })
        .collect();
    let flags = verified;

    // pass 2: repair flagged columns from the nearest UNFLAGGED neighbors
    let n_flagged = flags.iter().filter(|&&f| f).count();
    if n_flagged == 0 || n_flagged > w / 4 {
        // >25% flagged means the detector is confused — do nothing
        return BurstRepair { flags: vec![false; w], severity: sev, n_flagged: 0 };
    }
    let repaired: Vec<Option<Vec<f64>>> = (0..w)
        .into_par_iter()
        .map(|x| {
            if !flags[x] {
                return None;
            }
            // nearest 2 good frames on each side within +-15
            let mut samples: Vec<(f64, &[f32])> = Vec::new();
            let mut k = 1i64;
            let mut left = 0;
            let mut right = 0;
            while k <= 15 && (left < 2 || right < 2) {
                let xl = x as i64 - k;
                if left < 2 && xl >= 0 && !flags[xl as usize] {
                    samples.push((-(k as f64), &cols[xl as usize][..]));
                    left += 1;
                }
                let xr = x as i64 + k;
                if right < 2 && (xr as usize) < w && !flags[xr as usize] {
                    samples.push((k as f64, &cols[xr as usize][..]));
                    right += 1;
                }
                k += 1;
            }
            if samples.len() < 2 {
                return None;
            }
            Some(predict(&samples, 0.0, h))
        })
        .collect();

    for x in 0..w {
        if let Some(pred) = &repaired[x] {
            // blend by severity: fully replace well above threshold
            let keep = ((hi + 0.8 - sev[x]) / 0.8).clamp(0.0, 1.0) * 0.5;
            for y in 0..h {
                let v = keep * cols[x][y] as f64 + (1.0 - keep) * pred[y];
                disk.set(x, y, v.max(0.0) as f32);
            }
        }
    }
    // companion maps (velocity): same flags, nearest-good linear blend
    for comp in companions.iter_mut() {
        let ccols: Vec<Vec<f32>> = (0..w).map(|x| comp.column(x)).collect();
        for x in 0..w {
            if repaired[x].is_none() {
                continue;
            }
            // nearest good on each side
            let mut xl = x as i64 - 1;
            while xl >= 0 && flags[xl as usize] {
                xl -= 1;
            }
            let mut xr = x as i64 + 1;
            while (xr as usize) < w && flags[xr as usize] {
                xr += 1;
            }
            if xl < 0 || xr as usize >= w {
                continue;
            }
            let t = (x as f64 - xl as f64) / (xr as f64 - xl as f64);
            for y in 0..h {
                let v = (1.0 - t) * ccols[xl as usize][y] as f64 + t * ccols[xr as usize][y] as f64;
                comp.set(x, y, v as f32);
            }
        }
    }

    BurstRepair { flags, severity: sev, n_flagged }
}


// ---------------------------------------------------------------------------
// F11.5: temporal non-local-means smoothing.
//
// The scan oversamples: neighboring frames are near-duplicate observations
// of the same solar strip under independent seeing. Averaging each pixel
// with its temporal neighbors, weighted by vertical-patch similarity,
// suppresses per-frame seeing noise (the "jaggies") while real structure
// (limb, fibrils) protects itself: where content genuinely differs, the
// patch distance collapses the weight. The noise scale is estimated from
// the data (median adjacent-column difference), so on clean data the filter
// self-attenuates.
// ---------------------------------------------------------------------------

/// Shared parameter derivation for CPU and GPU NLM implementations.
pub fn nlm_params(disk: &Image, h_factor: f64) -> Option<(f64, f64, f32)> {
    let w = disk.w;
    let hgt = disk.h;
    let thresh = percentile_f32(&disk.data, 80.0) * 0.15;
    let mut diffs: Vec<f64> = Vec::new();
    let mut x = 8;
    while x + 1 < w && diffs.len() < 200_000 {
        for y in (0..hgt).step_by(3) {
            let a = disk.at(x, y);
            if a > thresh {
                diffs.push((a as f64 - disk.at(x + 1, y) as f64).abs());
            }
        }
        x += 13;
    }
    if diffs.len() < 1000 {
        return None;
    }
    let sigma = crate::mathutil::median_inplace(&mut diffs) * 1.4826 / std::f64::consts::SQRT_2;
    let h2 = (h_factor * sigma).powi(2).max(1e-12);
    Some((sigma, h2, thresh))
}

pub fn temporal_nlm(disk: &Image, radius: usize, h_factor: f64) -> Image {
    let w = disk.w;
    let hgt = disk.h;
    let cols: Vec<Vec<f32>> = (0..w).map(|x| disk.column(x)).collect();
    let Some((sigma, h2, thresh)) = nlm_params(disk, h_factor) else {
        return disk.clone();
    };
    const P: isize = 3; // vertical patch half-size
    let pn = (2 * P + 1) as f64;

    let out_cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|t| {
            let mut out = cols[t].clone();
            let lo = t.saturating_sub(radius);
            let hi = (t + radius + 1).min(w);
            for y in 0..hgt {
                if cols[t][y] <= thresh {
                    continue; // leave sky/prominence-faint pixels untouched
                }
                let mut acc = cols[t][y] as f64;
                let mut wsum = 1.0;
                for tt in lo..hi {
                    if tt == t {
                        continue;
                    }
                    // vertical patch distance
                    let mut d2 = 0.0;
                    for p in -P..=P {
                        let yy = (y as isize + p).clamp(0, hgt as isize - 1) as usize;
                        let d = cols[t][yy] as f64 - cols[tt][yy] as f64;
                        d2 += d * d;
                    }
                    d2 /= pn;
                    // subtract the noise contribution so equal-content
                    // patches (differing only by noise) get full weight
                    let excess = (d2 - 2.0 * sigma * sigma).max(0.0);
                    let wgt = (-excess / h2).exp();
                    acc += wgt * cols[tt][y] as f64;
                    wsum += wgt;
                }
                out[y] = (acc / wsum) as f32;
            }
            out
        })
        .collect();

    let mut out = Image::new(w, hgt);
    for (t, col) in out_cols.iter().enumerate() {
        out.set_column(t, col);
    }
    out
}
