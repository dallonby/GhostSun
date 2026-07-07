//! Scan-jitter and drift correction along the slit axis.
//!
//! Seeing and mount wobble shift each frame along the slit. INTI stacks raw
//! columns, producing ragged limbs and horizontally shredded fine structure.
//!
//! Two estimators, both absent from INTI:
//!
//! 1. Fast jitter: normalized cross-correlation of *derivative* column
//!    profiles (the derivative kills the smooth limb-darkening ramp that
//!    otherwise flattens the correlation peak) between each column and a
//!    local reference, with parabolic sub-pixel refinement, integrated and
//!    high-passed so real geometry is untouched.
//!
//! 2. Slow drift: for a circle under shear+scale, the midpoint of each
//!    vertical chord lies exactly on a straight line in x. Sub-pixel
//!    top/bottom limb crossings give midchord(x); the smoothed residual from
//!    a robust line fit is vertical drift the ellipse model cannot absorb.

use crate::image2d::Image;
use crate::mathutil::{bspline_eval, bspline_prefilter, percentile_f32, robust_trend};
use rayon::prelude::*;

pub struct JitterResult {
    pub corrected: Image,
    pub trajectory: Vec<f64>, // applied correction per column (px)
}

const MAX_SHIFT: isize = 4;

/// derivative of a column, lightly smoothed
fn column_derivative(col: &[f32]) -> Vec<f64> {
    let n = col.len();
    let mut d = vec![0.0f64; n];
    for y in 1..n - 1 {
        d[y] = (col[y + 1] as f64 - col[y - 1] as f64) / 2.0;
    }
    // 3-tap smoothing to tame photon noise
    let mut out = vec![0.0f64; n];
    for y in 1..n - 1 {
        out[y] = 0.25 * d[y - 1] + 0.5 * d[y] + 0.25 * d[y + 1];
    }
    out
}

fn ncc_lag(a: &[f64], b: &[f64], k: isize) -> f64 {
    let n = a.len() as isize;
    let mut sa = 0.0;
    let mut sb = 0.0;
    let mut cnt = 0.0;
    for y in 0..n {
        let yb = y + k;
        if yb < 0 || yb >= n {
            continue;
        }
        sa += a[y as usize];
        sb += b[yb as usize];
        cnt += 1.0;
    }
    if cnt < 32.0 {
        return f64::NEG_INFINITY;
    }
    let (ma, mb) = (sa / cnt, sb / cnt);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for y in 0..n {
        let yb = y + k;
        if yb < 0 || yb >= n {
            continue;
        }
        let va = a[y as usize] - ma;
        let vb = b[yb as usize] - mb;
        num += va * vb;
        da += va * va;
        db += vb * vb;
    }
    if da <= 1e-12 || db <= 1e-12 {
        return f64::NEG_INFINITY;
    }
    num / (da * db).sqrt()
}

/// Sub-pixel displacement of b relative to a (b(y) ~ a(y - d)), from
/// derivative profiles, plus a confidence in (0,1]. None if unreliable.
fn register(a: &[f64], b: &[f64]) -> Option<(f64, f64)> {
    let mut scores = [f64::NEG_INFINITY; (2 * MAX_SHIFT + 1) as usize];
    let mut best_k = 0isize;
    let mut best_v = f64::NEG_INFINITY;
    for k in -MAX_SHIFT..=MAX_SHIFT {
        let v = ncc_lag(a, b, k);
        scores[(k + MAX_SHIFT) as usize] = v;
        if v > best_v {
            best_v = v;
            best_k = k;
        }
    }
    if best_v < 0.4 || best_k.abs() == MAX_SHIFT {
        return None;
    }
    let i = (best_k + MAX_SHIFT) as usize;
    let (vm, v0, vp) = (scores[i - 1], scores[i], scores[i + 1]);
    let denom = vm - 2.0 * v0 + vp;
    let frac = if denom < -1e-12 { (0.5 * (vm - vp) / denom).clamp(-0.75, 0.75) } else { 0.0 };
    // sharper, higher peaks earn more weight in the trajectory solve
    let conf = (best_v.max(0.0)) * (-denom).min(0.5).max(0.02);
    Some((best_k as f64 + frac, conf))
}

fn shift_column(col: &[f32], c: f64) -> Vec<f32> {
    if c.abs() < 0.04 {
        // deadband: don't resample for shifts below the estimator noise floor
        return col.to_vec();
    }
    let h = col.len();
    let mut coef: Vec<f64> = col.iter().map(|&v| v as f64).collect();
    bspline_prefilter(&mut coef);
    (0..h)
        .map(|y| {
            let pos = (y as f64 + c).clamp(0.0, (h - 1) as f64);
            bspline_eval(&coef, pos) as f32
        })
        .collect()
}

pub fn apply_shifts(disk: &Image, shifts: &[f64]) -> Image {
    let cols: Vec<Vec<f32>> = (0..disk.w)
        .into_par_iter()
        .map(|x| shift_column(&disk.column(x), shifts[x]))
        .collect();
    let mut out = Image::new(disk.w, disk.h);
    for (x, col) in cols.iter().enumerate() {
        out.set_column(x, col);
    }
    out
}

/// First/last row above threshold: the vertical disk span of a column.
fn interior_span(col: &[f32], thresh: f32) -> Option<(usize, usize)> {
    let mut lo = None;
    let mut hi = None;
    for (y, &v) in col.iter().enumerate() {
        if v > thresh {
            if lo.is_none() {
                lo = Some(y);
            }
            hi = Some(y);
        }
    }
    match (lo, hi) {
        (Some(a), Some(b)) if b > a => Some((a, b)),
        _ => None,
    }
}

/// Fast jitter pass: pairwise derivative registration, integration,
/// high-pass, then a refinement pass against a 9-column local reference.
///
/// Registration deliberately EXCLUDES the limb rows: as the scan crosses the
/// disk, the top and bottom limb edges move in opposite directions (chord
/// growth/shrinkage) — a single correlation lag cannot represent that, and
/// the huge edge gradients would otherwise dominate and bias the estimate.
/// Only interior chromospheric texture carries clean shift information.
pub fn correct_jitter(disk: &Image, hp_window: usize) -> JitterResult {
    let w = disk.w;
    let thresh = percentile_f32(&disk.data, 75.0) * 0.25;

    // chord length per column; trust only the disk interior where adjacent
    // columns see nearly the same content (near the x-limbs the chord shrinks
    // so fast that registration locks onto the moving limb edges instead)
    let spans: Vec<Option<(usize, usize)>> = (0..w)
        .map(|x| interior_span(&disk.column(x), thresh))
        .collect();
    let chord: Vec<usize> = spans.iter().map(|s| s.map(|(a, b)| b - a).unwrap_or(0)).collect();
    let max_chord = *chord.iter().max().unwrap_or(&0);
    let has_signal: Vec<bool> = (0..w)
        .map(|x| chord[x] > (max_chord as f64 * 0.55) as usize && chord[x] > 60)
        .collect();
    // margin keeping the limb-edge gradient spikes out of the correlation
    let edge_margin = 12usize;
    let derivs: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| column_derivative(&disk.column(x)))
        .collect();

    // pass 1: multi-baseline shift graph + weighted least-squares trajectory.
    // Plain pairwise integration accumulates a random walk (sqrt-N growth);
    // constraints at lags 1..16 pin the trajectory at all scales so the
    // error grows only logarithmically.
    //
    // The trajectory is estimated TWICE, from the top and bottom halves of
    // each column's interior: real slit-direction jitter is common to both,
    // estimator noise is not. Their agreement calibrates a Wiener shrinkage
    // so that on data where the estimator noise rivals the true jitter
    // (real scans with strong texture evolution) the correction attenuates
    // itself instead of injecting combing noise.
    let solve_half = |half: u8| -> Vec<f64> {
        let lags: [usize; 5] = [1, 2, 4, 8, 16];
        let mut constraints: Vec<(usize, usize, f64, f64)> = Vec::new();
        for &lag in &lags {
            let cs: Vec<Option<(usize, usize, f64, f64)>> = (0..w.saturating_sub(lag))
                .into_par_iter()
                .map(|x| {
                    let x2 = x + lag;
                    if !(has_signal[x] && has_signal[x2]) {
                        return None;
                    }
                    let (a1, b1) = spans[x]?;
                    let (a2, b2) = spans[x2]?;
                    let mut lo = a1.max(a2) + edge_margin;
                    let mut hi = (b1.min(b2)).saturating_sub(edge_margin);
                    let mid = (lo + hi) / 2;
                    match half {
                        1 => hi = mid,
                        2 => lo = mid,
                        _ => {}
                    }
                    if hi <= lo + 40 {
                        return None;
                    }
                    let (d, conf) = register(&derivs[x][lo..hi], &derivs[x2][lo..hi])?;
                    if d.abs() > 2.5 {
                        return None;
                    }
                    Some((x, x2, d, conf))
                })
                .collect();
            constraints.extend(cs.into_iter().flatten());
        }
        // weighted Gauss-Seidel solve of J(x2) - J(x1) = d
        let mut traj = vec![0.0f64; w];
        let mut num = vec![0.0f64; w];
        let mut den = vec![0.0f64; w];
        for _ in 0..120 {
            num.iter_mut().for_each(|v| *v = 0.0);
            den.iter_mut().for_each(|v| *v = 0.0);
            for &(x1, x2, d, cw) in &constraints {
                num[x2] += cw * (traj[x1] + d);
                den[x2] += cw;
                num[x1] += cw * (traj[x2] - d);
                den[x1] += cw;
            }
            for x in 0..w {
                if den[x] > 1e-9 {
                    traj[x] = 0.5 * traj[x] + 0.5 * num[x] / den[x];
                }
            }
        }
        let trend = robust_trend(&traj, hp_window | 1, hp_window as f64 / 2.0);
        traj.iter().zip(&trend).map(|(t, s)| t - s).collect()
    };
    let c_top = solve_half(1);
    let c_bot = solve_half(2);
    // Wiener shrinkage from split-half agreement (trusted columns only)
    let trusted: Vec<usize> = (0..w).filter(|&x| has_signal[x]).collect();
    let mut shrink = 1.0;
    if trusted.len() > 50 {
        let mean = |v: &[f64]| -> f64 {
            trusted.iter().map(|&x| v[x]).sum::<f64>() / trusted.len() as f64
        };
        let avg: Vec<f64> = (0..w).map(|x| 0.5 * (c_top[x] + c_bot[x])).collect();
        let ma = mean(&avg);
        let var_avg = trusted.iter().map(|&x| (avg[x] - ma).powi(2)).sum::<f64>() / trusted.len() as f64;
        // noise of the average = Var(top - bot) / 4
        let var_noise = trusted
            .iter()
            .map(|&x| (c_top[x] - c_bot[x]).powi(2))
            .sum::<f64>()
            / trusted.len() as f64
            / 4.0;
        shrink = (1.0 - var_noise / var_avg.max(1e-12)).clamp(0.0, 1.0);
    }
    let mut correction: Vec<f64> = (0..w).map(|x| 0.5 * (c_top[x] + c_bot[x]) * shrink).collect();
    for x in 0..w {
        if !has_signal[x] {
            correction[x] = 0.0;
        }
    }

    let pass1 = apply_shifts(disk, &correction);

    // pass 2: register each column against the mean of its corrected
    // neighbors (excluding itself) — kills integration random-walk noise
    let derivs2: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| column_derivative(&pass1.column(x)))
        .collect();
    let h = disk.h;
    let refine: Vec<f64> = (0..w)
        .into_par_iter()
        .map(|x| {
            if !has_signal[x] {
                return 0.0;
            }
            let Some((sa, sb)) = spans[x] else { return 0.0 };
            let lo = sa + edge_margin;
            let hi = sb.saturating_sub(edge_margin);
            if hi <= lo + 60 {
                return 0.0;
            }
            let mut refprof = vec![0.0f64; h];
            let mut cnt = 0.0;
            for dx in -4i64..=4 {
                if dx == 0 {
                    continue;
                }
                let xx = x as i64 + dx;
                if xx < 0 || xx >= w as i64 || !has_signal[xx as usize] {
                    continue;
                }
                for y in 0..h {
                    refprof[y] += derivs2[xx as usize][y];
                }
                cnt += 1.0;
            }
            if cnt < 3.0 {
                return 0.0;
            }
            for v in refprof.iter_mut() {
                *v /= cnt;
            }
            // displacement of this column vs local reference (interior rows)
            register(&refprof[lo..hi], &derivs2[x][lo..hi])
                .map(|(d, _)| d.clamp(-1.5, 1.5))
                .unwrap_or(0.0)
        })
        .collect();
    // high-pass the refinement too (protect real geometry)
    let rtrend = robust_trend(&refine, hp_window | 1, hp_window as f64 / 2.0);
    let mut refine_hp: Vec<f64> = refine.iter().zip(&rtrend).map(|(r, s)| r - s).collect();
    for x in 0..w {
        if !has_signal[x] {
            refine_hp[x] = 0.0;
        }
    }

    let corrected = apply_shifts(&pass1, &refine_hp);
    for x in 0..w {
        correction[x] += refine_hp[x];
    }
    JitterResult { corrected, trajectory: correction }
}

/// Slow drift pass: midchord straight-line residual.
/// Returns the corrected image and the applied per-column shift.
pub fn correct_drift(disk: &Image) -> JitterResult {
    let w = disk.w;
    let thresh = percentile_f32(&disk.data, 75.0) * 0.35;
    let chords: Vec<usize> = (0..w)
        .map(|x| disk.column(x).iter().filter(|&&v| v > thresh).count())
        .collect();
    let max_chord = *chords.iter().max().unwrap_or(&0);
    let min_chord = ((max_chord as f64) * 0.45) as usize;

    // per-column top/bottom limb crossings (gradient centroid, sub-pixel)
    let mid: Vec<Option<(f64, f64)>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let col = disk.column(x);
            let n = col.len();
            // near-tangent columns: limb nearly horizontal, edge too weak
            if chords[x] < min_chord || chords[x] < 80 {
                return None;
            }
            let sm: Vec<f64> = {
                let raw: Vec<f64> = col.iter().map(|&v| v as f64).collect();
                crate::mathutil::gaussian_smooth(&raw, 2.0)
            };
            let grad: Vec<f64> = (0..n)
                .map(|y| {
                    let ym = y.saturating_sub(1);
                    let yp = (y + 1).min(n - 1);
                    (sm[yp] - sm[ym]) / 2.0
                })
                .collect();
            let mut imax = 0;
            let mut imin = 0;
            let mut vmax = f64::MIN;
            let mut vmin = f64::MAX;
            for y in 2..n - 2 {
                if grad[y] > vmax {
                    vmax = grad[y];
                    imax = y;
                }
                if grad[y] < vmin {
                    vmin = grad[y];
                    imin = y;
                }
            }
            if imin <= imax + 40 {
                return None;
            }
            let centroid = |x0: usize, rising: bool| -> Option<f64> {
                let a = x0.saturating_sub(5).max(1);
                let b = (x0 + 6).min(n - 1);
                let mut sw = 0.0;
                let mut swy = 0.0;
                for y in a..b {
                    let g = if rising { grad[y].max(0.0) } else { (-grad[y]).max(0.0) };
                    sw += g;
                    swy += g * y as f64;
                }
                if sw > 1e-9 {
                    Some(swy / sw)
                } else {
                    None
                }
            };
            let top = centroid(imax, true)?;
            let bot = centroid(imin, false)?;
            Some((x as f64, (top + bot) / 2.0))
        })
        .collect();

    let pts: Vec<(f64, f64)> = mid.into_iter().flatten().collect();
    if pts.len() < 60 {
        return JitterResult { corrected: disk.clone(), trajectory: vec![0.0; w] };
    }
    let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
    let ws = vec![1.0; xs.len()];
    // robust line: midchords of a sheared/scaled circle are exactly linear in x
    let Some(line) = crate::mathutil::polyfit_robust(&xs, &ys, &ws, 1, 4) else {
        return JitterResult { corrected: disk.clone(), trajectory: vec![0.0; w] };
    };
    // Hard-drop absurd midchord points BEFORE any trend estimation: a
    // contiguous run of bad edges (prominence, frame border) otherwise
    // becomes the trend itself and the correction runs away (observed
    // 814 px on real data). Sane drift is tens of px at most.
    let keep: Vec<usize> = (0..pts.len())
        .filter(|&i| (pts[i].1 - crate::mathutil::polyval(&line, pts[i].0)).abs() < 20.0)
        .collect();
    if keep.len() < 60 {
        return JitterResult { corrected: disk.clone(), trajectory: vec![0.0; w] };
    }
    let pts: Vec<(f64, f64)> = keep.iter().map(|&i| pts[i]).collect();
    let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    // refit the line on the surviving points
    let ys2: Vec<f64> = pts.iter().map(|p| p.1).collect();
    let ws2 = vec![1.0; xs.len()];
    let line = crate::mathutil::polyfit_robust(&xs, &ys2, &ws2, 1, 4).unwrap_or(line);
    // residuals -> Tukey-clip against a robust local trend (prominences and
    // active regions bias individual edge centroids), then smooth
    let res_raw: Vec<f64> = pts.iter().map(|p| p.1 - crate::mathutil::polyval(&line, p.0)).collect();
    let trend0 = crate::mathutil::robust_loess_quadratic(&res_raw, 41, 3);
    let mut dev: Vec<f64> = res_raw.iter().zip(&trend0).map(|(r, t)| (r - t).abs()).collect();
    let mad = crate::mathutil::median_inplace(&mut dev).max(1e-6);
    let res: Vec<f64> = res_raw
        .iter()
        .zip(&trend0)
        .map(|(r, t)| if (r - t).abs() > 3.0 * 1.4826 * mad { *t } else { *r })
        .collect();
    let res_sm = robust_trend(&res, 31, 12.0);
    // interpolate over all columns (nearest for gaps, 0 outside disk)
    let mut drift = vec![0.0f64; w];
    for x in 0..w {
        let xf = x as f64;
        // find bracketing sample indices
        match xs.binary_search_by(|v| v.partial_cmp(&xf).unwrap()) {
            Ok(i) => drift[x] = res_sm[i],
            Err(i) => {
                if i == 0 || i >= xs.len() {
                    drift[x] = 0.0; // off-disk: no correction
                } else {
                    let t = (xf - xs[i - 1]) / (xs[i] - xs[i - 1]).max(1e-9);
                    drift[x] = res_sm[i - 1] * (1.0 - t) + res_sm[i] * t;
                }
            }
        }
    }
    // F9.1: no magnitude clamp — the anchors are absolute (midchords of a
    // sheared circle are exactly collinear) and Tukey-robustified above, so
    // a clamp only truncates real mount drift (observed saturating at 3 and
    // then 6 px on real scans). Edge taper still protects tangent columns.
    let x_lo = xs[0];
    let x_hi = *xs.last().unwrap();
    let taper_w = 40.0;
    let corrections: Vec<f64> = drift
        .iter()
        .enumerate()
        .map(|(x, d)| {
            let xf = x as f64;
            let edge = ((xf - x_lo) / taper_w).min((x_hi - xf) / taper_w).clamp(0.0, 1.0);
            // generous sanity bound: real mount drift is < ~10 px per scan
            d.clamp(-12.0, 12.0) * edge
        })
        .collect();
    let corrected = apply_shifts(disk, &corrections);
    JitterResult { corrected, trajectory: corrections }
}


// ---------------------------------------------------------------------------
// F9.2: per-frame registration ALONG the scan (x) direction.
//
// Seeing displaces each frame along the scan too: the slit samples a sun
// strip up to ~1 px from its nominal position, scrambling column order in a
// way no y-correction can fix. Estimate: column t should be the linear
// blend of its lag-L neighbors; the best blend coefficient alpha gives the
// sub-frame position offset delta = L*(1 - 2*alpha). Confidence-weighted
// over lags {1,2}, Tukey-cleaned, split-half Wiener-shrunk (top vs bottom
// slit halves), robust-line removed (a linear trend is degenerate with the
// scan-scale sx). Applied by resampling every row with B-splines.
// ---------------------------------------------------------------------------

pub struct XRegResult {
    pub corrected: Image,
    pub delta: Vec<f64>, // applied x offset per column (frames)
}

/// Sub-frame x offset of `cur` relative to the grid of its four neighbors
/// at +-L and +-2L, via SSE minimization over Lagrange-4 interpolated
/// references (exact for cubic texture evolution — the 2-point linear blend
/// has model bias that rivals the signal). Offset returned in frames.
fn lagrange_delta(
    pm2: &[f64],
    pm1: &[f64],
    pp1: &[f64],
    pp2: &[f64],
    cur: &[f64],
    lag: f64,
) -> Option<(f64, f64)> {
    // Lagrange weights for position u (in units of L) on nodes -2,-1,1,2
    let weights = |u: f64| -> [f64; 4] {
        let n = [-2.0f64, -1.0, 1.0, 2.0];
        let mut w = [0.0f64; 4];
        for i in 0..4 {
            let mut p = 1.0;
            for j in 0..4 {
                if i != j {
                    p *= (u - n[j]) / (n[i] - n[j]);
                }
            }
            w[i] = p;
        }
        w
    };
    let sse = |u: f64| -> f64 {
        let w = weights(u);
        let mut s = 0.0;
        for i in 0..cur.len() {
            let r = cur[i] - (w[0] * pm2[i] + w[1] * pm1[i] + w[2] * pp1[i] + w[3] * pp2[i]);
            s += r * r;
        }
        s
    };
    // scan u in [-0.6, 0.6], parabola refine
    let steps = 7;
    let mut best = (0usize, f64::MAX);
    let us: Vec<f64> = (0..steps).map(|k| -0.6 + 1.2 * k as f64 / (steps - 1) as f64).collect();
    let ss: Vec<f64> = us.iter().map(|&u| sse(u)).collect();
    for (k, &v) in ss.iter().enumerate() {
        if v < best.1 {
            best = (k, v);
        }
    }
    if best.0 == 0 || best.0 == steps - 1 {
        return None;
    }
    let (vm, v0, vp) = (ss[best.0 - 1], ss[best.0], ss[best.0 + 1]);
    let den = vm - 2.0 * v0 + vp;
    if den <= 1e-12 {
        return None;
    }
    let u = us[best.0] + (us[1] - us[0]) * (0.5 * (vm - vp) / den).clamp(-0.6, 0.6);
    // confidence: curvature of the SSE valley normalized by signal energy
    let energy: f64 = cur.iter().map(|v| v * v).sum();
    if energy < 1e-9 {
        return None;
    }
    Some((u * lag, (den / energy).min(4.0)))
}

pub fn correct_x(disk: &Image, hp_window: usize) -> XRegResult {
    let w = disk.w;
    let thresh = percentile_f32(&disk.data, 75.0) * 0.25;
    let spans: Vec<Option<(usize, usize)>> = (0..w)
        .map(|x| interior_span(&disk.column(x), thresh))
        .collect();
    let chord: Vec<usize> = spans.iter().map(|s| s.map(|(a, b)| b - a).unwrap_or(0)).collect();
    let max_chord = *chord.iter().max().unwrap_or(&0);
    let trusted: Vec<bool> = (0..w)
        .map(|x| chord[x] > (max_chord as f64 * 0.55) as usize && chord[x] > 60)
        .collect();
    let edge_margin = 12usize;
    let derivs: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| column_derivative(&disk.column(x)))
        .collect();

    // The alpha-projection measures a frame's position relative to its own
    // (displaced) neighbors — a SECOND DIFFERENCE of the trajectory. Like
    // the y solve, low frequencies must be recovered from a multi-lag
    // constraint graph: 2*delta(t) - delta(t-L) - delta(t+L) = 2*m_L(t),
    // solved by damped weighted Gauss-Seidel per slit half.
    let estimate_half = |half: u8| -> (Vec<f64>, Vec<f64>) {
        let lags: [usize; 5] = [1, 2, 4, 8, 16];
        let mut cons: Vec<(usize, usize, f64, f64)> = Vec::new(); // (t, L, m, w)
        for &lag in &lags {
            let cs: Vec<Option<(usize, usize, f64, f64)>> = (2 * lag..w.saturating_sub(2 * lag))
                .into_par_iter()
                .map(|x| {
                    for &xx in &[x - 2 * lag, x - lag, x, x + lag, x + 2 * lag] {
                        if !trusted[xx] {
                            return None;
                        }
                    }
                    let sp: Vec<(usize, usize)> = [x - 2 * lag, x - lag, x, x + lag, x + 2 * lag]
                        .iter()
                        .filter_map(|&xx| spans[xx])
                        .collect();
                    if sp.len() != 5 {
                        return None;
                    }
                    let mut lo = sp.iter().map(|s| s.0).max().unwrap() + edge_margin;
                    let mut hi = sp.iter().map(|s| s.1).min().unwrap().saturating_sub(edge_margin);
                    let mid = (lo + hi) / 2;
                    match half {
                        1 => hi = mid,
                        2 => lo = mid,
                        _ => {}
                    }
                    if hi <= lo + 40 {
                        return None;
                    }
                    let (m, c) = lagrange_delta(
                        &derivs[x - 2 * lag][lo..hi],
                        &derivs[x - lag][lo..hi],
                        &derivs[x + lag][lo..hi],
                        &derivs[x + 2 * lag][lo..hi],
                        &derivs[x][lo..hi],
                        lag as f64,
                    )?;
                    Some((x, lag, m, c / (lag * lag) as f64))
                })
                .collect();
            cons.extend(cs.into_iter().flatten());
        }
        // ridge prior toward zero bounds the L^2 low-frequency noise
        // amplification of inverting a second-difference operator
        let mean_cw = if cons.is_empty() {
            0.0
        } else {
            cons.iter().map(|c| c.3).sum::<f64>() / cons.len() as f64
        };
        let ridge = 0.05 * mean_cw;
        let mut delta = vec![0.0f64; w];
        let mut wsum = vec![0.0f64; w];
        let mut num = vec![0.0f64; w];
        for _ in 0..150 {
            num.iter_mut().for_each(|v| *v = 0.0);
            wsum.iter_mut().for_each(|v| *v = 0.0);
            for &(t, lag, m, cw) in &cons {
                num[t] += cw * (m + 0.5 * (delta[t - lag] + delta[t + lag]));
                wsum[t] += cw;
            }
            for x in 0..w {
                if wsum[x] > 1e-9 {
                    delta[x] = 0.5 * delta[x] + 0.5 * num[x] / (wsum[x] + ridge);
                }
            }
        }
        (delta, wsum)
    };
    let (d_top, w_top) = estimate_half(1);
    let (d_bot, w_bot) = estimate_half(2);

    // combine, remove robust line (degenerate with sx), shrink by agreement
    let usable: Vec<usize> = (0..w).filter(|&x| w_top[x] > 0.0 && w_bot[x] > 0.0).collect();
    if usable.len() < 100 {
        return XRegResult { corrected: disk.clone(), delta: vec![0.0; w] };
    }
    let d_top = crate::mathutil::gaussian_smooth(&d_top, 4.0);
    let d_bot = crate::mathutil::gaussian_smooth(&d_bot, 4.0);
    let avg: Vec<f64> = (0..w).map(|x| 0.5 * (d_top[x] + d_bot[x])).collect();
    let xs: Vec<f64> = usable.iter().map(|&x| x as f64).collect();
    let ys: Vec<f64> = usable.iter().map(|&x| avg[x]).collect();
    let ws = vec![1.0; xs.len()];
    let line = crate::mathutil::polyfit_robust(&xs, &ys, &ws, 1, 3).unwrap_or(vec![0.0, 0.0]);
    let detrended: Vec<f64> = (0..w)
        .map(|x| {
            if w_top[x] > 0.0 && w_bot[x] > 0.0 {
                avg[x] - crate::mathutil::polyval(&line, x as f64)
            } else {
                0.0
            }
        })
        .collect();
    // Wiener shrinkage from split-half disagreement
    let mean_u = |v: &[f64]| usable.iter().map(|&x| v[x]).sum::<f64>() / usable.len() as f64;
    let md = mean_u(&detrended);
    let var_avg = usable.iter().map(|&x| (detrended[x] - md).powi(2)).sum::<f64>() / usable.len() as f64;
    let var_noise = usable
        .iter()
        .map(|&x| (d_top[x] - d_bot[x]).powi(2))
        .sum::<f64>()
        / usable.len() as f64
        / 4.0;
    let mut shrink = (1.0 - var_noise / var_avg.max(1e-12)).clamp(0.0, 1.0);
    if std::env::var("GS_NOSHRINK").is_ok() { shrink = 1.0; }
    if std::env::var("GS_DEBUG").is_ok() {
        eprintln!("xreg: usable {} var_avg {:.4} var_noise {:.4} shrink {:.3}", usable.len(), var_avg, var_noise, shrink);
    }
    // mild smoothing of the estimate itself (per-column noise floor), then
    // remove slow trend beyond the hp window? No: slow nonlinear scan-speed
    // variation is a REAL geometric error we want to correct; only the
    // linear part was removed. Smooth lightly to kill single-column spikes.
    // the seeing x-trajectory is correlated over ~10 frames; smoothing at
    // sigma 4 keeps the signal band and cuts estimator noise ~sqrt(8)x
    let smooth = crate::mathutil::gaussian_smooth(&detrended, 4.0);
    let delta: Vec<f64> = (0..w)
        .map(|x| if w_top[x] > 0.0 && w_bot[x] > 0.0 { smooth[x] * shrink } else { 0.0 })
        .collect();
    let _ = hp_window;

    let corrected = apply_x_offsets(disk, &delta);
    XRegResult { corrected, delta }
}

/// Resample every row at x - delta(x) (B-spline).
pub fn apply_x_offsets(disk: &Image, delta: &[f64]) -> Image {
    let w = disk.w;
    let h = disk.h;
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let row: Vec<f64> = (0..w).map(|x| disk.at(x, y) as f64).collect();
            let mut coef = row.clone();
            bspline_prefilter(&mut coef);
            (0..w)
                .map(|x| {
                    let d = delta[x];
                    if d.abs() < 0.04 {
                        disk.at(x, y)
                    } else {
                        bspline_eval(&coef, (x as f64 - d).clamp(0.0, (w - 1) as f64)) as f32
                    }
                })
                .collect()
        })
        .collect();
    let mut out = Image::new(w, h);
    for (y, r) in rows.iter().enumerate() {
        out.row_mut(y).copy_from_slice(r);
    }
    out
}


/// F9.3: limb-anchored x-offsets. At the disk's left/right tangent columns
/// the texture-based x-registration is gated off (too few rows), yet those
/// frames' x-displacement is directly visible as the horizontal residual of
/// each limb point from the fitted ellipse — an absolute anchor exactly
/// where the texture estimator is blind. y-jitter moves near-vertical limb
/// points ALONG the limb, so the horizontal residual isolates x cleanly.
pub fn limb_x_offsets(
    pts: &[crate::limb::EdgePoint],
    conic: &crate::ellipse::Conic,
    w: usize,
) -> Vec<f64> {
    let yc = {
        let mut ys: Vec<f64> = pts.iter().map(|p| p.y).collect();
        crate::mathutil::median_inplace(&mut ys)
    };
    let mut acc_top: Vec<Vec<f64>> = vec![Vec::new(); w];
    let mut acc_bot: Vec<Vec<f64>> = vec![Vec::new(); w];
    for p in pts {
        // solve the conic for x at this y; take the branch nearer the point
        let a = conic.a;
        let b = conic.b * p.y + conic.d;
        let c = conic.c * p.y * p.y + conic.e * p.y + conic.f;
        let disc = b * b - 4.0 * a * c;
        if disc <= 0.0 || a.abs() < 1e-15 {
            continue;
        }
        let r1 = (-b + disc.sqrt()) / (2.0 * a);
        let r2 = (-b - disc.sqrt()) / (2.0 * a);
        let xe = if (p.x - r1).abs() < (p.x - r2).abs() { r1 } else { r2 };
        let dx = p.x - xe;
        if dx.abs() > 6.0 {
            continue; // prominence / junk edge
        }
        let t = p.x.round() as i64;
        if t >= 0 && (t as usize) < w {
            if p.y < yc {
                acc_top[t as usize].push(dx);
            } else {
                acc_bot[t as usize].push(dx);
            }
        }
    }
    // per-frame medians per limb half; split-half Wiener shrinkage: real
    // per-frame x-displacement is common to both arcs, limb-point noise is
    // not — with no true signal (clean scans) the correction disables itself
    let med_of = |acc: &Vec<Vec<f64>>, x: usize| -> f64 {
        if acc[x].is_empty() {
            f64::NAN
        } else {
            let mut v = acc[x].clone();
            crate::mathutil::median_inplace(&mut v)
        }
    };
    let raw_top: Vec<f64> = (0..w).map(|x| med_of(&acc_top, x)).collect();
    let raw_bot: Vec<f64> = (0..w).map(|x| med_of(&acc_bot, x)).collect();
    let both: Vec<usize> = (0..w)
        .filter(|&x| raw_top[x].is_finite() && raw_bot[x].is_finite())
        .collect();
    let mut shrink = 0.0;
    if both.len() > 40 {
        let avg: Vec<f64> = both.iter().map(|&x| 0.5 * (raw_top[x] + raw_bot[x])).collect();
        let ma = avg.iter().sum::<f64>() / avg.len() as f64;
        let var_avg = avg.iter().map(|v| (v - ma).powi(2)).sum::<f64>() / avg.len() as f64;
        let var_noise = both
            .iter()
            .map(|&x| (raw_top[x] - raw_bot[x]).powi(2))
            .sum::<f64>()
            / both.len() as f64
            / 4.0;
        shrink = (1.0 - var_noise / var_avg.max(1e-12)).clamp(0.0, 1.0);
    }
    if shrink < 0.05 {
        return vec![0.0; w];
    }
    let raw: Vec<f64> = (0..w)
        .map(|x| match (raw_top[x].is_finite(), raw_bot[x].is_finite()) {
            (true, true) => 0.5 * (raw_top[x] + raw_bot[x]) * shrink,
            (true, false) => raw_top[x] * shrink,
            (false, true) => raw_bot[x] * shrink,
            _ => f64::NAN,
        })
        .collect();
    let sampled: Vec<usize> = (0..w).filter(|&x| raw[x].is_finite()).collect();
    if sampled.len() < 30 {
        return vec![0.0; w];
    }
    // fill short gaps by interpolation, but only within +-5 of a sample
    let mut delta = vec![0.0f64; w];
    for x in 0..w {
        if raw[x].is_finite() {
            delta[x] = raw[x];
        } else {
            let near = sampled.iter().min_by_key(|&&v| v.abs_diff(x)).unwrap();
            if near.abs_diff(x) <= 5 {
                delta[x] = raw[*near];
            }
        }
    }
    let smooth = crate::mathutil::gaussian_smooth(&delta, 2.0);
    smooth
        .iter()
        .map(|d| {
            // undo displacement: content shift -dx  =>  delta = -dx
            let v = -d.clamp(-3.0, 3.0);
            if v.abs() < 0.08 {
                0.0
            } else {
                v
            }
        })
        .collect()
}


/// F9.4: photometric x-anchors at the disk entry/exit ramps.
///
/// At tangent columns the chord flux is a steep function of scan position:
/// a frame displaced along the scan reads too dark or too bright for its
/// nominal x — visible on real data as dark vertical bands just inside the
/// left/right limb. Inverting that: dx = (F - E) / E', where E is the
/// smooth flux envelope and E' its slope. The sensor is strongest exactly
/// where texture x-registration and limb-count anchors are weakest.
/// Interior columns (flat E) are untouched; transparency residuals are
/// bounded by the clamp and by requiring a steep relative slope.
#[allow(dead_code)]
pub fn photometric_x_offsets(flux: &[f64]) -> Vec<f64> {
    let w = flux.len();
    if w < 200 {
        return vec![0.0; w];
    }
    // fill unmeasured flux (off-disk NaN) with zeros: ramps start at zero
    let f: Vec<f64> = flux.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
    let env = crate::mathutil::robust_loess_quadratic(&f, 81, 3);
    let fmax = env.iter().cloned().fold(f64::MIN, f64::max).max(1e-9);
    let mut delta = vec![0.0f64; w];
    for x in 2..w - 2 {
        let slope = (env[x + 2] - env[x - 2]) / 4.0;
        let e = env[x];
        // ramp gate: only where the relative slope is steep enough that a
        // plausible transparency residual (~2%) maps to <1 frame of error —
        // displacement must dominate the inversion
        if e < 0.04 * fmax || slope.abs() / e.max(1e-9) < 0.025 {
            continue;
        }
        let dx = (f[x] - e) / slope;
        delta[x] = dx.clamp(-2.5, 2.5);
    }
    let smooth = crate::mathutil::gaussian_smooth(&delta, 1.5);
    smooth
        .iter()
        .map(|d| {
            // content shift -dx  =>  sampling offset = -dx (see apply_x_offsets)
            let v = -(d * 0.8); // conservative gain
            if v.abs() < 0.3 { 0.0 } else { v }
        })
        .collect()
}
