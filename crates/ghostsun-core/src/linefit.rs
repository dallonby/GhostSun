//! Spectral line geometry: sub-pixel measurement of the absorption-line
//! center on the mean spectrum image, followed by a robust polynomial fit.
//!
//! Strategy (improves on JSolEx dual-seed):
//!   1. Restrict search to the *lit* spectral band (ignore dark margins).
//!   2. At mid-slit, rank many local absorption candidates by local depth ×
//!      continuum brightness (not just "darkest absolute sample").
//!   3. Continuously track each of the top-K candidates along the slit with a
//!      tight continuity window (JSolEx uses ±4; we start at ±4 and adapt to
//!      measured smile slope).
//!   4. Score full tracks by depth, coherence (post-trend residual), coverage,
//!      and continuum — pick the winner, not merely the darker of two seeds.
//!   5. Sub-pixel inverted-Gaussian refine + Tukey-IRLS smile poly.
//!
//! INTI baseline path remains whole-pixel argmin + sigma-clipped polyfit.

use crate::image2d::Image;
use crate::mathutil::{fit_inverted_gaussian, gaussian_smooth, polyfit_robust, polyval};

#[derive(Clone)]
pub struct LineGeometry {
    /// polynomial coefficients: x(y) = c[0] + c[1] y + c[2] y^2 (raw y coords)
    pub coeffs: Vec<f64>,
    pub y1: usize,
    pub y2: usize,
    pub rms: f64,
    pub n_rows_used: usize,
}

/// Median relative absorption depth of the darkest dip along each sampled row.
/// High when dispersion is **horizontal** (each row is a spectrum with a dark line).
pub fn absorption_contrast_along_rows(img: &Image) -> f64 {
    if img.w < 16 || img.h < 8 {
        return 0.0;
    }
    let step = (img.h / 64).max(1);
    let lo = 4usize;
    let hi = img.w.saturating_sub(4).max(lo + 1);
    let mut depths = Vec::new();
    for y in (0..img.h).step_by(step) {
        let row = img.row(y);
        let mut vmin = f32::MAX;
        let mut vmax = f32::MIN;
        for &v in &row[lo..hi] {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
        if vmax > 1.0 {
            depths.push(((vmax - vmin) / vmax) as f64);
        }
    }
    if depths.is_empty() {
        return 0.0;
    }
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    depths[depths.len() / 2]
}

/// Same score with axes swapped — high when dispersion is **vertical**.
pub fn absorption_contrast_along_cols(img: &Image) -> f64 {
    if img.w < 8 || img.h < 16 {
        return 0.0;
    }
    let step = (img.w / 64).max(1);
    let lo = 4usize;
    let hi = img.h.saturating_sub(4).max(lo + 1);
    let mut depths = Vec::new();
    for x in (0..img.w).step_by(step) {
        let mut vmin = f32::MAX;
        let mut vmax = f32::MIN;
        for y in lo..hi {
            let v = img.at(x, y);
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
        if vmax > 1.0 {
            depths.push(((vmax - vmin) / vmax) as f64);
        }
    }
    if depths.is_empty() {
        return 0.0;
    }
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    depths[depths.len() / 2]
}

/// Maximum absorption depth found when treating the given axis as dispersion.
///
/// Samples mid-band spatial rows/cols, builds a mean 1-D spectrum, and measures
/// local-minima depth against continuum ±80..150 px — wide enough for Hα
/// (FWHM often tens of px). Median min/max contrast alone is fooled by dark
/// sensor margins and many shallow lines on full-sensor multi-line frames
/// (16_00_03.ser: row contrast favoured the long axis while real Hα at ~69%
/// depth lives on the short axis).
fn max_line_depth_dispersion_along_width(img: &Image) -> f64 {
    if img.w < 32 || img.h < 16 {
        return 0.0;
    }
    let y0 = img.h / 4;
    let y1 = (3 * img.h / 4).max(y0 + 1);
    let step = ((y1 - y0) / 32).max(1);
    let mut prof = vec![0.0f64; img.w];
    let mut n = 0.0f64;
    let mut y = y0;
    while y < y1 {
        for (x, &v) in img.row(y).iter().enumerate() {
            prof[x] += v as f64;
        }
        n += 1.0;
        y += step;
    }
    if n < 1.0 {
        return 0.0;
    }
    for v in &mut prof {
        *v /= n;
    }
    let sm = gaussian_smooth(&prof, 2.0);
    // Exclude extreme margins (vignette looks like 100% absorption).
    let lo = (img.w / 12).max(20);
    let hi = img.w.saturating_sub(lo).max(lo + 16);
    let mut best = 0.0f64;
    for x in lo..hi {
        if !(sm[x] <= sm[x - 2]
            && sm[x] <= sm[x + 2]
            && sm[x] <= sm[x - 5]
            && sm[x] <= sm[x + 5])
        {
            continue;
        }
        let mut cont = 0.0f64;
        for &off in &[80usize, 100, 120, 150] {
            if x >= off {
                cont = cont.max(sm[x - off]);
            }
            if x + off < sm.len() {
                cont = cont.max(sm[x + off]);
            }
        }
        if cont < 1.0 {
            continue;
        }
        // A real absorption line recovers on BOTH sides. An illumination STEP --
        // the solar limb sitting near a slit end, or a vignetted margin -- only
        // recovers on one side, yet `cont` above takes the maximum across both
        // and so scores the bright side against the dark one as a near-total
        // absorption. That is not hypothetical: a disc offset 15 px along the
        // slit scored 0.984 transposed against 0.669 native, flipping the frame
        // orientation and transposing the entire reconstruction (900x160 rather
        // than 900x600), which collapsed limb detection and produced a radius
        // nine times too large. Requiring two-sided recovery is what separates a
        // line from an edge; depth alone cannot.
        let half = sm[x] + 0.5 * (cont - sm[x]);
        // Span sized to real spectral lines, which are NARROW: the Halpha core
        // is ~10-30 px at working dispersions (the ~200 px figure sometimes
        // quoted is the ROI crop height around the line, not the line). The
        // tight window also does double duty on the transposed axis: a sunspot
        // is a 50-200 px dip in the along-slit profile that recovers on BOTH
        // sides -- unlike the slit-end step -- so a generous span would let it
        // count as a "line" on the wrong axis. At 60 px its recovery lies
        // outside the window and it is rejected along with the step.
        let span = (img.w / 8).clamp(20, 60);
        let recovers = |dir: isize| {
            let mut i = x as isize;
            for _ in 0..span {
                i += dir;
                if i < 0 || i as usize >= sm.len() {
                    return false;
                }
                if sm[i as usize] >= half {
                    return true;
                }
            }
            false
        };
        if !recovers(-1) || !recovers(1) {
            continue;
        }
        let d = (cont - sm[x]) / cont;
        if d > best {
            best = d;
        }
    }
    best
}

/// Whether the SER frame should be transposed so dispersion is horizontal.
///
/// Classic Sol'Ex / SHG-700 files are often taller than wide after capture and
/// need a flip; full-sensor landscape spectra must choose the axis that holds
/// the **deepest real absorption line**, not merely the higher median contrast
/// (margins and multi-line clutter bias the old score).
///
/// Returns `(transpose, row_score, col_score)` where scores are max line depth
/// treating width (rows) or height (cols) as the dispersion axis.
pub fn should_transpose_for_dispersion(img: &Image) -> (bool, f64, f64) {
    // Score if we keep native: dispersion = width (sample along each row).
    let depth_native = max_line_depth_dispersion_along_width(img);
    // Score if we transpose: dispersion becomes current height.
    let depth_transposed = max_line_depth_dispersion_along_width(&img.transpose());
    // Also keep legacy median contrast as a weak tie-break signal.
    let row_c = absorption_contrast_along_rows(img);
    let col_c = absorption_contrast_along_cols(img);

    let prefer_transpose = if depth_native > 0.08 || depth_transposed > 0.08 {
        // Need a clear win on deep-line depth (≥8% absolute, or ≥15% relative).
        if depth_transposed > depth_native + 0.08
            || (depth_transposed > depth_native * 1.15 && depth_transposed > 0.25)
        {
            true
        } else if depth_native > depth_transposed + 0.08
            || (depth_native > depth_transposed * 1.15 && depth_native > 0.25)
        {
            false
        } else {
            // Ambiguous deep-line scores: fall back to contrast / aspect.
            let ratio = col_c / row_c.max(1e-6);
            if ratio > 1.15 {
                true
            } else if ratio < 1.0 / 1.15 {
                false
            } else {
                img.w > img.h
            }
        }
    } else if row_c > 0.02 || col_c > 0.02 {
        let ratio = col_c / row_c.max(1e-6);
        if ratio > 1.15 {
            true
        } else if ratio < 1.0 / 1.15 {
            false
        } else {
            img.w > img.h
        }
    } else {
        img.w > img.h
    };
    // Report deep-line depths as the diagnostic scores (pipeline logs these).
    (prefer_transpose, depth_native, depth_transposed)
}

/// Detect the vertical extent of the spectrum on the mean image
/// (rows where the slit actually saw light).
pub fn detect_spectrum_rows(mean_img: &Image) -> (usize, usize) {
    let h = mean_img.h;
    let mut prof: Vec<f64> = (0..h)
        .map(|y| mean_img.row(y).iter().map(|&v| v as f64).sum::<f64>() / mean_img.w as f64)
        .collect();
    prof = gaussian_smooth(&prof, 5.0);
    let pmax = prof.iter().cloned().fold(f64::MIN, f64::max);
    let pmin = prof.iter().cloned().fold(f64::MAX, f64::min);
    let thresh = pmin + 0.15 * (pmax - pmin);
    let mut y1 = 0;
    let mut y2 = h - 1;
    for (y, &v) in prof.iter().enumerate() {
        if v > thresh {
            y1 = y;
            break;
        }
    }
    for (y, &v) in prof.iter().enumerate().rev() {
        if v > thresh {
            y2 = y;
            break;
        }
    }
    (y1, y2)
}

/// Local continuum beside a candidate core: max of samples in two side
/// windows (skips the core itself). Wide offsets (±80..150) so broad Hα
/// (FWHM often tens of px) is not scored against its own wings.
fn local_continuum(profile: &[f64], x: usize) -> f64 {
    let n = profile.len();
    let mut c = f64::MIN;
    for &off in &[12usize, 20, 35, 55, 80, 100, 120, 150] {
        if x >= off {
            c = c.max(profile[x - off]);
        }
        if x + off < n {
            c = c.max(profile[x + off]);
        }
    }
    if c == f64::MIN {
        profile.get(x).copied().unwrap_or(0.0)
    } else {
        c
    }
}

/// Detect the horizontal extent of the *illuminated* spectrum along the
/// dispersion axis.
///
/// Full-sensor multi-line frames often have dark margins left/right of the
/// dispersed band. A global per-row argmin locks onto those margins (they look
/// like ~100% absorption) and never finds a real solar line. Returns
/// `(x_lo, x_hi)` inclusive bounds of columns whose mean flux exceeds 25% of
/// the 90th-percentile column flux over lit slit rows.
pub fn detect_spectrum_cols(mean_img: &Image) -> (usize, usize) {
    let w = mean_img.w;
    if w < 16 {
        return (0, w.saturating_sub(1));
    }
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let step = ((y2.saturating_sub(y1)) / 64).max(1);
    let mut col = vec![0.0f64; w];
    let mut n = 0.0f64;
    let mut y = y1;
    while y <= y2 {
        for (x, &v) in mean_img.row(y).iter().enumerate() {
            col[x] += v as f64;
        }
        n += 1.0;
        y += step;
    }
    if n > 0.0 {
        for v in &mut col {
            *v /= n;
        }
    }
    let sm = gaussian_smooth(&col, 5.0);
    let mut sorted = sm.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p90 = sorted[((0.90 * (w as f64 - 1.0)).round() as usize).min(w - 1)];
    if p90 <= 1.0 {
        return (4, w.saturating_sub(5).max(4));
    }
    let thresh = p90 * 0.25;
    let mut x1 = 0;
    let mut x2 = w - 1;
    for (x, &v) in sm.iter().enumerate() {
        if v > thresh {
            x1 = x;
            break;
        }
    }
    for (x, &v) in sm.iter().enumerate().rev() {
        if v > thresh {
            x2 = x;
            break;
        }
    }
    // Keep a usable interior; fall back to near-full width if the gate
    // collapsed (uniform illumination, no dark margins).
    if x2 <= x1 + 16 {
        return (4, w.saturating_sub(5).max(4));
    }
    (x1, x2)
}

/// Measure the line center on each row of the mean image and fit the smile
/// polynomial of the given degree.
pub fn fit_line_geometry(mean_img: &Image, deg: usize) -> Option<LineGeometry> {
    fit_line_geometry_impl(mean_img, deg, None, None)
}

/// Force the primary line near a spectral column (px). Used when AUTO picks
/// the wrong companion on multi-line full-sensor SERs.
pub fn fit_line_geometry_at(mean_img: &Image, deg: usize, seed_x: f64) -> Option<LineGeometry> {
    fit_line_geometry_impl(mean_img, deg, None, Some(seed_x))
}

/// Fit smile geometry for a companion line near a seed track.
///
/// `seed_track(y)` returns the expected spectral column at slit row `y`
/// (typically primary smile + constant offset). Continuous tracking keeps
/// the weak neighbour from being stolen by a deeper line on the same row.
pub fn fit_line_geometry_seeded(
    mean_img: &Image,
    deg: usize,
    half_win: usize,
    seed_track: &dyn Fn(f64) -> f64,
) -> Option<LineGeometry> {
    fit_line_geometry_impl(mean_img, deg, Some((half_win.max(4), seed_track)), None)
}

/// One absorption candidate at a single slit row.
#[derive(Clone, Copy, Debug)]
struct LineCand {
    x: f64,
    depth: f64,
    cont: f64,
    /// Rough half-width to half-depth (px); broad solar lines score higher.
    half_width: f64,
}

/// Ranked local minima in a 1-D spectrum (dispersion axis).
fn local_line_candidates(
    sm: &[f64],
    lo: usize,
    hi: usize,
    cont_floor: f64,
    min_depth: f64,
) -> Vec<LineCand> {
    let edge = 3usize;
    let x0 = lo + edge;
    let x1 = hi.saturating_sub(edge);
    if x1 <= x0 + 2 || sm.len() < 16 {
        return Vec::new();
    }
    let mut cont_hi = cont_floor;
    let step = ((x1 - x0) / 64).max(1);
    let mut x = x0;
    while x < x1 {
        cont_hi = cont_hi.max(local_continuum(sm, x));
        x += step;
    }
    let cont_hi = cont_hi.max(cont_floor * 5.0);

    let mut cands = Vec::new();
    for x in x0..x1 {
        if !(sm[x] <= sm[x - 1]
            && sm[x] <= sm[x + 1]
            && sm[x] <= sm[x - 2]
            && sm[x] <= sm[x + 2]
            && sm[x] <= sm[x - 3]
            && sm[x] <= sm[x + 3])
        {
            continue;
        }
        let cont = local_continuum(sm, x);
        if cont < cont_floor || cont < 0.40 * cont_hi {
            continue;
        }
        let depth = (cont - sm[x]) / cont;
        if depth < min_depth {
            continue;
        }
        // Half-width at half-depth (capped) — Hα is broad; tellurics/noise are not.
        let half_level = cont - 0.5 * (cont - sm[x]);
        let mut left = x;
        while left > x0 && sm[left] < half_level {
            left -= 1;
        }
        let mut right = x;
        while right + 1 < x1 && sm[right] < half_level {
            right += 1;
        }
        let half_width = 0.5 * (right - left) as f64;
        cands.push(LineCand {
            x: x as f64,
            depth,
            cont,
            half_width,
        });
    }
    // Merge near-duplicates (keep deeper).
    cands.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<LineCand> = Vec::new();
    for c in cands {
        if let Some(last) = merged.last_mut() {
            if (c.x - last.x).abs() < 6.0 {
                if c.depth > last.depth {
                    *last = c;
                }
                continue;
            }
        }
        merged.push(c);
    }
    // Prefer deep, broad lines on bright continuum (Hα over shallow narrow dips).
    merged.sort_by(|a, b| {
        let broad_a = (a.half_width / 8.0).clamp(0.5, 4.0);
        let broad_b = (b.half_width / 8.0).clamp(0.5, 4.0);
        let sa = a.depth * (a.cont / cont_hi).powi(2) * broad_a;
        let sb = b.depth * (b.cont / cont_hi).powi(2) * broad_b;
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    merged
}

/// Continuous track of one seed along the slit. Returns refined (y, x, weight)
/// samples plus mean core intensity (lower = stronger absorption).
fn track_line_continuous(
    mean_img: &Image,
    ya: usize,
    yb: usize,
    seed_x: f64,
    lo: usize,
    hi: usize,
    cont_floor: f64,
    min_depth: f64,
    max_dev: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64) {
    let mut xs = Vec::new(); // slit y
    let mut ys = Vec::new(); // spectral x
    let mut ws = Vec::new();
    let mut intensity_sum = 0.0f64;
    let mut depth_sum = 0.0f64;
    let mut n_i = 0.0f64;
    let mut prev_x = seed_x;

    // Adaptive continuity: start tight (JSolEx-like ±4), grow slightly with
    // observed row-to-row smile slope so curved lines stay locked.
    let mut dev = max_dev.max(3.0);

    for y in ya..yb {
        let row: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
        let sm = gaussian_smooth(&row, 1.5);
        let search_lo = ((prev_x - dev).floor() as isize)
            .max(lo as isize)
            .max(0) as usize;
        let search_hi = ((prev_x + dev).ceil() as isize)
            .min(hi as isize)
            .max(search_lo as isize + 1) as usize;
        if search_hi <= search_lo + 2 {
            continue;
        }
        // Prefer local min closest to predicted track among those with depth;
        // fall back to absolute min intensity in the continuity window.
        let cands = local_line_candidates(&sm, search_lo, search_hi, cont_floor, min_depth * 0.6);
        let xmin = if let Some(best) = cands
            .iter()
            .min_by(|a, b| {
                let da = (a.x - prev_x).abs() - 0.5 * a.depth; // slight depth bias
                let db = (b.x - prev_x).abs() - 0.5 * b.depth;
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.x)
        {
            best
        } else {
            // No structured local min: take darkest sample near prediction.
            let mut bx = search_lo;
            let mut bv = f64::MAX;
            for x in search_lo..search_hi {
                if sm[x] < bv {
                    bv = sm[x];
                    bx = x;
                }
            }
            bx as f64
        };

        let xi = xmin.round().clamp(0.0, (sm.len() - 1) as f64) as usize;
        let wfit = 14usize;
        let a = xi.saturating_sub(wfit).max(1);
        let b = (xi + wfit + 1).min(row.len() - 1);
        let win_x: Vec<f64> = (a..b).map(|x| x as f64).collect();
        let win_y: Vec<f64> = row[a..b].to_vec();
        let (mu, depth_w, core_i) =
            if let Some((mu, sigma, amp, off)) = fit_inverted_gaussian(&win_x, &win_y, xmin, 3.5) {
                if mu > a as f64
                    && mu < b as f64
                    && (0.6..=50.0).contains(&sigma)
                    && off >= cont_floor * 0.5
                {
                    let d = (amp / off.max(1e-9)).clamp(0.0, 1.0);
                    (mu, d.max(0.05), off - amp)
                } else {
                    let cont = local_continuum(&sm, xi);
                    let d = ((cont - sm[xi]) / cont.max(1e-9)).clamp(0.0, 1.0);
                    (xmin, d.max(0.05), sm[xi])
                }
            } else {
                let cont = local_continuum(&sm, xi);
                let d = ((cont - sm[xi]) / cont.max(1e-9)).clamp(0.0, 1.0);
                (xmin, d.max(0.05), sm[xi])
            };

        // Reject wild jumps (track loss).
        if (mu - prev_x).abs() > dev * 1.75 {
            continue;
        }
        // Adapt continuity window to local smile slope.
        let step = (mu - prev_x).abs();
        dev = (0.85 * dev + 0.15 * (step * 2.5 + max_dev)).clamp(max_dev, 12.0);
        prev_x = mu;

        xs.push(y as f64);
        ys.push(mu);
        ws.push(depth_w);
        intensity_sum += core_i;
        depth_sum += depth_w;
        n_i += 1.0;
    }

    let mean_i = if n_i > 0.0 {
        intensity_sum / n_i
    } else {
        f64::MAX
    };
    let mean_d = if n_i > 0.0 { depth_sum / n_i } else { 0.0 };
    (xs, ys, ws, mean_i, mean_d)
}

/// Score a continuous track: higher is better.
fn score_track(xs: &[f64], ys: &[f64], ws: &[f64], mean_depth: f64, mean_intensity: f64) -> f64 {
    if xs.len() < 30 {
        return f64::NEG_INFINITY;
    }
    // Coherence: residual after a robust quadratic (lower residual → single line).
    let ones = vec![1.0; xs.len()];
    let coeffs = match polyfit_robust(xs, ys, &ones, 2, 3) {
        Some(c) => c,
        None => return f64::NEG_INFINITY,
    };
    let mut ss = 0.0f64;
    let mut n = 0.0f64;
    for i in 0..xs.len() {
        let r = ys[i] - polyval(&coeffs, xs[i]);
        if r.abs() < 6.0 {
            ss += r * r;
            n += 1.0;
        }
    }
    let rms = (ss / n.max(1.0)).sqrt();
    let coverage = xs.len() as f64;
    let mean_w = ws.iter().sum::<f64>() / ws.len() as f64;
    // Depth and coverage dominate; intensity (lower core) helps break ties;
    // coherence penalises tracks that hop between lines.
    mean_depth * 100.0 + mean_w * 40.0 + coverage * 0.05 - rms * 8.0
        - (mean_intensity / 1e5).clamp(0.0, 5.0)
}

fn finish_geometry(
    xs: &[f64],
    ys: &[f64],
    ws: &[f64],
    deg: usize,
    y1: usize,
    y2: usize,
    lo: usize,
    hi: usize,
) -> Option<LineGeometry> {
    if xs.len() < 30 {
        return None;
    }
    let coeffs = polyfit_robust(xs, ys, ws, deg, 4)?;
    let mut n_in = 0usize;
    let mut ss = 0.0;
    let mut n = 0.0f64;
    for i in 0..xs.len() {
        let pred = polyval(&coeffs, xs[i]);
        let r = ys[i] - pred;
        if r.abs() < 3.0 {
            ss += r * r;
            n += 1.0;
        }
        if pred >= lo as f64 - 5.0 && pred <= hi as f64 + 5.0 {
            n_in += 1;
        }
    }
    if n_in < xs.len() / 2 {
        return None;
    }
    Some(LineGeometry {
        coeffs,
        y1,
        y2,
        rms: (ss / n.max(1.0)).sqrt(),
        n_rows_used: xs.len(),
    })
}

fn lit_band_and_floor(mean_img: &Image, ya: usize, yb: usize) -> (usize, usize, f64) {
    let (x_lit0, x_lit1) = detect_spectrum_cols(mean_img);
    let x_pad = ((x_lit1.saturating_sub(x_lit0)) / 40).clamp(12, 80);
    let lo = (x_lit0 + x_pad).max(4);
    let hi = x_lit1
        .saturating_sub(x_pad)
        .min(mean_img.w.saturating_sub(4))
        .max(lo + 16);
    let mut samples = Vec::new();
    let y_step = ((yb - ya) / 32).max(1);
    let x_step = ((hi - lo) / 64).max(1);
    let mut y = ya;
    while y < yb {
        let row = mean_img.row(y);
        let mut x = lo;
        while x < hi {
            samples.push(row[x] as f64);
            x += x_step;
        }
        y += y_step;
    }
    let cont_floor = if samples.is_empty() {
        1.0
    } else {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p90 = samples
            [((0.90 * (samples.len() as f64 - 1.0)).round() as usize).min(samples.len() - 1)];
        (p90 * 0.20).max(1.0)
    };
    (lo, hi, cont_floor)
}

fn fit_line_geometry_impl(
    mean_img: &Image,
    deg: usize,
    seeded: Option<(usize, &dyn Fn(f64) -> f64)>,
    force_seed_x: Option<f64>,
) -> Option<LineGeometry> {
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let margin = ((y2 - y1) / 20).clamp(5, 40);
    let (ya, yb) = (y1 + margin, y2.saturating_sub(margin));
    if yb <= ya + 10 {
        return None;
    }
    let (lo, hi, cont_floor) = lit_band_and_floor(mean_img, ya, yb);
    let min_depth = if seeded.is_some() || force_seed_x.is_some() {
        0.03
    } else {
        0.045
    };

    // ---- seeded companion path: follow seed_track(y) each row with a
    // tight window (does not free-run continuity — avoids snapping to a
    // deeper neighbour like Hα when tracking a weak companion).
    if let Some((half_win, seed_fn)) = seeded {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut ws = Vec::new();
        let half = half_win.max(6) as f64;
        for y in ya..yb {
            let pred = seed_fn(y as f64);
            let row: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
            let sm = gaussian_smooth(&row, 1.5);
            let search_lo = ((pred - half).floor() as isize)
                .max(lo as isize)
                .max(0) as usize;
            let search_hi = ((pred + half).ceil() as isize)
                .min(hi as isize)
                .max(search_lo as isize + 1) as usize;
            if search_hi <= search_lo + 2 {
                continue;
            }
            let cands =
                local_line_candidates(&sm, search_lo, search_hi, cont_floor, min_depth * 0.5);
            let xmin = cands
                .iter()
                .min_by(|a, b| {
                    (a.x - pred)
                        .abs()
                        .partial_cmp(&(b.x - pred).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|c| c.x)
                .unwrap_or_else(|| {
                    let mut bx = search_lo;
                    let mut bv = f64::MAX;
                    for x in search_lo..search_hi {
                        if sm[x] < bv {
                            bv = sm[x];
                            bx = x;
                        }
                    }
                    bx as f64
                });
            let xi = xmin.round().clamp(0.0, (sm.len() - 1) as f64) as usize;
            let wfit = 12usize;
            let a = xi.saturating_sub(wfit).max(1);
            let b = (xi + wfit + 1).min(row.len() - 1);
            let win_x: Vec<f64> = (a..b).map(|x| x as f64).collect();
            let win_y: Vec<f64> = row[a..b].to_vec();
            if let Some((mu, sigma, amp, off)) =
                fit_inverted_gaussian(&win_x, &win_y, xmin, 3.0)
            {
                if mu > a as f64
                    && mu < b as f64
                    && (0.6..=50.0).contains(&sigma)
                    && off >= cont_floor * 0.4
                    && (mu - pred).abs() <= half * 1.25
                {
                    xs.push(y as f64);
                    ys.push(mu);
                    ws.push((amp / off.max(1e-9)).clamp(0.05, 1.0));
                }
            }
        }
        return finish_geometry(&xs, &ys, &ws, deg, y1, y2, lo, hi);
    }

    // ---- forced spectral column ----
    if let Some(seed) = force_seed_x {
        let (xs, ys, ws, _, _) = track_line_continuous(
            mean_img, ya, yb, seed, lo, hi, cont_floor, min_depth, 6.0,
        );
        return finish_geometry(&xs, &ys, &ws, deg, y1, y2, lo, hi);
    }

    // ---- AUTO: multi-candidate continuous tracks (better than JSolEx dual-seed) ----
    // Harvest candidates across the mid-third of the slit (stable continuum).
    let y_mid0 = ya + (yb - ya) / 3;
    let y_mid1 = ya + 2 * (yb - ya) / 3;
    let mut seed_pool: Vec<LineCand> = Vec::new();
    let step = ((y_mid1 - y_mid0) / 24).max(1);
    let mut y = y_mid0;
    while y < y_mid1 {
        let row: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
        let sm = gaussian_smooth(&row, 1.5);
        for c in local_line_candidates(&sm, lo, hi, cont_floor, min_depth) {
            seed_pool.push(c);
        }
        y += step;
    }
    // Cluster seeds in spectral x (≈ same physical line).
    seed_pool.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    let mut clusters: Vec<(f64, f64, f64, usize)> = Vec::new(); // (x_sum, depth_sum, cont_sum, n)
    for c in &seed_pool {
        if let Some(cl) = clusters.last_mut() {
            let mean_x = cl.0 / cl.3 as f64;
            if (c.x - mean_x).abs() < 8.0 {
                cl.0 += c.x;
                cl.1 += c.depth;
                cl.2 += c.cont;
                cl.3 += 1;
                continue;
            }
        }
        clusters.push((c.x, c.depth, c.cont, 1));
    }
    let mut seeds: Vec<(f64, f64)> = clusters
        .iter()
        .map(|cl| {
            let x = cl.0 / cl.3 as f64;
            let score = (cl.1 / cl.3 as f64) * (cl.2 / cl.3 as f64).sqrt() * (cl.3 as f64).sqrt();
            (x, score)
        })
        .collect();
    seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Evaluate top-K full tracks (JSolEx only tries 2 raw minima).
    const MAX_TRACKS: usize = 6;
    seeds.truncate(MAX_TRACKS);

    let mut best: Option<(LineGeometry, f64)> = None;
    for (seed_x, _) in &seeds {
        let (xs, ys, ws, mean_i, mean_d) = track_line_continuous(
            mean_img, ya, yb, *seed_x, lo, hi, cont_floor, min_depth, 4.0,
        );
        let sc = score_track(&xs, &ys, &ws, mean_d, mean_i);
        if sc.is_finite() {
            if let Some(geom) = finish_geometry(&xs, &ys, &ws, deg, y1, y2, lo, hi) {
                // Bonus for smile spanning many rows with low residual —
                // full continuous tracks beat short noisy ones.
                let bonus = (geom.n_rows_used as f64).ln() * 2.0 - geom.rms * 3.0;
                let sc = sc + bonus;
                if best.as_ref().map_or(true, |(_, bs)| sc > *bs) {
                    best = Some((geom, sc));
                }
            }
        }
    }
    best.map(|(g, _)| g)
}

/// Detect companion line *seeds* for multi-line composite: ranked spectral
/// positions excluding the primary, suitable for continuous seeded tracking.
pub fn detect_companion_seeds(
    mean_img: &Image,
    primary_center: f64,
    min_depth: f64,
    min_sep_px: f64,
    max_companions: usize,
) -> Vec<f64> {
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let margin = ((y2 - y1) / 20).clamp(5, 40);
    let (ya, yb) = (y1 + margin, y2.saturating_sub(margin));
    let (lo, hi, cont_floor) = lit_band_and_floor(mean_img, ya, yb);
    let y_mid0 = ya + (yb - ya) / 3;
    let y_mid1 = ya + 2 * (yb - ya) / 3;
    let step = ((y_mid1 - y_mid0) / 20).max(1);
    let mut pool: Vec<LineCand> = Vec::new();
    let mut y = y_mid0;
    while y < y_mid1 {
        let row: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
        let sm = gaussian_smooth(&row, 1.5);
        for c in local_line_candidates(&sm, lo, hi, cont_floor, min_depth) {
            if (c.x - primary_center).abs() >= min_sep_px {
                pool.push(c);
            }
        }
        y += step;
    }
    pool.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    let mut clusters: Vec<(f64, f64, usize)> = Vec::new();
    for c in &pool {
        if let Some(cl) = clusters.last_mut() {
            let mx = cl.0 / cl.2 as f64;
            if (c.x - mx).abs() < 10.0 {
                cl.0 += c.x;
                cl.1 += c.depth;
                cl.2 += 1;
                continue;
            }
        }
        clusters.push((c.x, c.depth, 1));
    }
    let mut out: Vec<(f64, f64)> = clusters
        .iter()
        .map(|cl| (cl.0 / cl.2 as f64, cl.1 / cl.2 as f64))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(max_companions);
    out.into_iter().map(|(x, _)| x).collect()
}

/// Mean 1-D spectrum over the lit slit rows of a mean frame image.
pub fn mean_spectrum_1d(mean_img: &Image) -> Vec<f64> {
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let mut prof = vec![0.0f64; mean_img.w];
    let mut n = 0.0f64;
    let ya = y1 + ((y2 - y1) / 20).clamp(2, 20);
    let yb = y2.saturating_sub(((y2 - y1) / 20).clamp(2, 20));
    if yb <= ya {
        return prof;
    }
    for y in ya..=yb {
        let row = mean_img.row(y);
        for x in 0..mean_img.w {
            prof[x] += row[x] as f64;
        }
        n += 1.0;
    }
    if n > 0.0 {
        for v in &mut prof {
            *v /= n;
        }
    }
    prof
}

/// De-smiled mean spectrum: each row is resampled onto the mid-row spectral
/// column grid using the primary smile polynomial so weak companion lines do
/// not smear out of detectability.
pub fn mean_spectrum_desmiled(mean_img: &Image, geom: &LineGeometry) -> Vec<f64> {
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let mut prof = vec![0.0f64; mean_img.w];
    let mut n = 0.0f64;
    let ya = y1 + ((y2 - y1) / 20).clamp(2, 20);
    let yb = y2.saturating_sub(((y2 - y1) / 20).clamp(2, 20));
    if yb <= ya || mean_img.w < 8 {
        return prof;
    }
    let ymid = 0.5 * (ya + yb) as f64;
    let ref_c = polyval(&geom.coeffs, ymid);
    let w = mean_img.w;
    for y in ya..=yb {
        let row = mean_img.row(y);
        let shift = polyval(&geom.coeffs, y as f64) - ref_c;
        for x in 0..w {
            let src = x as f64 + shift;
            let x0 = src.floor() as isize;
            let f = (src - x0 as f64) as f32;
            let i0 = x0.clamp(0, (w - 1) as isize) as usize;
            let i1 = (x0 + 1).clamp(0, (w - 1) as isize) as usize;
            prof[x] += (row[i0] * (1.0 - f) + row[i1] * f) as f64;
        }
        n += 1.0;
    }
    if n > 0.0 {
        for v in &mut prof {
            *v /= n;
        }
    }
    prof
}

/// Detect absorption lines on the de-smiled mean spectrum, deepest first,
/// excluding the primary core (within `min_sep_px`). Used by multi-line
/// composite reconstruction to pick companion lines.
///
/// Detection runs on a continuum-normalized residual (wide Gaussian / flux)
/// so companions sitting on the wings of a deeper line still form clear
/// local minima, then each candidate is re-fit on the raw de-smiled profile.
pub fn detect_companion_line_centers(
    mean_img: &Image,
    geom: &LineGeometry,
    primary_center: f64,
    min_depth: f64,
    min_sep_px: f64,
    max_companions: usize,
) -> Vec<LineFit1d> {
    let prof = mean_spectrum_desmiled(mean_img, geom);
    if prof.len() < 16 {
        return Vec::new();
    }
    // Wide continuum envelope: companions on primary wings appear as dips
    // in residual = flux / continuum.
    let continuum = gaussian_smooth(&prof, 22.0);
    let residual: Vec<f64> = prof
        .iter()
        .zip(&continuum)
        .map(|(&f, &c)| if c > 1e-6 { f / c } else { 1.0 })
        .collect();
    // min_depth on residual is relative to local continuum (~1.0).
    let mut lines = fit_lines_1d(&residual, min_depth.max(0.02));
    // Re-fit accepted centres on the raw profile for true depth/FWHM.
    let mut refined: Vec<LineFit1d> = Vec::new();
    for l in lines.drain(..) {
        if (l.center - primary_center).abs() < min_sep_px {
            continue;
        }
        let seed = l.center.round() as usize;
        if seed < 4 || seed + 4 >= prof.len() {
            continue;
        }
        // Local inverted-Gaussian on raw spectrum around the residual min.
        let half = 10.min(seed).min(prof.len() - 1 - seed);
        let (a, b) = (seed - half, seed + half);
        let win_x: Vec<f64> = (a..=b).map(|i| i as f64).collect();
        let win_y: Vec<f64> = (a..=b).map(|i| prof[i]).collect();
        if let Some((mu, sigma, amp, off)) =
            fit_inverted_gaussian(&win_x, &win_y, seed as f64, 2.5)
        {
            if mu > a as f64
                && mu < b as f64
                && (0.7..=40.0).contains(&sigma)
                && off > 1e-9
            {
                let depth = (amp / off).clamp(0.0, 1.0);
                if depth >= min_depth * 0.5 {
                    refined.push(LineFit1d {
                        center: mu,
                        sigma,
                        fwhm: FWHM_PER_SIGMA * sigma,
                        depth,
                        continuum: off,
                    });
                }
            }
        } else if l.depth >= min_depth {
            // Residual detection only — keep residual fit translated to
            // absolute column (depth is residual-relative).
            refined.push(LineFit1d {
                center: l.center,
                sigma: l.sigma,
                fwhm: l.fwhm,
                depth: l.depth,
                continuum: continuum
                    .get(seed)
                    .copied()
                    .unwrap_or(1.0),
            });
        }
    }
    refined.sort_by(|a, b| {
        b.depth
            .partial_cmp(&a.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    refined.truncate(max_companions);
    refined
}

/// Comparison-baseline variant: whole-pixel argmin + 2-pass sigma-clipped
/// unweighted quadratic fit (mirrors Inti_recon.py).
pub fn fit_line_geometry_baseline(mean_img: &Image) -> Option<LineGeometry> {
    let (y1, y2) = detect_spectrum_rows(mean_img);
    let marge = 30usize;
    let (ya, yb) = (y1 + marge, y2.saturating_sub(marge));
    if yb <= ya + 10 {
        return None;
    }
    // Same lit-band gate as the main fitter — INTI files rarely have dark
    // spectral margins, but full-sensor captures do.
    let (x_lit0, x_lit1) = detect_spectrum_cols(mean_img);
    let x_pad = ((x_lit1.saturating_sub(x_lit0)) / 40).clamp(12, 80);
    let lo = (x_lit0 + x_pad).max(4);
    let hi = x_lit1
        .saturating_sub(x_pad)
        .min(mean_img.w.saturating_sub(4))
        .max(lo + 16);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for y in ya..yb {
        let row = mean_img.row(y);
        let mut xmin = lo;
        let mut vmin = f32::MAX;
        for x in lo..hi {
            let v = row[x];
            if v < vmin {
                vmin = v;
                xmin = x;
            }
        }
        xs.push(y as f64);
        ys.push(xmin as f64); // integer position: this is the INTI flaw
    }
    let mut mask: Vec<bool> = vec![true; xs.len()];
    let mut coeffs = vec![0.0; 3];
    for _ in 0..2 {
        let fx: Vec<f64> = xs.iter().zip(&mask).filter(|(_, &m)| m).map(|(&x, _)| x).collect();
        let fy: Vec<f64> = ys.iter().zip(&mask).filter(|(_, &m)| m).map(|(&y, _)| y).collect();
        let fw = vec![1.0; fx.len()];
        coeffs = crate::mathutil::polyfit_weighted(&fx, &fy, &fw, 2)?;
        let res: Vec<f64> = xs.iter().zip(&ys).map(|(&x, &y)| y - polyval(&coeffs, x)).collect();
        let std = {
            let sel: Vec<f64> = res.iter().zip(&mask).filter(|(_, &m)| m).map(|(&r, _)| r).collect();
            let mean = sel.iter().sum::<f64>() / sel.len() as f64;
            (sel.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / sel.len() as f64).sqrt()
        };
        for i in 0..mask.len() {
            mask[i] = res[i].abs() < 6.0 * std;
        }
    }
    Some(LineGeometry { coeffs, y1, y2, rms: 0.0, n_rows_used: xs.len() })
}

/// Sub-pixel fit of the deepest absorption line in a single 1-D spectrum.
#[derive(Clone, Copy, Debug)]
pub struct LineFit1d {
    /// Sub-pixel column of the line core.
    pub center: f64,
    /// Gaussian sigma of the line, in pixels.
    pub sigma: f64,
    /// Full width at half maximum, in pixels (2.3548·sigma).
    pub fwhm: f64,
    /// Relative line depth (continuum − core)/continuum, in 0..1.
    pub depth: f64,
    /// Fitted continuum level (flux units).
    pub continuum: f64,
}

/// FWHM / sigma for a Gaussian: 2·sqrt(2·ln 2).
pub const FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949;

/// Find the deepest absorption line in a 1-D spectrum and fit it sub-pixel with
/// the same inverted-Gaussian estimator the pipeline uses per row. This is the
/// single source of truth for line width, shared by live focusing (minimise
/// `fwhm`) and the `spectrum` diagnostic — so the number at the telescope is the
/// number in the reports. `min_depth` gates out noise (relative, e.g. 0.03).
pub fn fit_line_1d(profile: &[f64], min_depth: f64) -> Option<LineFit1d> {
    let n = profile.len();
    if n < 9 {
        return None;
    }
    // Coarse minimum on a lightly smoothed copy (robust to per-sample noise);
    // the Gaussian fit itself runs on the raw samples for sub-pixel accuracy.
    let sm = gaussian_smooth(profile, 1.5);
    let mut sorted = profile.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cont0 = sorted[((0.90 * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    if cont0 <= 1e-9 {
        return None;
    }
    let margin = 3usize;
    let mut xmin = margin;
    let mut vmin = f64::MAX;
    for i in margin..n - margin {
        if sm[i] < vmin {
            vmin = sm[i];
            xmin = i;
        }
    }
    if (cont0 - profile[xmin]) / cont0 < min_depth {
        return None;
    }
    // Symmetric window around the core for the fit (up to ±10 px, edge-limited).
    let half = 10.min(xmin).min(n - 1 - xmin);
    if half < 3 {
        return None;
    }
    let (a, b) = (xmin - half, xmin + half);
    let win_x: Vec<f64> = (a..=b).map(|i| i as f64).collect();
    let win_y: Vec<f64> = (a..=b).map(|i| profile[i]).collect();
    let (mu, sigma, amp, off) = fit_inverted_gaussian(&win_x, &win_y, xmin as f64, 2.5)?;
    // Reject sub-resolution fits: any real line spans several pixels, so a fit
    // that pins near the fitter's minimum-sigma clamp is a single-pixel noise
    // spike, not a line. (Without this a noisy frame yields a spurious
    // FWHM ≈ 0.71 px = 2.3548·0.3 that poisons the min-hold.)
    if !(mu > a as f64 && mu < b as f64) || sigma < 0.7 || sigma > 40.0 || off <= 1e-9 {
        return None;
    }
    Some(LineFit1d {
        center: mu,
        sigma,
        fwhm: FWHM_PER_SIGMA * sigma,
        depth: (amp / off).clamp(0.0, 1.0),
        continuum: off,
    })
}

/// Detect and sub-pixel-fit *every* absorption line in a 1-D spectrum whose
/// relative depth exceeds `min_depth`, deepest-independent. Lets a caller pick
/// the narrowest line (best focus reference) or one nearest a chosen position,
/// instead of only the single deepest dip that [`fit_line_1d`] returns.
pub fn fit_lines_1d(profile: &[f64], min_depth: f64) -> Vec<LineFit1d> {
    let n = profile.len();
    if n < 9 {
        return Vec::new();
    }
    let sm = gaussian_smooth(profile, 1.5);
    let mut sorted = profile.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cont0 = sorted[((0.90 * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    if cont0 <= 1e-9 {
        return Vec::new();
    }
    const NEIGH: usize = 3;
    let margin = 3usize;
    let mut out: Vec<LineFit1d> = Vec::new();
    let mut i = margin;
    while i + margin < n {
        // Local minimum on the smoothed profile, deep enough to be a line.
        let is_min = (1..=NEIGH).all(|k| sm[i] <= sm[i - k])
            && (1..=NEIGH).all(|k| sm[i] <= sm[i + k]);
        if is_min && (cont0 - profile[i]) / cont0 >= min_depth {
            let half = 10.min(i).min(n - 1 - i);
            if half >= 3 {
                let (a, b) = (i - half, i + half);
                let win_x: Vec<f64> = (a..=b).map(|j| j as f64).collect();
                let win_y: Vec<f64> = (a..=b).map(|j| profile[j]).collect();
                if let Some((mu, sigma, amp, off)) =
                    fit_inverted_gaussian(&win_x, &win_y, i as f64, 2.5)
                {
                    if mu > a as f64 && mu < b as f64 && (0.7..=40.0).contains(&sigma) && off > 1e-9 {
                        out.push(LineFit1d {
                            center: mu,
                            sigma,
                            fwhm: FWHM_PER_SIGMA * sigma,
                            depth: (amp / off).clamp(0.0, 1.0),
                            continuum: off,
                        });
                    }
                }
            }
            i += NEIGH; // step past this minimum
        } else {
            i += 1;
        }
    }
    // Merge near-coincident detections (keep the deeper).
    out.sort_by(|a, b| a.center.partial_cmp(&b.center).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<LineFit1d> = Vec::new();
    for f in out {
        if let Some(last) = merged.last_mut() {
            if (f.center - last.center).abs() < 2.0 {
                if f.depth > last.depth {
                    *last = f;
                }
                continue;
            }
        }
        merged.push(f);
    }
    merged
}

#[cfg(test)]
mod orientation_tests {
    use super::*;
    use crate::image2d::Image;

    /// 160 wide (dispersion) x 600 tall (slit): a narrow absorption line on the
    /// dispersion axis, and a dark band at one end of the SLIT -- the solar limb
    /// sitting near the slit end when the disc does not fill it.
    fn line_plus_slit_edge(dark_from: usize) -> Image {
        let (w, h) = (160usize, 600usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut v = 3000.0f32;
                // Narrow line at x = 80, ~6 px sigma: recovers on BOTH sides.
                let d = (x as f32 - 80.0) / 6.0;
                v *= 1.0 - 0.6 * (-0.5 * d * d).exp();
                // Illumination step along the slit: recovers on ONE side only.
                if y >= dark_from {
                    v *= 0.02;
                }
                img.set(x, y, v);
            }
        }
        img
    }

    #[test]
    fn slit_end_darkness_is_not_mistaken_for_a_spectral_line() {
        // The disc leaves the last 60 rows of the slit dark. Scoring on depth
        // alone read that step as a 98% "absorption line" and transposed the
        // frame, which collapsed limb detection and produced a radius nine
        // times too large.
        let img = line_plus_slit_edge(540);
        let (transpose, native, transposed) = should_transpose_for_dispersion(&img);
        assert!(!transpose, "native {native:.3} vs transposed {transposed:.3}");
        assert!(native > 0.4, "the real line should still score: {native:.3}");
        assert!(
            transposed < native,
            "a one-sided step must not outscore a real line: {transposed:.3}"
        );
    }

    #[test]
    fn a_fully_lit_slit_is_unaffected() {
        // No step at all: the same frame must still choose native.
        let img = line_plus_slit_edge(usize::MAX);
        let (transpose, native, _) = should_transpose_for_dispersion(&img);
        assert!(!transpose);
        assert!(native > 0.4, "{native:.3}");
    }

    #[test]
    fn real_frame_geometry_scores_native_despite_dust_spots_and_dark_ends() {
        // The user's capture in pipeline orientation (dispersion along width
        // after the SER transpose): a ~200 px crop around Halpha, slit along
        // height. The line core is narrow; dust is a few dark ROWS (fixed slit
        // positions, all wavelengths); a sunspot is a 150-row darker band; the
        // slit ends are dark. Native must win, and the sunspot dip -- which
        // recovers on both sides, unlike the step -- must be rejected by the
        // recovery WINDOW, not by one-sidedness.
        let (w, h) = (200usize, 600usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut v = 3000.0f32;
                let d = (x as f32 - 100.0) / 7.0; // core FWHM ~16 px
                v *= 1.0 - 0.65 * (-0.5 * d * d).exp();
                if (300..450).contains(&y) {
                    v *= 0.75; // sunspot band along the slit
                }
                for y0 in [120usize, 480, 520] {
                    if y >= y0 && y < y0 + 5 {
                        v *= 0.88; // dust rows
                    }
                }
                if y < 60 || y >= 560 {
                    v *= 0.02; // disc short of the slit ends
                }
                img.set(x, y, v);
            }
        }
        let (transpose, native, transposed) = should_transpose_for_dispersion(&img);
        assert!(!transpose, "native {native:.3} vs transposed {transposed:.3}");
        assert!(native > 0.4, "narrow real line must score: {native:.3}");
        assert!(
            transposed < 0.3,
            "slit-axis structure must stay below the real line: {transposed:.3}"
        );
    }

    #[test]
    fn a_genuine_line_on_the_other_axis_still_transposes() {
        // Guard against over-correcting: a real narrow line along the HEIGHT
        // axis must still be detected and the frame transposed.
        let (w, h) = (600usize, 160usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let d = (y as f32 - 80.0) / 6.0;
                img.set(x, y, 3000.0 * (1.0 - 0.6 * (-0.5 * d * d).exp()));
            }
        }
        let (transpose, native, transposed) = should_transpose_for_dispersion(&img);
        assert!(transpose, "native {native:.3} vs transposed {transposed:.3}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_multiple_lines_and_narrowest() {
        // Two lines: a deep broad one (σ4) and a shallow sharp one (σ1.5).
        let prof: Vec<f64> = (0..120)
            .map(|i| {
                let x = i as f64;
                let broad = 800.0 * (-(x - 35.0).powi(2) / (2.0 * 4.0 * 4.0)).exp();
                let sharp = 300.0 * (-(x - 85.0).powi(2) / (2.0 * 1.5 * 1.5)).exp();
                1000.0 - broad - sharp
            })
            .collect();
        let lines = fit_lines_1d(&prof, 0.03);
        assert!(lines.len() >= 2, "found {} lines", lines.len());
        let narrowest = lines.iter().min_by(|a, b| a.fwhm.partial_cmp(&b.fwhm).unwrap()).unwrap();
        assert!((narrowest.center - 85.0).abs() < 0.5, "narrowest at {}", narrowest.center);
        let deepest = lines.iter().max_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap()).unwrap();
        assert!((deepest.center - 35.0).abs() < 0.5, "deepest at {}", deepest.center);
    }

    #[test]
    fn ignores_dark_spectral_margins() {
        // Full-sensor style: dark left/right margins + bright continuum with
        // one real absorption line near the middle. Global argmin would pick
        // the left edge; lit-band gating must find the real line.
        let (w, h) = (200usize, 80usize);
        let mut img = Image::new(w, h);
        let line_x = 120.0;
        for y in 0..h {
            for x in 0..w {
                let v = if x < 25 || x > 175 {
                    50.0 // dark margin
                } else {
                    let dx = x as f64 - line_x;
                    let abs = 0.35 * (-dx * dx / (2.0 * 2.5 * 2.5)).exp();
                    1000.0 * (1.0 - abs)
                };
                img.set(x, y, v as f32);
            }
        }
        let (x0, x1) = detect_spectrum_cols(&img);
        assert!(x0 > 10 && x1 < 190, "lit band should exclude margins, got [{x0},{x1}]");
        let geom = fit_line_geometry(&img, 2).expect("geometry");
        let ymid = 0.5 * (geom.y1 + geom.y2) as f64;
        let cx = polyval(&geom.coeffs, ymid);
        assert!(
            (cx - line_x).abs() < 3.0,
            "expected line near {line_x}, got {cx} (would be ~12 if margin won)"
        );
    }

    #[test]
    fn detect_companions_on_desmiled_multi_line_image() {
        // Synthetic mean frame: dispersion horizontal, two vertical absorption
        // lines with mild smile (quadratic). Primary deep, companion shallower.
        let (w, h) = (160usize, 120usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            let yf = y as f64;
            let smile = 0.00005 * (yf - 60.0).powi(2);
            let c1 = 45.0 + smile;
            let c2 = 110.0 + smile;
            for x in 0..w {
                let xf = x as f64;
                let d1 = (-(xf - c1).powi(2) / (2.0 * 2.5 * 2.5)).exp();
                let d2 = (-(xf - c2).powi(2) / (2.0 * 2.0 * 2.0)).exp();
                let v = 1000.0 * (1.0 - 0.55 * d1 - 0.25 * d2);
                img.set(x, y, v as f32);
            }
        }
        let geom = fit_line_geometry(&img, 2).expect("primary geometry");
        let ymid = 0.5 * (geom.y1 + geom.y2) as f64;
        let primary = polyval(&geom.coeffs, ymid);
        assert!(
            (primary - 45.0).abs() < 2.0,
            "primary near 45, got {primary}"
        );
        let comps = detect_companion_line_centers(&img, &geom, primary, 0.08, 12.0, 3);
        assert!(
            !comps.is_empty(),
            "expected companion near 110, found none"
        );
        assert!(
            (comps[0].center - 110.0).abs() < 3.0,
            "companion at {}, want ~110",
            comps[0].center
        );
    }

    #[test]
    fn dispersion_axis_from_absorption_contrast() {
        // Landscape: dispersion along width, deep absorption line → no transpose.
        // Width large enough for ±150 px continuum samples used by deep-line score.
        let (w, h) = (400usize, 80usize);
        let mut landscape = crate::image2d::Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - 200.0;
                let line = 0.70 * (-dx * dx / (2.0 * 4.0 * 4.0)).exp();
                landscape.set(x, y, (1000.0 * (1.0 - line)) as f32);
            }
        }
        let (t, d_nat, d_tr) = should_transpose_for_dispersion(&landscape);
        assert!(
            !t,
            "landscape spectrum should not transpose (native_depth={d_nat:.3} transp_depth={d_tr:.3})"
        );
        assert!(d_nat > 0.4, "should detect deep line on width axis, got {d_nat}");
        let portrait = landscape.transpose();
        let (t2, d_nat2, d_tr2) = should_transpose_for_dispersion(&portrait);
        assert!(
            t2,
            "portrait spectrum should transpose (native_depth={d_nat2:.3} transp_depth={d_tr2:.3})"
        );
    }

    #[test]
    fn recovers_known_line_width() {
        // Synthetic absorption line: continuum 1000, sigma 3.0 px, centre 40.7.
        let (cont, sigma, mu, amp) = (1000.0_f64, 3.0_f64, 40.7_f64, 700.0_f64);
        let prof: Vec<f64> = (0..80)
            .map(|i| {
                let dx = i as f64 - mu;
                cont - amp * (-dx * dx / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let f = fit_line_1d(&prof, 0.03).expect("line found");
        assert!((f.center - mu).abs() < 0.05, "center {} vs {mu}", f.center);
        assert!((f.sigma - sigma).abs() < 0.05, "sigma {} vs {sigma}", f.sigma);
        assert!((f.fwhm - FWHM_PER_SIGMA * sigma).abs() < 0.15);
    }
}
