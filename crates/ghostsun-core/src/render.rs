//! Colorized presentation rendering: Hα-style false color, background
//! extraction to black, prominences preserved and boosted.
//!
//! Background model: for each radial bin outside the limb, the low
//! percentile across position angles is scattered light — prominences are
//! angularly localized so they survive the subtraction. Disk and off-disk
//! get separate tone curves, feathered across the limb.

use crate::image2d::Image;
use crate::metrics::{coarse_disk, fit_disk_polar, DiskFit};

pub struct ColorizeOptions {
    /// prominence brightness boost relative to the disk stretch
    pub prom_boost: f64,
    /// gamma applied to the disk tone curve
    pub gamma: f64,
}

impl Default for ColorizeOptions {
    fn default() -> Self {
        ColorizeOptions { prom_boost: 3.0, gamma: 0.7 }
    }
}

/// Hα false-color lookup: black -> deep red -> orange -> yellow-white.
fn lut(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = (2.6 * t).min(1.0);
    let g = t.powf(1.6);
    let b = t.powi(4) * 0.55;
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Radial sky (scattered light) model outside the limb: per-radius
/// percentiles across angles (prominences are angularly localized and sit
/// in the upper tail, so mid/low percentiles are pure sky), smoothed.
/// Returns per-bin (background, noise sigma).
fn sky_model(img: &Image, disk: &DiskFit) -> (Vec<f64>, Vec<f64>) {
    let rmax = ((img.w.max(img.h)) as f64 * 0.75) as usize;
    let bin = 4usize;
    let nbins = rmax / bin + 2;
    let mut bins: Vec<Vec<f32>> = vec![Vec::new(); nbins];
    for y in (0..img.h).step_by(2) {
        for x in (0..img.w).step_by(2) {
            let dx = x as f64 - disk.xc;
            let dy = y as f64 - disk.yc;
            let r = (dx * dx + dy * dy).sqrt();
            // start well clear of the limb: the spicule rim glow at
            // 1.0-1.1 R is real solar signal present at ALL position
            // angles — sampling it into the "sky" percentile subtracts it
            // and detaches prominences from the disk with a black ring
            if r > disk.r * 1.12 {
                let b = (r as usize) / bin;
                // exact zeros are NO-DATA (canvas edge beyond scan
                // coverage), not sky — including them poisons the
                // percentile and makes sigma explode, gating real
                // prominence signal to black
                let v = img.at(x, y);
                if b < nbins && v > 0.5 {
                    bins[b].push(v);
                }
            }
        }
    }
    let mut prof = vec![f64::NAN; nbins];
    let mut sig = vec![f64::NAN; nbins];
    for (b, v) in bins.iter().enumerate() {
        if v.len() > 30 {
            let p45 = crate::mathutil::percentile_f32(v, 45.0) as f64;
            let p16 = crate::mathutil::percentile_f32(v, 16.0) as f64;
            prof[b] = p45;
            sig[b] = (p45 - p16).max(1e-6); // ~1 sigma of the sky noise
        }
    }
    // fill gaps, light smoothing
    let valid: Vec<usize> = (0..nbins).filter(|&b| prof[b].is_finite()).collect();
    if valid.is_empty() {
        return (vec![0.0; nbins], vec![1.0; nbins]);
    }
    // inner bins (limb rim zone): hold the innermost measured sky value
    // rather than nearest-fill semantics that could grab something else
    for b in 0..nbins {
        if !prof[b].is_finite() {
            let nearest = valid.iter().min_by_key(|&&v| v.abs_diff(b)).unwrap();
            prof[b] = prof[*nearest];
            sig[b] = sig[*nearest];
        }
    }
    (
        crate::mathutil::gaussian_smooth(&prof, 2.0),
        crate::mathutil::gaussian_smooth(&sig, 4.0),
    )
}

/// One-time per-image preparation for colorizing: disk geometry, sky
/// model, normalization. Lets interactive UIs re-render with new tone
/// options at full speed (the per-pixel pass is cheap).
#[derive(Clone)]
pub struct ColorizePrep {
    pub disk: DiskFit,
    bg: Vec<f64>,
    sig: Vec<f64>,
    scale: f64,
}

pub fn prepare(img: &Image) -> Option<ColorizePrep> {
    let init = coarse_disk(img)?;
    let (disk, _) = fit_disk_polar(img, &init)?;
    let (bg, sig) = sky_model(img, &disk);
    // disk normalization level (robust bright end of disk interior)
    let mut disk_vals: Vec<f32> = Vec::new();
    for y in (0..img.h).step_by(3) {
        for x in (0..img.w).step_by(3) {
            let dx = x as f64 - disk.xc;
            let dy = y as f64 - disk.yc;
            if (dx * dx + dy * dy).sqrt() < disk.r * 0.95 {
                disk_vals.push(img.at(x, y));
            }
        }
    }
    if disk_vals.is_empty() {
        return None;
    }
    let peak = crate::mathutil::percentile_f32(&disk_vals, 99.7) as f64;
    let disk_bg = bg.first().cloned().unwrap_or(0.0);
    let scale = (peak - disk_bg).max(1e-6);
    Some(ColorizePrep { disk, bg, sig, scale })
}

/// Render an RGB colorized image. Returns (w, h, rgb bytes).
pub fn colorize(img: &Image, opts: &ColorizeOptions) -> Option<(usize, usize, Vec<u8>)> {
    let prep = prepare(img)?;
    Some(render_with(img, &prep, opts))
}

/// Fast per-pixel colorize pass with precomputed geometry (rayon over rows).
pub fn render_with(img: &Image, prep: &ColorizePrep, opts: &ColorizeOptions) -> (usize, usize, Vec<u8>) {
    use rayon::prelude::*;
    let bin = 4usize;
    let disk = &prep.disk;
    let (bg, sig, scale) = (&prep.bg, &prep.sig, prep.scale);
    let rows: Vec<Vec<u8>> = (0..img.h)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![0u8; img.w * 3];
            for x in 0..img.w {
            let dx = x as f64 - disk.xc;
            let dy = y as f64 - disk.yc;
            let r = (dx * dx + dy * dy).sqrt();
            let b = ((r as usize) / bin).min(bg.len() - 1);
            let raw = img.at(x, y);
            if raw <= 0.5 {
                // no-data (canvas edge): true black
                continue;
            }
            let v = (raw as f64 - bg[b]).max(0.0);

            // disk tone
            let t_disk = (v / scale).clamp(0.0, 1.0).powf(opts.gamma);
            // prominence tone: boosted, soft-saturating, noise-gated so the
            // sky renders black while prominences (well above the noise)
            // pass through untouched. The gate is RADIALLY WEIGHTED: at the
            // limb the faint prominence bases sit only ~1-2 sigma above sky
            // and an active gate paints a black moat between rim and
            // prominence — black-sky enforcement only matters away from the
            // limb, so the gate fades in from 1.02R to ~1.14R.
            let yv = v * opts.prom_boost / scale;
            let raw_gate = smoothstep((v - 1.5 * sig[b]) / (2.0 * sig[b]).max(1e-9));
            let gate_strength = smoothstep((r / disk.r - 1.02) / 0.12);
            let gate = 1.0 - gate_strength * (1.0 - raw_gate);
            let t_prom = (yv / (1.0 + yv)).powf(0.8) * gate;
            // feather across the limb (+-3 px)
            let w_disk = smoothstep((disk.r + 2.0 - r) / 5.0);
            let t = w_disk * t_disk + (1.0 - w_disk) * t_prom;

            let c = lut(t);
            row[x * 3] = c[0];
            row[x * 3 + 1] = c[1];
            row[x * 3 + 2] = c[2];
            }
            row
        })
        .collect();
    let mut rgb = vec![0u8; img.w * img.h * 3];
    for (y, row) in rows.iter().enumerate() {
        rgb[y * img.w * 3..(y + 1) * img.w * 3].copy_from_slice(row);
    }
    (img.w, img.h, rgb)
}
