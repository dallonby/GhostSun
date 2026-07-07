//! Disk reconstruction: sample every frame along the fitted line position.
//!
//! GhostSun mode: exact cubic B-spline interpolation (with prefiltering) per
//! row, full f32 output — no intermediate quantization, no ad-hoc smoothing.
//!
//! Baseline mode reproduces INTI: Catmull-Rom on 4 raw samples, blended
//! through the hard-coded 7-tap kernel [-2,3,6,7,6,3,-2]/21 at half-pixel
//! offsets, then rounded to uint16.

use crate::image2d::Image;
use crate::linefit::LineGeometry;
use crate::mathutil::{bspline_eval, bspline_prefilter, polyval};
use crate::ser::SerReader;
use rayon::prelude::*;

pub struct ExtractOptions {
    pub shift: f64, // wavelength offset in px from line center
    pub baseline: bool,
    pub transpose_input: bool, // dispersion vertical in SER -> transpose
    /// Optional Gaussian spectral window sigma (px) for noise/bandpass
    /// trade-off; 0 = pure point sampling.
    pub window_sigma: f64,
    /// Optional per-frame additive spectral offset (F3 flexure), px.
    pub frame_offsets: Option<Vec<f64>>,
}

/// Reconstruct the disk: output image has width = n_frames, height = slit_h.
pub fn reconstruct_disk(reader: &SerReader, geom: &LineGeometry, opts: &ExtractOptions) -> Image {
    let n = reader.header.frame_count;
    let slit_h = if opts.transpose_input { reader.header.width } else { reader.header.height };
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
                extract_column(&frame, &pos, opts.baseline, opts.window_sigma)
            } else {
                let shifted: Vec<f64> = pos.iter().map(|p| p + off).collect();
                extract_column(&frame, &shifted, opts.baseline, opts.window_sigma)
            }
        })
        .collect();

    for (t, col) in cols.iter().enumerate() {
        disk.set_column(t, col);
    }
    disk
}

fn extract_column(frame: &Image, pos: &[f64], baseline: bool, window_sigma: f64) -> Vec<f32> {
    let w = frame.w;
    let mut out = vec![0.0f32; frame.h];
    if baseline {
        // INTI: Catmull-Rom + 7-tap quadratic SG kernel at 0.5 px spacing
        const COEFFS: [f64; 7] = [-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0];
        const OFFSETS: [f64; 7] = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        for y in 0..frame.h {
            let row = frame.row(y);
            let mut acc = 0.0;
            for k in 0..7 {
                let x = pos[y] + OFFSETS[k];
                let xf = x.floor();
                let t = x - xf;
                let i = (xf as isize).clamp(2, w as isize - 3) as usize;
                let p0 = row[i - 1] as f64;
                let p1 = row[i] as f64;
                let p2 = row[i + 1] as f64;
                let p3 = row[i + 2] as f64;
                let v = 0.5
                    * ((2.0 * p1)
                        + (-p0 + p2) * t
                        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);
                acc += v * COEFFS[k] / 21.0;
            }
            // INTI clamps and rounds to uint16 here — reproduce that loss
            out[y] = acc.clamp(0.0, 65535.0).round() as f32;
        }
    } else {
        // GhostSun: exact B-spline interpolation on each row, keep f32.
        // Optional Gaussian spectral window (a principled version of what
        // INTI's fixed 7-tap kernel tries to do).
        let (offsets, weights): (Vec<f64>, Vec<f64>) = if window_sigma > 0.0 {
            let r = (2.5 * window_sigma).ceil();
            let mut offs = Vec::new();
            let mut wts = Vec::new();
            let mut o = -r;
            while o <= r + 1e-9 {
                offs.push(o);
                wts.push((-(o * o) / (2.0 * window_sigma * window_sigma)).exp());
                o += 0.5;
            }
            let s: f64 = wts.iter().sum();
            for w in wts.iter_mut() {
                *w /= s;
            }
            (offs, wts)
        } else {
            (vec![0.0], vec![1.0])
        };
        let mut coef = vec![0.0f64; w];
        for y in 0..frame.h {
            let row = frame.row(y);
            for (i, &v) in row.iter().enumerate() {
                coef[i] = v as f64;
            }
            bspline_prefilter(&mut coef);
            let mut acc = 0.0;
            for (o, wt) in offsets.iter().zip(&weights) {
                let x = (pos[y] + o).clamp(1.0, (w - 2) as f64);
                acc += wt * bspline_eval(&coef, x);
            }
            out[y] = acc as f32;
        }
    }
    out
}
