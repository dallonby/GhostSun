//! F5: multi-scan registration and stacking.
//!
//! Reconstructions of sequential scans are registered globally (disk fit:
//! scale + translation, NCC-refined), then a stiff block-matching optical
//! flow absorbs solar evolution between scans, and a sharpness-weighted
//! robust mean combines them. Flow must be heavily smoothed or it will
//! "correct" noise into the reference (hallucinated sharpness).

use crate::image2d::Image;
use crate::metrics::{fit_disk, DiskFit};
use nalgebra::{Matrix6, Vector6};
use rayon::prelude::*;

fn bilinear(img: &Image, x: f64, y: f64) -> f32 {
    let xf = x.floor();
    let yf = y.floor();
    let tx = (x - xf) as f32;
    let ty = (y - yf) as f32;
    let xi = xf as isize;
    let yi = yf as isize;
    let v00 = img.at_clamped(xi, yi);
    let v10 = img.at_clamped(xi + 1, yi);
    let v01 = img.at_clamped(xi, yi + 1);
    let v11 = img.at_clamped(xi + 1, yi + 1);
    v00 * (1.0 - tx) * (1.0 - ty) + v10 * tx * (1.0 - ty) + v01 * (1.0 - tx) * ty + v11 * tx * ty
}

/// High-frequency energy (sharpness proxy): variance of (img - blur2).
fn hf_energy(img: &Image, disk: &DiskFit) -> f64 {
    let blur = crate::mathutil::gaussian_blur_2d(img, 2.0, 2.0);
    let mut e = 0.0f64;
    let mut n = 0.0f64;
    for y in (0..img.h).step_by(2) {
        for x in (0..img.w).step_by(2) {
            let dx = x as f64 - disk.xc;
            let dy = y as f64 - disk.yc;
            if (dx * dx + dy * dy).sqrt() < disk.r * 0.9 {
                e += ((img.at(x, y) - blur.at(x, y)) as f64).powi(2);
                n += 1.0;
            }
        }
    }
    e / n.max(1.0)
}

/// Resample `img` onto the reference grid: global scale+translation from the
/// two disk fits plus an extra (dx, dy).
fn to_ref_grid(img: &Image, f: &DiskFit, rf: &DiskFit, size: (usize, usize), dx: f64, dy: f64) -> Image {
    let s = f.r / rf.r;
    let mut out = Image::new(size.0, size.1);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let xs = f.xc + (x as f64 - rf.xc) * s + dx;
            let ys = f.yc + (y as f64 - rf.yc) * s + dy;
            out.set(x, y, bilinear(img, xs, ys));
        }
    }
    out
}

fn ncc(a: &Image, b: &Image, disk: &DiskFit) -> f64 {
    let mut sa = 0.0;
    let mut sb = 0.0;
    let mut n = 0.0;
    let mut idx = Vec::new();
    for y in (0..a.h).step_by(2) {
        for x in (0..a.w).step_by(2) {
            let dx = x as f64 - disk.xc;
            let dy = y as f64 - disk.yc;
            if (dx * dx + dy * dy).sqrt() < disk.r * 0.95 {
                idx.push(y * a.w + x);
                sa += a.data[y * a.w + x] as f64;
                sb += b.data[y * a.w + x] as f64;
                n += 1.0;
            }
        }
    }
    let (ma, mb) = (sa / n, sb / n);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for &i in &idx {
        let va = a.data[i] as f64 - ma;
        let vb = b.data[i] as f64 - mb;
        num += va * vb;
        da += va * va;
        db += vb * vb;
    }
    num / (da * db).sqrt().max(1e-12)
}

/// Stiff block-matching optical flow of `img` relative to `refimg`.
/// Returns per-pixel (fx, fy) fields, heavily smoothed.
fn optical_flow(refimg: &Image, img: &Image, disk: &DiskFit, block: usize, search: isize) -> (Image, Image) {
    let nbx = img.w / block;
    let nby = img.h / block;
    let cells: Vec<(usize, usize, f64, f64, f64)> = (0..nby * nbx)
        .into_par_iter()
        .map(|cell| {
            let bx = cell % nbx;
            let by = cell / nbx;
            let x0 = bx * block;
            let y0 = by * block;
            let cx = x0 as f64 + block as f64 / 2.0 - disk.xc;
            let cy = y0 as f64 + block as f64 / 2.0 - disk.yc;
            if (cx * cx + cy * cy).sqrt() > disk.r {
                return (bx, by, 0.0, 0.0, 0.0); // off-disk: freeze flow
            }
            // NCC over integer shifts
            let score = |sx: isize, sy: isize| -> f64 {
                let mut num = 0.0;
                let mut da = 0.0;
                let mut db = 0.0;
                let mut sa = 0.0;
                let mut sb = 0.0;
                let mut n = 0.0;
                for y in y0..y0 + block {
                    for x in x0..x0 + block {
                        let a = refimg.at(x, y) as f64;
                        let b = img.at_clamped(x as isize + sx, y as isize + sy) as f64;
                        sa += a;
                        sb += b;
                        n += 1.0;
                        num += a * b;
                        da += a * a;
                        db += b * b;
                    }
                }
                let cov = num - sa * sb / n;
                let va = da - sa * sa / n;
                let vb = db - sb * sb / n;
                cov / (va * vb).sqrt().max(1e-9)
            };
            let mut best = (0isize, 0isize, f64::MIN);
            for sy in -search..=search {
                for sx in -search..=search {
                    let v = score(sx, sy);
                    if v > best.2 {
                        best = (sx, sy, v);
                    }
                }
            }
            if best.2 < 0.5 || best.0.abs() == search || best.1.abs() == search {
                return (bx, by, 0.0, 0.0, 0.0);
            }
            // parabolic sub-pixel in each axis
            let sub = |m: f64, c: f64, p: f64| -> f64 {
                let den = m - 2.0 * c + p;
                if den < -1e-12 { (0.5 * (m - p) / den).clamp(-0.6, 0.6) } else { 0.0 }
            };
            let fx = best.0 as f64 + sub(score(best.0 - 1, best.1), best.2, score(best.0 + 1, best.1));
            let fy = best.1 as f64 + sub(score(best.0, best.1 - 1), best.2, score(best.0, best.1 + 1));
            (bx, by, fx, fy, best.2)
        })
        .collect();

    // fill cell grids, Tukey-clip against the median, smooth, upsample
    let mut gx = vec![0.0f64; nbx * nby];
    let mut gy = vec![0.0f64; nbx * nby];
    for &(bx, by, fx, fy, _) in &cells {
        gx[by * nbx + bx] = fx;
        gy[by * nbx + bx] = fy;
    }
    let clip = |g: &mut Vec<f64>| {
        let mut v = g.clone();
        let med = crate::mathutil::median_inplace(&mut v);
        for x in g.iter_mut() {
            if (*x - med).abs() > 2.0 {
                *x = med;
            }
        }
    };
    clip(&mut gx);
    clip(&mut gy);
    // smooth the cell grid (separable)
    let smooth_grid = |g: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; g.len()];
        for by in 0..nby {
            let row: Vec<f64> = (0..nbx).map(|bx| g[by * nbx + bx]).collect();
            let sm = crate::mathutil::gaussian_smooth(&row, 1.5);
            for bx in 0..nbx {
                out[by * nbx + bx] = sm[bx];
            }
        }
        let mut out2 = vec![0.0; g.len()];
        for bx in 0..nbx {
            let col: Vec<f64> = (0..nby).map(|by| out[by * nbx + bx]).collect();
            let sm = crate::mathutil::gaussian_smooth(&col, 1.5);
            for by in 0..nby {
                out2[by * nbx + bx] = sm[by];
            }
        }
        out2
    };
    let gx = smooth_grid(&gx);
    let gy = smooth_grid(&gy);

    // bilinear upsample to full resolution, deadband 0.1 px
    let mut fx_img = Image::new(img.w, img.h);
    let mut fy_img = Image::new(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            let u = (x as f64 / block as f64 - 0.5).clamp(0.0, nbx as f64 - 1.0);
            let v = (y as f64 / block as f64 - 0.5).clamp(0.0, nby as f64 - 1.0);
            let (ui, vi) = (u.floor() as usize, v.floor() as usize);
            let (tu, tv) = (u - ui as f64, v - vi as f64);
            let (ui1, vi1) = ((ui + 1).min(nbx - 1), (vi + 1).min(nby - 1));
            let sample = |g: &[f64]| -> f64 {
                g[vi * nbx + ui] * (1.0 - tu) * (1.0 - tv)
                    + g[vi * nbx + ui1] * tu * (1.0 - tv)
                    + g[vi1 * nbx + ui] * (1.0 - tu) * tv
                    + g[vi1 * nbx + ui1] * tu * tv
            };
            let (fx, fy) = (sample(&gx), sample(&gy));
            fx_img.set(x, y, if fx.abs() < 0.1 { 0.0 } else { fx as f32 });
            fy_img.set(x, y, if fy.abs() < 0.1 { 0.0 } else { fy as f32 });
        }
    }
    (fx_img, fy_img)
}

#[allow(dead_code)]
pub struct StackReport {
    pub image: Image,
    pub n_used: usize,
    pub weights: Vec<f64>,
}

/// LS fit of ratio ref/img over the disk as a quadratic surface in
/// normalized disk coords. Returns the 6 coefficients.
fn fit_gain_surface(refimg: &Image, img: &Image, disk: &DiskFit) -> Option<Vector6<f64>> {
    let mut ata = Matrix6::<f64>::zeros();
    let mut atb = Vector6::<f64>::zeros();
    let mut count = 0.0;
    for y in (0..img.h).step_by(4) {
        for x in (0..img.w).step_by(4) {
            let dx = (x as f64 - disk.xc) / disk.r;
            let dy = (y as f64 - disk.yc) / disk.r;
            if (dx * dx + dy * dy).sqrt() >= 0.92 {
                continue;
            }
            let iv = img.at(x, y) as f64;
            let rv = refimg.at(x, y) as f64;
            if iv < 1e-3 {
                continue;
            }
            let ratio = (rv / iv).clamp(0.3, 3.0);
            let basis = Vector6::new(1.0, dx, dy, dx * dx, dx * dy, dy * dy);
            ata += basis * basis.transpose();
            atb += basis * ratio;
            count += 1.0;
        }
    }
    if count < 200.0 {
        return None;
    }
    ata.lu().solve(&atb)
}

fn eval_quad(c: &Vector6<f64>, x: f64, y: f64) -> f64 {
    c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y
}

/// Stack registered reconstructions. `flow` enables evolution compensation.
pub fn stack(images: &[Image], flow: bool, verbose: bool) -> Option<StackReport> {
    stack_with_reference(images, flow, verbose, None)
}

/// Stack with an explicit reference index (None = sharpest scan).
pub fn stack_with_reference(
    images: &[Image],
    flow: bool,
    verbose: bool,
    reference: Option<usize>,
) -> Option<StackReport> {
    if images.is_empty() {
        return None;
    }
    if images.len() == 1 {
        return Some(StackReport { image: images[0].clone(), n_used: 1, weights: vec![1.0] });
    }
    let fits: Vec<DiskFit> = images.iter().map(|i| fit_disk(i)).collect::<Option<Vec<_>>>()?;

    // reference = sharpest scan (or caller-specified)
    let energies: Vec<f64> = images.iter().zip(&fits).map(|(i, f)| hf_energy(i, f)).collect();
    let ref_idx = match reference {
        Some(i) => i.min(images.len() - 1),
        None => energies
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)?,
    };
    let rf = &fits[ref_idx];
    let size = (images[ref_idx].w, images[ref_idx].h);
    let refimg = &images[ref_idx];
    if verbose {
        println!("stack: reference = scan {ref_idx} (hf energy {:.1})", energies[ref_idx]);
    }

    // register each scan: global + NCC translation refine + optional flow
    let mut aligned: Vec<Image> = Vec::new();
    for (k, img) in images.iter().enumerate() {
        if k == ref_idx {
            aligned.push(img.clone());
            continue;
        }
        let mut best = (0.0f64, 0.0f64, f64::MIN);
        for step in [1.0f64, 0.25, 0.05] {
            let (cx, cy, _) = best;
            let mut local = best;
            let mut dy = cy - 2.0 * step;
            while dy <= cy + 2.0 * step + 1e-12 {
                let mut dx = cx - 2.0 * step;
                while dx <= cx + 2.0 * step + 1e-12 {
                    let r = to_ref_grid(img, &fits[k], rf, size, dx, dy);
                    let v = ncc(&r, refimg, rf);
                    if v > local.2 {
                        local = (dx, dy, v);
                    }
                    dx += step;
                }
                dy += step;
            }
            best = local;
        }
        let mut reg = to_ref_grid(img, &fits[k], rf, size, best.0, best.1);
        if flow {
            let (fx, fy) = optical_flow(refimg, &reg, rf, 32, 4);
            let mut warped = Image::new(size.0, size.1);
            for y in 0..size.1 {
                for x in 0..size.0 {
                    warped.set(
                        x,
                        y,
                        bilinear(&reg, x as f64 + fx.at(x, y) as f64, y as f64 + fy.at(x, y) as f64),
                    );
                }
            }
            reg = warped;
        }
        if verbose {
            println!("stack: scan {k} registered (ncc {:.4})", best.2);
        }
        aligned.push(reg);
    }

    // Photometric matching to the reference: each scan carries its own
    // slow transparency residual; a scalar gain leaves large-scale waves
    // that dominate PSNR. Fit a low-order (quadratic) gain surface per scan
    // over the disk and divide it out.
    for a in aligned.iter_mut() {
        if let Some(gain) = fit_gain_surface(refimg, a, rf) {
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let g = eval_quad(&gain, (x as f64 - rf.xc) / rf.r, (y as f64 - rf.yc) / rf.r);
                    let v = a.at(x, y) as f64 * g.clamp(0.5, 2.0);
                    a.set(x, y, v as f32);
                }
            }
        }
    }
    let scale: Vec<f64> = vec![1.0; aligned.len()];

    // sharpness weights (floored)
    let emax = energies.iter().cloned().fold(f64::MIN, f64::max).max(1e-12);
    let weights: Vec<f64> = energies.iter().map(|e| (e / emax).clamp(0.2, 1.0)).collect();

    // robust weighted mean per pixel: reject > 3*MAD from the median
    let k_scans = aligned.len();
    let mut out = Image::new(size.0, size.1);
    let mut vals = vec![0.0f64; k_scans];
    for i in 0..size.0 * size.1 {
        for (k, a) in aligned.iter().enumerate() {
            vals[k] = a.data[i] as f64 * scale[k];
        }
        let mut sorted = vals.clone();
        let med = crate::mathutil::median_inplace(&mut sorted);
        let mut devs: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
        let mad = crate::mathutil::median_inplace(&mut devs).max(1e-6);
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for k in 0..k_scans {
            if (vals[k] - med).abs() < 3.0 * 1.4826 * mad + 1e-3 * med.abs() + 1.0 {
                acc += weights[k] * vals[k];
                wsum += weights[k];
            }
        }
        out.data[i] = if wsum > 0.0 { (acc / wsum) as f32 } else { med as f32 };
    }
    Some(StackReport { image: out, n_used: k_scans, weights })
}
