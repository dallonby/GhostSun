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

/// Per-block NCC displacement measurement of `img` relative to `refimg`:
/// for each on-disk block, the (fx, fy) shift of `img` that best matches the
/// reference, with parabolic sub-pixel refinement and an NCC confidence.
/// Off-disk, low-confidence and search-saturated blocks report (0, 0, 0).
fn block_vectors(
    refimg: &Image,
    img: &Image,
    disk: &DiskFit,
    block: usize,
    search: isize,
) -> Vec<(usize, usize, f64, f64, f64)> {
    let nbx = img.w / block;
    let nby = img.h / block;
    (0..nby * nbx)
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
        .collect()
}

/// Stiff block-matching optical flow of `img` relative to `refimg`.
/// Returns per-pixel (fx, fy) fields, heavily smoothed.
fn optical_flow(refimg: &Image, img: &Image, disk: &DiskFit, block: usize, search: isize) -> (Image, Image) {
    let nbx = img.w / block;
    let nby = img.h / block;
    let cells = block_vectors(refimg, img, disk, block, search);

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
    /// Scans that arrived x-mirrored and were auto-flipped to match the
    /// reference. Nonzero means direction/flip bookkeeping upstream is wrong,
    /// even though the data was recovered.
    pub n_flipped: usize,
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

/// Combine images that are already on a shared canvas (e.g. multi-line
/// extractions warped with the same geometry). No disk re-fit or optical
/// flow — only per-image median gain matching to the reference + robust
/// sharpness-weighted mean.
pub fn stack_coregistered(images: &[Image], reference: usize) -> Option<StackReport> {
    if images.is_empty() {
        return None;
    }
    if images.len() == 1 {
        return Some(StackReport {
            n_flipped: 0,
            image: images[0].clone(),
            n_used: 1,
            weights: vec![1.0],
        });
    }
    let w = images[0].w;
    let h = images[0].h;
    if images.iter().any(|im| im.w != w || im.h != h) {
        return None;
    }
    let ref_idx = reference.min(images.len() - 1);
    let refimg = &images[ref_idx];

    // Per-image median gain vs reference over bright pixels.
    let mut aligned: Vec<Image> = Vec::with_capacity(images.len());
    let mut energies = Vec::with_capacity(images.len());
    for (k, img) in images.iter().enumerate() {
        let mut gimg = img.clone();
        if k != ref_idx {
            let mut ratios = Vec::new();
            let step = 4usize;
            for y in (0..h).step_by(step) {
                for x in (0..w).step_by(step) {
                    let a = refimg.at(x, y) as f64;
                    let b = img.at(x, y) as f64;
                    if a > 500.0 && b > 500.0 {
                        ratios.push(a / b);
                    }
                }
            }
            if ratios.len() > 50 {
                ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let g = ratios[ratios.len() / 2].clamp(0.25, 4.0) as f32;
                for v in gimg.data.iter_mut() {
                    *v *= g;
                }
            }
        }
        // HF energy as sharpness proxy (full image, no disk fit required).
        let blur = crate::mathutil::gaussian_blur_2d(&gimg, 2.0, 2.0);
        let mut e = 0.0f64;
        let mut n = 0.0f64;
        for i in (0..gimg.data.len()).step_by(4) {
            e += ((gimg.data[i] - blur.data[i]) as f64).powi(2);
            n += 1.0;
        }
        energies.push(e / n.max(1.0));
        aligned.push(gimg);
    }
    let emax = energies.iter().cloned().fold(f64::MIN, f64::max).max(1e-12);
    let weights: Vec<f64> = energies.iter().map(|e| (e / emax).clamp(0.25, 1.0)).collect();

    let k_scans = aligned.len();
    let mut out = Image::new(w, h);
    let mut vals = vec![0.0f64; k_scans];
    for i in 0..w * h {
        for (k, a) in aligned.iter().enumerate() {
            vals[k] = a.data[i] as f64;
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
        out.data[i] = if wsum > 0.0 {
            (acc / wsum) as f32
        } else {
            med as f32
        };
    }
    Some(StackReport {
        n_flipped: 0,
        image: out,
        n_used: k_scans,
        weights,
    })
}

/// Stack with an explicit reference index (None = sharpest scan).
/// Minimum NCC against the reference for a scan to join the stack in a given
/// orientation. Correct pairs measure ~0.999 and even photometrically poor
/// scans stay above ~0.95; a mirrored scan measures ~0.2 — below this it is
/// re-tried flipped before being given up on.
const NCC_MIN: f64 = 0.85;

/// Horizontal mirror, for re-trying a scan whose flip bookkeeping was wrong.
fn mirror_x(img: &Image) -> Image {
    let mut out = Image::new(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            out.set(x, y, img.at(img.w - 1 - x, y));
        }
    }
    out
}

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
        return Some(StackReport { n_flipped: 0, image: images[0].clone(), n_used: 1, weights: vec![1.0] });
    }
    // One unfittable scan used to abort the whole stack through `?`. That is the
    // wrong trade: dropping a single scan of N costs a little SNR, whereas
    // returning None discards every scan including the good ones -- and to the
    // caller it is indistinguishable from "no multi-scan result", which is how
    // an over-large dither presented itself.
    let fits: Vec<Option<DiskFit>> = images.iter().map(fit_disk).collect();
    let energies: Vec<f64> = images
        .iter()
        .zip(&fits)
        .map(|(i, f)| f.as_ref().map(|f| hf_energy(i, f)).unwrap_or(0.0))
        .collect();

    // reference = sharpest scan (or caller-specified); it must itself be fittable
    let fitted = |i: &usize| fits[*i].is_some();
    let ref_idx = match reference {
        Some(i) => {
            let i = i.min(images.len() - 1);
            if fits[i].is_some() {
                i
            } else {
                (0..images.len()).find(fitted)?
            }
        }
        None => (0..images.len())
            .filter(fitted)
            .max_by(|a, b| energies[*a].partial_cmp(&energies[*b]).unwrap())?,
    };
    let rf = fits[ref_idx].as_ref()?;
    let size = (images[ref_idx].w, images[ref_idx].h);
    let refimg = &images[ref_idx];
    if verbose {
        println!("stack: reference = scan {ref_idx} (hf energy {:.1})", energies[ref_idx]);
    }

    // register each scan: global + NCC translation refine + optional flow
    let mut aligned: Vec<Image> = Vec::new();
    // Indices actually stacked, so the sharpness weights stay in step with
    // `aligned` once scans can be dropped.
    let mut kept: Vec<usize> = Vec::new();
    let mut n_flipped = 0usize;
    for (k, img) in images.iter().enumerate() {
        if k == ref_idx {
            aligned.push(img.clone());
            kept.push(k);
            continue;
        }
        let Some(fk) = fits[k].as_ref() else {
            if verbose {
                println!("stack: scan {k} dropped (no disk fit)");
            }
            continue;
        };
        let search = |im: &Image, fit: &DiskFit| {
            let mut best = (0.0f64, 0.0f64, f64::MIN);
            for step in [1.0f64, 0.25, 0.05] {
                let (cx, cy, _) = best;
                let mut local = best;
                let mut dy = cy - 2.0 * step;
                while dy <= cy + 2.0 * step + 1e-12 {
                    let mut dx = cx - 2.0 * step;
                    while dx <= cx + 2.0 * step + 1e-12 {
                        let r = to_ref_grid(im, fit, rf, size, dx, dy);
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
            best
        };
        let mut best = search(img, fk);
        // A mirrored scan (direction/flip bookkeeping gone wrong) registers at
        // NCC ~0.2 against ~0.999 for a correct pair, and the robust mean
        // cannot reject a wholesale mirror -- with two scans the MAD clip has
        // no majority -- so unhandled it silently blends into a plausible-
        // looking 22 dB result. But orientation is DETERMINISTIC, not noise:
        // sweep direction is known upstream, and even when its bookkeeping
        // fails the mirror is trivially recoverable. So recover first -- flip
        // and re-register, with the enormous NCC margin as confirmation -- and
        // drop only a scan that correlates in NEITHER orientation, which is
        // genuinely bad data rather than a labelling mistake.
        let mut flipped_src: Option<(Image, DiskFit)> = None;
        if best.2 < NCC_MIN {
            let mirrored = mirror_x(img);
            let mfit = DiskFit {
                xc: (img.w as f64 - 1.0) - fk.xc,
                yc: fk.yc,
                r: fk.r,
            };
            let mbest = search(&mirrored, &mfit);
            if mbest.2 >= NCC_MIN {
                if verbose {
                    println!(
                        "stack: scan {k} arrived x-mirrored (ncc {:.3} → {:.3});                          auto-flipped — check direction/flip bookkeeping",
                        best.2, mbest.2
                    );
                }
                n_flipped += 1;
                best = mbest;
                flipped_src = Some((mirrored, mfit));
            } else {
                if verbose {
                    println!(
                        "stack: scan {k} correlates in neither orientation                          (ncc {:.3} direct, {:.3} mirrored) — dropped",
                        best.2, mbest.2
                    );
                }
                continue;
            }
        }
        let (src_img, src_fit) = flipped_src
            .as_ref()
            .map(|(i, f)| (i, f))
            .unwrap_or((img, fk));
        let mut reg = to_ref_grid(src_img, src_fit, rf, size, best.0, best.1);
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
        kept.push(k);
    }
    if aligned.len() < 2 {
        return None;
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
    let emax = kept.iter().map(|&k| energies[k]).fold(f64::MIN, f64::max).max(1e-12);
    let weights: Vec<f64> = kept.iter().map(|&k| (energies[k] / emax).clamp(0.2, 1.0)).collect();

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
    Some(StackReport { n_flipped, image: out, n_used: k_scans, weights })
}

// ---------------------------------------------------------------------------
// F5.5: native-domain stacking — one resample from acquisition to stack.
//
// The classic path warps each scan (Lanczos), registers with bilinear
// resampling, optionally flows with another bilinear pass: 2-3 cumulative
// interpolations of already-interpolated data. Here the inter-scan
// registration (similarity + residual affine + optional evolution flow) is
// COMPOSED with the per-scan geometric warp, and the corrected native disk
// is sampled exactly once, with the same anisotropic Lanczos-3 the single
// warp uses. Per-native-column quality weights (burst severity,
// transparency) ride along through the same transform.
// ---------------------------------------------------------------------------

use crate::ellipse::EllipseGeom;
use crate::warp::sample_lanczos3_aniso;

/// Row-major 2x3 affine: (x, y) -> (m0 x + m1 y + m2, m3 x + m4 y + m5).
#[derive(Clone, Copy, Debug)]
pub struct Affine2 {
    pub m: [f64; 6],
}

impl Affine2 {
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.m[0] * x + self.m[1] * y + self.m[2],
            self.m[3] * x + self.m[4] * y + self.m[5],
        )
    }

    /// self ∘ inner: apply `inner` first, then `self`.
    pub fn compose(&self, inner: &Affine2) -> Affine2 {
        let a = &self.m;
        let b = &inner.m;
        Affine2 {
            m: [
                a[0] * b[0] + a[1] * b[3],
                a[0] * b[1] + a[1] * b[4],
                a[0] * b[2] + a[1] * b[5] + a[2],
                a[3] * b[0] + a[4] * b[3],
                a[3] * b[1] + a[4] * b[4],
                a[3] * b[2] + a[4] * b[5] + a[5],
            ],
        }
    }
}

/// The exact inverse mapping used by `warp::warp_single`, as an affine from
/// this scan's output-canvas coordinates to native disk coordinates.
fn canvas_to_native(
    geom: &EllipseGeom,
    rotation_deg: f64,
    flip_x: bool,
    flip_y: bool,
    canvas_c: (f64, f64),
) -> Affine2 {
    let th = rotation_deg.to_radians();
    let (ct, st) = (th.cos(), th.sin());
    let fx = if flip_x { -1.0 } else { 1.0 };
    let fy = if flip_y { -1.0 } else { 1.0 };
    // u = fx*ct*(vx-ocx) - fy*st*(vy-ocy); v = fx*st*(vx-ocx) + fy*ct*(vy-ocy)
    // x = sx*(u + shear*v) + xc ; y = v + yc
    let (ocx, ocy) = canvas_c;
    let ux = fx * ct;
    let uy = -fy * st;
    let vx = fx * st;
    let vy = fy * ct;
    let m0 = geom.sx * (ux + geom.shear * vx);
    let m1 = geom.sx * (uy + geom.shear * vy);
    let m2 = geom.xc - m0 * ocx - m1 * ocy;
    let m3 = vx;
    let m4 = vy;
    let m5 = geom.yc - m3 * ocx - m4 * ocy;
    Affine2 { m: [m0, m1, m2, m3, m4, m5] }
}

/// One scan's inputs to the native-domain stacker.
pub struct NativeScan<'a> {
    /// Corrected native disk (`ReconReport::native_disk`).
    pub disk: &'a Image,
    /// Fitted geometry (`ReconReport::native_geom`).
    pub geom: EllipseGeom,
    /// The warp orientation this scan was reconstructed with.
    pub rotation_deg: f64,
    pub flip_x: bool,
    pub flip_y: bool,
    /// This scan's own warped-canvas center (`WarpOutput::xc/yc`).
    pub canvas_c: (f64, f64),
    /// The per-scan warped reconstruction — used only for disk fitting,
    /// sharpness weighting and registration scoring, never resampled into
    /// the result.
    pub warped: &'a Image,
    /// Per-native-column quality weight in (0, 1], `column_weights()`.
    pub col_weight: Vec<f32>,
    /// Match the warp's footprint-filtered x downscale.
    pub filtered_downscale: bool,
}

pub struct NativeStackParams {
    /// Evolution-compensating optical flow (composed into the one resample).
    pub flow: bool,
    /// Re-register every scan against the first stack and stack again.
    pub iterate: bool,
    pub verbose: bool,
    /// None = sharpest scan.
    pub reference: Option<usize>,
}

impl Default for NativeStackParams {
    fn default() -> Self {
        NativeStackParams { flow: true, iterate: true, verbose: false, reference: None }
    }
}

/// Combine burst severity and transparency gains into per-column weights.
/// severity 1 = normal seeing (weight 1); a column that needed a 2x
/// transparency gain or reads 2x blurred contributes ~a quarter the weight.
pub fn column_weights(severity: &[f64], gains: &[f64], w: usize) -> Vec<f32> {
    (0..w)
        .map(|x| {
            let sev = severity.get(x).copied().unwrap_or(1.0).max(1.0);
            let ws = (1.0 / (sev * sev)).clamp(0.15, 1.0);
            let g = gains.get(x).copied().unwrap_or(1.0);
            let wg = 1.0 / (1.0 + 8.0 * (g - 1.0) * (g - 1.0));
            (ws * wg).clamp(0.05, 1.0) as f32
        })
        .collect()
}

#[inline]
fn bilinear_native(img: &Image, x: f64, y: f64) -> f32 {
    // shared with `bilinear` above; kept separate to stay #[inline] on the
    // hot search path
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

/// Session-wide 1-D slit distortion (c2, c3): the residual limb error is
/// modelled as d(u) = (c2 u^2 + c3 u^3) * r in native slit coordinates,
/// u = (y - yc)/r. Real optics stretch the plate scale near the slit ends
/// (observed: a one-sided +5% "pear" bulge at the south limb) — no affine
/// can express that, so it is corrected inside the same single resample.
pub type Ydist = Option<(f64, f64)>;

#[inline]
fn apply_ydist(geom: &EllipseGeom, yd: Ydist, ny: f64) -> f64 {
    match yd {
        None => ny,
        Some((c2, c3)) => {
            let u = (ny - geom.yc) / geom.radius;
            ny - (c2 * u * u + c3 * u * u * u) * geom.radius
        }
    }
}

/// NCC between the reference image and a native scan sampled through `t`,
/// over a sparse grid inside 0.95 R of the reference disk. Bilinear native
/// sampling: this scores candidate registrations, it never produces data.
fn ncc_through(refimg: &Image, rf: &DiskFit, scan: &NativeScan, yd: Ydist, t: &Affine2) -> f64 {
    let step = 4usize;
    let rows: Vec<(f64, f64, f64, f64, f64, f64)> = (0..refimg.h / step)
        .into_par_iter()
        .map(|yy| {
            let y = yy * step;
            let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let dy = y as f64 - rf.yc;
            for x in (0..refimg.w).step_by(step) {
                let dx = x as f64 - rf.xc;
                if dx * dx + dy * dy >= (rf.r * 0.95) * (rf.r * 0.95) {
                    continue;
                }
                let a = refimg.at(x, y) as f64;
                let (nx, ny) = t.apply(x as f64, y as f64);
                let ny = apply_ydist(&scan.geom, yd, ny);
                let b = bilinear_native(scan.disk, nx, ny) as f64;
                sa += a;
                sb += b;
                saa += a * a;
                sbb += b * b;
                sab += a * b;
                n += 1.0;
            }
            (sa, sb, saa, sbb, sab, n)
        })
        .collect();
    let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for r in rows {
        sa += r.0;
        sb += r.1;
        saa += r.2;
        sbb += r.3;
        sab += r.4;
        n += r.5;
    }
    if n < 100.0 {
        return f64::MIN;
    }
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    cov / (va * vb).sqrt().max(1e-12)
}

/// Sample the native disk through `t` (plus an optional flow field in
/// reference coordinates) onto the reference grid — the ONE resample.
fn resample_native(
    scan: &NativeScan,
    yd: Ydist,
    t: &Affine2,
    flow: Option<(&Image, &Image)>,
    size: (usize, usize),
) -> Image {
    let kx = if scan.filtered_downscale { scan.geom.sx.max(1.0) } else { 1.0 };
    let rows: Vec<Vec<f32>> = (0..size.1)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![0.0f32; size.0];
            for (x, out) in row.iter_mut().enumerate() {
                let (mut px, mut py) = (x as f64, y as f64);
                if let Some((fx, fy)) = flow {
                    px += fx.at(x, y) as f64;
                    py += fy.at(x, y) as f64;
                }
                let (nx, ny) = t.apply(px, py);
                let ny = apply_ydist(&scan.geom, yd, ny);
                *out = sample_lanczos3_aniso(scan.disk, nx, ny, kx).max(0.0);
            }
            row
        })
        .collect();
    let mut img = Image::new(size.0, size.1);
    for (y, row) in rows.iter().enumerate() {
        img.row_mut(y).copy_from_slice(row);
    }
    img
}

/// Solve a 3x3 linear system (normal equations of the affine fit).
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&a);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut out = [0.0f64; 3];
    for k in 0..3 {
        let mut ak = a;
        for r in 0..3 {
            ak[r][k] = b[r];
        }
        out[k] = det(&ak) / d;
    }
    Some(out)
}

/// Robust (Tukey-reweighted) LS fit of an affine displacement field to the
/// block-match vectors: f(p) ~ B p + c. Returns the correction as an affine
/// `p -> p + f(p)` to compose into the scan transform, or None when too few
/// coherent blocks support a global model.
fn fit_residual_affine(
    cells: &[(usize, usize, f64, f64, f64)],
    block: usize,
    rf: &DiskFit,
) -> Option<Affine2> {
    let mut pts: Vec<(f64, f64, f64, f64, f64)> = cells
        .iter()
        .filter(|&&(_, _, fx, fy, conf)| conf > 0.5 || fx != 0.0 || fy != 0.0)
        .map(|&(bx, by, fx, fy, conf)| {
            let x = (bx * block + block / 2) as f64;
            let y = (by * block + block / 2) as f64;
            (x, y, fx, fy, conf.max(0.0))
        })
        .collect();
    if pts.len() < 12 {
        return None;
    }
    // normalize coordinates about the disk center for conditioning
    for p in pts.iter_mut() {
        p.0 = (p.0 - rf.xc) / rf.r;
        p.1 = (p.1 - rf.yc) / rf.r;
    }
    let mut wts: Vec<f64> = pts.iter().map(|p| p.4).collect();
    let mut cx = [0.0f64; 3]; // fx = cx0*x + cx1*y + cx2
    let mut cy = [0.0f64; 3];
    for _round in 0..3 {
        let fit1 = |get: &dyn Fn(&(f64, f64, f64, f64, f64)) -> f64| -> Option<[f64; 3]> {
            let mut ata = [[0.0f64; 3]; 3];
            let mut atb = [0.0f64; 3];
            for (p, &w) in pts.iter().zip(&wts) {
                let basis = [p.0, p.1, 1.0];
                let v = get(p);
                for r in 0..3 {
                    for c in 0..3 {
                        ata[r][c] += w * basis[r] * basis[c];
                    }
                    atb[r] += w * basis[r] * v;
                }
            }
            solve3(ata, atb)
        };
        cx = fit1(&|p| p.2)?;
        cy = fit1(&|p| p.3)?;
        // Tukey reweight against residuals
        let mut resid: Vec<f64> = pts
            .iter()
            .map(|p| {
                let rx = p.2 - (cx[0] * p.0 + cx[1] * p.1 + cx[2]);
                let ry = p.3 - (cy[0] * p.0 + cy[1] * p.1 + cy[2]);
                (rx * rx + ry * ry).sqrt()
            })
            .collect();
        let mut sorted = resid.clone();
        let mad = crate::mathutil::median_inplace(&mut sorted).max(0.02);
        let c_tukey = 4.685 * 1.4826 * mad;
        for (w, r) in wts.iter_mut().zip(resid.drain(..)) {
            let u = r / c_tukey;
            *w = if u < 1.0 { (1.0 - u * u) * (1.0 - u * u) } else { 0.0 };
        }
        if wts.iter().sum::<f64>() < 6.0 {
            return None;
        }
    }
    // de-normalize: fx(px) = cx0*(x-xc)/r + cx1*(y-yc)/r + cx2
    let (r, xc, yc) = (rf.r, rf.xc, rf.yc);
    let m0 = 1.0 + cx[0] / r;
    let m1 = cx[1] / r;
    let m2 = cx[2] - (cx[0] * xc + cx[1] * yc) / r;
    let m3 = cy[0] / r;
    let m4 = 1.0 + cy[1] / r;
    let m5 = cy[2] - (cy[0] * xc + cy[1] * yc) / r;
    Some(Affine2 { m: [m0, m1, m2, m3, m4, m5] })
}

/// Register one scan against the current reference image: similarity from
/// disk fits, pattern-search translation refine (coarse-to-fine, wide
/// capture), then a global residual affine from block matching. Returns the
/// composed canvas->native transform and the achieved NCC.
fn register_native(
    scan: &NativeScan,
    yd: Ydist,
    fit: &DiskFit,
    refimg: &Image,
    rf: &DiskFit,
    flipped: bool,
) -> (Affine2, f64) {
    let c2n = canvas_to_native(
        &scan.geom,
        scan.rotation_deg,
        if flipped { !scan.flip_x } else { scan.flip_x },
        scan.flip_y,
        scan.canvas_c,
    );
    let s = fit.r / rf.r;
    let sim = |dx: f64, dy: f64| -> Affine2 {
        // ref pixel -> this scan's canvas -> native
        let sim = Affine2 {
            m: [s, 0.0, fit.xc - rf.xc * s + dx, 0.0, s, fit.yc - rf.yc * s + dy],
        };
        c2n.compose(&sim)
    };
    if std::env::var("GS_STACK_DEBUG").is_ok() {
        let t0 = sim(0.0, 0.0);
        println!(
            "reg dbg: flipped={} fit=({:.1},{:.1},r={:.1}) rf=({:.1},{:.1},r={:.1}) s={:.4}",
            flipped, fit.xc, fit.yc, fit.r, rf.xc, rf.yc, rf.r, s
        );
        println!(
            "reg dbg: geom xc={:.1} yc={:.1} sx={:.4} shear={:.5} r={:.1} canvas_c=({:.1},{:.1}) disk={}x{} ncc0={:.4}",
            scan.geom.xc, scan.geom.yc, scan.geom.sx, scan.geom.shear, scan.geom.radius,
            scan.canvas_c.0, scan.canvas_c.1, scan.disk.w, scan.disk.h,
            ncc_through(refimg, rf, scan, yd, &t0)
        );
    }
    let mut best = (0.0f64, 0.0f64, f64::MIN);
    for step in [4.0f64, 1.0, 0.25, 0.05] {
        let (cx, cy, _) = best;
        let mut local = best;
        let mut dy = cy - 2.0 * step;
        while dy <= cy + 2.0 * step + 1e-12 {
            let mut dx = cx - 2.0 * step;
            while dx <= cx + 2.0 * step + 1e-12 {
                let v = ncc_through(refimg, rf, scan, yd, &sim(dx, dy));
                if v > local.2 {
                    local = (dx, dy, v);
                }
                dx += step;
            }
            dy += step;
        }
        best = local;
    }
    let mut t = sim(best.0, best.1);
    let mut ncc_best = best.2;
    if ncc_best < NCC_MIN {
        return (t, ncc_best);
    }
    // residual affine: measure block displacements on a real (Lanczos)
    // candidate, fit globally, compose, keep only if the score improves
    let cand = resample_native(scan, yd, &t, None, (refimg.w, refimg.h));
    let cells = block_vectors(refimg, &cand, rf, 32, 4);
    if let Some(res) = fit_residual_affine(&cells, 32, rf) {
        let t2 = t.compose(&res);
        let v2 = ncc_through(refimg, rf, scan, yd, &t2);
        if v2 > ncc_best {
            t = t2;
            ncc_best = v2;
        }
    }
    (t, ncc_best)
}

/// Tukey-reweighted line fit y = a + b t. Returns (a, b).
fn robust_line(ts: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    let mut w = vec![1.0f64; ts.len()];
    let mut ab = (0.0f64, 0.0f64);
    for _ in 0..3 {
        let (mut sw, mut st, mut sy, mut stt, mut sty) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for ((&t, &y), &wi) in ts.iter().zip(ys).zip(&w) {
            sw += wi;
            st += wi * t;
            sy += wi * y;
            stt += wi * t * t;
            sty += wi * t * y;
        }
        let det = sw * stt - st * st;
        if det.abs() < 1e-9 {
            return None;
        }
        ab = ((stt * sy - st * sty) / det, (sw * sty - st * sy) / det);
        let mut resid: Vec<f64> =
            ts.iter().zip(ys).map(|(&t, &y)| (y - ab.0 - ab.1 * t).abs()).collect();
        let mut sorted = resid.clone();
        let mad = crate::mathutil::median_inplace(&mut sorted).max(1e-9);
        let ct = 4.685 * 1.4826 * mad;
        for (wi, r) in w.iter_mut().zip(resid.drain(..)) {
            let u = r / ct;
            *wi = if u < 1.0 { (1.0 - u * u) * (1.0 - u * u) } else { 0.0 };
        }
    }
    Some(ab)
}

/// Photometric disc geometry from the corrected native disk — the stacking
/// replacement for the limb-point conic fit.
///
/// Rationale: on real data the per-scan conic fits scatter by ±10% in sx and
/// hundreds of px in center (polar-cap tangent smear + entry/exit ramp
/// artifacts bias the limb points), so every scan warps into a DIFFERENTLY
/// distorted canvas and no rigid inter-scan registration can succeed. The
/// whole-disc flux profiles are far better conditioned: for a circle seen
/// through x = sx*(u + shear*v), y = v, the squared column-flux F(x)^2 and
/// row-flux G(y)^2 are exact parabolas, giving (xc, rx) and (yc, ry = r)
/// robustly, and the vertical-midchord line slope b = shear/(sx*(1+shear^2))
/// together with rx = sx*r*sqrt(1+shear^2) closes the system:
///   S = Ab/sqrt(1-(Ab)^2), sx = A/sqrt(1+S^2), where A = rx/r.
pub fn photometric_geom(disk: &Image, seed: &EllipseGeom) -> Option<EllipseGeom> {
    let (w, h) = (disk.w, disk.h);
    // flux profiles (negative-clamped)
    let mut f = vec![0.0f64; w];
    let mut g = vec![0.0f64; h];
    for y in 0..h {
        let row = disk.row(y);
        for x in 0..w {
            let v = row[x].max(0.0) as f64;
            f[x] += v;
            g[y] += v;
        }
    }
    let peak = |p: &[f64]| -> f64 {
        let mut s: Vec<f64> = p.to_vec();
        let n = s.len();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[(n as f64 * 0.98) as usize % n]
    };
    // Half-max crossings, NOT a sqrt/parabola profile model: the pipeline's
    // transparency and transversalium corrections flatten the flux profiles
    // before we see them, so a chord-model fit inflates the axis estimates
    // anisotropically (observed: 3% wide bias in sx -> pumpkin-shaped
    // stacks). Crossing positions survive plateau-flattening, and using the
    // SAME estimator on both axes cancels the residual limb-darkening bias
    // in the rx/ry ratio that sx is derived from. For a uniform disc the
    // half-of-max chord sits at 0.866 r, hence the calibration divisor.
    let fit_profile = |p: &[f64]| -> Option<(f64, f64)> {
        let pk = peak(p);
        if pk <= 0.0 {
            return None;
        }
        let thr = 0.5 * pk;
        let above: Vec<usize> = (0..p.len()).filter(|&i| p[i] > thr).collect();
        if above.len() < 50 {
            return None;
        }
        let (i0, i1) = (above[0], *above.last().unwrap());
        let interp_left = if i0 > 0 && p[i0] > p[i0 - 1] {
            i0 as f64 - 1.0 + (thr - p[i0 - 1]) / (p[i0] - p[i0 - 1])
        } else {
            i0 as f64
        };
        let interp_right = if i1 + 1 < p.len() && p[i1] > p[i1 + 1] {
            i1 as f64 + (p[i1] - thr) / (p[i1] - p[i1 + 1])
        } else {
            i1 as f64
        };
        let tc = (interp_left + interp_right) / 2.0;
        let halfspan = (interp_right - interp_left) / 2.0 / 0.866_025_4;
        Some((tc, halfspan))
    };
    let (xc, rx) = fit_profile(&f)?;
    let (yc, ry) = fit_profile(&g)?;
    if rx < 20.0 || ry < 20.0 {
        return None;
    }

    // vertical midchord line: slope -> shear
    let fpk = peak(&f);
    let gthr = (peak(&g) / (2.0 * rx)).max(1e-9); // ~mean surface brightness
    let mut ts = Vec::new();
    let mut ms = Vec::new();
    for x in 0..w {
        if f[x] < 0.5 * fpk {
            continue;
        }
        let mut y0 = None;
        let mut y1 = None;
        for y in 0..h {
            if disk.at(x, y) as f64 > 0.35 * gthr {
                if y0.is_none() {
                    y0 = Some(y);
                }
                y1 = Some(y);
            }
        }
        if let (Some(a), Some(b)) = (y0, y1) {
            if b > a + 50 {
                ts.push(x as f64 - xc);
                ms.push((a + b) as f64 / 2.0);
            }
        }
    }
    if ts.len() < 50 {
        return None;
    }
    let (_a, b) = robust_line(&ts, &ms)?;

    let big_a = rx / ry;
    let ab = (big_a * b).clamp(-0.9, 0.9);
    let shear = ab / (1.0 - ab * ab).sqrt();
    let sx = big_a / (1.0 + shear * shear).sqrt();
    // plausibility: reject a fit that disagrees catastrophically with itself
    if !(0.3..=4.0).contains(&sx) {
        return None;
    }
    Some(EllipseGeom {
        xc,
        yc,
        an: seed.an,
        bn: seed.bn,
        cn: seed.cn,
        sx,
        shear,
        radius: ry,
    })
}

/// Steepest-gradient limb radius at `n_ang` position angles.
fn limb_radii(img: &Image, xc: f64, yc: f64, r0: f64, n_ang: usize) -> (Vec<f64>, Vec<f64>) {
    let n_r = 141usize;
    let mut ths = Vec::with_capacity(n_ang);
    let mut rads = Vec::with_capacity(n_ang);
    for ia in 0..n_ang {
        let th = ia as f64 / n_ang as f64 * std::f64::consts::TAU;
        let (ct, st) = (th.cos(), th.sin());
        let mut vals = Vec::with_capacity(n_r);
        let mut rs = Vec::with_capacity(n_r);
        for ir in 0..n_r {
            let r = r0 * (0.86 + 0.28 * ir as f64 / (n_r - 1) as f64);
            let x = xc + r * ct;
            let y = yc + r * st;
            if x < 1.0 || y < 1.0 || x > img.w as f64 - 2.0 || y > img.h as f64 - 2.0 {
                break;
            }
            vals.push(bilinear_native(img, x, y) as f64);
            rs.push(r);
        }
        if vals.len() < 40 {
            continue;
        }
        // steepest falloff
        let mut best = (0usize, 0.0f64);
        for i in 1..vals.len() - 1 {
            let g = vals[i + 1] - vals[i - 1];
            if g < best.1 {
                best = (i, g);
            }
        }
        if best.1 < 0.0 {
            ths.push(th);
            rads.push(rs[best.0]);
        }
    }
    (ths, rads)
}

/// Session 1-D slit distortion from a rendered probe: residual limb error
/// after removing radius + decenter, attributed to native-y via
/// d = δr/sinθ at u = sinθ·(r/r̄), robust-fit to d(u) = (c2 u² + c3 u³) r̄.
fn measure_y_distortion(img: &Image, xc: f64, yc: f64, r0: f64) -> Ydist {
    let (ths, rads) = limb_radii(img, xc, yc, r0, 180);
    if ths.len() < 80 {
        return None;
    }
    // remove radius + decenter (dipole)
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for (&th, &r) in ths.iter().zip(&rads) {
        let b = [1.0, th.cos(), th.sin()];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += b[i] * b[j];
            }
            atb[i] += b[i] * r;
        }
    }
    let c = solve3(ata, atb)?;
    let rbar = c[0];
    if rbar <= 0.0 {
        return None;
    }
    let mut us = Vec::new();
    let mut ds = Vec::new();
    for (&th, &r) in ths.iter().zip(&rads) {
        let st = th.sin();
        if st.abs() < 0.34 {
            continue; // δr/sinθ ill-conditioned near E/W
        }
        let delta = r - (c[0] + c[1] * th.cos() + c[2] * st);
        us.push(st * (r / rbar));
        ds.push(delta / st / rbar); // normalized: d in units of r̄
    }
    if us.len() < 40 {
        return None;
    }
    // robust LS on d(u) = c2 u^2 + c3 u^3
    let mut w = vec![1.0f64; us.len()];
    let mut c2 = 0.0f64;
    let mut c3 = 0.0f64;
    for _ in 0..3 {
        let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for ((&u, &d), &wi) in us.iter().zip(&ds).zip(&w) {
            let (f1, f2) = (u * u, u * u * u);
            a11 += wi * f1 * f1;
            a12 += wi * f1 * f2;
            a22 += wi * f2 * f2;
            b1 += wi * f1 * d;
            b2 += wi * f2 * d;
        }
        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-12 {
            return None;
        }
        c2 = (a22 * b1 - a12 * b2) / det;
        c3 = (a11 * b2 - a12 * b1) / det;
        let mut resid: Vec<f64> = us
            .iter()
            .zip(&ds)
            .map(|(&u, &d)| (d - c2 * u * u - c3 * u * u * u).abs())
            .collect();
        let mut sorted = resid.clone();
        let mad = crate::mathutil::median_inplace(&mut sorted).max(1e-6);
        let ct = 4.685 * 1.4826 * mad;
        for (wi, r) in w.iter_mut().zip(resid.drain(..)) {
            let t = r / ct;
            *wi = if t < 1.0 { (1.0 - t * t) * (1.0 - t * t) } else { 0.0 };
        }
    }
    // plausibility: correction at the slit ends must stay moderate
    let dmax = (c2.abs() + c3.abs()).max((c2 - c3).abs());
    if dmax > 0.12 {
        return None;
    }
    // ignore corrections below ~1.5 px at the limb
    if dmax * r0 < 1.5 {
        return None;
    }
    Some((c2, c3))
}

/// Measured limb axis ratio rx/ry of a rendered disc: steepest-gradient
/// limb radius at 72 position angles, LS-fit to r(θ) = r0 + c2 cos2θ +
/// s2 sin2θ. Feeds the one-shot sx self-calibration.
fn limb_axis_ratio(img: &Image, xc: f64, yc: f64, r0: f64) -> Option<f64> {
    let (ths, rads) = limb_radii(img, xc, yc, r0, 72);
    if ths.len() < 32 {
        return None;
    }
    // LS fit r(θ) = a0 + a1 cos2θ + a2 sin2θ (+ cosθ, sinθ absorb decenter)
    let mut ata = [[0.0f64; 5]; 5];
    let mut atb = [0.0f64; 5];
    for (&th, &r) in ths.iter().zip(&rads) {
        let b = [1.0, (2.0 * th).cos(), (2.0 * th).sin(), th.cos(), th.sin()];
        for i in 0..5 {
            for j in 0..5 {
                ata[i][j] += b[i] * b[j];
            }
            atb[i] += b[i] * r;
        }
    }
    // solve 5x5 by Gaussian elimination
    let mut m = ata;
    let mut v = atb;
    for col in 0..5 {
        let piv = (col..5).max_by(|&a, &b| m[a][col].abs().partial_cmp(&m[b][col].abs()).unwrap())?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        v.swap(col, piv);
        for row in 0..5 {
            if row != col {
                let f = m[row][col] / m[col][col];
                for k in col..5 {
                    m[row][k] -= f * m[col][k];
                }
                v[row] -= f * v[col];
            }
        }
    }
    let a0 = v[0] / m[0][0];
    let c2 = v[1] / m[1][1];
    if a0 <= 0.0 {
        return None;
    }
    Some((a0 + c2) / (a0 - c2))
}

/// Register one scan against a reference image, recovering a mislabelled
/// x-mirror (orientation is deterministic — a scan that fails direct
/// registration is retried flipped before being given up on).
/// Returns (transform, ncc, was_flipped) or None if neither orientation
/// correlates.
fn register_scan(
    scan: &NativeScan,
    yd: Ydist,
    fk: &DiskFit,
    refimg: &Image,
    rf: &DiskFit,
) -> Option<(Affine2, f64, bool)> {
    let (t, ncc_v) = register_native(scan, yd, fk, refimg, rf, false);
    if ncc_v >= NCC_MIN {
        return Some((t, ncc_v, false));
    }
    let mfit = DiskFit {
        xc: 2.0 * scan.canvas_c.0 - fk.xc,
        yc: fk.yc,
        r: fk.r,
    };
    let (tm, nm) = register_native(scan, yd, &mfit, refimg, rf, true);
    if nm >= NCC_MIN {
        return Some((tm, nm, true));
    }
    None
}

/// Native-domain multi-scan stack: one Lanczos resample per scan, global
/// similarity + residual affine registration, optional evolution flow
/// composed into that same resample, per-column quality weights, robust
/// MAD-clipped weighted mean, optional re-registration against the first
/// stack.
///
/// Reference selection is CONSENSUS-GATED: candidates are tried in
/// descending sharpness, and a reference that a majority of the other scans
/// refuse to register against is rejected. Raw HF energy alone is not a
/// safe criterion — a scan wrecked by a cloud passage can carry the highest
/// "sharpness" in its artifacts and, as the reference, silently vetoes
/// every healthy scan (observed on real data: 7/8 scans dropped).
pub fn stack_native(scans: &[NativeScan], p: &NativeStackParams) -> Option<StackReport> {
    if scans.is_empty() {
        return None;
    }
    if scans.len() == 1 {
        return Some(StackReport {
            n_flipped: 0,
            image: scans[0].warped.clone(),
            n_used: 1,
            weights: vec![1.0],
        });
    }
    // F5.5.1: photometric session geometry. Per-scan conic fits scatter by
    // ±10% in sx and hundreds of px in center on real data (limb-point bias
    // from tangent smear and entry/exit ramps), which warps every scan into
    // a DIFFERENTLY distorted canvas — unregistrable by any rigid model.
    // Refit every scan photometrically and pin the radius to the session
    // median: the sun does not change size between scans.
    let mut geoms: Vec<EllipseGeom> = scans
        .iter()
        .map(|s| photometric_geom(s.disk, &s.geom).unwrap_or(s.geom))
        .collect();
    let mut radii: Vec<f64> = geoms.iter().map(|g| g.radius).collect();
    let r_med = crate::mathutil::median_inplace(&mut radii).max(1.0);
    for g in geoms.iter_mut() {
        // keep the physical x half-span while pinning the common radius
        g.sx *= g.radius / r_med;
        g.radius = r_med;
    }
    if p.verbose {
        for (k, g) in geoms.iter().enumerate() {
            println!(
                "stack-native: scan {k} photometric geom: c=({:.1},{:.1}) sx={:.4} shear={:.5} r={:.1}",
                g.xc, g.yc, g.sx, g.shear, g.radius
            );
        }
    }
    // F5.5.2/F5.5.3: one-shot shape self-calibration on a rendered probe.
    // Step 1 measures the session's nonlinear slit distortion (a one-sided
    // "pear" bulge at the slit-end limb that no affine can express).
    // Step 2, on a distortion-corrected re-render, measures the residual
    // limb axis ratio (the two flux profiles are flattened by DIFFERENT
    // upstream corrections, so the half-max rx/ry carries a small uniform
    // bias) and folds it into every sx.
    let mut ydist: Ydist = None;
    {
        let mut sxs: Vec<f64> = geoms.iter().map(|g| g.sx).collect();
        let sx_med = crate::mathutil::median_inplace(&mut sxs);
        let probe_k = (0..scans.len())
            .min_by(|&a, &b| {
                (geoms[a].sx - sx_med)
                    .abs()
                    .partial_cmp(&(geoms[b].sx - sx_med).abs())
                    .unwrap()
            })
            .unwrap();
        let s = &scans[probe_k];
        let mk_probe = |geom: EllipseGeom| NativeScan {
            disk: s.disk,
            geom,
            rotation_deg: s.rotation_deg,
            flip_x: s.flip_x,
            flip_y: s.flip_y,
            canvas_c: s.canvas_c,
            warped: s.warped,
            col_weight: Vec::new(),
            filtered_downscale: s.filtered_downscale,
        };
        let probe = mk_probe(geoms[probe_k]);
        let t = canvas_to_native(&probe.geom, probe.rotation_deg, probe.flip_x, probe.flip_y, probe.canvas_c);
        let pimg = resample_native(&probe, None, &t, None, (s.warped.w, s.warped.h));
        ydist = measure_y_distortion(&pimg, s.canvas_c.0, s.canvas_c.1, r_med);
        if p.verbose {
            match ydist {
                Some((c2, c3)) => println!(
                    "stack-native: slit distortion on scan {probe_k}: d(u) = ({c2:+.4} u^2 {c3:+.4} u^3) r  (ends: {:+.1}/{:+.1} px)",
                    (c2 - c3) * r_med,
                    (c2 + c3) * r_med
                ),
                None => println!("stack-native: slit distortion below threshold — none applied"),
            }
        }
        let pimg2 = if ydist.is_some() {
            resample_native(&probe, ydist, &t, None, (s.warped.w, s.warped.h))
        } else {
            pimg
        };
        if let Some(ratio) = limb_axis_ratio(&pimg2, s.canvas_c.0, s.canvas_c.1, r_med) {
            if (ratio - 1.0).abs() < 0.2 && (ratio - 1.0).abs() > 0.002 {
                for g in geoms.iter_mut() {
                    g.sx *= ratio;
                }
                if p.verbose {
                    println!(
                        "stack-native: shape self-calibration on scan {probe_k}: rx/ry {ratio:.4} folded into sx"
                    );
                }
            }
        }
    }
    let scans_c: Vec<NativeScan> = scans
        .iter()
        .zip(&geoms)
        .map(|(s, g)| NativeScan {
            disk: s.disk,
            geom: *g,
            rotation_deg: s.rotation_deg,
            flip_x: s.flip_x,
            flip_y: s.flip_y,
            canvas_c: s.canvas_c,
            warped: s.warped,
            col_weight: s.col_weight.clone(),
            filtered_downscale: s.filtered_downscale,
        })
        .collect();
    let scans = &scans_c[..];
    // With consistent geometry every disc is circular (radius r_med) and
    // centered on its own canvas center BY CONSTRUCTION — the canvas disk
    // fit is synthetic, not measured from the (old-geometry) warped image.
    let fits: Vec<Option<DiskFit>> = scans
        .iter()
        .map(|s| Some(DiskFit { xc: s.canvas_c.0, yc: s.canvas_c.1, r: r_med }))
        .collect();
    let energies: Vec<f64> = scans
        .iter()
        .zip(&fits)
        .map(|(s, f)| hf_energy(s.warped, f.as_ref().unwrap()))
        .collect();
    let fitted: Vec<usize> = (0..scans.len()).collect();

    // Candidate references: an explicit choice is honored as-is; otherwise
    // the sharpest few are tried in order and gated on consensus.
    let candidates: Vec<usize> = match p.reference {
        Some(i) => {
            let i = i.min(scans.len() - 1);
            if fits[i].is_some() { vec![i] } else { vec![fitted[0]] }
        }
        None => {
            let mut c = fitted.clone();
            c.sort_by(|a, b| energies[*b].partial_cmp(&energies[*a]).unwrap());
            c.truncate(3);
            c
        }
    };
    // The reference image is rendered from the reference scan's NATIVE disk
    // through its corrected geometry — the stored warped canvas was made
    // with the old (unreliable) conic fit and would poison every NCC.
    let render_ref = |k: usize| -> Image {
        let s = &scans[k];
        let t = canvas_to_native(&s.geom, s.rotation_deg, s.flip_x, s.flip_y, s.canvas_c);
        resample_native(s, ydist, &t, None, (s.warped.w, s.warped.h))
    };
    let mut chosen: Option<(usize, Image, Vec<(usize, Affine2, f64, bool)>)> = None;
    for &cand in &candidates {
        let cref = render_ref(cand);
        let crf = fits[cand].as_ref().unwrap();
        let mut regs: Vec<(usize, Affine2, f64, bool)> = Vec::new();
        for &k in &fitted {
            match register_scan(&scans[k], ydist, fits[k].as_ref().unwrap(), &cref, crf) {
                Some((t, n, f)) => regs.push((k, t, n, f)),
                None => {
                    if p.verbose {
                        println!(
                            "stack-native: scan {k} refuses candidate reference {cand}"
                        );
                    }
                }
            }
        }
        if p.verbose {
            println!(
                "stack-native: candidate reference {cand} (hf {:.1}): {}/{} scans register",
                energies[cand],
                regs.len(),
                fitted.len()
            );
        }
        let all = regs.len() == fitted.len();
        if chosen.as_ref().map(|(_, _, r)| regs.len() > r.len()).unwrap_or(true) {
            chosen = Some((cand, cref, regs));
        }
        if all {
            break;
        }
    }
    let (ref_idx, ref_render, regs0) = chosen?;
    if regs0.len() < 2 {
        return None;
    }
    let size = (scans[ref_idx].warped.w, scans[ref_idx].warped.h);
    let mut refimg: Image = ref_render;
    let mut rf: DiskFit = fits[ref_idx].as_ref()?.clone();
    if p.verbose {
        println!(
            "stack-native: reference = scan {ref_idx} ({}/{} scans in consensus)",
            regs0.len(),
            fitted.len()
        );
    }

    let passes = if p.iterate { 2 } else { 1 };
    let mut n_flipped = 0usize;
    let mut result: Option<(Image, Vec<usize>, Vec<f64>)> = None;
    for pass in 0..passes {
        // pass 0 reuses the consensus registrations; later passes register
        // every fittable scan (reference included) against the stack itself
        let regs: Vec<(usize, Affine2, f64, bool)> = if pass == 0 {
            regs0.clone()
        } else {
            fitted
                .iter()
                .filter_map(|&k| {
                    register_scan(&scans[k], ydist, fits[k].as_ref().unwrap(), &refimg, &rf)
                        .map(|(t, n, f)| (k, t, n, f))
                })
                .collect()
        };
        if regs.len() < 2 {
            // a collapsed re-registration pass must not discard a valid
            // first-pass result
            break;
        }
        n_flipped = regs.iter().filter(|r| r.3).count();
        let mut aligned: Vec<Image> = Vec::new();
        let mut transforms: Vec<Affine2> = Vec::new();
        let mut kept: Vec<usize> = Vec::new();
        for &(k, t, ncc_v, _) in &regs {
            let scan = &scans[k];
            // evolution flow, composed into the same single resample
            let img = if p.flow {
                let cand = resample_native(scan, ydist, &t, None, size);
                let (fx, fy) = optical_flow(&refimg, &cand, &rf, 32, 4);
                resample_native(scan, ydist, &t, Some((&fx, &fy)), size)
            } else {
                resample_native(scan, ydist, &t, None, size)
            };
            if p.verbose {
                println!("stack-native[{pass}]: scan {k} registered (ncc {:.4})", ncc_v);
            }
            aligned.push(img);
            transforms.push(t);
            kept.push(k);
        }

        // photometric gain surface vs the reference image
        for a in aligned.iter_mut() {
            if let Some(gain) = fit_gain_surface(&refimg, a, &rf) {
                for y in 0..size.1 {
                    for x in 0..size.0 {
                        let g = eval_quad(&gain, (x as f64 - rf.xc) / rf.r, (y as f64 - rf.yc) / rf.r);
                        let v = a.at(x, y) as f64 * g.clamp(0.5, 2.0);
                        a.set(x, y, v as f32);
                    }
                }
            }
        }

        // global sharpness weights
        let emax = kept.iter().map(|&k| energies[k]).fold(f64::MIN, f64::max).max(1e-12);
        let weights: Vec<f64> = kept.iter().map(|&k| (energies[k] / emax).clamp(0.2, 1.0)).collect();

        // robust per-pixel combine with per-column quality weights carried
        // through each scan's transform
        let k_scans = aligned.len();
        let out_rows: Vec<Vec<f32>> = (0..size.1)
            .into_par_iter()
            .map(|y| {
                let mut row = vec![0.0f32; size.0];
                let mut vals = vec![0.0f64; k_scans];
                let mut wpix = vec![0.0f64; k_scans];
                for (x, out) in row.iter_mut().enumerate() {
                    for k in 0..k_scans {
                        vals[k] = aligned[k].at(x, y) as f64;
                        let scan = &scans[kept[k]];
                        let (nx, _) = transforms[k].apply(x as f64, y as f64);
                        let cw = if scan.col_weight.is_empty() {
                            1.0
                        } else {
                            let xi = (nx.round() as isize)
                                .clamp(0, scan.col_weight.len() as isize - 1)
                                as usize;
                            scan.col_weight[xi] as f64
                        };
                        wpix[k] = weights[k] * cw;
                    }
                    let mut sorted = vals.clone();
                    let med = crate::mathutil::median_inplace(&mut sorted);
                    let mut devs: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
                    let mad = crate::mathutil::median_inplace(&mut devs).max(1e-6);
                    let mut acc = 0.0;
                    let mut wsum = 0.0;
                    for k in 0..k_scans {
                        if (vals[k] - med).abs() < 3.0 * 1.4826 * mad + 1e-3 * med.abs() + 1.0 {
                            acc += wpix[k] * vals[k];
                            wsum += wpix[k];
                        }
                    }
                    *out = if wsum > 0.0 { (acc / wsum) as f32 } else { med as f32 };
                }
                row
            })
            .collect();
        let mut out = Image::new(size.0, size.1);
        for (y, row) in out_rows.iter().enumerate() {
            out.row_mut(y).copy_from_slice(row);
        }

        let is_last = pass + 1 == passes;
        result = Some((out, kept, weights));
        if !is_last {
            // next pass registers every scan (reference included) against
            // the stack itself, removing the single-scan reference bias
            let stacked = &result.as_ref().unwrap().0;
            rf = fit_disk(stacked).unwrap_or(rf);
            refimg = stacked.clone();
        }
    }
    let (image, kept, weights) = result?;
    Some(StackReport { n_flipped, image, n_used: kept.len(), weights })
}

#[cfg(test)]
mod guard_tests {
    use super::*;

    /// Limb-darkened disk with strong mirror-asymmetric texture. `mirror`
    /// evaluates the same field at the flipped x, i.e. exactly what a
    /// forgotten flip_x produces.
    fn textured_disk(mirror: bool) -> Image {
        let (w, h) = (240usize, 240usize);
        let (cx, cy, r) = (120.0f64, 120.0f64, 80.0f64);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let fx = if mirror { (w - 1 - x) as f64 } else { x as f64 };
                let d = ((fx - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt() / r;
                if d < 1.0 {
                    let mu = (1.0 - d * d).sqrt();
                    let limb = 1.0 - 0.6 * (1.0 - mu);
                    let tex = 1.0
                        + 0.5
                            * ((0.37 * fx + 0.11 * y as f64).sin()
                                * (0.23 * fx - 0.31 * y as f64).cos());
                    img.set(x, y, (10_000.0 * limb * tex) as f32);
                }
            }
        }
        img
    }

    fn identity_geom() -> EllipseGeom {
        EllipseGeom {
            xc: 120.0,
            yc: 120.0,
            an: 1.0,
            bn: 0.0,
            cn: 1.0,
            sx: 1.0,
            shear: 0.0,
            radius: 80.0,
        }
    }

    /// The native path must reproduce the invariants of the warped path:
    /// identical scans stack losslessly, and per-column weights don't
    /// disturb a uniform-quality stack.
    #[test]
    fn native_stack_of_identical_scans_matches_input() {
        let a = textured_disk(false);
        let geom = identity_geom();
        let scans: Vec<NativeScan> = (0..2)
            .map(|_| NativeScan {
                disk: &a,
                geom,
                rotation_deg: 0.0,
                flip_x: false,
                flip_y: false,
                canvas_c: (120.0, 120.0),
                warped: &a,
                col_weight: vec![1.0; a.w],
                filtered_downscale: false,
            })
            .collect();
        let rep = stack_native(
            &scans,
            &NativeStackParams { flow: false, iterate: false, verbose: false, reference: Some(0) },
        )
        .expect("identical scans must stack");
        assert_eq!(rep.n_used, 2);
        assert_eq!(rep.n_flipped, 0);
        // Pixel identity is NOT the contract any more: the stack re-derives
        // the disc geometry photometrically, which may differ from the
        // input warp by sub-pixel amounts. The guarantees are (1) high
        // correlation with the source and (2) no gross displacement.
        let df = DiskFit { xc: 120.0, yc: 120.0, r: 80.0 };
        let corr = ncc(&rep.image, &a, &df);
        assert!(
            corr > 0.98,
            "native stack decorrelated from its identical inputs (ncc {corr:.4})"
        );
        let centroid = |img: &Image| -> (f64, f64) {
            let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
            for y in 0..img.h {
                for x in 0..img.w {
                    let v = img.at(x, y).max(0.0) as f64;
                    sx += v * x as f64;
                    sy += v * y as f64;
                    sw += v;
                }
            }
            (sx / sw, sy / sw)
        };
        let (ax, ay) = centroid(&a);
        let (bx, by) = centroid(&rep.image);
        assert!(
            (ax - bx).abs() < 2.0 && (ay - by).abs() < 2.0,
            "native stack displaced the disc (({ax:.2},{ay:.2}) -> ({bx:.2},{by:.2}))"
        );
    }

    #[test]
    fn a_mirrored_scan_is_recovered_by_flipping_not_dropped() {
        let a = textured_disk(false);
        let m = textured_disk(true);
        // Orientation is deterministic, so a mirrored scan is good data with a
        // wrong label: the stack must flip it back and use it, reporting the
        // recovery so the upstream bookkeeping bug is visible.
        let rep = stack_with_reference(&[a.clone(), m], false, false, Some(0))
            .expect("a mirrored scan must be recovered, not rejected");
        assert_eq!(rep.n_used, 2, "both scans must survive");
        assert_eq!(rep.n_flipped, 1, "the recovery must be reported");
        // Sanity: the same disk twice stacks with no flips.
        let rep = stack_with_reference(&[a.clone(), a], false, false, Some(0))
            .expect("identical scans must stack");
        assert_eq!(rep.n_used, 2);
        assert_eq!(rep.n_flipped, 0);
    }
}
