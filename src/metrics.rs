//! Quality metrics against ground truth: registration (scale + sub-pixel
//! translation search), robust linear intensity matching, then PSNR and
//! SSIM over the disk interior and a limb annulus.

use crate::image2d::Image;

pub struct DiskFit {
    pub xc: f64,
    pub yc: f64,
    pub r: f64,
}

pub fn fit_disk(img: &Image) -> Option<DiskFit> {
    let init = coarse_disk(img)?;
    let (refined, _) = fit_disk_polar(img, &init)?;
    Some(refined)
}

/// Conic-free initialization: flux centroid + area-equivalent radius of the
/// thresholded disk. Conic fits are fragile on spicule-fuzzy limbs; for a
/// near-circular disk this coarse seed plus the polar refinement is stable.
pub fn coarse_disk(img: &Image) -> Option<DiskFit> {
    let thresh = crate::mathutil::percentile_f32(&img.data, 80.0) * 0.35;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut n = 0.0f64;
    for y in 0..img.h {
        for x in 0..img.w {
            if img.at(x, y) > thresh {
                sx += x as f64;
                sy += y as f64;
                n += 1.0;
            }
        }
    }
    if n < 500.0 {
        return None;
    }
    Some(DiskFit {
        xc: sx / n,
        yc: sy / n,
        r: (n / std::f64::consts::PI).sqrt(),
    })
}

/// Polar limb refinement: sub-pixel radial edge at many position angles,
/// robust Fourier fit r(theta) = r0 + a cos + b sin + c cos2 + d sin2.
/// Far more stable than conic fitting on near-circular, spicule-fuzzy
/// limbs. Returns the refined fit and the relative ellipticity
/// 2*sqrt(c^2+d^2)/r0.
pub fn fit_disk_polar(img: &Image, init: &DiskFit) -> Option<(DiskFit, f64)> {
    let mut xc = init.xc;
    let mut yc = init.yc;
    let mut r0 = init.r;
    let mut ellip = 0.0;
    for _ in 0..3 {
        let n_pa = 256usize;
        let mut thetas = Vec::new();
        let mut radii = Vec::new();
        for a in 0..n_pa {
            let th = a as f64 / n_pa as f64 * std::f64::consts::TAU;
            let (ct, st) = (th.cos(), th.sin());
            // radial profile over [0.85, 1.15] r0
            let n_s = 90usize;
            let mut prof = vec![0.0f64; n_s];
            let mut ok = true;
            for i in 0..n_s {
                let r = r0 * (0.85 + 0.30 * i as f64 / (n_s - 1) as f64);
                let x = xc + r * ct;
                let y = yc + r * st;
                if x < 1.0 || y < 1.0 || x > img.w as f64 - 2.0 || y > img.h as f64 - 2.0 {
                    ok = false;
                    break;
                }
                prof[i] = bilinear(img, x, y) as f64;
            }
            if !ok {
                continue;
            }
            let sm = crate::mathutil::gaussian_smooth(&prof, 2.0);
            // steepest descent (limb = falling edge outward), centroid refine
            let grad: Vec<f64> = (0..n_s)
                .map(|i| {
                    let a2 = i.saturating_sub(1);
                    let b2 = (i + 1).min(n_s - 1);
                    (sm[a2] - sm[b2]).max(0.0)
                })
                .collect();
            let imax = grad
                .iter()
                .enumerate()
                .max_by(|u, v| u.1.partial_cmp(v.1).unwrap())
                .map(|(i, _)| i)?;
            if grad[imax] <= 1e-9 || imax < 4 || imax > n_s - 5 {
                continue;
            }
            let lo = imax.saturating_sub(6);
            let hi = (imax + 7).min(n_s);
            let mut sw = 0.0;
            let mut swi = 0.0;
            for i in lo..hi {
                sw += grad[i];
                swi += grad[i] * i as f64;
            }
            let pos = swi / sw;
            let r = r0 * (0.85 + 0.30 * pos / (n_s - 1) as f64);
            thetas.push(th);
            radii.push(r);
        }
        if thetas.len() < 64 {
            return None;
        }
        // robust Fourier LS: r = r0 + a cos + b sin + c cos2 + d sin2
        let mut w = vec![1.0f64; thetas.len()];
        let mut coef = [0.0f64; 5];
        for _ in 0..4 {
            // normal equations 5x5
            let mut ata = [[0.0f64; 5]; 5];
            let mut atb = [0.0f64; 5];
            for (k, (&th, &r)) in thetas.iter().zip(&radii).enumerate() {
                let basis = [1.0, th.cos(), th.sin(), (2.0 * th).cos(), (2.0 * th).sin()];
                for i in 0..5 {
                    for j in 0..5 {
                        ata[i][j] += w[k] * basis[i] * basis[j];
                    }
                    atb[i] += w[k] * basis[i] * r;
                }
            }
            let m = nalgebra::Matrix5::from_fn(|i, j| ata[i][j]);
            let b = nalgebra::Vector5::from_fn(|i, _| atb[i]);
            let sol = m.lu().solve(&b)?;
            for i in 0..5 {
                coef[i] = sol[i];
            }
            // Tukey reweight
            let mut res: Vec<f64> = thetas
                .iter()
                .zip(&radii)
                .map(|(&th, &r)| {
                    r - (coef[0] + coef[1] * th.cos() + coef[2] * th.sin()
                        + coef[3] * (2.0 * th).cos() + coef[4] * (2.0 * th).sin())
                })
                .collect();
            let mut ares: Vec<f64> = res.iter().map(|r| r.abs()).collect();
            let mad = crate::mathutil::median_inplace(&mut ares).max(1e-6);
            let cs = 4.685 * 1.4826 * mad;
            for k in 0..w.len() {
                let u = res[k] / cs;
                w[k] = if u.abs() < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
            }
            res.clear();
        }
        // update: center shift from order-1 terms, radius from r0 term
        xc += coef[1];
        yc += coef[2];
        r0 = coef[0];
        ellip = 2.0 * (coef[3] * coef[3] + coef[4] * coef[4]).sqrt() / coef[0];
    }
    Some((DiskFit { xc, yc, r: r0 }, ellip))
}

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

/// Resample recon onto the ground-truth grid given disk fits and offsets.
fn resample_to_gt(recon: &Image, gt_size: (usize, usize), rf: &DiskFit, gf: &DiskFit, dx: f64, dy: f64) -> Image {
    let s = rf.r / gf.r;
    let mut out = Image::new(gt_size.0, gt_size.1);
    for y in 0..gt_size.1 {
        for x in 0..gt_size.0 {
            let xr = rf.xc + (x as f64 - gf.xc) * s + dx;
            let yr = rf.yc + (y as f64 - gf.yc) * s + dy;
            out.set(x, y, bilinear(recon, xr, yr));
        }
    }
    out
}

fn ncc_masked(a: &Image, b: &Image, mask: &[bool]) -> f64 {
    let mut sa = 0.0;
    let mut sb = 0.0;
    let mut n = 0.0;
    for i in 0..mask.len() {
        if mask[i] {
            sa += a.data[i] as f64;
            sb += b.data[i] as f64;
            n += 1.0;
        }
    }
    let (ma, mb) = (sa / n, sb / n);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..mask.len() {
        if mask[i] {
            let va = a.data[i] as f64 - ma;
            let vb = b.data[i] as f64 - mb;
            num += va * vb;
            da += va * va;
            db += vb * vb;
        }
    }
    num / (da * db).sqrt().max(1e-12)
}

/// Coarse-to-fine sub-pixel translation search maximizing masked NCC.
fn find_offset(recon: &Image, gt: &Image, rf: &DiskFit, gf: &DiskFit, mask: &[bool]) -> (f64, f64) {
    let mut best = (0.0f64, 0.0f64, f64::MIN);
    for step in [0.5f64, 0.1, 0.02] {
        let (cx, cy, _) = best;
        let mut local_best = best;
        let mut dy = cy - 2.0 * step;
        while dy <= cy + 2.0 * step + 1e-12 {
            let mut dx = cx - 2.0 * step;
            while dx <= cx + 2.0 * step + 1e-12 {
                let r = resample_to_gt(recon, (gt.w, gt.h), rf, gf, dx, dy);
                let v = ncc_masked(&r, gt, mask);
                if v > local_best.2 {
                    local_best = (dx, dy, v);
                }
                dx += step;
            }
            dy += step;
        }
        best = local_best;
    }
    (best.0, best.1)
}

pub struct EvalResult {
    pub psnr_disk: f64,
    pub ssim_disk: f64,
    pub psnr_limb: f64,
    pub radius_ratio: f64,
    /// erf-fit limb transition width of the aligned recon (px, lower=sharper)
    pub limb_sigma: f64,
    /// per-band residual SNR in dB (DoG bands, coarse->fine)
    pub band_snr: [f64; 4],
    /// max sector photometric deviation, percent
    pub flat_pct: f64,
}

pub fn evaluate(recon: &Image, gt: &Image) -> Option<EvalResult> {
    let rf = fit_disk(recon)?;
    let gf = fit_disk(gt)?;

    // masks on GT grid
    let n = gt.w * gt.h;
    let mut disk_mask = vec![false; n];
    let mut limb_mask = vec![false; n];
    for y in 0..gt.h {
        for x in 0..gt.w {
            let r = (((x as f64 - gf.xc).powi(2) + (y as f64 - gf.yc).powi(2)) as f64).sqrt() / gf.r;
            if r < 0.95 {
                disk_mask[y * gt.w + x] = true;
            }
            if (0.93..1.04).contains(&r) {
                limb_mask[y * gt.w + x] = true;
            }
        }
    }

    // sub-pixel translation search maximizing NCC over the disk
    let best = find_offset(recon, gt, &rf, &gf, &disk_mask);
    let aligned = resample_to_gt(recon, (gt.w, gt.h), &rf, &gf, best.0, best.1);

    // robust linear intensity match: recon ~ a*gt + b over disk
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut cnt = 0.0;
    for i in 0..n {
        if disk_mask[i] {
            let g = gt.data[i] as f64;
            let r = aligned.data[i] as f64;
            sx += g;
            sy += r;
            sxx += g * g;
            sxy += g * r;
            cnt += 1.0;
        }
    }
    let det = cnt * sxx - sx * sx;
    let a = (cnt * sxy - sx * sy) / det;
    let b = (sy * sxx - sx * sxy) / det;

    // normalize both to GT units, peak = 99.9 percentile of GT inside disk
    let mut gtvals: Vec<f32> = (0..n).filter(|&i| disk_mask[i]).map(|i| gt.data[i]).collect();
    gtvals.sort_by(|u, v| u.partial_cmp(v).unwrap());
    let peak = gtvals[(gtvals.len() as f64 * 0.999) as usize] as f64;

    let mut na = Image::new(gt.w, gt.h);
    let mut ng = Image::new(gt.w, gt.h);
    for i in 0..n {
        na.data[i] = (((aligned.data[i] as f64 - b) / a) / peak) as f32;
        ng.data[i] = (gt.data[i] as f64 / peak) as f32;
    }

    let psnr = |mask: &[bool]| {
        let mut se = 0.0;
        let mut cnt = 0.0;
        for i in 0..n {
            if mask[i] {
                let d = (na.data[i] - ng.data[i]) as f64;
                se += d * d;
                cnt += 1.0;
            }
        }
        10.0 * (1.0 / (se / cnt)).log10()
    };

    Some(EvalResult {
        psnr_disk: psnr(&disk_mask),
        ssim_disk: ssim_masked(&na, &ng, &disk_mask),
        psnr_limb: psnr(&limb_mask),
        radius_ratio: rf.r / gf.r,
        limb_sigma: limb_sigma(&na, &gf),
        band_snr: band_snr(&na, &ng, &disk_mask),
        flat_pct: sector_flatness(&na, &ng, &gf),
    })
}

/// Erf-transition width of the limb: average radial derivative profile over
/// many position angles, then second-moment width.
pub fn limb_sigma(img: &Image, disk: &DiskFit) -> f64 {
    let n_pa = 180;
    let r_lo = disk.r - 8.0;
    let n_r = 64usize; // 0.25 px steps
    let mut prof = vec![0.0f64; n_r];
    let mut cnt = vec![0.0f64; n_r];
    for a in 0..n_pa {
        let th = a as f64 / n_pa as f64 * std::f64::consts::TAU;
        let (ct, st) = (th.cos(), th.sin());
        for i in 0..n_r {
            let r = r_lo + i as f64 * 0.25;
            let x = disk.xc + r * ct;
            let y = disk.yc + r * st;
            if x < 1.0 || y < 1.0 || x > img.w as f64 - 2.0 || y > img.h as f64 - 2.0 {
                continue;
            }
            prof[i] += bilinear(img, x, y) as f64;
            cnt[i] += 1.0;
        }
    }
    for i in 0..n_r {
        if cnt[i] > 0.0 {
            prof[i] /= cnt[i];
        }
    }
    // derivative magnitude, background-slope removed
    let mut d: Vec<f64> = (1..n_r - 1).map(|i| (prof[i - 1] - prof[i + 1]).max(0.0)).collect();
    let base = {
        let mut tails: Vec<f64> = d[..6].to_vec();
        tails.extend_from_slice(&d[d.len() - 6..]);
        crate::mathutil::median_inplace(&mut tails)
    };
    for v in d.iter_mut() {
        *v = (*v - base).max(0.0);
    }
    let sw: f64 = d.iter().sum();
    if sw <= 1e-9 {
        return f64::NAN;
    }
    let mu: f64 = d.iter().enumerate().map(|(i, &v)| v * i as f64).sum::<f64>() / sw;
    let var: f64 = d.iter().enumerate().map(|(i, &v)| v * (i as f64 - mu).powi(2)).sum::<f64>() / sw;
    var.sqrt() * 0.25 // steps -> px
}

/// Residual SNR per difference-of-Gaussian band (coarse->fine), dB.
fn band_snr(na: &Image, ng: &Image, mask: &[bool]) -> [f64; 4] {
    let sigmas = [8.0f64, 4.0, 2.0, 1.0, 0.0]; // band i = G(s[i+1]) - G(s[i])
    let mut res = Image::new(na.w, na.h);
    for i in 0..res.data.len() {
        res.data[i] = na.data[i] - ng.data[i];
    }
    let mut out = [0.0f64; 4];
    let blur = |img: &Image, s: f64| {
        if s > 0.0 {
            crate::mathutil::gaussian_blur_2d(img, s, s)
        } else {
            img.clone()
        }
    };
    let mut g_res_prev = blur(&res, sigmas[0]);
    let mut g_sig_prev = blur(ng, sigmas[0]);
    for b in 0..4 {
        let g_res = blur(&res, sigmas[b + 1]);
        let g_sig = blur(ng, sigmas[b + 1]);
        let mut pr = 0.0f64;
        let mut ps = 0.0f64;
        let mut cnt = 0.0f64;
        for i in 0..mask.len() {
            if mask[i] {
                let br = (g_res.data[i] - g_res_prev.data[i]) as f64;
                let bs = (g_sig.data[i] - g_sig_prev.data[i]) as f64;
                pr += br * br;
                ps += bs * bs;
                cnt += 1.0;
            }
        }
        out[b] = 10.0 * ((ps / cnt.max(1.0)) / (pr / cnt.max(1.0)).max(1e-20)).log10();
        g_res_prev = g_res;
        g_sig_prev = g_sig;
    }
    out
}

/// Max deviation of recon/truth mean ratio over 4 radial x 4 azimuthal
/// sectors (percent). Catches slow photometric waves PSNR dilutes.
fn sector_flatness(na: &Image, ng: &Image, disk: &DiskFit) -> f64 {
    let mut sums = [[0.0f64; 2]; 16];
    for y in 0..na.h {
        for x in 0..na.w {
            let dx = x as f64 - disk.xc;
            let dy = y as f64 - disk.yc;
            let r = (dx * dx + dy * dy).sqrt() / disk.r;
            if r >= 0.9 {
                continue;
            }
            let ri = ((r / 0.9) * 4.0) as usize;
            let ai = (((dy.atan2(dx) + std::f64::consts::PI) / std::f64::consts::TAU * 4.0) as usize).min(3);
            let s = &mut sums[ri.min(3) * 4 + ai];
            s[0] += na.at(x, y) as f64;
            s[1] += ng.at(x, y) as f64;
        }
    }
    let mut worst = 0.0f64;
    for s in sums.iter() {
        if s[1] > 1e-9 {
            worst = worst.max((s[0] / s[1] - 1.0).abs());
        }
    }
    worst * 100.0
}

/// RMS velocity error (px) over the disk interior, after registering the
/// intensity pair (velocity maps share their images' grids).
pub fn evaluate_velocity(recon_i: &Image, gt_i: &Image, recon_v: &Image, gt_v: &Image) -> Option<f64> {
    let rf = fit_disk(recon_i)?;
    let gf = fit_disk(gt_i)?;
    let s = rf.r / gf.r;
    // register on intensity, apply the offset to the velocity sampling
    let n = gt_i.w * gt_i.h;
    let mut disk_mask = vec![false; n];
    for y in 0..gt_i.h {
        for x in 0..gt_i.w {
            let dx = x as f64 - gf.xc;
            let dy = y as f64 - gf.yc;
            if (dx * dx + dy * dy).sqrt() / gf.r < 0.95 {
                disk_mask[y * gt_i.w + x] = true;
            }
        }
    }
    let (odx, ody) = find_offset(recon_i, gt_i, &rf, &gf, &disk_mask);
    let mut se = 0.0;
    let mut cnt = 0.0;
    for y in 0..gt_v.h {
        for x in 0..gt_v.w {
            let dx = x as f64 - gf.xc;
            let dy = y as f64 - gf.yc;
            if (dx * dx + dy * dy).sqrt() / gf.r >= 0.9 {
                continue;
            }
            let xr = rf.xc + dx * s + odx;
            let yr = rf.yc + dy * s + ody;
            if xr < 1.0 || yr < 1.0 || xr > recon_v.w as f64 - 2.0 || yr > recon_v.h as f64 - 2.0 {
                continue;
            }
            let d = bilinear(recon_v, xr, yr) as f64 - gt_v.at(x, y) as f64;
            se += d * d;
            cnt += 1.0;
        }
    }
    if cnt < 100.0 {
        return None;
    }
    Some((se / cnt).sqrt())
}

/// Mean SSIM over masked pixels, 11x11 Gaussian window (sigma 1.5), L=1.
fn ssim_masked(a: &Image, b: &Image, mask: &[bool]) -> f64 {
    let (w, h) = (a.w, a.h);
    let c1 = 0.01f64 * 0.01;
    let c2 = 0.03f64 * 0.03;
    // gaussian kernel
    let r = 5isize;
    let sigma = 1.5f64;
    let mut k = Vec::new();
    let mut ks = 0.0;
    for j in -r..=r {
        for i in -r..=r {
            let v = (-((i * i + j * j) as f64) / (2.0 * sigma * sigma)).exp();
            k.push(v);
            ks += v;
        }
    }
    for v in k.iter_mut() {
        *v /= ks;
    }
    let mut total = 0.0f64;
    let mut cnt = 0.0f64;
    for y in (r as usize..h - r as usize).step_by(2) {
        for x in (r as usize..w - r as usize).step_by(2) {
            if !mask[y * w + x] {
                continue;
            }
            let mut ma = 0.0;
            let mut mb = 0.0;
            let mut ki = 0;
            for j in -r..=r {
                for i in -r..=r {
                    let av = a.at((x as isize + i) as usize, (y as isize + j) as usize) as f64;
                    let bv = b.at((x as isize + i) as usize, (y as isize + j) as usize) as f64;
                    ma += k[ki] * av;
                    mb += k[ki] * bv;
                    ki += 1;
                }
            }
            let mut va = 0.0;
            let mut vb = 0.0;
            let mut cov = 0.0;
            ki = 0;
            for j in -r..=r {
                for i in -r..=r {
                    let av = a.at((x as isize + i) as usize, (y as isize + j) as usize) as f64 - ma;
                    let bv = b.at((x as isize + i) as usize, (y as isize + j) as usize) as f64 - mb;
                    va += k[ki] * av * av;
                    vb += k[ki] * bv * bv;
                    cov += k[ki] * av * bv;
                    ki += 1;
                }
            }
            let s = ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            total += s;
            cnt += 1.0;
        }
    }
    total / cnt.max(1.0)
}
