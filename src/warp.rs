//! Geometric correction. GhostSun composes shear + anisotropic scale +
//! P-angle rotation + flips into ONE affine transform and resamples once
//! with Lanczos-3 in f32. INTI applies three successive bilinear warps with
//! uint16 rounding after each — three rounds of low-pass filtering and
//! quantization.

use crate::ellipse::EllipseGeom;
use crate::image2d::Image;
use crate::mathutil::lanczos3;
use rayon::prelude::*;

pub struct WarpParams {
    pub rotation_deg: f64, // P angle, applied about disk center
    pub flip_x: bool,
    pub flip_y: bool,
    pub margin_frac: f64, // canvas margin as a fraction of the radius
    /// F4: scale the x kernel support by the downsampling factor sx so the
    /// warp is a proper footprint-weighted (drizzle-equivalent) downscale —
    /// averages the oversampled scan frames instead of point-sampling them.
    pub filtered_downscale: bool,
    /// keep negative values (signed maps like velocity); intensity clamps
    pub allow_negative: bool,
}

/// Output canvas geometry.
#[allow(dead_code)]
pub struct WarpOutput {
    pub image: Image,
    pub xc: f64,
    pub yc: f64,
    pub radius: f64,
}

/// Lanczos-3 sampling with the x kernel stretched by `scale_x` (>=1):
/// correct anti-aliased, noise-averaging downsampling in x.
#[inline]
fn sample_lanczos3_aniso(img: &Image, x: f64, y: f64, scale_x: f64) -> f32 {
    let sx = scale_x.max(1.0);
    let rx = (3.0 * sx).ceil() as isize;
    let xf = x.floor() as isize;
    let yf = y.floor() as isize;
    if xf < -3 - rx || yf < -3 || xf > img.w as isize + 2 + rx || yf > img.h as isize + 2 {
        return 0.0;
    }
    let mut wy = [0.0f64; 6];
    for k in 0..6 {
        wy[k] = lanczos3(y - (yf - 2 + k as isize) as f64);
    }
    let mut acc = 0.0;
    let mut wsum = 0.0;
    for j in 0..6 {
        let yy = yf - 2 + j as isize;
        for i in -rx..=rx {
            let xx = xf + i;
            let w = lanczos3((x - xx as f64) / sx) * wy[j];
            if w == 0.0 {
                continue;
            }
            acc += w * img.at_clamped(xx, yy) as f64;
            wsum += w;
        }
    }
    if wsum.abs() < 1e-12 {
        0.0
    } else {
        (acc / wsum) as f32
    }
}

#[inline]
fn sample_bilinear(img: &Image, x: f64, y: f64) -> f32 {
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

/// Single composed warp: for each output pixel (centered coords, circular
/// sun geometry, optional rotation/flips) compute the source position in
/// the raw reconstruction and sample once with Lanczos-3.
pub fn warp_single(disk: &Image, geom: &EllipseGeom, p: &WarpParams) -> WarpOutput {
    let r = geom.radius;
    let margin = (p.margin_frac * r).max(40.0);
    let size = (2.0 * (r + margin)).ceil() as usize;
    let oc = size as f64 / 2.0;

    let th = p.rotation_deg.to_radians();
    let (ct, st) = (th.cos(), th.sin());
    let fx = if p.flip_x { -1.0 } else { 1.0 };
    let fy = if p.flip_y { -1.0 } else { 1.0 };
    // F4: when the scan oversamples (sx > 1 raw px per output px), widen the
    // x kernel to average the redundant frames (anti-alias + SNR gain)
    let kx = if p.filtered_downscale { geom.sx.max(1.0) } else { 1.0 };

    let rows: Vec<Vec<f32>> = (0..size)
        .into_par_iter()
        .map(|vy| {
            let mut row = vec![0.0f32; size];
            let v0 = vy as f64 - oc;
            for (vxi, out) in row.iter_mut().enumerate() {
                let u0 = vxi as f64 - oc;
                // flips (about center)
                let u1 = u0 * fx;
                let v1 = v0 * fy;
                // rotation about center
                let u = u1 * ct - v1 * st;
                let v = u1 * st + v1 * ct;
                // circle -> raw ellipse: x = sx*(u + shear*v), y = v
                let x = geom.sx * (u + geom.shear * v) + geom.xc;
                let y = v + geom.yc;
                let v = sample_lanczos3_aniso(disk, x, y, kx);
                *out = if p.allow_negative { v } else { v.max(0.0) };
            }
            row
        })
        .collect();

    let mut image = Image::new(size, size);
    for (y, row) in rows.iter().enumerate() {
        image.row_mut(y).copy_from_slice(row);
    }
    WarpOutput { image, xc: oc, yc: oc, radius: r }
}

/// INTI-baseline: three successive bilinear resamplings, each rounded to
/// uint16 (shear about the disk column, x-rescale, rotation about center).
pub fn warp_baseline(disk: &Image, geom: &EllipseGeom, p: &WarpParams) -> WarpOutput {
    let quantize = |img: &mut Image| {
        for v in img.data.iter_mut() {
            *v = v.clamp(0.0, 65535.0).round();
        }
    };

    // 1. shear (tilt correction): undo x = sx*(u + shear*v) by shifting each
    //    row horizontally, one bilinear pass + uint16 rounding like INTI's
    //    map_coordinates(order=1) + uint16 copy.
    let mut img1 = Image::new(disk.w, disk.h);
    for y in 0..disk.h {
        let dy = y as f64 - geom.yc;
        let xoff = geom.sx * geom.shear * dy;
        for x in 0..disk.w {
            img1.set(x, y, sample_bilinear(disk, x as f64 + xoff, y as f64));
        }
    }
    quantize(&mut img1);

    // 2. x-rescale by 1/sx (bilinear zoom)
    let new_w = (disk.w as f64 / geom.sx).round() as usize;
    let mut img2 = Image::new(new_w, disk.h);
    for y in 0..disk.h {
        for x in 0..new_w {
            let xs = x as f64 * geom.sx;
            img2.set(x, y, sample_bilinear(&img1, xs, y as f64));
        }
    }
    quantize(&mut img2);

    // 3. rotation about disk center (bilinear)
    let xc2 = geom.xc / geom.sx;
    let th = p.rotation_deg.to_radians();
    let (ct, st) = (th.cos(), th.sin());
    let mut img3 = Image::new(new_w, disk.h);
    for y in 0..disk.h {
        for x in 0..new_w {
            let dx = x as f64 - xc2;
            let dy = y as f64 - geom.yc;
            let sx_ = dx * ct - dy * st + xc2;
            let sy_ = dx * st + dy * ct + geom.yc;
            img3.set(x, y, sample_bilinear(&img2, sx_, sy_));
        }
    }
    quantize(&mut img3);

    // crop to square canvas around center for comparability
    let r = geom.radius;
    let margin = (p.margin_frac * r).max(40.0);
    let size = (2.0 * (r + margin)).ceil() as usize;
    let oc = size as f64 / 2.0;
    let mut out = Image::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let sx_ = x as f64 - oc + xc2;
            let sy_ = y as f64 - oc + geom.yc;
            let xi = sx_.round() as isize;
            let yi = sy_.round() as isize;
            if xi >= 0 && yi >= 0 && (xi as usize) < img3.w && (yi as usize) < img3.h {
                out.set(x, y, img3.at(xi as usize, yi as usize));
            }
        }
    }
    WarpOutput { image: out, xc: oc, yc: oc, radius: r }
}
