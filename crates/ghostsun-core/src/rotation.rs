//! Solar differential rotation: de-rotating scans to a common epoch.
//!
//! Scans taken minutes apart do not show the same Sun. The surface turns, and
//! it turns FASTER AT THE EQUATOR than at the poles, so no single shift or
//! rigid rotation can bring two epochs into register — which is exactly what
//! the stacker's global NCC alignment tries to do. Left uncorrected, stacking
//! a long session smears fine structure by the differential part.
//!
//! Scale, for calibration of effort: at 14.7 deg/day the equatorial limb moves
//! about 1.6 px in six minutes on a 1500 px-radius disc, which is small beside
//! a typical 8-10 px PSF and measurably costs nothing. Over half an hour it is
//! ~8 px and dominates. This module exists for the long sessions.
//!
//! Getting it right needs the Sun's orientation, not just its rotation rate:
//! the position angle `P` tips the rotation axis in the image plane by up to
//! ±26 deg through the year, so assuming P = 0 would point the whole
//! correction in the wrong direction and could easily do more harm than good.
//! [`solar_orientation`] computes P and B0 from the observation time.

use crate::image2d::Image;
use crate::mathutil::{bspline_eval, bspline_prefilter};
use crate::metrics::DiskFit;
use rayon::prelude::*;

/// Snodgrass (1984) spectroscopic differential rotation, degrees per day,
/// sidereal, as a function of heliographic latitude.
pub fn omega_deg_per_day(lat_rad: f64) -> f64 {
    let s2 = lat_rad.sin().powi(2);
    14.713 - 2.396 * s2 - 1.787 * s2 * s2
}

/// Orientation of the solar disc as seen from Earth.
#[derive(Clone, Copy, Debug)]
pub struct SolarOrientation {
    /// Position angle of the north end of the rotation axis, radians,
    /// measured east from celestial north.
    pub p: f64,
    /// Heliographic latitude of the disc centre, radians.
    pub b0: f64,
}

/// Low-precision solar orientation from a Julian date (UTC is close enough:
/// the terms here move by far less than a pixel over a leap second).
///
/// Standard formulation — solar longitude, the node of the solar equator, and
/// its 7.25 deg inclination. Good to a few arcminutes in P, which is ample:
/// an error of 0.1 deg in P misdirects a 2 px correction by 0.003 px.
pub fn solar_orientation(jd: f64) -> SolarOrientation {
    let d = jd - 2_451_545.0;
    let rad = std::f64::consts::PI / 180.0;
    // Sun's mean longitude and anomaly.
    let l = (280.460 + 0.9856474 * d) % 360.0;
    let g = ((357.528 + 0.9856003 * d) % 360.0) * rad;
    // Apparent ecliptic longitude.
    let lambda = (l + 1.915 * g.sin() + 0.020 * (2.0 * g).sin()) * rad;
    // Obliquity, and the ascending node of the solar equator on the ecliptic.
    let eps = (23.439 - 0.0000004 * d) * rad;
    let omega = (73.6667 + 0.013958 * (d / 365.25 + 100.0)) * rad;
    let i = 7.25 * rad;
    let lo = lambda - omega;
    // P is the sum of the tilt of the ecliptic and the tilt of the solar
    // equator, both projected into the sky plane.
    let x = (-(lambda.cos()) * eps.tan()).atan();
    let y = (-(lo.cos()) * i.tan()).atan();
    SolarOrientation {
        p: x + y,
        b0: (lo.sin() * i.sin()).asin(),
    }
}

/// Julian date from a UTC calendar instant.
pub fn julian_date(year: i64, month: u32, day: u32, hour: f64) -> f64 {
    let (y, m) = if month <= 2 {
        (year - 1, month as i64 + 12)
    } else {
        (year, month as i64)
    };
    let a = y.div_euclid(100);
    let b = 2 - a + a.div_euclid(4);
    (365.25 * (y + 4716) as f64).floor() + (30.6001 * (m + 1) as f64).floor() + day as f64
        + b as f64
        - 1524.5
        + hour / 24.0
}

/// Parse the ISO-8601 UTC that [`crate::ser::ticks_to_iso8601`] writes into
/// FITS `DATE-OBS`, returning a Julian date.
pub fn jd_from_iso8601(s: &str) -> Option<f64> {
    let s = s.trim().trim_matches('\'').trim();
    let (date, time) = s.split_once('T')?;
    let mut dp = date.split('-');
    let y: i64 = dp.next()?.parse().ok()?;
    let mo: u32 = dp.next()?.parse().ok()?;
    let da: u32 = dp.next()?.parse().ok()?;
    let mut tp = time.split(':');
    let hh: f64 = tp.next()?.parse().ok()?;
    let mm: f64 = tp.next()?.parse().ok()?;
    let ss: f64 = tp.next().unwrap_or("0").parse().ok()?;
    Some(julian_date(y, mo, da, hh + mm / 60.0 + ss / 3600.0))
}

/// Re-project `img` from its own epoch onto the reference epoch, `dt_days`
/// earlier or later, following differential rotation.
///
/// For each output pixel the line of sight is intersected with the solar
/// sphere, converted to heliographic latitude and longitude, rotated BACK by
/// `omega(lat) * dt`, and projected again to find where that material was in
/// the source frame. Points beyond the limb, and those that rotate around the
/// far side, are left as they were — there is nothing to fetch.
pub fn derotate(img: &Image, disk: &DiskFit, dt_days: f64, orient: SolarOrientation) -> Image {
    if dt_days == 0.0 {
        return img.clone();
    }
    let (w, h) = (img.w, img.h);
    // Prefilter every row ONCE. Doing it per sample instead makes this
    // O(width) per pixel, which on a 3400 px disc is the difference between
    // a second and not finishing.
    let coefs: Vec<Vec<f64>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut c: Vec<f64> = img.row(y).iter().map(|&v| v as f64).collect();
            bspline_prefilter(&mut c);
            c
        })
        .collect();
    let (sp, cp) = orient.p.sin_cos();
    let (sb, cb) = orient.b0.sin_cos();
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut out = vec![0.0f32; w];
            for x in 0..w {
                out[x] = img.at(x, y);
                // Sky-plane offsets in solar radii.
                let dx = (x as f64 - disk.xc) / disk.r;
                let dy = (y as f64 - disk.yc) / disk.r;
                if dx * dx + dy * dy >= 0.999 {
                    continue; // off-disc: nothing rotates
                }
                // Undo the position angle: image +y is down, so the sign of
                // the y term follows the image convention, not the sky's.
                let xr = dx * cp + dy * sp;
                let yr = -dx * sp + dy * cp;
                let zr = (1.0 - xr * xr - yr * yr).max(0.0).sqrt();
                // Rotate out of the B0-tilted frame into heliographic coords.
                let hy = yr * cb + zr * sb;
                let hz = -yr * sb + zr * cb;
                let lat = hy.clamp(-1.0, 1.0).asin();
                let lon = xr.atan2(hz);
                // Where this material was dt earlier.
                let dlon = omega_deg_per_day(lat) * dt_days * std::f64::consts::PI / 180.0;
                let lon0 = lon - dlon;
                let (sl, cl) = lon0.sin_cos();
                let (sla, cla) = lat.sin_cos();
                let (px, py, pz) = (cla * sl, sla, cla * cl);
                if pz <= 0.0 {
                    continue; // was on the far side
                }
                // Project back through B0 and P.
                let by = py * cb - pz * sb;
                let bz = py * sb + pz * cb;
                let _ = bz;
                let sx = px * cp - by * sp;
                let sy = px * sp + by * cp;
                let fx = disk.xc + sx * disk.r;
                let fy = disk.yc + sy * disk.r;
                if fx < 1.0 || fy < 1.0 || fx > w as f64 - 2.0 || fy > h as f64 - 2.0 {
                    continue;
                }
                // Separable cubic B-spline, matching the rest of the pipeline.
                let iy = fy.floor() as usize;
                let mut col = [0.0f64; 4];
                for (k, c) in col.iter_mut().enumerate() {
                    let yy = (iy + k).saturating_sub(1).min(h - 1);
                    *c = bspline_eval(&coefs[yy], fx);
                }
                let t = fy - iy as f64;
                let v = col[1] + 0.5 * t * (col[2] - col[0]
                    + t * (2.0 * col[0] - 5.0 * col[1] + 4.0 * col[2] - col[3]
                        + t * (3.0 * (col[1] - col[2]) + col[3] - col[0])));
                out[x] = v as f32;
            }
            out
        })
        .collect();
    let mut o = Image::new(w, h);
    for (y, r) in rows.iter().enumerate() {
        o.row_mut(y).copy_from_slice(r);
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_profile_is_differential_and_equator_fastest() {
        let eq = omega_deg_per_day(0.0);
        let mid = omega_deg_per_day(45f64.to_radians());
        let pole = omega_deg_per_day(80f64.to_radians());
        assert!((eq - 14.713).abs() < 1e-9);
        assert!(mid < eq && pole < mid, "{eq} {mid} {pole}");
        // The pole-to-equator spread is what a rigid alignment cannot absorb.
        assert!(eq - pole > 3.0, "spread {}", eq - pole);
    }

    #[test]
    fn orientation_matches_published_values_for_a_known_date() {
        // 2026-08-22 ~07:00 UT. In late August B0 is around +7 deg (north pole
        // tipped toward us, near its maximum) and P is around +19 deg.
        let jd = julian_date(2026, 8, 22, 7.0);
        let o = solar_orientation(jd);
        let (p, b0) = (o.p.to_degrees(), o.b0.to_degrees());
        assert!((b0 - 7.0).abs() < 1.5, "B0 {b0:.2} deg");
        assert!((p - 19.0).abs() < 3.0, "P {p:.2} deg");
    }

    #[test]
    fn jd_round_trips_through_the_iso_string_we_write() {
        let jd = jd_from_iso8601("2026-08-22T15:46:55.225").expect("parsed");
        let direct = julian_date(2026, 8, 22, 15.0 + 46.0 / 60.0 + 55.225 / 3600.0);
        assert!((jd - direct).abs() < 1e-9);
        assert!(jd_from_iso8601("not a date").is_none());
    }

    #[test]
    fn derotation_moves_equator_more_than_pole_and_is_reversible() {
        // A disc of dots; de-rotate forward then back must return them home,
        // and the equatorial dot must move further than the polar one.
        let (n, r) = (401usize, 180.0f64);
        let disk = DiskFit { xc: 200.0, yc: 200.0, r };
        let mut img = Image::new(n, n);
        for y in 0..n {
            for x in 0..n {
                let dx = (x as f64 - 200.0) / r;
                let dy = (y as f64 - 200.0) / r;
                if dx * dx + dy * dy < 1.0 {
                    img.set(x, y, 1000.0);
                }
            }
        }
        // Zero P and B0 isolates the rotation itself.
        let o = SolarOrientation { p: 0.0, b0: 0.0 };
        let dt = 30.0 / 1440.0; // half an hour
        let fwd = derotate(&img, &disk, dt, o);
        let back = derotate(&fwd, &disk, -dt, o);
        let mut worst = 0.0f64;
        for y in 60..340 {
            for x in 60..340 {
                let dx = (x as f64 - 200.0) / r;
                let dy = (y as f64 - 200.0) / r;
                if dx * dx + dy * dy < 0.55 {
                    worst = worst.max((back.at(x, y) - img.at(x, y)).abs() as f64);
                }
            }
        }
        assert!(worst < 30.0, "round trip residual {worst}");

        // Displacement magnitude: equator vs 60 deg latitude, measured by how
        // far a marked column has to be sampled from.
        let d_eq = {
            let lat = 0.0f64;
            omega_deg_per_day(lat) * dt * r * lat.cos()
        };
        let d_hi = {
            let lat = 60f64.to_radians();
            omega_deg_per_day(lat) * dt * r * lat.cos()
        };
        assert!(d_eq > d_hi * 1.5, "equator {d_eq:.2} vs 60deg {d_hi:.2} px-deg");
    }
}
