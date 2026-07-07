//! Synthetic spectroheliograph scan generator with known ground truth.
//!
//! Simulates the full physical chain: a limb-darkened chromosphere with
//! network/filaments/plage/prominences, scanned by a slit with spectral
//! smile (quadratic line curvature), slit tilt (shear), anisotropic scan
//! sampling, per-frame seeing jitter, sky-transparency fluctuations,
//! slit-dust transversalium, photon + read noise, 16-bit quantization.
//!
//! Phase-0 extensions (all default OFF to preserve reference numbers):
//! Doppler velocity field, spectral flexure drift, anisotropic PSF
//! (seeing + slit boxcar), multi-scan series with solar evolution,
//! exposure scaling.

use crate::image2d::Image;
use crate::ser::write_ser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::path::Path;

/// Fixed encoding for velocity PNGs: 0..65535 <-> -VEL_SCALE..+VEL_SCALE px.
pub const VEL_SCALE: f64 = 1.5;

pub struct SynthParams {
    pub spec_w: usize,   // spectral axis (px)
    pub slit_h: usize,   // slit axis (px)
    pub n_frames: usize, // scan length
    pub radius: f64,     // solar radius in slit px
    pub scan_step: f64,  // sun px advanced per frame (X anisotropy)
    pub tilt_deg: f64,   // slit tilt -> shear
    pub jitter_sigma: f64,
    /// seeing jitter ALONG the scan direction (sun px); 0 = off
    pub jitter_x_sigma: f64,
    pub seed: u64,
    /// disable noise, jitter, transparency, transversalium (ceiling tests)
    pub clean: bool,
    /// enable a Doppler velocity field (rotation + turbulence)
    pub doppler: bool,
    /// spectral dispersion for velocity <-> px conversion
    pub dispersion_kms_per_px: f64,
    /// peak slow spectral drift of the line over the scan (px); 0 = off
    pub flexure_px: f64,
    /// seeing blur sigma in sun px; 0 = off (also enables slit boxcar)
    pub psf_seeing_px: f64,
    /// number of sequential scans to emit (multi-scan stacking tests)
    pub n_scans: usize,
    /// signal level multiplier (low-SNR tests)
    pub exposure: f64,
    /// add two telluric absorption lines (shift with flexure only)
    pub telluric: bool,
    /// fraction of frames hit by seeing bursts (blur + displacement runs)
    pub bursts: f64,
}

impl Default for SynthParams {
    fn default() -> Self {
        SynthParams {
            spec_w: 160,
            slit_h: 600,
            n_frames: 900,
            radius: 255.0,
            scan_step: 0.62,
            tilt_deg: 2.0,
            jitter_sigma: 0.35,
            jitter_x_sigma: 0.0,
            seed: 42,
            clean: false,
            doppler: false,
            dispersion_kms_per_px: 5.0,
            flexure_px: 0.0,
            psf_seeing_px: 0.0,
            n_scans: 1,
            exposure: 1.0,
            telluric: false,
            bursts: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Procedural band-limited value noise (deterministic).
// ---------------------------------------------------------------------------

fn hash2(ix: i64, iy: i64, seed: u64) -> f64 {
    let mut h = seed
        .wrapping_add(ix as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (iy as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    (h as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    let sx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let sy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    let v00 = hash2(ix, iy, seed);
    let v10 = hash2(ix + 1, iy, seed);
    let v01 = hash2(ix, iy + 1, seed);
    let v11 = hash2(ix + 1, iy + 1, seed);
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sy
}

fn fbm(x: f64, y: f64, seed: u64, octaves: u32, base_wl: f64) -> f64 {
    let mut amp = 1.0;
    let mut freq = 1.0 / base_wl;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        sum += amp * value_noise(x * freq, y * freq, seed.wrapping_add(o as u64 * 1013));
        norm += amp;
        amp *= 0.55;
        freq *= 2.1;
    }
    sum / norm
}

// ---------------------------------------------------------------------------
// Solar model
// ---------------------------------------------------------------------------

struct Filament {
    x: f64,
    y: f64,
    len: f64,
    width: f64,
    angle: f64,
    depth: f64,
}

struct Prominence {
    pa: f64,     // position angle, rad
    height: f64, // px above limb
    width_pa: f64,
    bright: f64,
}

pub struct SunModel {
    radius: f64,
    seed: u64,
    filaments: Vec<Filament>,
    plages: Vec<Filament>, // reuse shape struct, positive contrast
    proms: Vec<Prominence>,
    /// network-texture advection offset (solar evolution between scans)
    evo_dx: f64,
    /// prominence brightness scale (evolution)
    prom_scale: f64,
}

impl SunModel {
    pub fn new(radius: f64, seed: u64) -> SunModel {
        Self::with_evolution(radius, seed, 0.0, 1.0)
    }

    pub fn with_evolution(radius: f64, seed: u64, evo_dx: f64, prom_scale: f64) -> SunModel {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(7919));
        let mut filaments = Vec::new();
        for _ in 0..7 {
            let r = rng.gen_range(0.05..0.75) * radius;
            let th = rng.gen_range(0.0..std::f64::consts::TAU);
            filaments.push(Filament {
                x: r * th.cos(),
                y: r * th.sin(),
                len: rng.gen_range(0.12..0.4) * radius,
                width: rng.gen_range(2.5..7.0),
                angle: rng.gen_range(0.0..std::f64::consts::PI),
                depth: rng.gen_range(0.35..0.6),
            });
        }
        let mut plages = Vec::new();
        for _ in 0..5 {
            let r = rng.gen_range(0.1..0.7) * radius;
            let th = rng.gen_range(0.0..std::f64::consts::TAU);
            plages.push(Filament {
                x: r * th.cos(),
                y: r * th.sin(),
                len: rng.gen_range(0.05..0.15) * radius,
                width: rng.gen_range(6.0..18.0),
                angle: rng.gen_range(0.0..std::f64::consts::PI),
                depth: rng.gen_range(0.25..0.5), // used as brightness boost
            });
        }
        let mut proms = Vec::new();
        for _ in 0..4 {
            proms.push(Prominence {
                pa: rng.gen_range(0.0..std::f64::consts::TAU),
                height: rng.gen_range(8.0..30.0),
                width_pa: rng.gen_range(0.06..0.18),
                bright: rng.gen_range(0.08..0.2),
            });
        }
        SunModel { radius, seed, filaments, plages, proms, evo_dx, prom_scale }
    }

    /// Chromospheric line-core intensity at sun coords (origin = disk center),
    /// normalized so disk-center quiet sun = 1.0.
    pub fn core_intensity(&self, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        let rn = r / self.radius;
        if rn < 1.0 {
            let mu = (1.0 - rn * rn).max(0.0).sqrt();
            // weak chromospheric limb darkening
            let ld = 1.0 - 0.35 * (1.0 - mu);
            // network texture: two scales (advected by evolution offset)
            let xa = x + self.evo_dx;
            let net = 1.0 + 0.22 * fbm(xa, y, self.seed, 4, 30.0) + 0.10 * fbm(xa, y, self.seed + 555, 3, 9.0);
            let mut v = ld * net.max(0.05);
            for f in &self.filaments {
                let (dx, dy) = (x - f.x, y - f.y);
                let (c, s) = (f.angle.cos(), f.angle.sin());
                let u = dx * c + dy * s;
                let w = -dx * s + dy * c;
                let g = (-(u * u) / (2.0 * f.len * f.len) - (w * w) / (2.0 * f.width * f.width)).exp();
                v *= 1.0 - f.depth * g;
            }
            for p in &self.plages {
                let (dx, dy) = (x - p.x, y - p.y);
                let (c, s) = (p.angle.cos(), p.angle.sin());
                let u = dx * c + dy * s;
                let w = -dx * s + dy * c;
                let g = (-(u * u) / (2.0 * p.len * p.len) - (w * w) / (2.0 * p.width * p.width)).exp();
                v *= 1.0 + p.depth * g;
            }
            v
        } else {
            // prominences: emission above the limb
            let pa = y.atan2(x);
            let h = r - self.radius;
            let mut v: f64 = 0.0;
            for p in &self.proms {
                let mut dpa = pa - p.pa;
                while dpa > std::f64::consts::PI { dpa -= std::f64::consts::TAU; }
                while dpa < -std::f64::consts::PI { dpa += std::f64::consts::TAU; }
                let radial = (-(h / p.height).powi(2)).exp() * (1.0 + 0.6 * fbm(x, y, self.seed + 99, 3, 12.0));
                let angular = (-(dpa / p.width_pa).powi(2)).exp();
                v = v.max(p.bright * radial * angular);
            }
            (v * self.prom_scale).max(0.0)
        }
    }

    /// Photospheric continuum intensity (strong limb darkening, granulation).
    pub fn continuum_intensity(&self, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        let rn = r / self.radius;
        if rn >= 1.0 {
            return 0.0;
        }
        let mu = (1.0 - rn * rn).max(0.0).sqrt();
        let ld = 1.0 - 0.85 * (1.0 - mu); // strong photospheric limb darkening
        let gran = 1.0 + 0.06 * fbm(x + self.evo_dx, y, self.seed + 1234, 3, 6.0);
        (ld * gran).max(0.0)
    }

    /// Line-of-sight velocity in spectral px: solar rotation (linear in x)
    /// plus small-scale turbulence. Zero off-disk.
    pub fn velocity_px(&self, x: f64, y: f64, dispersion_kms_per_px: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        if r >= self.radius {
            return 0.0;
        }
        let rot_kms = 2.0 * (x / self.radius); // +-2 km/s at the limbs
        let turb_px = 0.15 * fbm(x, y, self.seed + 777, 3, 20.0);
        rot_kms / dispersion_kms_per_px + turb_px
    }

    /// Render the canonical ground-truth line-core image on a square-pixel
    /// grid, disk centered, 1 sun px per image px. `stencil` optionally
    /// applies the same PSF blur as the scan simulation.
    pub fn render_ground_truth(&self, size: usize, stencil: &[(f64, f64, f64)]) -> Image {
        let mut img = Image::new(size, size);
        let c = size as f64 / 2.0;
        for yy in 0..size {
            for xx in 0..size {
                let (x, y) = (xx as f64 - c, yy as f64 - c);
                let mut v = 0.0;
                for &(ox, oy, w) in stencil {
                    v += w * self.core_intensity(x + ox, y + oy);
                }
                img.set(xx, yy, v as f32);
            }
        }
        img
    }

    pub fn render_ground_truth_velocity(&self, size: usize, dispersion: f64) -> Image {
        let mut img = Image::new(size, size);
        let c = size as f64 / 2.0;
        for yy in 0..size {
            for xx in 0..size {
                let v = self.velocity_px(xx as f64 - c, yy as f64 - c, dispersion);
                img.set(xx, yy, v as f32);
            }
        }
        img
    }
}

/// Sampling stencil (offsets + weights) modeling seeing (2-D Gaussian) and
/// the slit boxcar along the scan direction. Identity when both are zero.
fn psf_stencil(seeing_sigma: f64, slit_px: f64) -> Vec<(f64, f64, f64)> {
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    let seeing: Vec<(f64, f64, f64)> = if seeing_sigma > 0.0 {
        let mut s = Vec::new();
        let step = seeing_sigma; // 3x3 quadrature at +-1 sigma spacing
        for j in -1i32..=1 {
            for i in -1i32..=1 {
                let w = (-0.5 * ((i * i + j * j) as f64)).exp();
                s.push((i as f64 * step, j as f64 * step, w));
            }
        }
        s
    } else {
        vec![(0.0, 0.0, 1.0)]
    };
    let slit: Vec<(f64, f64)> = if slit_px > 0.0 {
        vec![(-slit_px / 3.0, 1.0), (0.0, 1.0), (slit_px / 3.0, 1.0)]
    } else {
        vec![(0.0, 1.0)]
    };
    for &(sx, sy, sw) in &seeing {
        for &(bx, bw) in &slit {
            pts.push((sx + bx, sy, sw * bw));
        }
    }
    let total: f64 = pts.iter().map(|p| p.2).sum();
    for p in pts.iter_mut() {
        p.2 /= total;
    }
    pts
}

// ---------------------------------------------------------------------------
// Scan simulation
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct SynthTruth {
    pub jitter: Vec<f64>,
    pub transparency: Vec<f64>,
    pub row_gain: Vec<f64>,
    pub smile: [f64; 3], // c0 + c1*yc + c2*yc^2, yc = y - slit_h/2
    pub flex: Vec<f64>,
    pub jitter_x: Vec<f64>,
    /// frames degraded by a seeing burst
    pub burst_mask: Vec<bool>,
    pub psf_seeing_px: f64,
    pub n_scans: usize,
}

pub fn generate(params: &SynthParams, out_ser: &Path, out_truth_png: &Path) -> std::io::Result<SynthTruth> {
    let p = params;
    let stencil = psf_stencil(
        p.psf_seeing_px,
        if p.psf_seeing_px > 0.0 { p.scan_step } else { 0.0 },
    );

    let yc_slit = p.slit_h as f64 / 2.0;
    let shear = p.tilt_deg.to_radians().tan();

    // spectral line geometry (smile)
    let c0 = p.spec_w as f64 * 0.5;
    let c1 = 0.008;
    let c2 = 6.0 / (yc_slit * yc_slit); // ~6 px sagitta at slit ends
    let sigma_line = 2.3;
    let line_depth_quiet: f64 = 0.82;

    let mut truth_out: Option<SynthTruth> = None;

    for scan in 0..p.n_scans.max(1) {
        let scan_seed = p.seed.wrapping_add(scan as u64 * 7717);
        let mut rng = StdRng::seed_from_u64(scan_seed);
        let evo_dx = scan as f64 * 0.8; // gentle fibril-scale evolution per scan
        let prom_scale = 1.0 + if scan == 0 { 0.0 } else { rng.gen_range(-0.1..0.1) };
        let model = SunModel::with_evolution(p.radius, p.seed, evo_dx, prom_scale);

        // per-scan pointing offset (0 for the first scan)
        let (off_x, off_y) = if scan == 0 {
            (0.0, 0.0)
        } else {
            (rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0))
        };
        let t_center = p.n_frames as f64 / 2.0 + off_x / p.scan_step;

        // transversalium: slit-fixed row gains (same slit every scan)
        let mut dust_rng = StdRng::seed_from_u64(p.seed.wrapping_mul(31));
        let mut row_gain = vec![1.0f64; p.slit_h];
        let n_dust = if p.clean { 0 } else { 14 };
        for _ in 0..n_dust {
            let y0 = dust_rng.gen_range(10.0..(p.slit_h as f64 - 10.0));
            let w = dust_rng.gen_range(0.6..2.8);
            let d = dust_rng.gen_range(0.02..0.12);
            for y in 0..p.slit_h {
                let g = (-((y as f64 - y0) / w).powi(2)).exp();
                row_gain[y] *= 1.0 - d * g;
            }
        }

        // transparency: slow sine + AR(1) + one cloud event
        let mut transparency = vec![1.0f64; p.n_frames];
        let mut ar = 0.0;
        let n01 = Normal::new(0.0, 1.0).unwrap();
        for t in 0..p.n_frames {
            ar = 0.92 * ar + 0.008 * n01.sample(&mut rng);
            let slow = 0.03 * (t as f64 / 140.0).sin();
            let cloud = if (300..345).contains(&t) {
                let u = (t as f64 - 322.0) / 12.0;
                -0.15 * (-u * u).exp()
            } else {
                0.0
            };
            transparency[t] = if p.clean { 1.0 } else { (1.0 + slow + ar + cloud).clamp(0.6, 1.2) };
        }

        // seeing jitter along slit: AR(1) + slow drift
        let mut jitter = vec![0.0f64; p.n_frames];
        let mut j = 0.0;
        for t in 0..p.n_frames {
            j = 0.90 * j + p.jitter_sigma * (1.0f64 - 0.90f64 * 0.90).sqrt() * n01.sample(&mut rng);
            let tt = t as f64 / p.n_frames as f64;
            let drift = 1.2 * (tt * 2.7).sin() * tt;
            jitter[t] = if p.clean { 0.0 } else { j + drift };
        }
        // seeing jitter along the scan direction (F9): AR(1), no drift
        let mut jitter_x = vec![0.0f64; p.n_frames];
        let mut jx = 0.0;
        for t in 0..p.n_frames {
            jx = 0.90 * jx + p.jitter_x_sigma * (1.0f64 - 0.90f64 * 0.90).sqrt() * n01.sample(&mut rng);
            jitter_x[t] = if p.clean { 0.0 } else { jx };
        }
        // seeing bursts: runs of 2-8 frames with strong blur + displacement
        let mut burst_mask = vec![false; p.n_frames];
        let mut burst_blur = vec![0.0f64; p.n_frames];
        if p.bursts > 0.0 && !p.clean {
            let mut t = 0;
            while t < p.n_frames {
                if rng.gen_range(0.0..1.0) < p.bursts / 5.0 {
                    let run = rng.gen_range(2..=8usize);
                    let blur = rng.gen_range(2.5..6.0);
                    let kick = rng.gen_range(-3.0..3.0);
                    for k in 0..run.min(p.n_frames - t) {
                        burst_mask[t + k] = true;
                        burst_blur[t + k] = blur;
                        jitter[t + k] += kick;
                        jitter_x[t + k] += kick * 0.7;
                    }
                    t += run;
                } else {
                    t += 1;
                }
            }
        }

        // spectral flexure: slow drift of the whole line over the scan
        let mut flex = vec![0.0f64; p.n_frames];
        if p.flexure_px > 0.0 && !p.clean {
            for t in 0..p.n_frames {
                let tt = t as f64 / p.n_frames as f64;
                flex[t] = p.flexure_px
                    * (0.6 * (std::f64::consts::TAU * tt * 1.3).sin() + 0.4 * tt);
            }
        }

        // photometric scale
        let full_scale = 30000.0 * p.exposure;
        let photon_gain = 1.2; // e-/ADU equivalent
        let read_noise = 22.0;

        let mut frames: Vec<Vec<u16>> = Vec::with_capacity(p.n_frames);
        for t in 0..p.n_frames {
            let mut frame = vec![0u16; p.spec_w * p.slit_h];
            let x_scan = (t as f64 - t_center) * p.scan_step + jitter_x[t];
            for y in 0..p.slit_h {
                let ycs = y as f64 - yc_slit;
                let sun_x = x_scan + shear * ycs;
                let sun_y = ycs + jitter[t] + off_y;
                // PSF-averaged intensities and velocity (bursts add y-blur)
                let mut icont = 0.0;
                let mut icore = 0.0;
                let mut vel = 0.0;
                let bb = burst_blur[t];
                for &(ox, oy, w) in &stencil {
                    let (mut ic, mut ik) = (0.0, 0.0);
                    if bb > 0.0 {
                        for &(bo, bw) in &[(-bb, 0.27), (0.0, 0.46), (bb, 0.27)] {
                            ic += bw * model.continuum_intensity(sun_x + ox, sun_y + oy + bo);
                            ik += bw * model.core_intensity(sun_x + ox, sun_y + oy + bo);
                        }
                    } else {
                        ic = model.continuum_intensity(sun_x + ox, sun_y + oy);
                        ik = model.core_intensity(sun_x + ox, sun_y + oy);
                    }
                    icont += w * ic;
                    icore += w * ik;
                    if p.doppler {
                        vel += w * model.velocity_px(sun_x + ox, sun_y + oy, p.dispersion_kms_per_px);
                    }
                }
                let line_center = c0 + c1 * ycs + c2 * ycs * ycs + flex[t] + vel;
                let smile_here = c0 + c1 * ycs + c2 * ycs * ycs;
                let tell = |x: f64| -> f64 {
                    if !p.telluric {
                        return 1.0;
                    }
                    // fixed wavelength offsets from the smile; move with flexure only
                    let t1 = smile_here + flex[t] - 20.0;
                    let t2 = smile_here + flex[t] + 26.0;
                    (1.0 - 0.10 * (-((x - t1) * (x - t1)) / (2.0 * 1.5 * 1.5)).exp())
                        * (1.0 - 0.06 * (-((x - t2) * (x - t2)) / (2.0 * 1.5 * 1.5)).exp())
                };
                let gain = row_gain[y] * transparency[t] * full_scale;

                let row = &mut frame[y * p.spec_w..(y + 1) * p.spec_w];
                if icont > 1e-4 {
                    // absorption profile whose core intensity equals icore
                    let depth = (1.0 - (icore * (1.0 - line_depth_quiet).max(0.02) / icont).min(1.0))
                        .max(0.0)
                        .min(0.995);
                    for (x, px) in row.iter_mut().enumerate() {
                        let dx = x as f64 - line_center;
                        let prof = 1.0 - depth * (-(dx * dx) / (2.0 * sigma_line * sigma_line)).exp();
                        let signal = (icont * prof * gain * tell(x as f64)).max(0.0);
                        let noise = if p.clean {
                            0.0
                        } else {
                            (signal.max(0.0) / photon_gain).sqrt() * n01.sample(&mut rng)
                                + read_noise * n01.sample(&mut rng)
                        };
                        let noisy = signal + noise + 200.0; // bias pedestal
                        *px = noisy.clamp(0.0, 65535.0) as u16;
                    }
                } else {
                    // off-disk: emission line (prominences) + sky background
                    for (x, px) in row.iter_mut().enumerate() {
                        let dx = x as f64 - line_center;
                        let em = icore * (1.0 - line_depth_quiet).max(0.02) * (-(dx * dx) / (2.0 * sigma_line * sigma_line)).exp();
                        let sky = 0.004; // scattered light
                        let signal = ((em + sky) * gain * tell(x as f64)).max(0.0);
                        let noise = if p.clean {
                            0.0
                        } else {
                            (signal.max(0.0) / photon_gain).sqrt() * n01.sample(&mut rng)
                                + read_noise * n01.sample(&mut rng)
                        };
                        let noisy = signal + noise + 200.0;
                        *px = noisy.clamp(0.0, 65535.0) as u16;
                    }
                }
            }
            frames.push(frame);
        }

        let ser_path = if scan == 0 {
            out_ser.to_path_buf()
        } else {
            let dir = out_ser.parent().unwrap_or(Path::new("."));
            dir.join(format!("synth_scan{scan}.ser"))
        };
        write_ser(&ser_path, p.spec_w, p.slit_h, &frames)?;

        if scan == 0 {
            truth_out = Some(SynthTruth {
                jitter,
                transparency,
                row_gain,
                smile: [c0, c1, c2],
                flex,
                jitter_x: jitter_x.clone(),
                burst_mask: burst_mask.clone(),
                psf_seeing_px: p.psf_seeing_px,
                n_scans: p.n_scans.max(1),
            });
        }
    }

    // ground truth images (scan-0 sun, no evolution)
    let model = SunModel::new(p.radius, p.seed);
    let gt_size = (2.0 * p.radius + 160.0) as usize;
    let identity = psf_stencil(0.0, 0.0);
    let gt = model.render_ground_truth(gt_size, &identity);
    crate::output::write_png16(out_truth_png, &gt, Some((0.0, gt.max())))?;
    let dir = out_truth_png.parent().unwrap_or(Path::new("."));
    if p.psf_seeing_px > 0.0 {
        let gtb = model.render_ground_truth(gt_size, &stencil);
        crate::output::write_png16(&dir.join("ground_truth_blurred.png"), &gtb, Some((0.0, gtb.max())))?;
    }
    if p.doppler {
        let gtv = model.render_ground_truth_velocity(gt_size, p.dispersion_kms_per_px);
        let mut enc = Image::new(gtv.w, gtv.h);
        for i in 0..gtv.data.len() {
            enc.data[i] = ((gtv.data[i] as f64 / VEL_SCALE + 1.0) / 2.0 * 65535.0).clamp(0.0, 65535.0) as f32;
        }
        crate::output::write_png16(&dir.join("ground_truth_velocity.png"), &enc, Some((0.0, 65535.0)))?;
    }

    Ok(truth_out.unwrap())
}
