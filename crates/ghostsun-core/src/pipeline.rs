//! End-to-end reconstruction orchestration.

use crate::deconv;
use crate::denoise;
use crate::ellipse;
use crate::extract::{reconstruct_disk, ExtractOptions, SpectralKernel};
use crate::flatfield;
use crate::image2d::Image;
use crate::jitter;
use crate::limb;
use crate::linefit;
use crate::mathutil::{percentile_f32, polyval, savgol_quadratic};
use crate::metrics::DiskFit;
use crate::profile::{self, ProfileTune};
use crate::quality;
use crate::ser::SerReader;
use crate::warp::{warp_baseline, warp_single, WarpOutput, WarpParams};
use std::path::Path;

/// All tunable magic numbers, sweepable via `bench --sweep name=v1,v2,...`.
#[derive(Clone)]
pub struct TuneParams {
    pub w_fit: f64,           // profile fit half-window (px)
    pub pca_k: f64,           // residual PCA components
    pub mu_range: f64,        // mu search range (px)
    pub depth_gate: f64,      // absorption/emission fallback gate
    pub transp_deadband: f64, // transparency gain deadband
    pub transv_deadband: f64, // transversalium gain deadband
    pub jitter_hp: f64,       // jitter high-pass window (frames)
    pub burst_thresh: f64,    // burst detection threshold (x local floor)
    pub nlm_radius: f64,      // temporal NLM neighbor radius (frames)
    pub nlm_h: f64,           // temporal NLM strength (x noise sigma)
    pub column_demix_strength: f64, // residual column-state correction (0..1)
    pub rl_iters: f64,        // Richardson-Lucy iterations
    pub rl_tv: f64,           // RL total-variation lambda
    pub rl_floor: f64,        // intrinsic limb-width floor (px)
    pub denoise_k: f64,       // wavelet soft-threshold multiple
}

impl Default for TuneParams {
    fn default() -> Self {
        TuneParams {
            w_fit: 8.0,
            pca_k: 3.0,
            mu_range: 3.0,
            depth_gate: 0.10,
            transp_deadband: 0.012,
            transv_deadband: 0.004,
            jitter_hp: 41.0,
            burst_thresh: 1.3,
            nlm_radius: 3.0,
            nlm_h: 1.8,
            column_demix_strength: 1.0,
            rl_iters: 15.0,
            rl_tv: 0.01,
            rl_floor: 1.2,
            denoise_k: 1.0,
        }
    }
}

impl TuneParams {
    pub fn set(&mut self, name: &str, v: f64) -> Result<(), String> {
        match name {
            "w_fit" => self.w_fit = v,
            "pca_k" => self.pca_k = v,
            "mu_range" => self.mu_range = v,
            "depth_gate" => self.depth_gate = v,
            "transp_deadband" => self.transp_deadband = v,
            "transv_deadband" => self.transv_deadband = v,
            "jitter_hp" => self.jitter_hp = v,
            "burst_thresh" => self.burst_thresh = v,
            "nlm_radius" => self.nlm_radius = v,
            "nlm_h" => self.nlm_h = v,
            "column_demix_strength" => self.column_demix_strength = v,
            "rl_iters" => self.rl_iters = v,
            "rl_tv" => self.rl_tv = v,
            "rl_floor" => self.rl_floor = v,
            "denoise_k" => self.denoise_k = v,
            _ => return Err(format!("unknown tune param: {name}")),
        }
        Ok(())
    }
}

pub struct ReconOptions {
    pub baseline: bool,
    pub shift: f64,
    pub window_sigma: f64,
    pub rotation_deg: f64,
    pub flip_x: bool,
    pub flip_y: bool,
    pub margin_frac: f64,
    pub jitter_correction: bool,
    pub jitter_fast: bool,
    pub jitter_drift: bool,
    pub transparency_correction: bool,
    pub transversalium_correction: bool,
    /// F1: profile-model extraction (false = plain B-spline sampling)
    pub profile_extraction: bool,
    /// F4: footprint-filtered downscaling warp
    pub filtered_warp: bool,
    /// F6: PSF estimation + Richardson-Lucy deconvolution
    pub deconv: bool,
    /// F7: variance-stabilized wavelet denoising
    pub denoise: bool,
    /// F9.2: per-frame registration along the scan direction
    pub x_registration: bool,
    /// F11: temporal burst detection and repair
    pub burst_repair: bool,
    /// F11.5: temporal non-local-means smoothing
    pub temporal_nlm: bool,
    /// M2: use wgpu compute kernels where available (CPU fallback)
    pub use_gpu: bool,
    /// F8: extra block-coordinate refinement iterations (0 = single pass)
    pub map_iterations: usize,
    pub tune: TuneParams,
    pub verbose: bool,
    /// optional log/progress sink (UI); when set, vlog! goes here too
    pub progress: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,
}

impl Default for ReconOptions {
    fn default() -> Self {
        ReconOptions {
            baseline: false,
            shift: 0.0,
            window_sigma: 0.0,
            rotation_deg: 0.0,
            flip_x: false,
            flip_y: false,
            margin_frac: 0.15,
            jitter_correction: true,
            jitter_fast: true,
            jitter_drift: true,
            transparency_correction: true,
            transversalium_correction: true,
            profile_extraction: true,
            filtered_warp: true,
            deconv: false,
            denoise: false,
            x_registration: true,
            burst_repair: true,
            temporal_nlm: true,
            use_gpu: true,
            map_iterations: 0,
            tune: TuneParams::default(),
            verbose: true,
            progress: None,
        }
    }
}

#[allow(dead_code)]
pub struct ReconReport {
    pub output: WarpOutput,
    /// Final-view comparison image immediately before column-state demixing,
    /// passed through the same NLM and geometric warp as `output`.
    pub demix_before: Option<Image>,
    pub raw_disk: Image,
    /// F2: warped line-core velocity map (px), when profile extraction is on
    pub velocity: Option<Image>,
    /// Wing-difference Dopplergram (R-B)/(R+B) at +-wing offset — the
    /// INTI-style, bisector-depth Doppler product (rotation-sensitive)
    pub wing_doppler: Option<Image>,
    /// F3: estimated per-frame flexure (px)
    pub flex: Vec<f64>,
    /// F6: fitted PSF (sigma_x, sigma_y) when deconvolution ran
    pub psf_sigma: Option<(f64, f64)>,
    pub line_rms: f64,
    pub ellipse_inliers: (usize, usize),
    pub ellipse_rms: f64,
    pub sx: f64,
    pub shear: f64,
    pub radius: f64,
    /// per-column vertical shift applied (jitter + drift), px
    pub jitter_applied: Vec<f64>,
    /// F9.2: per-column scan-direction offset removed (frames)
    pub xreg_applied: Vec<f64>,
    /// F11: burst-flagged columns
    pub burst_flags: Vec<bool>,
    /// per-column photometric gain divided out
    pub column_gain: Vec<f64>,
}

macro_rules! vlog {
    ($opts:expr, $($arg:tt)*) => {
        {
            let msg = format!($($arg)*);
            if let Some(cb) = &$opts.progress { cb(&msg); }
            if $opts.verbose { println!("{}", msg); }
        }
    };
}

pub fn reconstruct(ser_path: &Path, opts: &ReconOptions) -> Result<ReconReport, String> {
    let t_start = std::time::Instant::now();
    let mut t_last = t_start;
    macro_rules! stage {
        ($name:expr) => {
            {
                let now = std::time::Instant::now();
                vlog!(opts, "[t] {}: {:.2}s", $name, (now - t_last).as_secs_f64());
                #[allow(unused_assignments)]
                { t_last = now; }
            }
        };
    }
    let reader = SerReader::open(ser_path).map_err(|e| format!("SER open: {e}"))?;
    let hdr = &reader.header;
    vlog!(opts, "SER: {}x{} x{} frames, {} bit", hdr.width, hdr.height, hdr.frame_count, hdr.bit_depth);

    // orientation: slit must be vertical (dispersion horizontal)
    let transpose = hdr.width > hdr.height;

    // ---- mean image over frames with signal (rayon: this was the single
    // largest stage at ~11 s on a 9100-frame scan when serial) ----
    use rayon::prelude::*;
    let n = hdr.frame_count;
    let frame_means: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|t| {
            let f = reader.frame(t);
            let mut s = 0.0;
            let mut c = 0.0;
            let mut y = 0;
            while y < f.h {
                for &v in f.row(y) {
                    s += v as f64;
                    c += 1.0;
                }
                y += 4;
            }
            s / c
        })
        .collect();
    let mut sorted = frame_means.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = sorted[(sorted.len() as f64 * 0.9) as usize];
    let good_thresh = p90 * 0.4;
    let good: Vec<usize> = (0..n).filter(|&t| frame_means[t] > good_thresh).collect();
    let use_frames: &[usize] = if good.len() > 50 { &good } else { &(0..n).collect::<Vec<_>>() };
    let mean_img = {
        // parallel partial sums over frame chunks, then reduce
        let partials: Vec<(Vec<f64>, usize, usize)> = use_frames
            .par_chunks(256)
            .map(|chunk| {
                let mut acc: Option<Vec<f64>> = None;
                let mut w = 0;
                let mut h = 0;
                for &t in chunk {
                    let mut f = reader.frame(t);
                    if transpose {
                        f = f.transpose();
                    }
                    w = f.w;
                    h = f.h;
                    let a = acc.get_or_insert_with(|| vec![0.0; w * h]);
                    for (i, &v) in f.data.iter().enumerate() {
                        a[i] += v as f64;
                    }
                }
                (acc.unwrap_or_default(), w, h)
            })
            .collect();
        let (w, h) = partials
            .iter()
            .find(|p| p.1 > 0)
            .map(|p| (p.1, p.2))
            .ok_or("no frames")?;
        let mut total = vec![0.0f64; w * h];
        for (a, _, _) in &partials {
            for (i, v) in a.iter().enumerate() {
                total[i] += v;
            }
        }
        let mut m = Image::new(w, h);
        let cnt = use_frames.len() as f64;
        for (i, v) in total.iter().enumerate() {
            m.data[i] = (v / cnt) as f32;
        }
        m
    };
    vlog!(opts, "mean image from {}/{} frames", use_frames.len(), n);
    stage!("mean image");

    // ---- spectral line geometry ----
    let geom = if opts.baseline {
        linefit::fit_line_geometry_baseline(&mean_img)
    } else {
        linefit::fit_line_geometry(&mean_img, 2)
    }
    .ok_or("line geometry fit failed")?;
    vlog!(
        opts,
        "line poly: {:?} (rms {:.3} px, {} rows)",
        geom.coeffs.iter().map(|c| format!("{c:.4e}")).collect::<Vec<_>>(),
        geom.rms,
        geom.n_rows_used
    );

    // ---- extraction (F1 profile model or B-spline / baseline) ----
    let slit_h = if transpose { hdr.width } else { hdr.height };
    let smile: Vec<f64> = (0..slit_h).map(|y| polyval(&geom.coeffs, y as f64)).collect();

    let ptune = ProfileTune {
        w_fit: opts.tune.w_fit.round().max(4.0) as usize,
        pca_k: opts.tune.pca_k.round().max(0.0) as usize,
        mu_range: opts.tune.mu_range,
        depth_gate: opts.tune.depth_gate,
    };
    let use_profile = opts.profile_extraction && !opts.baseline;
    // Continuum offset for the transparency reference.  Compute this before
    // extraction so the profile spectrum can supply the same bin directly.
    let continuum_shift = {
        let iw = mean_img.w as f64;
        let ymid = (geom.y1 + geom.y2) / 2;
        let cx = polyval(&geom.coeffs, ymid as f64);
        let room_left = cx - 6.0;
        let room_right = iw - 7.0 - cx;
        let mag = 25.0f64.min(room_left.max(room_right).max(8.0));
        if room_right >= room_left { mag.min(room_right) } else { -mag.min(room_left) }
    };

    // The profile extractor already produces a de-smiled per-frame spectrum.
    // Retain its continuum bin so transparency correction does not need a
    // second full extraction of every SER frame.
    let mut profile_continuum_flux: Option<Vec<f64>> = None;
    let (mut disk, flex, mut velocity_raw): (Image, Vec<f64>, Option<Image>) = if use_profile {
        let (maps, on_gpu) =
            profile::extract_profile_auto(&reader, &geom, &mean_img, transpose, opts.shift, &ptune, opts.use_gpu);
        vlog!(opts, "extraction [{}]", if on_gpu { "gpu" } else { "cpu" });
        // The optimized GPU spectrum is disk-gated.  The CPU reference keeps
        // its historic all-lit-row spectrum, so retain the legacy continuum
        // extraction when GPU profile extraction is unavailable.
        if on_gpu && !maps.spec_offsets.is_empty() {
            if let Some((ki, _)) = maps
                .spec_offsets
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    ((*a - continuum_shift).abs()).total_cmp(&((*b - continuum_shift).abs()))
                })
            {
                profile_continuum_flux = Some(
                    maps.frame_spec
                        .iter()
                        .map(|s| s.get(ki).copied().unwrap_or(0.0) as f64)
                        .collect(),
                );
            }
        }
        // F3: flexure. Preferred source: telluric anchor lines (absolute,
        // immune to solar Doppler — keeps the full trend including the part
        // degenerate with rotation). Fallback: solar-line estimator
        // (nonlinear component only).
        let mut v_row: Option<Vec<f64>> = None;
        let flex = match profile::estimate_flexure_telluric(&maps, &smile, 3.0) {
            Some(tf) => {
                vlog!(
                    opts,
                    "flexure: telluric-anchored, {} line(s) at offsets {:?} px (dispersion {:.4} A/px)",
                    tf.n_lines,
                    tf.line_offsets.iter().map(|o| *o as i64).collect::<Vec<_>>(),
                    tf.dispersion
                );
                if std::env::var("GS_NO_VROW").is_err() {
                    v_row = profile::slit_velocity_from_telluric(
                        &mean_img, &smile, geom.y1, geom.y2, &tf.line_offsets,
                    );
                    if let Some(vr) = &v_row {
                        let amp = vr.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
                        vlog!(opts, "slit-velocity (telluric-referenced smile): +-{:.2} px", amp);
                    }
                }
                tf.flex
            }
            None => {
                vlog!(opts, "flexure: no telluric anchors, solar-line fallback");
                profile::estimate_flexure(&maps, &smile)
            }
        };
        let fmax = flex.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
        vlog!(opts, "flexure: max {:.3} px", fmax);
        // F2: velocity map (raw disk coords)
        let vel = profile::velocity_map(&maps, &smile, &flex, v_row.as_deref());
        (maps.core, flex, Some(vel))
    } else {
        let exopts = ExtractOptions {
            shift: opts.shift,
            transpose_input: transpose,
            kernel: if opts.baseline {
                SpectralKernel::LocalPolynomial
            } else if opts.window_sigma > 0.0 {
                SpectralKernel::Gaussian {
                    sigma: opts.window_sigma,
                }
            } else {
                SpectralKernel::Point
            },
            frame_offsets: None,
        };
        (reconstruct_disk(&reader, &geom, &exopts), vec![0.0; n], None)
    };
    let raw_disk = disk.clone();
    vlog!(opts, "raw disk: {}x{}", disk.w, disk.h);
    stage!("extraction");

    // ---- photometric & registration corrections ----
    let mut jitter_applied = vec![0.0f64; disk.w];
    let mut xreg_applied = vec![0.0f64; disk.w];
    let mut burst_flags = vec![false; disk.w];
    let mut column_gain = vec![1.0f64; disk.w];
    let mut demix_before_raw: Option<Image> = None;
    if opts.baseline {
        correct_transversalium_baseline(&mut disk);
    } else {
        if opts.transparency_correction {
            let fluxv = if let Some(flux) = profile_continuum_flux.take() {
                flux
            } else {
                let cont_opts = ExtractOptions {
                    shift: opts.shift + continuum_shift,
                    transpose_input: transpose,
                    kernel: SpectralKernel::Gaussian { sigma: 1.5 },
                    frame_offsets: if flex.iter().any(|f| f.abs() > 0.0) { Some(flex.clone()) } else { None },
                };
                let cont_disk = reconstruct_disk(&reader, &geom, &cont_opts);
                flatfield::measure_column_flux(&cont_disk)
            };
            column_gain = flatfield::transparency_gains(&fluxv, opts.tune.transp_deadband);
            flatfield::apply_column_gains(&mut disk, &column_gain);
            let worst = column_gain.iter().cloned().fold(1.0f64, |m, v| if (v - 1.0).abs() > (m - 1.0).abs() { v } else { m });
            vlog!(opts, "transparency (continuum dp {:+.0}): worst gain {:.3}", continuum_shift, worst);
            stage!("transparency");
        }
        // F8: block-coordinate refinement — registration and gain blocks are
        // re-estimated on the already-corrected disk; corrections compose.
        let passes = 1 + opts.map_iterations;
        for pass in 0..passes {
            if opts.jitter_correction {
                if opts.jitter_fast {
                    let jr = jitter::correct_jitter(&disk, opts.tune.jitter_hp.round() as usize);
                    let max_c = jr.trajectory.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
                    vlog!(opts, "jitter[{pass}]: max fast correction {:.2} px", max_c);
                    disk = jr.corrected;
                    if let Some(v) = velocity_raw.as_mut() {
                        *v = jitter::apply_shifts(v, &jr.trajectory);
                    }
                    for x in 0..disk.w {
                        jitter_applied[x] += jr.trajectory[x];
                    }
                }
                if opts.jitter_drift {
                    let dr = jitter::correct_drift(&disk);
                    let max_d = dr.trajectory.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
                    vlog!(opts, "drift[{pass}]: max midchord correction {:.2} px", max_d);
                    disk = dr.corrected;
                    if let Some(v) = velocity_raw.as_mut() {
                        *v = jitter::apply_shifts(v, &dr.trajectory);
                    }
                    for x in 0..disk.w {
                        jitter_applied[x] += dr.trajectory[x];
                    }
                }
                stage!("jitter+drift");
            }
            if opts.x_registration && pass == 0 {
                let xr = jitter::correct_x(&disk, opts.tune.jitter_hp.round() as usize);
                let max_x = xr.delta.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
                vlog!(opts, "x-registration: max offset {:.2} frames", max_x);
                disk = xr.corrected;
                if let Some(v) = velocity_raw.as_mut() {
                    *v = jitter::apply_x_offsets(v, &xr.delta);
                }
                xreg_applied = xr.delta;
                stage!("x-registration");
            }
            if opts.burst_repair && pass == 0 {
                let rep = {
                    let mut comps: Vec<&mut Image> = Vec::new();
                    if let Some(v) = velocity_raw.as_mut() {
                        comps.push(v);
                    }
                    quality::repair_bursts(&mut disk, &mut comps, opts.tune.burst_thresh)
                };
                vlog!(opts, "burst repair: {} column(s) repaired", rep.n_flagged);
                burst_flags = rep.flags;
                stage!("burst repair");
            }
            if opts.x_registration && pass == 0 {
                // F9.4: photometric x-anchors — at the disk entry/exit ramps
                // the chord flux is a steep invertible function of true scan
                // position; displaced frames read dark/bright (the vertical
                // bands seen on real data near the left/right limb)
                // F9.3: limb-anchored x-offsets (covers the tangent columns
                // where the texture x-registration is gated off).
                // NOTE: a photometric-x variant (flux-ramp inversion) was
                // tried and removed: transparency residuals alias into it
                // (jitter::photometric_x_offsets kept for reference).
                let pts0 = limb::detect_limb_points(&disk);
                if let Some(fit0) = ellipse::fit_robust(&pts0, 4242) {
                    let dl = jitter::limb_x_offsets(&pts0, &fit0.conic, disk.w);
                    let max_l = dl.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
                    let n_act = dl.iter().filter(|v| v.abs() > 0.0).count();
                    vlog!(opts, "limb-x anchors: {} column(s), max {:.2} frames", n_act, max_l);
                    if n_act > 0 {
                        disk = jitter::apply_x_offsets(&disk, &dl);
                        if let Some(v) = velocity_raw.as_mut() {
                            *v = jitter::apply_x_offsets(v, &dl);
                        }
                        for x in 0..disk.w {
                            xreg_applied[x] += dl[x];
                        }
                    }
                }
            }
            if opts.transversalium_correction {
                flatfield::correct_transversalium(&mut disk, opts.tune.transv_deadband);
                vlog!(opts, "transversalium corrected [{pass}]");
            }
        }
        // Joint residual self-calibration: after the explicit physical
        // corrections, fit the remaining column-coherent signal to gain,
        // additive bias, scan/slit displacement and blur modes. The GPU kernel uses the
        // paired solar limbs as a high-weight round-disk constraint.
        if opts.use_gpu
            && opts.tune.column_demix_strength > 0.0
            && std::env::var("GS_NO_COLUMN_DEMIX").is_err()
        {
            demix_before_raw = Some(disk.clone());
            if let Some((out, state)) = crate::gpu::demix_columns(
                &disk,
                opts.tune.column_demix_strength as f32,
            ) {
                let max_abs = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let active = state.gain.iter().filter(|g| g.abs() > 0.002).count();
                vlog!(
                    opts,
                    "column-state [gpu x{:.2}]: gain +-{:.2}% ({} active), offset +-{:.2}%, dx +-{:.3}, dy +-{:.3}, blur +-{:.3}",
                    opts.tune.column_demix_strength,
                    100.0 * max_abs(&state.gain),
                    active,
                    100.0 * max_abs(&state.offset),
                    max_abs(&state.x_shift),
                    max_abs(&state.y_shift),
                    max_abs(&state.blur),
                );
                disk = out;
                stage!("column-state demix");
            } else {
                demix_before_raw = None;
            }
        }
        if opts.temporal_nlm {
            stage!("pre-NLM stages");
            let radius = opts.tune.nlm_radius.round().max(1.0) as usize;
            let mut done_gpu = false;
            if opts.use_gpu {
                if let Some((sigma, h2, thresh)) = quality::nlm_params(&disk, opts.tune.nlm_h) {
                    if let Some(out) = crate::gpu::temporal_nlm(&disk, radius, h2, sigma, thresh) {
                        disk = out;
                        done_gpu = true;
                    }
                }
            }
            if !done_gpu {
                disk = quality::temporal_nlm(&disk, radius, opts.tune.nlm_h);
            }
            vlog!(
                opts,
                "temporal NLM [{}]: radius {} h {:.2}",
                if done_gpu { "gpu" } else { "cpu" },
                opts.tune.nlm_radius,
                opts.tune.nlm_h
            );
            stage!("temporal NLM");
        }
    }

    // ---- geometry: limb -> ellipse -> single warp ----
    let pts = if opts.baseline {
        limb::detect_limb_points_baseline(&disk)
    } else {
        limb::detect_limb_points(&disk)
    };
    vlog!(opts, "limb points: {}", pts.len());
    let fit = if opts.baseline {
        let conic = ellipse::fit_direct(&pts).ok_or("baseline ellipse fit failed")?;
        let geom2 = conic.geometry().ok_or("baseline conic not an ellipse")?;
        ellipse::RansacResult { conic, geom: geom2, inliers: pts.len(), total: pts.len(), residual_rms: 0.0 }
    } else {
        ellipse::fit_robust(&pts, 1234).ok_or("robust ellipse fit failed")?
    };
    vlog!(
        opts,
        "ellipse: center ({:.1},{:.1}) sx {:.4} shear {:.5} radius {:.1} (inliers {}/{}, rms {:.2})",
        fit.geom.xc, fit.geom.yc, fit.geom.sx, fit.geom.shear, fit.geom.radius,
        fit.inliers, fit.total, fit.residual_rms
    );

    let wp = WarpParams {
        rotation_deg: opts.rotation_deg,
        flip_x: opts.flip_x,
        flip_y: opts.flip_y,
        margin_frac: opts.margin_frac,
        filtered_downscale: opts.filtered_warp && !opts.baseline,
        allow_negative: false,
    };
    let mut output = if opts.baseline {
        warp_baseline(&disk, &fit.geom, &wp)
    } else if opts.use_gpu {
        match crate::gpu::warp_single(&disk, &fit.geom, &wp) {
            Some(o) => {
                vlog!(opts, "warp [gpu]");
                o
            }
            None => warp_single(&disk, &fit.geom, &wp),
        }
    } else {
        warp_single(&disk, &fit.geom, &wp)
    };
    // Real-time A/B viewer image. Apply the same NLM and the exact fitted
    // geometry used by the demixed result, so toggling isolates the new
    // column-state correction rather than changing registration or scale.
    let demix_before = demix_before_raw.map(|mut before| {
        if opts.temporal_nlm {
            let radius = opts.tune.nlm_radius.round().max(1.0) as usize;
            if let Some((sigma, h2, thresh)) = quality::nlm_params(&before, opts.tune.nlm_h) {
                before = crate::gpu::temporal_nlm(&before, radius, h2, sigma, thresh)
                    .unwrap_or_else(|| quality::temporal_nlm(&before, radius, opts.tune.nlm_h));
            }
        }
        if opts.use_gpu {
            crate::gpu::warp_single(&before, &fit.geom, &wp)
                .unwrap_or_else(|| warp_single(&before, &fit.geom, &wp))
                .image
        } else {
            warp_single(&before, &fit.geom, &wp).image
        }
    });
    vlog!(opts, "output: {}x{}", output.image.w, output.image.h);
    stage!("limb+ellipse+warp");

    // Wing-difference Dopplergram: intensities at +-wing_offset from the
    // (flexure-corrected) line center; the normalized difference cancels
    // column gains and transversalium and is maximally shift-sensitive
    // (wing slope), measuring at bisector depths where rotation is clean.
    let wing_doppler: Option<Image> = if use_profile && !opts.baseline {
        let wing = std::env::var("GS_WING_OFFSET")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(6.0);
        let offsets: Option<Vec<f64>> = if std::env::var("GS_WING_NOFLEX").is_ok() {
            None // INTI-condition: wings extracted without flexure correction
        } else if flex.iter().any(|f| f.abs() > 0.0) {
            Some(flex.clone())
        } else {
            None
        };
        let mk = |sh: f64| {
            reconstruct_disk(&reader, &geom, &ExtractOptions {
                shift: opts.shift + sh,
                transpose_input: transpose,
                kernel: SpectralKernel::Gaussian { sigma: 1.0 },
                frame_offsets: offsets.clone(),
            })
        };
        let blue = mk(-wing);
        let red = mk(wing);
        let ithresh = crate::mathutil::percentile_f32(&blue.data, 80.0) * 0.25;
        let mut wd = Image::new(blue.w, blue.h);
        for i in 0..wd.data.len() {
            let (b, r) = (blue.data[i] as f64, red.data[i] as f64);
            if b + r > 2.0 * ithresh as f64 {
                wd.data[i] = ((r - b) / (r + b)) as f32;
            }
        }
        // align with the corrected core image
        let mut wd = jitter::apply_shifts(&wd, &jitter_applied);
        wd = jitter::apply_x_offsets(&wd, &xreg_applied);
        vlog!(opts, "wing Dopplergram at +-{:.0} px", wing);
        Some(wd)
    } else {
        None
    };
    let wing_doppler = wing_doppler.map(|wd| {
        let wp_v = WarpParams { filtered_downscale: false, allow_negative: true, ..wp };
        if opts.use_gpu {
            crate::gpu::warp_single(&wd, &fit.geom, &wp_v)
                .unwrap_or_else(|| warp_single(&wd, &fit.geom, &wp_v))
                .image
        } else {
            warp_single(&wd, &fit.geom, &wp_v).image
        }
    });

    // warp the velocity map with identical geometry (unfiltered kernel: the
    // map is already smoothed and NaN-free)
    let velocity = velocity_raw.map(|v| {
        let wp_v = WarpParams { filtered_downscale: false, allow_negative: true, ..wp };
        if opts.use_gpu {
            crate::gpu::warp_single(&v, &fit.geom, &wp_v)
                .unwrap_or_else(|| warp_single(&v, &fit.geom, &wp_v))
                .image
        } else {
            warp_single(&v, &fit.geom, &wp_v).image
        }
    });

    let disk_fit = DiskFit { xc: output.xc, yc: output.yc, r: output.radius };

    // ---- F6: deconvolution ----
    let mut psf_sigma = None;
    if opts.deconv && !opts.baseline {
        match deconv::deconvolve(
            &output.image,
            &disk_fit,
            opts.tune.rl_iters.round().max(1.0) as usize,
            opts.tune.rl_tv,
            opts.tune.rl_floor,
        ) {
            Some((img, sig)) => {
                vlog!(opts, "deconv: PSF sigma ({:.2}, {:.2}) px, {} RL iters", sig.0, sig.1, opts.tune.rl_iters);
                output.image = img;
                psf_sigma = Some(sig);
            }
            None => vlog!(opts, "deconv: PSF below threshold, skipped"),
        }
    }

    // ---- F7: denoising ----
    if opts.denoise && !opts.baseline {
        output.image = denoise::denoise(&output.image, &disk_fit, opts.tune.denoise_k);
        vlog!(opts, "denoise: wavelet shrinkage k={:.2}", opts.tune.denoise_k);
    }

    vlog!(opts, "[t] TOTAL: {:.2}s", t_start.elapsed().as_secs_f64());
    Ok(ReconReport {
        output,
        demix_before,
        raw_disk,
        velocity,
        wing_doppler,
        flex,
        psf_sigma,
        line_rms: geom.rms,
        ellipse_inliers: (fit.inliers, fit.total),
        ellipse_rms: fit.residual_rms,
        sx: fit.geom.sx,
        shear: fit.geom.shear,
        radius: fit.geom.radius,
        jitter_applied,
        xreg_applied,
        burst_flags,
        column_gain,
    })
}

/// INTI-style transversalium: mean row profile over threshold pixels,
/// Savitzky-Golay smoothing, straight division, uint16 rounding.
fn correct_transversalium_baseline(disk: &mut Image) {
    let h = disk.h;
    let seuil_haut = percentile_f32(&disk.data, 90.0);
    let myseuil = seuil_haut * 0.5;

    let mut y1 = h;
    let mut y2 = 0;
    for y in 0..h {
        if disk.row(y).iter().any(|&v| v > myseuil) {
            if y < y1 {
                y1 = y;
            }
            y2 = y;
        }
    }
    if y2 <= y1 + 20 {
        return;
    }
    let w1 = {
        let mut v = (y2 - y1) / 4;
        if v % 2 == 0 {
            v += 1;
        }
        v
    };
    let w2 = {
        let mut v = (w1 as f64 * 0.3) as usize;
        if v % 2 == 0 {
            v += 1;
        }
        v.max(5)
    };
    for win in [w2, w1] {
        let mut prof = vec![0.0f64; h];
        for y in 0..h {
            let sel: Vec<f64> = disk.row(y).iter().filter(|&&v| v > myseuil).map(|&v| v as f64).collect();
            prof[y] = if sel.is_empty() {
                myseuil as f64
            } else {
                let mut s = sel.clone();
                crate::mathutil::median_inplace(&mut s)
            };
        }
        let seg: Vec<f64> = prof[y1..=y2].to_vec();
        let sm = savgol_quadratic(&seg, win);
        for (i, y) in (y1..=y2).enumerate() {
            let hf = if sm[i] > 1e-6 { seg[i] / sm[i] } else { 1.0 };
            if hf.abs() > 1e-9 {
                for x in 0..disk.w {
                    let v = disk.at(x, y) as f64 / hf;
                    disk.set(x, y, v.clamp(0.0, 65535.0).round() as f32);
                }
            }
        }
    }
}

/// De-smiled mean spectrum: average of all rows of the mean image, each
/// resampled on a grid of offsets relative to its fitted line center.
/// Offsets are relative to the line core (0 = core).
pub fn mean_spectrum(ser_path: &Path) -> Result<(Vec<i64>, Vec<f64>), String> {
    let reader = SerReader::open(ser_path).map_err(|e| format!("SER open: {e}"))?;
    let hdr = &reader.header;
    let transpose = hdr.width > hdr.height;
    let n = hdr.frame_count;
    // mean image over a subsample of frames with signal
    let step = (n / 600).max(1);
    let mut acc: Option<Vec<f64>> = None;
    let mut w = 0;
    let mut h = 0;
    let mut cnt = 0.0;
    let mut t = 0;
    while t < n {
        let mut f = reader.frame(t);
        if transpose {
            f = f.transpose();
        }
        let m: f64 = f.data.iter().map(|&v| v as f64).sum::<f64>() / f.data.len() as f64;
        w = f.w;
        h = f.h;
        if m > 500.0 {
            let a = acc.get_or_insert_with(|| vec![0.0; w * h]);
            for (i, &v) in f.data.iter().enumerate() {
                a[i] += v as f64;
            }
            cnt += 1.0;
        }
        t += step;
    }
    let a = acc.ok_or("no bright frames")?;
    let mut mean_img = Image::new(w, h);
    for (i, v) in a.iter().enumerate() {
        mean_img.data[i] = (v / cnt) as f32;
    }
    let geom = linefit::fit_line_geometry(&mean_img, 2).ok_or("line fit failed")?;
    // offsets covering the full window for all rows
    let mut cmin = f64::MAX;
    let mut cmax = f64::MIN;
    for y in geom.y1..=geom.y2 {
        let c = polyval(&geom.coeffs, y as f64);
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    let off_lo = (4.0 - cmin).ceil() as i64;
    let off_hi = (w as f64 - 5.0 - cmax).floor() as i64;
    let offsets: Vec<i64> = (off_lo..=off_hi).collect();
    let mut prof = vec![0.0f64; offsets.len()];
    let mut wsum = 0.0;
    let margin = ((geom.y2 - geom.y1) / 20).max(10);
    for y in geom.y1 + margin..geom.y2.saturating_sub(margin) {
        let mut coef: Vec<f64> = mean_img.row(y).iter().map(|&v| v as f64).collect();
        crate::mathutil::bspline_prefilter(&mut coef);
        let c = polyval(&geom.coeffs, y as f64);
        let rw = mean_img.row(y).iter().map(|&v| v as f64).sum::<f64>();
        for (k, &o) in offsets.iter().enumerate() {
            prof[k] += rw * crate::mathutil::bspline_eval(&coef, c + o as f64);
        }
        wsum += rw;
    }
    for v in prof.iter_mut() {
        *v /= wsum.max(1e-9);
    }
    Ok((offsets, prof))
}
