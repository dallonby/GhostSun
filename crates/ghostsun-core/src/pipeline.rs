//! End-to-end reconstruction orchestration.

use crate::deconv;
use crate::denoise;
use crate::ellipse;
use crate::extract::{self, reconstruct_disk, ExtractOptions, SpectralKernel};
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
use crate::stack;
use crate::warp::{warp_baseline, warp_single, WarpOutput, WarpParams};
use std::path::Path;

/// All tunable magic numbers, sweepable via `bench --sweep name=v1,v2,...`.
#[derive(Clone)]
pub struct TuneParams {
    pub w_fit: f64,           // profile fit half-window (px)
    pub pca_k: f64,           // residual PCA components
    pub kl_k: f64,            // F17 spectral-subspace rank (0 = off)
    /// F20 spectro-temporal NLM: h in units of noise sigma. 0 = off, and off
    /// is correct -- MEASURED AS A NO-OP, see quality::temporal_nlm_spectral
    /// for why. Retained so the finding is reproducible, not for use.
    pub nlm_spec: f64,
    pub w_kl: f64,            // F17 half-window (0 = auto from the smile)
    pub mu_range: f64,        // mu search range (px)
    pub depth_gate: f64,      // absorption/emission fallback gate
    pub transp_deadband: f64, // transparency gain deadband
    pub transv_deadband: f64, // transversalium gain deadband
    pub jitter_hp: f64,       // jitter high-pass window (frames)
    pub burst_thresh: f64,    // burst detection threshold (x local floor)
    pub nlm_radius: f64,      // temporal NLM neighbor radius (frames)
    pub nlm_h: f64,           // temporal NLM strength (x noise sigma)
    pub motion_strength: f64, // pre-extraction slit-motion correction (0..1.5)
    pub column_demix_strength: f64, // residual column-state correction (0..1)
    pub rl_iters: f64,        // Richardson-Lucy iterations
    pub rl_tv: f64,           // RL total-variation lambda
    pub rl_floor: f64,        // intrinsic limb-width floor (px)
    pub denoise_k: f64,       // wavelet soft-threshold multiple
    /// Dopplergram wing offset in px; 0 = choose it from the line profile.
    /// Exposed here so `bench --sweep wing_px=...` can scan it against truth.
    pub wing_px: f64,
    /// Wiener noise scaling: 1.0 is the classic filter, higher is more
    /// aggressive. Swept with `bench --sweep wiener_strength=...`.
    pub wiener_strength: f64,
}

impl Default for TuneParams {
    fn default() -> Self {
        TuneParams {
            w_fit: 8.0,
            pca_k: 3.0,
            kl_k: 3.0,
            w_kl: 0.0,
            nlm_spec: 0.0,
            mu_range: 3.0,
            depth_gate: 0.10,
            transp_deadband: 0.012,
            transv_deadband: 0.004,
            jitter_hp: 41.0,
            burst_thresh: 1.3,
            nlm_radius: 3.0,
            nlm_h: 1.8,
            motion_strength: 1.0,
            column_demix_strength: 1.0,
            rl_iters: 15.0,
            rl_tv: 0.01,
            rl_floor: 1.2,
            denoise_k: 1.0,
            wing_px: 0.0,
            wiener_strength: 4.0,
        }
    }
}

impl TuneParams {
    pub fn set(&mut self, name: &str, v: f64) -> Result<(), String> {
        match name {
            "w_fit" => self.w_fit = v,
            "pca_k" => self.pca_k = v,
            "kl_k" => self.kl_k = v,
            "nlm_spec" => self.nlm_spec = v,
            "w_kl" => self.w_kl = v,
            "mu_range" => self.mu_range = v,
            "depth_gate" => self.depth_gate = v,
            "transp_deadband" => self.transp_deadband = v,
            "transv_deadband" => self.transv_deadband = v,
            "jitter_hp" => self.jitter_hp = v,
            "burst_thresh" => self.burst_thresh = v,
            "nlm_radius" => self.nlm_radius = v,
            "nlm_h" => self.nlm_h = v,
            "motion_strength" => self.motion_strength = v,
            "column_demix_strength" => self.column_demix_strength = v,
            "rl_iters" => self.rl_iters = v,
            "rl_tv" => self.rl_tv = v,
            "rl_floor" => self.rl_floor = v,
            "denoise_k" => self.denoise_k = v,
            "wing_px" => self.wing_px = v,
            "wiener_strength" => self.wiener_strength = v,
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
    /// Use the SER's per-frame timestamps as the scan-axis coordinate when the
    /// file carries them (resamples onto a uniform time grid). No effect on
    /// files without a timestamp trailer.
    pub use_timing: bool,
    /// Spectral dispersion in A/px, when known from the optics. Supplies the
    /// scale the telluric anchors cannot establish on their own from a single
    /// line, and puts wavelength-domain settings on a physical footing.
    pub dispersion_a_per_px: Option<f64>,
    /// Dopplergram wing offset in Angstrom. `None` = choose it from the line
    /// profile (F14). Needs `dispersion_a_per_px` to convert to pixels.
    pub wing_offset_a: Option<f64>,
    /// Extra wavelength offsets (px from the line core) to emit alongside the
    /// primary product, all sharing the primary's geometry so they are
    /// pixel-for-pixel comparable.
    pub shift_series: Vec<f64>,
    /// F15: Wiener filtering against the image's own measured power spectrum.
    pub wiener: bool,
    /// M2: use wgpu compute kernels where available (CPU fallback)
    pub use_gpu: bool,
    /// F8: extra block-coordinate refinement iterations (0 = single pass)
    pub map_iterations: usize,
    pub tune: TuneParams,
    /// `None` = auto-detect from absorption contrast; `Some(true)` force
    /// transpose; `Some(false)` keep SER orientation (dispersion = width).
    pub force_transpose: Option<bool>,
    /// Opt-in multi-line composite: extract companion absorption lines on
    /// full-sensor / multi-line SERs, share primary registration + photometry,
    /// co-register and sharpness-weighted stack into the main disk product.
    /// Primary-only path is unchanged when false or when no companions exist.
    pub multi_line_composite: bool,
    /// Optional forced spectral column (px on the oriented mean frame) for
    /// the primary line. `None` = multi-candidate continuous AUTO tracking.
    pub line_center_x: Option<f64>,
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
            use_timing: true,
            dispersion_a_per_px: None,
            wing_offset_a: None,
            shift_series: Vec::new(),
            wiener: false,
            use_gpu: true,
            map_iterations: 0,
            tune: TuneParams::default(),
            force_transpose: None,
            multi_line_composite: false,
            line_center_x: None,
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
    /// F18: per-frame along-slit blur relative to the scan median (px^2).
    /// Positive = that frame was seeing-blurred worse than typical.
    pub frame_blur: Vec<f64>,
    /// F11: burst-flagged columns
    pub burst_flags: Vec<bool>,
    /// F14: extra wavelength products, `(offset px from core, image)`, on the
    /// primary's grid.
    pub shift_products: Vec<(f64, Image)>,
    /// F13: per-frame acquisition timing, when the SER carried timestamps.
    pub timing: Option<crate::timing::TimingSummary>,
    /// Grid columns interpolated across a dropped-frame gap rather than
    /// measured. Empty when no regrid happened.
    pub gap_columns: Vec<bool>,
    /// per-column photometric gain divided out
    pub column_gain: Vec<f64>,
    /// Multi-line composite: number of spectral lines stacked (primary +
    /// companions). `None` when the option was off or no companions found.
    pub composite_n_lines: Option<usize>,
    /// Spectral offsets (px) of composite lines relative to the primary core,
    /// primary first at 0.0 when composite ran.
    pub composite_line_offsets_px: Vec<f64>,
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

    // ---- F13: per-frame timing ----
    // Frame index is the scan-axis coordinate only if frames arrive on a
    // perfect clock. When the file carries timestamps we know where each
    // frame really belongs, and resample onto a uniform time grid after
    // extraction. Extraction itself stays frame-indexed, so every later
    // re-extraction (continuum, wings, companion lines) is regridded the same
    // way to keep all per-column vectors on one axis.
    let timing = if opts.use_timing && !opts.baseline {
        crate::timing::ScanTiming::from_reader(&reader)
    } else {
        None
    };
    let regrid_src: Option<Vec<f64>> = timing.as_ref().and_then(|t| {
        if t.worth_regridding() {
            Some(t.source_columns())
        } else {
            None
        }
    });
    if let Some(t) = &timing {
        vlog!(
            opts,
            "timing: {} [{}]",
            t.summary(),
            if regrid_src.is_some() { "regridding" } else { "uniform, no regrid" }
        );
    }
    let regrid = |img: Image| -> Image {
        match &regrid_src {
            Some(src) => crate::timing::regrid_columns(&img, src),
            None => img,
        }
    };

    // ---- mean image over frames with signal (rayon: this was the single
    // largest stage at ~11 s on a 9100-frame scan when serial) ----
    // Built in *native* SER orientation first so we can choose dispersion axis.
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
    let mean_native = {
        let partials: Vec<(Vec<f64>, usize, usize)> = use_frames
            .par_chunks(256)
            .map(|chunk| {
                let mut acc: Option<Vec<f64>> = None;
                let mut w = 0;
                let mut h = 0;
                for &t in chunk {
                    let f = reader.frame(t);
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

    // orientation: slit vertical, dispersion horizontal
    let (transpose, row_c, col_c) = match opts.force_transpose {
        Some(t) => (t, 0.0, 0.0),
        None => linefit::should_transpose_for_dispersion(&mean_native),
    };
    let mean_img = if transpose {
        mean_native.transpose()
    } else {
        mean_native
    };
    vlog!(
        opts,
        "orientation: {} (max-line-depth native {:.3} / transposed {:.3}; SER {}x{})",
        if transpose {
            "transposed → dispersion along width"
        } else {
            "native → dispersion along width"
        },
        row_c,
        col_c,
        hdr.width,
        hdr.height
    );

    // ---- spectral line geometry ----
    let (x_lit0, x_lit1) = linefit::detect_spectrum_cols(&mean_img);
    vlog!(
        opts,
        "lit spectral band: columns {}..{} (of {})",
        x_lit0,
        x_lit1,
        mean_img.w
    );
    let geom = if opts.baseline {
        linefit::fit_line_geometry_baseline(&mean_img)
    } else if let Some(seed) = opts.line_center_x {
        vlog!(opts, "line seed forced at {:.1} px", seed);
        linefit::fit_line_geometry_at(&mean_img, 2, seed)
    } else {
        linefit::fit_line_geometry(&mean_img, 2)
    }
    .ok_or("line geometry fit failed")?;
    let primary_mid = {
        let ymid = (geom.y1 + geom.y2) as f64 * 0.5;
        polyval(&geom.coeffs, ymid)
    };
    vlog!(
        opts,
        "line poly: {:?} (rms {:.3} px, {} rows, primary ~{:.1} px)",
        geom.coeffs.iter().map(|c| format!("{c:.4e}")).collect::<Vec<_>>(),
        geom.rms,
        geom.n_rows_used,
        primary_mid
    );

    // ---- extraction (F1 profile model or B-spline / baseline) ----
    let slit_h = if transpose { hdr.width } else { hdr.height };
    let smile: Vec<f64> = (0..slit_h).map(|y| polyval(&geom.coeffs, y as f64)).collect();

    let ptune = ProfileTune {
        w_fit: opts.tune.w_fit.round().max(4.0) as usize,
        pca_k: opts.tune.pca_k.round().max(0.0) as usize,
        kl_k: opts.tune.kl_k.round().max(0.0) as usize,
        w_kl: opts.tune.w_kl.round().max(0.0) as usize,
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

    // Solve slit-direction motion before fitting the spectral profile. The
    // preview touches only three continuum samples per detector row, then the
    // recovered sub-pixel trajectory is consumed directly by the extraction
    // kernel. This retains the raw spectrum instead of resampling an already
    // reconstructed line-core column.
    let motion_strength = opts.tune.motion_strength.clamp(0.0, 1.5);
    let pre_extraction_motion: Option<Vec<f64>> =
        if use_profile && opts.jitter_correction && motion_strength >= 1e-3 {
            let mut reference = extract::reconstruct_continuum_preview(
                &reader,
                &geom,
                transpose,
                continuum_shift,
            );
            let mut trajectory = vec![0.0f64; n];
            if opts.jitter_fast {
                let jr = jitter::correct_jitter(
                    &reference,
                    opts.tune.jitter_hp.round() as usize,
                );
                for (dst, src) in trajectory.iter_mut().zip(&jr.trajectory) {
                    *dst += *src;
                }
                reference = jr.corrected;
            }
            if opts.jitter_drift {
                let dr = jitter::correct_drift(&reference);
                for (dst, src) in trajectory.iter_mut().zip(&dr.trajectory) {
                    *dst += *src;
                }
            }
            for value in &mut trajectory {
                *value *= motion_strength;
            }
            let max_motion = trajectory
                .iter()
                .cloned()
                .fold(0.0f64, |m, v| m.max(v.abs()));
            let active = trajectory.iter().filter(|v| v.abs() >= 0.04).count();
            vlog!(
                opts,
                "motion registration [continuum -> extraction x{:.2}]: {} active, max {:.2} px",
                motion_strength,
                active,
                max_motion
            );
            stage!("motion preview");
            Some(trajectory)
        } else {
            None
        };

    // The profile extractor already produces a de-smiled per-frame spectrum.
    // Retain its continuum bin so transparency correction does not need a
    // second full extraction of every SER frame.
    let mut profile_continuum_flux: Option<Vec<f64>> = None;
    // F19: the flank offsets are decided BEFORE extraction so the core and
    // every flank come off one pass of the same learned subspace. The wing
    // offset depends only on the mean image and the smile, both of which
    // already exist here, so there is no ordering problem -- it was simply
    // computed later than it needed to be.
    // Wing offset, best source first: an explicit env override (diagnosis),
    // an explicit Angstrom request, then the profile-derived optimum, then
    // the historical constant. The constant is a poor last resort -- it was
    // right at 0.085 A/px and is barely off the core at 0.034 -- so it is
    // only reached when the line is too shallow to measure a flank.
    let wing_px = if let Some(v) = std::env::var("GS_WING_OFFSET")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        vlog!(opts, "wing offset: {:.1} px (GS_WING_OFFSET override)", v);
        v
    } else if opts.tune.wing_px > 0.0 {
        vlog!(opts, "wing offset: +-{:.1} px (tune)", opts.tune.wing_px);
        opts.tune.wing_px
    } else if let (Some(a), Some(d)) = (opts.wing_offset_a, opts.dispersion_a_per_px) {
        let px = a / d;
        vlog!(opts, "wing offset: +-{:.3} A = +-{:.1} px (requested)", a, px);
        px
    } else {
        match profile::optimal_wing_offset(&mean_img, &smile, geom.y1, geom.y2) {
            Some(wo) => {
                let in_a = opts
                    .dispersion_a_per_px
                    .map(|d| format!(" = +-{:.3} A", wo.px * d))
                    .unwrap_or_default();
                vlog!(
                    opts,
                    "wing offset: +-{:.1} px{} (auto: blue {:+.0}, red {:+.0}, HWHM {:.0} px)",
                    wo.px, in_a, wo.blue_px, wo.red_px, wo.hwhm_px
                );
                wo.px
            }
            None => {
                vlog!(opts, "wing offset: 6.0 px (line too shallow to measure a flank)");
                6.0
            }
        }
    };
    let mut wing_req: Vec<f64> = if use_profile && !opts.baseline {
        opts.shift_series.clone()
    } else {
        Vec::new()
    };
    let wing_doppler_idx = if use_profile && !opts.baseline {
        let i = wing_req.len();
        wing_req.push(-wing_px);
        wing_req.push(wing_px);
        Some(i)
    } else {
        None
    };
        let _ = &wing_doppler_idx;

    let (mut disk, flex, mut velocity_raw, kl_wings): (
        Image,
        Vec<f64>,
        Option<Image>,
        Vec<Image>,
    ) = if use_profile {
        let (maps, on_gpu) = profile::extract_profile_auto(
            &reader,
            &geom,
            &mean_img,
            transpose,
            opts.shift,
            &ptune,
            opts.use_gpu,
            pre_extraction_motion.as_deref(),
            &wing_req,
        );
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
        let flex = match profile::estimate_flexure_telluric(
            &maps, &smile, 3.0, opts.dispersion_a_per_px,
        ) {
            Some(tf) => {
                vlog!(
                    opts,
                    "flexure: telluric-anchored, {} line(s) at offsets {:?} px (dispersion {} A/px)",
                    tf.n_lines,
                    tf.line_offsets.iter().map(|o| *o as i64).collect::<Vec<_>>(),
                    match tf.dispersion {
                        Some(d) => format!("{d:.4}"),
                        None => "unknown".into(),
                    }
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
        (maps.core, flex, Some(vel), maps.wings)
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
        (reconstruct_disk(&reader, &geom, &exopts), vec![0.0; n], None, Vec::new())
    };
    if regrid_src.is_some() {
        let before = disk.w;
        disk = regrid(disk);
        velocity_raw = velocity_raw.map(&regrid);
        vlog!(opts, "time regrid: {} frames -> {} columns", before, disk.w);
        stage!("time regrid");
    }
    let raw_disk = disk.clone();
    vlog!(opts, "raw disk: {}x{}", disk.w, disk.h);
    // F18: measure the per-frame seeing before any correction stage touches
    // the disk — jitter registration and NLM both alter the along-slit
    // spectrum this reads, so measuring afterwards would measure the pipeline.
    let frame_blur = {
        let est = crate::momfbd::estimate_frame_blur(&disk, 12);
        vlog!(
            opts,
            "seeing: {} frame(s) measured, per-frame blur spread {:.3} px^2, acf {}",
            est.n_measured,
            est.spread,
            est.acf
                .iter()
                .enumerate()
                .map(|(i, v)| format!("{}:{:.2}", i + 1, v))
                .collect::<Vec<_>>()
                .join(" ")
        );
        est.dsigma2
    };
    stage!("extraction");

    // ---- photometric & registration corrections ----
    // Per-frame series recorded before the regrid are resampled onto the same
    // grid, so every per-column vector indexes the regridded disk.
    let mut jitter_applied = match (&pre_extraction_motion, &regrid_src) {
        (Some(m), Some(src)) => crate::timing::regrid_series(m, src),
        (Some(m), None) => m.clone(),
        (None, _) => vec![0.0f64; disk.w],
    };
    let mut xreg_applied = vec![0.0f64; disk.w];
    let mut burst_flags = vec![false; disk.w];
    let mut column_gain = vec![1.0f64; disk.w];
    let mut demix_before_raw: Option<Image> = None;
    if opts.baseline {
        correct_transversalium_baseline(&mut disk);
    } else {
        if opts.transparency_correction {
            let fluxv = if let Some(flux) = profile_continuum_flux.take() {
                match &regrid_src {
                    Some(src) => crate::timing::regrid_series(&flux, src),
                    None => flux,
                }
            } else {
                let cont_opts = ExtractOptions {
                    shift: opts.shift + continuum_shift,
                    transpose_input: transpose,
                    kernel: SpectralKernel::Gaussian { sigma: 1.5 },
                    frame_offsets: if flex.iter().any(|f| f.abs() > 0.0) { Some(flex.clone()) } else { None },
                };
                let cont_disk = regrid(reconstruct_disk(&reader, &geom, &cont_opts));
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
            if opts.jitter_correction && pre_extraction_motion.is_none() {
                if opts.jitter_fast {
                    let jr =
                        jitter::correct_jitter(&disk, opts.tune.jitter_hp.round() as usize);
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
        // additive bias, scan/slit displacement and independent scan/slit
        // blur modes. The GPU kernel uses the paired solar limbs as a
        // high-weight round-disk constraint. It deliberately works in native
        // acquisition coordinates: the one-shot warp below may make the slit
        // axis diagonal in the output, but no pre-rotation/resampling is
        // needed to correct it.
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
                    "column-state [gpu x{:.2}]: gain +-{:.2}% ({} active), offset +-{:.2}%, dx +-{:.3}, dy +-{:.3}, blur-x +-{:.3}, blur-y +-{:.3}",
                    opts.tune.column_demix_strength,
                    100.0 * max_abs(&state.gain),
                    active,
                    100.0 * max_abs(&state.offset),
                    max_abs(&state.x_shift),
                    max_abs(&state.y_shift),
                    max_abs(&state.blur_x),
                    max_abs(&state.blur_y),
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
            // F20: the flanks, brought onto the core's grid and through the
            // same geometric corrections, so the patch comparison lines up row
            // for row. Photometric corrections are skipped deliberately: this
            // is a SIMILARITY metric, and a per-column gain cancels in the
            // difference between two columns at the same row.
            let nlm_aux: Vec<Image> = match wing_doppler_idx {
                Some(i) if opts.tune.nlm_spec > 0.0 && kl_wings.len() > i + 1 => [i, i + 1]
                    .iter()
                    .map(|&k| {
                        let mut d = regrid(kl_wings[k].clone());
                        if jitter_applied.iter().any(|v| v.abs() > 1e-6) {
                            d = jitter::apply_shifts(&d, &jitter_applied);
                        }
                        if xreg_applied.iter().any(|v| v.abs() > 1e-6) {
                            d = jitter::apply_x_offsets(&d, &xreg_applied);
                        }
                        d
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let mut done_gpu = false;
            if opts.use_gpu && nlm_aux.is_empty() {
                if let Some((sigma, h2, thresh)) = quality::nlm_params(&disk, opts.tune.nlm_h) {
                    if let Some(out) = crate::gpu::temporal_nlm(&disk, radius, h2, sigma, thresh) {
                        disk = out;
                        done_gpu = true;
                    }
                }
            }
            if !done_gpu {
                disk = if nlm_aux.is_empty() {
                    quality::temporal_nlm(&disk, radius, opts.tune.nlm_h)
                } else {
                    let refs: Vec<&Image> = nlm_aux.iter().collect();
                    vlog!(
                        opts,
                        "spectro-temporal NLM: {} spectral channel(s), h {:.2} sigma",
                        refs.len() + 1,
                        opts.tune.nlm_spec
                    );
                    quality::temporal_nlm_spectral(&disk, &refs, radius, opts.tune.nlm_spec)
                };
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
    let aspect = disk.w.max(1) as f64 / disk.h.max(1) as f64;
    // Sparse full-disk scans can hit ~0.07 (JSolEx reports X/Y≈0.07 on 338-frame
    // 16_00_03.ser). Only reject truly absurd shapes.
    if !(0.05..=20.0).contains(&aspect) {
        return Err(format!(
            "raw disk is extremely elongated ({}×{}, aspect {:.2}). \
             Usually wrong SER orientation (full-sensor multi-line needs auto-detect or \
             --dispersion horizontal|vertical) or an incomplete scan (too few frames).",
            disk.w, disk.h, aspect
        ));
    }
    let mut fit = if opts.baseline {
        let conic = ellipse::fit_direct(&pts).ok_or("baseline ellipse fit failed")?;
        let geom2 = conic.geometry().ok_or("baseline conic not an ellipse")?;
        ellipse::RansacResult { conic, geom: geom2, inliers: pts.len(), total: pts.len(), residual_rms: 0.0 }
    } else {
        ellipse::fit_robust(&pts, 1234).ok_or_else(|| {
            format!(
                "robust ellipse fit failed ({} limb points on {}×{} disk). \
                 Check orientation (--dispersion), line selection, and that the scan covers a circular solar disk.",
                pts.len(),
                disk.w,
                disk.h
            )
        })?
    };
    let geom_before = fit.geom;
    fit.geom = ellipse::regularize_partial_fov(&fit.geom, &pts, disk.w, disk.h);
    if (fit.geom.radius - geom_before.radius).abs() > 1.0
        || (fit.geom.sx - geom_before.sx).abs() > 0.01
    {
        vlog!(
            opts,
            "ellipse regularized (partial-FOV / sparse scan): sx {:.4}→{:.4} r {:.1}→{:.1}",
            geom_before.sx,
            fit.geom.sx,
            geom_before.radius,
            fit.geom.radius
        );
    }
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
    vlog!(
        opts,
        "column/slit axis in output: {:+.2} deg from vertical (geometry-aware native-axis correction)",
        crate::warp::slit_axis_angle_deg(&fit.geom, &wp)
    );
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

    // ---- F14: co-registered wavelength series ----
    // Every offset is extracted in frame space, put through the SAME
    // photometric and registration corrections as the primary, and warped with
    // the PRIMARY's fitted geometry. Re-fitting the limb per wavelength is
    // what makes independent `--shift` runs incomparable: the limb profile
    // changes across the line and the fit follows it, so canvases came out
    // 2680-3454 px for one scan. Sharing fit.geom is what makes a Dopplergram,
    // a blink, or any difference between wavelengths meaningful.
    let mut shift_products: Vec<(f64, Image)> = Vec::new();
    if !opts.shift_series.is_empty() && !opts.baseline {
        let flex_off: Option<Vec<f64>> = if flex.iter().any(|f| f.abs() > 0.0) {
            Some(flex.clone())
        } else {
            None
        };
        for (si, &sh) in opts.shift_series.iter().enumerate() {
            // F19: read the flank off the SAME learned subspace as the core,
            // at the index that keeps the WAVELENGTH fixed. Falling back to a
            // fresh point-sampled extraction would put the products the user
            // actually wants -- the flanks, where the Halpha texture lives --
            // on a cruder estimator than the core, which shows the least.
            let mut d = if let Some(im) = kl_wings.get(si) {
                regrid(im.clone())
            } else {
                regrid(reconstruct_disk(
                    &reader,
                    &geom,
                    &ExtractOptions {
                        shift: opts.shift + sh,
                        transpose_input: transpose,
                        kernel: SpectralKernel::Point,
                        frame_offsets: flex_off.clone(),
                    },
                ))
            };
            if opts.transparency_correction {
                flatfield::apply_column_gains(&mut d, &column_gain);
            }
            if jitter_applied.iter().any(|v| v.abs() > 1e-6) {
                d = jitter::apply_shifts(&d, &jitter_applied);
            }
            if xreg_applied.iter().any(|v| v.abs() > 1e-6) {
                d = jitter::apply_x_offsets(&d, &xreg_applied);
            }
            if opts.transversalium_correction {
                flatfield::correct_transversalium(&mut d, opts.tune.transv_deadband);
            }
            if opts.temporal_nlm {
                let radius = opts.tune.nlm_radius.round().max(1.0) as usize;
                d = if opts.use_gpu {
                    quality::nlm_params(&d, opts.tune.nlm_h)
                        .and_then(|(sigma, h2, thresh)| {
                            crate::gpu::temporal_nlm(&d, radius, h2, sigma, thresh)
                        })
                        .unwrap_or_else(|| quality::temporal_nlm(&d, radius, opts.tune.nlm_h))
                } else {
                    quality::temporal_nlm(&d, radius, opts.tune.nlm_h)
                };
            }
            let warped = if opts.use_gpu {
                crate::gpu::warp_single(&d, &fit.geom, &wp)
                    .unwrap_or_else(|| warp_single(&d, &fit.geom, &wp))
            } else {
                warp_single(&d, &fit.geom, &wp)
            };
            shift_products.push((sh, warped.image));
        }
        vlog!(
            opts,
            "wavelength series: {} offset(s) {:?} px on the primary grid",
            shift_products.len(),
            opts.shift_series.iter().map(|s| format!("{s:+.1}")).collect::<Vec<_>>()
        );
        stage!("wavelength series");
    }

    // ---- multi-line composite (opt-in detail product) ----
    // Primary science path above is unchanged. When enabled and companions
    // exist, extract each neighbour line with the same shared corrections
    // (transparency, jitter, x-reg), warp with primary geometry, and
    // sharpness-weighted stack onto the primary (reference index 0).
    // Optical flow is off: formation-height differences would hallucinate
    // structure if we tried to warp features line-to-line.
    let mut composite_n_lines: Option<usize> = None;
    let mut composite_line_offsets_px: Vec<f64> = Vec::new();
    if opts.multi_line_composite && !opts.baseline {
        let primary_cx = {
            let ymid = (geom.y1 + geom.y2) as f64 * 0.5;
            polyval(&geom.coeffs, ymid)
        };
        // Continuous multi-candidate seeds (same engine as primary AUTO), not
        // just residual dips on a de-smiled profile — more reliable on
        // full-sensor multi-line frames.
        let companion_seeds =
            linefit::detect_companion_seeds(&mean_img, primary_cx, 0.04, 14.0, 5);
        if companion_seeds.is_empty() {
            vlog!(
                opts,
                "multi-line composite: no companion lines (primary at {:.1} px)",
                primary_cx
            );
        } else {
            let flex_off: Option<Vec<f64>> = if flex.iter().any(|f| f.abs() > 0.0) {
                Some(flex.clone())
            } else {
                None
            };
            let mut warped_lines: Vec<Image> = vec![output.image.clone()];
            composite_line_offsets_px.push(0.0);
            let mut used_mids: Vec<f64> = vec![primary_cx];
            for &seed_x in &companion_seeds {
                let offset = seed_x - primary_cx;
                // Track companion with primary smile slope + spectral offset.
                let seed = |y: f64| polyval(&geom.coeffs, y) + offset;
                let Some(cgeom) = linefit::fit_line_geometry_seeded(&mean_img, 2, 10, &seed)
                    .or_else(|| linefit::fit_line_geometry_at(&mean_img, 2, seed_x))
                else {
                    vlog!(opts, "multi-line: seed {:.1} px — track failed", seed_x);
                    continue;
                };
                let c_mid = {
                    let ymid = (cgeom.y1 + cgeom.y2) as f64 * 0.5;
                    polyval(&cgeom.coeffs, ymid)
                };
                // Reject tracks that wandered away from the seed or collapsed
                // onto another already-used line.
                if (c_mid - seed_x).abs() > 50.0 {
                    vlog!(
                        opts,
                        "multi-line: seed {:.1} wandered to {:.1}, skip",
                        seed_x,
                        c_mid
                    );
                    continue;
                }
                if cgeom.n_rows_used < 200 || cgeom.rms > 2.5 {
                    vlog!(
                        opts,
                        "multi-line: seed {:.1} weak track ({} rows, rms {:.2}), skip",
                        seed_x,
                        cgeom.n_rows_used,
                        cgeom.rms
                    );
                    continue;
                }
                if used_mids.iter().any(|m| (c_mid - m).abs() < 20.0) {
                    vlog!(opts, "multi-line: seed {:.1} duplicate of existing line, skip", seed_x);
                    continue;
                }
                let mut cdisk = regrid(reconstruct_disk(
                    &reader,
                    &cgeom,
                    &ExtractOptions {
                        shift: opts.shift,
                        transpose_input: transpose,
                        kernel: SpectralKernel::Point,
                        frame_offsets: flex_off.clone(),
                    },
                ));
                if opts.transparency_correction {
                    flatfield::apply_column_gains(&mut cdisk, &column_gain);
                }
                if jitter_applied.iter().any(|v| v.abs() > 1e-6) {
                    cdisk = jitter::apply_shifts(&cdisk, &jitter_applied);
                }
                if xreg_applied.iter().any(|v| v.abs() > 1e-6) {
                    cdisk = jitter::apply_x_offsets(&cdisk, &xreg_applied);
                }
                if opts.transversalium_correction {
                    flatfield::correct_transversalium(&mut cdisk, opts.tune.transv_deadband);
                }
                if opts.temporal_nlm {
                    let radius = opts.tune.nlm_radius.round().max(1.0) as usize;
                    if let Some((sigma, h2, thresh)) =
                        quality::nlm_params(&cdisk, opts.tune.nlm_h)
                    {
                        cdisk = if opts.use_gpu {
                            crate::gpu::temporal_nlm(&cdisk, radius, h2, sigma, thresh)
                                .unwrap_or_else(|| {
                                    quality::temporal_nlm(&cdisk, radius, opts.tune.nlm_h)
                                })
                        } else {
                            quality::temporal_nlm(&cdisk, radius, opts.tune.nlm_h)
                        };
                    }
                }
                let cwarp = if opts.use_gpu {
                    crate::gpu::warp_single(&cdisk, &fit.geom, &wp)
                        .unwrap_or_else(|| warp_single(&cdisk, &fit.geom, &wp))
                } else {
                    warp_single(&cdisk, &fit.geom, &wp)
                };
                warped_lines.push(cwarp.image);
                used_mids.push(c_mid);
                composite_line_offsets_px.push(c_mid - primary_cx);
                vlog!(
                    opts,
                    "multi-line: companion at {:+.1} px (seed {:.1}, smile rms {:.3}, {} rows)",
                    c_mid - primary_cx,
                    seed_x,
                    cgeom.rms,
                    cgeom.n_rows_used
                );
            }
            // Already co-registered via shared warp geometry — do not re-fit disks.
            match stack::stack_coregistered(&warped_lines, 0) {
                Some(sr) if sr.n_used > 1 => {
                    output.image = sr.image;
                    composite_n_lines = Some(sr.n_used);
                    vlog!(
                        opts,
                        "multi-line composite: stacked {} line(s) at offsets {:?} px (weights {:?})",
                        sr.n_used,
                        composite_line_offsets_px
                            .iter()
                            .map(|o| format!("{o:+.1}"))
                            .collect::<Vec<_>>(),
                        sr.weights
                            .iter()
                            .map(|w| format!("{w:.2}"))
                            .collect::<Vec<_>>()
                    );
                    stage!("multi-line composite");
                }
                _ => {
                    if warped_lines.len() == 1 {
                        vlog!(opts, "multi-line composite: no valid companions, keeping primary");
                    } else {
                        vlog!(opts, "multi-line composite: stack failed, keeping primary");
                    }
                    composite_line_offsets_px.clear();
                }
            }
        }
    }

    // Wing-difference Dopplergram: intensities at +-wing_offset from the
    // (flexure-corrected) line center; the normalized difference cancels
    // column gains and transversalium and is maximally shift-sensitive
    // (wing slope), measuring at bisector depths where rotation is clean.
    let wing_doppler: Option<Image> = if use_profile && !opts.baseline {
        let wing = wing_px;
        let offsets: Option<Vec<f64>> = if std::env::var("GS_WING_NOFLEX").is_ok() {
            None // INTI-condition: wings extracted without flexure correction
        } else if flex.iter().any(|f| f.abs() > 0.0) {
            Some(flex.clone())
        } else {
            None
        };
        let mk = |sh: f64| {
            regrid(reconstruct_disk(&reader, &geom, &ExtractOptions {
                shift: opts.shift + sh,
                transpose_input: transpose,
                kernel: SpectralKernel::Gaussian { sigma: 1.0 },
                frame_offsets: offsets.clone(),
            }))
        };
        let (blue, red) = match wing_doppler_idx {
            Some(i) if kl_wings.len() > i + 1 && std::env::var("GS_WING_POINT").is_err() => {
                (regrid(kl_wings[i].clone()), regrid(kl_wings[i + 1].clone()))
            }
            _ => (mk(-wing), mk(wing)),
        };
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

    // ---- F15: Wiener filtering ----
    // Last in the chain: it needs the final geometry and the final noise
    // level, and it is the stage that decides which spatial frequencies are
    // worth keeping at all.
    if opts.wiener && !opts.baseline {
        match denoise::wiener_psd(&output.image, opts.tune.wiener_strength, Some(&disk_fit)) {
            Some((img, rep)) => {
                vlog!(
                    opts,
                    "wiener: cutoff {:.1} px, removed {:.1}% of power (strength {:.2})",
                    rep.cutoff_px,
                    100.0 * rep.removed,
                    opts.tune.wiener_strength
                );
                output.image = img;
            }
            None => vlog!(opts, "wiener: image too small, skipped"),
        }
        stage!("wiener");
    }

    vlog!(opts, "[t] TOTAL: {:.2}s", t_start.elapsed().as_secs_f64());
    Ok(ReconReport {
        frame_blur,
        output,
        demix_before,
        raw_disk,
        velocity,
        wing_doppler,
        flex,
        psf_sigma,
        shift_products,
        timing: timing.as_ref().map(|t| t.summarize(regrid_src.is_some())),
        gap_columns: match (&timing, &regrid_src) {
            (Some(t), Some(src)) => crate::timing::gap_columns(t, src),
            _ => Vec::new(),
        },
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
        composite_n_lines,
        composite_line_offsets_px,
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
/// Open a scan and recover the three things every spectral diagnostic needs:
/// the dispersion orientation, the mean image, and the line geometry.
pub fn scan_setup(ser_path: &Path) -> Result<(SerReader, bool, Image, linefit::LineGeometry), String> {
    let reader = SerReader::open(ser_path).map_err(|e| format!("SER open: {e}"))?;
    let hdr = &reader.header;
    let n = hdr.frame_count;
    let sample_n = n.min(32).max(1);
    let mut probe_acc = vec![0.0f64; hdr.width * hdr.height];
    for t in 0..sample_n {
        let f = reader.frame(t);
        for (i, &v) in f.data.iter().enumerate() {
            probe_acc[i] += v as f64;
        }
    }
    let mut probe = Image::new(hdr.width, hdr.height);
    let pc = sample_n as f64;
    for (i, v) in probe_acc.iter().enumerate() {
        probe.data[i] = (v / pc) as f32;
    }
    let (transpose, _, _) = linefit::should_transpose_for_dispersion(&probe);
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
    Ok((reader, transpose, mean_img, geom))
}

pub fn mean_spectrum(ser_path: &Path) -> Result<(Vec<i64>, Vec<f64>), String> {
    let (_reader, _transpose, mean_img, geom) = scan_setup(ser_path)?;
    let w = mean_img.w;
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
