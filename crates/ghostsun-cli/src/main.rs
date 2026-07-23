use ghostsun_core::mathutil;
use ghostsun_core::metrics;
use ghostsun_core::output;
use ghostsun_core::pipeline;
use ghostsun_core::render;
use ghostsun_core::stack;
use ghostsun_core::synth;

use clap::{Parser, Subcommand};
use ghostsun_core::image2d::Image;
use std::path::{Path, PathBuf};
use synth::VEL_SCALE;

#[derive(Parser)]
#[command(name = "ghostsun", about = "High-fidelity solar disk reconstruction from spectroheliograph SER scans")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Parser, Clone)]
struct SynthArgs {
    #[arg(long, default_value_t = 900)]
    frames: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 2.0)]
    tilt: f64,
    #[arg(long, default_value_t = 0.62)]
    scan_step: f64,
    #[arg(long, default_value_t = 0.35)]
    jitter: f64,
    #[arg(long, default_value_t = 255.0)]
    radius: f64,
    /// disable all degradations (pipeline ceiling test)
    #[arg(long)]
    clean: bool,
    /// enable Doppler velocity field (rotation + turbulence)
    #[arg(long)]
    doppler: bool,
    /// rotation gradient along the slit (N-S scan geometry)
    #[arg(long)]
    doppler_ns: bool,
    /// spectral flexure amplitude in px (0 = off)
    #[arg(long, default_value_t = 0.0)]
    flexure: f64,
    /// seeing PSF sigma in px (0 = off; also enables slit boxcar)
    #[arg(long, default_value_t = 0.0)]
    psf: f64,
    /// number of sequential scans (multi-scan stacking tests)
    #[arg(long, default_value_t = 1)]
    scans: usize,
    /// signal level multiplier (low-SNR tests)
    #[arg(long, default_value_t = 1.0)]
    exposure: f64,
    /// scan-direction seeing jitter sigma in sun px (F9 tests)
    #[arg(long, default_value_t = 0.0)]
    jitter_x: f64,
    /// add telluric anchor lines to the synthetic spectrum
    #[arg(long)]
    telluric: bool,
    /// fraction of frames hit by seeing bursts (F11 tests)
    #[arg(long, default_value_t = 0.0)]
    bursts: f64,
}

impl SynthArgs {
    fn to_params(&self) -> synth::SynthParams {
        synth::SynthParams {
            n_frames: self.frames,
            seed: self.seed,
            tilt_deg: self.tilt,
            scan_step: self.scan_step,
            jitter_sigma: self.jitter,
            jitter_x_sigma: self.jitter_x,
            radius: self.radius,
            clean: self.clean,
            doppler: self.doppler,
            doppler_ns: self.doppler_ns,
            flexure_px: self.flexure,
            psf_seeing_px: self.psf,
            n_scans: self.scans,
            exposure: self.exposure,
            telluric: self.telluric,
            bursts: self.bursts,
            ..Default::default()
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a synthetic SER scan with known ground truth
    Synth {
        #[arg(long, default_value = "testdata")]
        out_dir: PathBuf,
        #[command(flatten)]
        args: SynthArgs,
    },
    /// Reconstruct a solar disk from a SER file
    Recon {
        ser: PathBuf,
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,
        /// Minimal comparison pipeline with adaptive local-polynomial extraction
        #[arg(long)]
        baseline: bool,
        /// wavelength shift in px from line center
        #[arg(long, default_value_t = 0.0)]
        shift: f64,
        /// Gaussian spectral window sigma in px (B-spline mode only)
        #[arg(long, default_value_t = 0.0)]
        window_sigma: f64,
        /// rotation to apply (P angle), degrees
        #[arg(long, default_value_t = 0.0)]
        rotation: f64,
        #[arg(long)]
        flip_x: bool,
        #[arg(long)]
        flip_y: bool,
        #[arg(long)]
        no_jitter: bool,
        #[arg(long)]
        no_transparency: bool,
        #[arg(long)]
        no_transversalium: bool,
        /// use plain B-spline extraction instead of the profile model (F1)
        #[arg(long)]
        no_profile: bool,
        /// disable the footprint-filtered warp (F4)
        #[arg(long)]
        no_filtered_warp: bool,
        /// write the Doppler velocity map (F2)
        #[arg(long)]
        velocity: bool,
        /// also write a colorized PNG (black background, prominences kept)
        #[arg(long)]
        colorize: bool,
        /// PSF estimation + Richardson-Lucy deconvolution (F6)
        #[arg(long)]
        deconv: bool,
        /// variance-stabilized wavelet denoising (F7)
        #[arg(long)]
        denoise: bool,
        /// disable scan-direction registration (F9.2)
        #[arg(long)]
        no_xreg: bool,
        /// disable temporal burst repair (F11)
        #[arg(long)]
        no_burst_repair: bool,
        /// disable temporal NLM smoothing (F11.5)
        #[arg(long)]
        no_nlm: bool,
        /// disable GPU compute kernels (CPU only)
        #[arg(long)]
        no_gpu: bool,
        /// extra block-coordinate refinement iterations (F8)
        #[arg(long, default_value_t = 0)]
        map_iterations: usize,
        /// tuning overrides, e.g. --tune pca_k=4,w_fit=10
        #[arg(long)]
        tune: Option<String>,
        /// output name stem
        #[arg(long, default_value = "recon")]
        name: String,
    },
    /// Fit the disk in an image and report residual ellipticity
    Diskfit {
        image: PathBuf,
    },
    /// Verify GPU kernels match CPU implementations (equivalence + timing)
    Gpucheck,
    /// Render a colorized PNG (black background, prominences preserved)
    Colorize {
        /// input reconstruction (.fits from `recon`/`stack`, or 16-bit PNG)
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// prominence brightness boost (relative to disk stretch)
        #[arg(long, default_value_t = 3.0)]
        prom_boost: f64,
        /// disk tone-curve gamma
        #[arg(long, default_value_t = 0.7)]
        gamma: f64,
    },
    /// Dump the de-smiled mean spectrum profile of a SER scan (CSV)
    Spectrum {
        ser: PathBuf,
        #[arg(long, default_value = "spectrum.csv")]
        out: PathBuf,
    },
    /// Compare a reconstruction against ground truth (16-bit PNGs)
    Eval {
        recon: PathBuf,
        truth: PathBuf,
    },
    /// Compare velocity maps (needs the intensity pair for registration)
    EvalVelocity {
        recon_v: PathBuf,
        truth_v: PathBuf,
        recon_i: PathBuf,
        truth_i: PathBuf,
    },
    /// Register and stack multiple reconstructions (F5)
    Stack {
        /// input reconstructions (.fits from `recon`)
        inputs: Vec<PathBuf>,
        #[arg(long, default_value = "stacked")]
        name: String,
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,
        /// disable optical-flow evolution compensation
        #[arg(long)]
        no_flow: bool,
    },
    /// Full benchmark: synth -> recon (ghostsun + baseline + ablations) -> eval
    Bench {
        #[arg(long, default_value = "testdata")]
        dir: PathBuf,
        #[command(flatten)]
        args: SynthArgs,
        /// also run single-stage ablations
        #[arg(long)]
        ablations: bool,
        /// sweep one tuning parameter: name=v1,v2,v3
        #[arg(long)]
        sweep: Option<String>,
        /// append results to this JSON-lines ledger
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

fn parse_tune(s: &str, tune: &mut pipeline::TuneParams) {
    for kv in s.split(',') {
        let mut it = kv.splitn(2, '=');
        let (k, v) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
        match v.parse::<f64>() {
            Ok(val) => {
                if let Err(e) = tune.set(k.trim(), val) {
                    eprintln!("warning: {e}");
                }
            }
            Err(_) => eprintln!("warning: bad tune value in '{kv}'"),
        }
    }
}

fn write_velocity_png(path: &Path, v: &Image) -> std::io::Result<()> {
    let mut enc = Image::new(v.w, v.h);
    for i in 0..v.data.len() {
        enc.data[i] = ((v.data[i] as f64 / VEL_SCALE + 1.0) / 2.0 * 65535.0).clamp(0.0, 65535.0) as f32;
    }
    output::write_png16(path, &enc, Some((0.0, 65535.0)))
}

fn decode_velocity_png(path: &Path) -> std::io::Result<Image> {
    let enc = output::read_png16(path)?;
    let mut v = Image::new(enc.w, enc.h);
    for i in 0..v.data.len() {
        v.data[i] = ((enc.data[i] as f64 / 65535.0 * 2.0 - 1.0) * VEL_SCALE) as f32;
    }
    Ok(v)
}

fn save_recon_outputs(dir: &Path, name: &str, rep: &pipeline::ReconReport, velocity: bool) {
    let img = &rep.output.image;
    let mx = img.max();
    output::write_png16(&dir.join(format!("{name}_linear.png")), img, Some((0.0, mx))).unwrap();
    output::write_png16(&dir.join(format!("{name}_display.png")), img, None).unwrap();
    output::write_fits_f32(&dir.join(format!("{name}.fits")), img).unwrap();
    if std::env::var_os("GS_SAVE_DEMIX_BEFORE").is_some() {
        if let Some(before) = &rep.demix_before {
            output::write_png16(&dir.join(format!("{name}_before_demix.png")), before, None).unwrap();
            output::write_fits_f32(&dir.join(format!("{name}_before_demix.fits")), before).unwrap();
        }
    }
    if velocity {
        if let Some(v) = &rep.velocity {
            write_velocity_png(&dir.join(format!("{name}_velocity.png")), v).unwrap();
            output::write_fits_f32(&dir.join(format!("{name}_velocity.fits")), v).unwrap();
        }
        if let Some(wd) = &rep.wing_doppler {
            output::write_fits_f32(&dir.join(format!("{name}_wingdopp.fits")), wd).unwrap();
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Synth { out_dir, args } => {
            std::fs::create_dir_all(&out_dir).unwrap();
            let params = args.to_params();
            let ser = out_dir.join("synth.ser");
            let gt = out_dir.join("ground_truth.png");
            let truth = synth::generate(&params, &ser, &gt).unwrap();
            println!("wrote {} ({} scan(s)) and {}", ser.display(), truth.n_scans, gt.display());
            println!(
                "truth: smile c0={:.2} c1={:.4} c2={:.6}, tilt {:.2} deg, step {:.3}",
                truth.smile[0], truth.smile[1], truth.smile[2], args.tilt, args.scan_step
            );
        }
        Cmd::Recon {
            ser, out_dir, baseline, shift, window_sigma, rotation, flip_x, flip_y,
            no_jitter, no_transparency, no_transversalium, no_profile, no_filtered_warp,
            velocity, colorize, deconv, denoise, no_xreg, no_burst_repair, no_nlm, no_gpu, map_iterations, tune, name,
        } => {
            std::fs::create_dir_all(&out_dir).unwrap();
            let mut opts = pipeline::ReconOptions {
                baseline,
                shift,
                window_sigma,
                rotation_deg: rotation,
                flip_x,
                flip_y,
                jitter_correction: !no_jitter,
                transparency_correction: !no_transparency,
                transversalium_correction: !no_transversalium,
                profile_extraction: !no_profile,
                filtered_warp: !no_filtered_warp,
                deconv,
                denoise,
                x_registration: !no_xreg,
                burst_repair: !no_burst_repair,
                temporal_nlm: !no_nlm,
                use_gpu: !no_gpu,
                map_iterations,
                ..Default::default()
            };
            if let Some(t) = &tune {
                parse_tune(t, &mut opts.tune);
            }
            match pipeline::reconstruct(&ser, &opts) {
                Ok(rep) => {
                    save_recon_outputs(&out_dir, &name, &rep, velocity);
                    if colorize {
                        let copts = render::ColorizeOptions::default();
                        if let Some((w, h, rgb)) = render::colorize(&rep.output.image, &copts) {
                            output::write_png_rgb(&out_dir.join(format!("{name}_color.png")), w, h, &rgb).unwrap();
                            println!("wrote {}/{name}_color.png", out_dir.display());
                        }
                    }
                    println!("wrote {}/{{{name}.fits,{name}_linear.png,{name}_display.png}}", out_dir.display());
                }
                Err(e) => {
                    eprintln!("reconstruction failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Gpucheck => {
            use ghostsun_core::{gpu, quality, warp as warpmod, ellipse as ellmod};
            // synthetic test image: smooth structure + disk + noise
            let (w, h) = (2400usize, 1200usize);
            let mut img = Image::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f64 - 1200.0;
                    let dy = y as f64 - 600.0;
                    let r = (dx * dx / 4.0 + dy * dy).sqrt();
                    let disk = if r < 500.0 { 20000.0 } else { 300.0 };
                    let tex = 2000.0 * ((x as f64 / 17.0).sin() * (y as f64 / 23.0).cos());
                    let noise = 150.0 * (((x * 7919 + y * 104729) % 1000) as f64 / 500.0 - 1.0);
                    img.set(x, y, (disk + tex * if r < 500.0 { 1.0 } else { 0.0 } + noise).max(0.0) as f32);
                }
            }
            let rel = |a: &Image, b: &Image| -> f64 {
                let peak = 20000.0f64;
                a.data
                    .iter()
                    .zip(&b.data)
                    .map(|(x, y)| ((x - y).abs() as f64) / peak)
                    .fold(0.0, f64::max)
            };
            // NLM
            let (sigma, h2, thresh) = quality::nlm_params(&img, 1.8).expect("params");
            let t0 = std::time::Instant::now();
            let cpu = quality::temporal_nlm(&img, 3, 1.8);
            let t_cpu = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            match gpu::temporal_nlm(&img, 3, h2, sigma, thresh) {
                Some(gout) => {
                    let t_gpu = t0.elapsed().as_secs_f64();
                    println!(
                        "NLM : max rel diff {:.2e}  cpu {:.3}s  gpu {:.3}s  ({:.1}x)",
                        rel(&cpu, &gout), t_cpu, t_gpu, t_cpu / t_gpu
                    );
                }
                None => println!("NLM : GPU unavailable"),
            }
            // Column-state regression: inject a repeating gain tooth plus
            // short additive-bias runs across otherwise untouched columns.
            // A successful demixer must reduce the known error while its
            // clean-scan evidence gate leaves the original bit-for-bit alone.
            let mut striped = img.clone();
            for x in 0..w {
                let gain = match x % 6 {
                    0 => 0.010,
                    1 => -0.008,
                    2 => 0.005,
                    3 => -0.004,
                    _ => 0.0,
                };
                let phase = x % 73;
                let bias = if (24..=29).contains(&phase) {
                    -160.0
                } else if (51..=54).contains(&phase) {
                    120.0
                } else {
                    0.0
                };
                for y in 0..h {
                    let i = y * w + x;
                    striped.data[i] = (striped.data[i] * (1.0 + gain) + bias).max(0.0);
                }
            }
            let rmse = |a: &Image, b: &Image| -> f64 {
                (a.data
                    .iter()
                    .zip(&b.data)
                    .map(|(x, y)| {
                        let d = (*x - *y) as f64;
                        d * d
                    })
                    .sum::<f64>()
                    / a.data.len() as f64)
                    .sqrt()
            };
            match (gpu::demix_columns(&striped, 1.0), gpu::demix_columns(&img, 1.0)) {
                (Some((fixed, state)), Some((clean, clean_state))) => {
                    let before = rmse(&striped, &img);
                    let after = rmse(&fixed, &img);
                    let clean_delta = rmse(&clean, &img);
                    let max_offset = state
                        .offset
                        .iter()
                        .fold(0.0f64, |m, v| m.max(v.abs()));
                    let clean_blur_x = clean_state
                        .blur_x
                        .iter()
                        .fold(0.0f64, |m, v| m.max(v.abs()));
                    let clean_blur_y = clean_state
                        .blur_y
                        .iter()
                        .fold(0.0f64, |m, v| m.max(v.abs()));
                    println!(
                        "COL : injected RMSE {:.2} -> {:.2} ({:.1}% removed), clean delta {:.3}, offset +- {:.3}%, clean blur x/y {:.3}/{:.3}",
                        before,
                        after,
                        100.0 * (1.0 - after / before),
                        clean_delta,
                        100.0 * max_offset,
                        clean_blur_x,
                        clean_blur_y,
                    );
                    if let (Some((fixed2, _)), Some((clean2, _))) =
                        (gpu::demix_columns(&striped, 0.5), gpu::demix_columns(&img, 0.5))
                    {
                        let after2 = rmse(&fixed2, &img);
                        println!(
                            "COL.5: injected RMSE {:.2} -> {:.2} ({:.1}% removed), clean delta {:.3}",
                            before,
                            after2,
                            100.0 * (1.0 - after2 / before),
                            rmse(&clean2, &img),
                        );
                    }
                }
                _ => println!("COL : GPU unavailable"),
            }
            // Directional column-PSF regression. Alternate acquisition
            // columns receive different amounts of slit-axis diffusion. The
            // final ellipse warp would render this native-y blur diagonally
            // when the scan has shear; the demixer must separate it from
            // scan-axis curvature before that warp.
            //
            // Add coherent, near-resolution slit detail to the truth. Unlike
            // uncorrelated detector noise, this is information shared by
            // neighbouring frames and is therefore legitimately recoverable.
            let mut directional_truth = img.clone();
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f64 - 1200.0;
                    let dy = y as f64 - 600.0;
                    if (dx * dx / 4.0 + dy * dy).sqrt() < 450.0 {
                        let phase = std::f64::consts::TAU
                            * (y as f64 / 4.6 + x as f64 / 700.0);
                        let i = y * w + x;
                        directional_truth.data[i] =
                            (directional_truth.data[i] + 2500.0 * phase.sin() as f32).max(0.0);
                    }
                }
            }
            let mut smeared = directional_truth.clone();
            for x in 0..w {
                let beta = match x % 6 {
                    0 => 0.24,
                    1 => 0.12,
                    2 => 0.18,
                    3 => 0.06,
                    _ => 0.0,
                };
                if beta == 0.0 {
                    continue;
                }
                for y in 1..h - 1 {
                    let i = y * w + x;
                    let curv_y = directional_truth.data[i - w] + directional_truth.data[i + w]
                        - 2.0 * directional_truth.data[i];
                    smeared.data[i] =
                        (directional_truth.data[i] + beta * curv_y).max(0.0);
                }
            }
            match gpu::demix_columns(&smeared, 1.0) {
                Some((fixed, state)) => {
                    let before = rmse(&smeared, &directional_truth);
                    let after = rmse(&fixed, &directional_truth);
                    let max_blur_y =
                        state.blur_y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    println!(
                        "COL-Y: directional RMSE {:.2} -> {:.2} ({:.1}% removed), blur-y +- {:.3}",
                        before,
                        after,
                        100.0 * (1.0 - after / before),
                        max_blur_y,
                    );
                }
                None => println!("COL-Y: GPU unavailable"),
            }
            // profile extraction (needs a synthetic SER scan)
            {
                use ghostsun_core::{linefit, profile};
                let dir = std::env::temp_dir().join("gs_gpucheck");
                std::fs::create_dir_all(&dir).unwrap();
                let ser = std::env::var("GS_CHECK_SER").map(std::path::PathBuf::from).unwrap_or(dir.join("synth.ser"));
                if !ser.exists() {
                    let params = synth::SynthParams { n_frames: 600, ..Default::default() };
                    synth::generate(&params, &ser, &dir.join("gt.png")).unwrap();
                }
                let reader = ghostsun_core::ser::SerReader::open(&ser).unwrap();
                let transpose = reader.header.width > reader.header.height;
                // mean image (subsampled)
                let n = reader.header.frame_count;
                let mut acc: Option<Vec<f64>> = None;
                let (mut mw, mut mh) = (0usize, 0usize);
                let mut cnt = 0.0;
                let mut t = 0;
                while t < n {
                    let mut f = reader.frame(t);
                    if transpose { f = f.transpose(); }
                    if f.mean() > 500.0 {
                        mw = f.w; mh = f.h;
                        let a = acc.get_or_insert_with(|| vec![0.0; mw * mh]);
                        for (i, &v) in f.data.iter().enumerate() { a[i] += v as f64; }
                        cnt += 1.0;
                    }
                    t += 3;
                }
                let mut mean_img = Image::new(mw, mh);
                for (i, v) in acc.unwrap().iter().enumerate() {
                    mean_img.data[i] = (v / cnt) as f32;
                }
                let geo = linefit::fit_line_geometry(&mean_img, 2).unwrap();
                let tune = profile::ProfileTune::default();
                let motion: Vec<f64> = (0..n)
                    .map(|t| 0.7 * (std::f64::consts::TAU * t as f64 / 37.0).sin())
                    .collect();
                let t0 = std::time::Instant::now();
                let cpu = profile::extract_profile(
                    &reader,
                    &geo,
                    &mean_img,
                    transpose,
                    0.0,
                    &tune,
                    Some(&motion),
                );
                let t_cpu = t0.elapsed().as_secs_f64();
                let t0 = std::time::Instant::now();
                match ghostsun_core::gpu_extract::extract_profile_gpu(
                    &reader,
                    &geo,
                    &mean_img,
                    transpose,
                    0.0,
                    &tune,
                    Some(&motion),
                ) {
                    Some(gout) => {
                        let t_gpu = t0.elapsed().as_secs_f64();
                        let peak = 30000.0f64;
                        let mut md_core = 0.0f64;
                        let mut md_mu = 0.0f64;
                        for i in 0..cpu.core.data.len() {
                            md_core = md_core.max((cpu.core.data[i] - gout.core.data[i]).abs() as f64 / peak);
                            let (a, b) = (cpu.mu.data[i], gout.mu.data[i]);
                            if a.is_finite() && b.is_finite() {
                                md_mu = md_mu.max((a - b).abs() as f64);
                            }
                        }
                        println!(
                            "EXTR-M: core max rel {:.2e}  mu max {:.2e} px  cpu {:.3}s  gpu {:.3}s  ({:.1}x)",
                            md_core, md_mu, t_cpu, t_gpu, t_cpu / t_gpu
                        );
                    }
                    None => println!("EXTR: GPU unavailable"),
                }
            }
            // warp
            let geom = ellmod::Conic {
                a: 1.0 / (1000.0f64 * 1000.0),
                b: 0.00000002,
                c: 1.0 / (505.0f64 * 505.0),
                d: -2.0 * 1200.0 / (1000.0f64 * 1000.0),
                e: -2.0 * 600.0 / (505.0f64 * 505.0),
                f: 1200.0 * 1200.0 / (1000.0f64 * 1000.0) + 600.0 * 600.0 / (505.0f64 * 505.0) - 1.0,
            }
            .geometry()
            .expect("geom");
            let wp = warpmod::WarpParams {
                rotation_deg: 1.5,
                flip_x: false,
                flip_y: false,
                margin_frac: 0.15,
                filtered_downscale: true,
                allow_negative: false,
            };
            let t0 = std::time::Instant::now();
            let cpu_w = warpmod::warp_single(&img, &geom, &wp);
            let t_cpu = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            match gpu::warp_single(&img, &geom, &wp) {
                Some(gout) => {
                    let t_gpu = t0.elapsed().as_secs_f64();
                    println!(
                        "WARP: max rel diff {:.2e}  cpu {:.3}s  gpu {:.3}s  ({:.1}x)",
                        rel(&cpu_w.image, &gout.image), t_cpu, t_gpu, t_cpu / t_gpu
                    );
                }
                None => println!("WARP: GPU unavailable"),
            }
        }
        Cmd::Colorize { input, out, prom_boost, gamma } => {
            let img = if input.extension().map(|e| e == "fits").unwrap_or(false) {
                output::read_fits_f32(&input).unwrap()
            } else {
                output::read_png16(&input).unwrap()
            };
            let copts = render::ColorizeOptions { prom_boost, gamma };
            match render::colorize(&img, &copts) {
                Some((w, h, rgb)) => {
                    let outp = out.unwrap_or_else(|| input.with_file_name(format!(
                        "{}_color.png",
                        input.file_stem().unwrap().to_string_lossy()
                    )));
                    output::write_png_rgb(&outp, w, h, &rgb).unwrap();
                    println!("wrote {}", outp.display());
                }
                None => {
                    eprintln!("colorize failed (disk fit)");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Spectrum { ser, out } => {
            match pipeline::mean_spectrum(&ser) {
                Ok((offsets, profile)) => {
                    let mut txt = String::from("offset_px,intensity\n");
                    for (o, v) in offsets.iter().zip(&profile) {
                        txt.push_str(&format!("{o},{v:.2}\n"));
                    }
                    std::fs::write(&out, txt).unwrap();
                    println!("wrote {} ({} samples)", out.display(), profile.len());
                }
                Err(e) => {
                    eprintln!("spectrum failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Diskfit { image } => {
            let img = if image.extension().map(|e| e == "fits").unwrap_or(false) {
                output::read_fits_f32(&image).unwrap()
            } else {
                output::read_png16(&image).unwrap()
            };
            match metrics::coarse_disk(&img) {
                Some(init) => match metrics::fit_disk_polar(&img, &init) {
                    Some((d, ellip)) => {
                        println!(
                            "polar fit: center ({:.1},{:.1}) radius {:.2}, relative ellipticity {:.4}",
                            d.xc, d.yc, d.r, ellip
                        );
                    }
                    None => eprintln!("polar refinement failed"),
                },
                None => eprintln!("coarse disk detection failed"),
            }
        }
        Cmd::Eval { recon, truth } => {
            let r = output::read_png16(&recon).unwrap();
            let g = output::read_png16(&truth).unwrap();
            match metrics::evaluate(&r, &g) {
                Some(m) => print_eval(&recon.display().to_string(), &m),
                None => {
                    eprintln!("evaluation failed (disk fit)");
                    std::process::exit(1);
                }
            }
        }
        Cmd::EvalVelocity { recon_v, truth_v, recon_i, truth_i } => {
            let rv = decode_velocity_png(&recon_v).unwrap();
            let gv = decode_velocity_png(&truth_v).unwrap();
            let ri = output::read_png16(&recon_i).unwrap();
            let gi = output::read_png16(&truth_i).unwrap();
            match metrics::evaluate_velocity(&ri, &gi, &rv, &gv) {
                Some(rms) => println!("velocity RMS: {:.4} px", rms),
                None => {
                    eprintln!("velocity evaluation failed");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Stack { inputs, name, out_dir, no_flow } => {
            std::fs::create_dir_all(&out_dir).unwrap();
            let images: Vec<Image> = inputs
                .iter()
                .map(|p| {
                    if p.extension().map(|e| e == "fits").unwrap_or(false) {
                        output::read_fits_f32(p).unwrap()
                    } else {
                        output::read_png16(p).unwrap()
                    }
                })
                .collect();
            match stack::stack(&images, !no_flow, true) {
                Some(rep) => {
                    let mx = rep.image.max();
                    output::write_png16(&out_dir.join(format!("{name}_linear.png")), &rep.image, Some((0.0, mx))).unwrap();
                    output::write_png16(&out_dir.join(format!("{name}_display.png")), &rep.image, None).unwrap();
                    output::write_fits_f32(&out_dir.join(format!("{name}.fits")), &rep.image).unwrap();
                    println!("stacked {} scans -> {}/{}.fits", rep.n_used, out_dir.display(), name);
                }
                None => {
                    eprintln!("stacking failed");
                    std::process::exit(1);
                }
            }
        }
        Cmd::Bench { dir, args, ablations, sweep, json } => {
            run_bench(&dir, &args, ablations, sweep.as_deref(), json.as_deref());
        }
    }
}

struct BenchRow {
    name: String,
    m: Option<metrics::EvalResult>,
    vrms: Option<f64>,
}

fn eval_variant(
    rep: &pipeline::ReconReport,
    gt: &Image,
    gt_vel: Option<&Image>,
) -> (Option<metrics::EvalResult>, Option<f64>) {
    let m = metrics::evaluate(&rep.output.image, gt);
    let vrms = match (gt_vel, &rep.velocity) {
        (Some(gv), Some(rv)) => metrics::evaluate_velocity(&rep.output.image, gt, rv, gv),
        _ => None,
    };
    (m, vrms)
}

fn run_bench(dir: &Path, args: &SynthArgs, ablations: bool, sweep: Option<&str>, json: Option<&Path>) {
    std::fs::create_dir_all(dir).unwrap();
    let params = args.to_params();
    let ser = dir.join("synth.ser");
    let gt_path = dir.join("ground_truth.png");
    let truth = synth::generate(&params, &ser, &gt_path).unwrap();
    println!("synthetic scan written; running reconstructions...\n");
    let gt = output::read_png16(&gt_path).unwrap();
    // deconvolution is judged against the unblurred truth; everything else
    // against the blurred truth when a PSF was simulated
    let gt_blurred = if params.psf_seeing_px > 0.0 {
        output::read_png16(&dir.join("ground_truth_blurred.png")).unwrap()
    } else {
        gt.clone()
    };
    let gt_vel: Option<Image> = if params.doppler {
        Some(decode_velocity_png(&dir.join("ground_truth_velocity.png")).unwrap())
    } else {
        None
    };

    // ---- sweep mode ----
    if let Some(spec) = sweep {
        let mut it = spec.splitn(2, '=');
        let pname = it.next().unwrap_or("").trim().to_string();
        let vals: Vec<f64> = it
            .next()
            .unwrap_or("")
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        println!("sweep {pname} over {vals:?}\n");
        print_header(gt_vel.is_some());
        for v in vals {
            let mut opts = pipeline::ReconOptions { verbose: false, ..Default::default() };
            if opts.tune.set(&pname, v).is_err() {
                eprintln!("unknown tune param {pname}");
                return;
            }
            // params that need feature flags to matter
            if pname.starts_with("rl_") {
                opts.deconv = true;
            }
            if pname.starts_with("denoise") {
                opts.denoise = true;
            }
            match pipeline::reconstruct(&ser, &opts) {
                Ok(rep) => {
                    let gtx = if opts.deconv { &gt } else { &gt_blurred };
                    let (m, vrms) = eval_variant(&rep, gtx, gt_vel.as_ref());
                    print_row(&format!("{pname}={v}"), &m, &vrms);
                }
                Err(e) => println!("{pname}={v}: recon failed: {e}"),
            }
        }
        return;
    }

    // ---- variant list ----
    let mut variants: Vec<(String, pipeline::ReconOptions)> = vec![
        (
            "Minimal-baseline".into(),
            pipeline::ReconOptions { baseline: true, verbose: false, ..Default::default() },
        ),
        (
            "GhostSun-full".into(),
            pipeline::ReconOptions { verbose: false, ..Default::default() },
        ),
    ];
    if ablations {
        let mk = |f: &dyn Fn(&mut pipeline::ReconOptions)| {
            let mut o = pipeline::ReconOptions { verbose: false, ..Default::default() };
            f(&mut o);
            o
        };
        variants.push(("Ghost-bspline".into(), mk(&|o| o.profile_extraction = false)));
        variants.push(("Ghost-no-pca".into(), mk(&|o| o.tune.pca_k = 0.0)));
        variants.push(("Ghost-no-fwarp".into(), mk(&|o| o.filtered_warp = false)));
        variants.push(("Ghost-no-jitter".into(), mk(&|o| o.jitter_correction = false)));
        variants.push(("Ghost-only-fast".into(), mk(&|o| o.jitter_drift = false)));
        variants.push(("Ghost-only-drift".into(), mk(&|o| o.jitter_fast = false)));
        variants.push(("Ghost-no-transp".into(), mk(&|o| o.transparency_correction = false)));
        variants.push(("Ghost-no-transv".into(), mk(&|o| o.transversalium_correction = false)));
        variants.push(("Ghost-no-xreg".into(), mk(&|o| o.x_registration = false)));
        variants.push(("Ghost-no-burst".into(), mk(&|o| o.burst_repair = false)));
        variants.push(("Ghost-no-nlm".into(), mk(&|o| o.temporal_nlm = false)));
        variants.push(("Ghost-map2".into(), mk(&|o| o.map_iterations = 2)));
    }
    if params.psf_seeing_px > 0.0 {
        let mut o = pipeline::ReconOptions { verbose: false, deconv: true, ..Default::default() };
        o.tune = pipeline::TuneParams::default();
        variants.push(("Ghost-deconv".into(), o));
    }
    if params.exposure < 0.99 {
        variants.push((
            "Ghost-denoise".into(),
            pipeline::ReconOptions { verbose: false, denoise: true, ..Default::default() },
        ));
    }

    print_header(gt_vel.is_some());
    let mut rows: Vec<BenchRow> = Vec::new();
    for (name, opts) in variants {
        match pipeline::reconstruct(&ser, &opts) {
            Ok(rep) => {
                let stem = name.to_lowercase().replace(' ', "_");
                save_recon_outputs(dir, &stem, &rep, gt_vel.is_some());
                // deconv is scored against unblurred truth; others vs blurred
                let gtx = if opts.deconv { &gt } else { &gt_blurred };
                let (m, vrms) = eval_variant(&rep, gtx, gt_vel.as_ref());
                print_row(&name, &m, &vrms);
                if name == "GhostSun-full" {
                    diagnostics(&rep, &truth, dir);
                }
                rows.push(BenchRow { name: name.clone(), m, vrms });
            }
            Err(e) => println!("{name:<18} recon failed: {e}"),
        }
    }

    // ---- multi-scan stacking benchmark (F5) ----
    if params.n_scans > 1 {
        println!("\nstacking {} scans...", params.n_scans);
        let mut recons: Vec<Image> = Vec::new();
        for k in 0..params.n_scans {
            let path = if k == 0 { ser.clone() } else { dir.join(format!("synth_scan{k}.ser")) };
            let opts = pipeline::ReconOptions { verbose: false, ..Default::default() };
            match pipeline::reconstruct(&path, &opts) {
                Ok(rep) => recons.push(rep.output.image),
                Err(e) => println!("scan {k} recon failed: {e}"),
            }
        }
        for (label, flow) in [("Stack-no-flow", false), ("Stack-flow", true)] {
            // reference scan 0: the ground truth is the scan-0 sun
            if let Some(srep) = stack::stack_with_reference(&recons, flow, false, Some(0)) {
                let stem = label.to_lowercase();
                let mx = srep.image.max();
                output::write_png16(&dir.join(format!("{stem}_linear.png")), &srep.image, Some((0.0, mx))).unwrap();
                match metrics::evaluate(&srep.image, &gt_blurred) {
                    Some(m) => print_row(label, &Some(m), &None),
                    None => println!("{label:<18} eval failed"),
                }
            }
        }
    }

    // ---- JSON ledger ----
    if let Some(jp) = json {
        let mut lines = String::new();
        for r in &rows {
            if let Some(m) = &r.m {
                lines.push_str(&format!(
                    "{{\"seed\":{},\"variant\":\"{}\",\"psnr\":{:.3},\"ssim\":{:.5},\"psnr_limb\":{:.3},\"limb_sigma\":{:.3},\"flat_pct\":{:.3},\"band4\":{:.2},\"vrms\":{}}}\n",
                    args.seed, r.name, m.psnr_disk, m.ssim_disk, m.psnr_limb, m.limb_sigma, m.flat_pct,
                    m.band_snr[3],
                    r.vrms.map(|v| format!("{v:.4}")).unwrap_or("null".into()),
                ));
            }
        }
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(jp).unwrap();
        f.write_all(lines.as_bytes()).unwrap();
        println!("\nappended {} rows to {}", rows.len(), jp.display());
    }
}

fn print_header(vel: bool) {
    print!(
        "{:<18} {:>9} {:>7} {:>9} {:>6} {:>6} {:>6}",
        "variant", "PSNR", "SSIM", "PSNRlimb", "limbσ", "flat%", "band4"
    );
    if vel {
        print!(" {:>7}", "vRMS");
    }
    println!();
}

fn print_row(name: &str, m: &Option<metrics::EvalResult>, vrms: &Option<f64>) {
    match m {
        Some(m) => {
            print!(
                "{:<18} {:>7.2}dB {:>7.4} {:>7.2}dB {:>6.2} {:>6.2} {:>6.1}",
                name, m.psnr_disk, m.ssim_disk, m.psnr_limb, m.limb_sigma, m.flat_pct, m.band_snr[3]
            );
            if let Some(v) = vrms {
                print!(" {:>7.4}", v);
            }
            println!();
        }
        None => println!("{name:<18} eval failed"),
    }
}

/// Correlate estimated corrections against synthetic truth.
fn diagnostics(rep: &pipeline::ReconReport, truth: &synth::SynthTruth, dir: &Path) {
    // jitter: applied correction should equal -jitter up to a linear ramp
    let nall = rep.jitter_applied.len().min(truth.jitter.len());
    let idx: Vec<usize> = (0..nall).filter(|&i| rep.jitter_applied[i].abs() > 1e-9).collect();
    let idx = if idx.len() < 50 { (0..nall).collect::<Vec<_>>() } else { idx };
    let n = idx.len();
    let xs: Vec<f64> = idx.iter().map(|&i| i as f64).collect();
    let target: Vec<f64> = idx.iter().map(|&i| -truth.jitter[i]).collect();
    let resid: Vec<f64> = (0..n).map(|k| rep.jitter_applied[idx[k]] - target[k]).collect();
    let ws = vec![1.0; n];
    let line = mathutil::polyfit_robust(&xs, &resid, &ws, 1, 3).unwrap_or(vec![0.0, 0.0]);
    let after: Vec<f64> = (0..n).map(|i| resid[i] - mathutil::polyval(&line, xs[i])).collect();
    let rms_after = (after.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let rms_before = {
        let l2 = mathutil::polyfit_robust(&xs, &target, &ws, 1, 3).unwrap_or(vec![0.0, 0.0]);
        let t2: Vec<f64> = (0..n).map(|i| target[i] - mathutil::polyval(&l2, xs[i])).collect();
        (t2.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt()
    };
    println!("  [diag] jitter RMS: {:.3} px uncorrected -> {:.3} px residual", rms_before, rms_after);

    // flexure tracking (only meaningful when synth flexure is on)
    let ftrue_max = truth.flex.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
    if ftrue_max > 0.0 {
        let m = rep.flex.len().min(truth.flex.len());
        let fr: Vec<f64> = (0..m)
            .map(|i| {
                let te = truth.flex[i] - truth.flex.iter().sum::<f64>() / m as f64;
                rep.flex[i] - te
            })
            .collect();
        let frms = (fr.iter().map(|v| v * v).sum::<f64>() / m as f64).sqrt();
        println!("  [diag] flexure: true max {:.3} px, residual RMS {:.3} px", ftrue_max, frms);
    }

    // x-registration vs truth (only when synth x-jitter is on)
    let jx_max = truth.jitter_x.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
    if jx_max > 0.0 {
        // truth is in sun px; applied delta is in frames: delta_true = jx/step
        // compare after robust line removal on columns actually corrected
        let m2 = rep.xreg_applied.len().min(truth.jitter_x.len());
        let used: Vec<usize> = (0..m2).filter(|&i| rep.xreg_applied[i].abs() > 1e-9).collect();
        if used.len() > 50 {
            // infer step from the ratio via LS to avoid plumbing scan_step here
            let mut num = 0.0;
            let mut den = 0.0;
            for &i in &used {
                num += rep.xreg_applied[i] * truth.jitter_x[i];
                den += truth.jitter_x[i] * truth.jitter_x[i];
            }
            let scale = num / den.max(1e-12); // frames per sun px (~1/step)
            let rms_before = (used.iter().map(|&i| (truth.jitter_x[i] * scale).powi(2)).sum::<f64>()
                / used.len() as f64)
                .sqrt();
            let rms_after = (used
                .iter()
                .map(|&i| (rep.xreg_applied[i] - truth.jitter_x[i] * scale).powi(2))
                .sum::<f64>()
                / used.len() as f64)
                .sqrt();
            println!(
                "  [diag] x-jitter (frames): {:.3} uncorrected -> {:.3} residual (scale {:.2})",
                rms_before, rms_after, scale
            );
        }
    }

    // burst detection hit-rate vs truth
    let n_true_bursts = truth.burst_mask.iter().filter(|&&b| b).count();
    if n_true_bursts > 0 {
        let m2 = rep.burst_flags.len().min(truth.burst_mask.len());
        let mut tp = 0;
        let mut fp = 0;
        for i in 0..m2 {
            if rep.burst_flags[i] && truth.burst_mask[i] {
                tp += 1;
            }
            if rep.burst_flags[i] && !truth.burst_mask[i] {
                fp += 1;
            }
        }
        println!(
            "  [diag] bursts: {} true, {} detected (recall {:.2}, false-pos {})",
            n_true_bursts,
            tp + fp,
            tp as f64 / n_true_bursts as f64,
            fp
        );
        if std::env::var("GS_DEBUG").is_ok() {
            let fps: Vec<usize> = (0..m2).filter(|&i| rep.burst_flags[i] && !truth.burst_mask[i]).collect();
            let trs: Vec<usize> = (0..m2).filter(|&i| truth.burst_mask[i]).collect();
            eprintln!("  false-pos cols: {:?}", &fps[..fps.len().min(40)]);
            eprintln!("  true burst cols: {:?}", &trs[..trs.len().min(40)]);
        }
    }

    // transparency: correlation of estimated gains with truth
    let m = rep.column_gain.len().min(truth.transparency.len());
    let est: Vec<f64> = rep.column_gain[..m].to_vec();
    let tru: Vec<f64> = truth.transparency[..m].to_vec();
    let me = est.iter().sum::<f64>() / m as f64;
    let mt = tru.iter().sum::<f64>() / m as f64;
    let mut num = 0.0;
    let mut de = 0.0;
    let mut dt = 0.0;
    for i in 0..m {
        let a = est[i] - me;
        let b = tru[i] - mt;
        num += a * b;
        de += a * a;
        dt += b * b;
    }
    println!("  [diag] transparency corr: {:.3}", num / (de * dt).sqrt().max(1e-12));

    // dump for offline inspection
    let mut out = String::from("col,gain_est,transp_true,jit_applied,jit_true\n");
    for i in 0..m.min(rep.jitter_applied.len()) {
        out.push_str(&format!(
            "{},{:.5},{:.5},{:.4},{:.4}\n",
            i, rep.column_gain[i], truth.transparency[i], rep.jitter_applied[i], truth.jitter[i]
        ));
    }
    let _ = std::fs::write(dir.join("diag.csv"), out);
}

fn print_eval(name: &str, m: &metrics::EvalResult) {
    println!("{name}:");
    println!("  PSNR (disk interior): {:.2} dB", m.psnr_disk);
    println!("  SSIM (disk interior): {:.4}", m.ssim_disk);
    println!("  PSNR (limb annulus) : {:.2} dB", m.psnr_limb);
    println!("  radius ratio        : {:.4}", m.radius_ratio);
    println!("  limb sigma          : {:.2} px", m.limb_sigma);
    println!("  flatness            : {:.2} %", m.flat_pct);
    println!("  band SNR            : {:?} dB", m.band_snr.map(|b| (b * 10.0).round() / 10.0));
}
