//! GhostSun desktop app: GPU-accelerated (via wgpu) solar reconstruction
//! viewer and processor for macOS and Windows.
#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod focus;
mod gong;

use eframe::egui;
use ghostsun_core::image2d::Image;
use ghostsun_core::mathutil::percentile_f32;
use ghostsun_core::{orientation, output, pipeline, render};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

const ACCENT: egui::Color32 = egui::Color32::from_rgb(255, 140, 40);
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(160, 84, 20);

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 940.0])
            .with_min_inner_size([980.0, 640.0])
            .with_title("GhostSun")
            .with_icon(app_icon()),
        // wgpu selects Metal on macOS and Direct3D 12/Vulkan on Windows.
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("GhostSun", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

/// Generate a small native window/taskbar icon without a platform-specific
/// asset bundle. Keeping it in RGBA form lets eframe use the same icon on
/// macOS, Windows, and Linux.
fn app_icon() -> egui::IconData {
    const SIZE: u32 = 128;
    let mut rgba = vec![0; (SIZE * SIZE * 4) as usize];
    let px = 1.0 / SIZE as f32;

    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = (x as f32 + 0.5) / SIZE as f32 - 0.5;
            let dy = (y as f32 + 0.5) / SIZE as f32 - 0.5;
            let radius = (dx * dx + dy * dy).sqrt();
            let angle = dy.atan2(dx);

            let disk = ((0.285 - radius) / px).clamp(0.0, 1.0);
            let ray_band = ((radius - 0.345) / (2.0 * px)).clamp(0.0, 1.0)
                * ((0.465 - radius) / (2.0 * px)).clamp(0.0, 1.0);
            let ray_direction = (((angle * 8.0).cos() - 0.91) / 0.09).clamp(0.0, 1.0);
            let rays = ray_band * ray_direction;
            let alpha = disk.max(rays);

            if alpha > 0.0 {
                let glow = (1.0 - radius / 0.285).clamp(0.0, 1.0);
                let i = ((y * SIZE + x) * 4) as usize;
                rgba[i] = 255;
                rgba[i + 1] = (126.0 + 68.0 * glow) as u8;
                rgba[i + 2] = (28.0 + 32.0 * glow) as u8;
                rgba[i + 3] = (255.0 * alpha) as u8;
            }
        }
    }

    egui::IconData { rgba, width: SIZE, height: SIZE }
}

// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Display,
    Color,
    Velocity,
    Focus,
}

enum Job {
    Log(String),
    Done {
        report: Box<pipeline::ReconReport>,
        source_ser: PathBuf,
    },
    Failed(String),
    Colorized { rgb: Vec<u8>, w: usize, h: usize, seq: u64 },
    OrientationDone(Box<OrientationApplied>),
    OrientationFailed(String),
}

struct OrientationApplied {
    image: Image,
    before_demix: Option<Image>,
    velocity: Option<Image>,
    prep: Option<render::ColorizePrep>,
    matched: orientation::OrientationMatch,
    reference_filename: String,
    reference_url: String,
    delta_seconds: i64,
}

struct Loaded {
    image: Arc<Image>,
    before_demix: Option<Arc<Image>>,
    velocity: Option<Image>,
    prep: Option<render::ColorizePrep>,
    name: String,
    source_ser: Option<PathBuf>,
    orientation_note: Option<String>,
    orientation_reference_url: Option<String>,
}

struct App {
    loaded: Option<Loaded>,
    running: bool,
    orientation_running: bool,
    log: Vec<String>,
    rx: Receiver<Job>,
    tx: Sender<Job>,
    mode: ViewMode,
    texture: Option<egui::TextureHandle>,
    tex_mode: Option<ViewMode>,
    zoom: f32,
    pan: egui::Vec2,
    fit_requested: bool,
    prom_boost: f32,
    gamma: f32,
    color_dirty: bool,
    color_seq: u64,
    color_inflight: bool,
    color_cache: Option<(Vec<u8>, usize, usize)>,
    opt_deconv: bool,
    opt_denoise: bool,
    opt_motion_strength: f64,
    opt_column_demix_strength: f64,
    selected_ser: Option<std::path::PathBuf>,
    pending_open: Option<PathBuf>,
    show_before_demix: bool,
    focus: focus::FocusState,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        style(&cc.egui_ctx);
        let (tx, rx) = channel::<Job>();
        App {
            loaded: None,
            running: false,
            orientation_running: false,
            log: Vec::new(),
            rx,
            tx,
            mode: ViewMode::Display,
            texture: None,
            tex_mode: None,
            zoom: 0.25,
            pan: egui::Vec2::ZERO,
            fit_requested: true,
            prom_boost: 3.0,
            gamma: 0.7,
            color_dirty: true,
            color_seq: 0,
            color_inflight: false,
            color_cache: None,
            opt_deconv: false,
            opt_denoise: false,
            opt_motion_strength: 1.0,
            opt_column_demix_strength: 1.0,
            selected_ser: None,
            pending_open: std::env::args().nth(1).map(PathBuf::from),
            show_before_demix: false,
            focus: focus::FocusState::default(),
        }
    }

    fn open_file(&mut self, path: PathBuf) {
        if self.running || self.orientation_running {
            self.log
                .push("processing is already in progress; wait before opening another file".into());
            return;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "ser" => {
                self.selected_ser = Some(path.clone());
                self.log.clear();
                self.log.push(format!("selected {}", path.display()));
                self.log.push("review the Pipeline settings, then click Process".into());
            }
            "fits" | "fit" => match output::read_fits_f32(&path) {
                Ok(img) => {
                    self.selected_ser = None;
                    let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                    self.set_loaded(img, None, None, name, None);
                    self.log.push(format!("loaded {}", path.display()));
                }
                Err(e) => self.log.push(format!("FITS load failed: {e}")),
            },
            "png" => match output::read_png16(&path) {
                Ok(img) => {
                    self.selected_ser = None;
                    let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                    self.set_loaded(img, None, None, name, None);
                }
                Err(e) => self.log.push(format!("PNG load failed: {e}")),
            },
            _ => self.log.push(format!("unsupported file type: {ext}")),
        }
    }

    fn set_loaded(
        &mut self,
        image: Image,
        velocity: Option<Image>,
        before_demix: Option<Image>,
        name: String,
        source_ser: Option<PathBuf>,
    ) {
        let prep = render::prepare(&image);
        self.loaded = Some(Loaded {
            image: Arc::new(image),
            before_demix: before_demix.map(Arc::new),
            velocity,
            prep,
            name,
            source_ser,
            orientation_note: None,
            orientation_reference_url: None,
        });
        self.show_before_demix = false;
        self.texture = None;
        self.tex_mode = None;
        self.color_cache = None;
        self.color_dirty = true;
        self.fit_requested = true;
    }

    fn run_pipeline(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.running = true;
        self.log.clear();
        self.log.push(format!("processing {} ...", path.display()));
        let tx = self.tx.clone();
        let tx_log = self.tx.clone();
        let egui_ctx = ctx.clone();
        let egui_ctx2 = ctx.clone();
        let mut tune = pipeline::TuneParams::default();
        tune.motion_strength = self.opt_motion_strength;
        tune.column_demix_strength = self.opt_column_demix_strength;
        let opts = pipeline::ReconOptions {
            verbose: false,
            deconv: self.opt_deconv,
            denoise: self.opt_denoise,
            tune,
            progress: Some(Arc::new(move |m: &str| {
                let _ = tx_log.send(Job::Log(m.to_string()));
                egui_ctx.request_repaint();
            })),
            ..Default::default()
        };
        std::thread::spawn(move || {
            match pipeline::reconstruct(&path, &opts) {
                Ok(rep) => {
                    let _ = tx.send(Job::Done {
                        report: Box::new(rep),
                        source_ser: path,
                    });
                }
                Err(e) => {
                    let _ = tx.send(Job::Failed(e));
                }
            }
            egui_ctx2.request_repaint();
        });
    }

    fn kick_gong_orientation(&mut self, ctx: &egui::Context) {
        let Some(loaded) = &self.loaded else { return };
        let Some(source_ser) = loaded.source_ser.clone() else {
            self.log.push(
                "GONG orientation requires the original SER so its UTC timestamp is available"
                    .into(),
            );
            return;
        };
        let Some(prep) = &loaded.prep else {
            self.log.push("cannot orient: solar disk geometry was not detected".into());
            return;
        };
        if self.orientation_running || loaded.orientation_note.is_some() {
            return;
        }

        self.orientation_running = true;
        self.log
            .push("GONG orientation: finding the nearest calibrated H-alpha reference ...".into());
        let disk = prep.disk.clone();
        let image = Arc::clone(&loaded.image);
        let before_demix = loaded.before_demix.clone();
        let velocity = loaded.velocity.clone();
        let tx = self.tx.clone();
        let egui_ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = (|| -> Result<OrientationApplied, String> {
                let reference = gong::download_nearest(&source_ser)?;
                let _ = tx.send(Job::Log(format!(
                    "GONG reference: {} ({} s from SER UTC)",
                    reference.filename, reference.delta_seconds
                )));
                let matched = orientation::match_to_reference(
                    &image,
                    &disk,
                    &reference.image,
                    &reference.disk,
                )?;
                let _ = tx.send(Job::Log(format!(
                    "feature match: {}rotation {:+.1} deg, NCC {:.3}, pose margin {:.3}",
                    if matched.mirrored { "horizontal mirror + " } else { "" },
                    matched.rotation_deg,
                    matched.score,
                    matched.confidence_margin(),
                )));
                if !matched.is_confident() {
                    return Err(format!(
                        "GONG feature match was not confident enough (NCC {:.3}, pose margin {:.3}); image left unchanged",
                        matched.score,
                        matched.confidence_margin()
                    ));
                }

                let transformed = orientation::apply_orientation(
                    &image,
                    &disk,
                    matched.mirrored,
                    matched.rotation_deg,
                );
                let transformed_before = before_demix.as_deref().map(|before| {
                    orientation::apply_orientation(
                        before,
                        &disk,
                        matched.mirrored,
                        matched.rotation_deg,
                    )
                });
                let transformed_velocity = velocity.as_ref().map(|map| {
                    orientation::apply_orientation(
                        map,
                        &disk,
                        matched.mirrored,
                        matched.rotation_deg,
                    )
                });
                let transformed_prep = render::prepare(&transformed);
                Ok(OrientationApplied {
                    image: transformed,
                    before_demix: transformed_before,
                    velocity: transformed_velocity,
                    prep: transformed_prep,
                    matched,
                    reference_filename: reference.filename,
                    reference_url: reference.url,
                    delta_seconds: reference.delta_seconds,
                })
            })();

            match result {
                Ok(applied) => {
                    let _ = tx.send(Job::OrientationDone(Box::new(applied)));
                }
                Err(error) => {
                    let _ = tx.send(Job::OrientationFailed(error));
                }
            }
            egui_ctx.request_repaint();
        });
    }

    fn kick_colorize(&mut self, ctx: &egui::Context) {
        let Some(loaded) = &self.loaded else { return };
        let Some(prep) = &loaded.prep else { return };
        if self.color_inflight {
            return;
        }
        self.color_inflight = true;
        self.color_dirty = false;
        self.color_seq += 1;
        let seq = self.color_seq;
        let img = Arc::clone(&loaded.image);
        let prep = prep.clone();
        let copts = render::ColorizeOptions {
            prom_boost: self.prom_boost as f64,
            gamma: self.gamma as f64,
        };
        let tx = self.tx.clone();
        let egui_ctx = ctx.clone();
        std::thread::spawn(move || {
            let (w, h, rgb) = render::render_with(&img, &prep, &copts);
            let _ = tx.send(Job::Colorized { rgb, w, h, seq });
            egui_ctx.request_repaint();
        });
    }

    fn pump_jobs(&mut self) {
        let mut msgs = Vec::new();
        while let Ok(m) = self.rx.try_recv() {
            msgs.push(m);
        }
        for m in msgs {
            match m {
                Job::Log(s) => {
                    self.log.push(s);
                    if self.log.len() > 400 {
                        self.log.drain(..100);
                    }
                }
                Job::Done { report, source_ser } => {
                    self.running = false;
                    self.log.push("done.".into());
                    let rep = *report;
                    self.set_loaded(
                        rep.output.image,
                        rep.velocity,
                        rep.demix_before,
                        "reconstruction".into(),
                        Some(source_ser),
                    );
                }
                Job::Failed(e) => {
                    self.running = false;
                    self.log.push(format!("FAILED: {e}"));
                }
                Job::Colorized { rgb, w, h, seq } => {
                    self.color_inflight = false;
                    if seq == self.color_seq {
                        self.color_cache = Some((rgb, w, h));
                        if self.mode == ViewMode::Color {
                            self.texture = None;
                        }
                    } else {
                        self.color_dirty = true;
                    }
                }
                Job::OrientationDone(applied) => {
                    self.orientation_running = false;
                    let applied = *applied;
                    if let Some(loaded) = &mut self.loaded {
                        loaded.image = Arc::new(applied.image);
                        loaded.before_demix = applied.before_demix.map(Arc::new);
                        loaded.velocity = applied.velocity;
                        loaded.prep = applied.prep;
                        loaded.orientation_note = Some(format!(
                            "GONG: north up, east left · {}{:+.1}° · NCC {:.3}",
                            if applied.matched.mirrored { "mirror + " } else { "" },
                            applied.matched.rotation_deg,
                            applied.matched.score,
                        ));
                        loaded.orientation_reference_url = Some(applied.reference_url);
                    }
                    self.texture = None;
                    self.tex_mode = None;
                    self.color_cache = None;
                    self.color_dirty = true;
                    self.fit_requested = true;
                    self.log.push(format!(
                        "orientation applied from {} ({} s from acquisition): solar north is up, east is left",
                        applied.reference_filename, applied.delta_seconds
                    ));
                }
                Job::OrientationFailed(error) => {
                    self.orientation_running = false;
                    self.log.push(format!("GONG orientation failed: {error}"));
                }
            }
        }
    }

    fn build_texture(&mut self, ctx: &egui::Context) {
        let Some(loaded) = &self.loaded else { return };
        if self.texture.is_some() && self.tex_mode == Some(self.mode) {
            return;
        }
        let color_image = match self.mode {
            ViewMode::Display => {
                let img = if self.show_before_demix {
                    loaded.before_demix.as_deref().unwrap_or(loaded.image.as_ref())
                } else {
                    loaded.image.as_ref()
                };
                let lo = percentile_f32(&img.data, 0.05);
                let hi = percentile_f32(&img.data, 99.95).max(lo + 1e-3);
                gray_to_color(img, lo, hi)
            }
            ViewMode::Velocity => {
                if let Some(v) = &loaded.velocity {
                    velocity_to_color(v)
                } else {
                    return;
                }
            }
            ViewMode::Color => {
                if let Some((rgb, w, h)) = &self.color_cache {
                    egui::ColorImage::from_rgb([*w, *h], rgb)
                } else {
                    return;
                }
            }
            ViewMode::Focus => return, // Focus mode renders its own live view
        };
        self.texture = Some(ctx.load_texture(
            "main",
            color_image,
            egui::TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                minification: egui::TextureFilter::Linear,
                ..Default::default()
            },
        ));
        self.tex_mode = Some(self.mode);
    }
}

fn gray_to_color(img: &Image, lo: f32, hi: f32) -> egui::ColorImage {
    let scale = 255.0 / (hi - lo);
    let mut px = Vec::with_capacity(img.w * img.h);
    for &v in &img.data {
        let g = ((v - lo) * scale).clamp(0.0, 255.0) as u8;
        px.push(egui::Color32::from_gray(g));
    }
    egui::ColorImage { size: [img.w, img.h], pixels: px }
}

fn velocity_to_color(v: &Image) -> egui::ColorImage {
    // normalize over measured (nonzero) pixels only — the background is
    // masked to exactly zero and must not set the scale
    let abs: Vec<f32> = v.data.iter().filter(|x| **x != 0.0).map(|x| x.abs()).collect();
    let mag = if abs.is_empty() { 1.0 } else { percentile_f32(&abs, 98.0).max(1e-3) };
    let mut px = Vec::with_capacity(v.w * v.h);
    for &val in &v.data {
        let t = (val / mag).clamp(-1.0, 1.0);
        let (r, g, b) = if t < 0.0 {
            let a = -t;
            ((255.0 * (1.0 - a)) as u8, (255.0 * (1.0 - a * 0.6)) as u8, 255u8)
        } else {
            (255u8, (255.0 * (1.0 - t * 0.6)) as u8, (255.0 * (1.0 - t)) as u8)
        };
        px.push(egui::Color32::from_rgb(r, g, b));
    }
    egui::ColorImage { size: [v.w, v.h], pixels: px }
}

fn style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = egui::Color32::from_rgb(16, 14, 13);
    v.window_fill = egui::Color32::from_rgb(22, 19, 17);
    v.extreme_bg_color = egui::Color32::from_rgb(10, 9, 8);
    v.faint_bg_color = egui::Color32::from_rgb(30, 26, 23);
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT);
    v.hyperlink_color = ACCENT;
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, ACCENT_DIM);
    v.widgets.active.bg_stroke = egui::Stroke::new(1.5_f32, ACCENT);
    ctx.set_visuals(v);
    let mut st = (*ctx.style()).clone();
    st.spacing.item_spacing = egui::vec2(10.0, 8.0);
    st.spacing.button_padding = egui::vec2(12.0, 6.0);
    st.spacing.slider_width = 150.0;
    ctx.set_style(st);
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump_jobs();
        if self.mode == ViewMode::Focus {
            self.focus.poll(ctx);
        }

        // open a file passed on the command line (once)
        if let Some(path) = self.pending_open.take() {
            self.open_file(path);
        }
        // drag & drop
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()
        });
        if let Some(p) = dropped.into_iter().next() {
            self.open_file(p);
        }

        if self.mode == ViewMode::Color
            && self.color_dirty
            && !self.running
            && !self.orientation_running
        {
            self.kick_colorize(ctx);
        }

        egui::TopBottomPanel::top("top").exact_height(48.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("☀ GhostSun").size(22.0).strong().color(ACCENT));
                ui.add_space(16.0);
                if ui.button("Open…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Solar data", &["ser", "fits", "fit", "png"])
                        .pick_file()
                    {
                        self.open_file(path);
                    }
                }
                if self.running {
                    ui.add(egui::Spinner::new().color(ACCENT));
                    ui.label(egui::RichText::new("processing…").italics());
                } else if self.orientation_running {
                    ui.add(egui::Spinner::new().color(ACCENT));
                    ui.label(egui::RichText::new("matching GONG features…").italics());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(6.0);
                    let has_vel =
                        self.loaded.as_ref().map(|l| l.velocity.is_some()).unwrap_or(false);
                    let mut mode = self.mode;
                    ui.selectable_value(&mut mode, ViewMode::Focus, "Focus");
                    ui.selectable_value(&mut mode, ViewMode::Color, "Hα Color");
                    if has_vel {
                        ui.selectable_value(&mut mode, ViewMode::Velocity, "Doppler");
                    }
                    ui.selectable_value(&mut mode, ViewMode::Display, "Grayscale");
                    if mode != self.mode {
                        // leaving Focus: stop the camera stream
                        if self.mode == ViewMode::Focus {
                            self.focus.stop();
                        }
                        // entering Focus: discover cameras once
                        if mode == ViewMode::Focus && self.focus.cameras.is_empty() {
                            self.focus.refresh_cameras();
                        }
                        self.mode = mode;
                        self.texture = None;
                        self.tex_mode = None;
                    }
                });
            });
        });

        egui::SidePanel::left("side")
            .resizable(true)
            .default_width(320.0)
            .width_range(260.0..=520.0)
            .show(ctx, |ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            if self.mode == ViewMode::Focus {
                self.focus.controls_ui(ui, ctx);
                return;
            }
            ui.add_space(8.0);
            if let Some(loaded) = &self.loaded {
                egui::Frame::group(ui.style()).fill(ui.visuals().faint_bg_color).show(ui, |ui| {
                    ui.add(egui::Label::new(egui::RichText::new(&loaded.name).strong()).wrap());
                    ui.label(format!("{} × {} px", loaded.image.w, loaded.image.h));
                    if let Some(prep) = &loaded.prep {
                        ui.label(format!("disk radius {:.0} px", prep.disk.r));
                    }
                    if let Some(note) = &loaded.orientation_note {
                        ui.label(egui::RichText::new(note).color(ACCENT));
                    }
                });
                ui.add_space(6.0);
            }

            ui.heading("Solar orientation");
            ui.label(
                egui::RichText::new(
                    "Feature-match the SER time against calibrated GONG H-alpha.\nTarget: north up, east left.",
                )
                .small()
                .weak(),
            );
            let orientation_applied = self
                .loaded
                .as_ref()
                .and_then(|loaded| loaded.orientation_note.as_ref())
                .is_some();
            let has_source_ser = self
                .loaded
                .as_ref()
                .and_then(|loaded| loaded.source_ser.as_ref())
                .is_some();
            if orientation_applied {
                if let Some(url) = self
                    .loaded
                    .as_ref()
                    .and_then(|loaded| loaded.orientation_reference_url.as_ref())
                {
                    ui.hyperlink_to("Open matched GONG reference", url);
                }
                ui.label(
                    egui::RichText::new("Reprocess the SER to restore the native scan pose.")
                        .small()
                        .weak(),
                );
            } else {
                let orient = egui::Button::new(
                    egui::RichText::new("Orient from GONG")
                        .strong()
                        .color(egui::Color32::WHITE),
                )
                .fill(ACCENT_DIM);
                if ui
                    .add_enabled(
                        has_source_ser && !self.running && !self.orientation_running,
                        orient,
                    )
                    .clicked()
                {
                    self.kick_gong_orientation(ctx);
                }
                if self.loaded.is_some() && !has_source_ser {
                    ui.label(
                        egui::RichText::new(
                            "Available after processing a timestamped .ser scan.",
                        )
                        .small()
                        .weak(),
                    );
                }
            }
            ui.add_space(10.0);

            ui.heading("Hα rendering");
            if self.loaded.as_ref().and_then(|l| l.before_demix.as_ref()).is_some() {
                ui.heading("Column-state comparison");
                let changed = ui
                    .checkbox(&mut self.show_before_demix, "Show before demixing")
                    .changed();
                ui.label(
                    egui::RichText::new(if self.show_before_demix {
                        "BEFORE: residual column pattern"
                    } else {
                        "AFTER: gain / offset / shifts / blur corrected"
                    })
                    .small()
                    .weak(),
                );
                if changed {
                    self.mode = ViewMode::Display;
                    self.texture = None;
                    self.tex_mode = None;
                }
                ui.add_space(10.0);
            }

            ui.spacing_mut().slider_width = (ui.available_width() - 120.0).max(120.0);
            ui.label("prominence boost");
            let r1 = ui.add(
                egui::Slider::new(&mut self.prom_boost, 1.0..=10.0).fixed_decimals(1),
            );
            ui.label("disk gamma");
            let r2 = ui.add(
                egui::Slider::new(&mut self.gamma, 0.4..=1.2).fixed_decimals(2),
            );
            if r1.changed() || r2.changed() {
                self.color_dirty = true;
            }
            ui.add_space(10.0);

            ui.heading("Pipeline");
            ui.label(
                egui::RichText::new("Select a .ser scan, check these settings,\nthen click Process.")
                    .small()
                    .weak(),
            );
            if let Some(ser) = &self.selected_ser {
                let name = ser.file_name().unwrap_or_default().to_string_lossy();
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!("Selected: {name}")).strong().color(ACCENT)
                    )
                    .wrap(),
                );
            }
            ui.checkbox(&mut self.opt_deconv, "PSF deconvolution");
            ui.checkbox(&mut self.opt_denoise, "wavelet denoise");
            ui.label("motion registration strength");
            ui.add(
                egui::Slider::new(&mut self.opt_motion_strength, 0.0..=1.5)
                    .fixed_decimals(2),
            );
            ui.label(
                egui::RichText::new("0 = off, 1 = measured, above 1 = more aggressive")
                    .small()
                    .weak(),
            );
            ui.label("column correction strength");
            ui.add(
                egui::Slider::new(&mut self.opt_column_demix_strength, 0.0..=1.0)
                    .fixed_decimals(2),
            );
            ui.label(
                egui::RichText::new("0 = off, 1 = full; detection is automatic per scan")
                    .small()
                    .weak(),
            );
            if let Some(ser) = self.selected_ser.clone() {
                let process = egui::Button::new(
                    egui::RichText::new("Process").strong().color(egui::Color32::WHITE)
                )
                .fill(ACCENT_DIM);
                if ui
                    .add_enabled(!self.running && !self.orientation_running, process)
                    .clicked()
                {
                    self.run_pipeline(ser, ctx);
                }
            } else {
                ui.label(
                    egui::RichText::new("Open a .ser scan to enable processing;\n.fits / .png files are view-only.")
                        .small()
                        .weak(),
                );
            }
            ui.add_space(10.0);

            if self.loaded.is_some() && ui.button("Save colorized PNG…").clicked() {
                if let Some((rgb, w, h)) = &self.color_cache {
                    if let Some(path) =
                        rfd::FileDialog::new().set_file_name("ghostsun_color.png").save_file()
                    {
                        let _ = output::write_png_rgb(&path, *w, *h, rgb);
                        self.log.push(format!("saved {}", path.display()));
                    }
                } else {
                    self.log.push("no color render yet — switch to Hα Color first".into());
                }
            }

            ui.add_space(8.0);
            ui.separator();
            ui.heading("Log");
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                    for line in &self.log {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(line).size(11.0).monospace().weak(),
                            )
                            .wrap(),
                        );
                    }
                });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::from_rgb(8, 7, 6)))
            .show(ctx, |ui| {
                if self.mode == ViewMode::Focus {
                    self.focus.view_ui(ui, ctx);
                    return;
                }
                self.build_texture(ctx);
                let avail = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());

                if let Some(tex) = &self.texture {
                    let tex_size = tex.size_vec2();
                    if self.fit_requested {
                        self.zoom = (avail.x / tex_size.x).min(avail.y / tex_size.y) * 0.96;
                        self.pan = egui::Vec2::ZERO;
                        self.fit_requested = false;
                    }
                    if response.hovered() {
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                        if scroll.abs() > 0.0 {
                            let old = self.zoom;
                            self.zoom = (self.zoom * (1.0 + scroll * 0.002)).clamp(0.02, 12.0);
                            if let Some(pos) = response.hover_pos() {
                                let center = rect.center() + self.pan;
                                let d = pos - center;
                                self.pan += d - d * (self.zoom / old);
                            }
                        }
                    }
                    if response.dragged() {
                        self.pan += response.drag_delta();
                    }
                    if response.double_clicked() {
                        self.fit_requested = true;
                    }

                    let size = tex_size * self.zoom;
                    let center = rect.center() + self.pan;
                    let img_rect = egui::Rect::from_center_size(center, size);
                    let painter = ui.painter_at(rect);
                    painter.image(
                        tex.id(),
                        img_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );

                    let mut status = format!("{:.0}%", self.zoom * 100.0);
                    if let (Some(pos), Some(loaded)) = (response.hover_pos(), &self.loaded) {
                        let u = (pos - img_rect.min) / size;
                        let px = u.x * tex_size.x;
                        let py = u.y * tex_size.y;
                        if px >= 0.0 && py >= 0.0 && px < tex_size.x && py < tex_size.y {
                            let inspect = if self.show_before_demix {
                                loaded
                                    .before_demix
                                    .as_deref()
                                    .unwrap_or(loaded.image.as_ref())
                            } else {
                                loaded.image.as_ref()
                            };
                            let v = inspect.at(px as usize, py as usize);
                            status += &format!("   ({:.0}, {:.0})  I = {:.0}", px, py, v);
                            if let Some(prep) = &loaded.prep {
                                let dx = px as f64 - prep.disk.xc;
                                let dy = py as f64 - prep.disk.yc;
                                let rr = (dx * dx + dy * dy).sqrt() / prep.disk.r;
                                status += &format!("   r/R = {rr:.3}");
                            }
                        }
                    }
                    painter.text(
                        rect.left_bottom() + egui::vec2(10.0, -10.0),
                        egui::Align2::LEFT_BOTTOM,
                        status,
                        egui::FontId::monospace(12.0),
                        egui::Color32::from_gray(180),
                    );
                } else {
                    ui.painter_at(rect).text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        if self.running {
                            "reconstructing…"
                        } else if self.orientation_running {
                            "matching solar features to GONG…"
                        } else if self.mode == ViewMode::Color && self.loaded.is_some() {
                            "rendering color…"
                        } else if self.selected_ser.is_some() {
                            "Scan selected — review settings and click Process"
                        } else {
                            "Open a .ser scan or .fits reconstruction"
                        },
                        egui::FontId::proportional(18.0),
                        egui::Color32::from_gray(120),
                    );
                }
            });
    }
}
