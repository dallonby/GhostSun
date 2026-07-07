//! GhostSun desktop app: Metal-accelerated (via wgpu) solar reconstruction
//! viewer and processor for Apple Silicon and beyond.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use ghostsun_core::image2d::Image;
use ghostsun_core::mathutil::percentile_f32;
use ghostsun_core::{output, pipeline, render};
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
            .with_title("GhostSun"),
        renderer: eframe::Renderer::Wgpu, // Metal on Apple Silicon
        ..Default::default()
    };
    eframe::run_native("GhostSun", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Display,
    Color,
    Velocity,
}

enum Job {
    Log(String),
    Done(Box<pipeline::ReconReport>),
    Failed(String),
    Colorized { rgb: Vec<u8>, w: usize, h: usize, seq: u64 },
}

struct Loaded {
    image: Arc<Image>,
    velocity: Option<Image>,
    prep: Option<render::ColorizePrep>,
    name: String,
}

struct App {
    loaded: Option<Loaded>,
    running: bool,
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
    pending_open: Option<PathBuf>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        style(&cc.egui_ctx);
        let (tx, rx) = channel::<Job>();
        App {
            loaded: None,
            running: false,
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
            pending_open: std::env::args().nth(1).map(PathBuf::from),
        }
    }

    fn open_file(&mut self, path: PathBuf, ctx: &egui::Context) {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "ser" => self.run_pipeline(path, ctx),
            "fits" | "fit" => match output::read_fits_f32(&path) {
                Ok(img) => {
                    let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                    self.set_loaded(img, None, name);
                    self.log.push(format!("loaded {}", path.display()));
                }
                Err(e) => self.log.push(format!("FITS load failed: {e}")),
            },
            "png" => match output::read_png16(&path) {
                Ok(img) => {
                    let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                    self.set_loaded(img, None, name);
                }
                Err(e) => self.log.push(format!("PNG load failed: {e}")),
            },
            _ => self.log.push(format!("unsupported file type: {ext}")),
        }
    }

    fn set_loaded(&mut self, image: Image, velocity: Option<Image>, name: String) {
        let prep = render::prepare(&image);
        self.loaded = Some(Loaded { image: Arc::new(image), velocity, prep, name });
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
        let opts = pipeline::ReconOptions {
            verbose: false,
            deconv: self.opt_deconv,
            denoise: self.opt_denoise,
            progress: Some(Arc::new(move |m: &str| {
                let _ = tx_log.send(Job::Log(m.to_string()));
                egui_ctx.request_repaint();
            })),
            ..Default::default()
        };
        std::thread::spawn(move || {
            match pipeline::reconstruct(&path, &opts) {
                Ok(rep) => {
                    let _ = tx.send(Job::Done(Box::new(rep)));
                }
                Err(e) => {
                    let _ = tx.send(Job::Failed(e));
                }
            }
            egui_ctx2.request_repaint();
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
                Job::Done(rep) => {
                    self.running = false;
                    self.log.push("done.".into());
                    let rep = *rep;
                    self.set_loaded(rep.output.image, rep.velocity, "reconstruction".into());
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
                let img = &loaded.image;
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
    let abs: Vec<f32> = v.data.iter().map(|x| x.abs()).collect();
    let mag = percentile_f32(&abs, 99.0).max(1e-3);
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
    v.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT_DIM);
    v.widgets.active.bg_stroke = egui::Stroke::new(1.5, ACCENT);
    ctx.set_visuals(v);
    let mut st = (*ctx.style()).clone();
    st.spacing.item_spacing = egui::vec2(10.0, 8.0);
    st.spacing.button_padding = egui::vec2(12.0, 6.0);
    st.spacing.slider_width = 170.0;
    ctx.set_style(st);
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump_jobs();

        // open a file passed on the command line (once)
        if let Some(path) = self.pending_open.take() {
            self.open_file(path, ctx);
        }
        // drag & drop
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()
        });
        if let Some(p) = dropped.into_iter().next() {
            self.open_file(p, ctx);
        }

        if self.mode == ViewMode::Color && self.color_dirty && !self.running {
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
                        self.open_file(path, ctx);
                    }
                }
                if self.running {
                    ui.add(egui::Spinner::new().color(ACCENT));
                    ui.label(egui::RichText::new("processing…").italics());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(6.0);
                    let has_vel =
                        self.loaded.as_ref().map(|l| l.velocity.is_some()).unwrap_or(false);
                    let mut mode = self.mode;
                    ui.selectable_value(&mut mode, ViewMode::Color, "Hα Color");
                    if has_vel {
                        ui.selectable_value(&mut mode, ViewMode::Velocity, "Doppler");
                    }
                    ui.selectable_value(&mut mode, ViewMode::Display, "Grayscale");
                    if mode != self.mode {
                        self.mode = mode;
                        self.texture = None;
                        self.tex_mode = None;
                    }
                });
            });
        });

        egui::SidePanel::left("side").exact_width(300.0).show(ctx, |ui| {
            ui.add_space(8.0);
            if let Some(loaded) = &self.loaded {
                egui::Frame::group(ui.style()).fill(ui.visuals().faint_bg_color).show(ui, |ui| {
                    ui.label(egui::RichText::new(&loaded.name).strong());
                    ui.label(format!("{} × {} px", loaded.image.w, loaded.image.h));
                    if let Some(prep) = &loaded.prep {
                        ui.label(format!("disk radius {:.0} px", prep.disk.r));
                    }
                });
                ui.add_space(6.0);
            }

            ui.heading("Hα rendering");
            let r1 = ui.add(
                egui::Slider::new(&mut self.prom_boost, 1.0..=10.0)
                    .text("prominence boost")
                    .fixed_decimals(1),
            );
            let r2 = ui.add(
                egui::Slider::new(&mut self.gamma, 0.4..=1.2).text("disk gamma").fixed_decimals(2),
            );
            if r1.changed() || r2.changed() {
                self.color_dirty = true;
            }
            ui.add_space(10.0);

            ui.heading("Pipeline");
            ui.checkbox(&mut self.opt_deconv, "PSF deconvolution");
            ui.checkbox(&mut self.opt_denoise, "wavelet denoise");
            ui.label(
                egui::RichText::new("Open a .ser scan to reconstruct;\n.fits / .png to view.")
                    .small()
                    .weak(),
            );
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
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for line in &self.log {
                    ui.label(egui::RichText::new(line).small().monospace());
                }
            });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::from_rgb(8, 7, 6)))
            .show(ctx, |ui| {
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
                            let v = loaded.image.at(px as usize, py as usize);
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
                        } else if self.mode == ViewMode::Color && self.loaded.is_some() {
                            "rendering color…"
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
