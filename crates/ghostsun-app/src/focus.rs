//! Live focus assistant for a spectroheliograph.
//!
//! Each camera frame carries two perpendicular families of dark lines, which
//! measure two different focus problems:
//!   * **Spectral absorption lines** run along the slit (⊥ dispersion). Their
//!     width is the *spectral* focus (camera-side wavelength resolution).
//!   * **Slit jaw / dust / defect lines** run along the dispersion axis
//!     (⊥ slit). Their width is the *spatial/slit* focus (telescope-on-slit +
//!     spectrograph imaging), and they are essentially always present.
//!
//! Averaging the frame along one axis cancels the lines parallel to that axis
//! and preserves the perpendicular family, so the two are measured cleanly and
//! independently every frame with the *same* core line fitter the
//! reconstruction uses. Both FWHMs are shown at once; a dispersion-axis toggle
//! only decides which readout is labelled "spectral" vs "slit". Backend-agnostic
//! via the `ghostsun_camera::Camera` trait (ToupTek / ZWO / synthetic).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;

use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Text, VLine};

use ghostsun_camera::{enumerate_all, open, CameraInfo, Roi};
use ghostsun_core::linefit::fit_lines_1d;
use ghostsun_core::lines::{calibrate, geometric_dispersion, identify, Calibration, LabeledLine};

const STRIP_W: usize = 1200;
const STRIP_H: usize = 280;
const HISTORY: usize = 600;
const DEPTH_GATE: f64 = 0.03;

const SPECTRAL_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 210, 255);
const SLIT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 170, 90);

#[derive(Clone, Copy)]
pub struct Fit {
    pub fwhm: f64,
    pub depth: f64,
    pub center: f64,
    pub sigma: f64,
    pub continuum: f64,
}

impl Fit {
    fn from(l: ghostsun_core::linefit::LineFit1d) -> Fit {
        Fit {
            fwhm: l.fwhm,
            depth: l.depth,
            center: l.center,
            sigma: l.sigma,
            continuum: l.continuum,
        }
    }
}

/// Which image axis carries the dispersion (wavelength). The spectral lines run
/// perpendicular to it; this only assigns the "spectral" vs "slit" labels.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DispAxis {
    Vertical,
    Horizontal,
}

/// How the reported spectral line is chosen from the detected candidates.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LineMode {
    /// Sharpest line — the best focus reference.
    Narrowest,
    /// Strongest line.
    Deepest,
    /// The line nearest a user-clicked position.
    Manual,
}

/// Pick one line from the candidates per the mode (Manual uses `picked`).
fn choose(lines: &[Fit], mode: LineMode, picked: Option<f64>) -> Option<Fit> {
    match mode {
        LineMode::Narrowest => {
            lines.iter().min_by(|a, b| a.fwhm.partial_cmp(&b.fwhm).unwrap()).copied()
        }
        LineMode::Deepest => {
            lines.iter().max_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap()).copied()
        }
        LineMode::Manual => picked.and_then(|pc| {
            lines
                .iter()
                .min_by(|a, b| {
                    (a.center - pc).abs().partial_cmp(&(b.center - pc).abs()).unwrap()
                })
                .copied()
        }),
    }
}

/// Per-axis measurement. `along_x` collapses rows → a horizontal profile whose
/// dips are **vertical** lines; `along_y` collapses columns → a vertical profile
/// whose dips are **horizontal** lines.
pub struct FocusUpdate {
    pub strip: Vec<u8>,
    pub strip_w: usize,
    pub strip_h: usize,
    pub prof_x: Vec<f32>,
    pub prof_y: Vec<f32>,
    pub lines_x: Vec<Fit>, // vertical-line candidates
    pub lines_y: Vec<Fit>, // horizontal-line candidates
    pub mean: f32,
    pub full_w: usize,
    pub full_h: usize,
    pub cur_exposure: Option<u32>,
    pub cur_gain: Option<u16>,
}

enum FocusMsg {
    Frame(Box<FocusUpdate>),
    Error(String),
}

enum FocusCmd {
    Exposure(u32),
    Gain(u16),
    AutoExposure(bool),
}

/// Min-hold + rolling history for one measured axis.
#[derive(Default)]
struct Track {
    min_hold: f64,
    history: VecDeque<f64>,
}

impl Track {
    fn new() -> Track {
        Track { min_hold: f64::INFINITY, history: VecDeque::with_capacity(HISTORY) }
    }
    fn push(&mut self, fit: &Option<Fit>) {
        if let Some(f) = fit {
            if f.depth > DEPTH_GATE {
                self.min_hold = self.min_hold.min(f.fwhm);
                if self.history.len() >= HISTORY {
                    self.history.pop_front();
                }
                self.history.push_back(f.fwhm);
            }
        }
    }
    fn reset(&mut self) {
        self.min_hold = f64::INFINITY;
        self.history.clear();
    }
}

pub struct FocusState {
    pub cameras: Vec<CameraInfo>,
    pub selected: usize,
    pub streaming: bool,
    pub exposure_us: u32,
    pub gain: u16,
    pub auto_exposure: bool,
    pub dispersion: DispAxis,
    pub dispersion_a_per_px: f64,
    pub line_mode: LineMode,
    pub picked_center: Option<f64>,
    // Spectral identification (assume sunlight).
    pub identify_lines: bool,
    pub grating_l_mm: f64,
    pub order: u32,
    pub focal_len_mm: f64,
    pub pixel_um: f64,
    pub central_wavelength: f64,
    calibration: Option<Calibration>,
    labels: Vec<LabeledLine>,
    pub status: String,
    sel_spectral: Option<Fit>,
    sel_slit: Option<Fit>,
    track_x: Track, // vertical lines
    track_y: Track, // horizontal lines
    last: Option<FocusUpdate>,
    tex: Option<egui::TextureHandle>,
    rx: Option<Receiver<FocusMsg>>,
    cmd: Option<Sender<FocusCmd>>,
    stop: Option<Arc<AtomicBool>>,
    handle: Option<JoinHandle<()>>,
}

impl Default for FocusState {
    fn default() -> Self {
        FocusState {
            cameras: Vec::new(),
            selected: 0,
            streaming: false,
            exposure_us: 10_000,
            gain: 200,
            auto_exposure: false,
            dispersion: DispAxis::Vertical,
            dispersion_a_per_px: 0.085,
            line_mode: LineMode::Narrowest,
            picked_center: None,
            identify_lines: false,
            grating_l_mm: 2400.0,
            order: 1,
            focal_len_mm: 125.0,
            pixel_um: 2.0,
            central_wavelength: 6562.79,
            calibration: None,
            labels: Vec::new(),
            status: String::new(),
            sel_spectral: None,
            sel_slit: None,
            track_x: Track::new(),
            track_y: Track::new(),
            last: None,
            tex: None,
            rx: None,
            cmd: None,
            stop: None,
            handle: None,
        }
    }
}

impl FocusState {
    pub fn refresh_cameras(&mut self) {
        self.cameras = enumerate_all();
        if self.selected >= self.cameras.len() {
            self.selected = 0;
        }
        let hardware = self.cameras.iter()
            .filter(|c| c.backend != ghostsun_camera::Backend::Synth)
            .count();
        self.status = if hardware > 0 {
            format!("{hardware} hardware camera(s) found")
        } else {
            match ghostsun_camera::toupcam::probe() {
                Ok(0) => "No hardware camera detected (ToupTek SDK loaded)".to_owned(),
                Ok(n) => format!(
                    "ToupTek reports {n} camera(s), but enumeration returned none; reconnect and refresh"
                ),
                Err(e) => format!("No hardware camera; {e}"),
            }
        };
    }

    pub fn start(&mut self, ctx: &egui::Context) {
        if self.streaming || self.cameras.is_empty() {
            return;
        }
        let info = self.cameras[self.selected].clone();
        let (tx, rx) = channel::<FocusMsg>();
        let (ctx_tx, ctx_rx) = channel::<FocusCmd>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let ctx = ctx.clone();
        let exposure = self.exposure_us;
        let gain = self.gain;
        let auto = self.auto_exposure;

        let handle = std::thread::spawn(move || {
            worker(info, tx, ctx_rx, stop_thread, ctx, exposure, gain, auto)
        });

        self.rx = Some(rx);
        self.cmd = Some(ctx_tx);
        self.stop = Some(stop);
        self.handle = Some(handle);
        self.streaming = true;
        self.track_x.reset();
        self.track_y.reset();
        self.status = "streaming…".into();
    }

    pub fn stop(&mut self) {
        if let Some(s) = &self.stop {
            s.store(true, Ordering::SeqCst);
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.rx = None;
        self.cmd = None;
        self.stop = None;
        self.streaming = false;
        self.status = "stopped".into();
    }

    fn send_cmd(&self, cmd: FocusCmd) {
        if let Some(tx) = &self.cmd {
            let _ = tx.send(cmd);
        }
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        let mut latest: Option<Box<FocusUpdate>> = None;
        let mut err = None;
        if let Some(rx) = &self.rx {
            loop {
                match rx.try_recv() {
                    Ok(FocusMsg::Frame(u)) => latest = Some(u),
                    Ok(FocusMsg::Error(e)) => {
                        err = Some(e);
                        break;
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                }
            }
        }
        if let Some(e) = err {
            self.status = format!("camera error: {e}");
            self.stop();
            return;
        }
        if let Some(u) = latest {
            if u.strip_w > 0 && u.strip_h > 0 {
                let pixels = u.strip.iter().map(|&g| egui::Color32::from_gray(g)).collect();
                let img = egui::ColorImage { size: [u.strip_w, u.strip_h], pixels };
                match &mut self.tex {
                    Some(t) => t.set(img, egui::TextureOptions::NEAREST),
                    None => {
                        self.tex =
                            Some(ctx.load_texture("focus_strip", img, egui::TextureOptions::NEAREST))
                    }
                }
            }
            // Under auto-exposure the sliders are read-only mirrors of what the
            // camera actually settled on.
            if self.auto_exposure {
                if let Some(e) = u.cur_exposure {
                    self.exposure_us = e;
                }
                if let Some(g) = u.cur_gain {
                    self.gain = g;
                }
            }
            // Choose which line each axis reports, per the user's mode/pick.
            let spec_is_y = self.spectral_is_y();
            let (spec, slit) = {
                let (spec_lines, slit_lines) =
                    if spec_is_y { (&u.lines_y, &u.lines_x) } else { (&u.lines_x, &u.lines_y) };
                (
                    choose(spec_lines, self.line_mode, self.picked_center),
                    choose(slit_lines, LineMode::Narrowest, None),
                )
            };
            if spec_is_y {
                self.track_y.push(&spec);
                self.track_x.push(&slit);
            } else {
                self.track_x.push(&spec);
                self.track_y.push(&slit);
            }
            self.sel_spectral = spec;
            self.sel_slit = slit;

            // Spectral line identification (sunlight): calibrate pixel→λ against
            // the Fraunhofer catalog, seeded by the grating geometry.
            if self.identify_lines {
                let spec_lines = if spec_is_y { &u.lines_y } else { &u.lines_x };
                let centers: Vec<f64> = spec_lines.iter().map(|f| f.center).collect();
                let depths: Vec<f64> = spec_lines.iter().map(|f| f.depth).collect();
                let approx = geometric_dispersion(
                    self.grating_l_mm,
                    self.order,
                    self.focal_len_mm,
                    self.pixel_um,
                    self.central_wavelength,
                )
                .unwrap_or(self.dispersion_a_per_px);
                if let Some(cal) = calibrate(&centers, &depths, approx, self.central_wavelength) {
                    self.dispersion_a_per_px = cal.a.abs();
                    let tol = (2.0 * cal.a.abs()).max(0.35);
                    self.labels = identify(&centers, &cal, tol);
                    self.calibration = Some(cal);
                } else {
                    self.calibration = None;
                    self.labels.clear();
                }
            }
            self.last = Some(*u);
        }
    }

    fn reset_holds(&mut self) {
        self.track_x.reset();
        self.track_y.reset();
    }

    /// Which geometric track is the spectral one, given the dispersion axis.
    /// Vertical dispersion ⇒ horizontal spectral lines ⇒ track_y is spectral.
    fn spectral_is_y(&self) -> bool {
        self.dispersion == DispAxis::Vertical
    }

    // -- UI ----------------------------------------------------------------

    pub fn controls_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(8.0);
        ui.heading("Focus assistant");
        ui.label(
            egui::RichText::new("Minimise FWHM — spectral (camera focus) and slit (scope-on-slit).")
                .small()
                .weak(),
        );
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui.button("⟳ Scan").clicked() {
                self.refresh_cameras();
            }
            ui.label(&self.status);
        });

        let names: Vec<String> = self
            .cameras
            .iter()
            .map(|c| format!("{} · {}", c.backend.label(), c.name))
            .collect();
        egui::ComboBox::from_label("camera")
            .selected_text(names.get(self.selected).cloned().unwrap_or_else(|| "—".into()))
            .show_ui(ui, |ui| {
                for (i, n) in names.iter().enumerate() {
                    ui.selectable_value(&mut self.selected, i, n);
                }
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let can_start = !self.cameras.is_empty();
            if !self.streaming {
                if ui.add_enabled(can_start, egui::Button::new("▶ Start")).clicked() {
                    self.start(ctx);
                }
            } else if ui.button("■ Stop").clicked() {
                self.stop();
            }
            if ui.button("reset min-hold").clicked() {
                self.reset_holds();
            }
        });

        ui.add_space(10.0);
        ui.spacing_mut().slider_width = (ui.available_width() - 130.0).max(120.0);

        if ui.checkbox(&mut self.auto_exposure, "auto-exposure").changed() {
            self.send_cmd(FocusCmd::AutoExposure(self.auto_exposure));
        }
        let manual = !self.auto_exposure;

        let (emin, emax) = self
            .cameras
            .get(self.selected)
            .map(|c| (*c.exposure_us.start(), *c.exposure_us.end()))
            .unwrap_or((100, 1_000_000));
        ui.label("exposure (µs)");
        if ui
            .add_enabled(
                manual,
                egui::Slider::new(&mut self.exposure_us, emin..=emax.min(2_000_000)).logarithmic(true),
            )
            .changed()
        {
            self.send_cmd(FocusCmd::Exposure(self.exposure_us));
        }

        let (gmin, gmax) = self
            .cameras
            .get(self.selected)
            .map(|c| (*c.gain.start(), *c.gain.end()))
            .unwrap_or((0, 600));
        self.gain = self.gain.clamp(gmin, gmax);
        ui.label(if gmin >= 100 { "gain (%, 100 = 1×)" } else { "gain" });
        if ui
            .add_enabled(manual, egui::Slider::new(&mut self.gain, gmin..=gmax))
            .changed()
        {
            self.send_cmd(FocusCmd::Gain(self.gain));
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("dispersion axis:");
            ui.selectable_value(&mut self.dispersion, DispAxis::Vertical, "⇕ vertical");
            ui.selectable_value(&mut self.dispersion, DispAxis::Horizontal, "⇔ horizontal");
        });
        ui.label(
            egui::RichText::new("Spectral lines run ⊥ to this. Sets only which readout is which.")
                .small()
                .weak(),
        );
        ui.horizontal(|ui| {
            ui.label("Å / px");
            ui.add_enabled(
                !self.identify_lines,
                egui::DragValue::new(&mut self.dispersion_a_per_px)
                    .speed(0.001)
                    .range(0.001..=1.0)
                    .fixed_decimals(3),
            );
        });

        ui.add_space(6.0);
        if ui
            .checkbox(&mut self.identify_lines, "identify lines (sunlight)")
            .changed()
            && !self.identify_lines
        {
            self.calibration = None;
            self.labels.clear();
        }
        if self.identify_lines {
            egui::Grid::new("optics").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                ui.label("grating l/mm");
                ui.add(egui::DragValue::new(&mut self.grating_l_mm).range(100.0..=5000.0));
                ui.end_row();
                ui.label("order");
                ui.add(egui::DragValue::new(&mut self.order).range(1..=5));
                ui.end_row();
                ui.label("focal length mm");
                ui.add(egui::DragValue::new(&mut self.focal_len_mm).range(10.0..=1000.0));
                ui.end_row();
                ui.label("pixel µm");
                ui.add(egui::DragValue::new(&mut self.pixel_um).speed(0.05).range(0.5..=20.0));
                ui.end_row();
                ui.label("central λ (Å)");
                ui.add(egui::DragValue::new(&mut self.central_wavelength).range(3000.0..=9000.0));
                ui.end_row();
            });
            let geo = geometric_dispersion(
                self.grating_l_mm,
                self.order,
                self.focal_len_mm,
                self.pixel_um,
                self.central_wavelength,
            );
            match (self.calibration, geo) {
                (Some(c), _) => {
                    ui.label(
                        egui::RichText::new(format!(
                            "locked · {:.4} Å/px · {} lines · rms {:.3} Å",
                            c.a.abs(),
                            c.n_matched,
                            c.rms
                        ))
                        .small()
                        .color(egui::Color32::LIGHT_GREEN),
                    );
                }
                (None, Some(g)) => {
                    ui.label(
                        egui::RichText::new(format!(
                            "geometry {g:.4} Å/px — no lock yet (need ≥3 catalog lines)"
                        ))
                        .small()
                        .weak(),
                    );
                }
                (None, None) => {
                    ui.label(egui::RichText::new("check optics values").small().weak());
                }
            }
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("target line:");
            let mut changed = false;
            changed |= ui
                .selectable_value(&mut self.line_mode, LineMode::Narrowest, "narrowest")
                .clicked();
            changed |= ui
                .selectable_value(&mut self.line_mode, LineMode::Deepest, "deepest")
                .clicked();
            if self.line_mode == LineMode::Manual {
                let _ = ui.selectable_label(true, "picked");
            }
            if changed {
                self.picked_center = None;
                self.reset_holds();
            }
        });
        if self.line_mode == LineMode::Manual {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("locked @ {:.0} px", self.picked_center.unwrap_or(0.0)))
                        .small()
                        .weak(),
                );
                if ui.small_button("clear").clicked() {
                    self.line_mode = LineMode::Narrowest;
                    self.picked_center = None;
                    self.reset_holds();
                }
            });
        } else {
            ui.label(egui::RichText::new("click the spectral plot to lock a line").small().weak());
        }

        ui.add_space(12.0);

        let spectral_fit = self.sel_spectral;
        let slit_fit = self.sel_slit;
        let (spectral_min, slit_min) = if self.spectral_is_y() {
            (self.track_y.min_hold, self.track_x.min_hold)
        } else {
            (self.track_x.min_hold, self.track_y.min_hold)
        };
        let a_per_px = self.dispersion_a_per_px;

        readout(ui, "Spectral line (dispersion)", SPECTRAL_COLOR, spectral_fit, spectral_min, Some(a_per_px));
        ui.add_space(6.0);
        readout(ui, "Slit jaws / dust (spatial)", SLIT_COLOR, slit_fit, slit_min, None);

        if let Some(l) = &self.last {
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!("frame {}×{}  mean {:.0}", l.full_w, l.full_h, l.mean))
                    .small()
                    .weak(),
            );
        }
    }

    pub fn view_ui(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        if let Some(tex) = &self.tex {
            let avail = ui.available_width();
            let aspect = tex.aspect_ratio();
            let h = (avail / aspect).min(260.0);
            ui.add(egui::Image::new(tex).fit_to_exact_size(egui::vec2(avail, h)));
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Start a camera to see the live spectrum.")
                        .size(18.0)
                        .weak(),
                );
            });
            return;
        }

        let spec_is_y = self.spectral_is_y();
        // Clone what the plots need so the borrow of self.last is released before
        // a click can mutate self.picked_center.
        let (spec_prof, slit_prof, spec_cands) = {
            let last = match &self.last {
                Some(l) => l,
                None => return,
            };
            let (sp, kp) = if spec_is_y {
                (last.prof_y.clone(), last.prof_x.clone())
            } else {
                (last.prof_x.clone(), last.prof_y.clone())
            };
            let cands: Vec<f64> = if spec_is_y {
                last.lines_y.iter().map(|f| f.center).collect()
            } else {
                last.lines_x.iter().map(|f| f.center).collect()
            };
            (sp, kp, cands)
        };
        let spec_fit = self.sel_spectral;
        let slit_fit = self.sel_slit;
        let picked = self.picked_center;
        let spec_labels: Vec<(f64, String)> = self
            .labels
            .iter()
            .map(|l| (l.x, format!("{} {:.1}", l.element, l.wavelength)))
            .collect();

        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Spectral profile (across dispersion) — click to lock a line")
                .small()
                .color(SPECTRAL_COLOR),
        );
        if let Some(x) = profile_plot(
            ui,
            "focus_spectral",
            &spec_prof,
            spec_fit,
            &spec_cands,
            picked,
            &spec_labels,
            SPECTRAL_COLOR,
            170.0,
        ) {
            self.picked_center = Some(x);
            self.line_mode = LineMode::Manual;
            self.reset_holds();
        }

        ui.add_space(4.0);
        ui.label(egui::RichText::new("Slit profile (across the slit)").small().color(SLIT_COLOR));
        profile_plot(ui, "focus_slit", &slit_prof, slit_fit, &[], None, &[], SLIT_COLOR, 150.0);

        // Combined FWHM trend + min-holds.
        let (spec_track, slit_track) = if self.spectral_is_y() {
            (&self.track_y, &self.track_x)
        } else {
            (&self.track_x, &self.track_y)
        };
        if !spec_track.history.is_empty() || !slit_track.history.is_empty() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new("FWHM history (px)").small().weak());
            let sh: Vec<f64> = spec_track.history.iter().copied().collect();
            let kh: Vec<f64> = slit_track.history.iter().copied().collect();
            let (smin, kmin) = (spec_track.min_hold, slit_track.min_hold);
            Plot::new("focus_trend").height(110.0).allow_scroll(false).show(ui, |p| {
                let sp: PlotPoints = sh.iter().enumerate().map(|(x, &y)| [x as f64, y]).collect();
                p.line(Line::new(sp).color(SPECTRAL_COLOR).name("spectral"));
                let kp: PlotPoints = kh.iter().enumerate().map(|(x, &y)| [x as f64, y]).collect();
                p.line(Line::new(kp).color(SLIT_COLOR).name("slit"));
                if smin.is_finite() {
                    p.hline(HLine::new(smin).color(SPECTRAL_COLOR));
                }
                if kmin.is_finite() {
                    p.hline(HLine::new(kmin).color(SLIT_COLOR));
                }
            });
        }
    }
}

fn readout(
    ui: &mut egui::Ui,
    title: &str,
    color: egui::Color32,
    fit: Option<Fit>,
    min_hold: f64,
    a_per_px: Option<f64>,
) {
    egui::Frame::group(ui.style()).fill(ui.visuals().faint_bg_color).show(ui, |ui| {
        ui.label(egui::RichText::new(title).small().weak());
        match fit {
            Some(f) if f.depth > DEPTH_GATE => {
                ui.label(egui::RichText::new(format!("{:.2} px", f.fwhm)).size(26.0).strong().color(color));
                let extra = match a_per_px {
                    Some(a) => format!("{:.3} Å   ·   depth {:.0}%", f.fwhm * a, f.depth * 100.0),
                    None => format!("depth {:.0}%", f.depth * 100.0),
                };
                ui.label(egui::RichText::new(extra).small());
            }
            _ => {
                ui.label(egui::RichText::new("— no line —").size(20.0).weak());
            }
        }
        let mh = if min_hold.is_finite() {
            format!("min-hold {min_hold:.2} px")
        } else {
            "min-hold —".into()
        };
        ui.label(egui::RichText::new(mh).small().color(egui::Color32::LIGHT_GREEN));
    });
}

/// Draws a profile with candidate-line markers and the selected fit. Returns
/// the clicked x (profile index) if the user clicked, for line locking.
#[allow(clippy::too_many_arguments)]
fn profile_plot(
    ui: &mut egui::Ui,
    id: &str,
    profile: &[f32],
    fit: Option<Fit>,
    candidates: &[f64],
    picked: Option<f64>,
    labels: &[(f64, String)],
    color: egui::Color32,
    height: f32,
) -> Option<f64> {
    let profile = profile.to_vec();
    let cands = candidates.to_vec();
    let labels = labels.to_vec();
    let ymin = profile.iter().cloned().fold(f32::MAX, f32::min) as f64;
    let ymax = profile.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let span = (ymax - ymin).max(1.0);
    let mut clicked = None;
    let mut plot = Plot::new(id).height(height).allow_scroll(false).show_axes([false, true]);
    if !labels.is_empty() {
        // Reserve headroom above the profile so the labels aren't clipped.
        plot = plot.include_y(ymin).include_y(ymax + 0.34 * span);
    }
    plot.show(ui, |p| {
        let pts: PlotPoints = profile.iter().enumerate().map(|(x, &y)| [x as f64, y as f64]).collect();
        p.line(Line::new(pts).color(color).name("profile"));
        // Faint marker at each detected candidate line.
        for &c in &cands {
            p.vline(VLine::new(c).color(egui::Color32::from_gray(80)));
        }
        // Identified lines: green marker + element/λ label, staggered in the
        // headroom so neighbours don't overprint and nothing clips at the top.
        let label_col = egui::Color32::from_rgb(140, 235, 165);
        for (k, (x, txt)) in labels.iter().enumerate() {
            p.vline(VLine::new(*x).color(label_col));
            let ly = ymax + span * (0.06 + 0.13 * (k % 2) as f64);
            p.text(
                Text::new(PlotPoint::new(*x, ly), egui::RichText::new(txt).size(11.0).color(label_col))
                    .anchor(egui::Align2::CENTER_BOTTOM),
            );
        }
        if let Some(f) = fit {
            if f.depth > DEPTH_GATE {
                let amp = f.depth * f.continuum;
                let lo = (f.center - 4.0 * f.sigma).max(0.0);
                let hi = f.center + 4.0 * f.sigma;
                let curve: PlotPoints = (0..=80)
                    .map(|i| {
                        let x = lo + (hi - lo) * i as f64 / 80.0;
                        let dx = x - f.center;
                        [x, f.continuum - amp * (-dx * dx / (2.0 * f.sigma * f.sigma)).exp()]
                    })
                    .collect();
                p.line(Line::new(curve).name("fit"));
                p.hline(HLine::new(f.continuum - amp / 2.0).name("half-max"));
                p.vline(VLine::new(f.center - f.fwhm / 2.0));
                p.vline(VLine::new(f.center + f.fwhm / 2.0));
            }
        }
        if let Some(pc) = picked {
            p.vline(VLine::new(pc).color(egui::Color32::WHITE));
        }
        if p.response().clicked() {
            if let Some(pt) = p.pointer_coordinate() {
                clicked = Some(pt.x);
            }
        }
    });
    clicked
}

// -- capture thread --------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn worker(
    info: CameraInfo,
    tx: Sender<FocusMsg>,
    cmd_rx: Receiver<FocusCmd>,
    stop: Arc<AtomicBool>,
    ctx: egui::Context,
    exposure_us: u32,
    gain: u16,
    auto_exposure: bool,
) {
    let mut cam = match open(&info) {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(FocusMsg::Error(e.to_string()));
            return;
        }
    };
    cam.set_exposure_us(exposure_us).ok();
    cam.set_gain(gain).ok();
    cam.set_auto_exposure(auto_exposure).ok();
    if let Err(e) = cam.start() {
        let _ = tx.send(FocusMsg::Error(e.to_string()));
        return;
    }

    while !stop.load(Ordering::SeqCst) {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                FocusCmd::Exposure(e) => {
                    cam.set_exposure_us(e).ok();
                }
                FocusCmd::Gain(g) => {
                    cam.set_gain(g).ok();
                }
                FocusCmd::AutoExposure(on) => {
                    cam.set_auto_exposure(on).ok();
                }
            }
        }
        match cam.next_frame(1000) {
            Ok(frame) => {
                // Both axes, every frame: the two line families separate cleanly
                // because averaging one axis cancels lines parallel to it.
                let prof_x = frame.mean_profile(true); // dips = vertical lines
                let prof_y = frame.mean_profile(false); // dips = horizontal lines
                let lines_x: Vec<Fit> = fit_lines_1d(&prof_x, 0.02).into_iter().map(Fit::from).collect();
                let lines_y: Vec<Fit> = fit_lines_1d(&prof_y, 0.02).into_iter().map(Fit::from).collect();
                let mean = if prof_x.is_empty() {
                    0.0
                } else {
                    (prof_x.iter().map(|&v| v as f64).sum::<f64>() / prof_x.len() as f64) as f32
                };
                let (strip, sw, sh) = make_strip(&frame);
                let cur_exposure = cam.current_exposure_us();
                let cur_gain = cam.current_gain();
                let _ = tx.send(FocusMsg::Frame(Box::new(FocusUpdate {
                    strip,
                    strip_w: sw,
                    strip_h: sh,
                    prof_x: prof_x.iter().map(|&v| v as f32).collect(),
                    prof_y: prof_y.iter().map(|&v| v as f32).collect(),
                    lines_x,
                    lines_y,
                    mean,
                    full_w: frame.width,
                    full_h: frame.height,
                    cur_exposure,
                    cur_gain,
                })));
                ctx.request_repaint();
            }
            Err(ghostsun_camera::CameraError::Timeout) => continue,
            Err(e) => {
                let _ = tx.send(FocusMsg::Error(e.to_string()));
                break;
            }
        }
    }
    cam.stop();
}

fn make_strip(frame: &ghostsun_camera::Frame) -> (Vec<u8>, usize, usize) {
    let (fw, fh) = (frame.width, frame.height);
    if fw == 0 || fh == 0 {
        return (Vec::new(), 0, 0);
    }
    let sw = fw.min(STRIP_W);
    let sh = fh.min(STRIP_H);
    let mut samp = vec![0u16; sw * sh];
    let mut lo = u16::MAX;
    let mut hi = 0u16;
    for y in 0..sh {
        let sy = y * fh / sh;
        for x in 0..sw {
            let sx = x * fw / sw;
            let v = frame.data[sy * fw + sx];
            samp[y * sw + x] = v;
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let span = (hi.saturating_sub(lo)).max(1) as f32;
    let out: Vec<u8> = samp
        .iter()
        .map(|&v| (((v.saturating_sub(lo)) as f32 / span) * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    (out, sw, sh)
}

impl Drop for FocusState {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(dead_code)]
pub fn full_roi(info: &CameraInfo) -> Roi {
    Roi { x: 0, y: 0, w: info.max_width, h: info.max_height }
}
