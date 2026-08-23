//! Live focus assistant for a spectroheliograph.
//!
//! Each camera frame carries two perpendicular families of dark lines, which
//! measure two different focus problems:
//!   * **Spectral absorption lines** run along the slit (⊥ dispersion). Their
//!     width is the *spectral* focus.
//!   * **Slit jaw / dust / defect lines** run along the dispersion axis
//!     (⊥ slit). Their width is the *spatial* focus, and they are essentially
//!     always present.
//!
//! Averaging the frame along one axis cancels the lines parallel to that axis
//! and preserves the perpendicular family, so the two are measured cleanly and
//! independently every frame with the *same* core line fitter the
//! reconstruction uses. Backend-agnostic via the `ghostsun_camera::Camera`
//! trait (ToupTek / ZWO / synthetic).
//!
//! # The three-stage procedure
//!
//! There are three focus unknowns — telescope, collimator, camera lens — and
//! only the camera lens has an obvious reference, which on many builds cannot
//! be reached because the lens and camera shoulder will not come off as a unit.
//! The instinctive move is to optimise solar sharpness against all three, which
//! is a shallow 3-D valley with no unique answer.
//!
//! It is not actually degenerate. Write out which observable responds to what:
//!
//! | observable                     | telescope | collimator | camera |
//! |--------------------------------|-----------|------------|--------|
//! | slit-jaw dust sharpness        | **no**    | yes        | yes    |
//! | spectral line FWHM             | **no**    | yes        | yes    |
//! | solar detail along the slit     | yes       | yes        | yes    |
//!
//! The system is *triangular*. Dust lies physically in the slit plane and the
//! spectral line is the dispersed image of that same plane, so the first two
//! rows carry no dependence on where the telescope's focal plane sits.
//!
//! **Stage A** (this module's [`Stage::Spectrograph`]) uses those two rows to
//! pin collimator and camera with zero telescope contamination. They are only
//! independent of *each other* because the grating is anamorphic — see
//! [`crate::vcurve::Split`] — which is why a high-dispersion grating makes this
//! solvable at all. Being telescope-blind, Stage A can be run at the bench with
//! a neon lamp on the slit: no sun, no seeing, no cloud.
//!
//! **Stage B** ([`Stage::Telescope`]) then has exactly one unknown left, and
//! uses row three. See [`crate::focusmetrics`] for why it is measured on
//! continuum columns rather than the line core.
//!
//! Order is load-bearing. Focus the telescope against a mis-collimated
//! spectrograph and it settles wherever best masks the *spatial* blur, while
//! being unable to touch the astigmatism — the telescope has no leverage in the
//! dispersion plane at all. The result is a soft compromise on both axes that
//! feels like a trade-off because it is one.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Points, Text, VLine};

use ghostsun_camera::{enumerate_all, open, Backend, CameraInfo, Roi};
use ghostsun_core::linefit::fit_lines_1d;
use ghostsun_core::lines::{calibrate, geometric_dispersion, identify, Calibration, LabeledLine};
use ghostsun_core::ser::{FrameTime, SerRecorder};

use crate::focusmetrics::{self, LuckyBuf, StructureSplit};
use crate::vcurve::{self, NullPoint, ParabolaFit, VCurve};

const STRIP_W: usize = 1200;
const STRIP_H: usize = 280;
const HISTORY: usize = 600;
const DEPTH_GATE: f64 = 0.03;
/// Frames averaged into one V-curve sample. At typical focus-tab frame rates
/// this is a second or two — long enough to beat down seeing, short enough that
/// nobody stops using it.
const DEFAULT_CAPTURE_FRAMES: usize = 40;
/// Rolling window for the live "lucky" Stage B readout.
const LUCKY_WINDOW: usize = 90;
/// Fraction of the brightest pixels averaged for the Sun-search signal.
///
/// A literal maximum is too easy for one hot pixel or cosmic ray to win.
/// Averaging the brightest one percent follows the illuminated spectrum while
/// remaining a peak-like metric rather than a whole-frame centroid or mean.
const SUN_PEAK_FRACTION: usize = 100;

const SPECTRAL_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 210, 255);
const SLIT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 170, 90);
const TELE_COLOR: egui::Color32 = egui::Color32::from_rgb(160, 235, 170);
const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 205, 100);

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LineMode {
    /// Sharpest line — the best focus reference.
    Narrowest,
    /// Strongest line.
    Deepest,
    /// The line nearest a user-clicked position.
    Manual,
}

/// Pick one line from the candidates per the mode (Manual uses `picked`).
///
/// Only lines deep enough to be *used* are candidates. The detector admits
/// anything 2% deep while tracking and capture require 3%, and on a real
/// profile photon noise reaches 2% — so without this filter `Narrowest`
/// reliably selects a noise spike, noise being narrower than any real line.
/// The genuine line is then never considered and every burst is rejected as
/// "no usable line" while a 40%-deep line sits in plain view on the plot.
/// A selection criterion must not be able to choose something the acceptance
/// criterion will throw away.
fn choose(lines: &[Fit], mode: LineMode, picked: Option<f64>) -> Option<Fit> {
    let usable = || lines.iter().filter(|f| f.depth > DEPTH_GATE);
    match mode {
        LineMode::Narrowest => usable()
            .min_by(|a, b| a.fwhm.partial_cmp(&b.fwhm).unwrap())
            .copied(),
        LineMode::Deepest => usable()
            .max_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap())
            .copied(),
        LineMode::Manual => picked.and_then(|pc| {
            usable()
                .min_by(|a, b| {
                    (a.center - pc)
                        .abs()
                        .partial_cmp(&(b.center - pc).abs())
                        .unwrap()
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
    pub peak: f32,
    pub full_w: usize,
    pub full_h: usize,
    pub cur_exposure: Option<u32>,
    pub cur_gain: Option<u16>,
    // -- Stage B (telescope) ------------------------------------------------
    /// Intensity along the slit, averaged over continuum columns only.
    pub slit_cut: Vec<f32>,
    /// Dispersion positions that passed the continuum mask.
    pub n_continuum: usize,
    /// Limb knife-edge FWHM in px, when a limb is on the slit.
    pub limb_width: Option<f64>,
    /// High-passed along-slit contrast, whole span and outer thirds.
    pub structure: StructureSplit,
    /// A hardware-ROI recording is active: frames are the capture band, so
    /// sensor-coordinate overlays do not apply.
    pub hw_roi_active: bool,
}

enum FocusMsg {
    /// Sent once after the camera opens: the open handle knows the true
    /// control ranges (e.g. the ToupTek per-model gain ceiling), which can be
    /// wider than the enumerate-time guess bounding the sliders.
    Opened(CameraInfo),
    Frame(Box<FocusUpdate>),
    RecordingStarted {
        path: PathBuf,
        width: usize,
        height: usize,
        hw_roi: bool,
    },
    RecordingProgress(usize),
    RecordingStopped {
        path: PathBuf,
        frames: usize,
        /// Achieved capture rate; 0 when unknown (fewer than two frames).
        fps: f64,
        hw_roi: bool,
        /// Per-frame times were written from the camera clock, not host stamps.
        device_clock: bool,
    },
    RecordingError(String),
    /// Non-fatal status the user needs to see (e.g. a stalled stream), which
    /// would otherwise only exist in the log file.
    Note(String),
    /// The live (outside-recording) hardware ROI was applied or released.
    LiveRoi {
        active: bool,
        /// Band actually accepted by the sensor, in sensor rows.
        y0: usize,
        h: usize,
    },
    Error(String),
}

enum FocusCmd {
    Exposure(u32),
    Gain(u16),
    AutoExposure(bool),
    /// The worker needs the dispersion axis to know which way to cut the frame
    /// for the Stage B profile; the Stage A readouts only need it for labels.
    Dispersion(bool),
    StartSer {
        path: PathBuf,
        capture_height: usize,
        anchor_y: f64,
        /// Ask the sensor to read only the capture band for the duration of
        /// the recording (falls back to the software crop if refused).
        hw_roi: bool,
    },
    StopSer,
    /// Apply (or release) the hardware ROI OUTSIDE a recording, so the live
    /// preview is the real capture band at the real cropped frame rate. The
    /// recording path is unchanged; this only decides what the sensor reads
    /// between scans.
    LiveRoi {
        capture_height: usize,
        anchor_y: f64,
        enable: bool,
    },
}

struct SerRequest {
    path: PathBuf,
    capture_height: usize,
    anchor_y: f64,
    /// Some(sensor row of the band top) when a hardware ROI is active.
    hw_roi_y0: Option<usize>,
}

struct ActiveSer {
    path: PathBuf,
    y0: usize,
    height: usize,
    recorder: SerRecorder,
    hw_roi_y0: Option<usize>,
    /// First-frame time, for the achieved-fps report.
    started: Instant,
}

/// Which stage of the procedure the panel is driving.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Collimator + camera lens, via the astigmatism null. Telescope-blind, so
    /// this runs at the bench with a lamp on the slit.
    Spectrograph,
    /// Telescope focuser, once Stage A is closed.
    Telescope,
}

/// Stage B metric. They fail differently — see [`crate::focusmetrics`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TeleMetric {
    /// Solar limb as a knife edge. Dust-immune; needs the limb on the slit.
    LimbEdge,
    /// High-passed along-slit contrast. Always available; dust adds a constant
    /// pedestal, and it is the only one that supports the top/bottom split.
    Structure,
}

impl TeleMetric {
    /// Whether best focus *minimises* this metric.
    fn want_min(self) -> bool {
        matches!(self, TeleMetric::LimbEdge)
    }
    fn label(self) -> &'static str {
        match self {
            TeleMetric::LimbEdge => "limb edge FWHM (px)",
            TeleMetric::Structure => "along-slit contrast",
        }
    }
}

/// An in-progress capture: accumulate N frames, then commit one V-curve point.
///
/// Committing a single frame would sample the seeing, not the focuser.
struct Capture {
    remaining: usize,
    total: usize,
    pos: f64,
    spec: Vec<f64>,
    slit: Vec<f64>,
    /// Deepest line seen per family during the burst, gated or not. A rejection
    /// that reports "deepest 1.4%, need 3%" is actionable; a bare "no usable
    /// line" leaves the user guessing between exposure, the wrong target line,
    /// and there being no such feature at all.
    best_spec_depth: f64,
    best_slit_depth: f64,
    tele: Vec<f64>,
    tele_top: Vec<f64>,
    tele_bot: Vec<f64>,
}

pub struct SearchCameraRestore {
    was_streaming: bool,
    exposure_us: u32,
    auto_exposure: bool,
}

/// Min-hold + rolling history for one measured axis.
#[derive(Default)]
struct Track {
    min_hold: f64,
    history: VecDeque<f64>,
}

impl Track {
    fn new() -> Track {
        Track {
            min_hold: f64::INFINITY,
            history: VecDeque::with_capacity(HISTORY),
        }
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
    /// The user has picked a camera explicitly, so auto-selection stops
    /// second-guessing them on the next scan.
    camera_chosen_by_user: bool,
    pub streaming: bool,
    pub exposure_us: u32,
    pub gain: u16,
    pub auto_exposure: bool,
    pub recording: bool,
    pub recorded_frames: usize,
    pub recording_path: Option<PathBuf>,
    pub recording_status: String,
    /// A hardware ROI is applied to the sensor outside of any recording, so
    /// the live preview IS the capture band at the cropped frame rate.
    pub live_roi_active: bool,
    pub live_roi_status: String,
    pub dispersion: DispAxis,
    pub dispersion_a_per_px: f64,
    pub line_mode: LineMode,
    pub picked_center: Option<f64>,
    /// Target selection for the slit/dust family, mirroring the spectral one.
    ///
    /// Stage A only works if the SAME feature is measured at every camera
    /// position. With several dust specks of differing intrinsic width,
    /// `Narrowest` hops between them frame to frame and that scatter lands
    /// directly in the slit V-curve, and so in the Delta it feeds.
    pub slit_line_mode: LineMode,
    pub slit_picked_center: Option<f64>,
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
    // -- three-stage focus procedure ---------------------------------------
    pub stage: Stage,
    pub tele_metric: TeleMetric,
    pub capture_frames: usize,
    /// Camera-lens micrometer reading, as the user reads it (Stage A x-axis).
    pub camera_pos_text: String,
    /// Collimator micrometer reading for the current Stage A sweep.
    pub collimator_pos_text: String,
    /// Telescope focuser reading (Stage B x-axis).
    pub focuser_pos_text: String,
    curve_spec: VCurve,
    curve_slit: VCurve,
    curve_tele: VCurve,
    curve_tele_top: VCurve,
    curve_tele_bot: VCurve,
    /// One (collimator, Δ) pair per completed Stage A sweep.
    null_points: Vec<NullPoint>,
    capture: Option<Capture>,
    lucky_limb: LuckyBuf,
    lucky_struct: LuckyBuf,
    pub stage_status: String,
    saved: Option<SavedFocus>,
    sel_spectral: Option<Fit>,
    sel_slit: Option<Fit>,
    track_x: Track, // vertical lines
    track_y: Track, // horizontal lines
    last: Option<FocusUpdate>,
    frame_seq: u64,
    tex: Option<egui::TextureHandle>,
    rx: Option<Receiver<FocusMsg>>,
    cmd: Option<Sender<FocusCmd>>,
    stop: Option<Arc<AtomicBool>>,
    /// True while one processed frame is waiting in `rx`.
    ///
    /// Without this gate an unbounded mpsc queue grows whenever acquisition is
    /// faster than egui, and the preview gradually falls behind real time.
    frame_pending: Option<Arc<AtomicBool>>,
    handle: Option<JoinHandle<()>>,
}

impl Default for FocusState {
    fn default() -> Self {
        FocusState {
            cameras: Vec::new(),
            selected: 0,
            camera_chosen_by_user: false,
            streaming: false,
            exposure_us: 10_000,
            gain: 200,
            auto_exposure: false,
            recording: false,
            recorded_frames: 0,
            recording_path: None,
            recording_status: "not recording".into(),
            live_roi_active: false,
            live_roi_status: "full sensor".into(),
            dispersion: DispAxis::Vertical,
            dispersion_a_per_px: 0.085,
            line_mode: LineMode::Narrowest,
            picked_center: None,
            slit_line_mode: LineMode::Narrowest,
            slit_picked_center: None,
            identify_lines: false,
            grating_l_mm: 2400.0,
            order: 1,
            focal_len_mm: 125.0,
            pixel_um: 2.0,
            central_wavelength: 6562.79,
            calibration: None,
            labels: Vec::new(),
            status: String::new(),
            stage: Stage::Spectrograph,
            tele_metric: TeleMetric::LimbEdge,
            capture_frames: DEFAULT_CAPTURE_FRAMES,
            camera_pos_text: String::new(),
            collimator_pos_text: String::new(),
            focuser_pos_text: String::new(),
            curve_spec: VCurve::new(true),
            curve_slit: VCurve::new(true),
            curve_tele: VCurve::new(true),
            curve_tele_top: VCurve::new(true),
            curve_tele_bot: VCurve::new(true),
            null_points: Vec::new(),
            capture: None,
            lucky_limb: LuckyBuf::new(LUCKY_WINDOW),
            lucky_struct: LuckyBuf::new(LUCKY_WINDOW),
            stage_status: String::new(),
            saved: SavedFocus::load(),
            sel_spectral: None,
            sel_slit: None,
            track_x: Track::new(),
            track_y: Track::new(),
            last: None,
            frame_seq: 0,
            tex: None,
            rx: None,
            cmd: None,
            stop: None,
            frame_pending: None,
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
        // The synthetic camera always enumerates first, so index 0 is never the
        // one someone with hardware attached wants. Prefer real hardware until
        // the user picks for themselves — after that their choice stands, so a
        // deliberate switch to the synthetic source survives a re-scan.
        if !self.camera_chosen_by_user {
            if let Some(hw) = self
                .cameras
                .iter()
                .position(|c| c.backend != ghostsun_camera::Backend::Synth)
            {
                self.selected = hw;
            }
        }
        let hardware = self
            .cameras
            .iter()
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
        let frame_pending = Arc::new(AtomicBool::new(false));
        let frame_pending_thread = frame_pending.clone();
        let ctx = ctx.clone();
        let exposure = self.exposure_us;
        let gain = self.gain;
        let auto = self.auto_exposure;
        let disp_h = self.dispersion == DispAxis::Horizontal;

        let handle = std::thread::spawn(move || {
            worker(
                info,
                tx,
                ctx_rx,
                stop_thread,
                ctx,
                exposure,
                gain,
                auto,
                disp_h,
                frame_pending_thread,
            )
        });

        self.rx = Some(rx);
        self.cmd = Some(ctx_tx);
        self.stop = Some(stop);
        self.frame_pending = Some(frame_pending);
        self.handle = Some(handle);
        self.streaming = true;
        self.track_x.reset();
        self.track_y.reset();
        self.status = "streaming…".into();
    }

    pub fn stop(&mut self) {
        // The worker owns the camera handle; when it exits the device is
        // released and reopens full-frame, so a live ROI cannot survive.
        self.live_roi_active = false;
        self.live_roi_status = "full sensor".into();
        if let Some(s) = &self.stop {
            s.store(true, Ordering::SeqCst);
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.rx = None;
        self.cmd = None;
        self.stop = None;
        self.frame_pending = None;
        self.streaming = false;
        if self.recording {
            self.recording = false;
            self.recording_status =
                format!("recording stopped with camera ({} frames)", self.recorded_frames);
        }
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
        let mut recording_event = None;
        if let Some(rx) = &self.rx {
            loop {
                match rx.try_recv() {
                    Ok(FocusMsg::Opened(info)) => {
                        // Adopt the open handle's true control ranges so the
                        // sliders (and their clamp) track the real device
                        // limits rather than the enumerate-time guess.
                        if let Some(c) = self
                            .cameras
                            .iter_mut()
                            .find(|c| c.backend == info.backend && c.id == info.id)
                        {
                            *c = info;
                        }
                    }
                    Ok(FocusMsg::Frame(u)) => {
                        if let Some(pending) = &self.frame_pending {
                            pending.store(false, Ordering::Release);
                        }
                        latest = Some(u);
                    }
                    Ok(
                        event @ (FocusMsg::RecordingStarted { .. }
                        | FocusMsg::RecordingProgress(_)
                        | FocusMsg::RecordingStopped { .. }
                        | FocusMsg::RecordingError(_)),
                    ) => recording_event = Some(event),
                    Ok(FocusMsg::Note(note)) => {
                        self.status = note;
                    }
                    Ok(FocusMsg::LiveRoi { active, y0, h }) => {
                        self.live_roi_active = active;
                        self.live_roi_status = if active {
                            format!("hardware ROI live · rows {y0}–{} ({h} px)", y0 + h)
                        } else {
                            "full sensor".into()
                        };
                    }
                    Ok(FocusMsg::Error(e)) => {
                        err = Some(e);
                        break;
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                }
            }
        }
        if let Some(event) = recording_event {
            match event {
                FocusMsg::RecordingStarted {
                    path,
                    width,
                    height,
                    hw_roi,
                } => {
                    self.recording = true;
                    self.recording_path = Some(path);
                    let mode = if hw_roi { " · hardware ROI" } else { "" };
                    self.recording_status = format!("recording {width}×{height} mono16{mode}");
                }
                FocusMsg::RecordingProgress(frames) => {
                    self.recorded_frames = frames;
                    self.recording_status = format!("recording · {frames} frames");
                }
                FocusMsg::RecordingStopped { path, frames, fps, hw_roi, device_clock } => {
                    self.recording = false;
                    self.recorded_frames = frames;
                    self.recording_path = Some(path);
                    let mode = if hw_roi { " · hardware ROI" } else { "" };
                    // The clock behind the per-frame timestamps decides how
                    // precisely reconstruction can place each column.
                    let clock = if device_clock { " · camera clock" } else { " · host clock" };
                    self.recording_status = if fps > 0.0 {
                        format!("saved {frames} frames · {fps:.1} fps{mode}{clock}")
                    } else {
                        format!("saved {frames} frames{mode}{clock}")
                    };
                }
                FocusMsg::RecordingError(error) => {
                    self.recording = false;
                    self.recording_status = format!("SER recording failed: {error}");
                }
                FocusMsg::Opened(_)
                | FocusMsg::Frame(_)
                | FocusMsg::Note(_)
                | FocusMsg::LiveRoi { .. }
                | FocusMsg::Error(_) => unreachable!(),
            }
        }
        if let Some(e) = err {
            self.recording = false;
            self.status = format!("camera error: {e}");
            self.stop();
            return;
        }
        if let Some(u) = latest {
            if u.strip_w > 0 && u.strip_h > 0 {
                let pixels = u
                    .strip
                    .iter()
                    .map(|&g| egui::Color32::from_gray(g))
                    .collect();
                let img = egui::ColorImage {
                    size: [u.strip_w, u.strip_h],
                    pixels,
                };
                match &mut self.tex {
                    Some(t) => t.set(img, egui::TextureOptions::NEAREST),
                    None => {
                        self.tex = Some(ctx.load_texture(
                            "focus_strip",
                            img,
                            egui::TextureOptions::NEAREST,
                        ))
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
                let (spec_lines, slit_lines) = if spec_is_y {
                    (&u.lines_y, &u.lines_x)
                } else {
                    (&u.lines_x, &u.lines_y)
                };
                (
                    choose(spec_lines, self.line_mode, self.picked_center),
                    choose(slit_lines, self.slit_line_mode, self.slit_picked_center),
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

            // Stage B live readouts: rolling "lucky" percentile, so the number
            // tracks the instrument's ceiling rather than the atmosphere.
            if let Some(lw) = u.limb_width {
                self.lucky_limb.push(lw);
            }
            if let Some(sc) = u.structure.all {
                self.lucky_struct.push(sc);
            }
            self.accumulate_capture(&spec, &slit, &u);

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
            self.frame_seq = self.frame_seq.wrapping_add(1);
        }
    }

    pub fn prepare_sun_search(
        &mut self,
        ctx: &egui::Context,
        requested_exposure_us: u32,
    ) -> Result<SearchCameraRestore, String> {
        if self.cameras.is_empty() {
            self.refresh_cameras();
        }
        if self
            .cameras
            .get(self.selected)
            .map(|camera| camera.backend == Backend::Synth)
            .unwrap_or(true)
        {
            if let Some(index) = self
                .cameras
                .iter()
                .position(|camera| camera.backend != Backend::Synth)
            {
                self.selected = index;
            }
        }
        let (camera_name, exposure_min, exposure_max) = self
            .cameras
            .get(self.selected)
            .filter(|camera| camera.backend != Backend::Synth)
            .map(|camera| {
                (
                    camera.name.clone(),
                    *camera.exposure_us.start(),
                    *camera.exposure_us.end(),
                )
            })
            .ok_or_else(|| "no hardware camera is available for Sun centering".to_owned())?;
        let target_exposure = requested_exposure_us.clamp(exposure_min, exposure_max);
        let restore = SearchCameraRestore {
            was_streaming: self.streaming,
            exposure_us: self.exposure_us,
            auto_exposure: self.auto_exposure,
        };
        if !self.streaming {
            self.start(ctx);
        }
        self.auto_exposure = false;
        self.exposure_us = target_exposure;
        self.send_cmd(FocusCmd::AutoExposure(false));
        self.send_cmd(FocusCmd::Exposure(target_exposure));
        self.status = format!(
            "Sun search: {} at {:.0} ms",
            camera_name,
            target_exposure as f64 / 1000.0
        );
        Ok(restore)
    }

    pub fn restore_after_sun_search(&mut self, restore: SearchCameraRestore) {
        self.exposure_us = restore.exposure_us;
        self.auto_exposure = restore.auto_exposure;
        self.send_cmd(FocusCmd::Exposure(restore.exposure_us));
        self.send_cmd(FocusCmd::AutoExposure(restore.auto_exposure));
        if !restore.was_streaming {
            self.stop();
        } else {
            self.status = "Sun search complete; camera settings restored".into();
        }
    }

    pub fn sun_signal_sample(&self) -> Option<(u64, f32)> {
        self.last.as_ref().map(|frame| (self.frame_seq, frame.peak))
    }

    /// Horizontal spectral-line candidates, expressed as sensor Y positions.
    ///
    /// SER acquisition uses one as the fixed centre of its vertical crop.
    pub fn vertical_anchor_lines(&self) -> Vec<(f64, f64)> {
        self.last
            .as_ref()
            .map(|frame| {
                let mut lines: Vec<(f64, f64)> = frame
                    .lines_y
                    .iter()
                    .filter(|line| line.depth > DEPTH_GATE)
                    .map(|line| (line.center, line.depth))
                    .collect();
                lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                lines
            })
            .unwrap_or_default()
    }

    pub fn current_frame_height(&self) -> Option<usize> {
        self.last.as_ref().map(|frame| frame.full_h)
    }

    /// Best available sensor height for bounding capture-band controls.
    ///
    /// Falls back to the SELECTED CAMERA's advertised height when no frame has
    /// arrived yet. Without this, anything gated on a live frame simply does
    /// not render before the preview is started — which is not a disabled
    /// control the user can reason about, it is a missing one.
    pub fn known_frame_height(&self) -> Option<usize> {
        let sensor_height = || {
            self.cameras
                .get(self.selected)
                .map(|c| c.max_height)
                .filter(|h| *h >= 32)
        };
        // With a live ROI the frames ARE the band, so bounding by them would
        // ratchet the capture height down to whatever was last applied and
        // leave no way to grow it again.
        if self.live_roi_active {
            return sensor_height().or_else(|| self.current_frame_height());
        }
        self.current_frame_height().or_else(sensor_height)
    }

    /// Apply or release the hardware ROI outside a recording.
    ///
    /// The checkbox alone only takes effect once a scan is already running,
    /// which gives no way to see or verify the capture band beforehand. This
    /// puts the sensor into the band now, so the preview shows exactly what a
    /// scan will record, at the frame rate it will record it.
    pub fn set_live_roi(
        &mut self,
        ctx: &egui::Context,
        capture_height: usize,
        anchor_y: f64,
        enable: bool,
    ) -> Result<(), String> {
        if self.recording {
            return Err("stop the recording before changing the sensor ROI".into());
        }
        if enable && !anchor_y.is_finite() {
            return Err("select a valid spectral-line anchor".into());
        }
        if enable && !self.streaming {
            self.start(ctx);
        }
        let Some(cmd) = &self.cmd else {
            return Err("start a camera before applying the ROI".into());
        };
        cmd.send(FocusCmd::LiveRoi {
            capture_height: capture_height.max(2),
            anchor_y,
            enable,
        })
        .map_err(|_| "camera worker is not running".to_owned())?;
        self.live_roi_status = if enable {
            "applying hardware ROI…".into()
        } else {
            "restoring the full sensor…".into()
        };
        Ok(())
    }

    pub fn slit_profile_sample(&self) -> Option<(u64, Vec<f32>)> {
        self.last
            .as_ref()
            .map(|frame| (self.frame_seq, frame.slit_cut.clone()))
    }

    pub fn start_ser_recording(
        &mut self,
        ctx: &egui::Context,
        path: PathBuf,
        capture_height: usize,
        anchor_y: f64,
        hw_roi: bool,
    ) -> Result<(), String> {
        if self.recording {
            return Err("a SER recording is already active".into());
        }
        if !anchor_y.is_finite() {
            return Err("select a valid spectral-line anchor".into());
        }
        if !self.streaming {
            self.start(ctx);
        }
        let Some(cmd) = &self.cmd else {
            return Err("start a camera before recording".into());
        };
        cmd.send(FocusCmd::StartSer {
            path: path.clone(),
            capture_height: capture_height.max(1),
            anchor_y,
            hw_roi,
        })
        .map_err(|_| "camera worker is not running".to_owned())?;
        self.recording = true;
        self.recorded_frames = 0;
        self.recording_path = Some(path);
        self.recording_status = "starting SER recording...".into();
        Ok(())
    }

    pub fn stop_ser_recording(&mut self) {
        if !self.recording {
            return;
        }
        self.recording_status = "stopping SER recording...".into();
        self.send_cmd(FocusCmd::StopSer);
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

    // -- V-curve capture ---------------------------------------------------

    /// The Stage B metric for one frame, per the selected metric.
    fn tele_frame_value(&self, u: &FocusUpdate) -> Option<f64> {
        match self.tele_metric {
            TeleMetric::LimbEdge => u.limb_width,
            TeleMetric::Structure => u.structure.all,
        }
    }

    /// Micrometer position for the active stage, parsed from its text field.
    fn active_position(&self) -> Option<f64> {
        let text = match self.stage {
            Stage::Spectrograph => &self.camera_pos_text,
            Stage::Telescope => &self.focuser_pos_text,
        };
        text.trim().parse::<f64>().ok().filter(|v| v.is_finite())
    }

    /// Start accumulating frames for one V-curve point.
    pub fn begin_capture(&mut self) {
        let Some(pos) = self.active_position() else {
            self.stage_status = "enter the micrometer reading first".into();
            return;
        };
        if !self.streaming {
            self.stage_status = "start the camera first".into();
            return;
        }
        let n = self.capture_frames.max(1);
        self.capture = Some(Capture {
            remaining: n,
            total: n,
            pos,
            spec: Vec::with_capacity(n),
            slit: Vec::with_capacity(n),
            best_spec_depth: 0.0,
            best_slit_depth: 0.0,
            tele: Vec::with_capacity(n),
            tele_top: Vec::with_capacity(n),
            tele_bot: Vec::with_capacity(n),
        });
        self.stage_status = format!("capturing {n} frames at {pos}…");
    }

    pub fn cancel_capture(&mut self) {
        self.capture = None;
        self.stage_status = "capture cancelled".into();
    }

    fn accumulate_capture(&mut self, spec: &Option<Fit>, slit: &Option<Fit>, u: &FocusUpdate) {
        let tele = self.tele_frame_value(u);
        let (top, bottom) = (u.structure.top, u.structure.bottom);
        let stage = self.stage;
        let Some(cap) = &mut self.capture else { return };

        match stage {
            Stage::Spectrograph => {
                // Only frames where both families actually produced a line are
                // usable; a dropout must not be averaged in as if it were data.
                if let Some(f) = spec {
                    cap.best_spec_depth = cap.best_spec_depth.max(f.depth);
                    if f.depth > DEPTH_GATE {
                        cap.spec.push(f.fwhm);
                    }
                }
                if let Some(f) = slit {
                    cap.best_slit_depth = cap.best_slit_depth.max(f.depth);
                    if f.depth > DEPTH_GATE {
                        cap.slit.push(f.fwhm);
                    }
                }
            }
            Stage::Telescope => {
                if let Some(v) = tele {
                    cap.tele.push(v);
                }
                if let Some(v) = top {
                    cap.tele_top.push(v);
                }
                if let Some(v) = bottom {
                    cap.tele_bot.push(v);
                }
            }
        }

        cap.remaining = cap.remaining.saturating_sub(1);
        if cap.remaining == 0 {
            self.commit_capture();
        }
    }

    fn commit_capture(&mut self) {
        let Some(cap) = self.capture.take() else { return };
        match self.stage {
            Stage::Spectrograph => {
                // Median over the burst: robust to the odd frame where the
                // fitter latched onto a neighbouring line.
                let spec = median(&cap.spec);
                let slit = median(&cap.slit);
                match (spec, slit) {
                    (Some(s), Some(k)) => {
                        self.curve_spec.push(cap.pos, s, cap.spec.len() as f64);
                        self.curve_slit.push(cap.pos, k, cap.slit.len() as f64);
                        self.stage_status = format!(
                            "captured @ {:.4}: spectral {s:.2} px, slit {k:.2} px ({}/{} frames)",
                            cap.pos,
                            cap.spec.len().min(cap.slit.len()),
                            cap.total
                        );
                    }
                    _ => {
                        self.stage_status = capture_failure(
                            spec.is_none(),
                            slit.is_none(),
                            cap.best_spec_depth,
                            cap.best_slit_depth,
                        );
                    }
                }
            }
            Stage::Telescope => {
                let want_min = self.tele_metric.want_min();
                // Lucky selection rather than the median: the burst is
                // seeing-limited, and the best frames measure the optics.
                let pick = |v: &Vec<f64>| lucky_of(v, want_min);
                match pick(&cap.tele) {
                    Some(t) => {
                        self.curve_tele.push(cap.pos, t, cap.tele.len() as f64);
                        if let Some(v) = pick(&cap.tele_top) {
                            self.curve_tele_top.push(cap.pos, v, cap.tele_top.len() as f64);
                        }
                        if let Some(v) = pick(&cap.tele_bot) {
                            self.curve_tele_bot.push(cap.pos, v, cap.tele_bot.len() as f64);
                        }
                        self.stage_status = format!(
                            "captured @ {:.4}: {} = {t:.3} ({}/{} frames)",
                            cap.pos,
                            self.tele_metric.label(),
                            cap.tele.len(),
                            cap.total
                        );
                    }
                    None => {
                        self.stage_status = match self.tele_metric {
                            TeleMetric::LimbEdge => {
                                "no limb on the slit in that burst — point at the limb, or switch \
                                 to along-slit contrast"
                                    .into()
                            }
                            TeleMetric::Structure => {
                                "no illuminated slit in that burst — check exposure".into()
                            }
                        };
                    }
                }
            }
        }
    }

    /// Set the Stage B curves' polarity to match the selected metric, and drop
    /// samples measured with the other one — they are not comparable.
    fn retarget_tele_curves(&mut self) {
        let want_min = self.tele_metric.want_min();
        self.curve_tele = VCurve::new(want_min);
        self.curve_tele_top = VCurve::new(want_min);
        self.curve_tele_bot = VCurve::new(want_min);
        self.lucky_limb.clear();
        self.lucky_struct.clear();
    }

    fn clear_stage_a(&mut self) {
        self.curve_spec.clear();
        self.curve_slit.clear();
        self.capture = None;
        self.stage_status = "Stage A sweep cleared".into();
    }

    fn clear_stage_b(&mut self) {
        self.curve_tele.clear();
        self.curve_tele_top.clear();
        self.curve_tele_bot.clear();
        self.capture = None;
        self.stage_status = "Stage B sweep cleared".into();
    }

    /// Bank the current sweep's Δ against its collimator reading, then clear the
    /// sweep so the next collimator setting starts fresh.
    fn record_null_point(&mut self) {
        let Some(sp) = vcurve::split(&self.curve_spec, &self.curve_slit) else {
            self.stage_status = "need a solved Δ before banking a point".into();
            return;
        };
        let Some(coll) = self
            .collimator_pos_text
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
        else {
            self.stage_status = "enter the collimator reading before banking".into();
            return;
        };
        self.null_points.push(NullPoint {
            collimator: coll,
            delta: sp.delta,
        });
        self.curve_spec.clear();
        self.curve_slit.clear();
        self.stage_status = format!(
            "banked Δ = {:+.4} at collimator {coll} ({} point(s))",
            sp.delta,
            self.null_points.len()
        );
    }

    fn save_current(&mut self) {
        let sp = vcurve::split(&self.curve_spec, &self.curve_slit);
        let rec = SavedFocus {
            camera: sp.map(|s| s.spatial.vertex),
            collimator: self
                .collimator_pos_text
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite()),
            delta: sp.map(|s| s.delta),
            focuser: self.curve_tele.fit().map(|f| f.vertex),
        };
        match rec.save() {
            Ok(path) => {
                self.stage_status = format!("saved to {}", path.display());
                self.saved = Some(rec);
            }
            Err(e) => self.stage_status = format!("save failed: {e}"),
        }
    }

    // -- UI ----------------------------------------------------------------

    pub fn controls_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(8.0);
        ui.heading("Focus assistant");
        ui.label(
            egui::RichText::new(
                "Minimise FWHM — spectral (camera focus) and slit (scope-on-slit).",
            )
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
            .selected_text(
                names
                    .get(self.selected)
                    .cloned()
                    .unwrap_or_else(|| "—".into()),
            )
            .show_ui(ui, |ui| {
                for (i, n) in names.iter().enumerate() {
                    if ui.selectable_value(&mut self.selected, i, n).clicked() {
                        self.camera_chosen_by_user = true;
                    }
                }
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let can_start = !self.cameras.is_empty();
            if !self.streaming {
                if ui
                    .add_enabled(can_start, egui::Button::new("▶ Start"))
                    .clicked()
                {
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

        if ui
            .checkbox(&mut self.auto_exposure, "auto-exposure")
            .changed()
        {
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
                egui::Slider::new(&mut self.exposure_us, emin..=emax.min(2_000_000))
                    .logarithmic(true),
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
        ui.label(if gmin >= 100 {
            "gain (%, 100 = 1×)"
        } else {
            "gain"
        });
        // Device ranges can run to thousands of percent (queried at open);
        // keep the everyday low end usable by going logarithmic when wide.
        let wide_gain = gmin > 0 && gmax / gmin >= 20;
        if ui
            .add_enabled(
                manual,
                egui::Slider::new(&mut self.gain, gmin..=gmax).logarithmic(wide_gain),
            )
            .changed()
        {
            self.send_cmd(FocusCmd::Gain(self.gain));
        }

        ui.add_space(8.0);
        let mut axis_changed = false;
        ui.horizontal(|ui| {
            ui.label("dispersion axis:");
            axis_changed |= ui
                .selectable_value(&mut self.dispersion, DispAxis::Vertical, "⇕ vertical")
                .clicked();
            axis_changed |= ui
                .selectable_value(&mut self.dispersion, DispAxis::Horizontal, "⇔ horizontal")
                .clicked();
        });
        if axis_changed {
            // Stage A only relabels, but Stage B genuinely cuts the frame the
            // other way, so the worker has to be told.
            self.send_cmd(FocusCmd::Dispersion(self.dispersion == DispAxis::Horizontal));
            self.lucky_limb.clear();
            self.lucky_struct.clear();
        }
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
            egui::Grid::new("optics")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
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
                    ui.add(
                        egui::DragValue::new(&mut self.pixel_um)
                            .speed(0.05)
                            .range(0.5..=20.0),
                    );
                    ui.end_row();
                    ui.label("central λ (Å)");
                    ui.add(
                        egui::DragValue::new(&mut self.central_wavelength).range(3000.0..=9000.0),
                    );
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
            ui.label("spectral target:");
        });
        let spec_changed = target_line_ui(
            ui,
            "spectral",
            &mut self.line_mode,
            &mut self.picked_center,
            SPECTRAL_COLOR,
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("slit target:");
        });
        let slit_changed = target_line_ui(
            ui,
            "slit",
            &mut self.slit_line_mode,
            &mut self.slit_picked_center,
            SLIT_COLOR,
        );
        if spec_changed || slit_changed {
            self.reset_holds();
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

        readout(
            ui,
            "Spectral line (dispersion)",
            SPECTRAL_COLOR,
            spectral_fit,
            spectral_min,
            Some(a_per_px),
        );
        ui.add_space(6.0);
        readout(
            ui,
            "Slit jaws / dust (spatial)",
            SLIT_COLOR,
            slit_fit,
            slit_min,
            None,
        );

        if let Some(l) = &self.last {
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "frame {}×{}  mean {:.0}  ·  peak {:.0}  ·  {} continuum px",
                    l.full_w, l.full_h, l.mean, l.peak, l.n_continuum
                ))
                .small()
                .weak(),
            );
        }

        ui.add_space(12.0);
        ui.separator();
        self.stage_ui(ui);
    }

    // -- three-stage procedure UI ------------------------------------------

    fn stage_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.heading("Focus procedure");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.stage, Stage::Spectrograph, "A · spectrograph");
            ui.selectable_value(&mut self.stage, Stage::Telescope, "B · telescope");
        });

        match self.stage {
            Stage::Spectrograph => self.stage_a_ui(ui),
            Stage::Telescope => self.stage_b_ui(ui),
        }

        if !self.stage_status.is_empty() {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(&self.stage_status).small().weak());
        }

        if let Some(s) = self.saved.filter(|s| !s.is_empty()) {
            ui.add_space(6.0);
            let mut parts = Vec::new();
            if let Some(v) = s.collimator {
                parts.push(format!("collimator {v}"));
            }
            if let Some(v) = s.camera {
                parts.push(format!("camera {v:.4}"));
            }
            if let Some(v) = s.delta {
                parts.push(format!("Δ {v:+.4}"));
            }
            if let Some(v) = s.focuser {
                parts.push(format!("focuser {v:.4}"));
            }
            ui.label(
                egui::RichText::new(format!("saved: {}", parts.join("  ·  ")))
                    .small()
                    .color(egui::Color32::LIGHT_GREEN),
            );
        }
    }

    fn stage_a_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new(
                "Telescope-blind: both metrics live in the slit plane, so this runs at the \
                 bench with a lamp on the slit. Step the CAMERA micrometer; the two minima \
                 coincide only when the collimator is truly collimating.",
            )
            .small()
            .weak(),
        );
        ui.add_space(6.0);

        // Label above the field, matching the exposure/gain controls higher up
        // this panel. A two-column Grid collapses its label column in the side
        // rail and wraps "collimator reading" one character per line.
        ui.label("collimator reading (fixed this sweep)");
        ui.add(
            egui::TextEdit::singleline(&mut self.collimator_pos_text)
                .desired_width(f32::INFINITY)
                .hint_text("e.g. 12.40"),
        );
        ui.add_space(4.0);
        ui.label("camera reading (stepped)");
        ui.add(
            egui::TextEdit::singleline(&mut self.camera_pos_text)
                .desired_width(f32::INFINITY)
                .hint_text("e.g. 8.15"),
        );

        self.capture_row_ui(ui);
        let n = self.curve_spec.samples.len().min(self.curve_slit.samples.len());
        ui.label(
            egui::RichText::new(format!("{n} sample(s) in this sweep"))
                .small()
                .weak(),
        );

        ui.add_space(8.0);
        match vcurve::split(&self.curve_spec, &self.curve_slit) {
            Some(sp) => {
                egui::Frame::group(ui.style())
                    .fill(ui.visuals().faint_bg_color)
                    .show(ui, |ui| {
                        vertex_line(ui, "spectral min", SPECTRAL_COLOR, &sp.spectral);
                        vertex_line(ui, "slit min", SLIT_COLOR, &sp.spatial);
                        ui.add_space(4.0);
                        let trusted = sp.spectral.trustworthy() && sp.spatial.trustworthy();
                        let (verdict, color) = if !trusted {
                            ("not solved yet", WARN_COLOR)
                        } else if sp.nulled() {
                            ("collimated — Δ is zero within 1σ", egui::Color32::LIGHT_GREEN)
                        } else {
                            ("astigmatic — move the collimator", WARN_COLOR)
                        };
                        let sigma = if sp.delta_sigma.is_finite() {
                            format!(" ± {:.4}", sp.delta_sigma)
                        } else {
                            String::new()
                        };
                        ui.label(
                            egui::RichText::new(format!("Δ = {:+.4}{sigma}", sp.delta))
                                .size(24.0)
                                .strong()
                                .color(color),
                        );
                        ui.label(egui::RichText::new(verdict).small().color(color));
                        for note in [
                            fit_note(&sp.spectral, "spectral"),
                            fit_note(&sp.spatial, "slit"),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            ui.label(egui::RichText::new(note).small().color(WARN_COLOR));
                        }
                    });
            }
            None => {
                ui.label(
                    egui::RichText::new(
                        "Δ needs ≥3 camera positions on both curves, spanning each minimum.",
                    )
                    .small()
                    .weak(),
                );
            }
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("bank Δ at this collimator").clicked() {
                self.record_null_point();
            }
            if ui.button("clear sweep").clicked() {
                self.clear_stage_a();
            }
        });

        if !self.null_points.is_empty() {
            ui.add_space(6.0);
            for p in &self.null_points {
                ui.label(
                    egui::RichText::new(format!(
                        "collimator {:.4}  →  Δ {:+.4}",
                        p.collimator, p.delta
                    ))
                    .small()
                    .weak(),
                );
            }
            match vcurve::solve_null(&self.null_points) {
                Some(sol) => {
                    ui.label(
                        egui::RichText::new(format!("set collimator to {:.4}", sol.collimator))
                            .size(18.0)
                            .strong()
                            .color(egui::Color32::LIGHT_GREEN),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "dΔ/dcollimator = {:+.4} from {} point(s){}",
                            sol.gain,
                            sol.n,
                            if sol.bracketed {
                                ""
                            } else {
                                " · extrapolated, expect one more iteration"
                            }
                        ))
                        .small()
                        .weak(),
                    );
                }
                None => {
                    ui.label(
                        egui::RichText::new(
                            "bank a second sweep at a different collimator setting to solve for \
                             the null (this also measures the sign, so it is never assumed)",
                        )
                        .small()
                        .weak(),
                    );
                }
            }
            if ui.button("clear banked points").clicked() {
                self.null_points.clear();
                self.stage_status = "banked points cleared".into();
            }
        }

        ui.add_space(6.0);
        if ui.button("💾 save converged settings").clicked() {
            self.save_current();
        }
    }

    fn stage_b_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new(
                "Run only after Stage A is closed — the telescope cannot correct grating \
                 astigmatism, so against a mis-collimated spectrograph it just finds a \
                 compromise. Measured on continuum columns, not the line core.",
            )
            .small()
            .weak(),
        );
        ui.add_space(6.0);

        let before = self.tele_metric;
        ui.horizontal(|ui| {
            ui.label("metric:");
            ui.selectable_value(&mut self.tele_metric, TeleMetric::LimbEdge, "limb edge");
            ui.selectable_value(&mut self.tele_metric, TeleMetric::Structure, "contrast");
        });
        if before != self.tele_metric {
            self.retarget_tele_curves();
            self.stage_status = "metric changed — Stage B sweep reset (values not comparable)".into();
        }
        ui.label(
            egui::RichText::new(match self.tele_metric {
                TeleMetric::LimbEdge => {
                    "Solar limb as a knife edge. Dust-immune, so trust this one. Needs the limb \
                     on the slit."
                }
                TeleMetric::Structure => {
                    "High-passed along-slit contrast. Always available, but slit dust adds a \
                     constant pedestal — the peak position is still right, the curve is just \
                     shallower. Only this metric supports the top/bottom split."
                }
            })
            .small()
            .weak(),
        );

        ui.add_space(6.0);
        let want_min = self.tele_metric.want_min();
        let live = match self.tele_metric {
            TeleMetric::LimbEdge => self.lucky_limb.lucky(want_min),
            TeleMetric::Structure => self.lucky_struct.lucky(want_min),
        };
        let n_lucky = match self.tele_metric {
            TeleMetric::LimbEdge => self.lucky_limb.len(),
            TeleMetric::Structure => self.lucky_struct.len(),
        };
        egui::Frame::group(ui.style())
            .fill(ui.visuals().faint_bg_color)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(format!("{} · best decile", self.tele_metric.label()))
                        .small()
                        .weak(),
                );
                match live {
                    Some(v) => ui.label(
                        egui::RichText::new(format!("{v:.3}"))
                            .size(26.0)
                            .strong()
                            .color(TELE_COLOR),
                    ),
                    None => ui.label(egui::RichText::new("— no signal —").size(20.0).weak()),
                };
                ui.label(
                    egui::RichText::new(format!("over {n_lucky} frames"))
                        .small()
                        .weak(),
                );
            });

        ui.add_space(6.0);
        ui.label("focuser reading (stepped)");
        ui.add(
            egui::TextEdit::singleline(&mut self.focuser_pos_text)
                .desired_width(f32::INFINITY)
                .hint_text("e.g. 4.80"),
        );
        self.capture_row_ui(ui);
        ui.label(
            egui::RichText::new(format!("{} sample(s)", self.curve_tele.samples.len()))
                .small()
                .weak(),
        );

        ui.add_space(8.0);
        match self.curve_tele.fit() {
            Some(f) => {
                egui::Frame::group(ui.style())
                    .fill(ui.visuals().faint_bg_color)
                    .show(ui, |ui| {
                        vertex_line(ui, "best focus", TELE_COLOR, &f);
                        if let Some(note) = fit_note(&f, "telescope") {
                            ui.label(egui::RichText::new(note).small().color(WARN_COLOR));
                        }
                    });
            }
            None => {
                ui.label(
                    egui::RichText::new("needs ≥3 focuser positions spanning the extremum")
                        .small()
                        .weak(),
                );
            }
        }

        if self.tele_metric == TeleMetric::Structure {
            ui.add_space(6.0);
            match (self.curve_tele_top.fit(), self.curve_tele_bot.fit()) {
                (Some(t), Some(b)) if t.trustworthy() && b.trustworthy() => {
                    let split = t.vertex - b.vertex;
                    let tol = (t.vertex_sigma.max(0.0) + b.vertex_sigma.max(0.0)).max(0.0);
                    let flat = tol.is_finite() && split.abs() <= tol;
                    ui.label(
                        egui::RichText::new(format!(
                            "top {:.4} · bottom {:.4} · split {split:+.4}",
                            t.vertex, b.vertex
                        ))
                        .small(),
                    );
                    ui.label(
                        egui::RichText::new(if flat {
                            "slit is flat to the field — no tilt or curvature to chase"
                        } else {
                            "top and bottom focus at different positions: field curvature or \
                             slit tilt, not focus. No single setting fixes it — focus for \
                             mid-disk radius, or fit a flattener."
                        })
                        .small()
                        .color(if flat { egui::Color32::LIGHT_GREEN } else { WARN_COLOR }),
                    );
                }
                _ => {
                    ui.label(
                        egui::RichText::new(
                            "top/bottom split: needs ≥3 positions with the slit lit end to end",
                        )
                        .small()
                        .weak(),
                    );
                }
            }
        }

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button("clear sweep").clicked() {
                self.clear_stage_b();
            }
            if ui.button("💾 save converged settings").clicked() {
                self.save_current();
            }
        });
    }

    /// Capture / undo row, shared by both stages.
    fn capture_row_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        let in_progress = self.capture.as_ref().map(|c| (c.total - c.remaining, c.total));
        ui.horizontal(|ui| match in_progress {
            Some((done, total)) => {
                ui.add(
                    egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                        .desired_width(150.0)
                        .text(format!("{done}/{total}")),
                );
                if ui.button("cancel").clicked() {
                    self.cancel_capture();
                }
            }
            None => {
                if ui.button("◉ capture").clicked() {
                    self.begin_capture();
                }
                if ui.button("undo last").clicked() {
                    match self.stage {
                        Stage::Spectrograph => {
                            self.curve_spec.undo();
                            self.curve_slit.undo();
                        }
                        Stage::Telescope => {
                            self.curve_tele.undo();
                            self.curve_tele_top.undo();
                            self.curve_tele_bot.undo();
                        }
                    }
                    self.stage_status = "last sample removed".into();
                }
                ui.add(
                    egui::DragValue::new(&mut self.capture_frames)
                        .range(5..=300)
                        .prefix("frames "),
                );
            }
        });
    }

    /// Draw the shared live-camera texture without opening another camera.
    ///
    /// The Mount tab uses this while auto-centering; Focus continues to add
    /// its analysis plots below the same image.
    pub fn camera_preview_ui(&self, ui: &mut egui::Ui, max_height: f32) -> bool {
        if let Some(tex) = &self.tex {
            let avail = ui.available_width().max(1.0);
            let aspect = tex.aspect_ratio();
            let h = (avail / aspect).min(260.0);
            let h = h.min(max_height).max(1.0);
            let w = (h * aspect).min(avail);
            ui.vertical_centered(|ui| {
                ui.add(egui::Image::new(tex).fit_to_exact_size(egui::vec2(w, h)));
            });
            true
        } else {
            let placeholder_height = max_height.min(140.0).max(60.0);
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), placeholder_height),
                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                |ui| {
                    ui.label(
                        egui::RichText::new("Start a camera to see the live spectrum.")
                            .size(18.0)
                            .weak(),
                    );
                },
            );
            false
        }
    }

    /// Draw the acquisition preview with sensor-Y spectral-line overlays.
    /// Returns the nearest detected line when the user clicks the image.
    pub fn acquisition_preview_ui(
        &self,
        ui: &mut egui::Ui,
        max_height: f32,
        selected_y: Option<f64>,
        capture_height: usize,
        selectable: bool,
    ) -> Option<f64> {
        let Some(tex) = &self.tex else {
            self.camera_preview_ui(ui, max_height);
            return None;
        };
        let Some(frame_height) = self.current_frame_height() else {
            self.camera_preview_ui(ui, max_height);
            return None;
        };
        let candidates = self.vertical_anchor_lines();
        let hw_roi_active = self
            .last
            .as_ref()
            .map(|update| update.hw_roi_active)
            .unwrap_or(false);
        // While the sensor is cropped the detected lines are in BAND rows,
        // not sensor rows, so accepting a click here would silently rewrite
        // the anchor into the wrong coordinate space. Release the ROI to
        // pick a different line.
        let selectable = selectable && !hw_roi_active;
        ui.vertical_centered(|ui| {
            let avail = ui.available_width().max(1.0);
            let aspect = tex.aspect_ratio();
            let height = (avail / aspect).min(max_height).max(1.0);
            let width = (height * aspect).min(avail);
            let sense = if selectable {
                egui::Sense::click()
            } else {
                egui::Sense::hover()
            };
            let response = ui
                .add(
                    egui::Image::new(tex)
                        .fit_to_exact_size(egui::vec2(width, height))
                        .sense(sense),
                )
                .on_hover_text(if selectable {
                    "Click a horizontal spectral line to select the acquisition anchor"
                } else {
                    "The spectral-line anchor is locked during acquisition"
                });
            let rect = response.rect;
            let sensor_to_screen_y = |sensor_y: f64| {
                rect.top()
                    + rect.height()
                        * (sensor_y / frame_height.max(1) as f64).clamp(0.0, 1.0) as f32
            };

            if hw_roi_active {
                ui.painter().text(
                    egui::pos2(rect.left() + 8.0, rect.top() + 6.0),
                    egui::Align2::LEFT_TOP,
                    "hardware ROI — the view is the capture band",
                    egui::FontId::proportional(12.0),
                    egui::Color32::from_rgb(255, 205, 75),
                );
            }
            for &(line_y, _) in candidates.iter().filter(|_| !hw_roi_active) {
                let y = sensor_to_screen_y(line_y);
                ui.painter().hline(
                    rect.x_range(),
                    y,
                    egui::Stroke::new(1.0_f32, SPECTRAL_COLOR.gamma_multiply(0.45)),
                );
            }
            if let Some(anchor) = selected_y.filter(|_| !hw_roi_active) {
                let half_crop = capture_height as f64 / 2.0;
                let y0 = sensor_to_screen_y(anchor - half_crop);
                let y1 = sensor_to_screen_y(anchor + half_crop);
                let crop_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left(), y0.min(y1)),
                    egui::pos2(rect.right(), y0.max(y1)),
                )
                .intersect(rect);
                ui.painter().rect_filled(
                    crop_rect,
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(255, 190, 70, 18),
                );
                let y = sensor_to_screen_y(anchor);
                ui.painter().hline(
                    rect.x_range(),
                    y,
                    egui::Stroke::new(2.5_f32, egui::Color32::from_rgb(255, 205, 75)),
                );
                ui.painter().text(
                    egui::pos2(rect.left() + 6.0, y - 4.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("selected Y {anchor:.1}"),
                    egui::FontId::proportional(12.0),
                    egui::Color32::from_rgb(255, 220, 110),
                );
            }

            if selectable && response.clicked() {
                if let Some(pointer) = response.interact_pointer_pos() {
                    let sensor_y =
                        ((pointer.y - rect.top()) / rect.height()) as f64 * frame_height as f64;
                    return candidates
                        .iter()
                        .min_by(|a, b| {
                            (a.0 - sensor_y)
                                .abs()
                                .total_cmp(&(b.0 - sensor_y).abs())
                        })
                        .map(|line| line.0);
                }
            }
            None
        })
        .inner
    }

    /// Acquisition-specific spectral profile. The x-axis is sensor Y because
    /// guided acquisition requires vertical dispersion.
    pub fn acquisition_spectral_profile_ui(
        &self,
        ui: &mut egui::Ui,
        selected_y: Option<f64>,
    ) -> Option<f64> {
        let last = self.last.as_ref()?;
        let profile = last.prof_y.clone();
        let candidates: Vec<f64> = last.lines_y.iter().map(|line| line.center).collect();
        let labels: Vec<(f64, String)> = if self.spectral_is_y() {
            self.labels
                .iter()
                .map(|line| (line.x, format!("{} {:.1}", line.element, line.wavelength)))
                .collect()
        } else {
            Vec::new()
        };
        profile_plot(
            ui,
            "acquire_spectral",
            &profile,
            None,
            &candidates,
            selected_y,
            &labels,
            SPECTRAL_COLOR,
            170.0,
        )
    }

    pub fn view_ui(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        if !self.camera_preview_ui(ui, 260.0) {
            return;
        }

        let spec_is_y = self.spectral_is_y();
        // Clone what the plots need so the borrow of self.last is released before
        // a click can mutate self.picked_center.
        let (spec_prof, slit_prof, spec_cands, slit_cands) = {
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
            let slit_cands: Vec<f64> = if spec_is_y {
                last.lines_x.iter().map(|f| f.center).collect()
            } else {
                last.lines_y.iter().map(|f| f.center).collect()
            };
            (sp, kp, cands, slit_cands)
        };
        let spec_fit = self.sel_spectral;
        let slit_fit = self.sel_slit;
        let picked = self.picked_center;
        let slit_picked = self.slit_picked_center;
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
        ui.label(
            egui::RichText::new("Slit profile (across the slit) — click to lock a dust line")
                .small()
                .color(SLIT_COLOR),
        );
        if let Some(x) = profile_plot(
            ui,
            "focus_slit",
            &slit_prof,
            slit_fit,
            &slit_cands,
            slit_picked,
            &[],
            SLIT_COLOR,
            150.0,
        ) {
            self.slit_picked_center = Some(x);
            self.slit_line_mode = LineMode::Manual;
            self.reset_holds();
        }

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
            Plot::new("focus_trend")
                .height(110.0)
                .allow_scroll(false)
                .show(ui, |p| {
                    let sp: PlotPoints =
                        sh.iter().enumerate().map(|(x, &y)| [x as f64, y]).collect();
                    p.line(Line::new(sp).color(SPECTRAL_COLOR).name("spectral"));
                    let kp: PlotPoints =
                        kh.iter().enumerate().map(|(x, &y)| [x as f64, y]).collect();
                    p.line(Line::new(kp).color(SLIT_COLOR).name("slit"));
                    if smin.is_finite() {
                        p.hline(HLine::new(smin).color(SPECTRAL_COLOR));
                    }
                    if kmin.is_finite() {
                        p.hline(HLine::new(kmin).color(SLIT_COLOR));
                    }
                });
        }

        // Stage B works on the continuum cut along the slit, not on either of
        // the profiles above — show it, so "no limb on the slit" is visible
        // rather than inferred.
        if self.stage == Stage::Telescope {
            if let Some(cut) = self.last.as_ref().map(|l| l.slit_cut.clone()) {
                if !cut.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Continuum cut along the slit (Stage B)")
                            .small()
                            .color(TELE_COLOR),
                    );
                    profile_plot(ui, "focus_slitcut", &cut, None, &[], None, &[], TELE_COLOR, 150.0);
                }
            }
        }

        // The V-curves being built for the active stage.
        match self.stage {
            Stage::Spectrograph => {
                if !self.curve_spec.is_empty() || !self.curve_slit.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "Stage A V-curves — FWHM (px) vs camera micrometer; the gap between \
                             the two vertices is Δ",
                        )
                        .small()
                        .weak(),
                    );
                    vcurve_plot(
                        ui,
                        "vcurve_stage_a",
                        &[
                            ("spectral", SPECTRAL_COLOR, &self.curve_spec),
                            ("slit", SLIT_COLOR, &self.curve_slit),
                        ],
                    );
                }
            }
            Stage::Telescope => {
                if !self.curve_tele.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "Stage B V-curve — {} vs focuser",
                            self.tele_metric.label()
                        ))
                        .small()
                        .weak(),
                    );
                    let mut series: Vec<(&str, egui::Color32, &VCurve)> =
                        vec![("focus", TELE_COLOR, &self.curve_tele)];
                    if self.tele_metric == TeleMetric::Structure {
                        series.push(("top", SPECTRAL_COLOR, &self.curve_tele_top));
                        series.push(("bottom", SLIT_COLOR, &self.curve_tele_bot));
                    }
                    vcurve_plot(ui, "vcurve_stage_b", &series);
                }
            }
        }
    }
}

/// Why a Stage A burst produced no sample.
///
/// Both families must succeed — a position measured on only one curve cannot
/// contribute to a difference between them — so a rejection has to say *which*
/// one failed, and whether it failed for want of depth or was never detected at
/// all. Those need opposite responses.
fn capture_failure(
    spec_failed: bool,
    slit_failed: bool,
    best_spec_depth: f64,
    best_slit_depth: f64,
) -> String {
    let gate = DEPTH_GATE * 100.0;
    let detail = |name: &str, best: f64, hint: &str| {
        if best <= 0.0 {
            format!("no {name} detected at all — {hint}")
        } else {
            format!(
                "{name} only {:.1}% deep, needs {gate:.0}% — {hint}",
                best * 100.0
            )
        }
    };
    match (spec_failed, slit_failed) {
        (true, true) => format!(
            "nothing usable in that burst: {}; {}",
            detail(
                "spectral line",
                best_spec_depth,
                "only absorption lines are detected, so an emission source (neon) will not work"
            ),
            detail("slit-jaw line", best_slit_depth, "jaws may be too clean")
        ),
        (true, false) => detail(
            "spectral line",
            best_spec_depth,
            "raise exposure without clipping, or click a deeper line in the spectral plot.              Only absorption lines are detected — an emission source will not register",
        ),
        (false, true) => detail(
            "slit-jaw line",
            best_slit_depth,
            "clean jaws give no signal: tape a hair across a jaw, or move the slit end into frame",
        ),
        (false, false) => {
            "burst produced no sample despite both families reporting lines — try more frames"
                .into()
        }
    }
}

fn vertex_line(ui: &mut egui::Ui, label: &str, color: egui::Color32, f: &ParabolaFit) {
    let sigma = if f.vertex_sigma.is_finite() {
        format!(" ± {:.4}", f.vertex_sigma)
    } else {
        String::new()
    };
    ui.label(
        egui::RichText::new(format!(
            "{label}: {:.4}{sigma}   (n={}, rms {:.3})",
            f.vertex, f.n, f.rms
        ))
        .small()
        .color(color),
    );
}

/// What is wrong with a fit, if anything — so the panel says "sample further
/// out" instead of quietly reporting a number that means nothing.
fn fit_note(f: &ParabolaFit, name: &str) -> Option<String> {
    if !f.shape_ok {
        Some(format!(
            "{name}: curve bends the wrong way — the range is probably too small to see the \
             extremum, or the metric has no signal"
        ))
    } else if !f.bracketed {
        Some(format!(
            "{name}: extremum is extrapolated beyond the samples — step past it and capture again"
        ))
    } else if f.n < 4 {
        Some(format!("{name}: a 4th sample gives an uncertainty"))
    } else {
        None
    }
}

/// Sample points plus their fitted parabolas. The parabola is reconstructed
/// from (vertex, curvature, extremum): y = extremum + ½·curvature·(x − vertex)².
fn vcurve_plot(ui: &mut egui::Ui, id: &str, series: &[(&str, egui::Color32, &VCurve)]) {
    if series.iter().all(|(_, _, c)| c.is_empty()) {
        return;
    }
    Plot::new(id)
        .height(170.0)
        .allow_scroll(false)
        .show(ui, |p| {
            for (name, color, curve) in series {
                if curve.is_empty() {
                    continue;
                }
                let pts: Vec<[f64; 2]> = curve.samples.iter().map(|s| [s.pos, s.value]).collect();
                p.points(
                    Points::new(PlotPoints::from(pts))
                        .color(*color)
                        .radius(3.5_f32)
                        .name(*name),
                );
                let Some(f) = curve.fit() else { continue };
                let lo = curve.samples.iter().map(|s| s.pos).fold(f64::MAX, f64::min);
                let hi = curve.samples.iter().map(|s| s.pos).fold(f64::MIN, f64::max);
                // Extend a little past the samples so a near-edge vertex is visible.
                let pad = 0.1 * (hi - lo).abs().max(1e-9);
                let (a, b) = (lo - pad, hi + pad);
                let curve_pts: PlotPoints = (0..=100)
                    .map(|i| {
                        let x = a + (b - a) * i as f64 / 100.0;
                        let d = x - f.vertex;
                        [x, f.extremum + 0.5 * f.curvature * d * d]
                    })
                    .collect();
                p.line(Line::new(curve_pts).color(*color).name(*name));
                if f.shape_ok {
                    p.vline(VLine::new(f.vertex).color(*color));
                }
            }
        });
}

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    Some(if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    })
}

/// Best decile of a burst, for the metric's polarity. Shares the live
/// readout's selection so a captured sample means the same thing as the number
/// the user watched before pressing capture.
fn lucky_of(v: &[f64], want_min: bool) -> Option<f64> {
    focusmetrics::lucky_mean(v, want_min, focusmetrics::LUCKY_FRACTION)
}

// -- converged-setting persistence -----------------------------------------

/// The converged readings from a closed procedure.
///
/// Stage A drifts far more slowly than Stage B — a short metal path with no
/// tube — so on a normal morning only the telescope is re-focused, and Stage A
/// is *verified* against these numbers rather than re-derived.
#[derive(Clone, Copy, Default)]
struct SavedFocus {
    camera: Option<f64>,
    collimator: Option<f64>,
    delta: Option<f64>,
    focuser: Option<f64>,
}

fn saved_focus_path() -> Option<std::path::PathBuf> {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(std::path::PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(|h| std::path::PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))
    }?;
    Some(base.join("GhostSun").join("focus.txt"))
}

impl SavedFocus {
    /// Hand-rolled key=value rather than a serde dependency: four optional
    /// numbers do not justify one, and the file stays readable at the scope.
    fn save(&self) -> Result<std::path::PathBuf, String> {
        let path = saved_focus_path().ok_or("no config directory")?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let mut s = String::from("# GhostSun converged focus settings\n");
        for (k, v) in [
            ("camera", self.camera),
            ("collimator", self.collimator),
            ("delta", self.delta),
            ("focuser", self.focuser),
        ] {
            if let Some(v) = v {
                s.push_str(&format!("{k}={v}\n"));
            }
        }
        std::fs::write(&path, s).map_err(|e| e.to_string())?;
        Ok(path)
    }

    fn load() -> Option<SavedFocus> {
        let text = std::fs::read_to_string(saved_focus_path()?).ok()?;
        let mut out = SavedFocus::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let Ok(v) = v.trim().parse::<f64>() else {
                continue;
            };
            match k.trim() {
                "camera" => out.camera = Some(v),
                "collimator" => out.collimator = Some(v),
                "delta" => out.delta = Some(v),
                "focuser" => out.focuser = Some(v),
                _ => {}
            }
        }
        Some(out)
    }

    fn is_empty(&self) -> bool {
        self.camera.is_none()
            && self.collimator.is_none()
            && self.delta.is_none()
            && self.focuser.is_none()
    }
}

/// Target-line selector for one family. Returns true when the selection changed,
/// so the caller can reset the min-holds — a held minimum measured on a
/// different feature is meaningless.
fn target_line_ui(
    ui: &mut egui::Ui,
    family: &str,
    mode: &mut LineMode,
    picked: &mut Option<f64>,
    color: egui::Color32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui
            .selectable_value(mode, LineMode::Narrowest, "narrowest")
            .clicked();
        changed |= ui
            .selectable_value(mode, LineMode::Deepest, "deepest")
            .clicked();
        if *mode == LineMode::Manual {
            let _ = ui.selectable_label(true, "picked");
        }
    });
    if changed {
        *picked = None;
    }
    if *mode == LineMode::Manual {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("locked @ {:.0} px", picked.unwrap_or(0.0)))
                    .small()
                    .color(color),
            );
            if ui.small_button("clear").clicked() {
                *mode = LineMode::Narrowest;
                *picked = None;
                changed = true;
            }
        });
    } else {
        ui.label(
            egui::RichText::new(format!("click the {family} plot to lock a line"))
                .small()
                .weak(),
        );
    }
    changed
}

fn readout(
    ui: &mut egui::Ui,
    title: &str,
    color: egui::Color32,
    fit: Option<Fit>,
    min_hold: f64,
    a_per_px: Option<f64>,
) {
    egui::Frame::group(ui.style())
        .fill(ui.visuals().faint_bg_color)
        .show(ui, |ui| {
            ui.label(egui::RichText::new(title).small().weak());
            match fit {
                Some(f) if f.depth > DEPTH_GATE => {
                    ui.label(
                        egui::RichText::new(format!("{:.2} px", f.fwhm))
                            .size(26.0)
                            .strong()
                            .color(color),
                    );
                    let extra = match a_per_px {
                        Some(a) => {
                            format!("{:.3} Å   ·   depth {:.0}%", f.fwhm * a, f.depth * 100.0)
                        }
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
            ui.label(
                egui::RichText::new(mh)
                    .small()
                    .color(egui::Color32::LIGHT_GREEN),
            );
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
    let mut plot = Plot::new(id)
        .height(height)
        .allow_scroll(false)
        .show_axes([false, true]);
    if !labels.is_empty() {
        // Reserve headroom above the profile so the labels aren't clipped.
        plot = plot.include_y(ymin).include_y(ymax + 0.34 * span);
    }
    plot.show(ui, |p| {
        let pts: PlotPoints = profile
            .iter()
            .enumerate()
            .map(|(x, &y)| [x as f64, y as f64])
            .collect();
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
                Text::new(
                    PlotPoint::new(*x, ly),
                    egui::RichText::new(txt).size(11.0).color(label_col),
                )
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
                        [
                            x,
                            f.continuum - amp * (-dx * dx / (2.0 * f.sigma * f.sigma)).exp(),
                        ]
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
    dispersion_horizontal: bool,
    frame_pending: Arc<AtomicBool>,
) {
    let mut disp_h = dispersion_horizontal;
    let mut cam = match open(&info) {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(FocusMsg::Error(e.to_string()));
            return;
        }
    };
    crate::applog!(
        "camera: opened {} ({}x{}) exposure={exposure_us}us gain={gain}",
        info.name,
        info.max_width,
        info.max_height
    );
    let _ = tx.send(FocusMsg::Opened(cam.info().clone()));
    cam.set_exposure_us(exposure_us).ok();
    cam.set_gain(gain).ok();
    cam.set_auto_exposure(auto_exposure).ok();
    if let Err(e) = cam.start() {
        crate::applog!("camera: initial start FAILED: {e}");
        let _ = tx.send(FocusMsg::Error(e.to_string()));
        return;
    }
    let mut pending_ser: Option<SerRequest> = None;
    let mut active_ser: Option<ActiveSer> = None;
    // The band when a live (outside-recording) ROI is applied. It outlives
    // recordings: a recording that ends must return to the band the user is
    // working in, not to the full sensor.
    let mut live_roi: Option<Roi> = None;
    // What the sensor is actually reading right now. The camera opens full
    // frame, and every change goes through ensure_roi so a redundant or
    // band-to-band transition can never stall the stream.
    let mut current_roi: Roi = full_frame_roi(&info);
    // Consecutive 1 s frame timeouts, so a stalled stream is reported rather
    // than spun on in silence.
    let mut stalled: u32 = 0;

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
                FocusCmd::Dispersion(h) => disp_h = h,
                FocusCmd::LiveRoi {
                    capture_height,
                    anchor_y,
                    enable,
                } => {
                    crate::applog!(
                        "live-roi: request enable={enable} height={capture_height} anchor={anchor_y:.1} recording={}",
                        active_ser.is_some() || pending_ser.is_some()
                    );
                    // Refuse to touch the sensor mid-recording: the SER's
                    // frame geometry is fixed at the first frame.
                    if active_ser.is_some() || pending_ser.is_some() {
                        let _ = tx.send(FocusMsg::Error(
                            "the sensor ROI cannot change while recording".into(),
                        ));
                    } else if enable {
                        let (y0, height) =
                            vertical_crop_bounds(info.max_height, capture_height, anchor_y);
                        let band = Roi {
                            x: 0,
                            y: y0 & !1,
                            w: info.max_width,
                            h: (height & !1).max(2),
                        };
                        match ensure_roi(&mut *cam, &info, &mut current_roi, band) {
                            Ok(()) => {
                                live_roi = Some(band);
                                let _ = tx.send(FocusMsg::LiveRoi {
                                    active: true,
                                    y0: band.y,
                                    h: band.h,
                                });
                            }
                            Err(roi_error) => {
                                // A refused ROI must not leave the camera
                                // stopped — apply_roi stops before it sets.
                                if let Err(error) = apply_roi(&mut *cam, full_frame_roi(&info)) {
                                    let _ = tx.send(FocusMsg::Error(format!(
                                        "hardware ROI failed ({roi_error}) and the camera did not recover: {error}"
                                    )));
                                    return;
                                }
                                current_roi = full_frame_roi(&info);
                                live_roi = None;
                                let _ = tx.send(FocusMsg::LiveRoi {
                                    active: false,
                                    y0: 0,
                                    h: info.max_height,
                                });
                                let _ = tx.send(FocusMsg::Error(format!(
                                    "the camera refused the hardware ROI: {roi_error}"
                                )));
                            }
                        }
                    } else {
                        let full = full_frame_roi(&info);
                        if let Err(error) = ensure_roi(&mut *cam, &info, &mut current_roi, full) {
                            let _ = tx.send(FocusMsg::Error(format!(
                                "the full sensor could not be restored: {error}"
                            )));
                            return;
                        }
                        live_roi = None;
                        let _ = tx.send(FocusMsg::LiveRoi {
                            active: false,
                            y0: 0,
                            h: info.max_height,
                        });
                    }
                }
                FocusCmd::StartSer {
                    path,
                    capture_height,
                    anchor_y,
                    hw_roi,
                } => {
                    if let Some(active) = active_ser.take() {
                        let had_hw = active.hw_roi_y0.is_some();
                        finish_ser(active, &tx);
                        if had_hw {
                            let rest = resting_roi(&info, live_roi);
                            if let Err(error) = ensure_roi(&mut *cam, &info, &mut current_roi, rest)
                            {
                                let _ = tx.send(FocusMsg::Error(format!(
                                    "the sensor could not be restored: {error}"
                                )));
                                return;
                            }
                        }
                    }
                    // Hardware ROI: have the sensor read only the capture band,
                    // so frame rate is exposure-limited rather than full-frame
                    // readout-limited (measured on a G3M678M: 23 → 176 fps at
                    // 256 rows). Frame rate is scan-axis sampling density, so
                    // this is a direct resolution win. Scoped strictly to the
                    // recording; a refusal falls back to the software crop --
                    // a cropped recording beats no recording.
                    let mut hw_roi_y0 = None;
                    if hw_roi {
                        let (y0, height) =
                            vertical_crop_bounds(info.max_height, capture_height, anchor_y);
                        let band = Roi {
                            x: 0,
                            y: y0 & !1,
                            w: info.max_width,
                            h: (height & !1).max(2),
                        };
                        // Usually a no-op: if the user already applied this
                        // band live, the sensor is reading it and the stream
                        // must NOT be cycled just to say so again.
                        match ensure_roi(&mut *cam, &info, &mut current_roi, band) {
                            Ok(()) => hw_roi_y0 = Some(band.y),
                            Err(roi_error) => {
                                let rest = resting_roi(&info, live_roi);
                                if let Err(error) =
                                    ensure_roi(&mut *cam, &info, &mut current_roi, rest)
                                {
                                    let _ = tx.send(FocusMsg::Error(format!(
                                        "hardware ROI failed ({roi_error}) and the camera did not recover: {error}"
                                    )));
                                    return;
                                }
                            }
                        }
                    }
                    pending_ser = Some(SerRequest {
                        path,
                        capture_height,
                        anchor_y,
                        hw_roi_y0,
                    });
                }
                FocusCmd::StopSer => {
                    let pending = pending_ser.take();
                    if let Some(active) = active_ser.take() {
                        let had_hw = active.hw_roi_y0.is_some();
                        let finished = finish_ser_message(active);
                        // Unwind BEFORE resuming preview. The recording's own
                        // ROI is scoped to the recording; where that leaves
                        // the sensor is the user's live band if they set one.
                        if had_hw {
                            let rest = resting_roi(&info, live_roi);
                            if let Err(error) = ensure_roi(&mut *cam, &info, &mut current_roi, rest)
                            {
                                let _ = tx.send(finished);
                                let _ = tx.send(FocusMsg::Error(format!(
                                    "SER saved, but the sensor could not be restored: {error}"
                                )));
                                return;
                            }
                        }
                        let preview = cam.resume_preview();
                        let _ = tx.send(finished);
                        if let Err(error) = preview {
                            let _ = tx.send(FocusMsg::Error(format!(
                                "SER saved, but live preview could not resume: {error}"
                            )));
                            return;
                        }
                    } else if let Some(request) = pending {
                        // No frame ever arrived. If the ROI was already
                        // applied it must still be unwound.
                        if request.hw_roi_y0.is_some() {
                            let rest = resting_roi(&info, live_roi);
                            if let Err(error) = ensure_roi(&mut *cam, &info, &mut current_roi, rest)
                            {
                                let _ = tx.send(FocusMsg::Error(format!(
                                    "the sensor could not be restored: {error}"
                                )));
                                return;
                            }
                        }
                        let _ = tx.send(FocusMsg::RecordingStopped {
                            path: request.path,
                            frames: 0,
                            fps: 0.0,
                            hw_roi: false,
                            device_clock: false,
                        });
                    }
                }
            }
        }
        // Preview is latency-first: vendor SDKs may buffer several completed
        // frames while focus analysis is busy, so ask each backend for the
        // newest available image. During SER recording we use the sequential
        // path and intentionally write every frame delivered to the app.
        let next = if active_ser.is_some() {
            cam.next_frame(1000)
        } else {
            cam.next_preview_frame(1000)
        };
        match next {
            Ok(frame) => {
                if stalled > 0 {
                    crate::applog!("camera: frames resumed after {stalled}s");
                    stalled = 0;
                }
                if let Some(request) = pending_ser.take() {
                    let (y0, height) = match request.hw_roi_y0 {
                        // The camera already delivers only the band.
                        Some(_) => (0, frame.height),
                        None => vertical_crop_bounds(
                            frame.height,
                            request.capture_height,
                            request.anchor_y,
                        ),
                    };
                    match SerRecorder::create(
                        &request.path,
                        frame.width,
                        height,
                        &info.name,
                        "Spectroheliograph",
                    ) {
                        Ok(recorder) => {
                            let _ = tx.send(FocusMsg::RecordingStarted {
                                path: request.path.clone(),
                                width: frame.width,
                                height,
                                hw_roi: request.hw_roi_y0.is_some(),
                            });
                            active_ser = Some(ActiveSer {
                                path: request.path,
                                y0,
                                height,
                                recorder,
                                hw_roi_y0: request.hw_roi_y0,
                                started: Instant::now(),
                            });
                        }
                        Err(error) => {
                            let _ = tx.send(FocusMsg::RecordingError(error.to_string()));
                        }
                    }
                }

                let mut recording_failed = None;
                if let Some(active) = active_ser.as_mut() {
                    if frame.width == 0
                        || active.y0.saturating_add(active.height) > frame.height
                    {
                        recording_failed =
                            Some("camera dimensions changed during recording".to_owned());
                    } else {
                        let start = active.y0 * frame.width;
                        let end = start + active.height * frame.width;
                        let ftime = FrameTime {
                            host: frame.host_time,
                            device_us: frame.device_time_us,
                        };
                        match active.recorder.write_frame(&frame.data[start..end], ftime) {
                            Ok(()) => {
                                let frames = active.recorder.frame_count();
                                if frames == 1 || frames % 10 == 0 {
                                    let _ = tx.send(FocusMsg::RecordingProgress(frames));
                                }
                            }
                            Err(error) => recording_failed = Some(error.to_string()),
                        }
                    }
                }
                if let Some(error) = recording_failed {
                    let had_hw = active_ser
                        .take()
                        .map(|active| active.hw_roi_y0.is_some())
                        .unwrap_or(false);
                    let _ = tx.send(FocusMsg::RecordingError(error));
                    if had_hw {
                        if let Err(error) = apply_roi(&mut *cam, full_frame_roi(&info)) {
                            let _ = tx.send(FocusMsg::Error(format!(
                                "the full sensor could not be restored: {error}"
                            )));
                            return;
                        }
                    }
                }

                // Always acquire (and therefore drain the camera/SDK), but do
                // not spend time processing or queueing another preview while
                // egui still has one waiting. This bounds preview latency.
                if frame_pending.load(Ordering::Acquire) {
                    continue;
                }
                // Both axes, every frame: the two line families separate cleanly
                // because averaging one axis cancels lines parallel to it.
                let prof_x = frame.mean_profile(true); // dips = vertical lines
                let prof_y = frame.mean_profile(false); // dips = horizontal lines
                let lines_x: Vec<Fit> = fit_lines_1d(&prof_x, 0.02)
                    .into_iter()
                    .map(Fit::from)
                    .collect();
                let lines_y: Vec<Fit> = fit_lines_1d(&prof_y, 0.02)
                    .into_iter()
                    .map(Fit::from)
                    .collect();
                let mean = if prof_x.is_empty() {
                    0.0
                } else {
                    (prof_x.iter().map(|&v| v as f64).sum::<f64>() / prof_x.len() as f64) as f32
                };
                let peak = robust_peak_signal(&frame.data);
                // Stage B: cut the frame along the slit using continuum
                // dispersion positions only. The line core is low-contrast
                // chromosphere and its focus curve is flattened by scattered
                // light; the continuum carries granulation and a hard limb.
                let spec_prof = if disp_h { &prof_x } else { &prof_y };
                let mask = focusmetrics::continuum_mask(spec_prof);
                let n_continuum = mask.iter().filter(|&&m| m).count();
                let slit_cut = focusmetrics::slit_profile_continuum(
                    &frame.data,
                    frame.width,
                    frame.height,
                    disp_h,
                    &mask,
                );
                let limb_width = focusmetrics::limb_edge_width(&slit_cut);
                let structure = focusmetrics::structure_split(&slit_cut);

                let (strip, sw, sh) = make_strip(&frame);
                let cur_exposure = cam.current_exposure_us();
                let cur_gain = cam.current_gain();
                let update = Box::new(FocusUpdate {
                    hw_roi_active: live_roi.is_some()
                        || active_ser
                            .as_ref()
                            .map(|active| active.hw_roi_y0.is_some())
                            .unwrap_or(false),
                    slit_cut: slit_cut.iter().map(|&v| v as f32).collect(),
                    n_continuum,
                    limb_width,
                    structure,
                    strip,
                    strip_w: sw,
                    strip_h: sh,
                    prof_x: prof_x.iter().map(|&v| v as f32).collect(),
                    prof_y: prof_y.iter().map(|&v| v as f32).collect(),
                    lines_x,
                    lines_y,
                    mean,
                    peak,
                    full_w: frame.width,
                    full_h: frame.height,
                    cur_exposure,
                    cur_gain,
                });
                if frame_pending
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    if tx.send(FocusMsg::Frame(update)).is_err() {
                        frame_pending.store(false, Ordering::Release);
                        break;
                    }
                    ctx.request_repaint();
                }
            }
            Err(ghostsun_camera::CameraError::Timeout) => {
                // Swallowing these silently is what made a stalled stream look
                // like a frozen UI with an empty log. The stream legitimately
                // pauses for a beat around a geometry change, so say nothing
                // for the first one and escalate from there.
                stalled += 1;
                if stalled == 2 || stalled == 5 || stalled % 15 == 0 {
                    crate::applog!(
                        "camera: no frame for {stalled}s (roi {}x{}+{}+{}, exposure {:?} us)",
                        current_roi.w,
                        current_roi.h,
                        current_roi.x,
                        current_roi.y,
                        cam.current_exposure_us()
                    );
                    let _ = tx.send(FocusMsg::Note(format!(
                        "no frames for {stalled}s — the camera stream has stalled"
                    )));
                    ctx.request_repaint();
                }
                continue;
            }
            Err(e) => {
                crate::applog!("camera: fatal stream error: {e}");
                if let Some(active) = active_ser.take() {
                    finish_ser(active, &tx);
                }
                let _ = tx.send(FocusMsg::Error(e.to_string()));
                break;
            }
        }
    }
    if let Some(active) = active_ser.take() {
        finish_ser(active, &tx);
    }
    crate::applog!("camera: worker exiting, closing device");
    cam.stop();
}

fn finish_ser(active: ActiveSer, tx: &Sender<FocusMsg>) {
    let _ = tx.send(finish_ser_message(active));
}

fn finish_ser_message(active: ActiveSer) -> FocusMsg {
    let path = active.path;
    let elapsed = active.started.elapsed().as_secs_f64();
    let hw_roi = active.hw_roi_y0.is_some();
    match active.recorder.finish() {
        Ok(summary) => {
            let frames = summary.frames;
            // Prefer the timestamp trailer: it spans first-to-last frame on
            // the camera clock, where `started` is a capture-thread Instant
            // taken at recording start.
            let fps = summary.fps().unwrap_or({
                // (frames - 1) intervals, since `started` is stamped at frame 1.
                if frames > 1 && elapsed > 0.0 {
                    (frames as f64 - 1.0) / elapsed
                } else {
                    0.0
                }
            });
            FocusMsg::RecordingStopped {
                path,
                frames,
                fps,
                hw_roi,
                device_clock: summary.device_clock,
            }
        }
        Err(error) => FocusMsg::RecordingError(error.to_string()),
    }
}

fn full_frame_roi(info: &CameraInfo) -> Roi {
    Roi {
        x: 0,
        y: 0,
        w: info.max_width,
        h: info.max_height,
    }
}

/// What the sensor should read when nothing is recording: the live band if
/// the user applied one, otherwise the whole sensor. A recording must unwind
/// to this, not unconditionally to full frame, or every scan would silently
/// cancel a live ROI the user had set up.
fn resting_roi(info: &CameraInfo, live: Option<Roi>) -> Roi {
    live.unwrap_or_else(|| full_frame_roi(info))
}

/// Stop → apply ROI → restart. Every backend applies ROI at stream start, so
/// the cycle is mandatory. A failure leaves the camera stopped; callers must
/// treat it as fatal for the session unless a full-frame restore succeeds.
fn apply_roi(cam: &mut dyn ghostsun_camera::Camera, roi: Roi) -> Result<(), String> {
    let t0 = Instant::now();
    crate::applog!(
        "roi: stop -> set {}x{}+{}+{} -> start",
        roi.w,
        roi.h,
        roi.x,
        roi.y
    );
    cam.stop();
    if let Err(error) = cam.set_roi(roi) {
        crate::applog!("roi: set_roi FAILED: {error}");
        return Err(error.to_string());
    }
    match cam.start() {
        Ok(()) => {
            crate::applog!("roi: started in {} ms", t0.elapsed().as_millis());
            Ok(())
        }
        Err(error) => {
            crate::applog!("roi: start FAILED after {} ms: {error}", t0.elapsed().as_millis());
            Err(error.to_string())
        }
    }
}

fn roi_eq(a: Roi, b: Roi) -> bool {
    a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h
}

/// Move the sensor to `target`, tracking what is currently applied.
///
/// Two rules the raw `apply_roi` cannot enforce on its own, both learned from
/// a camera that hung when a scan started on an already-cropped sensor:
///
/// 1. A redundant stop/set/start is not free — it is a real risk of a stalled
///    stream for no gain. If the sensor already reads `target`, do nothing.
/// 2. ROI offsets are full-sensor coordinates, but a band set while another
///    band is applied is not reliably interpreted that way. Normalise through
///    full frame before setting a different band.
fn ensure_roi(
    cam: &mut dyn ghostsun_camera::Camera,
    info: &CameraInfo,
    current: &mut Roi,
    target: Roi,
) -> Result<(), String> {
    let steps = roi_transition(*current, target, full_frame_roi(info));
    crate::applog!(
        "roi: {}x{}+{}+{} -> {}x{}+{}+{} ({} step(s))",
        current.w,
        current.h,
        current.x,
        current.y,
        target.w,
        target.h,
        target.x,
        target.y,
        steps.len()
    );
    for step in steps {
        apply_roi(cam, step)?;
        *current = step;
    }
    Ok(())
}

/// The ROIs to apply, in order, to get from `current` to `target`.
///
/// Empty when the sensor already reads the target — the case that matters,
/// since a scan starting on a band the user already applied live would
/// otherwise cycle the stream for nothing and could stall it.
fn roi_transition(current: Roi, target: Roi, full: Roi) -> Vec<Roi> {
    if roi_eq(current, target) {
        return Vec::new();
    }
    if roi_eq(current, full) {
        return vec![target];
    }
    if roi_eq(target, full) {
        return vec![full];
    }
    // Band to a DIFFERENT band: normalise through full frame, because ROI
    // offsets are full-sensor coordinates and are not reliably interpreted
    // as such while another band is applied.
    vec![full, target]
}

fn vertical_crop_bounds(
    frame_height: usize,
    requested_height: usize,
    anchor_y: f64,
) -> (usize, usize) {
    let height = requested_height.clamp(1, frame_height.max(1));
    let anchor = if anchor_y.is_finite() {
        anchor_y.round().clamp(0.0, frame_height.saturating_sub(1) as f64) as usize
    } else {
        frame_height / 2
    };
    let y0 = anchor
        .saturating_sub(height / 2)
        .min(frame_height.saturating_sub(height));
    (y0, height)
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

/// Mean of the brightest one percent of a 16-bit frame.
///
/// The 4096-bin histogram avoids sorting or cloning a multi-megapixel frame.
/// The partially used boundary bin is represented by that bin's actual mean.
fn robust_peak_signal(data: &[u16]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    const BINS: usize = 4096;
    let mut counts = [0u32; BINS];
    let mut sums = [0u64; BINS];
    for &value in data {
        let bin = (usize::from(value) * BINS) >> 16;
        counts[bin] += 1;
        sums[bin] += u64::from(value);
    }

    let target = data.len().div_ceil(SUN_PEAK_FRACTION).max(1) as u64;
    let mut remaining = target;
    let mut selected_sum = 0.0f64;
    for bin in (0..BINS).rev() {
        let count = u64::from(counts[bin]);
        if count == 0 {
            continue;
        }
        let take = remaining.min(count);
        selected_sum += (sums[bin] as f64 / count as f64) * take as f64;
        remaining -= take;
        if remaining == 0 {
            break;
        }
    }
    (selected_sum / target as f64) as f32
}

impl Drop for FocusState {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(dead_code)]
pub fn full_roi(info: &CameraInfo) -> Roi {
    Roi {
        x: 0,
        y: 0,
        w: info.max_width,
        h: info.max_height,
    }
}

#[cfg(test)]
mod roi_tests {
    use super::*;

    fn full() -> Roi {
        Roi { x: 0, y: 0, w: 3840, h: 2160 }
    }
    fn band(y: usize, h: usize) -> Roi {
        Roi { x: 0, y, w: 3840, h }
    }
    fn same(got: &[Roi], want: &[Roi]) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(a, b)| roi_eq(*a, *b))
    }

    /// The regression: with a band already applied live, starting a scan asks
    /// for the SAME band. Re-applying it cycles stop/set/start on a healthy
    /// stream, which hung the camera mid-acquisition ("SER recorder did not
    /// start"). An unchanged ROI must produce no camera calls at all.
    #[test]
    fn re_applying_the_same_band_touches_the_camera_not_at_all() {
        assert!(roi_transition(band(1026, 120), band(1026, 120), full()).is_empty());
        assert!(roi_transition(full(), full(), full()).is_empty());
    }

    #[test]
    fn band_to_band_normalises_through_full_frame() {
        // ROI offsets are full-sensor coordinates; setting a band while
        // another is applied cannot be trusted to mean that.
        assert!(same(
            &roi_transition(band(1026, 120), band(400, 256), full()),
            &[full(), band(400, 256)]
        ));
    }

    #[test]
    fn single_step_from_or_to_full_frame() {
        assert!(same(
            &roi_transition(full(), band(400, 256), full()),
            &[band(400, 256)]
        ));
        assert!(same(&roi_transition(band(400, 256), full(), full()), &[full()]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_peak_averages_the_brightest_one_percent() {
        let mut data = vec![100u16; 990];
        data.extend(std::iter::repeat_n(1_000u16, 10));
        assert!((robust_peak_signal(&data) - 1_000.0).abs() < 0.1);
    }

    #[test]
    fn robust_peak_is_not_owned_by_one_hot_pixel() {
        let mut data = vec![100u16; 9_900];
        data.extend(std::iter::repeat_n(1_000u16, 99));
        data.push(u16::MAX);
        let peak = robust_peak_signal(&data);
        assert!(peak > 1_000.0);
        assert!(peak < 2_000.0, "single hot pixel produced {peak}");
    }

    #[test]
    fn vertical_recording_crop_stays_in_sensor_bounds() {
        assert_eq!(vertical_crop_bounds(1_000, 200, 500.0), (400, 200));
        assert_eq!(vertical_crop_bounds(1_000, 200, 20.0), (0, 200));
        assert_eq!(vertical_crop_bounds(1_000, 200, 990.0), (800, 200));
        assert_eq!(vertical_crop_bounds(1_000, 2_000, 500.0), (0, 1_000));
    }

    fn fit_with(fwhm: f64) -> Fit {
        Fit {
            fwhm,
            depth: 0.5, // comfortably above DEPTH_GATE
            center: 100.0,
            sigma: fwhm / 2.3548,
            continuum: 1000.0,
        }
    }

    fn blank_update() -> FocusUpdate {
        FocusUpdate {
            strip: Vec::new(),
            strip_w: 0,
            strip_h: 0,
            prof_x: Vec::new(),
            prof_y: Vec::new(),
            lines_x: Vec::new(),
            lines_y: Vec::new(),
            mean: 0.0,
            peak: 0.0,
            full_w: 0,
            full_h: 0,
            cur_exposure: None,
            cur_gain: None,
            slit_cut: Vec::new(),
            n_continuum: 0,
            limb_width: None,
            structure: StructureSplit::default(),
            hw_roi_active: false,
        }
    }

    /// Drive one full capture burst at `pos` with the given per-frame values.
    fn capture_at(st: &mut FocusState, pos: f64, spec: Option<f64>, slit: Option<f64>) {
        st.camera_pos_text = pos.to_string();
        st.begin_capture();
        let u = blank_update();
        for _ in 0..st.capture_frames {
            st.accumulate_capture(&spec.map(fit_with), &slit.map(fit_with), &u);
        }
    }

    fn stage_a_state() -> FocusState {
        let mut st = FocusState::default();
        st.stage = Stage::Spectrograph;
        st.streaming = true; // begin_capture refuses to run without a camera
        st.capture_frames = 8;
        st
    }

    fn fit_deep(fwhm: f64, depth: f64) -> Fit {
        Fit {
            fwhm,
            depth,
            center: 100.0,
            sigma: fwhm / 2.3548,
            continuum: 1000.0,
        }
    }

    #[test]
    fn narrowest_ignores_sub_threshold_noise_spikes() {
        // The real failure seen on sky: a 42%-deep line alongside noise the
        // detector admitted at just over 2%. Noise is always narrower, so an
        // unfiltered "narrowest" picks it and the burst is then rejected by the
        // 3% capture gate.
        let lines = vec![
            fit_deep(1.8, 0.021), // noise spike, narrower than anything real
            fit_deep(2.2, 0.025), // ditto
            fit_deep(5.6, 0.42),  // the actual line
            fit_deep(7.9, 0.11),
        ];
        let pick = choose(&lines, LineMode::Narrowest, None).expect("a usable line exists");
        assert!(pick.depth > DEPTH_GATE, "picked depth {}", pick.depth);
        // Narrowest among the *usable* lines, not narrowest overall.
        assert!((pick.fwhm - 5.6).abs() < 1e-9, "picked fwhm {}", pick.fwhm);
    }

    #[test]
    fn choose_returns_nothing_when_every_candidate_is_too_shallow() {
        let lines = vec![fit_deep(1.8, 0.021), fit_deep(2.4, 0.029)];
        assert!(choose(&lines, LineMode::Narrowest, None).is_none());
        assert!(choose(&lines, LineMode::Deepest, None).is_none());
        assert!(choose(&lines, LineMode::Manual, Some(100.0)).is_none());
    }

    #[test]
    fn both_families_select_independently() {
        // The slit family used to be hardwired to Narrowest with no pick. Stage
        // A only works if the same feature is tracked at every camera position,
        // so each family needs its own lock.
        let mut st = stage_a_state();
        assert_eq!(st.slit_line_mode, LineMode::Narrowest);
        assert!(st.slit_picked_center.is_none());

        st.slit_line_mode = LineMode::Manual;
        st.slit_picked_center = Some(155.0);
        // Choosing for one family must not disturb the other.
        assert_eq!(st.line_mode, LineMode::Narrowest);
        assert!(st.picked_center.is_none());

        let mut near = fit_deep(3.0, 0.14);
        near.center = 154.0;
        let mut far = fit_deep(2.0, 0.07);
        far.center = 300.0;
        let lines = vec![near, far];
        let pick = choose(&lines, st.slit_line_mode, st.slit_picked_center).unwrap();
        assert!((pick.center - 154.0).abs() < 1e-9, "picked {}", pick.center);
        // Narrowest would have taken the other one; the lock overrides that.
        assert!(pick.fwhm > far.fwhm);
    }

    #[test]
    fn manual_pick_snaps_to_the_nearest_usable_line_not_the_nearest_noise() {
        let mut noise = fit_deep(1.5, 0.022);
        noise.center = 240.0; // right where the user clicked
        let mut real = fit_deep(6.0, 0.40);
        real.center = 244.0; // the line they meant
        let lines = vec![noise, real];
        let pick = choose(&lines, LineMode::Manual, Some(240.0)).unwrap();
        assert!((pick.center - 244.0).abs() < 1e-9, "picked {}", pick.center);
    }

    #[test]
    fn crop_bounds_centre_clamp_and_oversize() {
        // Centred anchor: band centred on the line.
        assert_eq!(vertical_crop_bounds(2160, 256, 1100.0), (972, 256));
        // Anchor near the top: clamped to the sensor edge, full height kept.
        assert_eq!(vertical_crop_bounds(2160, 256, 10.0), (0, 256));
        // Anchor near the bottom: clamped so the band stays on-sensor.
        assert_eq!(vertical_crop_bounds(2160, 256, 2155.0), (1904, 256));
        // Oversize request: the whole frame.
        assert_eq!(vertical_crop_bounds(2160, 9999, 1000.0), (0, 2160));
        // These bounds feed the hardware ROI directly, so clamping is what
        // keeps a near-edge line from producing an off-sensor ROI request.
    }

    #[test]
    fn stage_a_capture_builds_both_curves_and_signs_delta_spectral_minus_slit() {
        let mut st = stage_a_state();
        // Spectral minimum at 0.34, slit minimum at 0.28 ⇒ Δ = +0.06.
        for pos in [0.10, 0.20, 0.30, 0.40, 0.50] {
            let spec = 4.0 * (pos - 0.34) * (pos - 0.34) + 2.0;
            let slit = 3.0 * (pos - 0.28) * (pos - 0.28) + 3.0;
            capture_at(&mut st, pos, Some(spec), Some(slit));
        }
        assert_eq!(st.curve_spec.samples.len(), 5);
        assert_eq!(st.curve_slit.samples.len(), 5);

        let sp = vcurve::split(&st.curve_spec, &st.curve_slit).expect("Δ should solve");
        assert!((sp.spectral.vertex - 0.34).abs() < 1e-6, "{}", sp.spectral.vertex);
        assert!((sp.spatial.vertex - 0.28).abs() < 1e-6, "{}", sp.spatial.vertex);
        // The sign convention the whole procedure hangs on.
        assert!((sp.delta - 0.06).abs() < 1e-6, "delta {}", sp.delta);
        assert!(sp.spectral.trustworthy() && sp.spatial.trustworthy());
    }

    #[test]
    fn a_burst_with_no_usable_line_is_dropped_rather_than_recorded() {
        let mut st = stage_a_state();
        capture_at(&mut st, 0.20, None, None);
        assert!(st.curve_spec.is_empty(), "a dropout must not become a sample");
        assert!(st.curve_slit.is_empty());
        assert!(st.capture.is_none(), "capture should have completed, not stalled");
    }

    #[test]
    fn shallow_lines_below_the_depth_gate_are_not_averaged_in() {
        let mut st = stage_a_state();
        st.camera_pos_text = "0.2".into();
        st.begin_capture();
        let u = blank_update();
        let mut shallow = fit_with(3.0);
        shallow.depth = DEPTH_GATE * 0.5;
        for _ in 0..st.capture_frames {
            st.accumulate_capture(&Some(shallow), &Some(fit_with(4.0)), &u);
        }
        // The slit family was fine, the spectral one was not, so neither curve
        // gains a point — a half-measured position is not a sample.
        assert!(st.curve_spec.is_empty());
        assert!(st.curve_slit.is_empty());
    }

    #[test]
    fn capture_requires_a_parsed_position() {
        let mut st = stage_a_state();
        st.camera_pos_text = "about 12".into();
        st.begin_capture();
        assert!(st.capture.is_none());
        assert!(st.stage_status.contains("micrometer"));
    }

    #[test]
    fn undo_removes_the_last_sample_from_both_stage_a_curves() {
        let mut st = stage_a_state();
        for pos in [0.1, 0.2, 0.3] {
            capture_at(&mut st, pos, Some(2.0), Some(3.0));
        }
        st.curve_spec.undo();
        st.curve_slit.undo();
        assert_eq!(st.curve_spec.samples.len(), 2);
        assert_eq!(st.curve_slit.samples.len(), 2);
    }

    #[test]
    fn banking_a_null_point_clears_the_sweep_and_keeps_the_delta() {
        let mut st = stage_a_state();
        for pos in [0.10, 0.20, 0.30, 0.40, 0.50] {
            let spec = 4.0 * (pos - 0.34) * (pos - 0.34) + 2.0;
            let slit = 3.0 * (pos - 0.28) * (pos - 0.28) + 3.0;
            capture_at(&mut st, pos, Some(spec), Some(slit));
        }
        st.collimator_pos_text = "12.0".into();
        st.record_null_point();
        assert_eq!(st.null_points.len(), 1);
        assert!((st.null_points[0].delta - 0.06).abs() < 1e-6);
        assert!(st.curve_spec.is_empty(), "sweep should reset for the next setting");
    }

    #[test]
    fn banking_without_a_collimator_reading_is_refused() {
        let mut st = stage_a_state();
        for pos in [0.10, 0.20, 0.30, 0.40, 0.50] {
            capture_at(&mut st, pos, Some((pos - 0.3) * (pos - 0.3)), Some((pos - 0.2) * (pos - 0.2)));
        }
        st.collimator_pos_text.clear();
        st.record_null_point();
        assert!(st.null_points.is_empty());
        assert!(!st.curve_spec.is_empty(), "a refused bank must not discard the sweep");
    }

    #[test]
    fn stage_b_capture_uses_the_selected_metric_and_its_polarity() {
        let mut st = FocusState::default();
        st.stage = Stage::Telescope;
        st.streaming = true;
        st.capture_frames = 8;
        st.tele_metric = TeleMetric::LimbEdge;
        st.retarget_tele_curves();

        for pos in [10.0, 10.5, 11.0, 11.5, 12.0] {
            st.focuser_pos_text = pos.to_string();
            st.begin_capture();
            let mut u = blank_update();
            u.limb_width = Some(2.0 * (pos - 11.2) * (pos - 11.2) + 1.5);
            for _ in 0..st.capture_frames {
                st.accumulate_capture(&None, &None, &u);
            }
        }
        let f = st.curve_tele.fit().expect("Stage B curve should solve");
        assert!((f.vertex - 11.2).abs() < 1e-6, "vertex {}", f.vertex);
        assert!(f.curvature > 0.0, "limb width is minimised at focus");
        assert!(f.trustworthy());
    }

    #[test]
    fn switching_stage_b_metric_discards_incomparable_samples() {
        let mut st = FocusState::default();
        st.stage = Stage::Telescope;
        st.streaming = true;
        st.capture_frames = 4;
        st.tele_metric = TeleMetric::LimbEdge;

        st.focuser_pos_text = "11.0".into();
        st.begin_capture();
        let mut u = blank_update();
        u.limb_width = Some(3.0);
        for _ in 0..st.capture_frames {
            st.accumulate_capture(&None, &None, &u);
        }
        assert_eq!(st.curve_tele.samples.len(), 1);

        st.tele_metric = TeleMetric::Structure;
        st.retarget_tele_curves();
        assert!(st.curve_tele.is_empty(), "px and contrast are not the same axis");
        assert!(!st.curve_tele.want_min, "contrast is maximised, not minimised");
    }
}
