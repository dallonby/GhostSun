//! Bounded telescope autofocus. The worker owns USB; the UI supplies fresh,
//! timestamped limb fits. Movement never originates from discovery or connection.
use crate::vcurve::{fit_parabola, VSample};
use ghostsun_camera::focuser::{self, Focuser, Info, State};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct FrameSample {
    pub seq: u64,
    /// Earliest estimated exposure time; old/in-flight frames cannot enter a burst.
    pub captured: Instant,
    pub widths: [Option<f64>; 2],
    pub geometry: [usize; 2],
    pub exposure: u32,
    pub gain: u16,
    pub clipped: bool,
}
impl FrameSample {
    fn mask(self) -> u8 {
        self.widths.iter().enumerate().fold(0, |m, (i, v)| {
            m | if v.is_some_and(|w| w.is_finite() && w > 0.0) {
                1 << i
            } else {
                0
            }
        })
    }
    fn score(self, mask: u8) -> Option<f64> {
        if self.clipped || mask == 0 || self.mask() & mask != mask {
            return None;
        }
        let mut sum = 0.0;
        let mut n = 0;
        for i in 0..2 {
            if mask & (1 << i) != 0 {
                sum += self.widths[i]?;
                n += 1;
            }
        }
        Some(sum / n as f64)
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub minimum: i32,
    pub maximum: i32,
    pub step: i32,
    pub points: usize,
    /// Every measured/final position is approached in the increasing direction.
    pub approach: i32,
    pub frames: usize,
    pub settle_ms: u64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            minimum: 0,
            maximum: 0,
            step: 50,
            points: 7,
            approach: 0,
            frames: 30,
            settle_ms: 1000,
        }
    }
}
impl Settings {
    fn positions(&self, state: State) -> Result<Vec<i32>, String> {
        if state.moving {
            return Err("Wait for the focuser to stop".into());
        }
        if self.minimum < 0
            || self.maximum > state.max_step
            || self.minimum >= self.maximum
            || !(self.minimum..=self.maximum).contains(&state.position)
        {
            return Err(
                "Set safe travel limits containing the current position, within the device limits"
                    .into(),
            );
        }
        if self.step < 1
            || !(5..=11).contains(&self.points)
            || self.points % 2 == 0
            || self.approach < 0
            || !(20..=200).contains(&self.frames)
            || !(250..=10000).contains(&self.settle_ms)
        {
            return Err(
                "Use 5–11 odd points, a positive step, 20–200 frames and at least 250 ms settling"
                    .into(),
            );
        }
        let half = self.step as i64 * (self.points / 2) as i64;
        let start = state.position as i64 - half;
        let end = state.position as i64 + half;
        if start - (self.approach as i64) < self.minimum as i64 || end > self.maximum as i64 {
            return Err(
                "The sweep and approach move must fit inside your safe travel limits".into(),
            );
        }
        Ok((0..self.points)
            .map(|i| (start + i as i64 * self.step as i64) as i32)
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub position: i32,
    pub width: f64,
}
#[derive(Clone, Debug)]
enum Phase {
    Moving {
        target: i32,
        next: Option<i32>,
        since: Instant,
        verify: bool,
    },
    Settling {
        until: Instant,
        verify: bool,
    },
    Collecting {
        since: Instant,
        last_seq: u64,
        values: Vec<f64>,
        verify: bool,
    },
}
struct Run {
    settings: Settings,
    positions: Vec<i32>,
    index: usize,
    reference: FrameSample,
    mask: u8,
    phase: Phase,
    samples: Vec<VSample>,
    target: Option<i32>,
    outcome: Option<Outcome>,
}
fn fresh(frame: FrameSample, now: Instant) -> bool {
    now.checked_duration_since(frame.captured)
        .is_some_and(|d| d < Duration::from_secs(3))
}
fn burst_value(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    // Average the best quartile: several good-seeing frames, never one minimum.
    let n = sorted.len().div_ceil(4).max(1);
    sorted[..n].iter().sum::<f64>() / n as f64
}
fn fitted_target(samples: &[VSample], step: i32) -> Result<i32, String> {
    let base = samples[0].pos;
    let centred: Vec<_> = samples
        .iter()
        .map(|s| VSample {
            pos: s.pos - base,
            ..*s
        })
        .collect();
    let fit = fit_parabola(&centred, true).ok_or("Focus curve could not be fitted")?;
    let best = samples
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.value.total_cmp(&b.value))
        .unwrap()
        .0;
    let depth = samples[0].value.min(samples.last().unwrap().value) - samples[best].value;
    if !fit.trustworthy()
        || !fit.vertex_sigma.is_finite()
        || fit.vertex_sigma > step as f64
        || fit.extremum <= 0.0
        || best == 0
        || best + 1 == samples.len()
        || depth <= (3.0 * fit.rms).max(samples[best].value * 0.02)
    {
        return Err("No convincing bracketed minimum; adjust the sweep range and try again".into());
    }
    let vertex = base + fit.vertex;
    if vertex < samples[0].pos + step as f64 * 0.5
        || vertex > samples.last().unwrap().pos - step as f64 * 0.5
    {
        return Err("Best focus is too near the sweep boundary; recenter the sweep".into());
    }
    Ok(vertex.round() as i32)
}
impl Run {
    fn start(
        settings: Settings,
        state: State,
        frame: FrameSample,
        now: Instant,
    ) -> Result<(Self, i32), String> {
        let positions = settings.positions(state)?;
        let mask = frame.mask();
        if !fresh(frame, now) || frame.score(mask).is_none() {
            return Err("A fresh, unclipped solar limb measurement is required".into());
        }
        let target = positions[0] - settings.approach;
        let next = (settings.approach > 0).then_some(positions[0]);
        Ok((
            Self {
                settings,
                positions,
                index: 0,
                reference: frame,
                mask,
                phase: Phase::Moving {
                    target,
                    next,
                    since: now,
                    verify: false,
                },
                samples: Vec::new(),
                target: None,
                outcome: None,
            },
            target,
        ))
    }
    fn progress(&self) -> String {
        match &self.phase {
            Phase::Moving { target, .. } => format!("Moving to {target}"),
            Phase::Settling { .. } => "Waiting for the image to settle".into(),
            Phase::Collecting { values, verify, .. } => format!(
                "{}: {}/{} valid frames",
                if *verify {
                    "Verifying best focus".into()
                } else {
                    format!("Point {}/{}", self.index + 1, self.positions.len())
                },
                values.len(),
                self.settings.frames
            ),
        }
    }
    fn tick(
        &mut self,
        now: Instant,
        state: State,
        frame: Option<FrameSample>,
    ) -> Result<Option<i32>, String> {
        let frame = frame
            .filter(|f| fresh(*f, now))
            .ok_or("Camera frames stopped or became stale; autofocus stopped")?;
        if frame.geometry != self.reference.geometry
            || frame.exposure != self.reference.exposure
            || frame.gain != self.reference.gain
        {
            return Err("Camera geometry, exposure or gain changed; autofocus stopped".into());
        }
        if state.position < self.settings.minimum || state.position > self.settings.maximum {
            return Err("Focuser moved outside your safe travel limits".into());
        }
        match &mut self.phase {
            Phase::Moving {
                target,
                next,
                since,
                verify,
            } => {
                if now.duration_since(*since) > Duration::from_secs(60) {
                    return Err("Focuser movement timed out".into());
                }
                if !state.moving && state.position == *target {
                    if let Some(dest) = next.take() {
                        *target = dest;
                        *since = now;
                        return Ok(Some(dest));
                    }
                    self.phase = Phase::Settling {
                        until: now + Duration::from_millis(self.settings.settle_ms),
                        verify: *verify,
                    };
                } else if !state.moving && now.duration_since(*since) > Duration::from_secs(2) {
                    return Err("Focuser stopped short of the requested position".into());
                }
            }
            Phase::Settling { until, verify } => {
                let expected = if *verify {
                    self.target.unwrap()
                } else {
                    self.positions[self.index]
                };
                if state.moving || state.position != expected {
                    return Err("Unexpected focuser movement during settling".into());
                }
                if now >= *until {
                    self.phase = Phase::Collecting {
                        since: now,
                        last_seq: frame.seq,
                        values: Vec::new(),
                        verify: *verify,
                    };
                }
            }
            Phase::Collecting {
                since,
                last_seq,
                values,
                verify,
            } => {
                let expected = if *verify {
                    self.target.unwrap()
                } else {
                    self.positions[self.index]
                };
                if state.moving || state.position != expected {
                    return Err("Focuser moved during the measurement burst".into());
                }
                if now.duration_since(*since) > Duration::from_secs(30) {
                    return Err("Not enough valid limb frames in 30 seconds; check the limb, exposure and seeing".into());
                }
                if frame.seq != *last_seq && frame.captured > *since {
                    *last_seq = frame.seq;
                    if let Some(value) = frame.score(self.mask) {
                        values.push(value);
                    }
                }
                if values.len() >= self.settings.frames
                    && now.duration_since(*since) >= Duration::from_secs(1)
                {
                    let value = burst_value(values);
                    if *verify {
                        let best = self
                            .samples
                            .iter()
                            .map(|s| s.value)
                            .fold(f64::INFINITY, f64::min);
                        if value > best * 1.15 {
                            return Err("Verification is softer than the sweep minimum; best focus is not confirmed".into());
                        }
                        self.outcome = Some(Outcome {
                            position: expected,
                            width: value,
                        });
                        return Ok(None);
                    }
                    self.samples.push(VSample {
                        pos: expected as f64,
                        value,
                        weight: values.len() as f64,
                    });
                    self.index += 1;
                    let (target, next, verify) = if self.index == self.positions.len() {
                        let best = fitted_target(&self.samples, self.settings.step)?;
                        self.target = Some(best);
                        (
                            best - self.settings.approach,
                            (self.settings.approach > 0).then_some(best),
                            true,
                        )
                    } else {
                        (self.positions[self.index], None, false)
                    };
                    if !(self.settings.minimum..=self.settings.maximum).contains(&target) {
                        return Err("Final approach would exceed safe travel limits".into());
                    }
                    self.phase = Phase::Moving {
                        target,
                        next,
                        since: now,
                        verify,
                    };
                    return Ok(Some(target));
                }
            }
        }
        Ok(None)
    }
}

#[derive(Clone, Default)]
pub struct Snapshot {
    pub devices: Vec<Info>,
    pub state: Option<State>,
    pub connected: Option<String>,
    pub busy: bool,
    pub message: String,
    pub samples: Vec<VSample>,
    pub outcome: Option<Outcome>,
}
enum Command {
    Scan,
    Connect(Info),
    Disconnect,
    Move(i32, i32, i32),
    Start(Settings),
}
pub struct Controller {
    tx: mpsc::SyncSender<Command>,
    view: Arc<Mutex<Snapshot>>,
    frame: Arc<Mutex<Option<FrameSample>>>,
    cancel: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    pending: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    pub settings: Settings,
    selected: usize,
    jog: i32,
}
impl Default for Controller {
    fn default() -> Self {
        let (tx, rx) = mpsc::sync_channel(8);
        let view = Arc::new(Mutex::new(Snapshot::default()));
        let frame = Arc::new(Mutex::new(None));
        let cancel = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(AtomicBool::new(false));
        let p = pending.clone();
        let (v, f, c, s) = (
            view.clone(),
            frame.clone(),
            cancel.clone(),
            shutdown.clone(),
        );
        let worker = std::thread::spawn(move || {
            worker(rx, v, f, c, s, p, |info| {
                Focuser::open(&info.id).map(|d| Box::new(d) as Box<dyn Motor>)
            })
        });
        Self {
            tx,
            view,
            frame,
            cancel,
            shutdown,
            pending,
            worker: Some(worker),
            settings: Settings::default(),
            selected: 0,
            jog: 10,
        }
    }
}
impl Controller {
    pub fn snapshot(&self) -> Snapshot {
        let mut view = self.view.lock().unwrap().clone();
        view.busy |= self.pending.load(Ordering::Acquire);
        view
    }
    pub fn busy(&self) -> bool {
        self.snapshot().busy
    }
    pub fn feed(&self, frame: FrameSample) {
        *self.frame.lock().unwrap() = Some(frame);
    }
    pub fn stop(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    fn send(&self, command: Command) {
        let mut view = self.view.lock().unwrap();
        if view.busy || self.pending.swap(true, Ordering::AcqRel) {
            return;
        }
        match self.tx.try_send(command) {
            Ok(()) => view.busy = true,
            Err(e) => {
                self.pending.store(false, Ordering::Release);
                view.message = format!("Focuser command could not be queued: {e}");
            }
        }
    }
    pub fn ui(&mut self, ui: &mut egui::Ui, ready: bool, motion_allowed: bool) {
        let snapshot = self.snapshot();
        ui.separator();
        ui.heading("ToupTek USB focuser");
        if snapshot.busy || snapshot.connected.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        if ui
            .add_enabled(
                snapshot.connected.is_some() || snapshot.busy,
                egui::Button::new("■ STOP FOCUSER").fill(egui::Color32::from_rgb(140, 35, 25)),
            )
            .clicked()
        {
            self.stop();
        }
        ui.label(&snapshot.message);
        if let Some(outcome) = &snapshot.outcome {
            ui.label(format!(
                "Verified focus: {} steps · {:.2} px",
                outcome.position, outcome.width
            ));
        }
        ui.add_enabled_ui(!snapshot.busy, |ui| {
            if snapshot.connected.is_none() {
                if ui.button("Scan USB focusers").clicked() {
                    self.send(Command::Scan);
                }
                egui::ComboBox::from_id_salt("usb_focuser")
                    .selected_text(
                        snapshot
                            .devices
                            .get(self.selected)
                            .map(|d| d.name.as_str())
                            .unwrap_or("Select focuser"),
                    )
                    .show_ui(ui, |ui| {
                        for (i, d) in snapshot.devices.iter().enumerate() {
                            ui.selectable_value(&mut self.selected, i, &d.name);
                        }
                    });
                if let Some(info) = snapshot.devices.get(self.selected) {
                    if ui.button("Connect focuser").clicked() {
                        self.send(Command::Connect(info.clone()));
                    }
                }
            } else if ui.button("Disconnect focuser").clicked() {
                self.send(Command::Disconnect);
            }
        });
        if let Some(state) = snapshot.state {
            ui.label(format!(
                "Position: {} / {} steps{}",
                state.position,
                state.max_step,
                state
                    .temperature
                    .map(|t| format!(" · {t:.1} °C"))
                    .unwrap_or_default()
            ));
            ui.add_enabled_ui(!snapshot.busy && !state.moving, |ui| {
                ui.label("Set safe mechanical travel limits before moving:");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.settings.minimum).range(0..=state.max_step).prefix("min "));
                    ui.add(egui::DragValue::new(&mut self.settings.maximum).range(0..=state.max_step).prefix("max "));
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.jog).range(1..=10000).prefix("jog "));
                    for (label, delta) in [("− steps", -self.jog), ("+ steps", self.jog)] {
                        let target = state.position.saturating_add(delta);
                        let allowed = motion_allowed && self.settings.minimum < self.settings.maximum && (self.settings.minimum..=self.settings.maximum).contains(&target);
                        if ui.add_enabled(allowed, egui::Button::new(label)).clicked() { self.send(Command::Move(target, self.settings.minimum, self.settings.maximum)); }
                    }
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.settings.step).range(1..=10000).prefix("sweep step "));
                    egui::ComboBox::from_id_salt("af_points").selected_text(format!("{} points", self.settings.points)).show_ui(ui, |ui| {
                        for n in [5, 7, 9, 11] { ui.selectable_value(&mut self.settings.points, n, n.to_string()); }
                    });
                });
                ui.add(egui::DragValue::new(&mut self.settings.approach).range(0..=10000).prefix("backlash approach (steps) "));
                ui.add(egui::DragValue::new(&mut self.settings.settle_ms).range(250..=10000).prefix("settle (ms) "));
                ui.add(egui::DragValue::new(&mut self.settings.frames).range(20..=200).prefix("frames per point "));
                let plan = self.settings.positions(state);
                match &plan {
                    Ok(positions) => { ui.label(format!("Sweep {} → {}; first approach {}", positions[0], positions.last().unwrap(), positions[0] - self.settings.approach)); }
                    Err(e) => { ui.label(e); }
                }
                if !ready { ui.label("Start a camera with a visible solar limb, turn off auto-exposure, and stop recording to autofocus."); }
                if ui.add_enabled(ready && plan.is_ok(), egui::Button::new("Start solar-edge autofocus").fill(crate::ACCENT_DIM)).clicked() {
                    self.send(Command::Start(self.settings.clone()));
                }
            });
        }
    }
}
impl Drop for Controller {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

trait Motor {
    fn state(&self) -> Result<State, String>;
    fn move_to(&mut self, position: i32) -> Result<(), String>;
    fn halt(&mut self) -> Result<(), String>;
}
impl Motor for Focuser {
    fn state(&self) -> Result<State, String> {
        self.state()
    }
    fn move_to(&mut self, position: i32) -> Result<(), String> {
        self.move_to(position)
    }
    fn halt(&mut self) -> Result<(), String> {
        self.halt()
    }
}

fn worker(
    rx: mpsc::Receiver<Command>,
    view: Arc<Mutex<Snapshot>>,
    frame: Arc<Mutex<Option<FrameSample>>>,
    cancel: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    pending: Arc<AtomicBool>,
    connect: impl Fn(&Info) -> Result<Box<dyn Motor>, String>,
) {
    let mut device: Option<Box<dyn Motor>> = None;
    let mut run: Option<Run> = None;
    let mut manual: Option<(i32, Instant, i32, i32)> = None;
    let mut snapshot = Snapshot::default();
    let mut last_poll = Instant::now() - Duration::from_secs(1);
    loop {
        if cancel.swap(false, Ordering::AcqRel) || shutdown.load(Ordering::Acquire) {
            run = None;
            manual = None;
            while rx.try_recv().is_ok() {
                pending.store(false, Ordering::Release);
            }
            snapshot.message = match device.as_mut().map(|d| d.halt()).transpose() {
                Ok(_) => "Stopped; no return move requested".into(),
                Err(e) => {
                    device = None;
                    snapshot.connected = None;
                    snapshot.state = None;
                    format!("STOP failed: {e}; check the focuser")
                }
            };
            snapshot.busy = false;
            *view.lock().unwrap() = snapshot.clone();
            if shutdown.load(Ordering::Acquire) {
                break;
            }
        }
        // No USB calls on the egui thread. Cancellation is checked every 25 ms.
        let command = rx.recv_timeout(Duration::from_millis(25)).ok();
        let consumed = command.is_some();
        if cancel.load(Ordering::Acquire) || shutdown.load(Ordering::Acquire) {
            if consumed {
                pending.store(false, Ordering::Release);
            }
            continue;
        }
        let result = (|| -> Result<(), String> {
            if let Some(command) = command {
                if run.is_some() || manual.is_some() {
                    return Err("A focuser operation is already active".into());
                }
                match command {
                    Command::Scan => {
                        snapshot.devices = focuser::enumerate()?;
                        snapshot.message =
                            format!("Found {} ToupTek focuser(s)", snapshot.devices.len());
                    }
                    Command::Connect(info) => {
                        snapshot.outcome = None;
                        let d = connect(&info)?;
                        snapshot.state = Some(d.state()?);
                        snapshot.connected = Some(info.name);
                        snapshot.message =
                            "Connected. Set safe travel limits before moving.".into();
                        device = Some(d);
                    }
                    Command::Disconnect => {
                        snapshot.outcome = None;
                        if let Some(d) = device.as_mut() {
                            d.halt()?;
                        }
                        device = None;
                        snapshot.state = None;
                        snapshot.connected = None;
                        snapshot.message = "Focuser disconnected".into();
                    }
                    Command::Move(target, minimum, maximum) => {
                        let d = device.as_mut().ok_or("Focuser is not connected")?;
                        let state = d.state()?;
                        if state.moving
                            || minimum < 0
                            || maximum > state.max_step
                            || minimum >= maximum
                            || !(minimum..=maximum).contains(&state.position)
                            || !(minimum..=maximum).contains(&target)
                        {
                            return Err(
                                "Jog is outside safe travel limits, or focuser is already moving"
                                    .into(),
                            );
                        }
                        d.move_to(target)?;
                        snapshot.outcome = None;
                        manual = Some((target, Instant::now(), minimum, maximum));
                        snapshot.message = format!("Moving to {target}");
                    }
                    Command::Start(settings) => {
                        let d = device.as_mut().ok_or("Focuser is not connected")?;
                        let sample = frame
                            .lock()
                            .unwrap()
                            .ok_or("No camera frame is available")?;
                        let (new_run, target) =
                            Run::start(settings, d.state()?, sample, Instant::now())?;
                        crate::applog!("autofocus: starting {} points, step {}, safe range {}..{}, approach {}, {} frames per point", new_run.settings.points, new_run.settings.step, new_run.settings.minimum, new_run.settings.maximum, new_run.settings.approach, new_run.settings.frames);
                        snapshot.samples.clear();
                        snapshot.outcome = None;
                        d.move_to(target)?;
                        run = Some(new_run);
                    }
                }
            }
            if last_poll.elapsed() >= Duration::from_millis(100) {
                last_poll = Instant::now();
                if let Some(d) = device.as_mut() {
                    let state = d.state()?;
                    snapshot.state = Some(state);
                    if let Some((target, since, min, max)) = manual {
                        if state.position < min || state.position > max {
                            return Err("Position exceeded safe travel limits".into());
                        }
                        if !state.moving && state.position == target {
                            manual = None;
                            snapshot.message = format!("At {target} steps");
                        } else if since.elapsed() > Duration::from_secs(60)
                            || (!state.moving && since.elapsed() > Duration::from_secs(2))
                        {
                            return Err("Focuser failed to reach the jog target".into());
                        }
                    }
                    if let Some(active) = run.as_mut() {
                        let previous_count = active.samples.len();
                        let effect = active.tick(Instant::now(), state, *frame.lock().unwrap());
                        snapshot.samples = active.samples.clone();
                        if active.samples.len() > previous_count {
                            let sample = active.samples.last().unwrap();
                            crate::applog!(
                                "autofocus: position {} steps, limb width {:.4} px, {} frames",
                                sample.pos,
                                sample.value,
                                sample.weight
                            );
                        }
                        if let Some(target) = effect? {
                            if cancel.load(Ordering::Acquire) {
                                return Ok(());
                            }
                            d.move_to(target)?;
                        }
                        snapshot.message = active.progress();
                        if let Some(outcome) = &active.outcome {
                            snapshot.message = format!(
                                "Autofocus complete: verified at {} steps",
                                outcome.position
                            );
                            snapshot.outcome = Some(outcome.clone());
                            crate::applog!(
                                "autofocus: verified {} steps, {:.4} px",
                                outcome.position,
                                outcome.width
                            );
                            run = None;
                        }
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            let halt = device.as_mut().map(|d| d.halt()).transpose();
            let halted = halt.is_ok();
            snapshot.message = match halt {
                Ok(_) => format!("Stopped: {error}"),
                Err(e) => format!("{error}. STOP failed: {e}; check the focuser"),
            };
            crate::applog!("autofocus: {}", snapshot.message);
            snapshot.outcome = None;
            run = None;
            manual = None;
            // A rejected fit can be retried on the same connection; a failed
            // USB query or halt must invalidate the device and require reconnect.
            let state = device.as_ref().and_then(|d| d.state().ok());
            if !halted || state.is_none() {
                device = None;
                snapshot.connected = None;
            }
            snapshot.state = if device.is_some() { state } else { None };
        }
        snapshot.busy = run.is_some() || manual.is_some();
        *view.lock().unwrap() = snapshot.clone();
        if consumed {
            pending.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(position: i32) -> State {
        State {
            position,
            max_step: 100000,
            moving: false,
            temperature: None,
        }
    }
    fn settings() -> Settings {
        Settings {
            minimum: 100,
            maximum: 900,
            ..Settings::default()
        }
    }
    fn frame(now: Instant, seq: u64, position: i32) -> FrameSample {
        let width = 2.0 + ((position as f64 - 525.0) / 150.0).powi(2);
        FrameSample {
            seq,
            captured: now,
            widths: [Some(width), Some(width + 0.2)],
            geometry: [3840, 120],
            exposure: 10000,
            gain: 100,
            clipped: false,
        }
    }
    #[test]
    fn simulated_sweep_reaches_and_verifies_known_focus_with_same_direction_approach() {
        let start = Instant::now();
        let mut config = settings();
        config.approach = 20;
        config.frames = 20;
        config.settle_ms = 250;
        let (mut run, target) =
            Run::start(config, state(500), frame(start, 0, 500), start).unwrap();
        let mut position = target;
        let mut moves = vec![position];
        for i in 1..600 {
            let now = start + Duration::from_millis(i * 100);
            if let Some(target) = run
                .tick(now, state(position), Some(frame(now, i, position)))
                .unwrap()
            {
                position = target;
                moves.push(position);
            }
            if run.outcome.is_some() {
                break;
            }
        }
        let result = run.outcome.unwrap();
        assert!((result.position - 525).abs() <= 1);
        assert_eq!(run.samples.len(), 7);
        assert_eq!(&moves[..3], &[330, 350, 400]);
        assert_eq!(&moves[moves.len() - 2..], &[505, 525]);
        assert!(result.width < 2.11);
        assert!(moves.iter().all(|p| (100..=900).contains(p)));
    }
    #[test]
    fn safe_limits_include_backlash_and_never_silently_clamp_sweep() {
        let mut config = settings();
        config.minimum = 340;
        config.approach = 20;
        assert!(config.positions(state(500)).is_err());
        config.approach = 0;
        assert_eq!(config.positions(state(500)).unwrap()[0], 350);
        config.maximum = 649;
        assert!(config.positions(state(500)).is_err());
        config.maximum = 900;
        config.step = i32::MAX;
        assert!(config.positions(state(500)).is_err());
    }
    #[test]
    fn lost_camera_and_changed_settings_stop_a_sweep() {
        let now = Instant::now();
        let sample = frame(now, 0, 500);
        let (mut run, target) = Run::start(settings(), state(500), sample, now).unwrap();
        assert!(run
            .tick(now + Duration::from_secs(4), state(target), Some(sample))
            .is_err());
        let mut changed = sample;
        changed.exposure += 1;
        assert!(run.tick(now, state(target), Some(changed)).is_err());
        changed = sample;
        changed.geometry = [120, 3840];
        assert!(run.tick(now, state(target), Some(changed)).is_err());
    }
    #[test]
    fn bursts_exclude_duplicate_and_pre_settle_frames_and_keep_the_same_limbs() {
        let start = Instant::now();
        let (mut run, target) =
            Run::start(settings(), state(500), frame(start, 0, 500), start).unwrap();
        run.phase = Phase::Collecting {
            since: start,
            last_seq: 1,
            values: vec![],
            verify: false,
        };
        let now = start + Duration::from_millis(200);
        let mut sample = frame(now, 1, target);
        run.tick(now, state(target), Some(sample)).unwrap(); // duplicate
        sample.seq = 2;
        sample.captured = start;
        run.tick(now, state(target), Some(sample)).unwrap(); // exposure predates settling
        sample.seq = 3;
        sample.captured = now;
        sample.widths[1] = None;
        run.tick(now, state(target), Some(sample)).unwrap(); // cannot switch to only one limb
        sample.seq = 4;
        sample.widths[1] = Some(2.0);
        sample.clipped = true;
        run.tick(now, state(target), Some(sample)).unwrap();
        match run.phase {
            Phase::Collecting { values, .. } => assert!(values.is_empty()),
            _ => panic!(),
        }
    }
    #[test]
    fn unbracketed_flat_and_noisy_curves_are_not_accepted() {
        for f in [0, 1, 2] {
            let samples: Vec<_> = (0..7)
                .map(|i| VSample {
                    pos: i as f64 * 50.0,
                    weight: 30.0,
                    value: match f {
                        0 => 2.0,
                        1 => 2.0 + (i as f64).powi(2),
                        _ => [4.0, 2.0, 5.0, 2.0, 4.0, 2.0, 5.0][i],
                    },
                })
                .collect();
            assert!(fitted_target(&samples, 50).is_err());
        }
    }
    #[test]
    fn failed_verification_does_not_report_success() {
        let start = Instant::now();
        let (mut run, _) = Run::start(settings(), state(500), frame(start, 0, 500), start).unwrap();
        run.target = Some(525);
        run.samples = vec![VSample {
            pos: 525.0,
            value: 2.0,
            weight: 30.0,
        }];
        run.phase = Phase::Collecting {
            since: start,
            last_seq: 0,
            values: vec![5.0; 30],
            verify: true,
        };
        let now = start + Duration::from_secs(2);
        assert!(run
            .tick(now, state(525), Some(frame(now, 1, 525)))
            .unwrap_err()
            .contains("Verification"));
        assert!(run.outcome.is_none());
    }
    #[test]
    fn missing_limbs_clipping_stalls_and_timeout_are_rejected() {
        let now = Instant::now();
        let mut sample = frame(now, 0, 500);
        sample.widths = [None, None];
        assert!(Run::start(settings(), state(500), sample, now).is_err());
        sample = frame(now, 0, 500);
        sample.clipped = true;
        assert!(Run::start(settings(), state(500), sample, now).is_err());
        let (mut run, _) = Run::start(settings(), state(500), frame(now, 0, 500), now).unwrap();
        let later = now + Duration::from_secs(3);
        assert!(run
            .tick(later, state(500), Some(frame(later, 1, 500)))
            .is_err());
        run.phase = Phase::Collecting {
            since: now,
            last_seq: 0,
            values: vec![],
            verify: false,
        };
        let later = now + Duration::from_secs(31);
        assert!(run
            .tick(later, state(350), Some(frame(later, 1, 350)))
            .unwrap_err()
            .contains("30 seconds"));
    }
}

#[cfg(test)]
mod worker_tests {
    use super::*;
    #[derive(Default)]
    struct Fake {
        commands: Vec<String>,
        moving: bool,
        unplugged: bool,
    }
    struct FakeMotor(Arc<Mutex<Fake>>);
    impl Motor for FakeMotor {
        fn state(&self) -> Result<State, String> {
            let fake = self.0.lock().unwrap();
            if fake.unplugged {
                return Err("USB disconnected".into());
            }
            Ok(State {
                position: 500,
                max_step: 1000,
                moving: fake.moving,
                temperature: None,
            })
        }
        fn move_to(&mut self, position: i32) -> Result<(), String> {
            let mut fake = self.0.lock().unwrap();
            fake.commands.push(format!("move {position}"));
            fake.moving = true;
            Ok(())
        }
        fn halt(&mut self) -> Result<(), String> {
            let mut fake = self.0.lock().unwrap();
            fake.commands.push("halt".into());
            fake.moving = false;
            Ok(())
        }
    }
    fn wait_until(condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !condition() {
            assert!(Instant::now() < deadline, "worker did not respond");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn controller(fake: Arc<Mutex<Fake>>) -> Controller {
        let (tx, rx) = mpsc::sync_channel(8);
        let view = Arc::new(Mutex::new(Snapshot::default()));
        let frame = Arc::new(Mutex::new(None));
        let cancel = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(AtomicBool::new(false));
        let (v, f, c, s, p) = (
            view.clone(),
            frame.clone(),
            cancel.clone(),
            shutdown.clone(),
            pending.clone(),
        );
        let worker = std::thread::spawn(move || {
            worker(rx, v, f, c, s, p, |_| Ok(Box::new(FakeMotor(fake.clone()))))
        });
        Controller {
            tx,
            view,
            frame,
            cancel,
            shutdown,
            pending,
            worker: Some(worker),
            settings: Settings::default(),
            selected: 0,
            jog: 10,
        }
    }
    #[test]
    fn connect_is_read_only_and_stop_halts_without_returning_to_start() {
        let fake = Arc::new(Mutex::new(Fake::default()));
        let control = controller(fake.clone());
        control.send(Command::Connect(Info {
            name: "Fake AAF".into(),
            id: "fake".into(),
        }));
        wait_until(|| control.snapshot().connected.is_some() && !control.busy());
        assert!(fake.lock().unwrap().commands.is_empty());
        control.send(Command::Move(550, 100, 900));
        assert!(control.busy());
        wait_until(|| fake.lock().unwrap().moving);
        control.stop();
        wait_until(|| !control.busy() && !fake.lock().unwrap().moving);
        assert_eq!(fake.lock().unwrap().commands, ["move 550", "halt"]);
    }
    #[test]
    fn unplug_invalidates_connection_and_drop_halts_active_motion() {
        let fake = Arc::new(Mutex::new(Fake::default()));
        let control = controller(fake.clone());
        control.send(Command::Connect(Info {
            name: "Fake".into(),
            id: "fake".into(),
        }));
        wait_until(|| control.snapshot().connected.is_some() && !control.busy());
        control.send(Command::Move(550, 100, 900));
        wait_until(|| fake.lock().unwrap().moving);
        fake.lock().unwrap().unplugged = true;
        wait_until(|| control.snapshot().connected.is_none());
        assert!(!control.busy());
        assert!(!fake.lock().unwrap().moving);
        assert!(control.snapshot().message.contains("USB disconnected"));
        fake.lock().unwrap().unplugged = false;
        control.send(Command::Connect(Info {
            name: "Fake".into(),
            id: "fake".into(),
        }));
        wait_until(|| control.snapshot().connected.is_some() && !control.busy());
        control.send(Command::Move(600, 100, 900));
        wait_until(|| fake.lock().unwrap().moving);
        drop(control);
        assert!(!fake.lock().unwrap().moving);
        assert_eq!(fake.lock().unwrap().commands.last().unwrap(), "halt");
    }
}
