//! Guided, end-to-end SHG scan acquisition.
//!
//! This intentionally assumes the optical setup has already been focused and
//! the solar disc is reasonably centred. The UI exposes that assumption as a
//! mandatory checklist item before any mount motion is enabled.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use ghostsun_core::image2d::Image;
use ghostsun_core::{output, pipeline, stack};

use super::focus::{self, DispAxis};
use super::mount::{Direction, MountState};
use super::{ACCENT, ACCENT_DIM};

const PROBE_RATE_INDEX: u8 = 7;
const PROBE_RATE_MULTIPLE: f64 = 60.0;
const ACQUISITION_RATES: [(&str, u8, f64); 8] = [
    ("0.25×", 0, 0.25),
    ("0.5×", 1, 0.5),
    ("1×", 2, 1.0),
    ("2×", 3, 2.0),
    ("4× · recommended", 4, 4.0),
    ("8×", 5, 8.0),
    ("20×", 6, 20.0),
    ("60×", 7, 60.0),
];
const DEFAULT_ACQUISITION_RATE: usize = 4;
const SIDEREAL_DEG_PER_SEC: f64 = 15.0 / 3600.0;
const SETTLE_TIME: Duration = Duration::from_millis(1200);
const PRE_ROLL: Duration = Duration::from_millis(350);
const POST_ROLL: Duration = Duration::from_millis(350);
const OFF_AXIS_WARN_DEG: f64 = 10.0;
const PREPOSITION_STEP_DEG: f64 = 0.08;
const PREPOSITION_SETTLE: Duration = Duration::from_millis(250);
const PREPOSITION_SAMPLE_TIMEOUT: Duration = Duration::from_millis(1500);
const PREPOSITION_CLEAR_FRACTION: f32 = 0.25;
const PREPOSITION_REQUIRED_SAMPLES: usize = 2;
const SCAN_DISC_PRESENT_FRACTION: f32 = 0.45;
const SCAN_SIGNAL_REQUIRED_SAMPLES: usize = 2;
const SCAN_TAIL_CHECK: Duration = Duration::from_millis(1500);
const RECENTER_SETTLE: Duration = Duration::from_millis(400);
const RECENTER_CHECK_TIMEOUT: Duration = Duration::from_millis(1800);

pub struct AcquireOutput {
    pub image: Image,
    pub name: String,
    pub source_ser: PathBuf,
}

enum ProcessMessage {
    Log(String),
    Done(AcquireOutput),
    Failed(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunPhase {
    PrepositionMoving,
    PrepositionSampling,
    Settling,
    AwaitRecorder,
    PreRoll,
    Scanning,
    ScanTailCheck,
    PostRoll,
    WaitingForRecorder,
    Recentering,
    RecenterCheck,
    Processing,
}

struct ScanRun {
    phase: RunPhase,
    session_dir: PathBuf,
    files: Vec<(PathBuf, bool)>,
    scan_index: usize,
    direction: Direction,
    rate_code: u8,
    rate_multiple: f64,
    scan_span_deg: f64,
    preposition_baseline: f32,
    preposition_last_seq: u64,
    preposition_steps: usize,
    preposition_max_steps: usize,
    preposition_samples: usize,
    preposition_clear_samples: usize,
    scan_last_seq: u64,
    scan_present_samples: usize,
    scan_seen_disc: bool,
    scan_clear_samples: usize,
    scan_entry_at: Option<Instant>,
    scan_exit_at: Option<Instant>,
    recenter_last_seq: u64,
    recenter_present_samples: usize,
    settle_until: Instant,
    deadline: Instant,
}

#[derive(Clone, Copy)]
enum CalPhase {
    MoveNorth,
    SettleNorth(Instant),
    ReturnSouth,
    SettleReturn(Instant),
    MoveEast,
    SettleEast(Instant),
    ReturnWest,
    SettleFinal(Instant),
}

struct Calibration {
    baseline_seq: u64,
    baseline: Vec<f32>,
    north_shift: Option<f64>,
    phase: CalPhase,
}

pub struct AcquireState {
    capture_height: usize,
    /// Read only the capture band from the sensor while recording. Measured on
    /// the G3M678M: 23 fps full-frame → 176 fps at 256 rows, and frame rate is
    /// scan-axis sampling density. Restored the moment recording stops.
    use_hw_roi: bool,
    anchor_y: Option<f64>,
    output_dir: PathBuf,
    scan_span_deg: f64,
    scan_count: usize,
    scan_rate_index: usize,
    direction: Direction,
    probe_deg: f64,
    prepared_confirmed: bool,
    motion_confirmed: bool,
    meridian_ack_side: i8,
    off_axis_confirmed: bool,
    confirm_start_open: bool,
    off_axis_deg: Option<f64>,
    calibration: Option<Calibration>,
    run: Option<ScanRun>,
    status: String,
    log: Vec<String>,
    process_tx: Sender<ProcessMessage>,
    process_rx: Receiver<ProcessMessage>,
}

impl Default for AcquireState {
    fn default() -> Self {
        let (process_tx, process_rx) = channel();
        let output_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("GhostSun Captures");
        Self {
            // ~200 px around the line is the user's working crop: the Halpha
            // core is ~10-30 px and the pipeline's continuum offset and
            // telluric anchors live within +-90 px, so 256 covers science
            // needs at a fifth of the file size of the old 1024 default
            // (a 900-frame scan: ~1.3 GB rather than ~6.6 GB).
            capture_height: 256,
            use_hw_roi: true,
            anchor_y: None,
            output_dir,
            scan_span_deg: 0.80,
            scan_count: 1,
            scan_rate_index: DEFAULT_ACQUISITION_RATE,
            direction: Direction::East,
            probe_deg: 0.06,
            prepared_confirmed: false,
            motion_confirmed: false,
            meridian_ack_side: 0,
            off_axis_confirmed: false,
            confirm_start_open: false,
            off_axis_deg: None,
            calibration: None,
            run: None,
            status: "Ready for setup".into(),
            log: Vec::new(),
            process_tx,
            process_rx,
        }
    }
}

impl AcquireState {
    pub fn enter_tab(&mut self, focus: &mut focus::FocusState) {
        if focus.cameras.is_empty() {
            focus.refresh_cameras();
        }
    }

    pub fn leave_tab(&mut self, focus: &mut focus::FocusState, mount: &mut MountState) {
        let motion_or_recording = self
            .run
            .as_ref()
            .map(|run| run.phase != RunPhase::Processing)
            .unwrap_or(false)
            || self.calibration.is_some();
        if motion_or_recording {
            self.abort(focus, mount, "Acquisition stopped when leaving the tab");
        }
    }

    pub fn poll(
        &mut self,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
        mount: &mut MountState,
    ) -> Option<AcquireOutput> {
        let mut completed = None;
        while let Ok(message) = self.process_rx.try_recv() {
            match message {
                ProcessMessage::Log(line) => {
                    self.status = line.clone();
                    self.log.push(line);
                }
                ProcessMessage::Done(output) => {
                    self.status = "Acquisition, reconstruction, and output complete".into();
                    self.log.push(self.status.clone());
                    self.run = None;
                    completed = Some(output);
                }
                ProcessMessage::Failed(error) => {
                    self.status = format!("Processing failed: {error}");
                    self.log.push(self.status.clone());
                    self.run = None;
                }
            }
        }

        if let Some(error) = mount.take_acquisition_error() {
            self.abort(focus, mount, &format!("Mount error: {error}"));
            return completed;
        }

        self.poll_calibration(focus, mount);
        self.poll_scan(ctx, focus, mount);
        ctx.request_repaint_after(Duration::from_millis(40));
        completed
    }

    fn poll_calibration(&mut self, focus: &focus::FocusState, mount: &mut MountState) {
        let Some(mut cal) = self.calibration.take() else {
            return;
        };
        let result = (|| -> Result<bool, String> {
            match cal.phase {
                CalPhase::MoveNorth if mount.take_acquisition_nudge_done() => {
                    cal.phase = CalPhase::SettleNorth(Instant::now() + SETTLE_TIME);
                }
                CalPhase::SettleNorth(until) if Instant::now() >= until => {
                    let (_, profile) = fresh_profile(focus, cal.baseline_seq)
                        .ok_or("waiting for a fresh camera profile")?;
                    cal.north_shift = Some(
                        profile_shift(&cal.baseline, &profile, 160)
                            .ok_or("could not correlate the north probe")?,
                    );
                    self.start_probe(mount, Direction::South)?;
                    cal.phase = CalPhase::ReturnSouth;
                }
                CalPhase::ReturnSouth if mount.take_acquisition_nudge_done() => {
                    cal.phase = CalPhase::SettleReturn(Instant::now() + SETTLE_TIME);
                }
                CalPhase::SettleReturn(until) if Instant::now() >= until => {
                    self.start_probe(mount, Direction::East)?;
                    cal.phase = CalPhase::MoveEast;
                }
                CalPhase::MoveEast if mount.take_acquisition_nudge_done() => {
                    cal.phase = CalPhase::SettleEast(Instant::now() + SETTLE_TIME);
                }
                CalPhase::SettleEast(until) if Instant::now() >= until => {
                    let (_, profile) = fresh_profile(focus, cal.baseline_seq)
                        .ok_or("waiting for a fresh camera profile")?;
                    let east_shift = profile_shift(&cal.baseline, &profile, 160)
                        .ok_or("could not correlate the east probe")?;
                    let north_shift = cal.north_shift.unwrap_or(0.0);
                    self.direction = if north_shift.abs() <= east_shift.abs() {
                        Direction::North
                    } else {
                        Direction::East
                    };
                    self.off_axis_deg = Some(off_axis_angle(north_shift, east_shift));
                    self.off_axis_confirmed = false;
                    self.start_probe(mount, Direction::West)?;
                    cal.phase = CalPhase::ReturnWest;
                    self.log.push(format!(
                        "Direction calibration: N shift {north_shift:+.2} px, E shift {east_shift:+.2} px"
                    ));
                }
                CalPhase::ReturnWest if mount.take_acquisition_nudge_done() => {
                    cal.phase = CalPhase::SettleFinal(Instant::now() + SETTLE_TIME);
                }
                CalPhase::SettleFinal(until) if Instant::now() >= until => {
                    let angle = self.off_axis_deg.unwrap_or(0.0);
                    self.status = format!(
                        "Calibration complete: scan {} / {}, estimated sensor offset {:.1}°",
                        self.direction.label(),
                        self.direction.opposite().label(),
                        angle
                    );
                    self.log.push(self.status.clone());
                    return Ok(true);
                }
                _ => {}
            }
            Ok(false)
        })();

        match result {
            Ok(true) => {}
            Ok(false) => self.calibration = Some(cal),
            Err(error) if error == "waiting for a fresh camera profile" => {
                self.calibration = Some(cal);
            }
            Err(error) => {
                mount.stop_acquisition_motion();
                self.status = format!("Direction calibration failed: {error}");
                self.log.push(self.status.clone());
            }
        }
    }

    fn poll_scan(
        &mut self,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
        mount: &mut MountState,
    ) {
        let Some(mut run) = self.run.take() else {
            return;
        };
        if run.phase == RunPhase::Processing {
            self.run = Some(run);
            return;
        }

        let outcome = (|| -> Result<bool, String> {
            match run.phase {
                RunPhase::PrepositionMoving if mount.take_acquisition_nudge_done() => {
                    run.phase = RunPhase::PrepositionSampling;
                    run.settle_until = Instant::now() + PREPOSITION_SETTLE;
                    run.deadline = Instant::now() + PREPOSITION_SAMPLE_TIMEOUT;
                    run.preposition_samples = 0;
                    run.preposition_clear_samples = 0;
                    self.status = format!(
                        "Seeking first limb: moved {:.2}°, waiting for camera",
                        run.preposition_steps as f64 * PREPOSITION_STEP_DEG
                    );
                }
                RunPhase::PrepositionSampling if Instant::now() >= run.settle_until => {
                    if let Some((seq, signal)) = focus
                        .sun_signal_sample()
                        .filter(|(seq, _)| *seq > run.preposition_last_seq)
                    {
                        run.preposition_last_seq = seq;
                        run.preposition_samples += 1;
                        let fraction = signal / run.preposition_baseline;
                        if fraction <= PREPOSITION_CLEAR_FRACTION {
                            run.preposition_clear_samples += 1;
                        } else {
                            run.preposition_clear_samples = 0;
                        }
                        self.status = format!(
                            "Seeking first limb: {:.2}° moved, disc signal {:.0}%",
                            run.preposition_steps as f64 * PREPOSITION_STEP_DEG,
                            (fraction * 100.0).clamp(0.0, 999.0)
                        );

                        if run.preposition_clear_samples >= PREPOSITION_REQUIRED_SAMPLES {
                            let distance =
                                run.preposition_steps as f64 * PREPOSITION_STEP_DEG;
                            // The configured span is a minimum. If the camera
                            // proves that the first limb is farther away than
                            // the centred-disc assumption, extend the recorded
                            // sweep by the same excess distance.
                            run.scan_span_deg +=
                                (distance - run.scan_span_deg / 2.0).max(0.0);
                            run.phase = RunPhase::Settling;
                            run.settle_until = Instant::now() + SETTLE_TIME;
                            self.status = format!(
                                "Disc has cleared the slit; settling for a {:.2}° scan",
                                run.scan_span_deg
                            );
                        } else if run.preposition_samples >= PREPOSITION_REQUIRED_SAMPLES {
                            start_preposition_step(&mut run, mount)?;
                            self.status = format!(
                                "Disc still visible; extending edge search to {:.2}°",
                                run.preposition_steps as f64 * PREPOSITION_STEP_DEG
                            );
                        }
                    } else if Instant::now() >= run.deadline {
                        return Err(
                            "camera preview did not provide a fresh frame during edge search"
                                .into(),
                        );
                    }
                }
                RunPhase::Settling if Instant::now() >= run.settle_until => {
                    let path = scan_path(&run.session_dir, run.scan_index);
                    focus.start_ser_recording(
                        ctx,
                        path,
                        self.capture_height,
                        self.anchor_y.ok_or("spectral anchor was lost")?,
                        self.use_hw_roi,
                    )?;
                    run.phase = RunPhase::AwaitRecorder;
                    run.deadline = Instant::now() + Duration::from_secs(8);
                    self.status = format!(
                        "Scan {}/{}: opening SER",
                        run.scan_index + 1,
                        self.scan_count
                    );
                }
                RunPhase::AwaitRecorder if focus.recording_status.starts_with("recording") => {
                    run.phase = RunPhase::PreRoll;
                    run.deadline = Instant::now() + PRE_ROLL;
                    self.status = format!(
                        "Scan {}/{}: recording pre-roll",
                        run.scan_index + 1,
                        self.scan_count
                    );
                }
                RunPhase::AwaitRecorder if Instant::now() >= run.deadline => {
                    return Err(format!(
                        "SER recorder did not start: {}",
                        focus.recording_status
                    ));
                }
                RunPhase::PreRoll if Instant::now() >= run.deadline => {
                    let direction = scan_direction(run.direction, run.scan_index);
                    run.scan_last_seq = focus
                        .sun_signal_sample()
                        .map(|(seq, _)| seq)
                        .unwrap_or(run.scan_last_seq);
                    run.scan_present_samples = 0;
                    run.scan_seen_disc = false;
                    run.scan_clear_samples = 0;
                    run.scan_entry_at = None;
                    run.scan_exit_at = None;
                    mount.start_acquisition_nudge(
                        direction,
                        scan_duration(run.scan_span_deg, run.rate_multiple),
                        run.rate_code,
                    )?;
                    run.phase = RunPhase::Scanning;
                    self.status = format!(
                        "Scan {}/{}: moving {} and watching for the far limb",
                        run.scan_index + 1,
                        self.scan_count,
                        direction.label()
                    );
                }
                RunPhase::Scanning => {
                    if let Some((fraction, cleared)) = observe_scan_signal(&mut run, focus) {
                        self.status = if run.scan_seen_disc {
                            format!(
                                "Scan {}/{}: disc signal {:.0}%, seeking far limb",
                                run.scan_index + 1,
                                self.scan_count,
                                (fraction * 100.0).clamp(0.0, 999.0)
                            )
                        } else {
                            format!(
                                "Scan {}/{}: waiting for disc to enter ({:.0}%)",
                                run.scan_index + 1,
                                self.scan_count,
                                (fraction * 100.0).clamp(0.0, 999.0)
                            )
                        };
                        if cleared {
                            mount.stop_acquisition_motion();
                            run.phase = RunPhase::PostRoll;
                            run.deadline = Instant::now() + POST_ROLL;
                            self.status = format!(
                                "Scan {}/{}: far limb cleared; stopping and recording post-roll",
                                run.scan_index + 1,
                                self.scan_count
                            );
                        }
                    }
                    if run.phase == RunPhase::Scanning
                        && mount.take_acquisition_nudge_done()
                    {
                        // The last camera update can arrive just after the
                        // timed motion completion. Allow a short stationary
                        // tail before deciding the safety span was insufficient.
                        run.phase = RunPhase::ScanTailCheck;
                        run.deadline = Instant::now() + SCAN_TAIL_CHECK;
                        self.status = "Motion span complete; confirming the disc is off-sensor"
                            .into();
                    }
                }
                RunPhase::ScanTailCheck => {
                    if let Some((fraction, cleared)) = observe_scan_signal(&mut run, focus) {
                        self.status = format!(
                            "Confirming far limb: disc signal {:.0}%",
                            (fraction * 100.0).clamp(0.0, 999.0)
                        );
                        if cleared {
                            run.phase = RunPhase::PostRoll;
                            run.deadline = Instant::now() + POST_ROLL;
                            self.status = format!(
                                "Scan {}/{}: far limb confirmed; recording post-roll",
                                run.scan_index + 1,
                                self.scan_count
                            );
                        }
                    }
                    if run.phase == RunPhase::ScanTailCheck && Instant::now() >= run.deadline {
                        return Err(if run.scan_seen_disc {
                            format!(
                                "the disc did not clear the sensor within the {:.2}° safety span",
                                run.scan_span_deg
                            )
                        } else {
                            "the camera never detected the solar disc during the scan".into()
                        });
                    }
                }
                RunPhase::PostRoll if Instant::now() >= run.deadline => {
                    focus.stop_ser_recording();
                    run.phase = RunPhase::WaitingForRecorder;
                    self.status = "Finalising lossless mono16 SER".into();
                }
                RunPhase::WaitingForRecorder if !focus.recording => {
                    if focus.recording_status.starts_with("SER recording failed") {
                        return Err(focus.recording_status.clone());
                    }
                    let path = scan_path(&run.session_dir, run.scan_index);
                    let reversed = run.scan_index % 2 == 1;
                    run.files.push((path, reversed));
                    // The worker's summary carries capture mode and achieved fps.
                    self.log.push(focus.recording_status.clone());
                    run.scan_index += 1;
                    if run.scan_index < self.scan_count {
                        run.phase = RunPhase::Settling;
                        run.settle_until = Instant::now() + SETTLE_TIME;
                        self.status =
                            format!("Scan {} saved; settling for reverse scan", run.scan_index);
                    } else {
                        let final_direction =
                            scan_direction(run.direction, run.scan_index.saturating_sub(1));
                        let (distance_deg, duration) = recenter_motion(&run)?;
                        mount.start_acquisition_nudge(
                            final_direction.opposite(),
                            duration,
                            PROBE_RATE_INDEX,
                        )?;
                        run.phase = RunPhase::Recentering;
                        run.recenter_last_seq = focus
                            .sun_signal_sample()
                            .map(|(seq, _)| seq)
                            .unwrap_or(run.scan_last_seq);
                        run.recenter_present_samples = 0;
                        self.status = format!(
                            "Scans saved; re-centring {:.2}° toward {}",
                            distance_deg,
                            final_direction.opposite().label()
                        );
                    }
                }
                RunPhase::Recentering if mount.take_acquisition_nudge_done() => {
                    run.phase = RunPhase::RecenterCheck;
                    run.settle_until = Instant::now() + RECENTER_SETTLE;
                    run.deadline = Instant::now() + RECENTER_CHECK_TIMEOUT;
                    self.status = "Re-centred position reached; checking camera signal".into();
                }
                RunPhase::RecenterCheck if Instant::now() >= run.settle_until => {
                    if let Some((seq, signal)) = focus
                        .sun_signal_sample()
                        .filter(|(seq, signal)| {
                            *seq > run.recenter_last_seq && signal.is_finite()
                        })
                    {
                        run.recenter_last_seq = seq;
                        let fraction = signal / run.preposition_baseline;
                        if fraction >= SCAN_DISC_PRESENT_FRACTION {
                            run.recenter_present_samples += 1;
                        } else {
                            run.recenter_present_samples = 0;
                        }
                        self.status = format!(
                            "Verifying re-centre: disc signal {:.0}%",
                            (fraction * 100.0).clamp(0.0, 999.0)
                        );
                        if run.recenter_present_samples >= SCAN_SIGNAL_REQUIRED_SAMPLES {
                            self.status =
                                "Disc re-centred and verified; starting reconstruction".into();
                            self.begin_processing(&mut run, ctx);
                        }
                    }
                    if run.phase == RunPhase::RecenterCheck && Instant::now() >= run.deadline {
                        self.log.push(
                            "Warning: re-centre motion completed, but camera signal was not verified"
                                .into(),
                        );
                        self.status =
                            "Re-centre could not be verified; processing the saved scans".into();
                        self.begin_processing(&mut run, ctx);
                    }
                }
                _ => {}
            }
            Ok(false)
        })();

        match outcome {
            Ok(_) => self.run = Some(run),
            Err(error) => {
                mount.stop_acquisition_motion();
                focus.stop_ser_recording();
                self.status = format!("Acquisition failed: {error}");
                self.log.push(self.status.clone());
            }
        }
    }

    fn start_calibration(
        &mut self,
        focus: &focus::FocusState,
        mount: &mut MountState,
    ) -> Result<(), String> {
        self.validate_common(focus, mount)?;
        if !self.prepared_confirmed {
            return Err("confirm that the Sun is centred and focus is complete".into());
        }
        if !self.motion_confirmed {
            return Err("confirm that mount motion is safe first".into());
        }
        let (baseline_seq, baseline) = focus
            .slit_profile_sample()
            .ok_or("start the camera and wait for a profile")?;
        self.start_probe(mount, Direction::North)?;
        self.calibration = Some(Calibration {
            baseline_seq,
            baseline,
            north_shift: None,
            phase: CalPhase::MoveNorth,
        });
        self.status = "Calibrating scan direction: probing North".into();
        self.log.push(self.status.clone());
        Ok(())
    }

    fn start_probe(&self, mount: &mut MountState, direction: Direction) -> Result<(), String> {
        let duration = Duration::from_secs_f64(
            self.probe_deg.clamp(0.01, 0.15) / (PROBE_RATE_MULTIPLE * SIDEREAL_DEG_PER_SEC),
        );
        mount.start_acquisition_nudge(direction, duration, PROBE_RATE_INDEX)
    }

    fn start_scan(
        &mut self,
        focus: &focus::FocusState,
        mount: &mut MountState,
    ) -> Result<(), String> {
        self.validate_common(focus, mount)?;
        if !self.prepared_confirmed {
            return Err("confirm that the Sun is centred and focus is complete".into());
        }
        if !self.motion_confirmed {
            return Err("confirm that mount motion is safe".into());
        }
        if self
            .off_axis_deg
            .map(|angle| angle > OFF_AXIS_WARN_DEG)
            .unwrap_or(false)
            && !self.off_axis_confirmed
        {
            return Err("confirm the off-axis warning or realign the sensor".into());
        }
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| format!("cannot create output folder: {e}"))?;
        let session_dir = self.output_dir.join(format!("scan-{}", unix_timestamp()));
        std::fs::create_dir_all(&session_dir)
            .map_err(|e| format!("cannot create session folder: {e}"))?;
        let rate_index = self
            .scan_rate_index
            .min(ACQUISITION_RATES.len().saturating_sub(1));
        let (_, rate_code, rate_multiple) = ACQUISITION_RATES[rate_index];
        let (preposition_last_seq, preposition_baseline) = focus
            .sun_signal_sample()
            .filter(|(_, signal)| signal.is_finite() && *signal > 0.0)
            .ok_or("wait for a valid live camera signal before beginning")?;
        let configured_span = self.scan_span_deg.clamp(0.55, 2.0);
        let mut run = ScanRun {
            phase: RunPhase::PrepositionMoving,
            session_dir,
            files: Vec::new(),
            scan_index: 0,
            direction: self.direction,
            rate_code,
            rate_multiple,
            scan_span_deg: configured_span,
            preposition_baseline,
            preposition_last_seq,
            preposition_steps: 0,
            preposition_max_steps: (configured_span / PREPOSITION_STEP_DEG).ceil() as usize,
            preposition_samples: 0,
            preposition_clear_samples: 0,
            scan_last_seq: preposition_last_seq,
            scan_present_samples: 0,
            scan_seen_disc: false,
            scan_clear_samples: 0,
            scan_entry_at: None,
            scan_exit_at: None,
            recenter_last_seq: preposition_last_seq,
            recenter_present_samples: 0,
            settle_until: Instant::now(),
            deadline: Instant::now(),
        };
        start_preposition_step(&mut run, mount)?;
        self.run = Some(run);
        self.status = "Camera-guided pre-positioning: seeking the first solar limb".into();
        self.log.clear();
        self.log.push(self.status.clone());
        Ok(())
    }

    fn validate_common(&self, focus: &focus::FocusState, mount: &MountState) -> Result<(), String> {
        if !mount.is_connected() {
            return Err("connect the ZWO mount on the Mount tab".into());
        }
        if !focus.streaming {
            return Err("start the camera preview".into());
        }
        if !mount.tracking_is_on() {
            return Err("mount tracking must show On before acquisition".into());
        }
        if focus.dispersion != DispAxis::Vertical {
            return Err("set dispersion to Vertical on the Focus tab".into());
        }
        if self.anchor_y.is_none() {
            return Err("select a spectral-line anchor".into());
        }
        if let Some(minutes) = mount.sun_meridian_offset_minutes() {
            let side = if minutes < 0.0 { -1 } else { 1 };
            if minutes.abs() <= 30.0 && self.meridian_ack_side != side {
                return Err(if minutes < 0.0 {
                    format!(
                        "Sun reaches the meridian in {:.0} min; review and acknowledge the meridian warning",
                        -minutes
                    )
                } else {
                    format!(
                        "Sun crossed the meridian {:.0} min ago; confirm the mount has flipped and is tracking",
                        minutes
                    )
                });
            }
        }
        if self.run.is_some() || self.calibration.is_some() {
            return Err("an acquisition operation is already running".into());
        }
        Ok(())
    }

    fn start_processing(
        &mut self,
        files: Vec<(PathBuf, bool)>,
        session_dir: PathBuf,
        ctx: &egui::Context,
    ) {
        self.status = "Reconstructing scan(s) with the high-quality pipeline".into();
        self.log.push(self.status.clone());
        let tx = self.process_tx.clone();
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = process_scans(&files, &session_dir, &tx);
            match result {
                Ok(output) => {
                    let _ = tx.send(ProcessMessage::Done(output));
                }
                Err(error) => {
                    let _ = tx.send(ProcessMessage::Failed(error));
                }
            }
            repaint.request_repaint();
        });
    }

    fn begin_processing(&mut self, run: &mut ScanRun, ctx: &egui::Context) {
        let files = run.files.clone();
        let session_dir = run.session_dir.clone();
        self.start_processing(files, session_dir, ctx);
        run.phase = RunPhase::Processing;
    }

    fn abort(&mut self, focus: &mut focus::FocusState, mount: &mut MountState, reason: &str) {
        mount.stop_acquisition_motion();
        focus.stop_ser_recording();
        self.calibration = None;
        self.run = None;
        self.status = reason.into();
        self.log.push(reason.into());
    }

    pub fn controls_ui(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
        mount: &mut MountState,
    ) {
        ui.heading("Acquisition setup");
        egui::Frame::group(ui.style())
            .fill(ui.visuals().faint_bg_color)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Required observing state")
                        .strong()
                        .color(ACCENT),
                );
                ui.label(
                    egui::RichText::new(
                        "The filtered solar disc must already be reasonably centred and the telescope/spectrograph focused.",
                    )
                    .small(),
                );
                ui.checkbox(
                    &mut self.prepared_confirmed,
                    "Sun centred and focus completed",
                );
                ui.checkbox(
                    &mut self.motion_confirmed,
                    "Mount and cables are clear; motion is safe",
                );
                self.meridian_ack_ui(ui, mount);
            });

        ui.separator();
        ui.label(format!(
            "Mount: {}",
            if mount.is_connected() {
                "connected"
            } else {
                "not connected"
            }
        ));
        ui.label(format!(
            "Camera: {}",
            if focus.streaming {
                "streaming"
            } else {
                "stopped"
            }
        ));
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!focus.streaming, egui::Button::new("Start camera"))
                .clicked()
            {
                focus.start(ctx);
            }
            if ui
                .add_enabled(
                    focus.streaming && self.run.is_none(),
                    egui::Button::new("Stop camera"),
                )
                .clicked()
            {
                focus.stop();
            }
        });

        let lines = focus.vertical_anchor_lines();
        if self.anchor_y.is_none() {
            self.anchor_y = lines
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|line| line.0);
        }
        egui::ComboBox::from_label("Spectral-line anchor")
            .selected_text(
                self.anchor_y
                    .map(|y| format!("Y = {y:.1} px"))
                    .unwrap_or_else(|| "no line detected".into()),
            )
            .show_ui(ui, |ui| {
                for (y, depth) in &lines {
                    ui.selectable_value(
                        &mut self.anchor_y,
                        Some(*y),
                        format!("Y = {y:.1} px · depth {:.1}%", depth * 100.0),
                    );
                }
            });
        if let Some(frame_height) = focus.current_frame_height() {
            self.capture_height = self.capture_height.clamp(32, frame_height);
            ui.add(
                egui::DragValue::new(&mut self.capture_height)
                    .range(32..=frame_height)
                    .prefix("vertical capture ")
                    .suffix(" px"),
            );
        }
        ui.checkbox(&mut self.use_hw_roi, "hardware ROI while recording")
            .on_hover_text(
                "The sensor reads only the capture band, so frame rate is set by                  exposure instead of full-frame readout — measured 23 → 176 fps at                  256 px on a G3M678M. More frames per second means finer scan-axis                  sampling. The full sensor comes back the moment recording stops,                  and if the camera refuses the ROI the recording falls back to the                  software crop.",
            );

        ui.horizontal_wrapped(|ui| {
            ui.label("Output");
            ui.add(
                egui::Label::new(
                    egui::RichText::new(self.output_dir.display().to_string())
                        .small()
                        .monospace(),
                )
                .wrap(),
            );
            if ui.button("Choose…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.output_dir = path;
                }
            }
        });

        ui.separator();
        ui.heading("Scan direction");
        ui.horizontal(|ui| {
            for direction in [
                Direction::North,
                Direction::South,
                Direction::East,
                Direction::West,
            ] {
                ui.selectable_value(&mut self.direction, direction, direction.label());
            }
        });
        ui.add(
            egui::Slider::new(&mut self.probe_deg, 0.02..=0.12)
                .text("calibration probe °")
                .fixed_decimals(2),
        );
        let calibrate_enabled =
            self.run.is_none() && self.calibration.is_none() && self.motion_confirmed;
        if ui
            .add_enabled(
                calibrate_enabled,
                egui::Button::new("Auto-detect scan axis"),
            )
            .clicked()
        {
            if let Err(error) = self.start_calibration(focus, mount) {
                self.status = error;
            }
        }
        if let Some(angle) = self.off_axis_deg {
            let color = if angle > OFF_AXIS_WARN_DEG {
                egui::Color32::YELLOW
            } else {
                egui::Color32::LIGHT_GREEN
            };
            ui.label(
                egui::RichText::new(format!("Estimated sensor/slit-axis offset: {angle:.1}°"))
                    .color(color),
            );
            if angle > OFF_AXIS_WARN_DEG {
                ui.label(
                    egui::RichText::new(
                        "Warning: the sensor appears off-axis. Realignment is preferred; diagonal scans waste field and complicate reconstruction.",
                    )
                    .small()
                    .color(egui::Color32::YELLOW),
                );
                ui.checkbox(
                    &mut self.off_axis_confirmed,
                    "Proceed despite off-axis warning",
                );
            }
        }

        ui.separator();
        ui.heading("Recording");
        ui.add(
            egui::Slider::new(&mut self.scan_span_deg, 0.55..=1.50)
                .text("scan span °")
                .fixed_decimals(2),
        );
        self.scan_rate_index = self
            .scan_rate_index
            .min(ACQUISITION_RATES.len().saturating_sub(1));
        egui::ComboBox::from_label("science scan rate")
            .selected_text(ACQUISITION_RATES[self.scan_rate_index].0)
            .show_ui(ui, |ui| {
                for (index, (label, _, _)) in ACQUISITION_RATES.iter().enumerate() {
                    ui.selectable_value(&mut self.scan_rate_index, index, *label);
                }
            });
        ui.add(egui::Slider::new(&mut self.scan_count, 1..=8).text("alternating scans"));
        let (_, _, rate_multiple) = ACQUISITION_RATES[self.scan_rate_index];
        let duration = scan_duration(self.scan_span_deg, rate_multiple);
        let motion_per_exposure = rate_multiple
            * 15.0
            * (focus.exposure_us as f64 / 1_000_000.0);
        ui.label(format!(
            "{rate_multiple:.2}× sidereal · {:.1} s across {:.2}° · ~{motion_per_exposure:.2}″ per exposure",
            duration.as_secs_f64(),
            self.scan_span_deg,
        ));
        ui.label(
            egui::RichText::new(
                "Multiple scans alternate up/down (or left/right), reconstruct independently, register, and robust-stack for higher SNR.",
            )
            .small()
            .weak(),
        );

        let active = self.run.is_some() || self.calibration.is_some();
        let processing = self
            .run
            .as_ref()
            .map(|run| run.phase == RunPhase::Processing)
            .unwrap_or(false);
        ui.horizontal(|ui| {
            let start = egui::Button::new(
                egui::RichText::new("Review & start…")
                    .strong()
                    .color(egui::Color32::WHITE),
            )
            .fill(ACCENT_DIM);
            if ui.add_enabled(!active, start).clicked() {
                self.confirm_start_open = true;
            }
            if ui
                .add_enabled(active && !processing, egui::Button::new("STOP"))
                .clicked()
            {
                self.abort(focus, mount, "Acquisition stopped by user");
            }
        });
        ui.separator();
        ui.label(egui::RichText::new(&self.status).strong());
        if focus.recording {
            ui.label(format!(
                "{} · {} frames",
                focus.recording_status, focus.recorded_frames
            ));
        }
        self.show_start_confirmation(ctx, focus, mount);
    }

    fn show_start_confirmation(
        &mut self,
        ctx: &egui::Context,
        focus: &focus::FocusState,
        mount: &mut MountState,
    ) {
        if !self.confirm_start_open {
            return;
        }
        let mut open = self.confirm_start_open;
        let mut begin = false;
        egui::Window::new("Confirm guided acquisition")
            .collapsible(false)
            .resizable(false)
            .default_width(470.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(
                    "GhostSun will move the mount from the current position and record across the solar disc.",
                );
                ui.add_space(6.0);
                ui.checkbox(
                    &mut self.prepared_confirmed,
                    "The filtered Sun is reasonably centred and focus is complete",
                );
                ui.checkbox(
                    &mut self.motion_confirmed,
                    "The mount, telescope, instrument and cables can move safely",
                );
                self.meridian_ack_ui(ui, mount);
                if self
                    .off_axis_deg
                    .map(|angle| angle > OFF_AXIS_WARN_DEG)
                    .unwrap_or(false)
                {
                    ui.checkbox(
                        &mut self.off_axis_confirmed,
                        "Proceed despite the sensor off-axis warning",
                    );
                }
                ui.separator();
                let camera_ready = focus.streaming;
                let mount_ready = mount.is_connected();
                let tracking_ready = mount.tracking_is_on();
                let line_ready = self.anchor_y.is_some();
                let vertical = focus.dispersion == DispAxis::Vertical;
                let meridian_ready = mount
                    .sun_meridian_offset_minutes()
                    .map(|minutes| {
                        minutes.abs() > 30.0
                            || self.meridian_ack_side == if minutes < 0.0 { -1 } else { 1 }
                    })
                    .unwrap_or(true);
                for (ready, label) in [
                    (mount_ready, "ZWO mount connected"),
                    (tracking_ready, "Mount tracking is On"),
                    (camera_ready, "Camera streaming"),
                    (vertical, "Dispersion set to Vertical"),
                    (line_ready, "Spectral-line anchor selected"),
                    (meridian_ready, "Meridian guard satisfied"),
                ] {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} {label}",
                            if ready { "✓" } else { "○" }
                        ))
                        .color(if ready {
                            egui::Color32::LIGHT_GREEN
                        } else {
                            egui::Color32::YELLOW
                        }),
                    );
                }
                ui.add_space(6.0);
                let off_axis_ready = self
                    .off_axis_deg
                    .map(|angle| angle <= OFF_AXIS_WARN_DEG || self.off_axis_confirmed)
                    .unwrap_or(true);
                let ready = self.prepared_confirmed
                    && self.motion_confirmed
                    && camera_ready
                    && mount_ready
                    && tracking_ready
                    && line_ready
                    && vertical
                    && meridian_ready
                    && off_axis_ready;
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            ready,
                            egui::Button::new(
                                egui::RichText::new("Begin scan")
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(ACCENT_DIM),
                        )
                        .clicked()
                    {
                        begin = true;
                    }
                    if !ready {
                        ui.label(
                            egui::RichText::new("Complete the items above to continue.")
                                .small()
                                .weak(),
                        );
                    }
                });
            });
        if begin {
            match self.start_scan(focus, mount) {
                Ok(()) => open = false,
                Err(error) => self.status = error,
            }
        }
        self.confirm_start_open = open;
    }

    fn meridian_ack_ui(&mut self, ui: &mut egui::Ui, mount: &MountState) {
        let Some(minutes) = mount.sun_meridian_offset_minutes() else {
            return;
        };
        if minutes.abs() > 30.0 {
            return;
        }
        let side = if minutes < 0.0 { -1 } else { 1 };
        let mut acknowledged = self.meridian_ack_side == side;
        let text = if minutes < 0.0 {
            format!(
                "Meridian in {:.0} min: this scan is supervised and will finish safely",
                -minutes
            )
        } else {
            format!(
                "Meridian passed {:.0} min ago: mount has flipped and tracking is verified",
                minutes
            )
        };
        ui.label(
            egui::RichText::new("MERIDIAN GUARD")
                .strong()
                .color(egui::Color32::YELLOW),
        );
        if ui.checkbox(&mut acknowledged, text).changed() {
            self.meridian_ack_side = if acknowledged { side } else { 0 };
        }
    }

    pub fn view_ui(&mut self, ui: &mut egui::Ui, focus: &focus::FocusState) {
        ui.heading("Live acquisition view");
        let selectable = self.run.is_none();
        if let Some(anchor) = focus.acquisition_preview_ui(
            ui,
            420.0,
            self.anchor_y,
            self.capture_height,
            selectable,
        ) {
            self.anchor_y = Some(anchor);
            self.status = format!("Spectral-line anchor selected at Y = {anchor:.1} px");
        }
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(
                    self.anchor_y
                        .map(|y| format!("Selected spectral line: Y = {y:.1} px"))
                        .unwrap_or_else(|| "No spectral line selected".into()),
                )
                .strong()
                .color(if self.anchor_y.is_some() {
                    egui::Color32::from_rgb(255, 205, 75)
                } else {
                    egui::Color32::YELLOW
                }),
            );
            if !selectable {
                ui.label(egui::RichText::new("locked during acquisition").small().weak());
            }
        });
        ui.label(
            egui::RichText::new(
                "Spectral profile (sensor Y) — click a line to use it as the SER crop anchor",
            )
            .small()
            .color(ACCENT),
        );
        let profile_pick = ui
            .add_enabled_ui(selectable, |ui| {
                focus.acquisition_spectral_profile_ui(ui, self.anchor_y)
            })
            .inner;
        if let Some(clicked) = profile_pick {
            let anchor = focus
                .vertical_anchor_lines()
                .into_iter()
                .min_by(|a, b| {
                    (a.0 - clicked)
                        .abs()
                        .total_cmp(&(b.0 - clicked).abs())
                })
                .map(|line| line.0)
                .unwrap_or(clicked);
            self.anchor_y = Some(anchor);
            self.status = format!("Spectral-line anchor selected at Y = {anchor:.1} px");
        }
        ui.separator();
        ui.label(
            egui::RichText::new(
                "GhostSun records the selected line as lossless mono16 SER, including pre/post-roll. High-quality reconstruction starts automatically after the final scan.",
            )
            .small()
            .weak(),
        );
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in &self.log {
                    ui.label(egui::RichText::new(line).small().monospace());
                }
            });
    }
}

fn scan_direction(first: Direction, index: usize) -> Direction {
    if index % 2 == 0 {
        first
    } else {
        first.opposite()
    }
}

fn scan_duration(span_deg: f64, rate_multiple: f64) -> Duration {
    Duration::from_secs_f64(
        span_deg.clamp(0.55, 2.0)
            / (rate_multiple.max(0.25) * SIDEREAL_DEG_PER_SEC),
    )
}

fn start_preposition_step(run: &mut ScanRun, mount: &mut MountState) -> Result<(), String> {
    if run.preposition_steps >= run.preposition_max_steps {
        return Err(format!(
            "the solar limb was not found after moving {:.2}°; re-centre the disc or check exposure",
            run.preposition_steps as f64 * PREPOSITION_STEP_DEG
        ));
    }
    let duration = Duration::from_secs_f64(
        PREPOSITION_STEP_DEG / (PROBE_RATE_MULTIPLE * SIDEREAL_DEG_PER_SEC),
    );
    mount.start_acquisition_nudge(run.direction.opposite(), duration, PROBE_RATE_INDEX)?;
    run.preposition_steps += 1;
    run.phase = RunPhase::PrepositionMoving;
    Ok(())
}

fn observe_scan_signal(
    run: &mut ScanRun,
    focus: &focus::FocusState,
) -> Option<(f32, bool)> {
    let (seq, signal) = focus
        .sun_signal_sample()
        .filter(|(seq, signal)| *seq > run.scan_last_seq && signal.is_finite())?;
    run.scan_last_seq = seq;
    let fraction = signal / run.preposition_baseline;

    if !run.scan_seen_disc {
        if fraction >= SCAN_DISC_PRESENT_FRACTION {
            run.scan_present_samples += 1;
        } else {
            run.scan_present_samples = 0;
        }
        if run.scan_present_samples >= SCAN_SIGNAL_REQUIRED_SAMPLES {
            run.scan_seen_disc = true;
            run.scan_clear_samples = 0;
            run.scan_entry_at = Some(Instant::now());
        }
    } else if fraction <= PREPOSITION_CLEAR_FRACTION {
        run.scan_clear_samples += 1;
    } else {
        run.scan_clear_samples = 0;
    }

    let cleared =
        run.scan_seen_disc && run.scan_clear_samples >= SCAN_SIGNAL_REQUIRED_SAMPLES;
    if cleared && run.scan_exit_at.is_none() {
        run.scan_exit_at = Some(Instant::now());
    }
    Some((fraction, cleared))
}

fn recenter_motion(run: &ScanRun) -> Result<(f64, Duration), String> {
    let entry = run
        .scan_entry_at
        .ok_or("cannot re-centre because the near-limb time was not measured")?;
    let exit = run
        .scan_exit_at
        .ok_or("cannot re-centre because the far-limb time was not measured")?;
    let crossing_seconds = exit.saturating_duration_since(entry).as_secs_f64();
    if !crossing_seconds.is_finite() || crossing_seconds <= 0.0 {
        return Err("cannot re-centre because the measured disc crossing was invalid".into());
    }

    // The midpoint between camera-observed limb crossings is the disc centre.
    // Convert half of that science-rate transit to angular distance, then
    // return over the same distance at the fast positioning rate.
    let distance_deg = (crossing_seconds * run.rate_multiple * SIDEREAL_DEG_PER_SEC / 2.0)
        .clamp(0.05, run.scan_span_deg / 2.0);
    let duration = Duration::from_secs_f64(
        distance_deg / (PROBE_RATE_MULTIPLE * SIDEREAL_DEG_PER_SEC),
    );
    Ok((distance_deg, duration))
}

fn scan_path(session_dir: &Path, index: usize) -> PathBuf {
    session_dir.join(format!("scan-{:02}.ser", index + 1))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn fresh_profile(focus: &focus::FocusState, after_seq: u64) -> Option<(u64, Vec<f32>)> {
    focus
        .slit_profile_sample()
        .filter(|(seq, _)| *seq > after_seq)
}

fn profile_shift(reference: &[f32], sample: &[f32], max_shift: isize) -> Option<f64> {
    let n = reference.len().min(sample.len());
    if n < 32 {
        return None;
    }
    let max_shift = max_shift.min((n / 3) as isize).max(1);
    let mut scores = Vec::new();
    for shift in -max_shift..=max_shift {
        let start_ref = if shift < 0 { (-shift) as usize } else { 0 };
        let start_sample = if shift > 0 { shift as usize } else { 0 };
        let len = n.saturating_sub(start_ref.max(start_sample));
        if len < 24 {
            scores.push(f64::NEG_INFINITY);
            continue;
        }
        let a = &reference[start_ref..start_ref + len];
        let b = &sample[start_sample..start_sample + len];
        let ma = a.iter().map(|v| *v as f64).sum::<f64>() / len as f64;
        let mb = b.iter().map(|v| *v as f64).sum::<f64>() / len as f64;
        let mut numerator = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for (&va, &vb) in a.iter().zip(b) {
            let xa = va as f64 - ma;
            let xb = vb as f64 - mb;
            numerator += xa * xb;
            da += xa * xa;
            db += xb * xb;
        }
        scores.push(numerator / (da * db).sqrt().max(1e-12));
    }
    let best = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?
        .0;
    let integer = best as isize - max_shift;
    let refined = if best > 0 && best + 1 < scores.len() {
        let (ym, y0, yp) = (scores[best - 1], scores[best], scores[best + 1]);
        let denom = ym - 2.0 * y0 + yp;
        if denom.abs() > 1e-9 {
            0.5 * (ym - yp) / denom
        } else {
            0.0
        }
    } else {
        0.0
    };
    Some(integer as f64 + refined.clamp(-0.5, 0.5))
}

fn off_axis_angle(north_shift: f64, east_shift: f64) -> f64 {
    let major = north_shift.abs().max(east_shift.abs());
    let minor = north_shift.abs().min(east_shift.abs());
    if major < 0.1 {
        0.0
    } else {
        (minor / major).atan().to_degrees()
    }
}

fn process_scans(
    files: &[(PathBuf, bool)],
    session_dir: &Path,
    tx: &Sender<ProcessMessage>,
) -> Result<AcquireOutput, String> {
    let mut images = Vec::with_capacity(files.len());
    // Mid-scan epochs, so the stacker can compensate solar rotation.
    let mut epochs: Vec<Option<f64>> = Vec::with_capacity(files.len());
    // Acquisition metadata of the first scan, carried onto the stacked product.
    let mut stack_meta: Option<output::FitsMeta> = None;
    for (index, (path, reverse)) in files.iter().enumerate() {
        let _ = tx.send(ProcessMessage::Log(format!(
            "Reconstructing scan {}/{}{}",
            index + 1,
            files.len(),
            if *reverse { " (reverse)" } else { "" }
        )));
        let log_tx = tx.clone();
        let opts = pipeline::ReconOptions {
            flip_x: *reverse,
            verbose: false,
            progress: Some(Arc::new(move |line: &str| {
                let _ = log_tx.send(ProcessMessage::Log(line.to_owned()));
            })),
            ..Default::default()
        };
        let report = pipeline::reconstruct(path, &opts)?;
        let meta = report
            .timing
            .as_ref()
            .map(|t| t.fits_meta())
            .unwrap_or_default();
        // Keep the first scan's acquisition time for the stacked product.
        if stack_meta.is_none() {
            stack_meta = Some(meta.clone());
        }
        epochs.push(
            report
                .timing
                .as_ref()
                .map(|t| ghostsun_core::ser::ticks_to_iso8601(t.mid_utc_ticks))
                .and_then(|iso| ghostsun_core::rotation::jd_from_iso8601(&iso)),
        );
        let image = report.output.image;
        let fits = session_dir.join(format!("reconstruction-{:02}.fits", index + 1));
        let png = session_dir.join(format!("reconstruction-{:02}.png", index + 1));
        output::write_fits_f32_meta(&fits, &image, &meta)
            .map_err(|e| format!("write {}: {e}", fits.display()))?;
        output::write_png16(&png, &image, None)
            .map_err(|e| format!("write {}: {e}", png.display()))?;
        images.push(image);
    }

    let final_image = if images.len() == 1 {
        images.remove(0)
    } else {
        let _ = tx.send(ProcessMessage::Log(format!(
            "Registering and robust-stacking {} reconstructions",
            images.len()
        )));
        let n_in = images.len();
        // Same entry point the CLI uses, so the two cannot drift: de-rotation
        // to a common epoch (skipped below its deadband, or without epochs),
        // then registration, then Wiener filtering of the combined result.
        let scans: Vec<stack::StackInput> = images
            .drain(..)
            .zip(epochs.iter().copied())
            .map(|(image, jd)| stack::StackInput { image, jd })
            .collect();
        let missing = scans.iter().filter(|s| s.jd.is_none()).count();
        if missing > 0 {
            let _ = tx.send(ProcessMessage::Log(format!(
                "{missing} scan(s) had no acquisition time, so solar rotation was not compensated"
            )));
        }
        let sopts = stack::StackOptions {
            flow: true,
            derotate: true,
            wiener: Some(pipeline::TuneParams::default().wiener_strength),
            verbose: false,
        };
        let srep = stack::stack_scans(scans, &sopts)
            .ok_or("multi-scan registration/stacking failed")?;
        if srep.n_flipped > 0 {
            // Recovered, but never routine: the stacker had to deduce an
            // orientation this code claims to know.
            let _ = tx.send(ProcessMessage::Log(format!(
                "WARNING: {} scan(s) arrived mirrored and were auto-flipped; direction bookkeeping is wrong",
                srep.n_flipped
            )));
        }
        if srep.n_used < n_in {
            // A dropped scan is survivable but never routine -- it means a
            // reconstruction failed to fit or refused to correlate with the
            // reference (wrong direction/flip bookkeeping reads exactly so).
            let _ = tx.send(ProcessMessage::Log(format!(
                "WARNING: only {}/{} scans survived registration — check scan                  direction bookkeeping",
                srep.n_used, n_in
            )));
        }
        srep.image
    };
    let stem = if files.len() == 1 {
        "ghostsun-final"
    } else {
        "ghostsun-stacked"
    };
    let fits = session_dir.join(format!("{stem}.fits"));
    let png = session_dir.join(format!("{stem}.png"));
    output::write_fits_f32_meta(&fits, &final_image, &stack_meta.unwrap_or_default())
        .map_err(|e| format!("write {}: {e}", fits.display()))?;
    output::write_png16(&png, &final_image, None)
        .map_err(|e| format!("write {}: {e}", png.display()))?;
    Ok(AcquireOutput {
        image: final_image,
        name: if files.len() == 1 {
            "acquired reconstruction".into()
        } else {
            format!("stacked acquisition ({} scans)", files.len())
        },
        source_ser: files[0].0.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_correlation_recovers_shift() {
        let reference: Vec<f32> = (0..256)
            .map(|x| {
                let x = x as f32;
                (-(x - 73.0).powi(2) / 80.0).exp() + 0.6 * (-(x - 174.0).powi(2) / 180.0).exp()
            })
            .collect();
        let mut shifted = vec![0.0; reference.len()];
        shifted[7..].copy_from_slice(&reference[..reference.len() - 7]);
        let measured = profile_shift(&reference, &shifted, 20).unwrap();
        assert!((measured - 7.0).abs() < 0.2, "{measured}");
    }

    #[test]
    fn off_axis_angle_uses_minor_over_major_motion() {
        assert!((off_axis_angle(1.0, 10.0) - 5.7106).abs() < 0.01);
        assert!((off_axis_angle(10.0, 1.0) - 5.7106).abs() < 0.01);
    }

    #[test]
    fn alternating_scans_reverse_direction() {
        assert_eq!(scan_direction(Direction::East, 0), Direction::East);
        assert_eq!(scan_direction(Direction::East, 1), Direction::West);
        assert_eq!(scan_direction(Direction::East, 2), Direction::East);
    }

    #[test]
    fn recommended_scan_is_deliberately_slow() {
        let (_, code, multiple) = ACQUISITION_RATES[DEFAULT_ACQUISITION_RATE];
        assert_eq!(code, 4);
        assert_eq!(multiple, 4.0);
        assert!((scan_duration(0.8, multiple).as_secs_f64() - 48.0).abs() < 0.01);
    }
}
