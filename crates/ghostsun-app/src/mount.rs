//! Cross-platform ZWO AM-series mount control over its USB serial port.
//!
//! ZWO mounts expose an LX200-compatible command set at 9600 8N1. Keeping the
//! protocol here, rather than using Windows ASCOM, gives Windows and macOS the
//! same behaviour and keeps all blocking serial I/O off egui's render thread.

use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use serialport::{
    ClearBuffer, DataBits, FlowControl, Parity, SerialPort, SerialPortType, StopBits,
};

use super::{focus, ACCENT, ACCENT_DIM};

const POLL_INTERVAL: Duration = Duration::from_millis(750);
const SCAN_INTERVAL: Duration = Duration::from_secs(3);
const IO_DEADLINE: Duration = Duration::from_millis(1200);

/// Grid step for the expanding square spiral (matches the UI copy).
const SPIRAL_STEP_DEG: f32 = 0.2;
/// Local refinement step around the strongest coarse-grid sample.
const REFINE_STEP_DEG: f32 = SPIRAL_STEP_DEG / 2.0;
const COARSE_GRID_SCALE: i32 = 2;
/// Worker nudges always use ZWO rate 7 (60× sidereal) — see `WorkerCommand::Nudge`.
const NUDGE_SIDEREAL_MULT: f64 = 60.0;
const SIDEREAL_DEG_PER_S: f64 = 15.0 / 3600.0;
const SETTLE_AFTER_SLEW: Duration = Duration::from_secs(2);
const SAMPLE_FRAMES: usize = 3;
const SAMPLE_TIMEOUT: Duration = Duration::from_secs(8);
const SLEW_TIMEOUT: Duration = Duration::from_secs(180);
const AUTO_CENTER_TIMEOUT: Duration = Duration::from_secs(600);

/// Rate the auto-center spiral always uses (60x sidereal); see the note above.
const AUTO_CENTER_RATE: u8 = 7;
/// A return-to-mark beyond this separation is almost certainly a stale mark
/// rather than a jog to undo, so it asks before moving the telescope.
const RETURN_CONFIRM_DEG: f64 = 5.0;

/// (label, ZWO `:R<n>#` code, multiple of the sidereal rate).
const RATES: [(&str, u8, f64); 10] = [
    ("0.25x", 0, 0.25),
    ("0.5x", 1, 0.5),
    ("1x", 2, 1.0),
    ("2x", 3, 2.0),
    ("4x", 4, 4.0),
    ("8x", 5, 8.0),
    ("20x", 6, 20.0),
    ("60x", 7, 60.0),
    ("720x", 8, 720.0),
    ("1440x", 9, 1440.0),
];

/// Sidereal rate in arcseconds of sky per second of time.
const SIDEREAL_ARCSEC_PER_SEC: f64 = 15.041;

/// Approximate sky distance a timed jog covers, in arcminutes.
///
/// Nominal: the mount's actual rate is firmware-dependent and the declination
/// axis need not match right ascension exactly. It is shown to answer the
/// question the feature exists for — "will this cross the disc?", the Sun being
/// about 32 arcminutes — not as a calibration.
fn jog_arcmin(rate_multiple: f64, seconds: f64) -> f64 {
    rate_multiple * SIDEREAL_ARCSEC_PER_SEC * seconds / 60.0
}

#[derive(Clone)]
struct PortInfo {
    name: String,
    detail: String,
    is_zwo: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfirmedMotion {
    GoHome,
    Park,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoCenterOrigin {
    SunGoTo,
    CurrentPoint,
}

impl AutoCenterOrigin {
    fn label(self) -> &'static str {
        match self {
            Self::SunGoTo => "Sun GoTo + center",
            Self::CurrentPoint => "center from current point",
        }
    }
}

/// A recorded pointing to return to.
#[derive(Clone, Copy, Debug)]
struct Mark {
    ra_hours: f64,
    dec_deg: f64,
}

/// A timed jog in flight. The worker owns the deadline and reports completion,
/// so this only carries what the UI needs to show progress.
#[derive(Clone, Copy, Debug)]
struct TimedJog {
    direction: Direction,
    started: Instant,
    duration: Duration,
}

impl Direction {
    pub fn label(self) -> &'static str {
        match self {
            Direction::North => "N",
            Direction::South => "S",
            Direction::East => "E",
            Direction::West => "W",
        }
    }

    pub fn opposite(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::South => Self::North,
            Self::East => Self::West,
            Self::West => Self::East,
        }
    }
}

impl Direction {
    fn move_command(self) -> &'static str {
        match self {
            Direction::North => ":Mn#",
            Direction::South => ":Ms#",
            Direction::East => ":Me#",
            Direction::West => ":Mw#",
        }
    }

    fn stop_command(self) -> &'static str {
        match self {
            Direction::North => ":Qn#",
            Direction::South => ":Qs#",
            Direction::East => ":Qe#",
            Direction::West => ":Qw#",
        }
    }
}

#[derive(Default, Clone)]
struct MountSnapshot {
    ra: Option<String>,
    dec: Option<String>,
    altitude: Option<String>,
    azimuth: Option<String>,
    tracking: Option<String>,
    slewing: Option<bool>,
    home: Option<String>,
    park: Option<String>,
    flags: Option<String>,
}

enum WorkerCommand {
    Connect(String),
    Disconnect,
    Poll,
    SetRate(u8),
    Jog(Direction),
    Stop,
    /// Stop acquisition-owned motion without disabling RA tracking.
    StopAcquisition,
    GoHome,
    Park,
    Unpark,
    /// `true` enables tracking (`:Te#`), `false` disables (`:Td#`).
    SetTracking(bool),
    /// Push host clock + observing site so GoTo is not rejected with e7.
    SyncSiteTime {
        latitude_deg: f64,
        longitude_deg: f64,
        utc_offset_hours: f64,
    },
    Nudge {
        direction: Direction,
        duration_ms: u64,
        /// ZWO rate index (`:R<n>#`). The auto-center spiral always uses 7
        /// (60×); a user timed jog uses whatever the panel has selected.
        rate: u8,
        /// Re-enable tracking after the axis-specific stop. Acquisition sets
        /// this because a stopped RA drive lets the Sun drift out of the slit.
        ensure_tracking: bool,
    },
    SlewSun {
        ra_hours: f64,
        dec_deg: f64,
    },
    /// Read the current pointing so it can be returned to later.
    ///
    /// Deliberately a round-trip to the mount rather than a copy of the polled
    /// snapshot: the snapshot can be up to `POLL_INTERVAL` stale, and a mark
    /// taken from stale data is a mark that returns to the wrong place.
    MarkPosition,
    /// GoTo arbitrary coordinates, leaving the tracking rate untouched.
    SlewTo {
        ra_hours: f64,
        dec_deg: f64,
    },
    /// Select the SOLAR tracking rate (`:TS#`).
    ///
    /// Asserted before a scan rather than assumed: the rate was only ever set
    /// on the Sun-GoTo path, so centring the Sun any other way left the mount
    /// on whatever rate it powered up with.
    SetSolarRate,
    Shutdown,
}

enum WorkerMessage {
    Connected { port: String, model: String },
    Disconnected(String),
    Snapshot(MountSnapshot),
    Notice(String),
    NudgeDone,
    Marked { ra_hours: f64, dec_deg: f64 },
    Error(String),
}

enum AutoCenterPhase {
    AwaitingSlew {
        started: Instant,
        saw_motion: bool,
    },
    Settling {
        until: Instant,
    },
    Sampling {
        last_seq: u64,
        samples: Vec<f32>,
        deadline: Instant,
    },
    Moving {
        target: (i32, i32),
        sample_index: Option<usize>,
    },
    ReturnReady,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoCenterPass {
    Coarse,
    Refine,
}

struct AutoCenterState {
    restore: focus::SearchCameraRestore,
    origin: AutoCenterOrigin,
    pass: AutoCenterPass,
    points: Vec<(i32, i32)>,
    point_index: usize,
    current: (i32, i32),
    best: (i32, i32),
    best_signal: f32,
    max_units: i32,
    duration_ms: u64,
    overall_deadline: Instant,
    phase: AutoCenterPhase,
}

/// One grid step duration at the worker's fixed 60× nudge rate.
fn spiral_nudge_duration_ms() -> u64 {
    let deg_per_s = NUDGE_SIDEREAL_MULT * SIDEREAL_DEG_PER_S;
    ((f64::from(REFINE_STEP_DEG) / deg_per_s) * 1000.0).round() as u64
}

/// Expanding square spiral in grid units (Chebyshev radius `max_r`).
fn square_spiral(max_r: i32) -> Vec<(i32, i32)> {
    let mut out = vec![(0, 0)];
    for r in 1..=max_r {
        // East edge south→north: (r, 1-r) .. (r, r)
        for y in (1 - r)..=r {
            out.push((r, y));
        }
        // North edge east→west: (r-1, r) .. (-r, r)
        for k in 0..(2 * r) {
            out.push((r - 1 - k, r));
        }
        // West edge north→south: (-r, r-1) .. (-r, -r)
        for k in 0..(2 * r) {
            out.push((-r, r - 1 - k));
        }
        // South edge west→east: (1-r, -r) .. (r, -r)
        for x in (1 - r)..=r {
            out.push((x, -r));
        }
    }
    out
}

/// Coarse 0.2-degree scan represented in 0.1-degree fine-grid units.
fn coarse_spiral(max_r: i32) -> Vec<(i32, i32)> {
    square_spiral(max_r)
        .into_iter()
        .map(|(x, y)| (x * COARSE_GRID_SCALE, y * COARSE_GRID_SCALE))
        .collect()
}

/// Bounded 3x3 local search around the strongest coarse-grid sample.
fn refinement_grid(center: (i32, i32), max_units: i32) -> Vec<(i32, i32)> {
    square_spiral(1)
        .into_iter()
        .map(|(x, y)| (center.0 + x, center.1 + y))
        .filter(|(x, y)| x.abs() <= max_units && y.abs() <= max_units)
        .collect()
}

fn grid_step_direction(from: (i32, i32), to: (i32, i32)) -> Option<(Direction, (i32, i32))> {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    if dx != 0 {
        let step = dx.signum();
        let dir = if step > 0 {
            Direction::East
        } else {
            Direction::West
        };
        Some((dir, (from.0 + step, from.1)))
    } else if dy != 0 {
        let step = dy.signum();
        let dir = if step > 0 {
            Direction::North
        } else {
            Direction::South
        };
        Some((dir, (from.0, from.1 + step)))
    } else {
        None
    }
}

pub struct MountState {
    ports: Vec<PortInfo>,
    port_name: String,
    connected: bool,
    connecting: bool,
    connected_port: Option<String>,
    model: Option<String>,
    snapshot: MountSnapshot,
    rate_index: usize,
    active_direction: Option<Direction>,
    /// Pointing recorded as the origin for timed jogs, in equatorial coords so
    /// it survives tracking: the mount follows the sky, so a marked RA/Dec
    /// stays on the same solar feature while a marked alt/az would not.
    mark: Option<Mark>,
    /// A timed jog awaiting its `NudgeDone`.
    timed_jog: Option<TimedJog>,
    /// Motion owned by the acquisition workflow. Kept separate so its
    /// completion cannot advance a manual timed jog or Sun auto-center.
    acquisition_nudge_inflight: bool,
    acquisition_nudge_done: bool,
    acquisition_error: Option<String>,
    jog_seconds: f64,
    jog_direction: Direction,
    /// Rate for timed jogs, kept separate from the hold-to-jog rate: holding a
    /// button wants something slow and controllable, while crossing the disc in
    /// a few seconds wants something fast.
    jog_rate_index: usize,
    confirm_return: bool,
    /// Direction whose jog button the current press started on.
    ///
    /// egui cannot be asked "is this button still held": a plain `Button`
    /// senses clicks only, so `Response::is_pointer_button_down_on()` is backed
    /// by `potential_click_id`, which egui clears as soon as the press outlives
    /// `max_click_duration` (0.8 s, `interaction.rs`). The press is still
    /// physically down, but the button reports released — and the jog stopped
    /// after about a second, mid-slew. Latching the direction ourselves and
    /// releasing it on the actual pointer-up decouples "held" from egui's
    /// click/drag bookkeeping.
    jog_latch: Option<Direction>,
    confirm_motion: Option<ConfirmedMotion>,
    confirm_sun: bool,
    confirm_auto_center: bool,
    auto_center_origin: AutoCenterOrigin,
    auto_center: Option<AutoCenterState>,
    search_exposure_ms: u32,
    search_radius_deg: f32,
    capture_height: usize,
    capture_anchor_y: Option<f64>,
    /// East-positive degrees [-180, 180].
    site_latitude_deg: f64,
    /// East-positive degrees [-180, 180].
    site_longitude_deg: f64,
    /// Local − UTC in hours (e.g. BST = +1, PST = −8).
    site_utc_offset_hours: f64,
    site_place: String,
    site_search: String,
    site_search_hits: Vec<(String, f64, f64)>,
    site_search_inflight: bool,
    /// Background Nominatim results (never block / panic the UI thread).
    place_search_rx: Receiver<Result<Vec<(String, f64, f64)>, String>>,
    place_search_tx: Sender<Result<Vec<(String, f64, f64)>, String>>,
    auto_sync_site_on_connect: bool,
    /// Modal when connected without a saved observing site (GoTo needs it).
    site_prompt_open: bool,
    status: String,
    last_scan: Instant,
    last_poll: Instant,
    poll_inflight: bool,
    tx: Sender<WorkerCommand>,
    rx: Receiver<WorkerMessage>,
    worker: Option<JoinHandle<()>>,
}

impl Default for MountState {
    fn default() -> Self {
        let (command_tx, command_rx) = channel();
        let (message_tx, message_rx) = channel();
        let (place_tx, place_rx) = channel();
        let worker = thread::Builder::new()
            .name("ghostsun-mount".into())
            .spawn(move || worker_loop(command_rx, message_tx))
            .expect("failed to start mount worker");

        let mut state = Self {
            ports: Vec::new(),
            port_name: String::new(),
            connected: false,
            connecting: false,
            connected_port: None,
            model: None,
            snapshot: MountSnapshot::default(),
            rate_index: 4,
            active_direction: None,
            mark: None,
            timed_jog: None,
            acquisition_nudge_inflight: false,
            acquisition_nudge_done: false,
            acquisition_error: None,
            jog_seconds: 2.0,
            jog_direction: Direction::North,
            // 60x sidereal crosses the solar disc in roughly two seconds.
            jog_rate_index: 7,
            confirm_return: false,
            jog_latch: None,
            confirm_motion: None,
            confirm_sun: false,
            confirm_auto_center: false,
            auto_center_origin: AutoCenterOrigin::SunGoTo,
            auto_center: None,
            search_exposure_ms: 250,
            search_radius_deg: 0.6,
            capture_height: 1024,
            capture_anchor_y: None,
            site_latitude_deg: 0.0,
            site_longitude_deg: 0.0,
            site_utc_offset_hours: system_utc_offset_hours(),
            site_place: String::new(),
            site_search: String::new(),
            site_search_hits: Vec::new(),
            site_search_inflight: false,
            place_search_rx: place_rx,
            place_search_tx: place_tx,
            auto_sync_site_on_connect: true,
            site_prompt_open: false,
            status: "Not connected".into(),
            last_scan: Instant::now() - SCAN_INTERVAL,
            last_poll: Instant::now(),
            poll_inflight: false,
            tx: command_tx,
            rx: message_rx,
            worker: Some(worker),
        };
        state.load_site_settings();
        state.refresh_ports();
        state
    }
}

impl MountState {
    pub fn tracking_is_on(&self) -> bool {
        self.snapshot.tracking.as_deref() == Some("On")
    }

    /// Minutes from the Sun crossing the local meridian: negative before
    /// transit, positive after. Requires a configured observing longitude.
    pub fn sun_meridian_offset_minutes(&self) -> Option<f64> {
        if !self.site_is_configured() {
            return None;
        }
        let unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_secs_f64();
        Some(sun_hour_angle_hours_at_unix(
            unix_seconds,
            self.site_longitude_deg,
        ) * 60.0)
    }

    pub fn refresh_ports(&mut self) {
        self.last_scan = Instant::now();
        match discover_ports() {
            Ok(ports) => {
                self.ports = ports;
                if self.port_name.is_empty() {
                    if let Some(port) = self.ports.first() {
                        self.port_name = port.name.clone();
                    }
                }
                let detected_zwo = self.ports.iter().find(|port| port.is_zwo);
                self.status = if let Some(port) = detected_zwo {
                    format!("Detected ZWO USB mount on {}", port.name)
                } else if self.ports.is_empty() {
                    format!(
                        "No serial ports found using {}",
                        platform_scan_description()
                    )
                } else {
                    format!(
                        "Found {} serial port(s), but none identify as ZWO USB",
                        self.ports.len()
                    )
                };
            }
            Err(error) => {
                self.ports.clear();
                self.status = format!("Port discovery failed: {error}");
            }
        }
    }

    fn site_config_path() -> PathBuf {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".ghostsun")
                .join("mount_site.json");
        }
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata)
                .join("GhostSun")
                .join("mount_site.json");
        }
        PathBuf::from("ghostsun_mount_site.json")
    }

    fn load_site_settings(&mut self) {
        let path = Self::site_config_path();
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return;
        };
        for part in raw.split(&['{', '}', ',', '\n'][..]) {
            let part = part.trim();
            if let Some(v) = part.strip_prefix("\"lat\":") {
                if let Ok(x) = v.trim().parse() {
                    self.site_latitude_deg = x;
                }
            } else if let Some(v) = part.strip_prefix("\"lon\":") {
                if let Ok(x) = v.trim().parse() {
                    self.site_longitude_deg = x;
                }
            } else if let Some(v) = part.strip_prefix("\"utc_offset_hours\":") {
                if let Ok(x) = v.trim().parse() {
                    self.site_utc_offset_hours = x;
                }
            } else if let Some(v) = part.strip_prefix("\"auto_sync\":") {
                self.auto_sync_site_on_connect = v.trim() == "true";
            } else if let Some(v) = part.strip_prefix("\"place\":") {
                let v = v.trim().trim_matches('"');
                self.site_place = v.to_owned();
            }
        }
    }

    fn save_site_settings(&self) {
        let path = Self::site_config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let place = self.site_place.replace('\\', "\\\\").replace('"', "\\\"");
        let body = format!(
            "{{\n  \"lat\": {:.6},\n  \"lon\": {:.6},\n  \"utc_offset_hours\": {:.4},\n  \"auto_sync\": {},\n  \"place\": \"{place}\"\n}}\n",
            self.site_latitude_deg,
            self.site_longitude_deg,
            self.site_utc_offset_hours,
            self.auto_sync_site_on_connect,
        );
        let _ = std::fs::write(path, body);
    }

    fn site_is_configured(&self) -> bool {
        // Treat "never set" as both exactly 0 with empty place — valid Gulf of
        // Guinea sites must set a place name or non-zero coords intentionally.
        !(self.site_latitude_deg == 0.0
            && self.site_longitude_deg == 0.0
            && self.site_place.trim().is_empty())
    }

    fn request_site_time_sync(&mut self) {
        self.save_site_settings();
        match self.tx.send(WorkerCommand::SyncSiteTime {
            latitude_deg: self.site_latitude_deg.clamp(-90.0, 90.0),
            longitude_deg: self.site_longitude_deg.clamp(-180.0, 180.0),
            utc_offset_hours: self.site_utc_offset_hours.clamp(-14.0, 14.0),
        }) {
            Ok(()) => {
                self.status = "Syncing mount time and site coordinates...".into();
            }
            Err(_) => self.status = "Mount worker is not running".into(),
        }
    }

    fn run_place_search(&mut self) {
        let query = self.site_search.trim().to_owned();
        if query.is_empty() {
            self.status = "Enter a place name to search".into();
            return;
        }
        if self.site_search_inflight {
            self.status = "Place search already running…".into();
            return;
        }
        self.site_search_inflight = true;
        self.site_search_hits.clear();
        self.status = format!("Searching for “{query}”…");
        let tx = self.place_search_tx.clone();
        let _ = thread::Builder::new()
            .name("ghostsun-place-search".into())
            .spawn(move || {
                // Isolate panics (e.g. misconfigured TLS) from the UI process.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    nominatim_search(&query)
                }));
                let payload = match result {
                    Ok(inner) => inner,
                    Err(_) => {
                        Err("place search panicked (TLS/network). Enter lat/lon manually.".into())
                    }
                };
                let _ = tx.send(payload);
            });
    }

    fn drain_place_search(&mut self) {
        while let Ok(result) = self.place_search_rx.try_recv() {
            self.site_search_inflight = false;
            match result {
                Ok(hits) => {
                    if hits.is_empty() {
                        self.status = "No places found".into();
                        self.site_search_hits.clear();
                    } else {
                        self.status = format!("Found {} place(s) — pick one", hits.len());
                        self.site_search_hits = hits;
                    }
                }
                Err(error) => {
                    self.status = format!("Place search failed: {error}");
                    self.site_search_hits.clear();
                }
            }
        }
    }

    pub fn enter_tab(&mut self, focus: &mut focus::FocusState) {
        self.refresh_ports();
        if focus.cameras.is_empty() {
            focus.refresh_cameras();
        }
        self.last_poll = Instant::now() - POLL_INTERVAL;
    }

    pub fn leave_tab(&mut self, focus: &mut focus::FocusState) {
        self.cancel_auto_center(focus, "Sun auto-center cancelled");
        if focus.recording {
            focus.stop_ser_recording();
        }
        self.stop_motion();
        self.confirm_motion = None;
        self.confirm_sun = false;
        self.confirm_auto_center = false;
    }

    pub fn poll(&mut self, ctx: &egui::Context, focus: &mut focus::FocusState) {
        self.drain_place_search();
        while let Ok(message) = self.rx.try_recv() {
            match message {
                WorkerMessage::Connected { port, model } => {
                    self.connected = true;
                    self.connecting = false;
                    self.connected_port = Some(port.clone());
                    self.model = Some(model.clone());
                    self.status = format!("Connected to {model} on {port}");
                    self.last_poll = Instant::now() - POLL_INTERVAL;
                    if self.site_is_configured() {
                        if self.auto_sync_site_on_connect {
                            self.request_site_time_sync();
                        }
                        self.site_prompt_open = false;
                    } else {
                        self.site_prompt_open = true;
                        self.status = format!(
                            "Connected to {model} on {port} — set observing site (required for GoTo)"
                        );
                    }
                }
                WorkerMessage::Disconnected(reason) => {
                    self.cancel_auto_center(focus, "Sun auto-center stopped: mount disconnected");
                    if self.acquisition_nudge_inflight {
                        self.acquisition_error = Some("mount disconnected during acquisition motion".into());
                    }
                    self.acquisition_nudge_inflight = false;
                    self.connected = false;
                    self.connecting = false;
                    self.connected_port = None;
                    self.active_direction = None;
                    self.jog_latch = None;
                    self.poll_inflight = false;
                    self.status = reason;
                }
                WorkerMessage::Snapshot(snapshot) => {
                    self.snapshot = snapshot;
                    self.poll_inflight = false;
                }
                WorkerMessage::Notice(notice) => self.status = notice,
                WorkerMessage::NudgeDone => {
                    // The two cannot overlap -- a timed jog is refused while
                    // auto-center runs -- but route explicitly rather than
                    // relying on that invariant holding forever.
                    if self.acquisition_nudge_inflight {
                        self.acquisition_nudge_inflight = false;
                        self.acquisition_nudge_done = true;
                        self.status = "Acquisition motion complete".into();
                    } else if self.timed_jog.take().is_some() {
                        self.status = "Timed jog complete".into();
                    } else {
                        self.auto_center_nudge_done();
                    }
                }
                WorkerMessage::Marked { ra_hours, dec_deg } => {
                    self.mark = Some(Mark { ra_hours, dec_deg });
                    self.status = format!(
                        "Marked {} {}",
                        format_ra(ra_hours),
                        format_dec(dec_deg)
                    );
                }
                WorkerMessage::Error(error) => {
                    self.cancel_auto_center(focus, "Sun auto-center stopped by mount error");
                    if self.acquisition_nudge_inflight {
                        self.acquisition_error = Some(error.clone());
                    }
                    self.acquisition_nudge_inflight = false;
                    self.connecting = false;
                    self.poll_inflight = false;
                    self.status = format!("Mount error: {error}");
                }
            }
        }

        self.advance_auto_center(focus);
        if !self.connected && !self.connecting && self.last_scan.elapsed() >= SCAN_INTERVAL {
            self.refresh_ports();
        }
        if self.connected && !self.poll_inflight && self.last_poll.elapsed() >= POLL_INTERVAL {
            if self.tx.send(WorkerCommand::Poll).is_ok() {
                self.poll_inflight = true;
                self.last_poll = Instant::now();
            }
        }
        if self.connected && !self.site_is_configured() {
            self.site_prompt_open = true;
        }
        self.show_site_prompt_modal(ctx);
        ctx.request_repaint_after(Duration::from_millis(50));
    }

    fn show_site_prompt_modal(&mut self, ctx: &egui::Context) {
        if !self.site_prompt_open {
            return;
        }
        let mut open = true;
        egui::Window::new("Observing site required")
            .id(egui::Id::new("mount_site_prompt"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .order(egui::Order::Foreground)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                ui.label(
                    egui::RichText::new(
                        "The mount rejects GoTo (error e7) until time and location are set. Enter coordinates or search a place, then Save.",
                    )
                    .strong()
                    .color(egui::Color32::from_rgb(255, 190, 100)),
                );
                ui.add_space(8.0);
                self.site_editor_ui(ui, /*compact*/ true);
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            self.site_is_configured(),
                            egui::Button::new(
                                egui::RichText::new("Save & continue")
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(ACCENT_DIM),
                        )
                        .clicked()
                    {
                        if self.site_place.trim().is_empty() {
                            self.site_place = format!(
                                "{:.4},{:.4}",
                                self.site_latitude_deg, self.site_longitude_deg
                            );
                        }
                        self.save_site_settings();
                        self.site_prompt_open = false;
                        if self.connected {
                            self.request_site_time_sync();
                        } else {
                            self.status = "Observing site saved".into();
                        }
                    }
                    if ui.button("Later").clicked() {
                        self.site_prompt_open = false;
                        self.status =
                            "Site not set — GoTo will fail with e7 until you save a location"
                                .into();
                    }
                });
            });
        if !open {
            self.site_prompt_open = false;
        }
    }

    /// Shared lat/lon / search / sync controls for side panel and modal.
    fn site_editor_ui(&mut self, ui: &mut egui::Ui, compact: bool) {
        if !compact {
            ui.label(
                egui::RichText::new(
                    "GoTo fails with e7 until the mount has time + coordinates. GhostSun can push your system clock and this site on connect.",
                )
                .small()
                .weak(),
            );
        }
        if ui
            .checkbox(
                &mut self.auto_sync_site_on_connect,
                "Sync time & site on connect",
            )
            .changed()
        {
            self.save_site_settings();
        }
        ui.horizontal(|ui| {
            ui.label("lat °");
            ui.add(
                egui::DragValue::new(&mut self.site_latitude_deg)
                    .speed(0.01)
                    .range(-90.0..=90.0)
                    .fixed_decimals(5),
            );
            ui.label("lon °E");
            ui.add(
                egui::DragValue::new(&mut self.site_longitude_deg)
                    .speed(0.01)
                    .range(-180.0..=180.0)
                    .fixed_decimals(5),
            );
        });
        ui.horizontal(|ui| {
            ui.label("UTC offset h");
            ui.add(
                egui::DragValue::new(&mut self.site_utc_offset_hours)
                    .speed(0.25)
                    .range(-14.0..=14.0)
                    .fixed_decimals(2),
            );
            if ui.button("From system").clicked() {
                self.site_utc_offset_hours = system_utc_offset_hours();
                self.status = format!(
                    "UTC offset set from system: {:+.2} h",
                    self.site_utc_offset_hours
                );
            }
        });
        if !self.site_place.is_empty() {
            ui.label(
                egui::RichText::new(format!("Place: {}", self.site_place))
                    .small()
                    .weak(),
            );
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.site_search)
                    .hint_text("City or place name…")
                    .desired_width(if compact { 200.0 } else { 140.0 }),
            );
            let search_label = if self.site_search_inflight {
                "Searching…"
            } else {
                "Search"
            };
            if ui
                .add_enabled(!self.site_search_inflight, egui::Button::new(search_label))
                .clicked()
            {
                self.run_place_search();
            }
        });
        if !self.site_search_hits.is_empty() {
            egui::ScrollArea::vertical()
                .id_salt(if compact {
                    "site_hits_modal"
                } else {
                    "site_hits_side"
                })
                .max_height(if compact { 140.0 } else { 100.0 })
                .show(ui, |ui| {
                    let hits = self.site_search_hits.clone();
                    for (name, lat, lon) in hits {
                        if ui
                            .selectable_label(false, &name)
                            .on_hover_text(format!("{lat:.5}, {lon:.5}"))
                            .clicked()
                        {
                            self.site_latitude_deg = lat;
                            self.site_longitude_deg = lon;
                            self.site_place = name;
                            self.site_search_hits.clear();
                            self.save_site_settings();
                            self.status = format!(
                                "Site set to {} ({:.4}°, {:.4}°E)",
                                self.site_place, self.site_latitude_deg, self.site_longitude_deg
                            );
                        }
                    }
                });
        }
        if !compact {
            ui.horizontal(|ui| {
                if ui.button("Save site").clicked() {
                    if self.site_place.trim().is_empty() {
                        self.site_place = format!(
                            "{:.4},{:.4}",
                            self.site_latitude_deg, self.site_longitude_deg
                        );
                    }
                    self.save_site_settings();
                    self.site_prompt_open = false;
                    self.status = "Observing site saved".into();
                }
                if ui
                    .add_enabled(self.connected, egui::Button::new("Sync now"))
                    .on_hover_text(
                        "Push system time + this site to the mount (:SC/:SL/:SG/:St/:Sg)",
                    )
                    .clicked()
                {
                    if !self.site_is_configured() {
                        self.site_prompt_open = true;
                        self.status =
                            "Set latitude/longitude (or search a place) before syncing".into();
                    } else {
                        self.request_site_time_sync();
                    }
                }
                if ui
                    .button("Map…")
                    .on_hover_text("Open this site in OpenStreetMap (browser)")
                    .clicked()
                {
                    let url = format!(
                        "https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=10/{lat}/{lon}",
                        lat = self.site_latitude_deg,
                        lon = self.site_longitude_deg
                    );
                    self.status = match webbrowser::open(&url) {
                        Ok(()) => "Opened OpenStreetMap in browser".into(),
                        Err(error) => format!("Could not open map: {error}"),
                    };
                }
            });
            ui.label(
                egui::RichText::new(
                    "Home position / mechanical alignment still need ASI Mount or a hand controller.",
                )
                .small()
                .color(egui::Color32::from_rgb(255, 190, 100)),
            );
        }
    }

    pub fn controls_ui(&mut self, ui: &mut egui::Ui, focus: &mut focus::FocusState) {
        ui.add_space(8.0);
        ui.heading("Mount connection");
        ui.label(
            egui::RichText::new(format!(
                "Auto-scan: {} - 9600 baud",
                platform_scan_description()
            ))
            .small()
            .weak(),
        );

        let selected = if self.port_name.is_empty() {
            "Select or enter a serial port"
        } else {
            self.port_name.as_str()
        };
        egui::ComboBox::from_id_salt("mount_port")
            .selected_text(selected)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for port in &self.ports {
                    ui.selectable_value(
                        &mut self.port_name,
                        port.name.clone(),
                        format!("{} ({})", port.name, port.detail),
                    );
                }
            });
        ui.add(
            egui::TextEdit::singleline(&mut self.port_name)
                .hint_text(if cfg!(target_os = "macos") {
                    "/dev/cu.usbserial-..."
                } else {
                    "COM3"
                })
                .desired_width(f32::INFINITY),
        );

        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                self.refresh_ports();
            }
            if self.connected {
                if ui.button("Disconnect").clicked() {
                    self.stop_motion();
                    let _ = self.tx.send(WorkerCommand::Disconnect);
                }
            } else if ui
                .add_enabled(
                    !self.connecting && !self.port_name.trim().is_empty(),
                    egui::Button::new(if self.connecting {
                        "Connecting..."
                    } else {
                        "Connect"
                    }),
                )
                .clicked()
            {
                self.connecting = true;
                self.status = format!("Connecting to {}...", self.port_name.trim());
                let _ = self
                    .tx
                    .send(WorkerCommand::Connect(self.port_name.trim().to_owned()));
            }
        });

        let zwo_detected = self.ports.iter().any(|port| port.is_zwo);
        egui::CollapsingHeader::new(if zwo_detected {
            "Connection help"
        } else {
            "Mount not detected - connection help"
        })
        .default_open(!zwo_detected)
        .show(ui, |ui| {
            if !zwo_detected {
                ui.label(
                    egui::RichText::new(
                        "GhostSun cannot currently identify a ZWO USB mount.",
                    )
                    .strong()
                    .color(egui::Color32::from_rgb(255, 190, 100)),
                );
            }
            ui.label(connection_checklist());
            if ui.button(native_software_button_label()).clicked() {
                self.status = match launch_zwo_software() {
                    Ok(message) => message,
                    Err(error) => format!("Could not open ZWO software: {error}"),
                };
            }
            ui.hyperlink_to(
                "ZWO software and drivers",
                "https://www.zwoastro.com/software/",
            );
            ui.label(
                egui::RichText::new(
                    "After checking detection there, disconnect or close the ZWO application so it releases the serial port before connecting in GhostSun.",
                )
                .small()
                .weak(),
            );
        });

        ui.add_space(12.0);
        ui.heading("Mount status");
        if let Some(model) = &self.model {
            ui.label(format!("Model: {model}"));
        }
        if let Some(port) = &self.connected_port {
            ui.label(format!("Port: {port}"));
        }
        if let Some(flags) = &self.snapshot.flags {
            ui.label(format!("Flags: {flags}"));
        }
        ui.separator();
        ui.add(egui::Label::new(egui::RichText::new(&self.status).small()).wrap());

        ui.add_space(12.0);
        ui.horizontal(|ui| {
            ui.heading("Observing site");
            if !self.site_is_configured() {
                ui.label(
                    egui::RichText::new("required")
                        .small()
                        .strong()
                        .color(egui::Color32::from_rgb(255, 160, 80)),
                );
            }
        });
        if !self.site_is_configured() {
            ui.label(
                egui::RichText::new("No site saved — search a place or enter lat/lon.")
                    .small()
                    .color(egui::Color32::from_rgb(255, 190, 100)),
            );
            if ui.button("Set site now…").clicked() {
                self.site_prompt_open = true;
            }
        }
        self.site_editor_ui(ui, /*compact*/ false);

        ui.add_space(12.0);
        ui.heading("Auto-center camera");
        ui.horizontal(|ui| {
            if ui.button("Scan cameras").clicked() {
                focus.refresh_cameras();
            }
            ui.label(egui::RichText::new(&focus.status).small().weak());
        });
        let hardware: Vec<(usize, String)> = focus
            .cameras
            .iter()
            .enumerate()
            .filter(|(_, camera)| camera.backend != ghostsun_camera::Backend::Synth)
            .map(|(index, camera)| {
                (
                    index,
                    format!("{} - {}", camera.backend.label(), camera.name),
                )
            })
            .collect();
        let selected_camera = hardware
            .iter()
            .find(|(index, _)| *index == focus.selected)
            .map(|(_, name)| name.as_str())
            .unwrap_or("No hardware camera");
        egui::ComboBox::from_id_salt("sun_search_camera")
            .selected_text(selected_camera)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (index, name) in &hardware {
                    ui.selectable_value(&mut focus.selected, *index, name);
                }
            });
        ui.add_enabled_ui(self.auto_center.is_none(), |ui| {
            ui.label("search exposure");
            ui.add(
                egui::Slider::new(&mut self.search_exposure_ms, 50..=1000)
                    .suffix(" ms")
                    .logarithmic(true),
            );
            ui.label("maximum search radius");
            ui.add(
                egui::Slider::new(&mut self.search_radius_deg, 0.2..=1.2)
                    .suffix("°")
                    .step_by(0.2),
            );
        });
        ui.label(
            egui::RichText::new(
                "Uses a 0.2° coarse spiral at 60×, then a bounded 0.1° refinement around the strongest robust camera peak.",
            )
            .small()
            .weak(),
        );

        ui.add_space(16.0);
    }

    pub fn view_ui(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
    ) {
        // Central panel has no outer scroll — pack controls tightly and scroll
        // so Home / Park / Track / Sun GoTo never fall off the bottom.
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.view_ui_inner(ui, ctx, focus);
            });
    }

    fn view_ui_inner(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
    ) {
        ui.vertical_centered(|ui| {
            ui.add_space(14.0);
            ui.heading(egui::RichText::new("Mount control").size(26.0));
            ui.label(
                egui::RichText::new(if self.connected {
                    if self.site_is_configured() {
                        "Connected - hold a direction to jog; release to stop"
                    } else {
                        "Connected — set observing site before GoTo (prompt open or left panel)"
                    }
                } else {
                    "Connect to the ZWO mount in the left panel to enable controls"
                })
                .weak(),
            );
        });

        ui.add_space(12.0);
        ui.horizontal_wrapped(|ui| {
            ui.heading("Live auto-center camera");
            let camera_name = focus
                .cameras
                .get(focus.selected)
                .map(|camera| format!("{} · {}", camera.backend.label(), camera.name))
                .unwrap_or_else(|| "no camera selected".into());
            ui.label(egui::RichText::new(camera_name).small().weak());
            if focus.streaming {
                if ui
                    .add_enabled(
                        self.auto_center.is_none() && !focus.recording,
                        egui::Button::new("Stop camera"),
                    )
                    .clicked()
                {
                    focus.stop();
                }
            } else if ui
                .add_enabled(
                    !focus.cameras.is_empty(),
                    egui::Button::new("Start camera preview"),
                )
                .clicked()
            {
                focus.start(ctx);
            }
        });
        if focus.streaming {
            let peak = focus
                .sun_signal_sample()
                .map(|(_, peak)| format!("{peak:.0}"))
                .unwrap_or_else(|| "waiting".into());
            ui.label(
                egui::RichText::new(format!(
                    "Exposure {:.0} ms · gain {} · robust peak {peak}",
                    focus.exposure_us as f64 / 1000.0,
                    focus.gain
                ))
                .small()
                .weak(),
            );
        }
        focus.camera_preview_ui(ui, 240.0);
        self.ser_acquisition_ui(ui, ctx, focus);

        // Entire mount control surface is inert until Connect succeeds.
        let enabled = self.connected;
        ui.add_enabled_ui(enabled, |ui| {
            self.view_ui_connected(ui, ctx, focus);
        });
        if !enabled {
            ui.add_space(24.0);
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new("Mount controls locked until connected")
                        .size(16.0)
                        .weak(),
                );
            });
        }
    }

    fn view_ui_connected(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
    ) {
        ui.add_space(16.0);
        ui.horizontal_wrapped(|ui| {
            status_card(
                ui,
                "Right ascension",
                self.snapshot.ra.as_deref().unwrap_or("--"),
            );
            status_card(
                ui,
                "Declination",
                self.snapshot.dec.as_deref().unwrap_or("--"),
            );
            status_card(
                ui,
                "Altitude",
                self.snapshot.altitude.as_deref().unwrap_or("--"),
            );
            status_card(
                ui,
                "Azimuth",
                self.snapshot.azimuth.as_deref().unwrap_or("--"),
            );
            status_card(
                ui,
                "Tracking",
                self.snapshot.tracking.as_deref().unwrap_or("--"),
            );
            status_card(ui, "Home", self.snapshot.home.as_deref().unwrap_or("--"));
            status_card(ui, "Park", self.snapshot.park.as_deref().unwrap_or("--"));
        });
        if let Some(minutes) = self.sun_meridian_offset_minutes() {
            if minutes.abs() <= 30.0 {
                let message = if minutes < 0.0 {
                    format!(
                        "MERIDIAN WARNING · Sun transits in {:.0} min. Finish or defer automated scans and supervise the flip.",
                        -minutes
                    )
                } else {
                    format!(
                        "MERIDIAN WARNING · Sun transited {:.0} min ago. Confirm the mount has flipped and tracking is On before scanning.",
                        minutes
                    )
                };
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(92, 52, 8))
                    .stroke(egui::Stroke::new(2.0_f32, egui::Color32::YELLOW))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(message)
                                .strong()
                                .color(egui::Color32::YELLOW),
                        );
                    });
            }
        }

        ui.add_space(10.0);
        let enabled = self.connected;
        let jog_enabled = enabled && self.auto_center.is_none();
        // Seed from the latch so a held button survives egui dropping its
        // click interaction after 0.8 s; the buttons below only ever re-assert
        // it. Released at the bottom of this block on the real pointer-up.
        let mut held = self.jog_latch;
        // Losing window focus is treated as a release: a mouse-up delivered to
        // another application would otherwise leave the mount slewing.
        let pointer_held = ui.input(|i| i.pointer.primary_down() && i.focused);
        let wide_controls = ui.available_width() >= 700.0;

        if wide_controls {
            ui.columns(2, |columns| {
                let ui = &mut columns[0];
                // D-pad geometry: N/S share the same horizontal centre as STOP.
                // Zero egui item_spacing so only our `gap` separates W–STOP–E.
                {
            let btn = egui::vec2(58.0, 34.0);
            let stop_sz = egui::vec2(78.0, 38.0);
            let gap = 5.0_f32;
            let mid_w = btn.x + gap + stop_sz.x + gap + btn.x;
            let mid_h = btn.y.max(stop_sz.y);
            // STOP starts after W + gap; its centre is the N/S alignment axis.
            let stop_left = btn.x + gap;
            let ns_left = stop_left + stop_sz.x * 0.5 - btn.x * 0.5;
            let pad_h = btn.y + gap + mid_h + gap + btn.y;

            ui.horizontal(|ui| {
                let indent = ((ui.available_width() - mid_w) * 0.5).max(0.0);
                ui.add_space(indent);
                ui.allocate_ui_with_layout(
                    egui::vec2(mid_w, pad_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.spacing_mut().item_spacing.y = 0.0;

                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            ui.add_space(ns_left);
                            let north = ui.add_enabled(
                                jog_enabled,
                                egui::Button::new(egui::RichText::new("N").size(20.0).strong())
                                    .min_size(btn),
                            );
                            if north.is_pointer_button_down_on() {
                                held = Some(Direction::North);
                            }
                        });
                        ui.add_space(gap);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            let west = ui.add_enabled(
                                jog_enabled,
                                egui::Button::new(egui::RichText::new("W").size(20.0).strong())
                                    .min_size(btn),
                            );
                            if west.is_pointer_button_down_on() {
                                held = Some(Direction::West);
                            }
                            ui.add_space(gap);
                            let stop = egui::Button::new(
                                egui::RichText::new("STOP")
                                    .size(18.0)
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(160, 35, 25))
                            .min_size(stop_sz);
                            if ui.add(stop).clicked() {
                                self.cancel_auto_center(focus, "Sun auto-center stopped");
                                self.stop_motion();
                                held = None;
                            }
                            ui.add_space(gap);
                            let east = ui.add_enabled(
                                jog_enabled,
                                egui::Button::new(egui::RichText::new("E").size(20.0).strong())
                                    .min_size(btn),
                            );
                            if east.is_pointer_button_down_on() {
                                held = Some(Direction::East);
                            }
                        });
                        ui.add_space(gap);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            ui.add_space(ns_left);
                            let south = ui.add_enabled(
                                jog_enabled,
                                egui::Button::new(egui::RichText::new("S").size(20.0).strong())
                                    .min_size(btn),
                            );
                            if south.is_pointer_button_down_on() {
                                held = Some(Direction::South);
                            }
                        });
                    },
                );
            });

            // Jog rate lives in the main window (not the left rail).
            ui.add_space(4.0);
            ui.vertical_centered(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Jog rate").strong());
                    let old_rate = self.rate_index;
                    egui::ComboBox::from_id_salt("mount_rate_main")
                        .selected_text(RATES[self.rate_index].0)
                        .width(100.0)
                        .show_ui(ui, |ui| {
                            for (index, (label, _, _)) in RATES.iter().enumerate() {
                                ui.selectable_value(&mut self.rate_index, index, *label);
                            }
                        });
                    if self.rate_index != old_rate && self.connected {
                        self.stop_motion();
                        let _ = self
                            .tx
                            .send(WorkerCommand::SetRate(RATES[self.rate_index].1));
                    }
                });
                ui.label(
                    egui::RichText::new("Rate changes stop any active jog first.")
                        .small()
                        .weak(),
                );
            });
                }
                self.timed_jog_ui(&mut columns[1]);
            });
        } else {
            self.manual_jog_ui(ui, focus, jog_enabled, &mut held);
            self.timed_jog_ui(ui);
        }
        if !pointer_held {
            held = None;
        }
        self.jog_latch = held;
        self.update_held_direction(if jog_enabled { held } else { None });

        ui.add_space(8.0);
        let tracking_on = self.snapshot.tracking.as_deref() == Some("On");
        let equatorial = self
            .snapshot
            .flags
            .as_ref()
            .map(|flags| !flags.contains('Z'))
            .unwrap_or(true);
        let actions_ok = enabled && self.auto_center.is_none();

        // Always-clickable buttons with explicit status feedback. A fixed-height
        // allocate_ui previously clipped hit targets; disabled buttons gave no
        // feedback when the mount was only *detected* (not Connected).
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                let btn = |label: &str| {
                    egui::Button::new(egui::RichText::new(label).size(15.0).strong())
                        .min_size(egui::vec2(100.0, 40.0))
                };

                if ui
                    .add(btn("Home"))
                    .on_hover_text("Go to mechanical home (asks for confirmation)")
                    .clicked()
                {
                    if !enabled {
                        self.status = "Connect the mount before using Home".into();
                    } else if self.auto_center.is_some() {
                        self.status = "Cancel auto-center before using Home".into();
                    } else {
                        self.confirm_sun = false;
                        self.confirm_auto_center = false;
                        self.confirm_motion = Some(ConfirmedMotion::GoHome);
                        self.status = "Home: confirm the movement in the panel below".into();
                    }
                }

                ui.add_space(8.0);
                if ui
                    .add(btn("Park"))
                    .on_hover_text("Park (equatorial mode; asks for confirmation)")
                    .clicked()
                {
                    if !enabled {
                        self.status = "Connect the mount before using Park".into();
                    } else if self.auto_center.is_some() {
                        self.status = "Cancel auto-center before using Park".into();
                    } else if !equatorial {
                        self.status =
                            "Park is only available in equatorial mode (not alt-az)".into();
                    } else {
                        self.confirm_sun = false;
                        self.confirm_auto_center = false;
                        self.confirm_motion = Some(ConfirmedMotion::Park);
                        self.status = "Park: confirm the movement in the panel below".into();
                    }
                }

                ui.add_space(8.0);
                if ui
                    .add(btn("Unpark"))
                    .on_hover_text("Unpark the mount")
                    .clicked()
                {
                    if !enabled {
                        self.status = "Connect the mount before using Unpark".into();
                    } else if self.auto_center.is_some() {
                        self.status = "Cancel auto-center before using Unpark".into();
                    } else {
                        self.stop_motion();
                        match self.tx.send(WorkerCommand::Unpark) {
                            Ok(()) => self.status = "Requesting unpark...".into(),
                            Err(_) => self.status = "Mount worker is not running".into(),
                        }
                    }
                }

                ui.add_space(8.0);
                let track_label = if tracking_on {
                    "Track: On"
                } else {
                    "Track: Off"
                };
                let mut track = btn(track_label);
                if tracking_on {
                    track = track.fill(ACCENT_DIM);
                }
                if ui
                    .add(track)
                    .on_hover_text(if tracking_on {
                        "Stop tracking (:Td#)"
                    } else {
                        "Start tracking (:Te#)"
                    })
                    .clicked()
                {
                    if !actions_ok {
                        self.status = if !enabled {
                            "Connect the mount before using Track".into()
                        } else {
                            "Cancel auto-center before using Track".into()
                        };
                    } else {
                        let enable = !tracking_on;
                        match self.tx.send(WorkerCommand::SetTracking(enable)) {
                            Ok(()) => {
                                self.status = if enable {
                                    "Enabling tracking...".into()
                                } else {
                                    "Disabling tracking...".into()
                                };
                                self.last_poll = Instant::now() - POLL_INTERVAL;
                            }
                            Err(_) => self.status = "Mount worker is not running".into(),
                        }
                    }
                }
            });

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(&self.status)
                    .small()
                    .color(egui::Color32::from_rgb(220, 200, 160)),
            );
        });

        if let Some(action) = self.confirm_motion {
            let (title, detail, command) = match action {
                ConfirmedMotion::GoHome => (
                    "Confirm Home",
                    "The mount will move both axes to its mechanical home position.",
                    WorkerCommand::GoHome,
                ),
                ConfirmedMotion::Park => (
                    "Confirm Park",
                    "The mount will move to its configured park position (equatorial mode only).",
                    WorkerCommand::Park,
                ),
            };
            ui.add_space(12.0);
            ui.vertical_centered(|ui| {
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(70, 32, 18))
                    .stroke(egui::Stroke::new(
                        1.5_f32,
                        egui::Color32::from_rgb(255, 160, 80),
                    ))
                    .show(ui, |ui| {
                        ui.set_min_width(380.0);
                        ui.set_max_width(480.0);
                        ui.label(
                            egui::RichText::new(detail)
                                .strong()
                                .color(egui::Color32::from_rgb(255, 210, 160)),
                        );
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            let confirm = egui::Button::new(
                                egui::RichText::new(title)
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(ACCENT_DIM)
                            .min_size(egui::vec2(140.0, 36.0));
                            if ui.add(confirm).clicked() {
                                if !self.connected {
                                    self.status = "Connect the mount before confirming".into();
                                    self.confirm_motion = None;
                                } else {
                                    self.stop_motion();
                                    match self.tx.send(command) {
                                        Ok(()) => {
                                            self.status = format!("{title} sent to mount...");
                                            self.confirm_motion = None;
                                            self.last_poll = Instant::now() - POLL_INTERVAL;
                                        }
                                        Err(_) => {
                                            self.status = "Mount worker is not running".into();
                                        }
                                    }
                                }
                            }
                            if ui
                                .add(egui::Button::new("Cancel").min_size(egui::vec2(90.0, 36.0)))
                                .clicked()
                            {
                                self.confirm_motion = None;
                                self.status = "Home/Park cancelled".into();
                            }
                        });
                    });
            });
        }

        ui.add_space(18.0);
        ui.separator();
        ui.add_space(12.0);
        ui.vertical_centered(|ui| {
            let sun = sun_equatorial_now();
            ui.heading("Slew to the Sun");
            ui.label(format!(
                "Current target: RA {}  Dec {}",
                format_ra(sun.ra_hours),
                format_dec(sun.dec_deg)
            ));
            ui.label(
                egui::RichText::new(
                    "DANGER: Use a securely mounted, undamaged solar filter and verify a clear slew path.",
                )
                .strong()
                .color(egui::Color32::from_rgb(255, 170, 80)),
            );

            if self.confirm_sun {
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(50, 25, 16))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(
                                "Confirm only if the mount time/location/alignment are correct and the telescope is safe for solar use.",
                            )
                            .strong(),
                        );
                        ui.horizontal(|ui| {
                            let confirm = egui::Button::new(
                                egui::RichText::new("Confirm slew to Sun")
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(ACCENT_DIM);
                            if ui.add_enabled(enabled, confirm).clicked() {
                                self.stop_motion();
                                let _ = self.tx.send(WorkerCommand::SlewSun {
                                    ra_hours: sun.ra_hours,
                                    dec_deg: sun.dec_deg,
                                });
                                self.confirm_sun = false;
                                self.status = "Sending Sun coordinates to mount...".into();
                            }
                            if ui.button("Cancel").clicked() {
                                self.confirm_sun = false;
                            }
                        });
                    });
            } else if ui
                .add_enabled(
                    enabled && self.auto_center.is_none(),
                    egui::Button::new(
                        egui::RichText::new("Prepare Sun GoTo")
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(ACCENT_DIM),
                )
                .clicked()
            {
                self.confirm_auto_center = false;
                self.confirm_sun = true;
            }

            ui.add_space(16.0);
            ui.heading("Camera-assisted Sun center");
            ui.label(
                egui::RichText::new(
                    "Choose whether to slew to the calculated Sun first or search around the current pointing. A 0.2° coarse spiral is refined to 0.1° around the strongest robust peak.",
                )
                .small()
                .weak(),
            );
            if let Some(progress) = self.auto_center_progress_label() {
                ui.label(
                    egui::RichText::new(progress)
                        .strong()
                        .color(ACCENT),
                );
                if ui.button("Cancel auto-center").clicked() {
                    self.cancel_auto_center(focus, "Sun auto-center cancelled");
                }
            } else if self.confirm_auto_center {
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(50, 25, 16))
                    .show(ui, |ui| {
                        let warning = match self.auto_center_origin {
                            AutoCenterOrigin::SunGoTo => {
                                "DANGER: The mount will first slew to the Sun, then scan. Confirm only with a solar filter fitted, a clear slew path, and mount time/location/alignment set."
                            }
                            AutoCenterOrigin::CurrentPoint => {
                                "DANGER: The mount will scan around its current pointing. Confirm that a solar filter is fitted and the full search radius is safe."
                            }
                        };
                        ui.label(
                            egui::RichText::new(warning)
                                .strong()
                                .color(egui::Color32::from_rgb(255, 170, 80)),
                        );
                        ui.horizontal(|ui| {
                            let confirm = egui::Button::new(
                                egui::RichText::new(format!(
                                    "Confirm {}",
                                    self.auto_center_origin.label()
                                ))
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(ACCENT_DIM);
                            if ui.add_enabled(enabled, confirm).clicked() {
                                self.confirm_auto_center = false;
                                self.begin_auto_center(ctx, focus, self.auto_center_origin);
                            }
                            if ui.button("Cancel").clicked() {
                                self.confirm_auto_center = false;
                            }
                        });
                    });
            } else {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            enabled && self.auto_center.is_none(),
                            egui::Button::new(
                                egui::RichText::new("Prepare GoTo + center")
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(ACCENT_DIM),
                        )
                        .clicked()
                    {
                        self.confirm_sun = false;
                        self.auto_center_origin = AutoCenterOrigin::SunGoTo;
                        self.confirm_auto_center = true;
                    }
                    if ui
                        .add_enabled(
                            enabled && self.auto_center.is_none(),
                            egui::Button::new(
                                egui::RichText::new("Prepare center from here")
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(ACCENT_DIM),
                        )
                        .clicked()
                    {
                        self.confirm_sun = false;
                        self.auto_center_origin = AutoCenterOrigin::CurrentPoint;
                        self.confirm_auto_center = true;
                    }
                });
            }
        });
    }

    fn update_held_direction(&mut self, held: Option<Direction>) {
        if held == self.active_direction {
            return;
        }
        if self.active_direction.is_some() {
            let _ = self.tx.send(WorkerCommand::Stop);
        }
        if let Some(direction) = held {
            let _ = self
                .tx
                .send(WorkerCommand::SetRate(RATES[self.rate_index].1));
            let _ = self.tx.send(WorkerCommand::Jog(direction));
        }
        self.active_direction = held;
    }

    fn stop_motion(&mut self) {
        let _ = self.tx.send(WorkerCommand::Stop);
        self.active_direction = None;
        self.jog_latch = None;
        self.timed_jog = None;
        self.acquisition_nudge_inflight = false;
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Start one bounded move owned by the acquisition workflow.
    ///
    /// The rate is a ZWO rate index (0..=9), matching the mount protocol.
    pub fn start_acquisition_nudge(
        &mut self,
        direction: Direction,
        duration: Duration,
        rate: u8,
    ) -> Result<(), String> {
        if !self.connected {
            return Err("connect the mount first".into());
        }
        if self.auto_center.is_some()
            || self.timed_jog.is_some()
            || self.active_direction.is_some()
            || self.acquisition_nudge_inflight
        {
            return Err("another mount motion is already active".into());
        }
        let duration_ms = duration.as_millis().clamp(1, u64::MAX as u128) as u64;
        self.acquisition_nudge_done = false;
        self.acquisition_error = None;
        self.tx
            .send(WorkerCommand::Nudge {
                direction,
                duration_ms,
                rate: rate.min(9),
                ensure_tracking: true,
            })
            .map_err(|_| "mount worker is not running".to_owned())?;
        self.acquisition_nudge_inflight = true;
        self.status = format!(
            "Acquisition motion: {} for {:.2} s",
            direction.label(),
            duration.as_secs_f64()
        );
        Ok(())
    }

    /// Assert the solar tracking rate on the mount.
    ///
    /// Fire-and-forget: the LX200 dialect has no reliable "read the tracking
    /// rate" query, so the rate is SET before each run instead of checked.
    pub fn request_solar_tracking_rate(&mut self) {
        let _ = self.tx.send(WorkerCommand::SetSolarRate);
    }

    pub fn take_acquisition_nudge_done(&mut self) -> bool {
        std::mem::take(&mut self.acquisition_nudge_done)
    }

    pub fn take_acquisition_error(&mut self) -> Option<String> {
        self.acquisition_error.take()
    }

    pub fn stop_acquisition_motion(&mut self) {
        let _ = self.tx.send(WorkerCommand::StopAcquisition);
        self.active_direction = None;
        self.jog_latch = None;
        self.timed_jog = None;
        self.acquisition_nudge_inflight = false;
        self.acquisition_nudge_done = false;
    }

    // -- timed jog / return to mark ----------------------------------------

    /// Run one jog of a fixed duration at the panel's selected rate.
    ///
    /// The origin is marked first when nothing is marked yet, so "jog, look,
    /// come back" works without the user having to remember a setup step. The
    /// worker processes commands in order and `MarkPosition` is a synchronous
    /// round-trip, so the mark is always read before the mount starts moving.
    fn start_timed_jog(&mut self) {
        if !self.connected {
            self.status = "Connect the mount first".into();
            return;
        }
        if self.auto_center.is_some() {
            self.status = "Sun auto-center is running; stop it before jogging".into();
            return;
        }
        if self.timed_jog.is_some() {
            return;
        }
        let seconds = self.jog_seconds.clamp(0.1, 120.0);
        let duration = Duration::from_millis((seconds * 1000.0).round() as u64);
        if self.mark.is_none() {
            let _ = self.tx.send(WorkerCommand::MarkPosition);
        }
        let _ = self.tx.send(WorkerCommand::Nudge {
            direction: self.jog_direction,
            duration_ms: duration.as_millis() as u64,
            rate: RATES[self.jog_rate_index].1,
            ensure_tracking: false,
        });
        self.timed_jog = Some(TimedJog {
            direction: self.jog_direction,
            started: Instant::now(),
            duration,
        });
        self.status = format!(
            "Jogging {} for {seconds:.1} s at {}",
            self.jog_direction.label(),
            RATES[self.jog_rate_index].0
        );
    }

    fn cancel_timed_jog(&mut self) {
        self.timed_jog = None;
        self.stop_motion();
        self.status = "Timed jog stopped".into();
    }

    fn mark_here(&mut self) {
        if !self.connected {
            self.status = "Connect the mount first".into();
            return;
        }
        let _ = self.tx.send(WorkerCommand::MarkPosition);
    }

    /// Separation between the mark and where the mount currently reports being.
    /// `None` when either is unknown.
    fn return_separation_deg(&self) -> Option<f64> {
        let mark = self.mark?;
        let ra = parse_ra_hours(self.snapshot.ra.as_deref()?)?;
        let dec = parse_dec_deg(self.snapshot.dec.as_deref()?)?;
        Some(angular_separation_deg(
            mark.ra_hours,
            mark.dec_deg,
            ra,
            dec,
        ))
    }

    fn return_to_mark(&mut self, confirmed: bool) {
        let Some(mark) = self.mark else {
            self.status = "Nothing marked yet".into();
            return;
        };
        if !self.connected {
            self.status = "Connect the mount first".into();
            return;
        }
        // A GoTo moves at slew speed. Undoing a jog is arcminutes; anything
        // larger means the mark is stale, and that is worth a question before
        // the telescope crosses the sky on a single click.
        if !confirmed {
            if let Some(sep) = self.return_separation_deg() {
                if sep > RETURN_CONFIRM_DEG {
                    self.confirm_return = true;
                    self.status =
                        format!("Marked position is {sep:.1}° away — confirm before slewing");
                    return;
                }
            }
        }
        self.confirm_return = false;
        self.timed_jog = None;
        let _ = self.tx.send(WorkerCommand::SlewTo {
            ra_hours: mark.ra_hours,
            dec_deg: mark.dec_deg,
        });
    }

    fn manual_jog_ui(
        &mut self,
        ui: &mut egui::Ui,
        focus: &mut focus::FocusState,
        jog_enabled: bool,
        held: &mut Option<Direction>,
    ) {
        egui::Frame::group(ui.style())
            .fill(ui.visuals().faint_bg_color)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Manual jog").strong());
                    ui.label("rate");
                    let old_rate = self.rate_index;
                    egui::ComboBox::from_id_salt("mount_rate_main")
                        .selected_text(RATES[self.rate_index].0)
                        .width(72.0)
                        .show_ui(ui, |ui| {
                            for (index, (label, _, _)) in RATES.iter().enumerate() {
                                ui.selectable_value(&mut self.rate_index, index, *label);
                            }
                        });
                    if self.rate_index != old_rate && self.connected {
                        self.stop_motion();
                        *held = None;
                        let _ = self
                            .tx
                            .send(WorkerCommand::SetRate(RATES[self.rate_index].1));
                    }
                });

                let btn = egui::vec2(58.0, 34.0);
                let stop_sz = egui::vec2(78.0, 38.0);
                let gap = 5.0_f32;
                let mid_w = btn.x + gap + stop_sz.x + gap + btn.x;
                let mid_h = btn.y.max(stop_sz.y);
                let stop_left = btn.x + gap;
                let ns_left = stop_left + stop_sz.x * 0.5 - btn.x * 0.5;
                let pad_h = btn.y + gap + mid_h + gap + btn.y;

                ui.horizontal(|ui| {
                    let indent = ((ui.available_width() - mid_w) * 0.5).max(0.0);
                    ui.add_space(indent);
                    ui.allocate_ui_with_layout(
                        egui::vec2(mid_w, pad_h),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);

                            ui.horizontal(|ui| {
                                ui.add_space(ns_left);
                                let north = ui.add_enabled(
                                    jog_enabled,
                                    egui::Button::new(
                                        egui::RichText::new("N").size(16.0).strong(),
                                    )
                                    .min_size(btn),
                                );
                                if north.is_pointer_button_down_on() {
                                    *held = Some(Direction::North);
                                }
                            });
                            ui.add_space(gap);
                            ui.horizontal(|ui| {
                                let west = ui.add_enabled(
                                    jog_enabled,
                                    egui::Button::new(
                                        egui::RichText::new("W").size(16.0).strong(),
                                    )
                                    .min_size(btn),
                                );
                                if west.is_pointer_button_down_on() {
                                    *held = Some(Direction::West);
                                }
                                ui.add_space(gap);
                                let stop = egui::Button::new(
                                    egui::RichText::new("STOP")
                                        .size(15.0)
                                        .strong()
                                        .color(egui::Color32::WHITE),
                                )
                                .fill(egui::Color32::from_rgb(160, 35, 25))
                                .min_size(stop_sz);
                                if ui.add(stop).clicked() {
                                    self.cancel_auto_center(focus, "Sun auto-center stopped");
                                    self.stop_motion();
                                    *held = None;
                                }
                                ui.add_space(gap);
                                let east = ui.add_enabled(
                                    jog_enabled,
                                    egui::Button::new(
                                        egui::RichText::new("E").size(16.0).strong(),
                                    )
                                    .min_size(btn),
                                );
                                if east.is_pointer_button_down_on() {
                                    *held = Some(Direction::East);
                                }
                            });
                            ui.add_space(gap);
                            ui.horizontal(|ui| {
                                ui.add_space(ns_left);
                                let south = ui.add_enabled(
                                    jog_enabled,
                                    egui::Button::new(
                                        egui::RichText::new("S").size(16.0).strong(),
                                    )
                                    .min_size(btn),
                                );
                                if south.is_pointer_button_down_on() {
                                    *held = Some(Direction::South);
                                }
                            });
                        },
                    );
                });
                ui.label(
                    egui::RichText::new("Hold a direction; release to stop.")
                        .small()
                        .weak(),
                );
            });
    }

    fn ser_acquisition_ui(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
    ) {
        let anchors = focus.vertical_anchor_lines();
        if !focus.recording {
            let tracked = self.capture_anchor_y.and_then(|selected| {
                anchors
                    .iter()
                    .min_by(|a, b| {
                        (a.0 - selected)
                            .abs()
                            .partial_cmp(&(b.0 - selected).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .filter(|(center, _)| (*center - selected).abs() <= 20.0)
                    .map(|(center, _)| *center)
            });
            self.capture_anchor_y = tracked.or_else(|| {
                anchors
                    .iter()
                    .max_by(|a, b| {
                        a.1.partial_cmp(&b.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(center, _)| *center)
            });
        }

        // Shared helper, so this and the acquire tab cannot drift on how they
        // bound the capture band before a frame has arrived.
        let sensor_height = focus.known_frame_height().unwrap_or(1024).max(1);
        self.capture_height = self.capture_height.clamp(1, sensor_height);
        let vertical_dispersion = focus.dispersion == focus::DispAxis::Vertical;

        ui.add_space(8.0);
        egui::Frame::group(ui.style())
            .fill(ui.visuals().faint_bg_color)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("SER acquisition").strong());
                    ui.label(
                        egui::RichText::new(&focus.recording_status)
                            .small()
                            .color(if focus.recording {
                                egui::Color32::from_rgb(255, 110, 90)
                            } else {
                                ui.visuals().weak_text_color()
                            }),
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("anchor line");
                    let selected_text = self
                        .capture_anchor_y
                        .map(|center| format!("Y {center:.1} px"))
                        .unwrap_or_else(|| "no line detected".into());
                    ui.add_enabled_ui(!focus.recording && vertical_dispersion, |ui| {
                        egui::ComboBox::from_id_salt("ser_anchor_line")
                            .selected_text(selected_text)
                            .width(126.0)
                            .show_ui(ui, |ui| {
                                for (index, (center, depth)) in anchors.iter().enumerate() {
                                    ui.selectable_value(
                                        &mut self.capture_anchor_y,
                                        Some(*center),
                                        format!(
                                            "{} · Y {:.1} · depth {:.0}%",
                                            index + 1,
                                            center,
                                            depth * 100.0
                                        ),
                                    );
                                }
                            });
                    });
                    ui.label("vertical capture");
                    ui.add_enabled(
                        !focus.recording,
                        egui::DragValue::new(&mut self.capture_height)
                            .range(1..=sensor_height)
                            .speed(16.0)
                            .suffix(" px"),
                    );

                    if focus.recording {
                        let stop = egui::Button::new(
                            egui::RichText::new("■ Stop recording")
                                .strong()
                                .color(egui::Color32::WHITE),
                        )
                        .fill(egui::Color32::from_rgb(160, 35, 25));
                        if ui.add(stop).clicked() {
                            focus.stop_ser_recording();
                        }
                    } else {
                        let can_record = focus.streaming
                            && vertical_dispersion
                            && self.capture_anchor_y.is_some()
                            && self.auto_center.is_none();
                        let record = egui::Button::new(
                            egui::RichText::new("● Start recording")
                                .strong()
                                .color(egui::Color32::WHITE),
                        )
                        .fill(egui::Color32::from_rgb(145, 35, 28));
                        if ui.add_enabled(can_record, record).clicked() {
                            let stamp = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            if let Some(mut path) = rfd::FileDialog::new()
                                .add_filter("SER video", &["ser"])
                                .set_file_name(format!("ghostsun-{stamp}.ser"))
                                .save_file()
                            {
                                if path.extension().is_none() {
                                    path.set_extension("ser");
                                }
                                if let Some(anchor_y) = self.capture_anchor_y {
                                    if let Err(error) = focus.start_ser_recording(
                                        ctx,
                                        path,
                                        self.capture_height,
                                        anchor_y,
                                    ) {
                                        focus.recording_status = error;
                                    }
                                }
                            }
                        }
                    }
                });
                let hint = if !vertical_dispersion {
                    "Set dispersion to Vertical in Focus before selecting a horizontal spectral line."
                } else if anchors.is_empty() {
                    "Start the camera and expose a spectral line; detected lines appear in the anchor menu."
                } else {
                    "Records raw mono16 frames, vertically cropped around the fixed selected line."
                };
                ui.label(egui::RichText::new(hint).small().weak());
            });
    }

    fn timed_jog_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Timed jog and return").strong());
        ui.label(
            egui::RichText::new("Repeatable fixed move; the first jog marks its origin.")
                .small()
                .weak(),
        );

        let busy = self.timed_jog.is_some();
        let enabled = self.connected && self.auto_center.is_none() && !busy;

        ui.horizontal(|ui| {
            ui.label("direction:");
            for d in [
                Direction::North,
                Direction::South,
                Direction::East,
                Direction::West,
            ] {
                ui.add_enabled_ui(enabled, |ui| {
                    ui.selectable_value(&mut self.jog_direction, d, d.label());
                });
            }
        });
        ui.horizontal(|ui| {
            ui.label("duration");
            ui.add_enabled(
                enabled,
                egui::DragValue::new(&mut self.jog_seconds)
                    .range(0.1..=120.0)
                    .speed(0.1)
                    .suffix(" s")
                    .fixed_decimals(1),
            );
            ui.add_space(8.0);
            ui.label("speed");
            ui.add_enabled_ui(enabled, |ui| {
                egui::ComboBox::from_id_salt("timed_jog_rate")
                    .selected_text(RATES[self.jog_rate_index].0)
                    .width(80.0)
                    .show_ui(ui, |ui| {
                        for (index, (label, _, _)) in RATES.iter().enumerate() {
                            ui.selectable_value(&mut self.jog_rate_index, index, *label);
                        }
                    });
            });
        });

        // The distance is what makes a duration meaningful; the solar diameter
        // is the reference the user is actually working against.
        let arcmin = jog_arcmin(RATES[self.jog_rate_index].2, self.jog_seconds);
        let span = if arcmin >= 60.0 {
            format!("{:.2}°", arcmin / 60.0)
        } else {
            format!("{arcmin:.1}′")
        };
        ui.label(
            egui::RichText::new(format!("≈ {span} of sky  ·  solar diameter is ~32′"))
                .small()
                .weak(),
        );

        ui.add_space(4.0);
        match self.timed_jog {
            Some(jog) => {
                let elapsed = jog.started.elapsed().as_secs_f32();
                let total = jog.duration.as_secs_f32().max(0.001);
                ui.horizontal(|ui| {
                    ui.add(
                        egui::ProgressBar::new((elapsed / total).clamp(0.0, 1.0))
                            .desired_width(150.0)
                            .text(format!("{} {elapsed:.1}/{total:.1} s", jog.direction.label())),
                    );
                    if ui.button("stop").clicked() {
                        self.cancel_timed_jog();
                    }
                });
            }
            None => {
                let label = format!(
                    "▶ jog {} for {:.1} s",
                    self.jog_direction.label(),
                    self.jog_seconds
                );
                if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                    self.start_timed_jog();
                }
            }
        }

        ui.add_space(4.0);
        ui.separator();
        match self.mark {
            Some(mark) => {
                let sep = self
                    .return_separation_deg()
                    .map(|s| {
                        if s < 1.0 {
                            format!("{:.1}′ away", s * 60.0)
                        } else {
                            format!("{s:.2}° away")
                        }
                    })
                    .unwrap_or_else(|| "separation unknown".into());
                ui.label(
                    egui::RichText::new(format!(
                        "marked {} {}  ·  {sep}",
                        format_ra(mark.ra_hours),
                        format_dec(mark.dec_deg)
                    ))
                    .small()
                    .weak(),
                );
            }
            None => {
                ui.label(
                    egui::RichText::new("nothing marked — the first timed jog marks the origin")
                        .small()
                        .weak(),
                );
            }
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.connected, egui::Button::new("⌖ mark here"))
                .clicked()
            {
                self.mark_here();
            }
            if ui
                .add_enabled(
                    self.connected && self.mark.is_some() && self.auto_center.is_none(),
                    egui::Button::new("⟲ return to mark"),
                )
                .clicked()
            {
                self.return_to_mark(false);
            }
            if self.mark.is_some() && ui.button("clear").clicked() {
                self.mark = None;
                self.confirm_return = false;
            }
        });
        ui.label(
            egui::RichText::new("Return is a GoTo — the mount slews at full speed.")
                .small()
                .weak(),
        );

        if self.confirm_return {
            let sep = self
                .return_separation_deg()
                .map(|s| format!("{s:.1}°"))
                .unwrap_or_else(|| "an unknown distance".into());
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "DANGER: the marked position is {sep} away. Confirm a solar filter is \
                     fitted and the slew path is clear."
                ))
                .small()
                .color(egui::Color32::from_rgb(255, 120, 90)),
            );
            ui.horizontal(|ui| {
                if ui.button("Confirm slew").clicked() {
                    self.return_to_mark(true);
                }
                if ui.button("Cancel").clicked() {
                    self.confirm_return = false;
                }
            });
        }
    }

    fn auto_center_progress_label(&self) -> Option<String> {
        let state = self.auto_center.as_ref()?;
        let n = state.points.len().max(1);
        let i = state.point_index.min(n - 1) + 1;
        let pass = match state.pass {
            AutoCenterPass::Coarse => "coarse",
            AutoCenterPass::Refine => "refine",
        };
        let phase = match &state.phase {
            AutoCenterPhase::AwaitingSlew { .. } => "waiting for slew".into(),
            AutoCenterPhase::Settling { .. } => match state.origin {
                AutoCenterOrigin::SunGoTo => "settling after slew".into(),
                AutoCenterOrigin::CurrentPoint => "settling at current point".into(),
            },
            AutoCenterPhase::Sampling { samples, .. } => {
                format!(
                    "{pass} sampling {i}/{n} ({}/{})",
                    samples.len(),
                    SAMPLE_FRAMES
                )
            }
            AutoCenterPhase::Moving { target, .. } => {
                format!(
                    "{pass} nudge to ({:.1}°, {:.1}°) — point {i}/{n}",
                    target.0 as f32 * REFINE_STEP_DEG,
                    target.1 as f32 * REFINE_STEP_DEG
                )
            }
            AutoCenterPhase::ReturnReady => "returning to refined peak".into(),
            AutoCenterPhase::Finished => "finishing".into(),
        };
        let best = if state.best_signal.is_finite() {
            format!(
                " · best peak {:.0} at ({:.1}°, {:.1}°)",
                state.best_signal,
                state.best.0 as f32 * REFINE_STEP_DEG,
                state.best.1 as f32 * REFINE_STEP_DEG
            )
        } else {
            String::new()
        };
        Some(format!(
            "Auto-center ({}) — {phase}{best}",
            state.origin.label()
        ))
    }

    fn begin_auto_center(
        &mut self,
        ctx: &egui::Context,
        focus: &mut focus::FocusState,
        origin: AutoCenterOrigin,
    ) {
        if self.auto_center.is_some() {
            return;
        }
        if !self.connected {
            self.status = "Connect the mount before Sun auto-center".into();
            return;
        }
        let exposure_us = self.search_exposure_ms.saturating_mul(1000);
        let restore = match focus.prepare_sun_search(ctx, exposure_us) {
            Ok(restore) => restore,
            Err(error) => {
                self.status = error;
                return;
            }
        };

        let max_r = (self.search_radius_deg / SPIRAL_STEP_DEG)
            .round()
            .clamp(1.0, 6.0) as i32;
        let max_units = max_r * COARSE_GRID_SCALE;
        let points = coarse_spiral(max_r);
        let duration_ms = spiral_nudge_duration_ms().max(100);

        self.stop_motion();
        self.confirm_sun = false;
        self.confirm_motion = None;
        let phase = match origin {
            AutoCenterOrigin::SunGoTo => {
                let sun = sun_equatorial_now();
                let _ = self.tx.send(WorkerCommand::SlewSun {
                    ra_hours: sun.ra_hours,
                    dec_deg: sun.dec_deg,
                });
                AutoCenterPhase::AwaitingSlew {
                    started: Instant::now(),
                    saw_motion: false,
                }
            }
            AutoCenterOrigin::CurrentPoint => AutoCenterPhase::Settling {
                until: Instant::now() + SETTLE_AFTER_SLEW,
            },
        };

        self.auto_center = Some(AutoCenterState {
            restore,
            origin,
            pass: AutoCenterPass::Coarse,
            points,
            point_index: 0,
            current: (0, 0),
            best: (0, 0),
            best_signal: f32::NEG_INFINITY,
            max_units,
            duration_ms,
            overall_deadline: Instant::now() + AUTO_CENTER_TIMEOUT,
            phase,
        });
        self.status = format!(
            "Auto-center started: {} (radius {:.1}°, coarse {:.1}°, refine {:.1}°)",
            origin.label(),
            max_r as f32 * SPIRAL_STEP_DEG,
            SPIRAL_STEP_DEG,
            REFINE_STEP_DEG
        );
    }

    fn cancel_auto_center(&mut self, focus: &mut focus::FocusState, reason: &str) {
        let Some(state) = self.auto_center.take() else {
            return;
        };
        self.stop_motion();
        focus.restore_after_sun_search(state.restore);
        self.confirm_auto_center = false;
        self.status = reason.into();
    }

    fn auto_center_nudge_done(&mut self) {
        let Some(state) = self.auto_center.as_mut() else {
            return;
        };
        let AutoCenterPhase::Moving { target, .. } = state.phase else {
            return;
        };
        let target = target;
        if state.current == target {
            // Arrival handled in advance_auto_center (needs focus for sampling).
            state.phase = AutoCenterPhase::Moving {
                target,
                sample_index: Some(state.point_index),
            };
        } else if let Some((direction, next)) = grid_step_direction(state.current, target) {
            let duration_ms = state.duration_ms;
            state.current = next;
            let _ = self.tx.send(WorkerCommand::Nudge {
                direction,
                duration_ms,
                rate: AUTO_CENTER_RATE,
                ensure_tracking: true,
            });
        } else {
            state.phase = AutoCenterPhase::Moving {
                target,
                sample_index: Some(state.point_index),
            };
        }
    }

    fn advance_auto_center(&mut self, focus: &mut focus::FocusState) {
        if self.auto_center.is_none() {
            return;
        }
        if self
            .auto_center
            .as_ref()
            .is_some_and(|s| Instant::now() >= s.overall_deadline)
        {
            self.cancel_auto_center(focus, "Sun auto-center timed out (10 min limit)");
            return;
        }

        enum Next {
            Idle,
            BeginSample,
            FinishSuccess,
            Cancel(&'static str),
            StartMove { target: (i32, i32) },
        }

        let next = {
            let state = self.auto_center.as_mut().expect("checked");
            match &mut state.phase {
                AutoCenterPhase::AwaitingSlew {
                    started,
                    saw_motion,
                } => {
                    if self.snapshot.slewing == Some(true) {
                        *saw_motion = true;
                    }
                    if *saw_motion && self.snapshot.slewing == Some(false) {
                        state.phase = AutoCenterPhase::Settling {
                            until: Instant::now() + SETTLE_AFTER_SLEW,
                        };
                        self.status = "Sun auto-center: slew complete, settling…".into();
                        Next::Idle
                    } else if started.elapsed() >= SLEW_TIMEOUT {
                        Next::Cancel("Sun auto-center: slew timed out")
                    } else if !*saw_motion
                        && started.elapsed() >= Duration::from_secs(12)
                        && self.snapshot.slewing != Some(true)
                    {
                        // Some firmwares never report the slewing flag; don't hang forever.
                        state.phase = AutoCenterPhase::Settling {
                            until: Instant::now() + SETTLE_AFTER_SLEW,
                        };
                        self.status =
                            "Sun auto-center: no slew flag seen; settling and sampling…".into();
                        Next::Idle
                    } else {
                        Next::Idle
                    }
                }
                AutoCenterPhase::Settling { until } => {
                    if Instant::now() >= *until {
                        Next::BeginSample
                    } else {
                        Next::Idle
                    }
                }
                AutoCenterPhase::Sampling {
                    last_seq,
                    samples,
                    deadline,
                } => {
                    if let Some((seq, peak)) = focus.sun_signal_sample() {
                        if seq > *last_seq {
                            *last_seq = seq;
                            samples.push(peak);
                        }
                    }
                    let sample_ready =
                        samples.len() >= SAMPLE_FRAMES || Instant::now() >= *deadline;
                    if !sample_ready {
                        Next::Idle
                    } else if samples.is_empty() {
                        Next::Cancel("Sun auto-center: no camera frames while sampling")
                    } else {
                        // Average several robust per-frame peaks to suppress
                        // scintillation while still seeking the local maximum.
                        let peak = samples.iter().sum::<f32>() / samples.len() as f32;
                        let at = state
                            .points
                            .get(state.point_index)
                            .copied()
                            .unwrap_or(state.current);
                        if peak > state.best_signal {
                            state.best_signal = peak;
                            state.best = at;
                        }
                        let pass = match state.pass {
                            AutoCenterPass::Coarse => "coarse",
                            AutoCenterPass::Refine => "refine",
                        };
                        self.status = format!(
                            "Sun auto-center: {pass} point {}/{} peak {:.0} (best {:.0})",
                            state.point_index + 1,
                            state.points.len(),
                            peak,
                            state.best_signal
                        );
                        state.point_index += 1;
                        if state.point_index < state.points.len() {
                            let target = state.points[state.point_index];
                            Next::StartMove { target }
                        } else {
                            match state.pass {
                                AutoCenterPass::Coarse => {
                                    let coarse_best = state.best;
                                    state.pass = AutoCenterPass::Refine;
                                    state.points =
                                        refinement_grid(coarse_best, state.max_units);
                                    state.point_index = 0;
                                    state.best = coarse_best;
                                    state.best_signal = f32::NEG_INFINITY;
                                    let target = state.points[0];
                                    self.status =
                                        "Sun auto-center: coarse maximum found; refining at 0.1°"
                                            .into();
                                    Next::StartMove { target }
                                }
                                AutoCenterPass::Refine => {
                                    state.phase = AutoCenterPhase::ReturnReady;
                                    Next::Idle
                                }
                            }
                        }
                    }
                }
                AutoCenterPhase::Moving {
                    target,
                    sample_index,
                } => {
                    if sample_index.is_some() || state.current == *target {
                        Next::BeginSample
                    } else {
                        Next::Idle
                    }
                }
                AutoCenterPhase::ReturnReady => {
                    if state.current == state.best {
                        state.phase = AutoCenterPhase::Finished;
                        Next::FinishSuccess
                    } else {
                        let target = state.best;
                        Next::StartMove { target }
                    }
                }
                AutoCenterPhase::Finished => Next::FinishSuccess,
            }
        };

        match next {
            Next::Idle => {}
            Next::Cancel(reason) => self.cancel_auto_center(focus, reason),
            Next::FinishSuccess => {
                let (origin, best, peak) = self
                    .auto_center
                    .as_ref()
                    .map(|s| (s.origin, s.best, s.best_signal))
                    .unwrap_or((AutoCenterOrigin::CurrentPoint, (0, 0), 0.0));
                self.cancel_auto_center(
                    focus,
                    &format!(
                        "Auto-center complete ({}): peak {:.0} at offset ({:.1}°, {:.1}°)",
                        origin.label(),
                        peak,
                        best.0 as f32 * REFINE_STEP_DEG,
                        best.1 as f32 * REFINE_STEP_DEG
                    ),
                );
            }
            Next::BeginSample => {
                let seq = focus.sun_signal_sample().map(|(s, _)| s).unwrap_or(0);
                if let Some(state) = self.auto_center.as_mut() {
                    state.phase = AutoCenterPhase::Sampling {
                        last_seq: seq,
                        samples: Vec::new(),
                        deadline: Instant::now() + SAMPLE_TIMEOUT,
                    };
                }
            }
            Next::StartMove { target } => {
                if let Some(state) = self.auto_center.as_mut() {
                    if state.current == target {
                        let seq = focus.sun_signal_sample().map(|(s, _)| s).unwrap_or(0);
                        state.phase = AutoCenterPhase::Sampling {
                            last_seq: seq,
                            samples: Vec::new(),
                            deadline: Instant::now() + SAMPLE_TIMEOUT,
                        };
                    } else if let Some((direction, next_pos)) =
                        grid_step_direction(state.current, target)
                    {
                        let duration_ms = state.duration_ms;
                        state.current = next_pos;
                        state.phase = AutoCenterPhase::Moving {
                            target,
                            sample_index: None,
                        };
                        let _ = self.tx.send(WorkerCommand::Nudge {
                            direction,
                            duration_ms,
                            rate: AUTO_CENTER_RATE,
                            ensure_tracking: true,
                        });
                    } else {
                        let seq = focus.sun_signal_sample().map(|(s, _)| s).unwrap_or(0);
                        state.phase = AutoCenterPhase::Sampling {
                            last_seq: seq,
                            samples: Vec::new(),
                            deadline: Instant::now() + SAMPLE_TIMEOUT,
                        };
                    }
                }
            }
        }

        // ReturnReady → StartMove sets Moving; if we just set ReturnReady and
        // best != current, re-enter once so the first return nudge is issued
        // without waiting another poll tick.
        if matches!(
            self.auto_center.as_ref().map(|s| &s.phase),
            Some(AutoCenterPhase::ReturnReady)
        ) {
            self.advance_auto_center(focus);
        }
    }
}

impl Drop for MountState {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerCommand::Stop);
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn discover_ports() -> Result<Vec<PortInfo>, String> {
    let mut ports: Vec<PortInfo> = serialport::available_ports()
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|port| {
            let is_zwo = matches!(
                &port.port_type,
                SerialPortType::UsbPort(usb) if usb.vid == 0x03c3
            );
            let detail = match port.port_type {
                SerialPortType::UsbPort(usb) => {
                    let product = usb.product.unwrap_or_else(|| "USB serial".into());
                    format!("{product} - {:04X}:{:04X}", usb.vid, usb.pid)
                }
                SerialPortType::BluetoothPort => "Bluetooth".into(),
                SerialPortType::PciPort => "PCI serial".into(),
                SerialPortType::Unknown => "Serial port".into(),
            };
            PortInfo {
                name: port.port_name,
                detail,
                is_zwo,
            }
        })
        .collect();

    #[cfg(target_os = "macos")]
    {
        // IOKit can expose both tty.* and cu.* names for one adapter. cu.* is
        // the correct outgoing endpoint, and scanning /dev is a useful fallback
        // for adapters whose USB metadata is not returned by IOKit.
        ports.retain(|port| !port.name.starts_with("/dev/tty."));
        if let Ok(entries) = std::fs::read_dir("/dev") {
            for entry in entries.flatten() {
                let filename = entry.file_name();
                let filename = filename.to_string_lossy();
                if !filename.starts_with("cu.") {
                    continue;
                }
                let name = format!("/dev/{filename}");
                if !ports.iter().any(|port| port.name == name) {
                    ports.push(PortInfo {
                        name,
                        detail: "macOS /dev/cu fallback".into(),
                        is_zwo: false,
                    });
                }
            }
        }
    }

    ports.sort_by(|a, b| (!a.is_zwo, &a.name).cmp(&(!b.is_zwo, &b.name)));
    Ok(ports)
}

#[cfg(target_os = "windows")]
fn platform_scan_description() -> &'static str {
    "Windows SetupAPI COM ports"
}

#[cfg(target_os = "macos")]
fn platform_scan_description() -> &'static str {
    "macOS IOKit plus /dev/cu.*"
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn platform_scan_description() -> &'static str {
    "system serial ports"
}

#[cfg(target_os = "windows")]
fn connection_checklist() -> &'static str {
    "1. Power on the mount.\n2. Connect a data-capable cable to the mount's USB control port.\n3. Check Device Manager > Ports for \"USB Serial Device (COMx)\".\n4. Open ASI Mount and confirm it detects the mount, then return here and refresh."
}

#[cfg(target_os = "macos")]
fn connection_checklist() -> &'static str {
    "1. Power on the mount.\n2. Connect a data-capable cable to the mount's USB control port.\n3. Check System Information > USB for the mount.\n4. Open ASIStudio and confirm it detects the mount, then return here and refresh."
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn connection_checklist() -> &'static str {
    "Power on the mount, use a data-capable USB cable, confirm it in the system USB/serial device list, then refresh."
}

#[cfg(target_os = "windows")]
fn native_software_button_label() -> &'static str {
    "Open ASI Mount"
}

#[cfg(target_os = "macos")]
fn native_software_button_label() -> &'static str {
    "Open ASIStudio"
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn native_software_button_label() -> &'static str {
    "Open ZWO software page"
}

#[cfg(target_os = "windows")]
fn launch_zwo_software() -> Result<String, String> {
    let mut candidates = Vec::new();
    if let Some(root) = std::env::var_os("ProgramFiles(x86)") {
        let root = PathBuf::from(root);
        candidates.push(
            root.join("Common Files")
                .join("ASCOM")
                .join("ZWO")
                .join("ASIMount")
                .join("ASCOM.ASIMount.Server.exe"),
        );
        candidates.push(root.join("ZWO").join("ASIStudio").join("ASIStudio.exe"));
    }
    if let Some(root) = std::env::var_os("ProgramFiles") {
        let root = PathBuf::from(root);
        candidates.push(root.join("ZWO").join("ASIStudio").join("ASIStudio.exe"));
        candidates.push(root.join("ASIStudio").join("ASIStudio.exe"));
    }

    for executable in candidates {
        if executable.is_file() {
            Command::new(&executable)
                .spawn()
                .map_err(|error| format!("{}: {error}", executable.display()))?;
            return Ok(format!("Opened {}", executable.display()));
        }
    }
    webbrowser::open("https://www.zwoastro.com/software/").map_err(|error| error.to_string())?;
    Ok("ASI Mount was not found locally; opened the ZWO software page".into())
}

#[cfg(target_os = "macos")]
fn launch_zwo_software() -> Result<String, String> {
    let mut candidates = vec![
        PathBuf::from("/Applications/ASIStudio.app"),
        PathBuf::from("/Applications/ASI Mount.app"),
    ];
    if let Some(home) = std::env::var_os("HOME") {
        let applications = PathBuf::from(home).join("Applications");
        candidates.push(applications.join("ASIStudio.app"));
        candidates.push(applications.join("ASI Mount.app"));
    }
    for application in candidates {
        if application.is_dir() {
            let status = Command::new("open")
                .arg(&application)
                .status()
                .map_err(|error| error.to_string())?;
            if status.success() {
                return Ok(format!("Opened {}", application.display()));
            }
        }
    }
    webbrowser::open("https://www.zwoastro.com/software/").map_err(|error| error.to_string())?;
    Ok("ASIStudio was not found locally; opened the ZWO software page".into())
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn launch_zwo_software() -> Result<String, String> {
    webbrowser::open("https://www.zwoastro.com/software/").map_err(|error| error.to_string())?;
    Ok("Opened the ZWO software page".into())
}

fn status_card(ui: &mut egui::Ui, title: &str, value: &str) {
    egui::Frame::group(ui.style())
        .fill(ui.visuals().faint_bg_color)
        .show(ui, |ui| {
            ui.set_min_width(145.0);
            ui.label(egui::RichText::new(title).small().weak());
            ui.label(egui::RichText::new(value).size(18.0).strong().color(ACCENT));
        });
}

fn worker_loop(rx: Receiver<WorkerCommand>, tx: Sender<WorkerMessage>) {
    let mut port: Option<Box<dyn SerialPort>> = None;
    let mut timed_move: Option<(Instant, Direction, bool)> = None;

    loop {
        if timed_move
            .map(|(deadline, _, _)| Instant::now() >= deadline)
            .unwrap_or(false)
        {
            let (_, direction, ensure_tracking) = timed_move.take().unwrap();
            if let Some(opened) = port.as_deref_mut() {
                let _ = blind(opened, direction.stop_command());
                if ensure_tracking {
                    let _ = blind(opened, ":Te#");
                }
            }
            let _ = tx.send(WorkerMessage::NudgeDone);
        }
        let wait = timed_move
            .map(|(deadline, _, _)| deadline.saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::from_millis(100))
            .min(Duration::from_millis(100));
        match rx.recv_timeout(wait) {
            Ok(WorkerCommand::Connect(name)) => {
                timed_move = None;
                if let Some(mut old) = port.take() {
                    let _ = blind(&mut *old, ":Q#");
                }
                match open_and_probe(&name) {
                    Ok((opened, model)) => {
                        port = Some(opened);
                        let _ = tx.send(WorkerMessage::Connected { port: name, model });
                    }
                    Err(error) => {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::Disconnect) => {
                timed_move = None;
                if let Some(mut opened) = port.take() {
                    let _ = blind(&mut *opened, ":Q#");
                }
                let _ = tx.send(WorkerMessage::Disconnected("Disconnected".into()));
            }
            Ok(WorkerCommand::Poll) => {
                if let Some(opened) = port.as_deref_mut() {
                    let flags = query(opened, ":GU#").ok();
                    let home = flags.as_ref().map(|value| {
                        if value.contains('H') {
                            "At home".into()
                        } else {
                            "Away".into()
                        }
                    });
                    let slewing = flags.as_ref().map(|value| !value.contains('N'));
                    let park = query(opened, ":Gps#")
                        .ok()
                        .map(|value| match value.as_str() {
                            "1" => "Parking".into(),
                            "2" => "Parked".into(),
                            "3" => "Park error".into(),
                            _ => "Not parked".into(),
                        });
                    let snapshot = MountSnapshot {
                        ra: query(opened, ":GR#").ok(),
                        dec: query(opened, ":GD#").ok(),
                        altitude: query(opened, ":GA#").ok(),
                        azimuth: query(opened, ":GZ#").ok(),
                        tracking: query(opened, ":GAT#").ok().map(|v| {
                            if v.starts_with('1') {
                                "On".into()
                            } else {
                                "Off".into()
                            }
                        }),
                        slewing,
                        home,
                        park,
                        flags,
                    };
                    let _ = tx.send(WorkerMessage::Snapshot(snapshot));
                } else {
                    let _ = tx.send(WorkerMessage::Disconnected("Not connected".into()));
                }
            }
            Ok(WorkerCommand::SetRate(rate)) => {
                if let Some(opened) = port.as_deref_mut() {
                    if let Err(error) = blind(opened, &format!(":R{rate}#")) {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::Jog(direction)) => {
                if let Some(opened) = port.as_deref_mut() {
                    if let Err(error) = blind(opened, direction.move_command()) {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::Stop) => {
                timed_move = None;
                if let Some(opened) = port.as_deref_mut() {
                    if let Err(error) = blind(opened, ":Q#") {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::StopAcquisition) => {
                let direction = timed_move.take().map(|(_, direction, _)| direction);
                if let Some(opened) = port.as_deref_mut() {
                    let result = if let Some(direction) = direction {
                        blind(opened, direction.stop_command())
                    } else {
                        Ok(())
                    }
                    .and_then(|_| blind(opened, ":Te#"));
                    if let Err(error) = result {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::GoHome) => {
                timed_move = None;
                if let Some(opened) = port.as_deref_mut() {
                    let result = blind(opened, ":Q#").and_then(|_| blind(opened, ":hC#"));
                    match result {
                        Ok(()) => {
                            let _ = tx.send(WorkerMessage::Notice(
                                "Go Home started; STOP aborts the movement".into(),
                            ));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::Park) => {
                timed_move = None;
                if let Some(opened) = port.as_deref_mut() {
                    let result = blind(opened, ":Q#").and_then(|_| blind(opened, ":hP#"));
                    match result {
                        Ok(()) => {
                            let _ = tx.send(WorkerMessage::Notice(
                                "Park started; STOP aborts the movement".into(),
                            ));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::Unpark) => {
                if let Some(opened) = port.as_deref_mut() {
                    match expect_ack(opened, ":Spu#", "unpark") {
                        Ok(()) => {
                            let _ = tx.send(WorkerMessage::Notice("Mount is unparked".into()));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::SetSolarRate) => {
                if let Some(opened) = port.as_deref_mut() {
                    match blind(opened, ":TS#") {
                        Ok(()) => {
                            let _ = tx.send(WorkerMessage::Notice(
                                "Solar tracking rate selected".into(),
                            ));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::SetTracking(enable)) => {
                if let Some(opened) = port.as_deref_mut() {
                    // ZWO AM-series LX200 dialect: :Te# enable tracking, :Td# disable.
                    // GhostSun is a solar instrument, so enabling tracking selects
                    // the SOLAR rate: leaving it as-is meant the Sun was tracked at
                    // the sidereal rate and drifted whenever it had not arrived via
                    // a Sun GoTo.
                    if enable {
                        let _ = blind(opened, ":TS#");
                    }
                    let cmd = if enable { ":Te#" } else { ":Td#" };
                    match blind(opened, cmd) {
                        Ok(()) => {
                            let _ = tx.send(WorkerMessage::Notice(if enable {
                                "Tracking enabled".into()
                            } else {
                                "Tracking disabled".into()
                            }));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::SyncSiteTime {
                latitude_deg,
                longitude_deg,
                utc_offset_hours,
            }) => {
                if let Some(opened) = port.as_deref_mut() {
                    match sync_site_and_time(opened, latitude_deg, longitude_deg, utc_offset_hours)
                    {
                        Ok(summary) => {
                            let _ = tx.send(WorkerMessage::Notice(summary));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::MarkPosition) => {
                let result = match port.as_deref_mut() {
                    Some(opened) => query(opened, ":GR#").and_then(|ra| {
                        let dec = query(opened, ":GD#")?;
                        let ra_hours = parse_ra_hours(&ra)
                            .ok_or_else(|| format!("could not read right ascension: {ra:?}"))?;
                        let dec_deg = parse_dec_deg(&dec)
                            .ok_or_else(|| format!("could not read declination: {dec:?}"))?;
                        Ok((ra_hours, dec_deg))
                    }),
                    None => Err("mount is not connected".into()),
                };
                match result {
                    Ok((ra_hours, dec_deg)) => {
                        let _ = tx.send(WorkerMessage::Marked { ra_hours, dec_deg });
                    }
                    Err(error) => {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::SlewTo { ra_hours, dec_deg }) => {
                timed_move = None;
                let result = match port.as_deref_mut() {
                    Some(opened) => slew_to_coords(opened, ra_hours, dec_deg, false),
                    None => Err("mount is not connected".into()),
                };
                match result {
                    Ok(()) => {
                        let _ = tx.send(WorkerMessage::Notice(
                            "Returning to the marked position".into(),
                        ));
                    }
                    Err(error) => {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::Nudge {
                direction,
                duration_ms,
                rate,
                ensure_tracking,
            }) => {
                let previous = timed_move.take().map(|(_, direction, _)| direction);
                if let Some(opened) = port.as_deref_mut() {
                    let result = if let Some(previous) = previous {
                        blind(opened, previous.stop_command())
                    } else {
                        Ok(())
                    }
                    .and_then(|_| {
                        if ensure_tracking {
                            // Solar rate BEFORE re-enabling tracking. `:Te#`
                            // alone never selected one, so a mount sitting on
                            // the sidereal rate resumed 0.041 arcsec/s fast
                            // after every nudge — ~2.5 arcmin per hour, a
                            // twelfth of the disc, walking the slit off the
                            // feature mid-scan.
                            blind(opened, ":TS#").and_then(|_| blind(opened, ":Te#"))
                        } else {
                            Ok(())
                        }
                    })
                        .and_then(|_| blind(opened, &format!(":R{}#", rate.min(9))))
                        .and_then(|_| blind(opened, direction.move_command()));
                    match result {
                        Ok(()) => {
                            timed_move = Some((
                                Instant::now() + Duration::from_millis(duration_ms),
                                direction,
                                ensure_tracking,
                            ));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::SlewSun { ra_hours, dec_deg }) => {
                timed_move = None;
                let result = match port.as_deref_mut() {
                    Some(opened) => slew_to_sun(opened, ra_hours, dec_deg),
                    None => Err("mount is not connected".into()),
                };
                match result {
                    Ok(()) => {
                        let _ = tx.send(WorkerMessage::Notice(
                            "Sun slew started; solar tracking requested".into(),
                        ));
                    }
                    Err(error) => {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::Shutdown) => {
                if let Some(mut opened) = port.take() {
                    let _ = blind(&mut *opened, ":Q#");
                }
                break;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Local − UTC in hours from the OS clock (best-effort).
fn system_utc_offset_hours() -> f64 {
    #[cfg(windows)]
    {
        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "[TimeZoneInfo]::Local.GetUtcOffset([DateTime]::UtcNow).TotalHours",
            ])
            .output();
        if let Ok(out) = output {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(v) = s.trim().parse::<f64>() {
                    return v;
                }
            }
        }
        0.0
    }
    #[cfg(not(windows))]
    {
        let output = Command::new("date").arg("+%z").output();
        if let Ok(out) = output {
            if let Ok(s) = String::from_utf8(out.stdout) {
                let s = s.trim();
                // ±HHMM
                if s.len() >= 3 {
                    if let (Ok(h), Ok(m)) = (s[..3].parse::<i32>(), s[3..].parse::<i32>()) {
                        let sign = if h < 0 { -1.0 } else { 1.0 };
                        return h as f64 + sign * (m.abs() as f64) / 60.0;
                    }
                }
            }
        }
        0.0
    }
}

fn format_lat_lx200(lat_deg: f64) -> String {
    let sign = if lat_deg >= 0.0 { '+' } else { '-' };
    let a = lat_deg.abs().min(90.0);
    let mut d = a.floor() as i32;
    let mut m = ((a - f64::from(d)) * 60.0).round() as i32;
    if m >= 60 {
        d += 1;
        m = 0;
    }
    format!("{sign}{d:02}*{m:02}")
}

/// Meade/ZWO serial longitude is degrees **West** of Greenwich, 0–360.
fn format_lon_lx200_west(east_positive_lon: f64) -> String {
    let mut west = (-east_positive_lon).rem_euclid(360.0);
    if west >= 360.0 {
        west = 0.0;
    }
    let mut d = west.floor() as i32;
    let mut m = ((west - f64::from(d)) * 60.0).round() as i32;
    if m >= 60 {
        d += 1;
        m = 0;
    }
    if d >= 360 {
        d = 0;
    }
    format!("{d:03}*{m:02}")
}

fn format_meade_utc_offset(local_minus_utc_hours: f64) -> String {
    // Meade :SG is hours to *add* to local time to get UTC (= −(local−UTC)).
    let hours_to_utc = -local_minus_utc_hours;
    let sign = if hours_to_utc >= 0.0 { '+' } else { '-' };
    let a = hours_to_utc.abs();
    let h = a.floor() as i32;
    let frac = a - f64::from(h);
    if frac < 0.05 {
        format!("{sign}{h:02}")
    } else {
        format!(
            "{sign}{h:02}.{}",
            ((frac * 10.0).round() as i32).clamp(0, 9)
        )
    }
}

fn host_local_date_time() -> Result<(String, String), String> {
    // MM/DD/YY and HH:MM:SS in the machine's local timezone.
    #[cfg(windows)]
    {
        let date = Command::new("powershell")
            .args(["-NoProfile", "-Command", "Get-Date -Format 'MM/dd/yy'"])
            .output()
            .map_err(|e| e.to_string())?;
        let time = Command::new("powershell")
            .args(["-NoProfile", "-Command", "Get-Date -Format 'HH:mm:ss'"])
            .output()
            .map_err(|e| e.to_string())?;
        let d = String::from_utf8_lossy(&date.stdout).trim().to_owned();
        let t = String::from_utf8_lossy(&time.stdout).trim().to_owned();
        if d.is_empty() || t.is_empty() {
            return Err("could not read local date/time".into());
        }
        Ok((d, t))
    }
    #[cfg(not(windows))]
    {
        let date = Command::new("date")
            .arg("+%m/%d/%y")
            .output()
            .map_err(|e| e.to_string())?;
        let time = Command::new("date")
            .arg("+%H:%M:%S")
            .output()
            .map_err(|e| e.to_string())?;
        let d = String::from_utf8_lossy(&date.stdout).trim().to_owned();
        let t = String::from_utf8_lossy(&time.stdout).trim().to_owned();
        if d.is_empty() || t.is_empty() {
            return Err("could not read local date/time".into());
        }
        Ok((d, t))
    }
}

fn sync_site_and_time(
    port: &mut dyn SerialPort,
    latitude_deg: f64,
    longitude_deg: f64,
    utc_offset_hours: f64,
) -> Result<String, String> {
    let (date, time) = host_local_date_time()?;
    let lat = format_lat_lx200(latitude_deg);
    let lon = format_lon_lx200_west(longitude_deg);
    let sg = format_meade_utc_offset(utc_offset_hours);

    // Order matches common LX200/AM setup: date, local time, UTC offset, lat, lon.
    expect_ack(port, &format!(":SC{date}#"), "calendar date")?;
    expect_ack(port, &format!(":SL{time}#"), "local time")?;
    expect_ack(port, &format!(":SG{sg}#"), "UTC offset")?;
    expect_ack(port, &format!(":St{lat}#"), "latitude")?;
    expect_ack(port, &format!(":Sg{lon}#"), "longitude")?;

    let gt = query(port, ":Gt#").unwrap_or_else(|_| "?".into());
    let gg = query(port, ":Gg#").unwrap_or_else(|_| "?".into());
    let gl = query(port, ":GL#").unwrap_or_else(|_| "?".into());
    Ok(format!(
        "Site/time synced (local {date} {time}, UTC{sg}, lat {lat}, lonW {lon}). Mount reports Gt={gt} Gg={gg} GL={gl}"
    ))
}

/// Same TLS stack as GONG: this crate enables `ureq/native-tls` only. Bare
/// `ureq::get` defaults to Rustls and **panics** with
/// `uri scheme is https, provider is Rustls but feature is not enabled`.
fn native_tls_agent() -> ureq::Agent {
    use ureq::tls::{RootCerts, TlsConfig, TlsProvider};
    let config = ureq::config::Config::builder()
        .tls_config(
            TlsConfig::builder()
                .provider(TlsProvider::NativeTls)
                .root_certs(RootCerts::PlatformVerifier)
                .build(),
        )
        .build();
    config.new_agent()
}

/// OpenStreetMap Nominatim place search (no Google key required).
fn nominatim_search(query: &str) -> Result<Vec<(String, f64, f64)>, String> {
    let url = format!(
        "https://nominatim.openstreetmap.org/search?format=json&limit=6&q={}",
        urlencoding_minimal(query)
    );
    let agent = native_tls_agent();
    let mut response = agent
        .get(&url)
        .header(
            "User-Agent",
            "GhostSun/0.2 (solar spectroheliograph; mount site setup)",
        )
        .header("Accept", "application/json")
        .call()
        .map_err(|e| format!("place search request failed: {e}"))?;
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("cannot read place search response: {e}"))?;
    parse_nominatim_json(&body)
}

fn urlencoding_minimal(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn parse_nominatim_json(body: &str) -> Result<Vec<(String, f64, f64)>, String> {
    // Minimal JSON scrape — avoids a serde dependency for a tiny payload.
    let mut hits = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find('{') {
        rest = &rest[start..];
        let Some(end) = rest.find('}') else {
            break;
        };
        let obj = &rest[..=end];
        rest = &rest[end + 1..];
        let lat = json_string_field(obj, "lat").and_then(|s| s.parse().ok());
        let lon = json_string_field(obj, "lon").and_then(|s| s.parse().ok());
        let name = json_string_field(obj, "display_name");
        if let (Some(lat), Some(lon), Some(name)) = (lat, lon, name) {
            hits.push((truncate_chars(&name, 72), lat, lon));
        }
        if hits.len() >= 6 {
            break;
        }
    }
    Ok(hits)
}

fn json_string_field(obj: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":\"");
    let start = obj.find(&pattern)? + pattern.len();
    let bytes = obj.as_bytes();
    let mut out = String::new();
    let mut i = start;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'"' {
            return Some(out);
        }
        if c == b'\\' && i + 1 < bytes.len() {
            // Keep escapes simple; Nominatim names rarely need full JSON decode.
            out.push(bytes[i + 1] as char);
            i += 2;
            continue;
        }
        // Decode UTF-8 sequences so multi-byte place names stay valid.
        let width = match c {
            0x00..=0x7f => 1,
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => 1,
        };
        if i + width <= bytes.len() {
            if let Ok(s) = std::str::from_utf8(&bytes[i..i + width]) {
                out.push_str(s);
                i += width;
                continue;
            }
        }
        i += 1;
    }
    None
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut it = s.chars();
    let head: String = it.by_ref().take(max_chars).collect();
    if it.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

fn open_and_probe(name: &str) -> Result<(Box<dyn SerialPort>, String), String> {
    let mut port = serialport::new(name, 9_600)
        .data_bits(DataBits::Eight)
        .stop_bits(StopBits::One)
        .parity(Parity::None)
        .flow_control(FlowControl::None)
        .timeout(Duration::from_millis(120))
        .open()
        .map_err(|error| format!("cannot open {name}: {error}"))?;
    thread::sleep(Duration::from_millis(100));
    let model = query(&mut *port, ":GVP#")
        .map_err(|error| format!("{name} did not answer as an LX200/ZWO mount: {error}"))?;
    if model.trim().is_empty() {
        return Err(format!("{name} returned an empty mount identity"));
    }
    Ok((port, model))
}

fn slew_to_sun(port: &mut dyn SerialPort, ra_hours: f64, dec_deg: f64) -> Result<(), String> {
    slew_to_coords(port, ra_hours, dec_deg, true)
}

/// GoTo the given coordinates.
///
/// `solar_rate` selects the solar tracking rate before the slew. That is right
/// when acquiring the Sun, but wrong for a return-to-mark, which must not
/// silently change a tracking mode the user chose.
fn slew_to_coords(
    port: &mut dyn SerialPort,
    ra_hours: f64,
    dec_deg: f64,
    solar_rate: bool,
) -> Result<(), String> {
    blind(port, ":Q#")?;
    expect_ack(
        port,
        &format!(":Sr{}#", format_ra(ra_hours)),
        "right ascension",
    )?;
    expect_ack(port, &format!(":Sd{}#", format_dec(dec_deg)), "declination")?;
    if solar_rate {
        // A successful ZWO GoTo enables tracking on arrival. Select the solar
        // rate first so a separate tracking command cannot race the active slew.
        blind(port, ":TS#")?;
    }
    let response = transaction(port, ":MS#", false)?;
    match response.chars().next() {
        Some('0') => Ok(()),
        _ if response.starts_with('e') => Err(goto_error(&response)),
        _ => Err(format!("mount rejected GoTo: {response:?}")),
    }
}

/// Parse an LX200 `:GR#` reply into hours.
///
/// The mount answers in either low precision (`HH:MM.T`) or high precision
/// (`HH:MM:SS`) depending on its own setting, and GhostSun never forces one, so
/// both must be accepted.
fn parse_ra_hours(text: &str) -> Option<f64> {
    let t = text.trim().trim_end_matches('#').trim();
    let (h, rest) = t.split_once(':')?;
    let h: f64 = h.trim().parse().ok()?;
    let (m, s) = match rest.split_once(':') {
        // HH:MM:SS
        Some((m, s)) => (m.trim().parse::<f64>().ok()?, s.trim().parse::<f64>().ok()?),
        // HH:MM.T — the fraction is tenths of a minute, not seconds.
        None => (rest.trim().parse::<f64>().ok()?, 0.0),
    };
    let hours = h.abs() + m / 60.0 + s / 3600.0;
    hours.is_finite().then_some(hours)
}

/// Parse an LX200 `:GD#` reply into degrees.
///
/// Degrees are separated by `*` (or `°`/`:` on some firmwares) and the reply is
/// either `sDD*MM` or `sDD*MM:SS`. The sign belongs to the whole value, so it
/// must be applied after the arcminute and arcsecond terms are added — negating
/// only the degree field puts a target at −0°30′ on the wrong side of the
/// equator.
fn parse_dec_deg(text: &str) -> Option<f64> {
    let t = text.trim().trim_end_matches('#').trim();
    let negative = t.starts_with('-');
    let body = t.trim_start_matches(['+', '-']);
    let idx = body.find(['*', '°', ':'])?;
    let (d, rest) = body.split_at(idx);
    let d: f64 = d.trim().parse().ok()?;
    let rest = &rest[rest.chars().next()?.len_utf8()..];
    let (m, s) = match rest.split_once([':', '\'']) {
        Some((m, s)) => (
            m.trim().parse::<f64>().ok()?,
            s.trim().trim_end_matches('"').parse::<f64>().ok().unwrap_or(0.0),
        ),
        None => (rest.trim().parse::<f64>().ok()?, 0.0),
    };
    let deg = d.abs() + m / 60.0 + s / 3600.0;
    if !deg.is_finite() {
        return None;
    }
    Some(if negative { -deg } else { deg })
}

/// Great-circle separation between two equatorial positions, in degrees.
///
/// Used to sanity-check a return-to-mark before it moves the telescope: a mark
/// taken hours ago on a different target would otherwise slew across the sky on
/// a one-click button.
fn angular_separation_deg(ra1_hours: f64, dec1_deg: f64, ra2_hours: f64, dec2_deg: f64) -> f64 {
    let (d1, d2) = (dec1_deg.to_radians(), dec2_deg.to_radians());
    let dra = ((ra1_hours - ra2_hours) * 15.0).to_radians();
    let cos_sep = d1.sin() * d2.sin() + d1.cos() * d2.cos() * dra.cos();
    cos_sep.clamp(-1.0, 1.0).acos().to_degrees()
}

fn goto_error(response: &str) -> String {
    let detail = match response.trim_start_matches('e').parse::<u8>().ok() {
        Some(1) => "target parameter is out of range",
        Some(2) => "target parameter format was rejected",
        Some(3) => "mount is already homing, slewing, or performing a GoTo",
        Some(4) => "mount is already moving",
        Some(5) => "Sun is below the horizon",
        Some(6) => "Sun is below the configured altitude limit",
        Some(7) => {
            "mount time and location have not been synchronized — use Observing site → Sync now"
        }
        Some(8) => "mount has passed its tracking meridian limit",
        Some(9) => "sync target and mount are on opposite sides of the meridian",
        Some(10) => "alt-azimuth altitude would reverse",
        Some(11) => "sync is not allowed in the polar region",
        Some(12) => "sync target is too far from the mount position",
        _ => "unknown mount error",
    };
    format!("GoTo rejected ({response}): {detail}")
}

fn expect_ack(port: &mut dyn SerialPort, command: &str, field: &str) -> Result<(), String> {
    let response = transaction(port, command, true)?;
    if response.starts_with('1') {
        Ok(())
    } else {
        Err(format!("mount rejected target {field}"))
    }
}

fn query(port: &mut dyn SerialPort, command: &str) -> Result<String, String> {
    transaction(port, command, false)
}

fn blind(port: &mut dyn SerialPort, command: &str) -> Result<(), String> {
    port.write_all(command.as_bytes())
        .map_err(|error| format!("write {command}: {error}"))?;
    port.flush()
        .map_err(|error| format!("flush {command}: {error}"))
}

fn transaction(
    port: &mut dyn SerialPort,
    command: &str,
    single_byte: bool,
) -> Result<String, String> {
    let _ = port.clear(ClearBuffer::Input);
    blind(port, command)?;

    let start = Instant::now();
    let mut bytes = Vec::with_capacity(32);
    let mut buffer = [0_u8; 64];
    while start.elapsed() < IO_DEADLINE {
        match port.read(&mut buffer) {
            Ok(count) if count > 0 => {
                bytes.extend_from_slice(&buffer[..count]);
                if single_byte || bytes.contains(&b'#') {
                    break;
                }
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {
                if !bytes.is_empty() {
                    break;
                }
            }
            Err(error) => return Err(format!("read {command}: {error}")),
        }
    }
    let end = bytes
        .iter()
        .position(|byte| *byte == b'#')
        .unwrap_or(bytes.len());
    let response = String::from_utf8_lossy(&bytes[..end]).trim().to_owned();
    if response.is_empty() {
        Err(format!("no response to {command}"))
    } else {
        Ok(response)
    }
}

#[derive(Clone, Copy)]
struct SunPosition {
    ra_hours: f64,
    dec_deg: f64,
}

fn sun_equatorial_now() -> SunPosition {
    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    sun_equatorial_at_unix(unix_seconds)
}

fn sun_equatorial_at_unix(unix_seconds: f64) -> SunPosition {
    let julian_date = unix_seconds / 86_400.0 + 2_440_587.5;
    let n = julian_date - 2_451_545.0;
    let mean_longitude = (280.460 + 0.985_647_4 * n).rem_euclid(360.0);
    let mean_anomaly = (357.528 + 0.985_600_3 * n).to_radians();
    let ecliptic_longitude =
        (mean_longitude + 1.915 * mean_anomaly.sin() + 0.020 * (2.0 * mean_anomaly).sin())
            .to_radians();
    let obliquity = (23.439 - 0.000_000_4 * n).to_radians();
    let ra = (obliquity.cos() * ecliptic_longitude.sin())
        .atan2(ecliptic_longitude.cos())
        .to_degrees()
        .rem_euclid(360.0)
        / 15.0;
    let dec = (obliquity.sin() * ecliptic_longitude.sin())
        .asin()
        .to_degrees();
    SunPosition {
        ra_hours: ra,
        dec_deg: dec,
    }
}

fn local_sidereal_hours_at_unix(unix_seconds: f64, longitude_deg: f64) -> f64 {
    let julian_date = unix_seconds / 86_400.0 + 2_440_587.5;
    let days_since_j2000 = julian_date - 2_451_545.0;
    let centuries = days_since_j2000 / 36_525.0;
    let gmst_deg = 280.460_618_37
        + 360.985_647_366_29 * days_since_j2000
        + 0.000_387_933 * centuries * centuries
        - centuries * centuries * centuries / 38_710_000.0;
    ((gmst_deg + longitude_deg) / 15.0).rem_euclid(24.0)
}

fn sun_hour_angle_hours_at_unix(unix_seconds: f64, longitude_deg: f64) -> f64 {
    let sun = sun_equatorial_at_unix(unix_seconds);
    let lst = local_sidereal_hours_at_unix(unix_seconds, longitude_deg);
    (lst - sun.ra_hours + 12.0).rem_euclid(24.0) - 12.0
}

fn format_ra(hours: f64) -> String {
    let total = (hours.rem_euclid(24.0) * 3600.0).round() as i64 % 86_400;
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        total / 60 % 60,
        total % 60
    )
}

fn format_dec(degrees: f64) -> String {
    let sign = if degrees.is_sign_negative() { '-' } else { '+' };
    let total = (degrees.abs().min(90.0) * 3600.0).round() as i64;
    format!(
        "{sign}{:02}*{:02}:{:02}",
        total / 3600,
        total / 60 % 60,
        total % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jog_distance_matches_the_rate_table() {
        // 60x sidereal is 15.041 * 60 arcsec/s = ~15.04 arcmin/s, so a two
        // second jog is about one solar diameter. That relationship is the
        // whole reason the default is 60x.
        let two_sec_at_60x = jog_arcmin(60.0, 2.0);
        assert!(
            (two_sec_at_60x - 30.08).abs() < 0.1,
            "{two_sec_at_60x} arcmin"
        );
        assert!(two_sec_at_60x > 25.0 && two_sec_at_60x < 40.0, "~1 solar diameter");
        // Linear in both arguments.
        assert!((jog_arcmin(20.0, 6.0) - jog_arcmin(60.0, 2.0)).abs() < 1e-9);
        assert!((jog_arcmin(4.0, 1.0) * 2.0 - jog_arcmin(4.0, 2.0)).abs() < 1e-9);
        // A slow hold-to-jog rate barely moves: 1x sidereal for a second is
        // 15 arcsec, a quarter of an arcminute.
        assert!((jog_arcmin(1.0, 1.0) - 0.2507).abs() < 1e-3);
    }

    #[test]
    fn rate_table_multiples_agree_with_their_labels() {
        for (label, _, multiple) in RATES {
            let parsed: f64 = label.trim_end_matches('x').parse().unwrap();
            assert!(
                (parsed - multiple).abs() < 1e-9,
                "{label} carries multiple {multiple}"
            );
        }
    }

    #[test]
    fn parses_high_and_low_precision_right_ascension() {
        let hp = parse_ra_hours("12:34:56").unwrap();
        assert!((hp - (12.0 + 34.0 / 60.0 + 56.0 / 3600.0)).abs() < 1e-9, "{hp}");
        // Low precision: the fraction is tenths of a MINUTE, not seconds.
        let lp = parse_ra_hours("12:34.5").unwrap();
        assert!((lp - (12.0 + 34.5 / 60.0)).abs() < 1e-9, "{lp}");
        // The mount's own reply may still carry its terminator.
        assert!((parse_ra_hours("12:34:56#").unwrap() - hp).abs() < 1e-9);
        assert!(parse_ra_hours("garbage").is_none());
    }

    #[test]
    fn parses_declination_with_the_sign_applied_to_the_whole_value() {
        let north = parse_dec_deg("+12*34:56").unwrap();
        assert!((north - (12.0 + 34.0 / 60.0 + 56.0 / 3600.0)).abs() < 1e-9, "{north}");
        // The trap: negating only the degree field would give -0 + 30/60 =
        // +0.5°, putting the target on the wrong side of the equator.
        let just_south = parse_dec_deg("-00*30:00").unwrap();
        assert!((just_south + 0.5).abs() < 1e-9, "{just_south}");
        let south = parse_dec_deg("-12*34").unwrap();
        assert!((south + (12.0 + 34.0 / 60.0)).abs() < 1e-9, "{south}");
        // Some firmwares use a degree glyph rather than '*'.
        assert!((parse_dec_deg("+12°34:56").unwrap() - north).abs() < 1e-9);
        assert!(parse_dec_deg("nonsense").is_none());
    }

    #[test]
    fn coordinate_parse_round_trips_through_the_formatters() {
        for (ra, dec) in [(0.0, 0.0), (12.5827, -23.4561), (23.999, 89.5), (6.25, -0.25)] {
            let ra_back = parse_ra_hours(&format_ra(ra)).unwrap();
            let dec_back = parse_dec_deg(&format_dec(dec)).unwrap();
            // The formatters round to whole seconds/arcseconds.
            assert!((ra_back - ra).abs() < 1.0 / 3600.0 + 1e-9, "RA {ra} -> {ra_back}");
            assert!(
                (dec_back - dec).abs() < 1.0 / 3600.0 + 1e-9,
                "Dec {dec} -> {dec_back}"
            );
        }
    }

    #[test]
    fn angular_separation_matches_known_geometry() {
        // Same point.
        assert!(angular_separation_deg(5.0, 20.0, 5.0, 20.0) < 1e-9);
        // One hour of RA at the equator is exactly 15 degrees.
        let h = angular_separation_deg(0.0, 0.0, 1.0, 0.0);
        assert!((h - 15.0).abs() < 1e-9, "{h}");
        // Pure declination difference.
        let d = angular_separation_deg(3.0, 10.0, 3.0, -5.0);
        assert!((d - 15.0).abs() < 1e-9, "{d}");
        // RA separation shrinks with the cosine of declination.
        let high = angular_separation_deg(0.0, 60.0, 1.0, 60.0);
        assert!(high < 15.0 && high > 7.0, "{high}");
    }

    #[test]
    fn a_small_return_stays_under_the_confirmation_threshold() {
        // A jog of a few arcminutes -- the case the return button exists for --
        // must not trip the slew confirmation.
        let sep = angular_separation_deg(12.0, 20.0, 12.0 + 5.0 / 60.0 / 15.0, 20.05);
        assert!(sep < RETURN_CONFIRM_DEG, "{sep}");
        // A stale mark on another target must trip it.
        let stale = angular_separation_deg(12.0, 20.0, 14.0, 35.0);
        assert!(stale > RETURN_CONFIRM_DEG, "{stale}");
    }

    #[test]
    fn formats_mount_coordinates_with_carry() {
        assert_eq!(format_ra(23.999_999_9), "00:00:00");
        assert_eq!(format_ra(5.5), "05:30:00");
        assert_eq!(format_dec(-12.5), "-12*30:00");
        assert_eq!(format_dec(90.5), "+90*00:00");
    }

    #[test]
    fn solar_declination_tracks_the_seasons() {
        // 2024-03-20 12:00 UTC and 2024-06-20 12:00 UTC.
        let march = sun_equatorial_at_unix(1_710_936_000.0);
        let june = sun_equatorial_at_unix(1_718_884_800.0);
        assert!(
            march.dec_deg.abs() < 1.0,
            "March declination {}",
            march.dec_deg
        );
        assert!(
            (june.dec_deg - 23.44).abs() < 0.5,
            "June declination {}",
            june.dec_deg
        );
        assert!((0.0..24.0).contains(&march.ra_hours));
    }

    #[test]
    fn zwo_rate_and_goto_error_tables_match_protocol_v2_1() {
        assert_eq!(RATES.first(), Some(&("0.25x", 0, 0.25)));
        assert_eq!(RATES.last(), Some(&("1440x", 9, 1440.0)));
        assert!(goto_error("e5").contains("below the horizon"));
        assert!(goto_error("e7").contains("time and location"));
    }

    #[test]
    fn square_spiral_covers_chebyshev_disk_once() {
        let pts = square_spiral(2);
        assert_eq!(pts[0], (0, 0));
        assert_eq!(pts.len(), 25); // (2*2+1)^2
        let mut seen = std::collections::BTreeSet::new();
        for p in &pts {
            assert!(p.0.abs().max(p.1.abs()) <= 2);
            assert!(seen.insert(*p), "duplicate {p:?}");
        }
        assert_eq!(seen.len(), 25);
    }

    #[test]
    fn spiral_nudge_duration_matches_0_1_deg_at_60x() {
        // 60× × 15″/s = 0.25 °/s → 0.1° takes 400 ms.
        assert_eq!(spiral_nudge_duration_ms(), 400);
    }

    #[test]
    fn coarse_spiral_uses_two_fine_units_per_point() {
        let pts = coarse_spiral(1);
        assert_eq!(pts.len(), 9);
        assert!(pts.iter().all(|(x, y)| x % 2 == 0 && y % 2 == 0));
        assert!(pts.contains(&(2, 0)));
    }

    #[test]
    fn refinement_grid_is_local_and_respects_radius() {
        let middle = refinement_grid((2, -1), 4);
        assert_eq!(middle.len(), 9);
        assert!(middle.iter().all(|(x, y)| {
            (x - 2).abs() <= 1 && (y + 1).abs() <= 1
        }));

        let edge = refinement_grid((4, 4), 4);
        assert!(edge.len() < 9);
        assert!(edge
            .iter()
            .all(|(x, y)| x.abs() <= 4 && y.abs() <= 4));
    }

    #[test]
    fn grid_step_prefers_east_west_then_north_south() {
        let (dir, next) = grid_step_direction((0, 0), (2, 1)).unwrap();
        assert_eq!(dir, Direction::East);
        assert_eq!(next, (1, 0));
        let (dir, next) = grid_step_direction((1, 0), (1, -2)).unwrap();
        assert_eq!(dir, Direction::South);
        assert_eq!(next, (1, -1));
        assert!(grid_step_direction((1, 1), (1, 1)).is_none());
    }

    #[test]
    fn timed_motion_stops_only_its_requested_direction() {
        assert_eq!(Direction::North.stop_command(), ":Qn#");
        assert_eq!(Direction::South.stop_command(), ":Qs#");
        assert_eq!(Direction::East.stop_command(), ":Qe#");
        assert_eq!(Direction::West.stop_command(), ":Qw#");
    }

    #[test]
    fn meridian_estimate_matches_blackpool_solar_noon() {
        // 2026-07-30 13:18 BST, independently published as local solar noon.
        let unix_seconds = 1_785_413_880.0;
        let hour_angle = sun_hour_angle_hours_at_unix(unix_seconds, -2.989_722);
        assert!(
            hour_angle.abs() < 2.0 / 60.0,
            "solar hour angle was {hour_angle:.4} h"
        );
    }

    #[test]
    fn lx200_site_formats_match_meade_style() {
        assert_eq!(format_lat_lx200(51.5), "+51*30");
        assert_eq!(format_lat_lx200(-33.9), "-33*54");
        // 2.3°E → 357.7°W
        assert_eq!(format_lon_lx200_west(2.3), "357*42");
        // 98°W stored as −98 east-positive → 98°W
        assert_eq!(format_lon_lx200_west(-98.0), "098*00");
        // Local = UTC−5 → Meade SG = +05 (add to local to get UTC)
        assert_eq!(format_meade_utc_offset(-5.0), "+05");
        assert_eq!(format_meade_utc_offset(1.0), "-01");
    }

    #[test]
    fn live_nominatim_blackpool_does_not_panic() {
        let result = std::panic::catch_unwind(|| nominatim_search("Blackpool"));
        assert!(
            result.is_ok(),
            "nominatim_search panicked (TLS agent misconfigured?)"
        );
        let result = result.unwrap();
        assert!(
            result.is_ok(),
            "nominatim error: {:?}",
            result.as_ref().err()
        );
        let hits = result.unwrap();
        assert!(
            hits.iter()
                .any(|(n, _, _)| n.to_lowercase().contains("blackpool")),
            "unexpected hits: {hits:?}"
        );
    }

    #[test]
    fn nominatim_json_scrapes_display_name_and_coords() {
        let body = r#"[{"lat":"51.5074","lon":"-0.1278","display_name":"London, UK"}]"#;
        let hits = parse_nominatim_json(body).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].0.contains("London"));
        assert!((hits[0].1 - 51.5074).abs() < 1e-6);
        assert!((hits[0].2 + 0.1278).abs() < 1e-6);
    }

    #[test]
    fn truncate_chars_is_utf8_safe() {
        let s = format!("{}€{}", "a".repeat(70), "b".repeat(10));
        let t = truncate_chars(&s, 72);
        assert!(t.ends_with('…'));
        assert!(t.is_char_boundary(t.len()));
        assert!(std::panic::catch_unwind(|| truncate_chars(&s, 72)).is_ok());
    }
}
