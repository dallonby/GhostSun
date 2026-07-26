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

const RATES: [(&str, u8); 10] = [
    ("0.25x", 0),
    ("0.5x", 1),
    ("1x", 2),
    ("2x", 3),
    ("4x", 4),
    ("8x", 5),
    ("20x", 6),
    ("60x", 7),
    ("720x", 8),
    ("1440x", 9),
];

#[derive(Clone)]
struct PortInfo {
    name: String,
    detail: String,
    is_zwo: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
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

impl Direction {
    fn move_command(self) -> &'static str {
        match self {
            Direction::North => ":Mn#",
            Direction::South => ":Ms#",
            Direction::East => ":Me#",
            Direction::West => ":Mw#",
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
    GoHome,
    Park,
    Unpark,
    Nudge {
        direction: Direction,
        duration_ms: u64,
    },
    SlewSun { ra_hours: f64, dec_deg: f64 },
    Shutdown,
}

enum WorkerMessage {
    Connected { port: String, model: String },
    Disconnected(String),
    Snapshot(MountSnapshot),
    Notice(String),
    NudgeDone,
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

struct AutoCenterState {
    restore: focus::SearchCameraRestore,
    points: Vec<(i32, i32)>,
    point_index: usize,
    current: (i32, i32),
    best: (i32, i32),
    best_signal: f32,
    duration_ms: u64,
    phase: AutoCenterPhase,
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
    confirm_motion: Option<ConfirmedMotion>,
    confirm_sun: bool,
    confirm_auto_center: bool,
    auto_center: Option<AutoCenterState>,
    search_exposure_ms: u32,
    search_radius_deg: f32,
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
            confirm_motion: None,
            confirm_sun: false,
            confirm_auto_center: false,
            auto_center: None,
            search_exposure_ms: 250,
            search_radius_deg: 0.6,
            status: "Not connected".into(),
            last_scan: Instant::now() - SCAN_INTERVAL,
            last_poll: Instant::now(),
            poll_inflight: false,
            tx: command_tx,
            rx: message_rx,
            worker: Some(worker),
        };
        state.refresh_ports();
        state
    }
}

impl MountState {
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

    pub fn enter_tab(&mut self, focus: &mut focus::FocusState) {
        self.refresh_ports();
        if focus.cameras.is_empty() {
            focus.refresh_cameras();
        }
        self.last_poll = Instant::now() - POLL_INTERVAL;
    }

    pub fn leave_tab(&mut self, focus: &mut focus::FocusState) {
        self.cancel_auto_center(focus, "Sun auto-center cancelled");
        self.stop_motion();
        self.confirm_motion = None;
        self.confirm_sun = false;
        self.confirm_auto_center = false;
    }

    pub fn poll(&mut self, ctx: &egui::Context, focus: &mut focus::FocusState) {
        while let Ok(message) = self.rx.try_recv() {
            match message {
                WorkerMessage::Connected { port, model } => {
                    self.connected = true;
                    self.connecting = false;
                    self.connected_port = Some(port.clone());
                    self.model = Some(model.clone());
                    self.status = format!("Connected to {model} on {port}");
                    self.last_poll = Instant::now() - POLL_INTERVAL;
                }
                WorkerMessage::Disconnected(reason) => {
                    self.cancel_auto_center(focus, "Sun auto-center stopped: mount disconnected");
                    self.connected = false;
                    self.connecting = false;
                    self.connected_port = None;
                    self.active_direction = None;
                    self.poll_inflight = false;
                    self.status = reason;
                }
                WorkerMessage::Snapshot(snapshot) => {
                    self.snapshot = snapshot;
                    self.poll_inflight = false;
                }
                WorkerMessage::Notice(notice) => self.status = notice,
                WorkerMessage::NudgeDone => self.auto_center_nudge_done(),
                WorkerMessage::Error(error) => {
                    self.cancel_auto_center(focus, "Sun auto-center stopped by mount error");
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
        ctx.request_repaint_after(Duration::from_millis(50));
    }

    pub fn controls_ui(
        &mut self,
        ui: &mut egui::Ui,
        focus: &mut focus::FocusState,
    ) {
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
                "Uses a 0.2° square spiral at 60×, averages three frames per point, then returns to the strongest signal.",
            )
            .small()
            .weak(),
        );

        ui.add_space(12.0);
        ui.heading("Jog rate");
        let old_rate = self.rate_index;
        egui::ComboBox::from_id_salt("mount_rate")
            .selected_text(RATES[self.rate_index].0)
            .show_ui(ui, |ui| {
                for (index, (label, _)) in RATES.iter().enumerate() {
                    ui.selectable_value(&mut self.rate_index, index, *label);
                }
            });
        if self.rate_index != old_rate && self.connected {
            self.stop_motion();
            let _ = self
                .tx
                .send(WorkerCommand::SetRate(RATES[self.rate_index].1));
        }
        ui.label(
            egui::RichText::new("Rate changes stop any active jog first.")
                .small()
                .weak(),
        );

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

        ui.add_space(10.0);
        ui.label(
            egui::RichText::new(
                "Before Sun GoTo, set the mount's time, location, home position, and alignment in ASI Mount.",
            )
            .small()
            .color(egui::Color32::from_rgb(255, 190, 100)),
        );
    }

    pub fn view_ui(
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
                    "Connected - hold a direction to jog; release to stop"
                } else {
                    "Connect to the ZWO mount in the left panel"
                })
                .weak(),
            );
        });

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

        ui.add_space(20.0);
        let enabled = self.connected;
        let mut held = None;
        ui.vertical_centered(|ui| {
            let north = ui.add_enabled(
                enabled,
                egui::Button::new(egui::RichText::new("N").size(20.0).strong())
                    .min_size(egui::vec2(86.0, 52.0)),
            );
            if north.is_pointer_button_down_on() {
                held = Some(Direction::North);
            }
            ui.horizontal_centered(|ui| {
                let west = ui.add_enabled(
                    enabled,
                    egui::Button::new(egui::RichText::new("W").size(20.0).strong())
                        .min_size(egui::vec2(86.0, 52.0)),
                );
                if west.is_pointer_button_down_on() {
                    held = Some(Direction::West);
                }

                let stop = egui::Button::new(
                    egui::RichText::new("STOP")
                        .size(18.0)
                        .strong()
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(160, 35, 25))
                .min_size(egui::vec2(112.0, 58.0));
                if ui.add(stop).clicked() {
                    self.stop_motion();
                    held = None;
                }

                let east = ui.add_enabled(
                    enabled,
                    egui::Button::new(egui::RichText::new("E").size(20.0).strong())
                        .min_size(egui::vec2(86.0, 52.0)),
                );
                if east.is_pointer_button_down_on() {
                    held = Some(Direction::East);
                }
            });
            let south = ui.add_enabled(
                enabled,
                egui::Button::new(egui::RichText::new("S").size(20.0).strong())
                    .min_size(egui::vec2(86.0, 52.0)),
            );
            if south.is_pointer_button_down_on() {
                held = Some(Direction::South);
            }
            ui.label(
                egui::RichText::new(format!("Selected jog rate: {}", RATES[self.rate_index].0))
                    .small()
                    .weak(),
            );
        });
        self.update_held_direction(held);

        ui.add_space(10.0);
        ui.vertical_centered(|ui| {
            ui.horizontal_centered(|ui| {
                if ui
                    .add_enabled(enabled, egui::Button::new("Go Home"))
                    .clicked()
                {
                    self.confirm_sun = false;
                    self.confirm_motion = Some(ConfirmedMotion::GoHome);
                }
                let equatorial = self
                    .snapshot
                    .flags
                    .as_ref()
                    .map(|flags| !flags.contains('Z'))
                    .unwrap_or(true);
                if ui
                    .add_enabled(enabled && equatorial, egui::Button::new("Park"))
                    .clicked()
                {
                    self.confirm_sun = false;
                    self.confirm_motion = Some(ConfirmedMotion::Park);
                }
                if ui
                    .add_enabled(enabled, egui::Button::new("Unpark"))
                    .clicked()
                {
                    self.stop_motion();
                    let _ = self.tx.send(WorkerCommand::Unpark);
                    self.status = "Requesting unpark...".into();
                }
            });

            if let Some(action) = self.confirm_motion {
                let (title, detail, command) = match action {
                    ConfirmedMotion::GoHome => (
                        "Confirm Go Home",
                        "The mount will move both axes to its mechanical home position.",
                        WorkerCommand::GoHome,
                    ),
                    ConfirmedMotion::Park => (
                        "Confirm Park",
                        "The mount will move to its configured park position (equatorial mode only).",
                        WorkerCommand::Park,
                    ),
                };
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(42, 28, 20))
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new(detail).strong());
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    enabled,
                                    egui::Button::new(title).fill(ACCENT_DIM),
                                )
                                .clicked()
                            {
                                self.stop_motion();
                                let _ = self.tx.send(command);
                                self.status = format!("{title} sent...");
                                self.confirm_motion = None;
                            }
                            if ui.button("Cancel").clicked() {
                                self.confirm_motion = None;
                            }
                        });
                    });
            }
        });

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
                    enabled,
                    egui::Button::new(
                        egui::RichText::new("Prepare Sun GoTo")
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(ACCENT_DIM),
                )
                .clicked()
            {
                self.confirm_sun = true;
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
    let mut timed_move_deadline: Option<Instant> = None;

    loop {
        if timed_move_deadline
            .map(|deadline| Instant::now() >= deadline)
            .unwrap_or(false)
        {
            if let Some(opened) = port.as_deref_mut() {
                let _ = blind(opened, ":Q#");
            }
            timed_move_deadline = None;
            let _ = tx.send(WorkerMessage::NudgeDone);
        }
        let wait = timed_move_deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::from_millis(100))
            .min(Duration::from_millis(100));
        match rx.recv_timeout(wait) {
            Ok(WorkerCommand::Connect(name)) => {
                timed_move_deadline = None;
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
                timed_move_deadline = None;
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
                    let park = query(opened, ":Gps#").ok().map(|value| match value.as_str() {
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
                timed_move_deadline = None;
                if let Some(opened) = port.as_deref_mut() {
                    if let Err(error) = blind(opened, ":Q#") {
                        let _ = tx.send(WorkerMessage::Error(error));
                    }
                }
            }
            Ok(WorkerCommand::GoHome) => {
                timed_move_deadline = None;
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
                timed_move_deadline = None;
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
                            let _ =
                                tx.send(WorkerMessage::Notice("Mount is unparked".into()));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::Nudge {
                direction,
                duration_ms,
            }) => {
                timed_move_deadline = None;
                if let Some(opened) = port.as_deref_mut() {
                    let result = blind(opened, ":Q#")
                        .and_then(|_| blind(opened, ":R7#"))
                        .and_then(|_| blind(opened, direction.move_command()));
                    match result {
                        Ok(()) => {
                            timed_move_deadline =
                                Some(Instant::now() + Duration::from_millis(duration_ms));
                        }
                        Err(error) => {
                            let _ = tx.send(WorkerMessage::Error(error));
                        }
                    }
                }
            }
            Ok(WorkerCommand::SlewSun { ra_hours, dec_deg }) => {
                timed_move_deadline = None;
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
                timed_move_deadline = None;
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
    blind(port, ":Q#")?;
    expect_ack(
        port,
        &format!(":Sr{}#", format_ra(ra_hours)),
        "right ascension",
    )?;
    expect_ack(port, &format!(":Sd{}#", format_dec(dec_deg)), "declination")?;
    // A successful ZWO GoTo enables tracking on arrival. Select the solar
    // rate first so a separate tracking command cannot race the active slew.
    blind(port, ":TS#")?;
    let response = transaction(port, ":MS#", false)?;
    match response.chars().next() {
        Some('0') => Ok(()),
        _ if response.starts_with('e') => Err(goto_error(&response)),
        _ => Err(format!("mount rejected GoTo: {response:?}")),
    }
}

fn goto_error(response: &str) -> String {
    let detail = match response.trim_start_matches('e').parse::<u8>().ok() {
        Some(1) => "target parameter is out of range",
        Some(2) => "target parameter format was rejected",
        Some(3) => "mount is already homing, slewing, or performing a GoTo",
        Some(4) => "mount is already moving",
        Some(5) => "Sun is below the horizon",
        Some(6) => "Sun is below the configured altitude limit",
        Some(7) => "mount time and location have not been synchronized",
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
        assert_eq!(RATES.first(), Some(&("0.25x", 0)));
        assert_eq!(RATES.last(), Some(&("1440x", 9)));
        assert!(goto_error("e5").contains("below the horizon"));
        assert!(goto_error("e7").contains("time and location"));
    }
}
