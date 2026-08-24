//! Offline review: browse past sessions, see what the seeing gave you, and
//! stack only the scans worth stacking.
//!
//! Capture and reconstruction are deliberately separate (see `acquire`). This
//! is the other half: point at a folder, get a FAST look at every scan in it,
//! and choose. The quick pass exists to answer "which of these is sharp?" —
//! not to make a final image — so every stage that costs minutes is off.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use ghostsun_core::image2d::Image;
use ghostsun_core::{metrics, pipeline, ser};

use crate::acquire::{process_scans, AcquireOutput, ProcessMessage};

const ACCENT: egui::Color32 = egui::Color32::from_rgb(255, 205, 75);
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(120, 90, 30);
/// Long edge of the cached preview. Big enough to judge detail in the enlarged
/// view, small enough that a folder of thirty scans is not a texture problem.
const PREVIEW_PX: usize = 384;
const TILE_PX: f32 = 190.0;

/// Where a scan has got to in the quick pass.
#[derive(Clone)]
enum TileState {
    Queued,
    /// Running, carrying the pipeline's latest stage line.
    Working(String),
    Ready,
    Failed(String),
}

/// One scan, and what the quick pass learned about it.
struct Tile {
    path: PathBuf,
    session: String,
    label: String,
    frames: usize,
    captured: String,
    reverse: bool,
    /// Limb sharpness in px. LOWER IS SHARPER: it is the width of the limb
    /// edge, so good seeing gives a small number.
    sharpness: Option<f64>,
    gray: Option<(Vec<u8>, usize, usize)>,
    tex: Option<egui::TextureHandle>,
    state: TileState,
    selected: bool,
}

enum PreviewMsg {
    Started {
        index: usize,
    },
    Step {
        index: usize,
        line: String,
    },
    Preview {
        index: usize,
        gray: Vec<u8>,
        w: usize,
        h: usize,
        sharpness: Option<f64>,
    },
    Failed {
        index: usize,
        error: String,
    },
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortBy {
    Sharpness,
    Captured,
}

pub struct ProcessState {
    root: Option<PathBuf>,
    tiles: Vec<Tile>,
    order: Vec<usize>,
    sort: SortBy,
    preview_rx: Receiver<PreviewMsg>,
    preview_tx: Sender<PreviewMsg>,
    previewing: bool,
    previewed: usize,
    /// Seconds each finished scan took, so the remaining wait can be quoted
    /// from measurement rather than guessed.
    per_file: Vec<f64>,
    file_started: Option<Instant>,
    cancel: Arc<AtomicBool>,
    stack_rx: Receiver<ProcessMessage>,
    stack_tx: Sender<ProcessMessage>,
    stacking: bool,
    /// Selection has been touched, so the automatic pre-tick must not stomp it.
    user_picked: bool,
    enlarged: Option<usize>,
    status: String,
    log: Vec<String>,
}

impl Default for ProcessState {
    fn default() -> Self {
        let (preview_tx, preview_rx) = channel();
        let (stack_tx, stack_rx) = channel();
        Self {
            root: None,
            tiles: Vec::new(),
            order: Vec::new(),
            sort: SortBy::Sharpness,
            preview_rx,
            preview_tx,
            previewing: false,
            previewed: 0,
            per_file: Vec::new(),
            file_started: None,
            cancel: Arc::new(AtomicBool::new(false)),
            stack_rx,
            stack_tx,
            stacking: false,
            user_picked: false,
            enlarged: None,
            status: "Choose a folder of captures to review".into(),
            log: Vec::new(),
        }
    }
}

impl ProcessState {
    pub fn poll(&mut self, ctx: &egui::Context) -> Option<AcquireOutput> {
        while let Ok(message) = self.preview_rx.try_recv() {
            match message {
                PreviewMsg::Started { index } => {
                    self.file_started = Some(Instant::now());
                    if let Some(tile) = self.tiles.get_mut(index) {
                        tile.state = TileState::Working("starting".into());
                    }
                }
                PreviewMsg::Step { index, line } => {
                    if let Some(tile) = self.tiles.get_mut(index) {
                        tile.state = TileState::Working(line);
                    }
                }
                PreviewMsg::Preview {
                    index,
                    gray,
                    w,
                    h,
                    sharpness,
                } => {
                    if let Some(tile) = self.tiles.get_mut(index) {
                        tile.gray = Some((gray, w, h));
                        tile.sharpness = sharpness;
                        tile.state = TileState::Ready;
                    }
                    self.finish_one();
                }
                PreviewMsg::Failed { index, error } => {
                    if let Some(tile) = self.tiles.get_mut(index) {
                        tile.state = TileState::Failed(error);
                    }
                    self.finish_one();
                }
                PreviewMsg::Done => {
                    self.previewing = false;
                    self.file_started = None;
                    self.resort();
                    self.pretick();
                    self.status = format!(
                        "{} scans reviewed — {} selected",
                        self.tiles.len(),
                        self.selected_count()
                    );
                }
            }
            ctx.request_repaint();
        }

        let mut finished = None;
        while let Ok(message) = self.stack_rx.try_recv() {
            match message {
                ProcessMessage::Log(line) => {
                    self.status = line.clone();
                    self.log.push(line);
                }
                ProcessMessage::Done(output) => {
                    self.stacking = false;
                    self.status = "Stack complete".into();
                    self.log.push(self.status.clone());
                    finished = Some(output);
                }
                ProcessMessage::Failed(error) => {
                    self.stacking = false;
                    self.status = format!("Stacking failed: {error}");
                    self.log.push(self.status.clone());
                }
            }
            ctx.request_repaint();
        }
        finished
    }

    /// One scan done: bank how long it took, and restate the wait.
    fn finish_one(&mut self) {
        if let Some(start) = self.file_started.take() {
            self.per_file.push(start.elapsed().as_secs_f64());
        }
        self.previewed += 1;
        self.status = match self.remaining_estimate() {
            Some(text) => format!(
                "Quick look: {}/{} scans · about {text} left",
                self.previewed,
                self.tiles.len()
            ),
            None => format!("Quick look: {}/{} scans", self.previewed, self.tiles.len()),
        };
    }

    /// Time left, from the mean of the scans already done.
    ///
    /// `None` until at least one has finished: an estimate from no samples is
    /// worse than no estimate.
    fn remaining_estimate(&self) -> Option<String> {
        if self.per_file.is_empty() {
            return None;
        }
        let left = self.tiles.len().saturating_sub(self.previewed);
        if left == 0 {
            return None;
        }
        let mean = self.per_file.iter().sum::<f64>() / self.per_file.len() as f64;
        Some(human_duration(mean * left as f64))
    }

    fn progress_fraction(&self) -> f32 {
        if self.tiles.is_empty() {
            return 0.0;
        }
        self.previewed as f32 / self.tiles.len() as f32
    }

    fn selected_count(&self) -> usize {
        self.tiles.iter().filter(|t| t.selected).count()
    }

    /// Rank the tiles. Sharpest first, because that is the decision being made.
    fn resort(&mut self) {
        let mut order: Vec<usize> = (0..self.tiles.len()).collect();
        match self.sort {
            SortBy::Sharpness => order.sort_by(|&a, &b| {
                // Unmeasured scans sink: they are failures, not candidates.
                let ka = self.tiles[a].sharpness.unwrap_or(f64::INFINITY);
                let kb = self.tiles[b].sharpness.unwrap_or(f64::INFINITY);
                ka.total_cmp(&kb)
            }),
            SortBy::Captured => {
                order.sort_by(|&a, &b| self.tiles[a].captured.cmp(&self.tiles[b].captured))
            }
        }
        self.order = order;
    }

    /// Pre-select the better half, once, unless the user has already chosen.
    fn pretick(&mut self) {
        if self.user_picked {
            return;
        }
        let mut measured: Vec<usize> = (0..self.tiles.len())
            .filter(|&i| self.tiles[i].sharpness.is_some())
            .collect();
        measured.sort_by(|&a, &b| {
            self.tiles[a]
                .sharpness
                .unwrap_or(f64::INFINITY)
                .total_cmp(&self.tiles[b].sharpness.unwrap_or(f64::INFINITY))
        });
        let keep = measured.len().div_ceil(2);
        for tile in self.tiles.iter_mut() {
            tile.selected = false;
        }
        for &index in measured.iter().take(keep) {
            self.tiles[index].selected = true;
        }
    }

    /// Find every scan under `root`, one level deep as well as in `root` itself.
    fn load_folder(&mut self, root: PathBuf, ctx: &egui::Context) {
        self.cancel.store(true, Ordering::SeqCst);
        self.cancel = Arc::new(AtomicBool::new(false));
        self.tiles.clear();
        self.order.clear();
        self.enlarged = None;
        self.user_picked = false;
        self.previewed = 0;
        self.per_file.clear();
        self.file_started = None;

        let sessions = discover_sessions(&root);

        for session in &sessions {
            let name = session
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("session")
                .to_owned();
            for (index, path) in session_scans(session).into_iter().enumerate() {
                let (frames, captured) = ser_summary(&path);
                self.tiles.push(Tile {
                    label: path
                        .file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or("scan")
                        .to_owned(),
                    session: name.clone(),
                    path,
                    frames,
                    captured,
                    // Sweeps alternate, exactly as during the run.
                    reverse: index % 2 == 1,
                    sharpness: None,
                    gray: None,
                    tex: None,
                    state: TileState::Queued,
                    selected: false,
                });
            }
        }
        self.resort();
        self.root = Some(root);

        if self.tiles.is_empty() {
            self.previewing = false;
            self.status = "No scan-NN.ser files found in that folder or its subfolders".into();
            return;
        }

        let jobs: Vec<(usize, PathBuf, bool)> = self
            .tiles
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.path.clone(), t.reverse))
            .collect();
        let tx = self.preview_tx.clone();
        let cancel = self.cancel.clone();
        let repaint = ctx.clone();
        self.previewing = true;
        self.status = format!("Quick look: 0/{} scans", jobs.len());
        thread::spawn(move || {
            for (index, path, reverse) in jobs {
                if cancel.load(Ordering::SeqCst) {
                    break;
                }
                let _ = tx.send(PreviewMsg::Started { index });
                repaint.request_repaint();
                // Caught: one malformed SER must not take the whole review with it.
                let job_tx = tx.clone();
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    quick_preview(&path, reverse, index, job_tx)
                }));
                let message = match outcome {
                    Ok(Ok((gray, w, h, sharpness))) => PreviewMsg::Preview {
                        index,
                        gray,
                        w,
                        h,
                        sharpness,
                    },
                    Ok(Err(error)) => PreviewMsg::Failed { index, error },
                    Err(_) => PreviewMsg::Failed {
                        index,
                        error: "reconstruction panicked on this file".into(),
                    },
                };
                let _ = tx.send(message);
                repaint.request_repaint();
            }
            let _ = tx.send(PreviewMsg::Done);
            repaint.request_repaint();
        });
    }

    fn start_stack(&mut self, ctx: &egui::Context) {
        let files: Vec<(PathBuf, bool)> = self
            .order
            .iter()
            .filter_map(|&i| self.tiles.get(i))
            .filter(|t| t.selected)
            .map(|t| (t.path.clone(), t.reverse))
            .collect();
        if files.is_empty() {
            self.status = "Tick at least one scan first".into();
            return;
        }
        let Some(dir) = self.root.clone() else {
            return;
        };
        self.stacking = true;
        self.status = format!("Reconstructing and stacking {} scans", files.len());
        self.log.clear();
        self.log.push(self.status.clone());
        let tx = self.stack_tx.clone();
        let repaint = ctx.clone();
        thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                process_scans(&files, &dir, &tx)
            }));
            let message = match outcome {
                Ok(Ok(output)) => ProcessMessage::Done(output),
                Ok(Err(error)) => ProcessMessage::Failed(error),
                Err(_) => ProcessMessage::Failed("stacking panicked".into()),
            };
            let _ = tx.send(message);
            repaint.request_repaint();
        });
    }
}

// -- UI ---------------------------------------------------------------------

impl ProcessState {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Review & stack");
        ui.label(
            egui::RichText::new(
                "Point at a night's captures. Every scan gets a fast, plain \
                 reconstruction — no deconvolution or denoising — purely so you \
                 can see which ones the seeing was kind to. Tick those, then \
                 stack them at full quality.",
            )
            .small()
            .weak(),
        );
        ui.add_space(6.0);

        let busy = self.previewing || self.stacking;
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("Choose folder…"))
                .on_hover_text(
                    "A single session folder, or a parent holding several — \
                     both are scanned for scan-NN.ser files.",
                )
                .clicked()
            {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.load_folder(dir, ctx);
                }
            }
            if self.previewing && ui.button("Stop quick look").clicked() {
                self.cancel.store(true, Ordering::SeqCst);
            }
            if let Some(root) = &self.root {
                ui.label(
                    egui::RichText::new(root.display().to_string())
                        .small()
                        .monospace()
                        .weak(),
                );
            }
        });

        if self.tiles.is_empty() {
            ui.add_space(8.0);
            ui.label(egui::RichText::new(&self.status).strong());
            return;
        }

        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("sort:");
            let mut sort = self.sort;
            ui.selectable_value(&mut sort, SortBy::Sharpness, "sharpest first");
            ui.selectable_value(&mut sort, SortBy::Captured, "capture order");
            if sort != self.sort {
                self.sort = sort;
                self.resort();
            }
            ui.separator();
            if ui.button("all").clicked() {
                self.user_picked = true;
                self.tiles.iter_mut().for_each(|t| t.selected = t.gray.is_some());
            }
            if ui.button("none").clicked() {
                self.user_picked = true;
                self.tiles.iter_mut().for_each(|t| t.selected = false);
            }
            if ui
                .button("best half")
                .on_hover_text("Re-apply the automatic pick: the sharper half of the measured scans.")
                .clicked()
            {
                self.user_picked = false;
                self.pretick();
                self.user_picked = true;
            }
        });

        if self.previewing {
            ui.add_space(4.0);
            let done = self.previewed;
            let total = self.tiles.len();
            let label = match self.remaining_estimate() {
                Some(left) => format!("{done}/{total} · about {left} left"),
                None => format!("{done}/{total}"),
            };
            ui.add(
                egui::ProgressBar::new(self.progress_fraction())
                    .text(label)
                    .fill(ACCENT_DIM),
            )
            .on_hover_text(
                "Each scan gets a stripped-down reconstruction. The estimate is \
                 the mean of the ones already done, so it settles after a few.",
            );
        }

        ui.add_space(6.0);
        let selected = self.selected_count();
        ui.horizontal(|ui| {
            if self.stacking {
                ui.spinner();
            }
            ui.label(egui::RichText::new(&self.status).strong());
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !busy && selected > 0,
                    egui::Button::new(
                        egui::RichText::new(format!("Reconstruct & stack {selected} selected"))
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(ACCENT_DIM),
                )
                .on_disabled_hover_text(if busy {
                    "a job is already running"
                } else {
                    "tick at least one scan"
                })
                .clicked()
            {
                self.start_stack(ctx);
            }
            ui.label(
                egui::RichText::new("full pipeline, written beside the captures")
                    .small()
                    .weak(),
            );
        });

        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| {
            self.tile_grid(ui, ctx);
        });
        self.enlarged_window(ctx);
    }

    fn tile_grid(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let order = self.order.clone();
        let mut toggled = None;
        let mut enlarge = None;
        ui.horizontal_wrapped(|ui| {
            for index in order {
                let Some(tile) = self.tiles.get_mut(index) else {
                    continue;
                };
                // The texture is built here, on the UI thread, from the grey
                // bytes the worker produced.
                if tile.tex.is_none() {
                    if let Some((gray, w, h)) = &tile.gray {
                        let pixels = gray.iter().map(|&g| egui::Color32::from_gray(g)).collect();
                        tile.tex = Some(ctx.load_texture(
                            format!("preview_{index}"),
                            egui::ColorImage {
                                size: [*w, *h],
                                pixels,
                            },
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }
                let selected = tile.selected;
                let response = ui.allocate_ui(egui::vec2(TILE_PX, TILE_PX + 62.0), |ui| {
                    egui::Frame::none()
                        .stroke(egui::Stroke::new(
                            if selected { 2.0 } else { 1.0 },
                            if selected {
                                ACCENT
                            } else {
                                ui.visuals().widgets.noninteractive.bg_stroke.color
                            },
                        ))
                        .inner_margin(4.0)
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                let image_size = egui::vec2(TILE_PX - 10.0, TILE_PX - 10.0);
                                match &tile.tex {
                                    Some(tex) => {
                                        ui.add(
                                            egui::Image::new(tex)
                                                .fit_to_exact_size(image_size)
                                                .sense(egui::Sense::click()),
                                        );
                                    }
                                    None => {
                                        let (rect, _) = ui.allocate_exact_size(
                                            image_size,
                                            egui::Sense::click(),
                                        );
                                        let (caption, colour) = match &tile.state {
                                            TileState::Queued => (
                                                "queued".to_owned(),
                                                ui.visuals().weak_text_color(),
                                            ),
                                            TileState::Working(step) => {
                                                (step.clone(), ACCENT)
                                            }
                                            TileState::Failed(_) => (
                                                "failed".to_owned(),
                                                egui::Color32::from_rgb(255, 120, 120),
                                            ),
                                            TileState::Ready => (
                                                "…".to_owned(),
                                                ui.visuals().weak_text_color(),
                                            ),
                                        };
                                        if matches!(tile.state, TileState::Working(_)) {
                                            ui.painter().rect_filled(
                                                egui::Rect::from_min_max(
                                                    rect.left_top(),
                                                    egui::pos2(rect.right(), rect.top() + 3.0),
                                                ),
                                                0.0,
                                                ACCENT,
                                            );
                                        }
                                        // Stage names get long; wrap rather than
                                        // spill outside the tile.
                                        ui.painter().text(
                                            rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            ellipsize(&caption, 22),
                                            egui::FontId::proportional(12.0),
                                            colour,
                                        );
                                    }
                                }
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(if selected { "☑" } else { "☐" })
                                            .color(if selected { ACCENT } else { ui.visuals().weak_text_color() }),
                                    );
                                    ui.label(
                                        egui::RichText::new(&tile.label).small().strong(),
                                    );
                                    if tile.tex.is_some()
                                        && ui
                                            .small_button("⤢")
                                            .on_hover_text("Open a large view")
                                            .clicked()
                                    {
                                        enlarge = Some(index);
                                    }
                                });
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} · {} frames",
                                        tile.session, tile.frames
                                    ))
                                    .small()
                                    .weak(),
                                );
                                ui.label(
                                    egui::RichText::new(match tile.sharpness {
                                        // Stated as a width so the direction is
                                        // unambiguous: smaller edge = sharper.
                                        Some(s) => format!("{}  ·  limb {s:.2} px", tile.captured),
                                        None => tile.captured.clone(),
                                    })
                                    .small()
                                    .color(if tile.sharpness.is_some() { ACCENT } else { ui.visuals().weak_text_color() }),
                                );
                                if let TileState::Failed(error) = &tile.state {
                                    ui.label(
                                        egui::RichText::new(ellipsize(error, 40))
                                            .small()
                                            .color(egui::Color32::from_rgb(255, 120, 120)),
                                    )
                                    .on_hover_text(error.clone());
                                }
                            });
                        });
                });
                // Whole tile is the hit target for the frequent action.
                if response.response.interact(egui::Sense::click()).clicked() {
                    toggled = Some(index);
                }
            }
        });
        if let Some(index) = enlarge {
            self.enlarged = Some(index);
        } else if let Some(index) = toggled {
            if let Some(tile) = self.tiles.get_mut(index) {
                if tile.gray.is_some() {
                    tile.selected = !tile.selected;
                    self.user_picked = true;
                }
            }
        }
    }

    fn enlarged_window(&mut self, ctx: &egui::Context) {
        let Some(index) = self.enlarged else { return };
        let Some(tile) = self.tiles.get(index) else {
            self.enlarged = None;
            return;
        };
        let mut open = true;
        let title = format!("{} · {}", tile.session, tile.label);
        let tex = tile.tex.clone();
        let caption = match tile.sharpness {
            Some(s) => format!(
                "{} · {} frames · limb edge {s:.2} px (smaller is sharper)",
                tile.captured, tile.frames
            ),
            None => format!("{} · {} frames", tile.captured, tile.frames),
        };
        let mut toggle = false;
        let selected = tile.selected;
        egui::Window::new(title)
            .open(&mut open)
            .resizable(true)
            .default_width(560.0)
            .show(ctx, |ui| {
                if let Some(tex) = &tex {
                    let side = ui.available_width().min(720.0).max(200.0);
                    ui.add(egui::Image::new(tex).fit_to_exact_size(egui::vec2(side, side)));
                }
                ui.label(egui::RichText::new(caption).small().weak());
                if ui
                    .button(if selected {
                        "✔ selected — click to drop"
                    } else {
                        "select for stacking"
                    })
                    .clicked()
                {
                    toggle = true;
                }
            });
        if toggle {
            if let Some(tile) = self.tiles.get_mut(index) {
                tile.selected = !tile.selected;
                self.user_picked = true;
            }
        }
        if !open {
            self.enlarged = None;
        }
    }
}

// -- the quick pass ---------------------------------------------------------

/// Session folders under `root`, in name order.
///
/// A folder is a session if it directly holds `scan-NN.ser` files. `root`
/// itself counts, so pointing at either one session or a night's parent folder
/// both work — which is how the captures are actually laid out on disk.
fn discover_sessions(root: &Path) -> Vec<PathBuf> {
    if !session_scans(root).is_empty() {
        return vec![root.to_path_buf()];
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && !session_scans(p).is_empty())
        .collect();
    dirs.sort();
    dirs
}

/// `scan-NN.ser` files in one folder, in capture order.
fn session_scans(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e.eq_ignore_ascii_case("ser"))
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("scan-"))
        })
        .collect();
    paths.sort();
    paths
}

/// Frame count and capture time, read from the SER header alone.
///
/// Deliberately cheap: the grid is populated from this before any
/// reconstruction runs, so a folder shows its contents immediately.
fn ser_summary(path: &Path) -> (usize, String) {
    match ser::SerReader::open(path) {
        Ok(reader) => {
            let frames = reader.header.frame_count;
            let ticks = reader.scan_mid_utc_ticks();
            (frames, ser::ticks_to_iso8601(ticks).replace('T', " "))
        }
        Err(_) => (0, "unknown time".into()),
    }
}

/// Reconstruction options for the review pass.
///
/// Everything whose job is to make a FINAL image is off; what is left is the
/// geometry needed for the disc to be recognisable and for its limb to be
/// measurable. That is the whole contract of this pass.
fn quick_options(reverse: bool) -> pipeline::ReconOptions {
    pipeline::ReconOptions {
        flip_x: reverse,
        verbose: false,
        // Minutes each, and none of them change which scan is sharpest.
        deconv: false,
        denoise: false,
        wiener: false,
        temporal_nlm: false,
        burst_repair: false,
        x_registration: false,
        profile_extraction: false,
        filtered_warp: false,
        transparency_correction: false,
        transversalium_correction: false,
        // Kept: without it the disc is sheared and the limb unmeasurable.
        jitter_correction: true,
        jitter_fast: true,
        jitter_drift: false,
        ..Default::default()
    }
}

fn quick_preview(
    path: &Path,
    reverse: bool,
    index: usize,
    tx: Sender<PreviewMsg>,
) -> Result<(Vec<u8>, usize, usize, Option<f64>), String> {
    let mut options = quick_options(reverse);
    // Route the pipeline's stage lines onto the tile, so a scan that takes a
    // while says what it is doing rather than sitting on an ellipsis.
    let step_tx = tx.clone();
    options.progress = Some(Arc::new(move |line: &str| {
        let _ = step_tx.send(PreviewMsg::Step {
            index,
            line: line.to_owned(),
        });
    }));
    let report = pipeline::reconstruct(path, &options)?;
    let _ = tx.send(PreviewMsg::Step {
        index,
        line: "measuring the limb".into(),
    });
    let image = report.output.image;
    // Limb edge width: a direct read on the seeing, and the number the tiles
    // are ranked by.
    let sharpness = metrics::fit_disk(&image)
        .map(|disk| metrics::limb_sigma(&image, &disk))
        .filter(|s| s.is_finite() && *s > 0.0);
    let (gray, w, h) = to_thumbnail(&image, PREVIEW_PX);
    Ok((gray, w, h, sharpness))
}

/// Trim to `max` characters, so a long pipeline stage name cannot push a tile
/// out of shape.
fn ellipsize(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// "2m 10s" / "45s", for a wait the user is deciding whether to sit through.
fn human_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    }
}

/// Downsample to `max_side` and stretch to 8-bit for display.
///
/// Percentile clipping, not min/max: a single hot pixel or a dead column would
/// otherwise set the whole scale and flatten the disc to grey.
fn to_thumbnail(image: &Image, max_side: usize) -> (Vec<u8>, usize, usize) {
    if image.w == 0 || image.h == 0 {
        return (Vec::new(), 0, 0);
    }
    let scale = (image.w.max(image.h) as f64 / max_side as f64).max(1.0);
    let w = ((image.w as f64 / scale).round() as usize).max(1);
    let h = ((image.h as f64 / scale).round() as usize).max(1);

    let mut finite: Vec<f32> = image.data.iter().copied().filter(|v: &f32| v.is_finite()).collect();
    if finite.is_empty() {
        return (vec![0; w * h], w, h);
    }
    finite.sort_by(f32::total_cmp);
    let lo = finite[finite.len() / 200] as f64;
    let hi = finite[finite.len() - 1 - finite.len() / 200] as f64;
    let span = (hi - lo).max(1e-6);

    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let sy = ((y as f64 * scale) as usize).min(image.h - 1);
        for x in 0..w {
            let sx = ((x as f64 * scale) as usize).min(image.w - 1);
            let v = image.data[sy * image.w + sx] as f64;
            out[y * w + x] = (((v - lo) / span) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    (out, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ghostsun-process-{tag}-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch_scans(dir: &Path, n: usize) {
        std::fs::create_dir_all(dir).unwrap();
        for i in 0..n {
            std::fs::write(dir.join(format!("scan-{:02}.ser", i + 1)), b"x").unwrap();
        }
    }

    #[test]
    fn a_single_session_folder_is_itself_the_session() {
        let dir = scratch("single");
        touch_scans(&dir, 2);
        assert_eq!(discover_sessions(&dir), vec![dir.clone()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_parent_folder_finds_every_session_under_it() {
        // The real layout: one volume holding scan-<timestamp> folders.
        let root = scratch("parent");
        touch_scans(&root.join("scan-1787568793"), 3);
        touch_scans(&root.join("scan-1787569064"), 1);
        std::fs::create_dir_all(root.join("not-a-session")).unwrap();
        let sessions = discover_sessions(&root);
        assert_eq!(sessions.len(), 2, "{sessions:?}");
        assert!(sessions.iter().all(|p| !session_scans(p).is_empty()));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn only_scan_files_are_listed_and_they_stay_in_capture_order() {
        let dir = scratch("order");
        touch_scans(&dir, 3);
        std::fs::write(dir.join("ghostsun-final.png"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let scans = session_scans(&dir);
        assert_eq!(scans.len(), 3);
        let names: Vec<String> = scans
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["scan-01.ser", "scan-02.ser", "scan-03.ser"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn tile(sharpness: Option<f64>, captured: &str) -> Tile {
        Tile {
            path: PathBuf::from("x.ser"),
            session: "s".into(),
            label: "scan-01".into(),
            frames: 10,
            captured: captured.into(),
            reverse: false,
            sharpness,
            gray: Some((vec![0], 1, 1)),
            tex: None,
            state: TileState::Ready,
            selected: false,
        }
    }

    #[test]
    fn the_sharpest_scans_sort_first_and_failures_sink() {
        let mut state = ProcessState::default();
        state.tiles = vec![
            tile(Some(3.0), "c"),
            tile(None, "a"),
            tile(Some(1.0), "b"),
        ];
        state.resort();
        // Lower limb sigma is sharper, so index 2 leads; the unmeasured one is last.
        assert_eq!(state.order, vec![2, 0, 1]);
    }

    #[test]
    fn the_better_half_is_preselected() {
        let mut state = ProcessState::default();
        state.tiles = vec![
            tile(Some(4.0), "a"),
            tile(Some(1.0), "b"),
            tile(Some(2.0), "c"),
            tile(None, "d"),
        ];
        state.pretick();
        let picked: Vec<bool> = state.tiles.iter().map(|t| t.selected).collect();
        // Three measured -> ceil(3/2) = 2 kept: the 1.0 and 2.0 scans.
        assert_eq!(picked, vec![false, true, true, false]);
        assert!(!state.tiles[3].selected, "an unmeasured scan is never auto-picked");
    }

    #[test]
    fn a_users_selection_is_not_overwritten() {
        let mut state = ProcessState::default();
        state.tiles = vec![tile(Some(9.0), "a"), tile(Some(1.0), "b")];
        state.tiles[0].selected = true;
        state.user_picked = true;
        state.pretick();
        assert!(state.tiles[0].selected, "the automatic pick must not stomp a choice");
    }

    #[test]
    fn no_estimate_is_offered_before_anything_has_finished() {
        // A figure derived from zero samples is worse than none.
        let mut state = ProcessState::default();
        state.tiles = vec![tile(None, "a"), tile(None, "b")];
        assert!(state.remaining_estimate().is_none());
    }

    #[test]
    fn the_estimate_uses_the_mean_of_finished_scans() {
        let mut state = ProcessState::default();
        state.tiles = vec![tile(None, "a"), tile(None, "b"), tile(None, "c")];
        state.per_file = vec![10.0, 20.0];
        state.previewed = 2;
        // mean 15 s, one left.
        assert_eq!(state.remaining_estimate().as_deref(), Some("15s"));
    }

    #[test]
    fn the_estimate_stops_once_the_pass_is_done() {
        let mut state = ProcessState::default();
        state.tiles = vec![tile(None, "a")];
        state.per_file = vec![5.0];
        state.previewed = 1;
        assert!(state.remaining_estimate().is_none());
        assert_eq!(state.progress_fraction(), 1.0);
    }

    #[test]
    fn durations_read_in_minutes_when_long() {
        assert_eq!(human_duration(9.4), "9s");
        assert_eq!(human_duration(130.0), "2m 10s");
        assert_eq!(human_duration(-3.0), "0s");
    }

    #[test]
    fn long_stage_names_are_trimmed_to_fit_a_tile() {
        assert_eq!(ellipsize("short", 22), "short");
        let long = ellipsize("a stage name far too long for one tile", 10);
        assert_eq!(long.chars().count(), 10);
        assert!(long.ends_with('…'));
    }

    #[test]
    fn progress_is_zero_with_nothing_loaded() {
        assert_eq!(ProcessState::default().progress_fraction(), 0.0);
    }

    #[test]
    fn thumbnails_downsample_and_survive_a_hot_pixel() {
        let mut image = Image::new(800, 400);
        for (i, v) in image.data.iter_mut().enumerate() {
            *v = (i % 100) as f32;
        }
        image.data[0] = 1.0e9; // one runaway pixel must not flatten the stretch
        let (gray, w, h) = to_thumbnail(&image, 100);
        assert_eq!((w, h), (100, 50));
        assert_eq!(gray.len(), w * h);
        let hi = gray.iter().copied().max().unwrap();
        let lo = gray.iter().copied().min().unwrap();
        assert!(hi > 200 && lo < 50, "stretch collapsed: {lo}..{hi}");
    }

    #[test]
    fn an_empty_image_does_not_panic() {
        let (gray, w, h) = to_thumbnail(&Image::new(0, 0), 64);
        assert!(gray.is_empty() && w == 0 && h == 0);
    }
}
