//! ghostsun-camera: live camera capture for the focus assistant.
//!
//! A single [`Camera`] trait abstracts over vendor SDKs (ToupTek, ZWO) and a
//! synthetic source used to verify the focus pipeline offline against a known
//! line width. Vendor SDKs are loaded at runtime with `libloading`, so the app
//! launches and reconstructs files even when no SDK dylib or camera is present
//! — a missing backend simply contributes no devices instead of failing.
//!
//! Frames are always delivered as row-major 16-bit mono ([`Frame`]); 8-bit
//! sensors are scaled up so the downstream fitter sees one representation.

use std::fmt;
use std::ops::RangeInclusive;

pub mod synth;
pub mod toupcam;
pub mod asi;

/// Which SDK a camera belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Synth,
    Toupcam,
    Asi,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Synth => "Synthetic",
            Backend::Toupcam => "ToupTek",
            Backend::Asi => "ZWO ASI",
        }
    }
}

/// One captured mono frame, row-major, 16-bit (8-bit sensors scaled to 16).
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
}

impl Frame {
    /// Collapse to a 1-D spectrum by averaging along the slit axis, ready for
    /// the line fitter. `dispersion_horizontal` = true averages rows (result
    /// length = width); false averages columns (result length = height).
    pub fn mean_profile(&self, dispersion_horizontal: bool) -> Vec<f64> {
        if dispersion_horizontal {
            let mut prof = vec![0f64; self.width];
            for y in 0..self.height {
                let row = &self.data[y * self.width..(y + 1) * self.width];
                for (p, &v) in prof.iter_mut().zip(row) {
                    *p += v as f64;
                }
            }
            let inv = 1.0 / self.height.max(1) as f64;
            prof.iter_mut().for_each(|p| *p *= inv);
            prof
        } else {
            let mut prof = vec![0f64; self.height];
            for y in 0..self.height {
                let row = &self.data[y * self.width..(y + 1) * self.width];
                prof[y] = row.iter().map(|&v| v as f64).sum::<f64>() / self.width.max(1) as f64;
            }
            prof
        }
    }
}

/// A camera discovered by a backend. `id` is the opaque handle passed to
/// [`open`]; `exposure_us`/`gain` bound the UI sliders.
#[derive(Clone, Debug)]
pub struct CameraInfo {
    pub backend: Backend,
    pub id: String,
    pub name: String,
    pub max_width: usize,
    pub max_height: usize,
    pub exposure_us: RangeInclusive<u32>,
    pub gain: RangeInclusive<u16>,
}

/// Region of interest, in full-sensor pixels.
#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

#[derive(Debug)]
pub enum CameraError {
    /// The vendor SDK dylib could not be located or loaded.
    LibraryUnavailable(String),
    /// No camera matched the requested id.
    NotFound,
    /// The SDK reported an error.
    Sdk(String),
    /// Timed out waiting for a frame.
    Timeout,
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CameraError::LibraryUnavailable(s) => write!(f, "camera SDK unavailable: {s}"),
            CameraError::NotFound => write!(f, "camera not found"),
            CameraError::Sdk(s) => write!(f, "camera SDK error: {s}"),
            CameraError::Timeout => write!(f, "timed out waiting for frame"),
        }
    }
}

impl std::error::Error for CameraError {}

pub type Result<T> = std::result::Result<T, CameraError>;

/// A live camera. Implementors own vendor resources and are moved into the
/// capture thread; all methods are called from that one thread.
pub trait Camera: Send {
    fn info(&self) -> &CameraInfo;
    fn set_exposure_us(&mut self, us: u32) -> Result<()>;
    fn set_gain(&mut self, gain: u16) -> Result<()>;
    /// Enable/disable hardware auto-exposure. Default: no-op (e.g. synthetic).
    fn set_auto_exposure(&mut self, _on: bool) -> Result<()> {
        Ok(())
    }
    /// Current exposure (µs) the camera is actually using — reflects auto-exposure.
    fn current_exposure_us(&mut self) -> Option<u32> {
        None
    }
    /// Current gain the camera is actually using — reflects auto-exposure.
    fn current_gain(&mut self) -> Option<u16> {
        None
    }
    fn set_roi(&mut self, roi: Roi) -> Result<()>;
    /// Begin streaming. Must be called before [`Camera::next_frame`].
    fn start(&mut self) -> Result<()>;
    /// Block up to `timeout_ms` for the next frame.
    fn next_frame(&mut self, timeout_ms: u32) -> Result<Frame>;
    fn stop(&mut self);
}

/// Enumerate every camera across all backends. Backends whose SDK is missing
/// contribute nothing rather than erroring — the app stays usable with none.
pub fn enumerate_all() -> Vec<CameraInfo> {
    let mut v = Vec::new();
    v.extend(synth::enumerate());
    v.extend(toupcam::enumerate());
    v.extend(asi::enumerate());
    v
}

/// Open a specific camera previously returned by [`enumerate_all`].
pub fn open(info: &CameraInfo) -> Result<Box<dyn Camera>> {
    match info.backend {
        Backend::Synth => synth::open(info),
        Backend::Toupcam => toupcam::open(info),
        Backend::Asi => asi::open(info),
    }
}
