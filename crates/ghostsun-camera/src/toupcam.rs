//! ToupTek (toupcam) backend.
//!
//! `libtoupcam` is loaded at runtime with `libloading`; if it is absent the app
//! still launches and this backend simply reports no devices. ABI (structs,
//! constants, signatures) mirrors the ToupTek SDK, including its UTF-16
//! device/model strings on Windows and UTF-8 strings on macOS/Linux.
//!
//! Capture uses the SDK's pull model: an event callback (fired on the SDK's own
//! thread) signals "frame ready" over a channel, and [`Camera::next_frame`]
//! does the actual `PullImageV3` on the capture thread. Frames are pulled as
//! 16-bit mono (RAW, high bit depth) so the fitter sees linear data.

#![allow(non_camel_case_types)]

#[cfg(not(target_os = "windows"))]
use std::os::raw::c_char;
use std::os::raw::{c_int, c_uint, c_ushort, c_void};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use libloading::{Library, Symbol};

use crate::{Backend, Camera, CameraError, CameraInfo, Frame, Roi};

// --- ABI ------------------------------------------------------------------

type HToupcam = *mut c_void;

#[cfg(target_os = "windows")]
type ToupChar = u16;
#[cfg(not(target_os = "windows"))]
type ToupChar = c_char;

const TOUPCAM_MAX: usize = 128;
const OPTION_RAW: c_uint = 0x04;
const OPTION_BITDEPTH: c_uint = 0x06;
const EVENT_IMAGE: c_uint = 0x0004;
const EVENT_ERROR: c_uint = 0x0080;
const EVENT_DISCONNECTED: c_uint = 0x0081;
const EVENT_NOFRAMETIMEOUT: c_uint = 0x0082;
const EVENT_NOPACKETTIMEOUT: c_uint = 0x0085;
const E_PENDING: c_int = 0x8000_000a_u32 as c_int;

#[repr(C)]
#[derive(Clone, Copy)]
struct Resolution {
    width: c_uint,
    height: c_uint,
}

#[repr(C)]
struct ModelV2 {
    name: *const ToupChar,
    flag: u64,
    maxspeed: c_uint,
    preview: c_uint,
    still: c_uint,
    maxfanspeed: c_uint,
    ioctrol: c_uint,
    xpixsz: f32,
    ypixsz: f32,
    res: [Resolution; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceV2 {
    displayname: [ToupChar; 64],
    id: [ToupChar; 64],
    model: *const ModelV2,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FrameInfoV3 {
    width: c_uint,
    height: c_uint,
    flag: c_uint,
    seq: c_uint,
    timestamp: u64,
    shutterseq: c_uint,
    expotime: c_uint,
    expogain: c_ushort,
    blacklevel: c_ushort,
}

type EventCb = unsafe extern "C" fn(c_uint, *mut c_void);

type FnEnumV2 = unsafe extern "C" fn(*mut DeviceV2) -> c_uint;
type FnOpenByIndex = unsafe extern "C" fn(c_uint) -> HToupcam;
type FnClose = unsafe extern "C" fn(HToupcam);
type FnStartPull = unsafe extern "C" fn(HToupcam, Option<EventCb>, *mut c_void) -> c_int;
type FnPullV3 =
    unsafe extern "C" fn(HToupcam, *mut c_void, c_int, c_int, c_int, *mut FrameInfoV3) -> c_int;
type FnStop = unsafe extern "C" fn(HToupcam) -> c_int;
type FnPutOption = unsafe extern "C" fn(HToupcam, c_uint, c_int) -> c_int;
type FnPutExpoTime = unsafe extern "C" fn(HToupcam, c_uint) -> c_int;
type FnPutExpoAGain = unsafe extern "C" fn(HToupcam, c_ushort) -> c_int;
type FnPutAutoExpo = unsafe extern "C" fn(HToupcam, c_int) -> c_int;
type FnGetExpoTime = unsafe extern "C" fn(HToupcam, *mut c_uint) -> c_int;
type FnGetExpoAGain = unsafe extern "C" fn(HToupcam, *mut c_ushort) -> c_int;
type FnPutRoi = unsafe extern "C" fn(HToupcam, c_uint, c_uint, c_uint, c_uint) -> c_int;
type FnGetFinalSize = unsafe extern "C" fn(HToupcam, *mut c_int, *mut c_int) -> c_int;
type FnPutRealTime = unsafe extern "C" fn(HToupcam, c_int) -> c_int;

#[cfg(target_os = "macos")]
const LIBNAME: &str = "libtoupcam.dylib";
#[cfg(target_os = "linux")]
const LIBNAME: &str = "libtoupcam.so";
#[cfg(target_os = "windows")]
const LIBNAME: &str = "toupcam.dll";

/// Resolved SDK entry points. Holds the `Library` so the code pages stay mapped
/// for the lifetime of the extracted function pointers.
struct Api {
    _lib: Library,
    enum_v2: FnEnumV2,
    open_by_index: FnOpenByIndex,
    close: FnClose,
    start_pull: FnStartPull,
    pull_v3: FnPullV3,
    stop: FnStop,
    put_option: FnPutOption,
    put_expo: FnPutExpoTime,
    put_gain: FnPutExpoAGain,
    put_auto_expo: FnPutAutoExpo,
    get_expo: FnGetExpoTime,
    get_gain: FnGetExpoAGain,
    put_roi: FnPutRoi,
    get_final_size: FnGetFinalSize,
    put_real_time: Option<FnPutRealTime>,
}

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    let s: Symbol<T> = lib
        .get(name)
        .map_err(|e| CameraError::Sdk(format!("missing {}: {e}", String::from_utf8_lossy(name))))?;
    Ok(*s)
}

unsafe fn optional_sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    lib.get::<T>(name).ok().map(|symbol| *symbol)
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("GHOSTSUN_TOUPCAM_LIB") {
        v.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(LIBNAME)); // alongside the binary
            v.push(dir.join("..").join("Frameworks").join(LIBNAME)); // macOS .app bundle
        }
    }
    // The library the installer bundles, so development builds match releases.
    if let Some(p) = crate::vendored_lib(LIBNAME) {
        v.push(p);
    }
    #[cfg(target_os = "windows")]
    {
        // Camera drivers commonly install the SDK DLL privately rather than
        // on PATH. Consider known 64-bit copies only: loading the similarly
        // named x86 DLL into GhostSun's x64 process would fail.
        let program_files = std::env::var_os("ProgramW6432")
            .or_else(|| std::env::var_os("ProgramFiles"))
            .map(PathBuf::from);
        let program_files_x86 = std::env::var_os("ProgramFiles(x86)").map(PathBuf::from);

        if let Some(root) = &program_files {
            v.push(
                root.join("N.I.N.A. - Nighttime Imaging 'N' Astronomy")
                    .join("External")
                    .join("x64")
                    .join("ToupTek")
                    .join(LIBNAME),
            );
            // SharpCap versions its directory, so discover its 64-bit
            // installs instead of baking in a release number.
            if let Ok(entries) = std::fs::read_dir(root) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with("SharpCap ") && name.contains("64 bit") {
                        v.push(entry.path().join(LIBNAME));
                    }
                }
            }
        }
        if let Some(root) = &program_files_x86 {
            v.push(
                root.join("Common Files")
                    .join("ASCOM")
                    .join("x64")
                    .join(LIBNAME),
            );
        }
    }
    // Development fallback: borrow the dylib KStars/INDI ships.
    #[cfg(target_os = "macos")]
    v.push(PathBuf::from("/Applications/kstars.app/Contents/Frameworks").join(LIBNAME));
    v.push(PathBuf::from(LIBNAME)); // system search paths
    v
}

impl Api {
    fn load() -> crate::Result<Api> {
        let mut tried = Vec::new();
        for path in candidate_paths() {
            match unsafe { Library::new(&path) } {
                Ok(lib) => return unsafe { Api::bind(lib) },
                Err(e) => tried.push(format!("{}: {e}", path.display())),
            }
        }
        // Every candidate, not just the last: knowing which locations were
        // consulted is the whole diagnostic value of this message.
        Err(CameraError::LibraryUnavailable(format!(
            "{LIBNAME} not found; tried:\n  {}",
            tried.join("\n  ")
        )))
    }

    unsafe fn bind(lib: Library) -> crate::Result<Api> {
        Ok(Api {
            enum_v2: sym(&lib, b"Toupcam_EnumV2")?,
            open_by_index: sym(&lib, b"Toupcam_OpenByIndex")?,
            close: sym(&lib, b"Toupcam_Close")?,
            start_pull: sym(&lib, b"Toupcam_StartPullModeWithCallback")?,
            pull_v3: sym(&lib, b"Toupcam_PullImageV3")?,
            stop: sym(&lib, b"Toupcam_Stop")?,
            put_option: sym(&lib, b"Toupcam_put_Option")?,
            put_expo: sym(&lib, b"Toupcam_put_ExpoTime")?,
            put_gain: sym(&lib, b"Toupcam_put_ExpoAGain")?,
            put_auto_expo: sym(&lib, b"Toupcam_put_AutoExpoEnable")?,
            get_expo: sym(&lib, b"Toupcam_get_ExpoTime")?,
            get_gain: sym(&lib, b"Toupcam_get_ExpoAGain")?,
            put_roi: sym(&lib, b"Toupcam_put_Roi")?,
            get_final_size: sym(&lib, b"Toupcam_get_FinalSize")?,
            put_real_time: optional_sym(&lib, b"Toupcam_put_RealTime"),
            _lib: lib,
        })
    }
}

#[cfg(not(target_os = "windows"))]
fn toup_string(buf: &[ToupChar]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(target_os = "windows")]
fn toup_string(buf: &[ToupChar]) -> String {
    let units: Vec<u16> = buf.iter().copied().take_while(|&c| c != 0).collect();
    String::from_utf16_lossy(&units)
}

/// Probe the ToupTek SDK while preserving loader errors for UI diagnostics.
/// Regular enumeration intentionally remains failure-tolerant.
pub fn probe() -> crate::Result<usize> {
    let api = Api::load()?;
    let mut arr: [DeviceV2; TOUPCAM_MAX] = unsafe { std::mem::zeroed() };
    Ok(unsafe { (api.enum_v2)(arr.as_mut_ptr()) } as usize)
}

pub fn enumerate() -> Vec<CameraInfo> {
    let api = match Api::load() {
        Ok(a) => a,
        Err(_) => return Vec::new(), // no SDK ⇒ no devices, never an error
    };
    let mut arr: [DeviceV2; TOUPCAM_MAX] = unsafe { std::mem::zeroed() };
    let n = unsafe { (api.enum_v2)(arr.as_mut_ptr()) } as usize;
    let mut out = Vec::new();
    for (i, dev) in arr.iter().enumerate().take(n.min(TOUPCAM_MAX)) {
        let name = toup_string(&dev.displayname);
        // res[0] is the largest sensor resolution; used to bound ROI sliders.
        let (mw, mh) = if dev.model.is_null() {
            (0, 0)
        } else {
            let r = unsafe { (*dev.model).res[0] };
            (r.width as usize, r.height as usize)
        };
        out.push(CameraInfo {
            backend: Backend::Toupcam,
            id: i.to_string(), // opened by enumeration index
            name: if name.is_empty() {
                format!("ToupTek camera {i}")
            } else {
                name
            },
            max_width: mw,
            max_height: mh,
            exposure_us: 100..=15_000_000,
            gain: 100..=1000,
        });
    }
    out
}

pub fn open(info: &CameraInfo) -> crate::Result<Box<dyn Camera>> {
    let api = Api::load()?;
    let index: c_uint = info.id.parse().map_err(|_| CameraError::NotFound)?;
    let h = unsafe { (api.open_by_index)(index) };
    if h.is_null() {
        return Err(CameraError::Sdk("Toupcam_OpenByIndex returned null".into()));
    }
    // Disable auto-exposure — it is ON by default and overrides BOTH manual
    // exposure and gain, making the sliders appear to do nothing.
    unsafe { (api.put_auto_expo)(h, 0) };
    // Best-effort: raw, linear output. Default 16-bit; `set_bit_depth(8)`
    // switches OPTION_BITDEPTH to 0. Ignore failures on models that don't
    // support an option — pulling still works via upconversion.
    unsafe {
        (api.put_option)(h, OPTION_RAW, 1);
        (api.put_option)(h, OPTION_BITDEPTH, 1);
    }
    // Keep one buffering policy for the entire USB session. Repeatedly
    // switching FIFO/real-time mode, or stopping and restarting the pull
    // stream between SER files, can wedge some ToupTek-derived cameras after
    // several alternating scans. At SHG acquisition frame rates GhostSun
    // consumes each callback synchronously, so real-time mode still records
    // every frame the application receives while bounding preview latency.
    if let Some(put_real_time) = api.put_real_time {
        unsafe {
            put_real_time(h, 1);
        }
    }
    let (tx, rx) = channel();
    let signal = Box::into_raw(Box::new(Signal { tx }));
    Ok(Box::new(ToupcamCam {
        api,
        h,
        info: info.clone(),
        rx,
        signal,
        width: 0,
        height: 0,
        bit_depth: 16,
        pending_roi: None,
        started: false,
        buf: Vec::new(),
    }))
}

/// Passed to the SDK as the callback context; the callback only sends on `tx`.
struct Signal {
    tx: Sender<c_uint>,
}

unsafe extern "C" fn on_event(n_event: c_uint, ctx: *mut c_void) {
    if !ctx.is_null() {
        let sig = &*(ctx as *const Signal);
        let _ = sig.tx.send(n_event);
    }
}

pub struct ToupcamCam {
    api: Api,
    h: HToupcam,
    info: CameraInfo,
    rx: Receiver<c_uint>,
    signal: *mut Signal,
    width: usize,
    height: usize,
    /// 8 or 16; controls OPTION_BITDEPTH and PullImageV3 bits/pitch.
    bit_depth: u8,
    pending_roi: Option<Roi>,
    started: bool,
    buf: Vec<u8>,
}

// The SDK handle and callback context are only touched from the capture thread
// (plus the SDK's callback, which merely sends on the Sender — itself Send).
unsafe impl Send for ToupcamCam {}

impl ToupcamCam {
    fn sdk_failure(call: &str, hr: c_int) -> CameraError {
        CameraError::Sdk(format!("{call} failed (HRESULT 0x{:08x})", hr as u32))
    }

    fn stop_checked(&mut self) -> crate::Result<()> {
        if !self.started {
            return Ok(());
        }
        let hr = unsafe { (self.api.stop)(self.h) };
        if hr < 0 {
            return Err(Self::sdk_failure("Toupcam_Stop", hr));
        }
        self.started = false;
        while self.rx.try_recv().is_ok() {}
        Ok(())
    }

    fn bytes_per_pixel(&self) -> usize {
        if self.bit_depth <= 8 {
            1
        } else {
            2
        }
    }

    fn refresh_size(&mut self) -> crate::Result<()> {
        let (mut w, mut h) = (0i32, 0i32);
        let hr = unsafe { (self.api.get_final_size)(self.h, &mut w, &mut h) };
        if hr < 0 || w <= 0 || h <= 0 {
            return Err(CameraError::Sdk("get_FinalSize failed".into()));
        }
        self.width = w as usize;
        self.height = h as usize;
        self.buf
            .resize(self.width * self.height * self.bytes_per_pixel(), 0);
        Ok(())
    }

    fn pull_into_buffer(&mut self) -> crate::Result<(usize, usize)> {
        let mut fi = FrameInfoV3::default();
        let bpp = self.bytes_per_pixel();
        let bits = if bpp == 1 { 8 } else { 16 };
        let pitch = (self.width * bpp) as c_int;
        let hr = unsafe {
            (self.api.pull_v3)(
                self.h,
                self.buf.as_mut_ptr() as *mut c_void,
                0,
                bits,
                pitch,
                &mut fi,
            )
        };
        if hr == E_PENDING {
            return Err(CameraError::Timeout);
        }
        if hr < 0 {
            return Err(CameraError::Sdk("PullImageV3 failed".into()));
        }
        let (w, h) = (fi.width as usize, fi.height as usize);
        if w == 0 || h == 0 || w * h * bpp > self.buf.len() {
            return Err(CameraError::Sdk(
                "PullImageV3 returned bad dimensions".into(),
            ));
        }
        Ok((w, h))
    }

    fn frame_from_buffer(&self, w: usize, h: usize) -> Frame {
        let mut data = vec![0u16; w * h];
        if self.bytes_per_pixel() == 1 {
            for (i, px) in data.iter_mut().enumerate() {
                // Stretch 8-bit samples into the u16 range the fitter expects.
                *px = u16::from(self.buf[i]) * 257;
            }
        } else {
            for (i, px) in data.iter_mut().enumerate() {
                *px = u16::from_le_bytes([self.buf[2 * i], self.buf[2 * i + 1]]);
            }
        }
        Frame {
            width: w,
            height: h,
            data,
        }
    }

    fn wait_for_image(&self, timeout_ms: u32) -> crate::Result<()> {
        let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms as u64);
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            match self.rx.recv_timeout(remaining) {
                Ok(EVENT_IMAGE) => return Ok(()),
                Ok(EVENT_ERROR) => {
                    return Err(CameraError::Sdk("camera reported EVENT_ERROR".into()))
                }
                Ok(EVENT_DISCONNECTED) => {
                    return Err(CameraError::Sdk("camera disconnected".into()))
                }
                Ok(EVENT_NOFRAMETIMEOUT) => {
                    return Err(CameraError::Sdk("camera reported no-frame timeout".into()))
                }
                Ok(EVENT_NOPACKETTIMEOUT) => {
                    return Err(CameraError::Sdk("camera reported no-packet timeout".into()))
                }
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => return Err(CameraError::Timeout),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(CameraError::Sdk("event channel closed".into()))
                }
            }
        }
    }
}

impl Camera for ToupcamCam {
    fn info(&self) -> &CameraInfo {
        &self.info
    }

    fn set_exposure_us(&mut self, us: u32) -> crate::Result<()> {
        let hr = unsafe { (self.api.put_expo)(self.h, us) };
        if hr < 0 {
            return Err(CameraError::Sdk("put_ExpoTime failed".into()));
        }
        Ok(())
    }

    fn set_gain(&mut self, gain: u16) -> crate::Result<()> {
        let hr = unsafe { (self.api.put_gain)(self.h, gain.max(100)) };
        if hr < 0 {
            return Err(CameraError::Sdk("put_ExpoAGain failed".into()));
        }
        Ok(())
    }

    fn set_auto_exposure(&mut self, on: bool) -> crate::Result<()> {
        let hr = unsafe { (self.api.put_auto_expo)(self.h, i32::from(on)) };
        if hr < 0 {
            return Err(CameraError::Sdk("put_AutoExpoEnable failed".into()));
        }
        Ok(())
    }

    fn current_exposure_us(&mut self) -> Option<u32> {
        let mut v: c_uint = 0;
        if unsafe { (self.api.get_expo)(self.h, &mut v) } >= 0 {
            Some(v)
        } else {
            None
        }
    }

    fn current_gain(&mut self) -> Option<u16> {
        let mut v: c_ushort = 0;
        if unsafe { (self.api.get_gain)(self.h, &mut v) } >= 0 {
            Some(v)
        } else {
            None
        }
    }

    fn set_roi(&mut self, roi: Roi) -> crate::Result<()> {
        // ROI must be applied while stopped; the app stops → set_roi → starts.
        self.pending_roi = Some(roi);
        Ok(())
    }

    fn set_bit_depth(&mut self, bits: u8) -> crate::Result<()> {
        if bits != 8 && bits != 16 {
            return Err(CameraError::Sdk(format!(
                "unsupported bit depth {bits} (want 8 or 16)"
            )));
        }
        if self.started {
            return Err(CameraError::Sdk(
                "set_bit_depth must be called while stopped".into(),
            ));
        }
        // ToupTek: OPTION_BITDEPTH 0 = 8-bit, 1 = 16-bit.
        let opt = if bits == 8 { 0 } else { 1 };
        let hr = unsafe { (self.api.put_option)(self.h, OPTION_BITDEPTH, opt) };
        if hr < 0 {
            return Err(CameraError::Sdk(format!(
                "put_Option(BITDEPTH,{opt}) failed for {bits}-bit"
            )));
        }
        self.bit_depth = bits;
        Ok(())
    }

    fn start(&mut self) -> crate::Result<()> {
        if let Some(r) = self.pending_roi.take() {
            // Offsets/sizes align to 2 px; 0×0 means full frame.
            let a = |v: usize| (v & !1) as c_uint;
            let hr = unsafe { (self.api.put_roi)(self.h, a(r.x), a(r.y), a(r.w), a(r.h)) };
            if hr < 0 {
                return Err(Self::sdk_failure("Toupcam_put_Roi", hr));
            }
        }
        // Re-assert bit depth in case an earlier start left the SDK elsewhere.
        let opt = if self.bit_depth == 8 { 0 } else { 1 };
        let hr = unsafe { (self.api.put_option)(self.h, OPTION_BITDEPTH, opt) };
        if hr < 0 {
            return Err(Self::sdk_failure("Toupcam_put_Option(BITDEPTH)", hr));
        }
        let hr =
            unsafe { (self.api.start_pull)(self.h, Some(on_event), self.signal as *mut c_void) };
        if hr < 0 {
            return Err(Self::sdk_failure("Toupcam_StartPullModeWithCallback", hr));
        }
        self.started = true;
        if let Err(error) = self.refresh_size() {
            self.stop();
            return Err(error);
        }
        Ok(())
    }

    fn reconfigure_roi(&mut self, roi: Roi) -> crate::Result<()> {
        self.stop_checked()?;
        // Stop is synchronous, but several ToupTek-derived USB cameras need a
        // short quiet interval before their next geometry/start transaction.
        std::thread::sleep(Duration::from_millis(75));
        self.pending_roi = Some(roi);
        self.start()?;

        let exposure_ms = self.current_exposure_us().unwrap_or(250_000) / 1000;
        let first_frame_timeout = exposure_ms
            .saturating_mul(3)
            .saturating_add(1500)
            .clamp(2000, 10_000);
        if let Err(error) = self.wait_for_image(first_frame_timeout) {
            let _ = self.stop_checked();
            return Err(CameraError::Sdk(format!(
                "hardware ROI stream did not deliver a frame after restart: {error}"
            )));
        }
        Ok(())
    }

    fn next_frame(&mut self, timeout_ms: u32) -> crate::Result<Frame> {
        if !self.started {
            return Err(CameraError::Sdk("camera not started".into()));
        }
        self.wait_for_image(timeout_ms)?;
        let (w, h) = self.pull_into_buffer()?;
        Ok(self.frame_from_buffer(w, h))
    }

    fn next_preview_frame(&mut self, timeout_ms: u32) -> crate::Result<Frame> {
        if !self.started {
            return Err(CameraError::Sdk("camera not started".into()));
        }
        // Callback notifications can accumulate independently of the SDK's
        // real-time image slot. Discard stale notifications, wait for the next
        // completed exposure, then pull the SDK's newest whole frame. Calling
        // Toupcam_Flush here can race USB delivery and produce horizontally
        // torn frames on some cameras.
        while self.rx.try_recv().is_ok() {}
        self.wait_for_image(timeout_ms)?;
        let (w, h) = self.pull_into_buffer()?;
        Ok(self.frame_from_buffer(w, h))
    }

    fn resume_preview(&mut self) -> crate::Result<()> {
        if !self.started {
            return Err(CameraError::Sdk("camera not started".into()));
        }
        // Recording and preview share one uninterrupted pull stream. Only
        // discard callback notifications accumulated while the SER footer was
        // being finalised; the next preview call waits for a newly completed
        // whole frame.
        while self.rx.try_recv().is_ok() {}
        Ok(())
    }

    fn stop(&mut self) {
        let _ = self.stop_checked();
    }
}

impl Drop for ToupcamCam {
    fn drop(&mut self) {
        self.stop();
        if !self.h.is_null() {
            unsafe { (self.api.close)(self.h) };
            self.h = std::ptr::null_mut();
        }
        // Free the callback context only after Close, so no in-flight callback
        // can reference it.
        if !self.signal.is_null() {
            unsafe { drop(Box::from_raw(self.signal)) };
            self.signal = std::ptr::null_mut();
        }
    }
}
