//! ToupTek (toupcam) backend.
//!
//! `libtoupcam` is loaded at runtime with `libloading`; if it is absent the app
//! still launches and this backend simply reports no devices. ABI (structs,
//! constants, signatures) mirrors the ToupTek SDK — the macOS/Linux variant
//! where string parameters are `char` (UTF-8), not `wchar_t`.
//!
//! Capture uses the SDK's pull model: an event callback (fired on the SDK's own
//! thread) signals "frame ready" over a channel, and [`Camera::next_frame`]
//! does the actual `PullImageV3` on the capture thread. Frames are pulled as
//! 16-bit mono (RAW, high bit depth) so the fitter sees linear data.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_uint, c_ushort, c_void};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use libloading::{Library, Symbol};

use crate::{Backend, Camera, CameraError, CameraInfo, Frame, Roi};

// --- ABI ------------------------------------------------------------------

type HToupcam = *mut c_void;

const TOUPCAM_MAX: usize = 128;
const OPTION_RAW: c_uint = 0x04;
const OPTION_BITDEPTH: c_uint = 0x06;
const EVENT_IMAGE: c_uint = 0x0004;

#[repr(C)]
#[derive(Clone, Copy)]
struct Resolution {
    width: c_uint,
    height: c_uint,
}

#[repr(C)]
struct ModelV2 {
    name: *const c_char,
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
    displayname: [c_char; 64],
    id: [c_char; 64],
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
type FnPullV3 = unsafe extern "C" fn(HToupcam, *mut c_void, c_int, c_int, c_int, *mut FrameInfoV3) -> c_int;
type FnStop = unsafe extern "C" fn(HToupcam) -> c_int;
type FnPutOption = unsafe extern "C" fn(HToupcam, c_uint, c_int) -> c_int;
type FnPutExpoTime = unsafe extern "C" fn(HToupcam, c_uint) -> c_int;
type FnPutExpoAGain = unsafe extern "C" fn(HToupcam, c_ushort) -> c_int;
type FnPutAutoExpo = unsafe extern "C" fn(HToupcam, c_int) -> c_int;
type FnGetExpoTime = unsafe extern "C" fn(HToupcam, *mut c_uint) -> c_int;
type FnGetExpoAGain = unsafe extern "C" fn(HToupcam, *mut c_ushort) -> c_int;
type FnPutRoi = unsafe extern "C" fn(HToupcam, c_uint, c_uint, c_uint, c_uint) -> c_int;
type FnGetFinalSize = unsafe extern "C" fn(HToupcam, *mut c_int, *mut c_int) -> c_int;

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
}

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    let s: Symbol<T> = lib.get(name).map_err(|e| {
        CameraError::Sdk(format!("missing {}: {e}", String::from_utf8_lossy(name)))
    })?;
    Ok(*s)
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
    // Development fallback: borrow the dylib KStars/INDI ships.
    v.push(PathBuf::from("/Applications/kstars.app/Contents/Frameworks").join(LIBNAME));
    v.push(PathBuf::from(LIBNAME)); // system search paths
    v
}

impl Api {
    fn load() -> crate::Result<Api> {
        let mut last = String::new();
        for path in candidate_paths() {
            match unsafe { Library::new(&path) } {
                Ok(lib) => return unsafe { Api::bind(lib) },
                Err(e) => last = format!("{}: {e}", path.display()),
            }
        }
        Err(CameraError::LibraryUnavailable(format!(
            "{LIBNAME} not found ({last})"
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
            _lib: lib,
        })
    }
}

fn cstr_to_string(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
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
        let name = cstr_to_string(&dev.displayname);
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
            name: if name.is_empty() { format!("ToupTek camera {i}") } else { name },
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
    // Best-effort: raw, linear, 16-bit output. Ignore failures on models that
    // don't support an option — pulling 16 bits still works via upconversion.
    unsafe {
        (api.put_option)(h, OPTION_RAW, 1);
        (api.put_option)(h, OPTION_BITDEPTH, 1);
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
        pending_roi: None,
        started: false,
        buf: Vec::new(),
    }))
}

/// Passed to the SDK as the callback context; the callback only sends on `tx`.
struct Signal {
    tx: Sender<()>,
}

unsafe extern "C" fn on_event(n_event: c_uint, ctx: *mut c_void) {
    if n_event == EVENT_IMAGE && !ctx.is_null() {
        let sig = &*(ctx as *const Signal);
        let _ = sig.tx.send(());
    }
}

pub struct ToupcamCam {
    api: Api,
    h: HToupcam,
    info: CameraInfo,
    rx: Receiver<()>,
    signal: *mut Signal,
    width: usize,
    height: usize,
    pending_roi: Option<Roi>,
    started: bool,
    buf: Vec<u8>,
}

// The SDK handle and callback context are only touched from the capture thread
// (plus the SDK's callback, which merely sends on the Sender — itself Send).
unsafe impl Send for ToupcamCam {}

impl ToupcamCam {
    fn refresh_size(&mut self) -> crate::Result<()> {
        let (mut w, mut h) = (0i32, 0i32);
        let hr = unsafe { (self.api.get_final_size)(self.h, &mut w, &mut h) };
        if hr < 0 || w <= 0 || h <= 0 {
            return Err(CameraError::Sdk("get_FinalSize failed".into()));
        }
        self.width = w as usize;
        self.height = h as usize;
        self.buf.resize(self.width * self.height * 2, 0);
        Ok(())
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

    fn start(&mut self) -> crate::Result<()> {
        if let Some(r) = self.pending_roi.take() {
            // Offsets/sizes align to 2 px; 0×0 means full frame.
            let a = |v: usize| (v & !1) as c_uint;
            unsafe { (self.api.put_roi)(self.h, a(r.x), a(r.y), a(r.w), a(r.h)) };
        }
        let hr = unsafe { (self.api.start_pull)(self.h, Some(on_event), self.signal as *mut c_void) };
        if hr < 0 {
            return Err(CameraError::Sdk("StartPullModeWithCallback failed".into()));
        }
        self.started = true;
        self.refresh_size()
    }

    fn next_frame(&mut self, timeout_ms: u32) -> crate::Result<Frame> {
        if !self.started {
            return Err(CameraError::Sdk("camera not started".into()));
        }
        match self.rx.recv_timeout(Duration::from_millis(timeout_ms as u64)) {
            Ok(()) => {}
            Err(RecvTimeoutError::Timeout) => return Err(CameraError::Timeout),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(CameraError::Sdk("event channel closed".into()))
            }
        }
        // Collapse any backlog so we always pull the freshest frame.
        while self.rx.try_recv().is_ok() {}

        let mut fi = FrameInfoV3::default();
        let pitch = (self.width * 2) as c_int;
        let hr = unsafe {
            (self.api.pull_v3)(self.h, self.buf.as_mut_ptr() as *mut c_void, 0, 16, pitch, &mut fi)
        };
        if hr < 0 {
            return Err(CameraError::Sdk("PullImageV3 failed".into()));
        }
        let (w, h) = (fi.width as usize, fi.height as usize);
        if w == 0 || h == 0 || w * h * 2 > self.buf.len() {
            return Err(CameraError::Sdk("PullImageV3 returned bad dimensions".into()));
        }
        let mut data = vec![0u16; w * h];
        for (i, px) in data.iter_mut().enumerate() {
            *px = u16::from_le_bytes([self.buf[2 * i], self.buf[2 * i + 1]]);
        }
        Ok(Frame { width: w, height: h, data })
    }

    fn stop(&mut self) {
        if self.started {
            unsafe { (self.api.stop)(self.h) };
            self.started = false;
        }
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
