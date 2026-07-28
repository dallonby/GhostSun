//! QHYCCD backend (libqhyccd / qhyccd.dll).
//!
//! Loaded at runtime with `libloading`. If the library is missing the backend
//! contributes no devices — same pattern as ASI/ToupTek. Capture uses the
//! live stream path (`SetQHYCCDStreamMode` + `BeginQHYCCDLive` +
//! `GetQHYCCDLiveFrame`) as 16-bit mono, which suits planetary/guide sensors
//! such as the QHY5III678M used for spectroheliograph focus assist.
//!
//! **Calling convention:** QHYCCD headers define `STDCALL` as `__stdcall` on
//! Win32 and empty on Unix. Rust maps that with `extern "system"` on Windows
//! (stdcall on x86, Microsoft x64 on x86_64) and `extern "C"` elsewhere.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_double, c_int, c_void};
use std::path::PathBuf;
use std::sync::Mutex;

use libloading::{Library, Symbol};

use crate::{Backend, Camera, CameraError, CameraInfo, Frame, Roi};

// --- ABI constants (qhyccderr.h / qhyccdstruct.h CONTROL_ID) -------------

const QHYCCD_SUCCESS: u32 = 0;
const CONTROL_GAIN: c_int = 6;
const CONTROL_EXPOSURE: c_int = 8; // microseconds
const CONTROL_AUTOEXPOSURE: c_int = 88;
const STREAM_LIVE: u8 = 1;

type Handle = *mut c_void;

// Platform calling convention for QHYCCD STDCALL exports.
#[cfg(windows)]
macro_rules! qhy_fn {
    (fn($($arg:ty),* $(,)?) -> $ret:ty) => {
        unsafe extern "system" fn($($arg),*) -> $ret
    };
}
#[cfg(not(windows))]
macro_rules! qhy_fn {
    (fn($($arg:ty),* $(,)?) -> $ret:ty) => {
        unsafe extern "C" fn($($arg),*) -> $ret
    };
}

type FnInitRes = qhy_fn!(fn() -> u32);
type FnReleaseRes = qhy_fn!(fn() -> u32);
type FnScan = qhy_fn!(fn() -> u32);
type FnGetId = qhy_fn!(fn(u32, *mut c_char) -> u32);
type FnGetModel = qhy_fn!(fn(*mut c_char, *mut c_char) -> u32);
type FnOpen = qhy_fn!(fn(*mut c_char) -> Handle);
type FnClose = qhy_fn!(fn(Handle) -> u32);
type FnStreamMode = qhy_fn!(fn(Handle, u8) -> u32);
type FnInitCam = qhy_fn!(fn(Handle) -> u32);
type FnSetParam = qhy_fn!(fn(Handle, c_int, c_double) -> u32);
type FnGetParam = qhy_fn!(fn(Handle, c_int) -> c_double);
type FnGetParamMms = qhy_fn!(fn(Handle, c_int, *mut c_double, *mut c_double, *mut c_double) -> u32);
type FnIsCtrl = qhy_fn!(fn(Handle, c_int) -> u32);
type FnSetRes = qhy_fn!(fn(Handle, u32, u32, u32, u32) -> u32);
type FnSetBin = qhy_fn!(fn(Handle, u32, u32) -> u32);
type FnSetBits = qhy_fn!(fn(Handle, u32) -> u32);
type FnDebayer = qhy_fn!(fn(Handle, bool) -> u32);
type FnMemLen = qhy_fn!(fn(Handle) -> u32);
type FnChipInfo = qhy_fn!(
    fn(
        Handle,
        *mut c_double,
        *mut c_double,
        *mut u32,
        *mut u32,
        *mut c_double,
        *mut c_double,
        *mut u32,
    ) -> u32
);
type FnBeginLive = qhy_fn!(fn(Handle) -> u32);
type FnStopLive = qhy_fn!(fn(Handle) -> u32);
type FnLiveFrame = qhy_fn!(fn(Handle, *mut u32, *mut u32, *mut u32, *mut u32, *mut u8) -> u32);

#[cfg(target_os = "macos")]
const LIBNAME: &str = "libqhyccd.dylib";
#[cfg(target_os = "linux")]
const LIBNAME: &str = "libqhyccd.so";
#[cfg(target_os = "windows")]
const LIBNAME: &str = "qhyccd.dll";

struct Api {
    _lib: Library,
    init_res: FnInitRes,
    release_res: FnReleaseRes,
    scan: FnScan,
    get_id: FnGetId,
    get_model: Option<FnGetModel>,
    open: FnOpen,
    close: FnClose,
    stream_mode: FnStreamMode,
    init_cam: FnInitCam,
    set_param: FnSetParam,
    get_param: FnGetParam,
    get_param_mms: FnGetParamMms,
    is_ctrl: FnIsCtrl,
    set_res: FnSetRes,
    set_bin: FnSetBin,
    set_bits: FnSetBits,
    debayer: Option<FnDebayer>,
    mem_len: FnMemLen,
    chip_info: FnChipInfo,
    begin_live: FnBeginLive,
    stop_live: FnStopLive,
    live_frame: FnLiveFrame,
}

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    let s: Symbol<T> = lib
        .get(name)
        .map_err(|e| CameraError::Sdk(format!("missing {}: {e}", String::from_utf8_lossy(name))))?;
    Ok(*s)
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("GHOSTSUN_QHY_LIB") {
        v.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(LIBNAME));
            v.push(dir.join("..").join("Frameworks").join(LIBNAME));
        }
    }
    // The library the installer bundles, so development builds match releases.
    if let Some(p) = crate::vendored_lib(LIBNAME) {
        v.push(p);
    }
    // Common install locations (QHYCCD SDK / KStars / Homebrew).
    #[cfg(target_os = "macos")]
    {
        v.push(PathBuf::from("/usr/local/lib").join(LIBNAME));
        v.push(PathBuf::from("/opt/homebrew/lib").join(LIBNAME));
        v.push(PathBuf::from("/usr/local/lib/libqhyccd.20.dylib"));
        v.push(PathBuf::from("/Applications/kstars.app/Contents/Frameworks").join(LIBNAME));
    }
    #[cfg(target_os = "linux")]
    {
        v.push(PathBuf::from("/usr/lib").join(LIBNAME));
        v.push(PathBuf::from("/usr/local/lib").join(LIBNAME));
        v.push(PathBuf::from("/usr/lib/x86_64-linux-gnu").join(LIBNAME));
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                // Beside GhostSun.exe (Windows package layout).
                v.push(dir.join(LIBNAME));
            }
        }
        for key in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Ok(pf) = std::env::var(key) {
                let root = PathBuf::from(pf).join("QHYCCD");
                v.push(root.join("SDK").join(LIBNAME));
                v.push(root.join("SDK").join("x64").join(LIBNAME));
                v.push(root.join("SDK").join("x86").join(LIBNAME));
                v.push(root.join(LIBNAME));
            }
        }
    }
    v.push(PathBuf::from(LIBNAME));
    v
}

static SDK_INIT: Mutex<bool> = Mutex::new(false);

impl Api {
    fn load() -> crate::Result<Api> {
        let mut tried = Vec::new();
        for path in candidate_paths() {
            match unsafe { Library::new(&path) } {
                Ok(lib) => {
                    let api = unsafe { Api::bind(lib)? };
                    api.ensure_resource()?;
                    return Ok(api);
                }
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
            init_res: sym(&lib, b"InitQHYCCDResource")?,
            release_res: sym(&lib, b"ReleaseQHYCCDResource")?,
            scan: sym(&lib, b"ScanQHYCCD")?,
            get_id: sym(&lib, b"GetQHYCCDId")?,
            get_model: sym(&lib, b"GetQHYCCDModel").ok(),
            open: sym(&lib, b"OpenQHYCCD")?,
            close: sym(&lib, b"CloseQHYCCD")?,
            stream_mode: sym(&lib, b"SetQHYCCDStreamMode")?,
            init_cam: sym(&lib, b"InitQHYCCD")?,
            set_param: sym(&lib, b"SetQHYCCDParam")?,
            get_param: sym(&lib, b"GetQHYCCDParam")?,
            get_param_mms: sym(&lib, b"GetQHYCCDParamMinMaxStep")?,
            is_ctrl: sym(&lib, b"IsQHYCCDControlAvailable")?,
            set_res: sym(&lib, b"SetQHYCCDResolution")?,
            set_bin: sym(&lib, b"SetQHYCCDBinMode")?,
            set_bits: sym(&lib, b"SetQHYCCDBitsMode")?,
            debayer: sym(&lib, b"SetQHYCCDDebayerOnOff").ok(),
            mem_len: sym(&lib, b"GetQHYCCDMemLength")?,
            chip_info: sym(&lib, b"GetQHYCCDChipInfo")?,
            begin_live: sym(&lib, b"BeginQHYCCDLive")?,
            stop_live: sym(&lib, b"StopQHYCCDLive")?,
            live_frame: sym(&lib, b"GetQHYCCDLiveFrame")?,
            _lib: lib,
        })
    }

    fn ensure_resource(&self) -> crate::Result<()> {
        let mut g = SDK_INIT.lock().unwrap_or_else(|e| e.into_inner());
        if !*g {
            let r = unsafe { (self.init_res)() };
            if r != QHYCCD_SUCCESS {
                return Err(CameraError::Sdk(format!("InitQHYCCDResource failed ({r})")));
            }
            *g = true;
        }
        Ok(())
    }
}

fn cstr_buf(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    v
}

fn from_c_buf(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Probe for connected QHY cameras. Missing SDK → empty list.
pub fn enumerate() -> Vec<CameraInfo> {
    let Ok(api) = Api::load() else {
        return Vec::new();
    };
    let n = unsafe { (api.scan)() };
    if n == 0 || n > 64 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..n {
        let mut id = [0i8; 64];
        if unsafe { (api.get_id)(i, id.as_mut_ptr()) } != QHYCCD_SUCCESS {
            continue;
        }
        let id_str = from_c_buf(&id);
        if id_str.is_empty() {
            continue;
        }
        let mut model = id_str.clone();
        if let Some(get_model) = api.get_model {
            let mut mbuf = [0i8; 64];
            let mut id_mut = cstr_buf(&id_str);
            if unsafe { get_model(id_mut.as_mut_ptr() as *mut c_char, mbuf.as_mut_ptr()) }
                == QHYCCD_SUCCESS
            {
                let m = from_c_buf(&mbuf);
                if !m.is_empty() {
                    model = m;
                }
            }
        }

        // Open briefly to read chip size and exposure/gain ranges.
        let mut id_mut = cstr_buf(&id_str);
        let h = unsafe { (api.open)(id_mut.as_mut_ptr() as *mut c_char) };
        if h.is_null() {
            out.push(CameraInfo {
                backend: Backend::Qhy,
                id: id_str,
                name: format!("QHY {model}"),
                max_width: 1920,
                max_height: 1080,
                exposure_us: 100..=10_000_000,
                gain: 0..=400,
            });
            continue;
        }
        let _ = unsafe { (api.stream_mode)(h, STREAM_LIVE) };
        let _ = unsafe { (api.init_cam)(h) };

        let mut chipw = 0.0;
        let mut chiph = 0.0;
        let mut imagew = 0u32;
        let mut imageh = 0u32;
        let mut pixelw = 0.0;
        let mut pixelh = 0.0;
        let mut bpp = 0u32;
        let _ = unsafe {
            (api.chip_info)(
                h,
                &mut chipw,
                &mut chiph,
                &mut imagew,
                &mut imageh,
                &mut pixelw,
                &mut pixelh,
                &mut bpp,
            )
        };

        let mut exp_min = 100.0;
        let mut exp_max = 10_000_000.0;
        let mut exp_step = 1.0;
        let mut gain_min = 0.0;
        let mut gain_max = 400.0;
        let mut gain_step = 1.0;
        if unsafe { (api.is_ctrl)(h, CONTROL_EXPOSURE) } == QHYCCD_SUCCESS {
            let _ = unsafe {
                (api.get_param_mms)(
                    h,
                    CONTROL_EXPOSURE,
                    &mut exp_min,
                    &mut exp_max,
                    &mut exp_step,
                )
            };
        }
        if unsafe { (api.is_ctrl)(h, CONTROL_GAIN) } == QHYCCD_SUCCESS {
            let _ = unsafe {
                (api.get_param_mms)(
                    h,
                    CONTROL_GAIN,
                    &mut gain_min,
                    &mut gain_max,
                    &mut gain_step,
                )
            };
        }
        let _ = unsafe { (api.close)(h) };

        let max_w = imagew.max(1) as usize;
        let max_h = imageh.max(1) as usize;
        let e0 = exp_min.clamp(1.0, 60_000_000.0) as u32;
        let e1 = exp_max.clamp(e0 as f64, 60_000_000.0) as u32;
        let g0 = gain_min.max(0.0) as u16;
        let g1 = gain_max.max(g0 as f64) as u16;

        out.push(CameraInfo {
            backend: Backend::Qhy,
            id: id_str,
            name: format!("QHY {model}"),
            max_width: max_w,
            max_height: max_h,
            exposure_us: e0..=e1.max(e0),
            gain: g0..=g1.max(g0),
        });
    }
    out
}

pub fn open(info: &CameraInfo) -> crate::Result<Box<dyn Camera>> {
    let api = Api::load()?;
    let mut id_mut = cstr_buf(&info.id);
    let h = unsafe { (api.open)(id_mut.as_mut_ptr() as *mut c_char) };
    if h.is_null() {
        return Err(CameraError::Sdk(format!(
            "OpenQHYCCD failed for {}",
            info.id
        )));
    }
    // Stream mode must be set before InitQHYCCD.
    if unsafe { (api.stream_mode)(h, STREAM_LIVE) } != QHYCCD_SUCCESS {
        let _ = unsafe { (api.close)(h) };
        return Err(CameraError::Sdk("SetQHYCCDStreamMode(live) failed".into()));
    }
    if unsafe { (api.init_cam)(h) } != QHYCCD_SUCCESS {
        let _ = unsafe { (api.close)(h) };
        return Err(CameraError::Sdk("InitQHYCCD failed".into()));
    }
    let _ = unsafe { (api.set_bin)(h, 1, 1) };
    let _ = unsafe { (api.set_bits)(h, 16) };
    if let Some(debayer) = api.debayer {
        let _ = unsafe { debayer(h, false) };
    }
    let _ = unsafe { (api.set_res)(h, 0, 0, info.max_width as u32, info.max_height as u32) };

    Ok(Box::new(QhyCam {
        api,
        handle: h,
        info: info.clone(),
        width: info.max_width,
        height: info.max_height,
        streaming: false,
        last_exposure_us: 10_000,
        last_gain: 10,
        buf: Vec::new(),
    }))
}

struct QhyCam {
    api: Api,
    handle: Handle,
    info: CameraInfo,
    width: usize,
    height: usize,
    streaming: bool,
    last_exposure_us: u32,
    last_gain: u16,
    buf: Vec<u8>,
}

// SDK handle is only used from the capture thread (Camera: Send).
unsafe impl Send for QhyCam {}

impl Camera for QhyCam {
    fn info(&self) -> &CameraInfo {
        &self.info
    }

    fn set_exposure_us(&mut self, us: u32) -> crate::Result<()> {
        let us = us.clamp(*self.info.exposure_us.start(), *self.info.exposure_us.end());
        let r = unsafe { (self.api.set_param)(self.handle, CONTROL_EXPOSURE, us as f64) };
        if r != QHYCCD_SUCCESS {
            return Err(CameraError::Sdk(format!(
                "SetQHYCCDParam EXPOSURE failed ({r})"
            )));
        }
        self.last_exposure_us = us;
        Ok(())
    }

    fn set_gain(&mut self, gain: u16) -> crate::Result<()> {
        let gain = gain.clamp(*self.info.gain.start(), *self.info.gain.end());
        let r = unsafe { (self.api.set_param)(self.handle, CONTROL_GAIN, gain as f64) };
        if r != QHYCCD_SUCCESS {
            return Err(CameraError::Sdk(format!(
                "SetQHYCCDParam GAIN failed ({r})"
            )));
        }
        self.last_gain = gain;
        Ok(())
    }

    fn set_auto_exposure(&mut self, on: bool) -> crate::Result<()> {
        if unsafe { (self.api.is_ctrl)(self.handle, CONTROL_AUTOEXPOSURE) } != QHYCCD_SUCCESS {
            return Ok(()); // optional
        }
        let r = unsafe {
            (self.api.set_param)(
                self.handle,
                CONTROL_AUTOEXPOSURE,
                if on { 1.0 } else { 0.0 },
            )
        };
        if r != QHYCCD_SUCCESS {
            return Err(CameraError::Sdk(format!(
                "SetQHYCCDParam AUTOEXPOSURE failed ({r})"
            )));
        }
        Ok(())
    }

    fn current_exposure_us(&mut self) -> Option<u32> {
        let v = unsafe { (self.api.get_param)(self.handle, CONTROL_EXPOSURE) };
        if v.is_finite() && v > 0.0 {
            Some(v.round() as u32)
        } else {
            Some(self.last_exposure_us)
        }
    }

    fn current_gain(&mut self) -> Option<u16> {
        let v = unsafe { (self.api.get_param)(self.handle, CONTROL_GAIN) };
        if v.is_finite() && v >= 0.0 {
            Some(v.round() as u16)
        } else {
            Some(self.last_gain)
        }
    }

    fn set_roi(&mut self, roi: Roi) -> crate::Result<()> {
        if self.streaming {
            self.stop();
        }
        let w = roi.w.max(1) as u32;
        let h = roi.h.max(1) as u32;
        let x = roi.x as u32;
        let y = roi.y as u32;
        let r = unsafe { (self.api.set_res)(self.handle, x, y, w, h) };
        if r != QHYCCD_SUCCESS {
            return Err(CameraError::Sdk(format!(
                "SetQHYCCDResolution failed ({r})"
            )));
        }
        self.width = w as usize;
        self.height = h as usize;
        Ok(())
    }

    fn start(&mut self) -> crate::Result<()> {
        if self.streaming {
            return Ok(());
        }
        let need = unsafe { (self.api.mem_len)(self.handle) } as usize;
        let need = need.max(self.width * self.height * 2);
        self.buf.resize(need, 0);
        let r = unsafe { (self.api.begin_live)(self.handle) };
        if r != QHYCCD_SUCCESS {
            return Err(CameraError::Sdk(format!("BeginQHYCCDLive failed ({r})")));
        }
        self.streaming = true;
        Ok(())
    }

    fn next_frame(&mut self, _timeout_ms: u32) -> crate::Result<Frame> {
        if !self.streaming {
            return Err(CameraError::Sdk("camera not started".into()));
        }
        if self.buf.is_empty() {
            let need = unsafe { (self.api.mem_len)(self.handle) } as usize;
            self.buf.resize(need.max(self.width * self.height * 2), 0);
        }
        let mut w = 0u32;
        let mut h = 0u32;
        let mut bpp = 0u32;
        let mut channels = 0u32;
        let r = unsafe {
            (self.api.live_frame)(
                self.handle,
                &mut w,
                &mut h,
                &mut bpp,
                &mut channels,
                self.buf.as_mut_ptr(),
            )
        };
        if r != QHYCCD_SUCCESS {
            // No frame ready yet — treat as timeout for the focus poll loop.
            return Err(CameraError::Timeout);
        }
        if w == 0 || h == 0 {
            return Err(CameraError::Timeout);
        }
        self.width = w as usize;
        self.height = h as usize;
        let n = self.width * self.height;
        let mut data = vec![0u16; n];
        match bpp {
            16 if channels <= 1 => {
                let bytes = n * 2;
                if self.buf.len() < bytes {
                    return Err(CameraError::Sdk("live buffer shorter than frame".into()));
                }
                for i in 0..n {
                    let lo = self.buf[i * 2] as u16;
                    let hi = self.buf[i * 2 + 1] as u16;
                    data[i] = lo | (hi << 8);
                }
            }
            8 if channels <= 1 => {
                if self.buf.len() < n {
                    return Err(CameraError::Sdk("live buffer shorter than frame".into()));
                }
                for i in 0..n {
                    data[i] = (self.buf[i] as u16) * 257;
                }
            }
            _ => {
                // Colour or unexpected layout: average first channel bytes.
                let stride = ((bpp / 8).max(1) as usize) * channels.max(1) as usize;
                if self.buf.len() < n * stride {
                    return Err(CameraError::Sdk(format!(
                        "unsupported live frame bpp={bpp} ch={channels}"
                    )));
                }
                for i in 0..n {
                    let b = self.buf[i * stride];
                    data[i] = if bpp >= 16 {
                        let lo = self.buf[i * stride] as u16;
                        let hi = self.buf[i * stride + 1] as u16;
                        lo | (hi << 8)
                    } else {
                        (b as u16) * 257
                    };
                }
            }
        }
        Ok(Frame {
            width: self.width,
            height: self.height,
            data,
        })
    }

    fn stop(&mut self) {
        if self.streaming {
            let _ = unsafe { (self.api.stop_live)(self.handle) };
            self.streaming = false;
        }
    }
}

impl Drop for QhyCam {
    fn drop(&mut self) {
        self.stop();
        if !self.handle.is_null() {
            let _ = unsafe { (self.api.close)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

// Silence unused release_res — kept for potential SDK teardown.
#[allow(dead_code)]
fn _release_sdk(api: &Api) {
    let _ = unsafe { (api.release_res)() };
}
