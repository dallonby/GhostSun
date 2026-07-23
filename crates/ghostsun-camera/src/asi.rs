//! ZWO (ASI) backend.
//!
//! `libASICamera2` is loaded at runtime with `libloading`; if it is absent the
//! app still launches and this backend reports no devices. The ASI capture
//! model is a blocking pull (`ASIGetVideoData`), so no callback plumbing is
//! needed. Frames are pulled as 16-bit mono (`ASI_IMG_RAW16`).

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_double, c_int, c_long, c_uchar};
use std::path::PathBuf;

use libloading::{Library, Symbol};

use crate::{Backend, Camera, CameraError, CameraInfo, Frame, Roi};

// --- ABI (ASICamera2.h) ---------------------------------------------------

const ASI_IMG_RAW16: c_int = 2;
const ASI_GAIN: c_int = 0;
const ASI_EXPOSURE: c_int = 1;
const ASI_FALSE: c_int = 0;
const ASI_SUCCESS: c_int = 0;
const ASI_ERROR_TIMEOUT: c_int = 11;

#[repr(C)]
#[derive(Clone, Copy)]
struct AsiCameraInfo {
    name: [c_char; 64],
    camera_id: c_int,
    max_height: c_long,
    max_width: c_long,
    is_color: c_int,
    bayer: c_int,
    supported_bins: [c_int; 16],
    supported_formats: [c_int; 8],
    pixel_size: c_double,
    mechanical_shutter: c_int,
    st4: c_int,
    is_cooler: c_int,
    is_usb3_host: c_int,
    is_usb3: c_int,
    elec_per_adu: f32,
    bit_depth: c_int,
    is_trigger: c_int,
    unused: [c_char; 16],
}

type FnNumCams = unsafe extern "C" fn() -> c_int;
type FnGetProp = unsafe extern "C" fn(*mut AsiCameraInfo, c_int) -> c_int;
type FnOpen = unsafe extern "C" fn(c_int) -> c_int;
type FnInit = unsafe extern "C" fn(c_int) -> c_int;
type FnCloseCam = unsafe extern "C" fn(c_int) -> c_int;
type FnSetRoi = unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int) -> c_int;
type FnSetStartPos = unsafe extern "C" fn(c_int, c_int, c_int) -> c_int;
type FnSetControl = unsafe extern "C" fn(c_int, c_int, c_long, c_int) -> c_int;
type FnGetControl = unsafe extern "C" fn(c_int, c_int, *mut c_long, *mut c_int) -> c_int;
type FnStartVideo = unsafe extern "C" fn(c_int) -> c_int;
type FnStopVideo = unsafe extern "C" fn(c_int) -> c_int;
type FnGetVideoData = unsafe extern "C" fn(c_int, *mut c_uchar, c_long, c_int) -> c_int;

#[cfg(target_os = "macos")]
const LIBNAME: &str = "libASICamera2.dylib";
#[cfg(target_os = "linux")]
const LIBNAME: &str = "libASICamera2.so";
#[cfg(target_os = "windows")]
const LIBNAME: &str = "ASICamera2.dll";

struct Api {
    _lib: Library,
    num_cams: FnNumCams,
    get_prop: FnGetProp,
    open_cam: FnOpen,
    init_cam: FnInit,
    close_cam: FnCloseCam,
    set_roi: FnSetRoi,
    set_start_pos: Option<FnSetStartPos>,
    set_control: FnSetControl,
    get_control: FnGetControl,
    start_video: FnStartVideo,
    stop_video: FnStopVideo,
    get_video: FnGetVideoData,
}

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> crate::Result<T> {
    let s: Symbol<T> = lib.get(name).map_err(|e| {
        CameraError::Sdk(format!("missing {}: {e}", String::from_utf8_lossy(name)))
    })?;
    Ok(*s)
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("GHOSTSUN_ASI_LIB") {
        v.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(LIBNAME));
            v.push(dir.join("..").join("Frameworks").join(LIBNAME));
        }
    }
    v.push(PathBuf::from("/Applications/kstars.app/Contents/Frameworks").join(LIBNAME));
    v.push(PathBuf::from(LIBNAME));
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
        Err(CameraError::LibraryUnavailable(format!("{LIBNAME} not found ({last})")))
    }

    unsafe fn bind(lib: Library) -> crate::Result<Api> {
        Ok(Api {
            num_cams: sym(&lib, b"ASIGetNumOfConnectedCameras")?,
            get_prop: sym(&lib, b"ASIGetCameraProperty")?,
            open_cam: sym(&lib, b"ASIOpenCamera")?,
            init_cam: sym(&lib, b"ASIInitCamera")?,
            close_cam: sym(&lib, b"ASICloseCamera")?,
            set_roi: sym(&lib, b"ASISetROIFormat")?,
            set_start_pos: sym(&lib, b"ASISetStartPos").ok(),
            set_control: sym(&lib, b"ASISetControlValue")?,
            get_control: sym(&lib, b"ASIGetControlValue")?,
            start_video: sym(&lib, b"ASIStartVideoCapture")?,
            stop_video: sym(&lib, b"ASIStopVideoCapture")?,
            get_video: sym(&lib, b"ASIGetVideoData")?,
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
        Err(_) => return Vec::new(),
    };
    let n = unsafe { (api.num_cams)() };
    let mut out = Vec::new();
    for i in 0..n {
        let mut info: AsiCameraInfo = unsafe { std::mem::zeroed() };
        if unsafe { (api.get_prop)(&mut info, i) } != ASI_SUCCESS {
            continue;
        }
        out.push(CameraInfo {
            backend: Backend::Asi,
            id: info.camera_id.to_string(),
            name: cstr_to_string(&info.name),
            max_width: info.max_width as usize,
            max_height: info.max_height as usize,
            exposure_us: 32..=15_000_000,
            gain: 0..=600,
        });
    }
    out
}

pub fn open(info: &CameraInfo) -> crate::Result<Box<dyn Camera>> {
    let api = Api::load()?;
    let id: c_int = info.id.parse().map_err(|_| CameraError::NotFound)?;
    if unsafe { (api.open_cam)(id) } != ASI_SUCCESS {
        return Err(CameraError::Sdk("ASIOpenCamera failed".into()));
    }
    if unsafe { (api.init_cam)(id) } != ASI_SUCCESS {
        unsafe { (api.close_cam)(id) };
        return Err(CameraError::Sdk("ASIInitCamera failed".into()));
    }
    Ok(Box::new(AsiCam {
        api,
        id,
        info: info.clone(),
        width: 0,
        height: 0,
        pending_roi: None,
        started: false,
        buf: Vec::new(),
        last_exposure_us: 10_000,
        last_gain: 200,
    }))
}

pub struct AsiCam {
    api: Api,
    id: c_int,
    info: CameraInfo,
    width: usize,
    height: usize,
    pending_roi: Option<Roi>,
    started: bool,
    buf: Vec<u8>,
    // Retained so auto-exposure can be toggled: ASI sets auto per-control and
    // still needs a seed value on the same call.
    last_exposure_us: c_long,
    last_gain: c_long,
}

impl Camera for AsiCam {
    fn info(&self) -> &CameraInfo {
        &self.info
    }

    fn set_exposure_us(&mut self, us: u32) -> crate::Result<()> {
        self.last_exposure_us = us as c_long;
        let r = unsafe { (self.api.set_control)(self.id, ASI_EXPOSURE, us as c_long, ASI_FALSE) };
        if r != ASI_SUCCESS {
            return Err(CameraError::Sdk("set exposure failed".into()));
        }
        Ok(())
    }

    fn set_gain(&mut self, gain: u16) -> crate::Result<()> {
        self.last_gain = gain as c_long;
        let r = unsafe { (self.api.set_control)(self.id, ASI_GAIN, gain as c_long, ASI_FALSE) };
        if r != ASI_SUCCESS {
            return Err(CameraError::Sdk("set gain failed".into()));
        }
        Ok(())
    }

    fn set_auto_exposure(&mut self, on: bool) -> crate::Result<()> {
        // ASI auto is per-control; each call still carries a seed value.
        let b = if on { 1 } else { ASI_FALSE };
        unsafe {
            (self.api.set_control)(self.id, ASI_EXPOSURE, self.last_exposure_us, b);
            (self.api.set_control)(self.id, ASI_GAIN, self.last_gain, b);
        }
        Ok(())
    }

    fn current_exposure_us(&mut self) -> Option<u32> {
        let (mut v, mut auto): (c_long, c_int) = (0, 0);
        if unsafe { (self.api.get_control)(self.id, ASI_EXPOSURE, &mut v, &mut auto) } == ASI_SUCCESS {
            Some(v.max(0) as u32)
        } else {
            None
        }
    }

    fn current_gain(&mut self) -> Option<u16> {
        let (mut v, mut auto): (c_long, c_int) = (0, 0);
        if unsafe { (self.api.get_control)(self.id, ASI_GAIN, &mut v, &mut auto) } == ASI_SUCCESS {
            Some(v.clamp(0, u16::MAX as c_long) as u16)
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
        // ASI constraint: width % 8 == 0, height % 2 == 0.
        let roi = self.pending_roi.take().unwrap_or(Roi {
            x: 0,
            y: 0,
            w: self.info.max_width,
            h: self.info.max_height,
        });
        let w = (roi.w & !7).max(8);
        let h = (roi.h & !1).max(2);
        if unsafe { (self.api.set_roi)(self.id, w as c_int, h as c_int, 1, ASI_IMG_RAW16) } != ASI_SUCCESS
        {
            return Err(CameraError::Sdk("ASISetROIFormat failed".into()));
        }
        if let Some(set_pos) = self.api.set_start_pos {
            unsafe { set_pos(self.id, (roi.x & !7) as c_int, (roi.y & !1) as c_int) };
        }
        if unsafe { (self.api.start_video)(self.id) } != ASI_SUCCESS {
            return Err(CameraError::Sdk("ASIStartVideoCapture failed".into()));
        }
        self.width = w;
        self.height = h;
        self.buf.resize(w * h * 2, 0);
        self.started = true;
        Ok(())
    }

    fn next_frame(&mut self, timeout_ms: u32) -> crate::Result<Frame> {
        if !self.started {
            return Err(CameraError::Sdk("camera not started".into()));
        }
        let r = unsafe {
            (self.api.get_video)(
                self.id,
                self.buf.as_mut_ptr(),
                self.buf.len() as c_long,
                timeout_ms as c_int,
            )
        };
        if r == ASI_ERROR_TIMEOUT {
            return Err(CameraError::Timeout);
        }
        if r != ASI_SUCCESS {
            return Err(CameraError::Sdk(format!("ASIGetVideoData error {r}")));
        }
        let (w, h) = (self.width, self.height);
        let mut data = vec![0u16; w * h];
        for (i, px) in data.iter_mut().enumerate() {
            *px = u16::from_le_bytes([self.buf[2 * i], self.buf[2 * i + 1]]);
        }
        Ok(Frame { width: w, height: h, data })
    }

    fn stop(&mut self) {
        if self.started {
            unsafe { (self.api.stop_video)(self.id) };
            self.started = false;
        }
    }
}

impl Drop for AsiCam {
    fn drop(&mut self) {
        self.stop();
        unsafe { (self.api.close_cam)(self.id) };
    }
}
