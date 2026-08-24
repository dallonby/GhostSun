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
use std::time::{Duration, SystemTime};

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
// Which fields of the frame info the camera actually filled in. The structs
// are always returned; only these bits say a field means anything.
const FRAMEINFO_FLAG_SEQ: c_uint = 0x0000_0001;
const FRAMEINFO_FLAG_TIMESTAMP: c_uint = 0x0000_0002;
const FRAMEINFO_FLAG_GPS: c_uint = 0x0000_0040;
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

/// GPS block of `ToupcamFrameInfoV4`. `utcstart`/`utcend` are nanoseconds
/// since the Unix epoch — absolute exposure timing straight from the camera,
/// present only on GPS-equipped models (flagged by `FRAMEINFO_FLAG_GPS`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Gps {
    utcstart: u64,
    utcend: u64,
    longitude: c_int,
    latitude: c_int,
    altitude: c_int,
    satellite: c_ushort,
    reserved: c_ushort,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FrameInfoV4 {
    v3: FrameInfoV3,
    reserved: c_uint,
    u_lum: c_uint,
    u_fv: u64,
    timecount: u64,
    framecount: c_uint,
    tricount: c_uint,
    gps: Gps,
}

type EventCb = unsafe extern "C" fn(c_uint, *mut c_void);

type FnEnumV2 = unsafe extern "C" fn(*mut DeviceV2) -> c_uint;
type FnOpenByIndex = unsafe extern "C" fn(c_uint) -> HToupcam;
type FnClose = unsafe extern "C" fn(HToupcam);
type FnStartPull = unsafe extern "C" fn(HToupcam, Option<EventCb>, *mut c_void) -> c_int;
type FnPullV3 =
    unsafe extern "C" fn(HToupcam, *mut c_void, c_int, c_int, c_int, *mut FrameInfoV3) -> c_int;
type FnPullV4 =
    unsafe extern "C" fn(HToupcam, *mut c_void, c_int, c_int, c_int, *mut FrameInfoV4) -> c_int;
type FnStop = unsafe extern "C" fn(HToupcam) -> c_int;
type FnPutOption = unsafe extern "C" fn(HToupcam, c_uint, c_int) -> c_int;
type FnPutExpoTime = unsafe extern "C" fn(HToupcam, c_uint) -> c_int;
type FnPutExpoAGain = unsafe extern "C" fn(HToupcam, c_ushort) -> c_int;
type FnPutAutoExpo = unsafe extern "C" fn(HToupcam, c_int) -> c_int;
type FnGetExpoTime = unsafe extern "C" fn(HToupcam, *mut c_uint) -> c_int;
type FnGetExpoAGain = unsafe extern "C" fn(HToupcam, *mut c_ushort) -> c_int;
type FnGetExpoAGainRange =
    unsafe extern "C" fn(HToupcam, *mut c_ushort, *mut c_ushort, *mut c_ushort) -> c_int;
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
    /// V4 adds GPS exposure timestamps; absent from older SDK builds.
    pull_v4: Option<FnPullV4>,
    stop: FnStop,
    put_option: FnPutOption,
    put_expo: FnPutExpoTime,
    put_gain: FnPutExpoAGain,
    put_auto_expo: FnPutAutoExpo,
    get_expo: FnGetExpoTime,
    get_gain: FnGetExpoAGain,
    get_gain_range: Option<FnGetExpoAGainRange>,
    put_roi: FnPutRoi,
    get_final_size: FnGetFinalSize,
    put_real_time: Option<FnPutRealTime>,
    /// Frame-speed level and its ceiling. Read-only here: GhostSun never sets
    /// them, so reporting them is the only way to tell whether the SDK moved
    /// the camera to a slower level behind our back.
    get_speed: Option<FnGetSpeed>,
    get_max_speed: Option<FnGetMaxSpeed>,
}

type FnGetSpeed = unsafe extern "C" fn(HToupcam, *mut c_ushort) -> c_int;
type FnGetMaxSpeed = unsafe extern "C" fn(HToupcam) -> c_uint;

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
            pull_v4: optional_sym(&lib, b"Toupcam_PullImageV4"),
            stop: sym(&lib, b"Toupcam_Stop")?,
            put_option: sym(&lib, b"Toupcam_put_Option")?,
            put_expo: sym(&lib, b"Toupcam_put_ExpoTime")?,
            put_gain: sym(&lib, b"Toupcam_put_ExpoAGain")?,
            put_auto_expo: sym(&lib, b"Toupcam_put_AutoExpoEnable")?,
            get_expo: sym(&lib, b"Toupcam_get_ExpoTime")?,
            get_gain: sym(&lib, b"Toupcam_get_ExpoAGain")?,
            get_gain_range: optional_sym(&lib, b"Toupcam_get_ExpoAGainRange"),
            put_roi: sym(&lib, b"Toupcam_put_Roi")?,
            get_final_size: sym(&lib, b"Toupcam_get_FinalSize")?,
            put_real_time: optional_sym(&lib, b"Toupcam_put_RealTime"),
            get_speed: optional_sym(&lib, b"Toupcam_get_Speed"),
            get_max_speed: optional_sym(&lib, b"Toupcam_get_MaxSpeed"),
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
            // Conservative guess; the true per-model ceiling needs an open
            // handle, so `open` re-queries it via get_ExpoAGainRange.
            gain: 100..=1000,
        });
    }
    out
}

/// Open a handle and apply the options that hold for a whole USB session.
fn open_handle(api: &Api, index: c_uint) -> crate::Result<HToupcam> {
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
    Ok(h)
}

pub fn open(info: &CameraInfo) -> crate::Result<Box<dyn Camera>> {
    let api = Api::load()?;
    let index: c_uint = info.id.parse().map_err(|_| CameraError::NotFound)?;
    let h = open_handle(&api, index)?;
    // Replace the enumerate-time gain guess with the device's real range
    // (e.g. IMX678-class models allow far more than 1000%).
    let mut info = info.clone();
    if let Some(get_gain_range) = api.get_gain_range {
        let (mut lo, mut hi, mut def) = (0 as c_ushort, 0 as c_ushort, 0 as c_ushort);
        if unsafe { get_gain_range(h, &mut lo, &mut hi, &mut def) } >= 0 && lo < hi {
            info.gain = lo..=hi;
        }
    }
    let (tx, rx) = channel();
    let signal = Box::into_raw(Box::new(Signal { tx }));
    Ok(Box::new(ToupcamCam {
        api,
        h,
        info,
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
    /// Host time at which the SDK announced a ready frame — the earliest
    /// host-side stamp available, ahead of the pull on the capture thread.
    tx: Sender<SystemTime>,
}

unsafe extern "C" fn on_event(n_event: c_uint, ctx: *mut c_void) {
    if n_event == EVENT_IMAGE && !ctx.is_null() {
        let sig = &*(ctx as *const Signal);
        let _ = sig.tx.send(SystemTime::now());
    }
}

/// Per-frame timing the camera reported, filtered by its validity flags.
#[derive(Clone, Copy, Default)]
struct FrameTiming {
    device_us: Option<u64>,
    seq: Option<u64>,
    /// GPS exposure start, nanoseconds since the Unix epoch.
    utc_ns: Option<u64>,
}

pub struct ToupcamCam {
    api: Api,
    h: HToupcam,
    info: CameraInfo,
    rx: Receiver<SystemTime>,
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

    fn pull_into_buffer(&mut self) -> crate::Result<(usize, usize, FrameTiming)> {
        let mut fi = FrameInfoV3::default();
        let mut fi4 = FrameInfoV4::default();
        let bpp = self.bytes_per_pixel();
        let bits = if bpp == 1 { 8 } else { 16 };
        let pitch = (self.width * bpp) as c_int;
        let hr = match self.api.pull_v4 {
            Some(pull_v4) => {
                let hr = unsafe {
                    pull_v4(
                        self.h,
                        self.buf.as_mut_ptr() as *mut c_void,
                        0,
                        bits,
                        pitch,
                        &mut fi4,
                    )
                };
                fi = fi4.v3;
                hr
            }
            None => unsafe {
                (self.api.pull_v3)(
                    self.h,
                    self.buf.as_mut_ptr() as *mut c_void,
                    0,
                    bits,
                    pitch,
                    &mut fi,
                )
            },
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
        // Only the fields the camera flagged as filled in are trusted; the
        // structs come back populated either way.
        Ok((
            w,
            h,
            FrameTiming {
                device_us: (fi.flag & FRAMEINFO_FLAG_TIMESTAMP != 0).then_some(fi.timestamp),
                seq: (fi.flag & FRAMEINFO_FLAG_SEQ != 0).then(|| u64::from(fi.seq)),
                utc_ns: (self.api.pull_v4.is_some() && fi4.v3.flag & FRAMEINFO_FLAG_GPS != 0)
                    .then_some(fi4.gps.utcstart),
            },
        ))
    }

    fn frame_from_buffer(
        &self,
        w: usize,
        h: usize,
        host_time: SystemTime,
        t: &FrameTiming,
    ) -> Frame {
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
            // Best absolute time the camera offered: a GPS exposure start
            // beats the host stamp outright, since it needs no latency
            // correction at all.
            host_time: match t.utc_ns {
                Some(ns) => std::time::UNIX_EPOCH + Duration::from_nanos(ns),
                None => host_time,
            },
            // The camera's own free-running clock. Exact in SPACING even when
            // it has no idea what the absolute time is, which is what the SER
            // recorder needs to remove host-side delivery jitter.
            device_time_us: t.device_us,
            seq: t.seq,
        }
    }

    /// Block for the next frame-ready notification; returns the host time at
    /// which the SDK raised it.
    fn wait_for_image(&self, timeout_ms: u32) -> crate::Result<SystemTime> {
        match self
            .rx
            .recv_timeout(Duration::from_millis(timeout_ms as u64))
        {
            Ok(t) => Ok(t),
            Err(RecvTimeoutError::Timeout) => Err(CameraError::Timeout),
            Err(RecvTimeoutError::Disconnected) => {
                Err(CameraError::Sdk("event channel closed".into()))
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
        // Clamp to the device range queried at open; the SDK rejects
        // out-of-range values outright rather than saturating.
        let gain = gain.clamp(*self.info.gain.start(), *self.info.gain.end());
        let hr = unsafe { (self.api.put_gain)(self.h, gain) };
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

    fn speed_level(&mut self) -> Option<(u16, u32)> {
        let get = self.api.get_speed?;
        let mut level: c_ushort = 0;
        if unsafe { get(self.h, &mut level) } < 0 {
            return None;
        }
        let max = self.api.get_max_speed.map(|f| unsafe { f(self.h) }).unwrap_or(0);
        Some((level, max))
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
        // Bit depth FIRST. It reconfigures the readout, and doing it after the
        // ROI risks discarding the geometry we just set — the ordering was the
        // other way round and ROI changes were intermittently wedging the
        // stream badly enough to need a replug.
        let opt = if self.bit_depth == 8 { 0 } else { 1 };
        unsafe {
            (self.api.put_option)(self.h, OPTION_BITDEPTH, opt);
        }
        if let Some(r) = self.pending_roi.take() {
            // Offsets/sizes align to 2 px; 0×0 means full frame.
            let a = |v: usize| (v & !1) as c_uint;
            // The ROI is CLEARED with all-zero arguments. Asking for the whole
            // sensor by its explicit dimensions is a different request and is
            // not reliably honoured: releasing a live ROI that way left the
            // stream wedged, sometimes until the camera was replugged.
            let wants_full = r.x == 0
                && r.y == 0
                && (r.w == 0 || r.w >= self.info.max_width)
                && (r.h == 0 || r.h >= self.info.max_height);
            let hr = if wants_full {
                unsafe { (self.api.put_roi)(self.h, 0, 0, 0, 0) }
            } else {
                unsafe { (self.api.put_roi)(self.h, a(r.x), a(r.y), a(r.w), a(r.h)) }
            };
            // Previously discarded, so a refused ROI started the stream on a
            // geometry nobody had agreed on instead of failing where callers
            // can fall back.
            if hr < 0 {
                return Err(CameraError::Sdk(format!(
                    "put_Roi({}, {}, {}, {}) failed (hr {hr})",
                    r.x, r.y, r.w, r.h
                )));
            }
        }
        // Drop frame-ready notifications queued for the PREVIOUS geometry.
        // The callback fires per frame regardless of who is listening, so a
        // stale one makes the first pull after a resize read a frame whose
        // size no longer matches the buffer.
        while self.rx.try_recv().is_ok() {}
        let hr =
            unsafe { (self.api.start_pull)(self.h, Some(on_event), self.signal as *mut c_void) };
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
        let host_time = self.wait_for_image(timeout_ms)?;
        let (w, h, t) = self.pull_into_buffer()?;
        Ok(self.frame_from_buffer(w, h, host_time, &t))
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
        let host_time = self.wait_for_image(timeout_ms)?;
        let (w, h, t) = self.pull_into_buffer()?;
        Ok(self.frame_from_buffer(w, h, host_time, &t))
    }

    /// Close the device and acquire a fresh handle.
    ///
    /// Changing the ROI needs the stream stopped, and stopping/restarting the
    /// pull stream on a live handle is exactly what wedges these cameras —
    /// the same failure `open` already warns about for alternating scans.
    /// Observed: one ROI apply killed frame delivery outright, and the device
    /// stayed dead across a full close/open, needing a physical replug. So a
    /// geometry change goes through a genuinely new handle instead.
    fn reopen(&mut self) -> crate::Result<()> {
        let index: c_uint = self.info.id.parse().map_err(|_| CameraError::NotFound)?;
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
        // Let the driver release the endpoint before it is claimed again.
        std::thread::sleep(Duration::from_millis(250));
        let h = open_handle(&self.api, index)?;
        // A fresh channel as well: notifications from the old handle refer to
        // a geometry that no longer exists.
        let (tx, rx) = channel();
        self.signal = Box::into_raw(Box::new(Signal { tx }));
        self.rx = rx;
        self.h = h;
        self.width = 0;
        self.height = 0;
        self.pending_roi = None;
        self.started = false;
        self.buf.clear();
        Ok(())
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
