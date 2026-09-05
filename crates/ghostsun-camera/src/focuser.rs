//! ToupTek AAF USB focuser. Opening and querying never request movement.
//! Uses SDK device IDs rather than enumeration indices, which change on hotplug.
use libloading::Library;
use std::ffi::c_void;

#[cfg(windows)]
type Char = u16;
#[cfg(not(windows))]
type Char = std::ffi::c_char;
type Handle = *mut c_void;
type Enum = unsafe extern "system" fn(*mut Device) -> u32;
type Open = unsafe extern "system" fn(*const Char) -> Handle;
type Close = unsafe extern "system" fn(Handle);
type Aaf = unsafe extern "system" fn(Handle, i32, i32, *mut i32) -> i32;
const FLAG: u64 = 0x0002_0000_0000_0000;

// Only the fixed prefix is accessed; the SDK owns the trailing model data.
#[repr(C)]
struct Model {
    name: *const Char,
    flag: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Device {
    name: [Char; 64],
    id: [Char; 64],
    model: *const Model,
}

#[derive(Clone, Debug)]
pub struct Info {
    pub name: String,
    pub id: String,
}
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub position: i32,
    pub max_step: i32,
    pub moving: bool,
    pub temperature: Option<f64>,
}
struct Api {
    enumerate: Enum,
    open: Open,
    close: Close,
    aaf: Aaf,
    _library: Library,
}
impl Api {
    fn load() -> Result<Self, String> {
        let mut errors = Vec::new();
        for path in crate::toupcam::candidate_paths() {
            let loaded = (|| unsafe {
                let lib = Library::new(&path).map_err(|e| e.to_string())?;
                Ok::<_, String>(Self {
                    enumerate: *lib.get(b"Toupcam_EnumV2").map_err(|e| e.to_string())?,
                    open: *lib.get(b"Toupcam_Open").map_err(|e| e.to_string())?,
                    close: *lib.get(b"Toupcam_Close").map_err(|e| e.to_string())?,
                    aaf: *lib.get(b"Toupcam_AAF").map_err(|e| e.to_string())?,
                    _library: lib,
                })
            })();
            match loaded {
                Ok(api) => return Ok(api),
                Err(e) => errors.push(format!("{}: {e}", path.display())),
            }
        }
        Err(format!(
            "ToupTek focuser SDK unavailable: {}",
            errors.join("; ")
        ))
    }
    fn devices(&self) -> Vec<Info> {
        let mut devices: [Device; 128] = unsafe { std::mem::zeroed() };
        let n = unsafe { (self.enumerate)(devices.as_mut_ptr()) } as usize;
        devices
            .iter()
            .take(n.min(devices.len()))
            .filter_map(|d| {
                if d.model.is_null() || unsafe { (*d.model).flag } & FLAG == 0 {
                    return None;
                }
                Some(Info {
                    name: decode(&d.name),
                    id: decode(&d.id),
                })
            })
            .collect()
    }
}
#[cfg(windows)]
fn decode(buf: &[Char]) -> String {
    String::from_utf16_lossy(
        &buf.iter()
            .copied()
            .take_while(|&c| c != 0)
            .collect::<Vec<_>>(),
    )
}
#[cfg(not(windows))]
fn decode(buf: &[Char]) -> String {
    String::from_utf8_lossy(
        &buf.iter()
            .copied()
            .take_while(|&c| c != 0)
            .map(|c| c as u8)
            .collect::<Vec<_>>(),
    )
    .into_owned()
}
pub fn enumerate() -> Result<Vec<Info>, String> {
    Ok(Api::load()?.devices())
}

/// Owned by one worker thread; no handle sharing or unsafe Send implementation.
pub struct Focuser {
    api: Api,
    handle: Handle,
    commanded: bool,
}
impl Focuser {
    pub fn open(id: &str) -> Result<Self, String> {
        let api = Api::load()?;
        if !api.devices().iter().any(|d| d.id == id) {
            return Err("Selected ToupTek focuser is no longer connected".into());
        }
        #[cfg(windows)]
        let encoded: Vec<Char> = id.encode_utf16().chain(Some(0)).collect();
        #[cfg(not(windows))]
        let encoded: Vec<Char> = id.bytes().map(|b| b as Char).chain(Some(0)).collect();
        let handle = unsafe { (api.open)(encoded.as_ptr()) };
        if handle.is_null() {
            return Err("Cannot open focuser; disconnect it from other applications first".into());
        }
        Ok(Self {
            api,
            handle,
            commanded: false,
        })
    }
    fn action(&self, action: i32, value: i32) -> Result<i32, String> {
        let mut result = 0;
        let hr = unsafe { (self.api.aaf)(self.handle, action, value, &mut result) };
        if hr < 0 {
            Err(format!(
                "ToupTek AAF action {action:#04x} failed ({:#010x})",
                hr as u32
            ))
        } else {
            Ok(result)
        }
    }
    pub fn state(&self) -> Result<State, String> {
        let position = self.action(0x02, 0)?;
        let max_step = self.action(0x1c, 0)?;
        if max_step <= 0 || position < 0 || position > max_step {
            return Err("Focuser returned invalid position or travel limit".into());
        }
        Ok(State {
            position,
            max_step,
            moving: self.action(0x16, 0)? != 0,
            temperature: self
                .action(0x14, 0)
                .ok()
                .filter(|&v| (-1000..=1000).contains(&v))
                .map(|v| v as f64 / 10.0),
        })
    }
    pub fn move_to(&mut self, position: i32) -> Result<(), String> {
        let state = self.state()?;
        if !(0..=state.max_step).contains(&position) {
            return Err("Target exceeds focuser travel limits".into());
        }
        // Mark before sending: even a failed response can leave motion uncertain.
        self.commanded = true;
        self.action(0x01, position).map(|_| ())
    }
    pub fn halt(&mut self) -> Result<(), String> {
        self.action(0x17, 0)?;
        self.commanded = false;
        Ok(())
    }
}
impl Drop for Focuser {
    fn drop(&mut self) {
        if self.commanded {
            let _ = self.halt();
        }
        unsafe {
            (self.api.close)(self.handle);
        }
    }
}
