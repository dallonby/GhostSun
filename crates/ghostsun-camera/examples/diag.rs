//! Diagnostic: which libtoupcam path loads, and how many cameras EnumV2 sees.

use libloading::{Library, Symbol};
use std::os::raw::{c_char, c_uint};
use std::path::PathBuf;

#[repr(C)]
#[derive(Clone, Copy)]
struct Res {
    w: c_uint,
    h: c_uint,
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
    res: [Res; 16],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceV2 {
    displayname: [c_char; 64],
    id: [c_char; 64],
    model: *const ModelV2,
}

fn main() {
    let candidates = [
        std::env::var("GHOSTSUN_TOUPCAM_LIB").unwrap_or_default(),
        "/Applications/kstars.app/Contents/Frameworks/libtoupcam.dylib".into(),
        "libtoupcam.dylib".into(),
    ];
    for path in candidates.iter().filter(|p| !p.is_empty()) {
        print!("load {path} … ");
        let lib = match unsafe { Library::new(PathBuf::from(path)) } {
            Ok(l) => {
                println!("OK");
                l
            }
            Err(e) => {
                println!("FAIL: {e}");
                continue;
            }
        };
        unsafe {
            if let Ok(ver) = lib.get::<unsafe extern "C" fn() -> *const c_char>(b"Toupcam_Version") {
                let v = std::ffi::CStr::from_ptr(ver()).to_string_lossy().into_owned();
                println!("  SDK version: {v}");
            }
            let enum_v2: Symbol<unsafe extern "C" fn(*mut DeviceV2) -> c_uint> =
                lib.get(b"Toupcam_EnumV2").expect("EnumV2");
            let mut arr: [DeviceV2; 128] = std::mem::zeroed();
            let n = enum_v2(arr.as_mut_ptr());
            println!("  EnumV2 -> {n} camera(s)");
            for dev in arr.iter().take(n as usize) {
                let name: Vec<u8> = dev.displayname.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
                let id: Vec<u8> = dev.id.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
                println!(
                    "    name='{}' id='{}'",
                    String::from_utf8_lossy(&name),
                    String::from_utf8_lossy(&id)
                );
            }
        }
        return;
    }
}
