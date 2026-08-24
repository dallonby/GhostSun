//! Where GhostSun keeps its settings, and how much room a capture volume has.
//!
//! Both halves exist for the same reason: captures do not belong on the boot
//! volume. The chosen folder has to survive an upgrade, and the space warning
//! has to describe the volume the frames actually land on.

use std::path::{Path, PathBuf};

/// Per-user configuration directory.
///
/// Deliberately carries NO version in the path, so a new GhostSun build reads
/// the settings the previous one wrote. This is what makes the capture folder
/// persist across upgrades.
pub fn config_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    }?;
    Some(base.join("GhostSun"))
}

fn capture_dir_path() -> Option<PathBuf> {
    Some(config_dir()?.join("capture-dir.txt"))
}

/// The capture folder chosen in a previous session, if it is still usable.
///
/// A remembered path on a volume that is not mounted right now is discarded
/// rather than returned: silently recording to a freshly created folder at a
/// mount point is how a night ends up on the boot disk.
pub fn load_capture_dir() -> Option<PathBuf> {
    let text = std::fs::read_to_string(capture_dir_path()?).ok()?;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))?;
    let path = PathBuf::from(line);
    path.is_dir().then_some(path)
}

pub fn save_capture_dir(dir: &Path) -> Result<(), String> {
    let path = capture_dir_path().ok_or("no configuration directory")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        &path,
        format!("# GhostSun capture folder\n{}\n", dir.display()),
    )
    .map_err(|e| e.to_string())
}

/// Free bytes on the volume holding `path`.
///
/// The path itself need not exist yet — the nearest existing ancestor is
/// measured, which is the same volume a session folder would be created on.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let mut probe = path;
    loop {
        if probe.exists() {
            return free_bytes_existing(probe);
        }
        probe = probe.parent()?;
    }
}

#[cfg(unix)]
fn free_bytes_existing(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated path and `stat` is written
    // only by statvfs, which reports success before we read it.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return None;
        }
        // f_bavail, not f_bfree: blocks an unprivileged process may actually
        // use, which is what a capture gets.
        (stat.f_bavail as u64).checked_mul(stat.f_frsize as u64)
    }
}

#[cfg(windows)]
fn free_bytes_existing(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut available: u64 = 0;
    // SAFETY: `wide` is NUL-terminated and outlives the call; the out-param is
    // read only when the call reports success.
    let ok = unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (ok != 0).then_some(available)
}

#[cfg(not(any(unix, windows)))]
fn free_bytes_existing(_path: &Path) -> Option<u64> {
    None
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_space_is_reported_for_an_existing_directory() {
        let free = free_bytes(&std::env::temp_dir());
        assert!(free.is_some_and(|b| b > 0), "{free:?}");
    }

    #[test]
    fn free_space_falls_back_to_the_nearest_existing_ancestor() {
        // The capture folder is measured before it is created.
        let missing = std::env::temp_dir().join("ghostsun-not-created/deeper/still");
        assert!(!missing.exists());
        assert_eq!(
            free_bytes(&missing).is_some(),
            free_bytes(&std::env::temp_dir()).is_some()
        );
    }

    #[test]
    fn byte_sizes_read_in_sensible_units() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1_500), "1.5 kB");
        assert_eq!(human_bytes(2_000_000_000), "2.0 GB");
    }

    #[test]
    fn the_config_directory_carries_no_version() {
        // A versioned path would reset the capture folder on every upgrade.
        if let Some(dir) = config_dir() {
            let text = dir.display().to_string();
            assert!(text.contains("GhostSun"), "{text}");
            assert!(!text.contains(env!("CARGO_PKG_VERSION")), "{text}");
        }
    }
}
