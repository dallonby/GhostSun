//! Session log to a file on disk.
//!
//! The in-app pane only carries what the UI thread chose to surface, so the
//! failures that matter most — a camera that stops delivering frames, an SDK
//! call that returned an error nobody propagated — left no trace at all. This
//! writes every line straight through to disk and flushes it, so the record
//! survives a wedged stream, a force-quit, or a device that had to be
//! unplugged.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
static PATH: OnceLock<PathBuf> = OnceLock::new();

fn default_path() -> PathBuf {
    // Beside the executable when that is writable (portable unzip-and-run,
    // which is how this ships), otherwise the roaming profile.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("ghostsun.log");
            if OpenOptions::new()
                .create(true)
                .append(true)
                .open(&candidate)
                .is_ok()
            {
                return candidate;
            }
        }
    }
    let mut dir = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    dir.push("GhostSun");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("ghostsun.log")
}

/// Open (or truncate to a bounded size) the log. Safe to call more than once.
pub fn init() -> PathBuf {
    let path = PATH.get_or_init(default_path).clone();
    FILE.get_or_init(|| {
        // Keep sessions from accumulating without bound, but never discard the
        // session in progress: roll once at open.
        if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > 4 * 1024 * 1024 {
            let _ = std::fs::rename(&path, path.with_extension("log.1"));
        }
        Mutex::new(OpenOptions::new().create(true).append(true).open(&path).ok())
    });
    path
}

#[allow(dead_code)]
pub fn path() -> Option<&'static Path> {
    PATH.get().map(|p| p.as_path())
}

/// UTC wall clock, so lines can be lined up against anything else on the
/// machine. Deliberately not a date: a session log is read within the day.
fn stamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}.{:03}", now.subsec_millis())
}

pub fn write(line: &str) {
    let Some(lock) = FILE.get() else { return };
    let Ok(mut guard) = lock.lock() else { return };
    if let Some(file) = guard.as_mut() {
        let _ = writeln!(file, "{} {line}", stamp());
        // Flush every line: an unflushed buffer is exactly what is lost when
        // the process is killed, which is the case worth logging for.
        let _ = file.flush();
    }
}

#[macro_export]
macro_rules! applog {
    ($($arg:tt)*) => {
        $crate::applog::write(&format!($($arg)*))
    };
}
