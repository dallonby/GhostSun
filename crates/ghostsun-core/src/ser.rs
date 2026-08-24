//! SER v3 video file reader/writer (mono).
//! Header layout: 14-byte FileID, 7 x i32, 3 x 40-byte strings, 2 x i64 = 178 bytes.
//!
//! After the image data the format allows an optional trailer of one little-
//! endian i64 per frame: the frame's UTC time in .NET ticks (100 ns since
//! 0001-01-01). The writer always emits it and the reader exposes it as
//! [`SerReader::timestamps`]; the pipeline uses it as the scan-axis coordinate
//! (see `timing.rs`), so dropped frames and cadence breaks no longer shift
//! everything after them by a column.

use crate::image2d::Image;
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const HEADER_SIZE: usize = 178;
const DOTNET_UNIX_EPOCH_SECONDS: i64 = 62_135_596_800;
/// .NET ticks per second (100 ns resolution).
pub const TICKS_PER_SECOND: i64 = 10_000_000;
/// Synthetic recordings (`write_ser`) are stamped from this fixed UTC epoch
/// at a fixed cadence so generated files are byte-for-byte reproducible.
const SYNTH_EPOCH_TICKS: i64 = (DOTNET_UNIX_EPOCH_SECONDS + 1_577_880_000) * TICKS_PER_SECOND; // 2020-01-01T12:00Z
const SYNTH_CADENCE_TICKS: i64 = TICKS_PER_SECOND / 100; // 100 fps

/// Seconds since the Unix epoch for a .NET tick count.
pub fn ticks_to_unix_seconds(ticks: i64) -> f64 {
    (ticks - DOTNET_UNIX_EPOCH_SECONDS * TICKS_PER_SECOND) as f64 / TICKS_PER_SECOND as f64
}

/// .NET ticks for a host wall-clock instant.
pub fn system_time_to_ticks(t: SystemTime) -> i64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => DOTNET_UNIX_EPOCH_SECONDS
            .saturating_mul(TICKS_PER_SECOND)
            .saturating_add((d.as_nanos() / 100).min(i64::MAX as u128) as i64),
        // Before 1970 (clock not set): still a valid, if absurd, tick count.
        Err(e) => DOTNET_UNIX_EPOCH_SECONDS
            .saturating_mul(TICKS_PER_SECOND)
            .saturating_sub((e.duration().as_nanos() / 100).min(i64::MAX as u128) as i64),
    }
}

/// ISO-8601 UTC (`YYYY-MM-DDThh:mm:ss.mmm`) for a .NET tick count — the form
/// FITS `DATE-OBS` expects. Proleptic Gregorian, days-from-civil inverse
/// (Howard Hinnant's algorithm); valid for any date the tick range can hold.
pub fn ticks_to_iso8601(ticks: i64) -> String {
    let unix_ms = (ticks - DOTNET_UNIX_EPOCH_SECONDS * TICKS_PER_SECOND).div_euclid(TICKS_PER_SECOND / 1000);
    let days = unix_ms.div_euclid(86_400_000);
    let ms_of_day = unix_ms.rem_euclid(86_400_000);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let (h, m, sec, ms) = (
        ms_of_day / 3_600_000,
        (ms_of_day / 60_000) % 60,
        (ms_of_day / 1000) % 60,
        ms_of_day % 1000,
    );
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{sec:02}.{ms:03}")
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SerHeader {
    pub color_id: i32,
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub frame_count: usize,
    pub observer: String,
    pub instrument: String,
    pub telescope: String,
    pub date_time: i64,
    pub date_time_utc: i64,
}

pub struct SerReader {
    pub header: SerHeader,
    /// Per-frame UTC (.NET ticks) from the optional trailer, when the file
    /// carries one that is plausible (every entry after 1970). `None` for
    /// files written without per-frame timing.
    pub timestamps: Option<Vec<i64>>,
    mmap: Mmap,
    bytes_per_px: usize,
}

fn read_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_i64(buf: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

fn read_str(buf: &[u8], off: usize, len: usize) -> String {
    String::from_utf8_lossy(&buf[off..off + len])
        .trim_end_matches(['\0', ' '])
        .to_string()
}

impl SerReader {
    pub fn open(path: &Path) -> io::Result<SerReader> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small for SER header"));
        }
        let h = &mmap[..HEADER_SIZE];
        let color_id = read_i32(h, 18);
        let width = read_i32(h, 26) as usize;
        let height = read_i32(h, 30) as usize;
        let bit_depth = read_i32(h, 34) as u32;
        let frame_count = read_i32(h, 38) as usize;
        if color_id > 19 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "RGB SER not supported (mono spectroheliograph data expected)"));
        }
        let bytes_per_px = if bit_depth > 8 { 2 } else { 1 };
        let needed = HEADER_SIZE + frame_count * width * height * bytes_per_px;
        if mmap.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("SER truncated: need {} bytes, have {}", needed, mmap.len()),
            ));
        }
        let header = SerHeader {
            color_id,
            width,
            height,
            bit_depth,
            frame_count,
            observer: read_str(h, 42, 40),
            instrument: read_str(h, 82, 40),
            telescope: read_str(h, 122, 40),
            date_time: read_i64(h, 162),
            date_time_utc: read_i64(h, 170),
        };
        // Optional per-frame timestamp trailer directly after the image data.
        let timestamps = if frame_count > 0 && mmap.len() >= needed + 8 * frame_count {
            let ticks: Vec<i64> = (0..frame_count)
                .map(|i| read_i64(&mmap, needed + 8 * i))
                .collect();
            let floor = DOTNET_UNIX_EPOCH_SECONDS * TICKS_PER_SECOND;
            if ticks.iter().all(|&t| t > floor) {
                Some(ticks)
            } else {
                None
            }
        } else {
            None
        };
        Ok(SerReader { header, timestamps, mmap, bytes_per_px })
    }

    /// UTC (.NET ticks) of frame `idx`, when the file has per-frame timing.
    pub fn frame_utc_ticks(&self, idx: usize) -> Option<i64> {
        self.timestamps.as_ref().and_then(|t| t.get(idx).copied())
    }

    /// (first, last) frame UTC in .NET ticks: from the trailer when present,
    /// otherwise the header's single stream-start time for both.
    pub fn scan_utc_ticks(&self) -> (i64, i64) {
        match &self.timestamps {
            Some(t) if !t.is_empty() => {
                let lo = *t.iter().min().unwrap();
                let hi = *t.iter().max().unwrap();
                (lo, hi)
            }
            _ => (self.header.date_time_utc, self.header.date_time_utc),
        }
    }

    /// Mid-scan UTC in .NET ticks — the epoch that describes the whole
    /// reconstructed disk (ephemeris, reference-image matching).
    pub fn scan_mid_utc_ticks(&self) -> i64 {
        let (a, b) = self.scan_utc_ticks();
        a + (b - a) / 2
    }

    /// Raw bytes of `count` consecutive frames starting at `start`.
    pub fn raw_frames(&self, start: usize, count: usize) -> &[u8] {
        let (w, h) = (self.header.width, self.header.height);
        let fsize = w * h * self.bytes_per_px;
        let off = HEADER_SIZE + start * fsize;
        &self.mmap[off..off + fsize * count]
    }

    pub fn bytes_per_px(&self) -> usize {
        self.bytes_per_px
    }

    /// Load frame as f32 image in native orientation.
    /// 8-bit data is scaled by 257 to occupy the 16-bit range like 16-bit data.
    pub fn frame(&self, idx: usize) -> Image {
        let (w, h) = (self.header.width, self.header.height);
        let fsize = w * h * self.bytes_per_px;
        let off = HEADER_SIZE + idx * fsize;
        let raw = &self.mmap[off..off + fsize];
        let mut img = Image::new(w, h);
        if self.bytes_per_px == 2 {
            for (i, px) in img.data.iter_mut().enumerate() {
                *px = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) as f32;
            }
        } else {
            for (i, px) in img.data.iter_mut().enumerate() {
                *px = raw[i] as f32 * 257.0;
            }
        }
        img
    }
}

/// Acquisition time of one frame, handed to [`SerRecorder::write_frame`].
#[derive(Clone, Copy, Debug)]
pub struct FrameTime {
    /// Host wall-clock at which the frame became available.
    pub host: SystemTime,
    /// The camera's own microsecond clock for the same frame, when the SDK
    /// provides one. If every frame of a recording carries a strictly
    /// increasing device time, the trailer is written on that clock (aligned
    /// to host UTC by the minimum-latency pairing), so inter-frame spacing is
    /// crystal-exact instead of carrying USB/scheduling jitter.
    pub device_us: Option<u64>,
}

impl FrameTime {
    /// Host time now, no device clock.
    pub fn now() -> Self {
        FrameTime { host: SystemTime::now(), device_us: None }
    }
}

/// What [`SerRecorder::finish`] reports about the file it closed.
#[derive(Clone, Copy, Debug)]
pub struct SerSummary {
    pub frames: usize,
    /// UTC (.NET ticks) of the first and last frame as written to the trailer.
    pub first_utc_ticks: Option<i64>,
    pub last_utc_ticks: Option<i64>,
    /// Trailer times come from the camera clock (true) or host stamps (false).
    pub device_clock: bool,
}

impl SerSummary {
    /// Mean frame rate over the recording from the trailer itself.
    pub fn fps(&self) -> Option<f64> {
        match (self.first_utc_ticks, self.last_utc_ticks) {
            (Some(a), Some(b)) if self.frames > 1 && b > a => {
                Some((self.frames as f64 - 1.0) * TICKS_PER_SECOND as f64 / (b - a) as f64)
            }
            _ => None,
        }
    }
}

/// Per-frame UTC ticks for the trailer.
///
/// The camera clock is preferred when every frame carries one and it is
/// strictly increasing, because its SPACING is clean where host stamps jitter
/// by whatever the USB stack and scheduler were doing.
///
/// Its SCALE, however, cannot be assumed. The field is documented in
/// microseconds, but on real hardware it has been measured running 1.8x slow
/// against the wall clock: eight consecutive scans reported ~708 fps on the
/// camera clock and ~385 fps on host stamps, and cross-checking each scan's
/// span against the gap to the next scan's first frame showed the camera-clock
/// spans short by exactly that factor (an "impossible" 20 s of dead air
/// between scans where host-clocked ones showed 2 s). Anchoring only the
/// offset — as this did — leaves that error in the file, so every duration and
/// frame rate derived from it is wrong.
///
/// So fit BOTH scale and offset against the host stamps: keep the camera's
/// spacing, take the host's time base. The offset is still the minimum
/// residual rather than the mean, because a host stamp can only ever be late.
fn trailer_ticks(times: &[(i64, Option<u64>)]) -> (Vec<i64>, bool) {
    let device: Option<Vec<i64>> = times
        .iter()
        .map(|t| t.1.map(|us| (us.min(i64::MAX as u64 / 10) as i64) * 10))
        .collect();
    if let Some(dev) = device {
        if dev.len() >= 2 && dev.windows(2).all(|w| w[1] > w[0]) {
            if let Some(fitted) = device_on_host_timebase(times, &dev) {
                return (fitted, true);
            }
        }
    }
    (monotonic_host(times), false)
}

/// Map the camera clock onto the host time base, preserving its spacing.
///
/// `None` when the fit is degenerate or the implied rate is not credible, in
/// which case the caller falls back to host stamps: a bad fit would be worse
/// than the jitter it set out to remove.
fn device_on_host_timebase(times: &[(i64, Option<u64>)], dev: &[i64]) -> Option<Vec<i64>> {
    let n = dev.len();
    if n < 2 {
        return None;
    }
    // EVERYTHING relative to the first frame. .NET ticks are ~6.4e17 today,
    // where one f64 ulp is 128 ticks (12.8 us) — doing this arithmetic on
    // absolute values quantises the whole trailer to that step. Offsets from
    // the first frame are at most a few times 1e8 for a scan of any length,
    // which f64 carries exactly, and the absolute anchor stays an i64.
    let dev0 = dev[0];
    let host0 = times[0].0;
    let x: Vec<f64> = dev.iter().map(|&d| (d - dev0) as f64).collect();
    let y: Vec<f64> = times.iter().map(|t| (t.0 - host0) as f64).collect();

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (xi, yi) in x.iter().zip(&y) {
        let dx = xi - mean_x;
        sxx += dx * dx;
        sxy += dx * (yi - mean_y);
    }
    if sxx <= 0.0 {
        return None;
    }
    // Least squares for the rate. Host stamps are late by a variable latency,
    // never early, so this slightly under-estimates — by parts in a thousand,
    // against a scale error measured in tens of percent.
    let fitted = sxy / sxx;
    // The field is DOCUMENTED in microseconds, so 1.0 is the prior and it is
    // only overridden on strong evidence. Host latency jitter is a few ms; over
    // a short clip that is a large share of the span and the fit is noise, so a
    // rate is believed only from a long recording, and only when it is far
    // enough from 1.0 to be a real scale error rather than fitted jitter.
    let host_span = (times[n - 1].0 - host0) as f64 / TICKS_PER_SECOND as f64;
    let credible = n >= 64 && host_span >= 1.0 && fitted.is_finite();
    let scale = if credible && (fitted - 1.0).abs() > 0.02 {
        fitted
    } else {
        1.0
    };
    // A camera clock an order of magnitude out in either direction is not a
    // clock we understand; take the host stamps instead of inventing times.
    if !(0.1..=10.0).contains(&scale) {
        return None;
    }

    // Scaled offsets from the first frame, as integers.
    let scaled: Vec<i64> = x.iter().map(|xi| (scale * xi).round() as i64).collect();
    // Host stamps can only be late, so the smallest residual is the pairing
    // with the least latency and the best estimate of the true start.
    let anchor = times
        .iter()
        .zip(&scaled)
        .map(|(t, &s)| t.0 - s)
        .min()?;

    // Compressing the scale can tie adjacent frames; the trailer only has to
    // be non-decreasing.
    let mut out = Vec::with_capacity(n);
    let mut last = i64::MIN;
    for s in scaled {
        last = last.max(anchor + s);
        out.push(last);
    }
    Some(out)
}

/// Host stamps, monotonized so an NTP step cannot send the trailer backwards.
fn monotonic_host(times: &[(i64, Option<u64>)]) -> Vec<i64> {
    let mut out = Vec::with_capacity(times.len());
    let mut last = i64::MIN;
    for t in times {
        last = last.max(t.0);
        out.push(last);
    }
    out
}

/// Incremental mono16 SER writer for live acquisition.
pub struct SerRecorder {
    file: File,
    width: usize,
    height: usize,
    frame_count: usize,
    buffer: Vec<u8>,
    /// (host UTC ticks, device microseconds) per written frame.
    times: Vec<(i64, Option<u64>)>,
    finished: bool,
    summary: Option<SerSummary>,
}

impl SerRecorder {
    pub fn create(
        path: &Path,
        width: usize,
        height: usize,
        instrument: &str,
        telescope: &str,
    ) -> io::Result<Self> {
        if width == 0 || height == 0 || width > i32::MAX as usize || height > i32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SER dimensions must be non-zero 32-bit values",
            ));
        }
        let mut file = File::create(path)?;
        let mut header = vec![0u8; HEADER_SIZE];
        header[..14].copy_from_slice(b"LUCAM-RECORDER");
        put_i32(&mut header, 14, 0); // LuID
        put_i32(&mut header, 18, 0); // MONO
        put_i32(&mut header, 22, 0); // little endian
        put_i32(&mut header, 26, width as i32);
        put_i32(&mut header, 30, height as i32);
        put_i32(&mut header, 34, 16);
        put_i32(&mut header, 38, 0); // patched on finish
        put_header_str(&mut header, 42, "GhostSun");
        put_header_str(&mut header, 82, instrument);
        put_header_str(&mut header, 122, telescope);
        // Provisional stream-start time; `finalize` replaces it with the first
        // frame's trailer time. (Both DateTime fields hold UTC: the format's
        // local-time field is left equal to UTC rather than guessing a zone.)
        let ticks = dotnet_ticks_now();
        header[162..170].copy_from_slice(&ticks.to_le_bytes());
        header[170..178].copy_from_slice(&ticks.to_le_bytes());
        file.write_all(&header)?;
        Ok(Self {
            file,
            width,
            height,
            frame_count: 0,
            buffer: Vec::with_capacity(width * height * 2),
            times: Vec::new(),
            finished: false,
            summary: None,
        })
    }

    /// Append one frame with its acquisition time.
    pub fn write_frame(&mut self, pixels: &[u16], time: FrameTime) -> io::Result<()> {
        let expected = self.width * self.height;
        if pixels.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "SER frame has {} pixels; expected {expected}",
                    pixels.len()
                ),
            ));
        }
        self.buffer.clear();
        for &pixel in pixels {
            self.buffer.extend_from_slice(&pixel.to_le_bytes());
        }
        self.file.write_all(&self.buffer)?;
        self.frame_count += 1;
        self.times
            .push((system_time_to_ticks(time.host), time.device_us));
        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Write the timestamp trailer, patch the header, and close.
    pub fn finish(mut self) -> io::Result<SerSummary> {
        self.finalize()?;
        Ok(self.summary.unwrap_or(SerSummary {
            frames: self.frame_count,
            first_utc_ticks: None,
            last_utc_ticks: None,
            device_clock: false,
        }))
    }

    fn finalize(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }
        if self.frame_count > i32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SER frame count exceeded the format limit",
            ));
        }
        self.file.flush()?;
        // Trailer: one UTC tick count per frame, directly after the image
        // data. Position explicitly rather than trusting the cursor.
        let (ticks, device_clock) = trailer_ticks(&self.times);
        let image_end = HEADER_SIZE as u64 + (self.frame_count * self.width * self.height * 2) as u64;
        self.file.seek(SeekFrom::Start(image_end))?;
        let mut trailer = Vec::with_capacity(ticks.len() * 8);
        for t in &ticks {
            trailer.extend_from_slice(&t.to_le_bytes());
        }
        self.file.write_all(&trailer)?;
        // Header: stream start = first frame's time, then the frame count.
        if let Some(&t0) = ticks.first() {
            self.file.seek(SeekFrom::Start(162))?;
            self.file.write_all(&t0.to_le_bytes())?;
            self.file.write_all(&t0.to_le_bytes())?;
        }
        self.file.seek(SeekFrom::Start(38))?;
        self.file
            .write_all(&(self.frame_count as i32).to_le_bytes())?;
        self.file.flush()?;
        self.summary = Some(SerSummary {
            frames: self.frame_count,
            first_utc_ticks: ticks.first().copied(),
            last_utc_ticks: ticks.last().copied(),
            device_clock,
        });
        self.finished = true;
        Ok(())
    }
}

impl Drop for SerRecorder {
    fn drop(&mut self) {
        let _ = self.finalize();
    }
}

fn put_i32(header: &mut [u8], offset: usize, value: i32) {
    header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_header_str(header: &mut [u8], offset: usize, value: &str) {
    let bytes = value.as_bytes();
    let count = bytes.len().min(40);
    header[offset..offset + count].copy_from_slice(&bytes[..count]);
}

fn dotnet_ticks_now() -> i64 {
    let since_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    DOTNET_UNIX_EPOCH_SECONDS
        .saturating_mul(10_000_000)
        .saturating_add((since_unix.as_nanos() / 100).min(i64::MAX as u128) as i64)
}

/// Write a mono SER file from 16-bit frames with explicit per-frame UTC ticks
/// (used by the synthetic generator; `ticks.len()` must equal `frames.len()`).
pub fn write_ser_timed(
    path: &Path,
    width: usize,
    height: usize,
    frames: &[Vec<u16>],
    ticks: &[i64],
) -> io::Result<()> {
    if ticks.len() != frames.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "one timestamp per frame required",
        ));
    }
    let mut recorder = SerRecorder::create(path, width, height, "Synth", "Synth")?;
    for (frame, &t) in frames.iter().zip(ticks) {
        let host = UNIX_EPOCH + std::time::Duration::from_nanos(
            ((t - DOTNET_UNIX_EPOCH_SECONDS * TICKS_PER_SECOND).max(0) as u64) * 100,
        );
        recorder.write_frame(frame, FrameTime { host, device_us: None })?;
    }
    recorder.finish().map(|_| ())
}

/// UTC ticks of synthetic frame `index` on the fixed synthetic cadence.
pub fn synth_frame_ticks(index: usize) -> i64 {
    SYNTH_EPOCH_TICKS + index as i64 * SYNTH_CADENCE_TICKS
}

/// UTC ticks at a fractional position on the synthetic cadence — a frame that
/// arrived a fraction of an interval early or late.
pub fn synth_frame_ticks_at(position: f64) -> i64 {
    SYNTH_EPOCH_TICKS + (position * SYNTH_CADENCE_TICKS as f64).round() as i64
}

/// Write a mono SER file from 16-bit frames at the fixed synthetic cadence
/// (100 fps from a fixed epoch, so generated files are reproducible).
pub fn write_ser(path: &Path, width: usize, height: usize, frames: &[Vec<u16>]) -> io::Result<()> {
    let ticks: Vec<i64> = (0..frames.len()).map(synth_frame_ticks).collect();
    write_ser_timed(path, width, height, frames, &ticks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_recorder_patches_count_and_timestamp() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ghostsun-ser-recorder-{}-{unique}.ser",
            std::process::id()
        ));
        let mut recorder =
            SerRecorder::create(&path, 3, 2, "Test camera", "Test SHG").unwrap();
        let t0 = SystemTime::now();
        recorder.write_frame(&[1, 2, 3, 4, 5, 6], FrameTime { host: t0, device_us: None }).unwrap();
        recorder
            .write_frame(
                &[7, 8, 9, 10, 11, 12],
                FrameTime { host: t0 + std::time::Duration::from_millis(10), device_us: None },
            )
            .unwrap();
        let summary = recorder.finish().unwrap();
        assert_eq!(summary.frames, 2);
        assert!(!summary.device_clock);

        let reader = SerReader::open(&path).unwrap();
        assert_eq!(reader.header.width, 3);
        assert_eq!(reader.header.height, 2);
        assert_eq!(reader.header.frame_count, 2);
        assert_eq!(reader.header.instrument, "Test camera");
        assert!(reader.header.date_time_utc > DOTNET_UNIX_EPOCH_SECONDS * 10_000_000);
        assert_eq!(reader.frame(1).data, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        // Trailer: two host ticks 10 ms apart, header start = frame 0.
        let ts = reader.timestamps.clone().expect("trailer written");
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[1] - ts[0], 100_000);
        assert_eq!(reader.header.date_time_utc, ts[0]);
        assert_eq!(ts[0], system_time_to_ticks(t0));
        std::fs::remove_file(path).unwrap();
    }

    fn temp_ser(tag: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ghostsun-ser-{tag}-{}-{unique}.ser",
            std::process::id()
        ))
    }

    #[test]
    fn device_clock_trailer_is_jitter_free_and_anchored_by_min_latency() {
        // Camera clock: exactly 5000 us apart. Host stamps: late by a random
        // 1-4 ms on top of a 2 ms USB floor. The trailer must reproduce the
        // camera spacing exactly and sit at host - floor.
        let path = temp_ser("devclock");
        let base = SystemTime::now();
        let mut rec = SerRecorder::create(&path, 1, 1, "cam", "shg").unwrap();
        let lat_ms = [3u64, 2, 4, 2, 5, 2, 3];
        for (i, lat) in lat_ms.iter().enumerate() {
            let dev = 1_000_000 + 5000 * i as u64;
            let host = base + std::time::Duration::from_micros(dev + lat * 1000);
            rec.write_frame(&[i as u16], FrameTime { host, device_us: Some(dev) }).unwrap();
        }
        let summary = rec.finish().unwrap();
        assert!(summary.device_clock);
        let fps = summary.fps().unwrap();
        assert!((fps - 200.0).abs() < 1e-6, "fps {fps}");
        let reader = SerReader::open(&path).unwrap();
        let ts = reader.timestamps.clone().unwrap();
        for w in ts.windows(2) {
            assert_eq!(w[1] - w[0], 50_000, "exact 5 ms spacing in ticks");
        }
        // Anchor: min latency was 2 ms, so frame 0 = its host stamp - 1 ms.
        let host0 = system_time_to_ticks(base + std::time::Duration::from_micros(1_000_000 + 3000));
        assert_eq!(ts[0], host0 - 10_000);
        assert_eq!(reader.scan_utc_ticks(), (ts[0], ts[6]));
        assert_eq!(reader.scan_mid_utc_ticks(), ts[0] + (ts[6] - ts[0]) / 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_slow_camera_clock_is_rescaled_onto_the_host_time_base() {
        // The measured hardware fault: the camera clock advanced 1413 us per
        // frame where the wall clock advanced 2544, so the file claimed ~708
        // fps for a camera really running ~393. Spacing is the camera's; the
        // RATE has to be the host's.
        let path = temp_ser("slowclock");
        let base = SystemTime::now();
        let mut rec = SerRecorder::create(&path, 1, 1, "cam", "shg").unwrap();
        let n = 4000usize;
        for i in 0..n {
            let dev = 1_000_000 + 1413 * i as u64;
            // Real time advances 2544 us per frame, plus a little late jitter.
            let real_us = 2544 * i as u64 + (i as u64 % 7) * 120;
            rec.write_frame(
                &[0u16],
                FrameTime {
                    host: base + std::time::Duration::from_micros(real_us),
                    device_us: Some(dev),
                },
            )
            .unwrap();
        }
        let summary = rec.finish().unwrap();
        assert!(summary.device_clock, "camera spacing should still be used");
        let fps = summary.fps().unwrap();
        assert!((fps - 393.0).abs() < 2.0, "fps {fps} — should follow the wall clock");

        let reader = SerReader::open(&path).unwrap();
        let ts = reader.timestamps.clone().unwrap();
        // Still perfectly even: the camera's spacing survived the rescale.
        let steps: Vec<i64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
        let lo = *steps.iter().min().unwrap();
        let hi = *steps.iter().max().unwrap();
        assert!(hi - lo <= 1, "spacing should stay jitter-free: {lo}..{hi}");
        // And the duration is the real one, not the camera's compressed view.
        let span = (ts[n - 1] - ts[0]) as f64 / TICKS_PER_SECOND as f64;
        assert!((span - 2544.0 * (n as f64 - 1.0) / 1e6).abs() < 0.1, "span {span}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_correct_camera_clock_is_left_alone() {
        // A camera whose microseconds really are microseconds must not be
        // "corrected" by fitting to host jitter.
        let path = temp_ser("goodclock");
        let base = SystemTime::now();
        let mut rec = SerRecorder::create(&path, 1, 1, "cam", "shg").unwrap();
        let n = 2000usize;
        for i in 0..n {
            let dev = 500_000 + 2000 * i as u64;
            rec.write_frame(
                &[0u16],
                FrameTime {
                    host: base + std::time::Duration::from_micros(2000 * i as u64 + (i as u64 % 5) * 300),
                    device_us: Some(dev),
                },
            )
            .unwrap();
        }
        let summary = rec.finish().unwrap();
        let fps = summary.fps().unwrap();
        assert!((fps - 500.0).abs() < 0.5, "fps {fps}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn trailer_times_are_not_quantised_by_float_precision() {
        // .NET ticks are ~6.4e17, where one f64 ulp is 128 ticks. Doing the
        // fit on absolute values silently rounded every frame to 12.8 us.
        let path = temp_ser("precision");
        let base = SystemTime::now();
        let mut rec = SerRecorder::create(&path, 1, 1, "cam", "shg").unwrap();
        for i in 0..200u64 {
            let dev = 1_000_000 + 1000 * i;
            rec.write_frame(
                &[0u16],
                FrameTime {
                    host: base + std::time::Duration::from_micros(dev + 2000),
                    device_us: Some(dev),
                },
            )
            .unwrap();
        }
        rec.finish().unwrap();
        let reader = SerReader::open(&path).unwrap();
        let ts = reader.timestamps.clone().unwrap();
        for w in ts.windows(2) {
            assert_eq!(w[1] - w[0], 10_000, "1 ms steps must survive exactly");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn host_fallback_when_device_clock_is_missing_or_flat() {
        // A model whose SDK leaves the timestamp at 0 must not pretend to
        // have a device clock; host stamps are used and kept monotonic.
        let path = temp_ser("hostfallback");
        let base = SystemTime::now();
        let mut rec = SerRecorder::create(&path, 1, 1, "cam", "shg").unwrap();
        let host_ms = [0u64, 10, 8, 30]; // an NTP step backwards at frame 2
        for (i, ms) in host_ms.iter().enumerate() {
            let host = base + std::time::Duration::from_millis(*ms);
            rec.write_frame(&[i as u16], FrameTime { host, device_us: Some(0) }).unwrap();
        }
        let summary = rec.finish().unwrap();
        assert!(!summary.device_clock);
        let ts = SerReader::open(&path).unwrap().timestamps.unwrap();
        assert!(ts.windows(2).all(|w| w[1] >= w[0]), "monotonic: {ts:?}");
        assert_eq!(ts[2], ts[1]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reader_reports_no_timestamps_for_legacy_files() {
        // Hand-built header + frames, no trailer.
        let path = temp_ser("legacy");
        let mut header = vec![0u8; HEADER_SIZE];
        header[..14].copy_from_slice(b"LUCAM-RECORDER");
        put_i32(&mut header, 26, 2);
        put_i32(&mut header, 30, 1);
        put_i32(&mut header, 34, 16);
        put_i32(&mut header, 38, 2);
        let mut bytes = header;
        bytes.extend_from_slice(&[1, 0, 2, 0, 3, 0, 4, 0]);
        std::fs::write(&path, &bytes).unwrap();
        let reader = SerReader::open(&path).unwrap();
        assert!(reader.timestamps.is_none());
        assert_eq!(reader.scan_utc_ticks(), (0, 0));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn synthetic_cadence_is_deterministic_and_iso_dates_round_trip() {
        assert_eq!(synth_frame_ticks(1) - synth_frame_ticks(0), TICKS_PER_SECOND / 100);
        assert_eq!(ticks_to_iso8601(synth_frame_ticks(0)), "2020-01-01T12:00:00.000");
        // 2026-08-22T07:08:09.250Z
        let unix = 1_787_382_489i64;
        let ticks = (DOTNET_UNIX_EPOCH_SECONDS + unix) * TICKS_PER_SECOND + 2_500_000;
        assert_eq!(ticks_to_iso8601(ticks), "2026-08-22T07:08:09.250");
        assert!((ticks_to_unix_seconds(ticks) - (unix as f64 + 0.25)).abs() < 1e-6);
    }
}

/// A [`SerRecorder`] driven from a background thread, so disk writes never
/// block the capture loop.
///
/// This matters more than it sounds. The capture thread must keep pulling, or
/// the vendor SDK — in real-time mode — discards whatever arrives while it is
/// busy. Writing ~900 KB synchronously per frame at several hundred frames a
/// second puts hundreds of MB/s of blocking I/O directly in that path, and the
/// dropped frames are silent. A real session recorded at 612 fps kept only
/// 49% of its frames this way, every surviving interval an exact multiple of
/// the true 1.634 ms cadence.
///
/// Frames are handed to a writer thread through a bounded queue. Bounded, not
/// unbounded: if the disk genuinely cannot keep up, memory must not grow
/// without limit, and the overflow is COUNTED and reported rather than lost
/// quietly.
pub struct AsyncSerRecorder {
    tx: Option<std::sync::mpsc::SyncSender<(Vec<u16>, FrameTime)>>,
    handle: Option<std::thread::JoinHandle<io::Result<SerSummary>>>,
    queued: usize,
    dropped: usize,
    depth: usize,
}

impl AsyncSerRecorder {
    /// `depth` frames of slack absorb write-latency spikes; 64 frames of a
    /// 3840x120 16-bit band is about 59 MB.
    pub fn create(
        path: &Path,
        width: usize,
        height: usize,
        instrument: &str,
        telescope: &str,
        depth: usize,
    ) -> io::Result<Self> {
        let mut rec = SerRecorder::create(path, width, height, instrument, telescope)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<u16>, FrameTime)>(depth.max(2));
        let handle = std::thread::spawn(move || -> io::Result<SerSummary> {
            while let Ok((px, t)) = rx.recv() {
                rec.write_frame(&px, t)?;
            }
            rec.finish()
        });
        Ok(AsyncSerRecorder {
            tx: Some(tx),
            handle: Some(handle),
            queued: 0,
            dropped: 0,
            depth: depth.max(2),
        })
    }

    /// Queue a frame. Returns `false` when the queue was full and the frame
    /// had to be dropped — the caller should surface that, because it means
    /// the disk cannot sustain the frame rate and the scan will have gaps.
    pub fn write_frame(&mut self, pixels: &[u16], time: FrameTime) -> bool {
        let Some(tx) = &self.tx else { return false };
        match tx.try_send((pixels.to_vec(), time)) {
            Ok(()) => {
                self.queued += 1;
                true
            }
            Err(_) => {
                self.dropped += 1;
                false
            }
        }
    }

    /// Frames accepted for writing.
    pub fn frame_count(&self) -> usize {
        self.queued
    }

    /// Frames refused because the writer could not keep up.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    pub fn queue_depth(&self) -> usize {
        self.depth
    }

    /// Close the queue and wait for the writer to flush and finalise.
    pub fn finish(mut self) -> io::Result<SerSummary> {
        drop(self.tx.take());
        match self.handle.take() {
            Some(h) => h
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("SER writer thread panicked"))),
            None => Err(io::Error::other("SER writer already finished")),
        }
    }
}

#[cfg(test)]
mod async_tests {
    use super::*;

    #[test]
    fn async_recorder_writes_every_queued_frame_and_reports_overflow() {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir()
            .join(format!("ghostsun-async-{}-{unique}.ser", std::process::id()));
        let (w, h) = (64usize, 8usize);
        let mut rec = AsyncSerRecorder::create(&path, w, h, "test", "test", 8).unwrap();
        let base = SystemTime::now();
        let mut accepted = 0usize;
        for i in 0..200 {
            let px = vec![i as u16; w * h];
            let t = FrameTime {
                host: base + std::time::Duration::from_micros(i as u64 * 1000),
                device_us: Some(i as u64 * 1000),
            };
            if rec.write_frame(&px, t) {
                accepted += 1;
            }
        }
        let dropped = rec.dropped();
        let summary = rec.finish().unwrap();
        // Whatever was accepted must all be on disk, and accounted for.
        assert_eq!(summary.frames, accepted, "every accepted frame written");
        assert_eq!(accepted + dropped, 200, "no frame unaccounted for");
        let reader = SerReader::open(&path).unwrap();
        assert_eq!(reader.header.frame_count, accepted);
        assert_eq!(reader.timestamps.as_ref().unwrap().len(), accepted);
        // Frame content must survive the hand-off in order.
        let f0 = reader.frame(0);
        assert!(f0.data.iter().all(|&v| v == 0.0), "first frame intact");
        std::fs::remove_file(path).unwrap();
    }
}
