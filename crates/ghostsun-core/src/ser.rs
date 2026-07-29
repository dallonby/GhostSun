//! SER v3 video file reader/writer (mono).
//! Header layout: 14-byte FileID, 7 x i32, 3 x 40-byte strings, 2 x i64 = 178 bytes.

use crate::image2d::Image;
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const HEADER_SIZE: usize = 178;
const DOTNET_UNIX_EPOCH_SECONDS: i64 = 62_135_596_800;

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
        Ok(SerReader { header, mmap, bytes_per_px })
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

/// Incremental mono16 SER writer for live acquisition.
pub struct SerRecorder {
    file: File,
    width: usize,
    height: usize,
    frame_count: usize,
    buffer: Vec<u8>,
    finished: bool,
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
            finished: false,
        })
    }

    pub fn write_frame(&mut self, pixels: &[u16]) -> io::Result<()> {
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
        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn finish(mut self) -> io::Result<usize> {
        self.finalize()?;
        Ok(self.frame_count)
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
        self.file.seek(SeekFrom::Start(38))?;
        self.file
            .write_all(&(self.frame_count as i32).to_le_bytes())?;
        self.file.flush()?;
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

/// Write a mono SER file from 16-bit frames (used by the synthetic generator).
pub fn write_ser(path: &Path, width: usize, height: usize, frames: &[Vec<u16>]) -> io::Result<()> {
    let mut recorder = SerRecorder::create(path, width, height, "Synth", "Synth")?;
    for frame in frames {
        recorder.write_frame(frame)?;
    }
    recorder.finish().map(|_| ())
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
        recorder.write_frame(&[1, 2, 3, 4, 5, 6]).unwrap();
        recorder.write_frame(&[7, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(recorder.finish().unwrap(), 2);

        let reader = SerReader::open(&path).unwrap();
        assert_eq!(reader.header.width, 3);
        assert_eq!(reader.header.height, 2);
        assert_eq!(reader.header.frame_count, 2);
        assert_eq!(reader.header.instrument, "Test camera");
        assert!(reader.header.date_time_utc > DOTNET_UNIX_EPOCH_SECONDS * 10_000_000);
        assert_eq!(reader.frame(1).data, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        std::fs::remove_file(path).unwrap();
    }
}
