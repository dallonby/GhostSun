//! SER v3 video file reader/writer (mono).
//! Header layout: 14-byte FileID, 7 x i32, 3 x 40-byte strings, 2 x i64 = 178 bytes.

use crate::image2d::Image;
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

pub const HEADER_SIZE: usize = 178;

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

/// Write a mono SER file from 16-bit frames (used by the synthetic generator).
pub fn write_ser(path: &Path, width: usize, height: usize, frames: &[Vec<u16>]) -> io::Result<()> {
    let mut f = File::create(path)?;
    let mut header = vec![0u8; HEADER_SIZE];
    header[..14].copy_from_slice(b"LUCAM-RECORDER");
    let put_i32 = |h: &mut [u8], off: usize, v: i32| h[off..off + 4].copy_from_slice(&v.to_le_bytes());
    put_i32(&mut header, 14, 0); // LuID
    put_i32(&mut header, 18, 0); // MONO
    put_i32(&mut header, 22, 0); // endianness flag (ignored by most readers)
    put_i32(&mut header, 26, width as i32);
    put_i32(&mut header, 30, height as i32);
    put_i32(&mut header, 34, 16); // bit depth
    put_i32(&mut header, 38, frames.len() as i32);
    header[42..42 + 8].copy_from_slice(b"GhostSun");
    header[82..82 + 5].copy_from_slice(b"Synth");
    header[122..122 + 5].copy_from_slice(b"Synth");
    f.write_all(&header)?;
    let mut buf = Vec::with_capacity(width * height * 2);
    for frame in frames {
        buf.clear();
        for &v in frame {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    Ok(())
}
