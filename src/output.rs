//! Output: 16-bit PNG and FITS (32-bit float — no quantization at all in the
//! science product). Minimal FITS reader for round-tripping our own files.

use crate::image2d::Image;
use crate::mathutil::percentile_f32;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Write a 16-bit grayscale PNG. `range`: explicit (lo, hi) mapping to
/// 0..65535; None = robust percentile autoscale (display product).
pub fn write_png16(path: &Path, img: &Image, range: Option<(f32, f32)>) -> io::Result<()> {
    let (lo, hi) = match range {
        Some(r) => r,
        None => {
            let lo = percentile_f32(&img.data, 0.05);
            let hi = percentile_f32(&img.data, 99.95);
            (lo, (hi - lo).max(1e-6) + lo)
        }
    };
    let scale = 65535.0 / (hi - lo).max(1e-9);
    let mut buf: Vec<u16> = Vec::with_capacity(img.w * img.h);
    for &v in &img.data {
        buf.push(((v - lo) * scale).clamp(0.0, 65535.0) as u16);
    }
    let mut bytes: Vec<u8> = Vec::with_capacity(buf.len() * 2);
    for v in buf {
        bytes.extend_from_slice(&v.to_be_bytes()); // PNG is big-endian
    }
    let file = File::create(path)?;
    let w = io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, img.w as u32, img.h as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&bytes)?;
    Ok(())
}

/// Read a 16-bit grayscale PNG back into an Image.
pub fn read_png16(path: &Path) -> io::Result<Image> {
    let decoder = png::Decoder::new(File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let mut img = Image::new(w, h);
    match info.bit_depth {
        png::BitDepth::Sixteen => {
            for i in 0..w * h {
                img.data[i] = u16::from_be_bytes([buf[2 * i], buf[2 * i + 1]]) as f32;
            }
        }
        png::BitDepth::Eight => {
            for i in 0..w * h {
                img.data[i] = buf[i] as f32 * 257.0;
            }
        }
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported PNG depth")),
    }
    Ok(img)
}

/// Write an 8-bit RGB PNG.
pub fn write_png_rgb(path: &Path, w: usize, h: usize, rgb: &[u8]) -> io::Result<()> {
    let file = File::create(path)?;
    let bw = io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(bw, w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgb)?;
    Ok(())
}

fn fits_card(key: &str, value: &str) -> [u8; 80] {
    let mut card = [b' '; 80];
    let s = format!("{:<8}= {:>20}", key, value);
    card[..s.len().min(80)].copy_from_slice(&s.as_bytes()[..s.len().min(80)]);
    card
}

/// Write a BITPIX=-32 (f32) FITS file — the lossless science product.
pub fn write_fits_f32(path: &Path, img: &Image) -> io::Result<()> {
    let mut f = File::create(path)?;
    let mut header: Vec<u8> = Vec::new();
    let mut simple = [b' '; 80];
    simple[..30].copy_from_slice(b"SIMPLE  =                    T");
    header.extend_from_slice(&simple);
    header.extend_from_slice(&fits_card("BITPIX", "-32"));
    header.extend_from_slice(&fits_card("NAXIS", "2"));
    header.extend_from_slice(&fits_card("NAXIS1", &img.w.to_string()));
    header.extend_from_slice(&fits_card("NAXIS2", &img.h.to_string()));
    header.extend_from_slice(&fits_card("BZERO", "0.0"));
    header.extend_from_slice(&fits_card("BSCALE", "1.0"));
    let mut creator = [b' '; 80];
    let cs = b"CREATOR = 'GhostSun'";
    creator[..cs.len()].copy_from_slice(cs);
    header.extend_from_slice(&creator);
    let mut end = [b' '; 80];
    end[..3].copy_from_slice(b"END");
    header.extend_from_slice(&end);
    while header.len() % 2880 != 0 {
        header.push(b' ');
    }
    f.write_all(&header)?;
    // FITS row order: first row is bottom; write flipped, big-endian
    let mut data: Vec<u8> = Vec::with_capacity(img.w * img.h * 4);
    for y in (0..img.h).rev() {
        for &v in img.row(y) {
            data.extend_from_slice(&v.to_be_bytes());
        }
    }
    while data.len() % 2880 != 0 {
        data.push(0);
    }
    f.write_all(&data)?;
    Ok(())
}

/// Minimal FITS f32 reader (for files we wrote).
#[allow(dead_code)]
pub fn read_fits_f32(path: &Path) -> io::Result<Image> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let mut w = 0usize;
    let mut h = 0usize;
    let mut data_start = 0usize;
    'outer: for block in 0.. {
        let off = block * 2880;
        if off + 2880 > buf.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "no END card"));
        }
        for c in 0..36 {
            let card = &buf[off + c * 80..off + (c + 1) * 80];
            let text = String::from_utf8_lossy(card);
            if text.starts_with("NAXIS1") {
                w = text[10..30].trim().parse().unwrap_or(0);
            } else if text.starts_with("NAXIS2") {
                h = text[10..30].trim().parse().unwrap_or(0);
            } else if text.starts_with("END") {
                data_start = off + 2880;
                break 'outer;
            }
        }
    }
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = data_start + ((h - 1 - y) * w + x) * 4;
            img.set(x, y, f32::from_be_bytes(buf[i..i + 4].try_into().unwrap()));
        }
    }
    Ok(img)
}
