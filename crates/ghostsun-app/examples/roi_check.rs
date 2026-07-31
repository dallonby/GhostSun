//! Hardware-ROI throughput check.
//!
//! Acquisition quality is frame-rate-bound: each frame is one column of the
//! reconstructed disk, so fps IS scan-axis sampling density. Full-sensor
//! 16-bit frames cap USB throughput near ~24 fps on a 4K camera, undersampling
//! the scan axis ~2x even at slow rates. A hardware ROI of the capture band
//! should lift that ceiling by an order of magnitude — this example measures
//! whether the connected camera actually honours it, what dimensions it
//! delivers, and the achieved fps in each mode, including after restore.
//!
//! Also compares 16-bit vs 8-bit mono readout at the same exposure/ROI.
//!
//! ```sh
//! cargo run --release -p ghostsun-app --example roi_check
//! ```

use std::time::{Duration, Instant};

use ghostsun_camera::{enumerate_all, open, Backend, Roi};

const ROI_ROWS: usize = 256;
const MEASURE: Duration = Duration::from_secs(3);
// Match typical live capture (~1 ms).
const EXPOSURE_US: u32 = 1_000;

fn measure(cam: &mut dyn ghostsun_camera::Camera, label: &str, roi: Roi) {
    cam.stop();
    if let Err(e) = cam.set_roi(roi) {
        eprintln!("{label}: set_roi failed: {e}");
        return;
    }
    if let Err(e) = cam.start() {
        eprintln!("{label}: start failed: {e}");
        return;
    }
    // Warm-up: let exposure settle and any mode switch complete.
    for _ in 0..5 {
        let _ = cam.next_frame(2000);
    }
    let t0 = Instant::now();
    let mut frames = 0usize;
    let mut dims = (0usize, 0usize);
    while t0.elapsed() < MEASURE {
        match cam.next_frame(2000) {
            Ok(f) => {
                dims = (f.width, f.height);
                frames += 1;
            }
            Err(e) => {
                eprintln!("{label}: frame error: {e}");
                break;
            }
        }
    }
    let fps = frames as f64 / t0.elapsed().as_secs_f64();
    let honoured = dims.1 == roi.h || (roi.h == 0 && dims.1 > 0);
    println!(
        "  {label:<22} requested {}x{} -> delivered {}x{}  {fps:6.1} fps  {}",
        roi.w,
        roi.h,
        dims.0,
        dims.1,
        if honoured {
            ""
        } else {
            "  <-- DIMENSIONS NOT HONOURED"
        }
    );
}

fn main() {
    let cams = enumerate_all();
    let Some(info) = cams.iter().find(|c| c.backend != Backend::Synth) else {
        eprintln!("no hardware camera connected");
        std::process::exit(1);
    };
    println!(
        "camera: {} {} ({}x{})",
        info.backend.label(),
        info.name,
        info.max_width,
        info.max_height
    );
    println!("exposure: {EXPOSURE_US} µs");

    let mut cam = match open(info) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("open: {e}");
            std::process::exit(1);
        }
    };
    cam.set_auto_exposure(false).ok();
    cam.set_exposure_us(EXPOSURE_US).ok();
    cam.set_gain(200).ok();

    let full = Roi {
        x: 0,
        y: 0,
        w: info.max_width,
        h: info.max_height,
    };
    let band = Roi {
        x: 0,
        y: (info.max_height / 2 - ROI_ROWS / 2) & !1,
        w: info.max_width,
        h: ROI_ROWS,
    };

    for bits in [16u8, 8u8] {
        println!("--- {bits}-bit ---");
        cam.stop();
        if let Err(e) = cam.set_bit_depth(bits) {
            eprintln!("  set_bit_depth({bits}) failed: {e}");
            continue;
        }
        for (label, roi) in [
            (format!("{bits}-bit full"), full),
            (format!("{bits}-bit ROI {ROI_ROWS}"), band),
        ] {
            measure(cam.as_mut(), &label, roi);
        }
    }

    // Leave the camera back in the app default (16-bit full frame).
    cam.stop();
    let _ = cam.set_bit_depth(16);
    measure(cam.as_mut(), "restored 16-bit full", full);
    cam.stop();
}
