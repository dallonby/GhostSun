//! Hardware end-to-end check of the per-frame timing chain (F13):
//! camera clock -> `Frame` -> SER timestamp trailer -> `ScanTiming`.
//!
//! Records a short SER from the first real camera into a small ROI, then
//! reopens it and reports what the reconstruction pipeline would see. Proves
//! the recorder picked the camera clock over host stamps, and that the trailer
//! survives the round trip.
//!
//!   cargo run -p ghostsun-camera --example record --release

use ghostsun_camera::{enumerate_all, open, Backend, Roi};
use ghostsun_core::ser::{FrameTime, SerReader, SerRecorder, TICKS_PER_SECOND};
use ghostsun_core::timing::ScanTiming;

const FRAMES: usize = 120;
const ROI_W: usize = 1024;
const ROI_H: usize = 128;

fn main() {
    let cams = enumerate_all();
    let Some(info) = cams.iter().find(|c| c.backend != Backend::Synth) else {
        println!("No hardware camera found (synthetic only).");
        return;
    };
    println!("Recording from {} …", info.name);
    let mut cam = match open(info) {
        Ok(c) => c,
        Err(e) => {
            println!("open failed: {e}");
            return;
        }
    };
    cam.set_exposure_us(5_000).ok();
    cam.set_gain(200).ok();
    // Small ROI: this is a timing test, not an imaging one, and a full-frame
    // recording would be gigabytes.
    cam.set_roi(Roi { x: 0, y: 0, w: ROI_W, h: ROI_H }).ok();
    if let Err(e) = cam.start() {
        println!("start failed: {e}");
        return;
    }

    let path = std::env::temp_dir().join("ghostsun-timing-check.ser");
    let mut rec = None;
    let mut n_device = 0usize;
    let mut last_seq: Option<u64> = None;
    let mut seq_gaps = 0usize;
    for _ in 0..FRAMES {
        let frame = match cam.next_frame(3000) {
            Ok(f) => f,
            Err(e) => {
                println!("frame error: {e}");
                break;
            }
        };
        if frame.device_time_us.is_some() {
            n_device += 1;
        }
        // A jump in the camera's own counter means the SDK dropped frames
        // before they ever reached us — invisible in the file otherwise.
        if let (Some(prev), Some(cur)) = (last_seq, frame.seq) {
            if cur > prev + 1 {
                seq_gaps += (cur - prev - 1) as usize;
            }
        }
        last_seq = frame.seq;
        let r = rec.get_or_insert_with(|| {
            SerRecorder::create(&path, frame.width, frame.height, &info.name, "Timing check")
                .expect("create SER")
        });
        r.write_frame(
            &frame.data,
            FrameTime { host: frame.host_time, device_us: frame.device_time_us },
        )
        .expect("write frame");
    }
    cam.stop();

    let Some(rec) = rec else {
        println!("no frames captured");
        return;
    };
    let summary = rec.finish().expect("finish SER");
    println!(
        "\nwrote {} — {} frames, {}/{} carried a camera clock, {} frame(s) dropped by the SDK (seq gaps)",
        path.display(), summary.frames, n_device, summary.frames, seq_gaps
    );
    println!(
        "  trailer clock : {}",
        if summary.device_clock { "CAMERA (crystal spacing)" } else { "host stamps" }
    );
    if let Some(fps) = summary.fps() {
        println!("  trailer fps   : {fps:.3}");
    }

    // Reopen and report exactly what reconstruction would see.
    let reader = SerReader::open(&path).expect("reopen SER");
    match reader.timestamps.as_ref() {
        None => println!("  TRAILER MISSING on read-back"),
        Some(t) => println!("  read back     : {} timestamps", t.len()),
    }
    match ScanTiming::from_reader(&reader) {
        None => println!("  ScanTiming declined the timestamps"),
        Some(st) => {
            println!("  ScanTiming    : {}", st.summary());
            println!(
                "  regrid        : {}",
                if st.worth_regridding() {
                    format!("yes -> {} columns", st.grid_w)
                } else {
                    "no (cadence uniform within the deadband)".to_string()
                }
            );
            // Spacing steadiness, the whole point of preferring the camera
            // clock: intervals in ms, and how far they spread.
            let mut iv: Vec<f64> = st.seconds.windows(2).map(|w| (w[1] - w[0]) * 1e3).collect();
            iv.sort_by(|a, b| a.total_cmp(b));
            if !iv.is_empty() {
                println!(
                    "  interval ms   : min {:.4}  median {:.4}  max {:.4}  (spread {:.4})",
                    iv[0],
                    iv[iv.len() / 2],
                    iv[iv.len() - 1],
                    iv[iv.len() - 1] - iv[0]
                );
            }
            println!(
                "  first frame   : {} UTC",
                ghostsun_core::ser::ticks_to_iso8601(st.first_utc_ticks)
            );
            let drift_s = (st.last_utc_ticks - st.first_utc_ticks) as f64 / TICKS_PER_SECOND as f64;
            println!("  span          : {drift_s:.3} s");
        }
    }
    let _ = std::fs::remove_file(&path);
}
