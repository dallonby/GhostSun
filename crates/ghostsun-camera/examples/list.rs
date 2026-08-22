//! Hardware smoke test: enumerate all backends, and for the first real camera
//! (ToupTek or ZWO) open it, pull one frame, and measure the line FWHM through
//! the same core estimator the focus view uses.
//!
//!   cargo run -p ghostsun-camera --example list --release

use ghostsun_camera::{enumerate_all, open, Backend};
use ghostsun_core::linefit::fit_line_1d;

fn main() {
    let cams = enumerate_all();
    println!("Discovered {} camera(s):", cams.len());
    for c in &cams {
        println!(
            "  [{}] {:<8} {}  ({}x{})",
            c.id,
            c.backend.label(),
            c.name,
            c.max_width,
            c.max_height
        );
    }

    let Some(info) = cams.iter().find(|c| c.backend != Backend::Synth) else {
        println!("\nNo hardware camera found (synthetic only). Nothing to open.");
        return;
    };

    println!("\nOpening {} …", info.name);
    let mut cam = match open(info) {
        Ok(c) => c,
        Err(e) => {
            println!("open failed: {e}");
            return;
        }
    };
    // Ranges as the open handle reports them — the gain ceiling is queried
    // from the device and may exceed the enumerate-time guess.
    let oi = cam.info();
    println!(
        "  exposure {}..{} µs   gain {}..{}",
        oi.exposure_us.start(),
        oi.exposure_us.end(),
        oi.gain.start(),
        oi.gain.end()
    );
    cam.set_exposure_us(10_000).ok();
    cam.set_gain(200).ok();
    if let Err(e) = cam.start() {
        println!("start failed: {e}");
        return;
    }
    // Per-frame timing: the camera clock (device_time_us) should step with
    // crystal regularity while the host stamp jitters around it.
    let mut prev: Option<(std::time::SystemTime, Option<u64>)> = None;
    for i in 0..12 {
        match cam.next_frame(3000) {
            Ok(f) => {
                let prof = f.mean_profile(true);
                let mean: f64 = prof.iter().sum::<f64>() / prof.len().max(1) as f64;
                let fwhm = fit_line_1d(&prof, 0.02)
                    .map(|l| format!("{:.2} px (depth {:.0}%)", l.fwhm, l.depth * 100.0))
                    .unwrap_or_else(|| "no line".into());
                let host_d = prev
                    .and_then(|(h, _)| f.host_time.duration_since(h).ok())
                    .map(|d| format!("{:8.3} ms", d.as_secs_f64() * 1e3))
                    .unwrap_or_else(|| "       --".into());
                let dev_d = match (prev.and_then(|(_, d)| d), f.device_time_us) {
                    (Some(a), Some(b)) => format!("{:8.3} ms", (b as f64 - a as f64) / 1e3),
                    _ => "       --".into(),
                };
                let tier = match (f.device_time_us, f.seq) {
                    (Some(_), _) => "camera clock",
                    (None, Some(_)) => "seq only",
                    _ => "host clock",
                };
                println!(
                    "frame {i:2}: {}x{}  mean={:.0}  seq={:?}  dev_us={:?}  [{tier}]  Δhost={host_d}  Δdev={dev_d}  FWHM={fwhm}",
                    f.width, f.height, mean, f.seq, f.device_time_us
                );
                prev = Some((f.host_time, f.device_time_us));
            }
            Err(e) => println!("frame {i}: {e}"),
        }
    }
    cam.stop();
    println!("done.");
}
