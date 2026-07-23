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
    cam.set_exposure_us(10_000).ok();
    cam.set_gain(200).ok();
    if let Err(e) = cam.start() {
        println!("start failed: {e}");
        return;
    }
    for i in 0..5 {
        match cam.next_frame(3000) {
            Ok(f) => {
                let prof = f.mean_profile(true);
                let mean: f64 = prof.iter().sum::<f64>() / prof.len().max(1) as f64;
                let fwhm = fit_line_1d(&prof, 0.02)
                    .map(|l| format!("{:.2} px (depth {:.0}%)", l.fwhm, l.depth * 100.0))
                    .unwrap_or_else(|| "no line".into());
                println!(
                    "frame {i}: {}x{}  mean={:.0}  deepest-line FWHM(horiz)={fwhm}",
                    f.width, f.height, mean
                );
            }
            Err(e) => println!("frame {i}: {e}"),
        }
    }
    cam.stop();
    println!("done.");
}
