//! ToupTek diagnostic using the same ABI and DLL discovery as the app.
//!
//!     cargo run -p ghostsun-camera --example diag --release

use ghostsun_camera::{enumerate_all, toupcam, Backend};

fn main() {
    match toupcam::probe() {
        Ok(n) => println!("ToupTek SDK loaded; EnumV2 reports {n} camera(s)"),
        Err(e) => {
            println!("{e}");
            println!(
                "Place the 64-bit toupcam.dll beside GhostSun.exe or set \
                 GHOSTSUN_TOUPCAM_LIB to its full path."
            );
            return;
        }
    }

    for camera in enumerate_all()
        .into_iter()
        .filter(|c| c.backend == Backend::Toupcam)
    {
        println!(
            "[{}] {} ({}x{})",
            camera.id, camera.name, camera.max_width, camera.max_height
        );
    }
}
