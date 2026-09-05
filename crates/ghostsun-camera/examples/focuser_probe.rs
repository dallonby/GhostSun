//! Read-only SDK check: enumerate and query AAF devices, without moving them.
fn main() -> Result<(), String> {
    let devices = ghostsun_camera::focuser::enumerate()?;
    println!("Found {} ToupTek focuser(s)", devices.len());
    for info in devices {
        println!("{} ({})", info.name, info.id);
        let device = ghostsun_camera::focuser::Focuser::open(&info.id)?;
        println!("{:?}", device.state()?);
    }
    for info in ghostsun_camera::toupcam::enumerate() {
        println!("Camera (separate list): {} ({})", info.name, info.id);
    }
    Ok(())
}
