//! The synthetic camera must honour hardware ROI exactly like a vendor
//! backend: the acquisition recording path relies on frames arriving
//! pre-cropped when a ROI is active, and on the full sensor returning after
//! the stop → set_roi → start restore cycle.

use ghostsun_camera::{enumerate_all, open, Backend, Roi};

#[test]
fn synth_camera_honours_roi_and_restore() {
    let info = enumerate_all()
        .into_iter()
        .find(|c| c.backend == Backend::Synth)
        .expect("synthetic camera always enumerates");
    let mut cam = open(&info).expect("open synth");

    cam.set_roi(Roi {
        x: 0,
        y: 40,
        w: info.max_width,
        h: 64,
    })
    .expect("set band roi");
    cam.start().expect("start with roi");
    let frame = cam.next_frame(2000).expect("band frame");
    assert_eq!(
        (frame.width, frame.height),
        (info.max_width, 64),
        "the delivered frame must be the requested band"
    );

    // The recording teardown: stop → full ROI → start.
    cam.stop();
    cam.set_roi(Roi {
        x: 0,
        y: 0,
        w: info.max_width,
        h: info.max_height,
    })
    .expect("restore full roi");
    cam.start().expect("restart full");
    let frame = cam.next_frame(2000).expect("full frame");
    assert_eq!(
        (frame.width, frame.height),
        (info.max_width, info.max_height),
        "the full sensor must come back after restore"
    );
}
