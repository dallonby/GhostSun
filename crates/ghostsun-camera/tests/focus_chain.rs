//! End-to-end verification of the focus measurement chain, no hardware:
//! synth frame → mean profile → core `fit_line_1d`. The recovered FWHM must
//! track the *known* injected sigma across a simulated focus sweep, and the
//! min-hold must land near the sharpest point (sigma 1.4). This is the ground
//! truth gate for the whole feature before a camera is ever attached.

use ghostsun_camera::{open, synth, Backend};
use ghostsun_core::linefit::{fit_line_1d, FWHM_PER_SIGMA};

#[test]
fn measured_fwhm_tracks_injected_sigma() {
    let info = synth::enumerate()
        .into_iter()
        .find(|c| c.backend == Backend::Synth)
        .expect("synth camera present");
    let mut cam = open(&info).expect("open synth");
    cam.start().unwrap();

    let mut max_err = 0.0_f64;
    let mut min_measured_fwhm = f64::MAX;
    // Two full sweeps' worth of frames.
    for frame in 0..480u64 {
        let f = cam.next_frame(100).expect("frame");
        let prof = f.mean_profile(true); // synth spectrum is dispersion-horizontal
        let fit = fit_line_1d(&prof, 0.03).expect("line fit");
        let truth_fwhm = FWHM_PER_SIGMA * synth::swept_sigma(frame);
        max_err = max_err.max((fit.fwhm - truth_fwhm).abs());
        min_measured_fwhm = min_measured_fwhm.min(fit.fwhm);
    }

    // Recovered width tracks truth to well under a pixel across the sweep...
    assert!(max_err < 0.5, "max FWHM error {max_err:.3} px too large");
    // ...and the min-hold finds the sharp end (sigma 1.4 => FWHM ~3.30 px).
    let sharp = FWHM_PER_SIGMA * 1.4;
    assert!(
        (min_measured_fwhm - sharp).abs() < 0.4,
        "min-hold FWHM {min_measured_fwhm:.3} px, expected ~{sharp:.3}"
    );
}
