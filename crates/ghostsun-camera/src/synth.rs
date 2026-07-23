//! Synthetic camera: renders a spectrum frame with an absorption line whose
//! width sweeps triangularly over time, simulating racking through focus. It
//! lets the whole focus UI + FWHM estimator be exercised and verified against a
//! *known* line width with no hardware attached — the same measured-truth
//! discipline the reconstruction pipeline uses.

use crate::{Backend, Camera, CameraInfo, Frame, Roi};

const W: usize = 512;
const H: usize = 200;
const CONTINUUM: f64 = 42_000.0;
const CORE_COL: f64 = 0.44 * W as f64; // main (deep) line
const TELLURIC_COL: f64 = 0.72 * W as f64; // a shallow narrow line elsewhere
const SWEEP_FRAMES: u64 = 240; // one focus in-and-out cycle

/// Sigma (px) of the main line at a given frame: triangular 1.4 → 5.4 → 1.4.
/// Exposed so tests can assert the estimator recovers exactly this.
pub fn swept_sigma(frame: u64) -> f64 {
    let phase = (frame % SWEEP_FRAMES) as f64 / SWEEP_FRAMES as f64;
    let tri = if phase < 0.5 { phase * 2.0 } else { 2.0 - phase * 2.0 };
    1.4 + tri * 4.0
}

pub fn enumerate() -> Vec<CameraInfo> {
    vec![CameraInfo {
        backend: Backend::Synth,
        id: "synth-0".into(),
        name: "Synthetic spectrum (focus sweep)".into(),
        max_width: W,
        max_height: H,
        exposure_us: 100..=1_000_000,
        gain: 0..=100,
    }]
}

pub fn open(info: &CameraInfo) -> crate::Result<Box<dyn Camera>> {
    Ok(Box::new(SynthCam {
        info: info.clone(),
        frame: 0,
        roi: Roi { x: 0, y: 0, w: W, h: H },
    }))
}

struct SynthCam {
    info: CameraInfo,
    frame: u64,
    roi: Roi,
}

/// Tiny deterministic PRNG (xorshift64*) for read noise — no external dep.
fn rng(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11; // 53-bit mantissa
    (v as f64) / (1u64 << 53) as f64 // 0..1
}

impl Camera for SynthCam {
    fn info(&self) -> &CameraInfo {
        &self.info
    }
    fn set_exposure_us(&mut self, _us: u32) -> crate::Result<()> {
        Ok(())
    }
    fn set_gain(&mut self, _gain: u16) -> crate::Result<()> {
        Ok(())
    }
    fn set_roi(&mut self, roi: Roi) -> crate::Result<()> {
        self.roi = roi;
        Ok(())
    }
    fn start(&mut self) -> crate::Result<()> {
        Ok(())
    }
    fn next_frame(&mut self, _timeout_ms: u32) -> crate::Result<Frame> {
        let sigma = swept_sigma(self.frame);
        let core_amp = 0.70 * CONTINUUM;
        let tell_amp = 0.18 * CONTINUUM;
        let tell_sigma = 1.1; // stays sharp — an intrinsically narrow anchor
        let two_s2 = 2.0 * sigma * sigma;
        let two_ts2 = 2.0 * tell_sigma * tell_sigma;
        let mut seed = 0x9E37_79B9_7F4A_7C15 ^ self.frame.wrapping_mul(0x1000_0001B);

        // Full-sensor render, then crop to ROI — mirrors how a real SDK applies
        // ROI, and keeps the fitter honest about coordinates.
        let mut full = vec![0u16; W * H];
        for y in 0..H {
            // gentle vertical continuum gradient (slit-illumination shape)
            let vy = 0.85 + 0.15 * (std::f64::consts::PI * y as f64 / H as f64).sin();
            for x in 0..W {
                let dxc = x as f64 - CORE_COL;
                let dxt = x as f64 - TELLURIC_COL;
                let line = core_amp * (-dxc * dxc / two_s2).exp()
                    + tell_amp * (-dxt * dxt / two_ts2).exp();
                let noise = (rng(&mut seed) - 0.5) * 0.01 * CONTINUUM;
                let v = (CONTINUUM * vy - line + noise).clamp(0.0, 65_535.0);
                full[y * W + x] = v as u16;
            }
        }

        let Roi { x, y, w, h } = self.roi;
        let (x, y) = (x.min(W - 1), y.min(H - 1));
        let (w, h) = (w.min(W - x).max(1), h.min(H - y).max(1));
        let mut data = vec![0u16; w * h];
        for r in 0..h {
            let src = (y + r) * W + x;
            data[r * w..(r + 1) * w].copy_from_slice(&full[src..src + w]);
        }

        self.frame = self.frame.wrapping_add(1);
        Ok(Frame { width: w, height: h, data })
    }
    fn stop(&mut self) {}
}
