//! Per-frame acquisition timing: the SER timestamp trailer as the scan-axis
//! coordinate.
//!
//! The rest of the pipeline assumes `column index == frame index == time`.
//! That is an approximation. A spectroheliograph scan IS a time series: the
//! sun drifts across the slit at a rate set by the mount, so a frame's true
//! scan-axis position is proportional to its acquisition TIME, not to its
//! index in the file. Two things break the identity:
//!
//! * **cadence jitter** — USB delivery and exposure scheduling make the frame
//!   interval wobble by a few percent. A frame captured late belongs a
//!   fraction of a column to the right of where its index puts it.
//! * **dropped frames** — the sun keeps moving while nothing is recorded, so
//!   every subsequent frame is displaced by a whole column (or more) and the
//!   gap itself is silently closed up.
//!
//! Both are pure scan-axis displacements, which is exactly what F9.2's
//! x-registration estimates from image texture — noisily, since its raw
//! measurement is a second difference of the trajectory. Timestamps measure
//! the same quantity directly and without noise, so when a file carries them
//! the geometry is *known* rather than inferred, and x-registration is left
//! with only the seeing-induced part it is actually good at.
//!
//! [`ScanTiming::from_reader`] builds the model; [`regrid_columns`] resamples
//! a raw disk from frame-index columns onto a uniform time grid. Both are
//! no-ops (`None` / identity) for files without a trailer, so legacy data and
//! synthetic scans behave exactly as before.

use crate::image2d::Image;
use crate::mathutil::{bspline_eval, bspline_prefilter, median_inplace, polyfit_robust};
use crate::ser::{SerReader, TICKS_PER_SECOND};
use rayon::prelude::*;

/// Below this displacement (in frames) regridding is skipped: resampling to
/// move content by less than the B-spline's own error is a loss, not a gain.
/// Matches the deadband in `jitter::apply_x_offsets`.
const DEADBAND_FRAMES: f64 = 0.04;

/// Acquisition timing of one scan, in units of the nominal frame interval.
#[derive(Clone, Debug)]
pub struct ScanTiming {
    /// Seconds since the first frame, one entry per frame.
    pub seconds: Vec<f64>,
    /// Nominal frame interval (s) — the robust slope of time vs frame index.
    pub interval: f64,
    /// True scan-axis position of each frame, in nominal-frame units, with
    /// frame 0 at 0. Strictly increasing.
    pub position: Vec<f64>,
    /// Departure of each frame from uniform sampling, in columns, after
    /// removing the overall rate (a uniformly fast camera scores zero).
    pub residual: Vec<f64>,
    /// Local cadence non-uniformity: rms of `interval/typical - 1` over
    /// non-gap intervals.
    pub jitter_rms: f64,
    /// Largest local interval departure (fraction of a column).
    pub jitter_max: f64,
    /// RMS cumulative displacement from uniform sampling, in columns. This is
    /// the quantity that misplaces solar features when timing is ignored.
    pub drift_rms: f64,
    /// Largest cumulative displacement from uniform sampling, in columns.
    pub drift_max: f64,
    /// Scan slots with no frame in them — dropped frames, counted globally.
    pub dropped: usize,
    /// Width of the uniform time grid covering the scan. Equals the frame
    /// count when nothing was dropped.
    pub grid_w: usize,
    /// UTC (.NET ticks) of the first and last frame.
    pub first_utc_ticks: i64,
    pub last_utc_ticks: i64,
}

impl ScanTiming {
    /// Build the timing model from a SER file's trailer. `None` when the file
    /// has no per-frame timestamps, has fewer than 8 frames (too few to fit a
    /// cadence), or its timestamps are degenerate (no elapsed time).
    pub fn from_reader(reader: &SerReader) -> Option<ScanTiming> {
        let ticks = reader.timestamps.as_ref()?;
        if ticks.len() < 8 {
            return None;
        }
        Self::from_ticks(ticks)
    }

    /// Build the model from raw .NET tick counts (one per frame).
    pub fn from_ticks(ticks: &[i64]) -> Option<ScanTiming> {
        let n = ticks.len();
        if n < 8 {
            return None;
        }
        let t0 = ticks[0];
        let seconds: Vec<f64> = ticks
            .iter()
            .map(|&t| (t - t0) as f64 / TICKS_PER_SECOND as f64)
            .collect();
        // A non-monotonic trailer means the writer's clock was disciplined
        // backwards mid-scan; the index is then more trustworthy than the
        // clock, so decline rather than reorder frames.
        if seconds.windows(2).any(|w| w[1] < w[0]) {
            return None;
        }
        let span = *seconds.last()?;
        if !(span.is_finite() && span > 0.0) {
            return None;
        }

        // Grid spacing = the TYPICAL inter-frame interval.
        //
        // Real cameras free-run: they deliver as fast as USB and exposure
        // allow, so there is no underlying frame clock to snap frames to (the
        // user's scans show a continuous 1.4-2.9 ms spread around a 2.2 ms
        // median, not a quantized grid). The output grid spacing is therefore
        // ours to choose, and the typical interval is the honest choice: it
        // neither up-samples (inventing resolution) nor down-samples
        // (discarding it) where frames are dense, and spans real gaps with
        // however many columns the elapsed time deserves.
        //
        // The median sets a robust scale (immune to however many frames went
        // missing), then the mean of the non-gap intervals refines it: the
        // median alone quantizes onto one observed interval, while the mean
        // uses the whole delivery distribution. Two iterations are enough for
        // the gap cut to settle.
        let diffs: Vec<f64> = seconds.windows(2).map(|w| w[1] - w[0]).collect();
        let mut scale = {
            let mut m = diffs.clone();
            median_inplace(&mut m).max(1e-12)
        };
        for _ in 0..2 {
            let body: Vec<f64> = diffs.iter().copied().filter(|d| *d <= 1.5 * scale).collect();
            if body.is_empty() {
                break;
            }
            let mean = body.iter().sum::<f64>() / body.len() as f64;
            if !(mean.is_finite() && mean > 0.0) {
                break;
            }
            scale = mean;
        }
        let mut interval = scale;
        if !(interval.is_finite() && interval > 0.0) {
            interval = span / (n as f64 - 1.0);
        }

        let position: Vec<f64> = seconds.iter().map(|s| s / interval).collect();
        let grid_w = ((position[n - 1] + 1e-6).round() as usize)
            .saturating_add(1)
            .max(2);

        // Two different things are worth reporting, and conflating them hides
        // the one that matters:
        //
        //   jitter  - LOCAL non-uniformity: how much each interval departs
        //             from the typical one. Visible as column-to-column
        //             sampling irregularity.
        //   drift   - CUMULATIVE departure from uniform sampling, the running
        //             sum of that jitter minus its own linear trend. This is
        //             what actually displaces solar features, and because the
        //             per-interval errors random-walk it grows as sqrt(n):
        //             small local jitter still reaches many columns by the end
        //             of a long scan.
        let idx: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let w0 = vec![1.0; n];
        // Gaps are excluded from the local-jitter statistic: a 60 ms stall is
        // missing data, not a description of how steadily the camera ran.
        let nongap: Vec<f64> = diffs
            .iter()
            .copied()
            .filter(|d| *d <= 1.5 * interval)
            .map(|d| d / interval - 1.0)
            .collect();
        let jitter_rms = if nongap.is_empty() {
            0.0
        } else {
            (nongap.iter().map(|d| d * d).sum::<f64>() / nongap.len() as f64).sqrt()
        };
        let jitter_max = nongap.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        // Drift: residual of position about its own straight line. A camera
        // running uniformly fast or slow is a pure scale error and scores
        // zero here — the ellipse fit absorbs that later.
        let residual: Vec<f64> = match polyfit_robust(&idx, &position, &w0, 1, 4) {
            Some(c) => position
                .iter()
                .enumerate()
                .map(|(k, p)| p - (c[0] + c[1] * k as f64))
                .collect(),
            None => vec![0.0; n],
        };
        let drift_rms =
            (residual.iter().map(|d| d * d).sum::<f64>() / residual.len() as f64).sqrt();
        let drift_max = residual.iter().fold(0.0f64, |m, d| m.max(d.abs()));

        // Grid columns with no frame in them: dropped frames, plus the slack
        // that gaps open up. Counted from the grid, not per-interval, because
        // a single 60 ms stall is many missing columns.
        let dropped = grid_w.saturating_sub(n);
        Some(ScanTiming {
            seconds,
            interval,
            position,
            residual,
            jitter_rms,
            jitter_max,
            drift_rms,
            drift_max,
            dropped,
            grid_w,
            first_utc_ticks: ticks[0],
            last_utc_ticks: ticks[n - 1],
        })
    }

    /// Mean frame rate over the scan (Hz).
    pub fn fps(&self) -> f64 {
        1.0 / self.interval
    }

    /// Total scan duration (s).
    pub fn duration(&self) -> f64 {
        *self.seconds.last().unwrap_or(&0.0)
    }

    /// Mid-scan UTC in .NET ticks — the epoch describing the whole disk.
    pub fn mid_utc_ticks(&self) -> i64 {
        self.first_utc_ticks + (self.last_utc_ticks - self.first_utc_ticks) / 2
    }

    /// Is the departure from uniform cadence large enough to be worth a
    /// resample? Sub-deadband jitter with no dropped frames is not: the
    /// interpolation would cost more than the misplacement it corrects.
    pub fn worth_regridding(&self) -> bool {
        self.dropped > 0 || self.drift_max > DEADBAND_FRAMES
    }

    /// Input-column coordinate to sample for each column of the uniform time
    /// grid: `src[x]` is the (fractional) frame index whose true scan position
    /// is `x`. The inverse of `position`, by linear interpolation between
    /// bracketing frames — exact at the frames themselves, and across a gap it
    /// interpolates the missing columns from the frames on either side.
    pub fn source_columns(&self) -> Vec<f64> {
        let n = self.position.len();
        let mut src = Vec::with_capacity(self.grid_w);
        let mut k = 0usize;
        for x in 0..self.grid_w {
            let xf = x as f64;
            // position is strictly increasing, so a single forward walk
            // suffices for the whole grid.
            while k + 2 < n && self.position[k + 1] <= xf {
                k += 1;
            }
            let (p0, p1) = (self.position[k], self.position[k + 1]);
            let f = if (p1 - p0).abs() > 1e-12 {
                (xf - p0) / (p1 - p0)
            } else {
                0.0
            };
            src.push((k as f64 + f).clamp(0.0, (n - 1) as f64));
        }
        src
    }

    /// Compact record for `ReconReport` and the FITS header — the scalars,
    /// without the per-frame vectors.
    pub fn summarize(&self, regridded: bool) -> TimingSummary {
        TimingSummary {
            frames: self.position.len(),
            fps: self.fps(),
            duration_s: self.duration(),
            jitter_rms: self.jitter_rms,
            jitter_max: self.jitter_max,
            drift_rms: self.drift_rms,
            drift_max: self.drift_max,
            dropped: self.dropped,
            grid_w: self.grid_w,
            regridded,
            first_utc_ticks: self.first_utc_ticks,
            last_utc_ticks: self.last_utc_ticks,
            mid_utc_ticks: self.mid_utc_ticks(),
        }
    }

    /// One-line summary for the reconstruction log.
    pub fn summary(&self) -> String {
        format!(
            "{:.2} fps ({:.1} s, {} frames), cadence jitter {:.1}% rms, drift {:.2} col rms / {:.2} max, {} empty column(s) -> grid {}",
            self.fps(),
            self.duration(),
            self.position.len(),
            100.0 * self.jitter_rms,
            self.drift_rms,
            self.drift_max,
            self.dropped,
            self.grid_w
        )
    }
}

/// Scan timing as recorded in a reconstruction report / FITS header.
#[derive(Clone, Copy, Debug)]
pub struct TimingSummary {
    pub frames: usize,
    pub fps: f64,
    pub duration_s: f64,
    pub jitter_rms: f64,
    pub jitter_max: f64,
    pub drift_rms: f64,
    pub drift_max: f64,
    pub dropped: usize,
    pub grid_w: usize,
    /// The disk was resampled onto the uniform time grid.
    pub regridded: bool,
    pub first_utc_ticks: i64,
    pub last_utc_ticks: i64,
    pub mid_utc_ticks: i64,
}

impl TimingSummary {
    /// Acquisition cards for the FITS header.
    pub fn fits_meta(&self) -> crate::output::FitsMeta {
        use crate::ser::ticks_to_iso8601;
        crate::output::FitsMeta {
            date_obs: Some(ticks_to_iso8601(self.mid_utc_ticks)),
            date_beg: Some(ticks_to_iso8601(self.first_utc_ticks)),
            date_end: Some(ticks_to_iso8601(self.last_utc_ticks)),
            exptime: Some(self.duration_s),
            cadence_fps: Some(self.fps),
            frames: Some(self.frames),
            dropped: Some(self.dropped),
        }
    }
}

/// Resample an image's columns: output column `x` is sampled from input
/// column `src[x]`, with prefiltered cubic B-splines along the scan axis (the
/// same interpolator the extraction and x-registration stages use, so the
/// spatial-frequency response of the pipeline stays uniform).
///
/// Output width is `src.len()`, which may exceed the input width when frames
/// were dropped — the missing columns are then interpolated from the frames
/// bracketing the gap, placing surviving structure at its true scan position
/// instead of closing the gap up.
pub fn regrid_columns(img: &Image, src: &[f64]) -> Image {
    let (w, h, out_w) = (img.w, img.h, src.len());
    if w < 2 || out_w == 0 {
        return img.clone();
    }
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut coef: Vec<f64> = img.row(y).iter().map(|&v| v as f64).collect();
            bspline_prefilter(&mut coef);
            src.iter()
                .enumerate()
                .map(|(x, &s)| {
                    // Untouched columns keep their exact samples: B-spline
                    // evaluation at an integer position is exact, but skipping
                    // it avoids accumulating float error over many stages.
                    if (s - x as f64).abs() < DEADBAND_FRAMES && x < w {
                        img.at(x, y)
                    } else {
                        bspline_eval(&coef, s.clamp(0.0, (w - 1) as f64)) as f32
                    }
                })
                .collect()
        })
        .collect();
    let mut out = Image::new(out_w, h);
    for (y, r) in rows.iter().enumerate() {
        out.row_mut(y).copy_from_slice(r);
    }
    out
}

/// Resample a per-frame signal (a jitter trajectory, a flexure curve) onto the
/// same uniform time grid, so per-column vectors stay aligned with the regridded
/// disk. Linear interpolation: these are already smooth, low-order curves.
pub fn regrid_series(values: &[f64], src: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0; src.len()];
    }
    let n = values.len();
    src.iter()
        .map(|&s| {
            let s = s.clamp(0.0, (n - 1) as f64);
            let i = s.floor() as usize;
            if i + 1 >= n {
                values[n - 1]
            } else {
                let f = s - i as f64;
                values[i] * (1.0 - f) + values[i + 1] * f
            }
        })
        .collect()
}

/// Which grid columns were interpolated across a gap rather than measured.
/// A column counts as unmeasured when no real frame lies within half a frame
/// of the position it samples.
pub fn gap_columns(timing: &ScanTiming, src: &[f64]) -> Vec<bool> {
    let pos = &timing.position;
    src.iter()
        .enumerate()
        .map(|(x, &s)| {
            let k = s.round().clamp(0.0, (pos.len() - 1) as f64) as usize;
            (pos[k] - x as f64).abs() > 0.5
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ser::{synth_frame_ticks, TICKS_PER_SECOND};

    fn ticks_at(seconds: &[f64]) -> Vec<i64> {
        let base = synth_frame_ticks(0);
        seconds
            .iter()
            .map(|s| base + (s * TICKS_PER_SECOND as f64).round() as i64)
            .collect()
    }

    #[test]
    fn perfect_cadence_is_an_exact_identity() {
        let secs: Vec<f64> = (0..64).map(|k| k as f64 * 0.01).collect();
        let t = ScanTiming::from_ticks(&ticks_at(&secs)).unwrap();
        assert!((t.fps() - 100.0).abs() < 1e-6, "fps {}", t.fps());
        assert_eq!(t.dropped, 0);
        assert_eq!(t.grid_w, 64);
        assert!(t.jitter_max < 1e-6, "jitter {}", t.jitter_max);
        assert!(t.drift_max < 1e-6, "drift {}", t.drift_max);
        assert!(!t.worth_regridding(), "uniform cadence must not resample");
        // src == identity, so a regrid would return the image untouched.
        for (x, s) in t.source_columns().iter().enumerate() {
            assert!((s - x as f64).abs() < 1e-6, "src[{x}] = {s}");
        }
    }

    #[test]
    fn dropped_frames_widen_the_grid_and_place_survivors_correctly() {
        // 40 nominal slots at 100 fps, but slots 10, 11 and 25 never arrive.
        let secs: Vec<f64> = (0..40)
            .filter(|k| ![10, 11, 25].contains(k))
            .map(|k| k as f64 * 0.01)
            .collect();
        let t = ScanTiming::from_ticks(&ticks_at(&secs)).unwrap();
        assert_eq!(t.dropped, 3);
        assert_eq!(t.grid_w, 40, "grid spans the true scan extent");
        assert!(t.drift_max > 0.5, "drops displace frames: {}", t.drift_max);
        assert!(t.worth_regridding());
        let src = t.source_columns();
        // Frame index of the data that belongs in output column 30: three
        // frames are missing before it, so it is input column 27.
        assert!((src[30] - 27.0).abs() < 1e-6, "src[30] = {}", src[30]);
        assert!((src[9] - 9.0).abs() < 1e-6);
        let gaps = gap_columns(&t, &src);
        assert!(gaps[10] && gaps[11] && gaps[25], "gap slots flagged");
        assert!(!gaps[9] && !gaps[12] && !gaps[30], "measured slots not flagged");
        assert_eq!(gaps.iter().filter(|g| **g).count(), 3);
    }

    #[test]
    fn regrid_reconstructs_a_ramp_across_a_gap() {
        // A linear ramp along the scan axis, sampled with a missing frame:
        // regridding must restore the true value at the missing slot, which
        // simply dropping the column cannot do.
        let secs: Vec<f64> = (0..32)
            .filter(|k| *k != 16)
            .map(|k| k as f64 * 0.01)
            .collect();
        let t = ScanTiming::from_ticks(&ticks_at(&secs)).unwrap();
        let src = t.source_columns();
        let mut img = Image::new(31, 3);
        // Column values follow the frame's TRUE position, not its index.
        let positions: Vec<f64> = (0..32).filter(|k| *k != 16).map(|k| k as f64).collect();
        for (x, p) in positions.iter().enumerate() {
            for y in 0..3 {
                img.set(x, y, (2.0 * p + 1.0) as f32);
            }
        }
        let out = regrid_columns(&img, &src);
        assert_eq!(out.w, 32);
        for x in 0..32 {
            let want = 2.0 * x as f32 + 1.0;
            assert!(
                (out.at(x, 1) - want).abs() < 1e-3,
                "column {x}: {} vs {want}",
                out.at(x, 1)
            );
        }
    }

    #[test]
    fn cadence_jitter_is_measured_but_a_slow_rate_error_is_not() {
        // Continuous delivery jitter of roughly +-0.15 of a frame interval —
        // the scale of a real USB hiccup at 100 fps. Real jitter is a spread,
        // not an alternation, so the intervals form one broad mode.
        let mut secs = vec![0.0f64];
        let mut t = 0.0f64;
        for k in 1..64 {
            t += 0.01 * (1.0 + 0.15 * ((k as f64) * 1.7).sin());
            secs.push(t);
        }
        let t = ScanTiming::from_ticks(&ticks_at(&secs)).unwrap();
        assert!(t.jitter_rms > 0.05, "jitter_rms {}", t.jitter_rms);
        assert!(t.drift_max > 0.05, "drift_max {}", t.drift_max);
        assert!(t.worth_regridding());
        assert_eq!(t.dropped, 0, "jitter is not a dropped frame");
        assert_eq!(t.grid_w, 64);
        assert!((t.fps() - 100.0).abs() < 2.0, "fps {}", t.fps());

        // A camera running 3% slower than nominal is a pure scale error: the
        // ellipse fit absorbs it, so it must NOT show up as jitter or drift.
        let slow: Vec<f64> = (0..64).map(|k| k as f64 * 0.0103).collect();
        let ts = ScanTiming::from_ticks(&ticks_at(&slow)).unwrap();
        assert!(ts.jitter_max < 1e-6, "rate error leaked as jitter: {}", ts.jitter_max);
        assert!(ts.drift_max < 1e-6, "rate error leaked as drift: {}", ts.drift_max);
        assert!(!ts.worth_regridding());
        assert!((ts.fps() - 97.087).abs() < 0.01, "fps {}", ts.fps());
    }

    #[test]
    fn a_free_running_camera_gets_a_grid_matched_to_its_typical_spacing() {
        // Real cameras deliver on a continuous spread of intervals rather
        // than a clock. The grid must track the TYPICAL spacing, so dense
        // stretches are neither up- nor down-sampled, and the occasional
        // long stall becomes empty columns instead of stretching the scan.
        let mut secs = vec![0.0f64];
        let mut t = 0.0;
        for k in 1..400 {
            // 2.2 ms typical with a deterministic +-30% wobble, and one 60 ms
            // stall partway through — the shape of the user's real scans.
            let wobble = 1.0 + 0.3 * ((k as f64) * 2.399).sin();
            t += if k == 200 { 0.060 } else { 0.0022 * wobble };
            secs.push(t);
        }
        let st = ScanTiming::from_ticks(&ticks_at(&secs)).unwrap();
        assert!((st.fps() - 454.0).abs() < 25.0, "fps {}", st.fps());
        assert!(st.jitter_rms > 0.1, "local wobble measured: {}", st.jitter_rms);
        // The 60 ms stall is ~27 typical intervals: empty columns, not cadence.
        assert!(st.dropped >= 20, "stall became {} empty columns", st.dropped);
        assert!(st.grid_w > 400, "grid spans the stall: {}", st.grid_w);
        assert!(st.worth_regridding());
        let src = st.source_columns();
        assert_eq!(src.len(), st.grid_w);
        assert!(src.windows(2).all(|w| w[1] >= w[0]), "monotonic mapping");
        // Every real frame is still reachable from the grid.
        assert!(*src.last().unwrap() >= 398.0, "last frame reachable: {}", src.last().unwrap());
        let gaps = gap_columns(&st, &src);
        assert!(gaps.iter().filter(|g| **g).count() >= 20, "stall columns flagged");
    }

    #[test]
    fn degenerate_or_backwards_clocks_are_declined() {
        assert!(ScanTiming::from_ticks(&[]).is_none());
        assert!(ScanTiming::from_ticks(&ticks_at(&[0.0, 0.01, 0.02])).is_none(), "too few frames");
        let flat: Vec<f64> = vec![0.0; 32];
        assert!(ScanTiming::from_ticks(&ticks_at(&flat)).is_none(), "no elapsed time");
        let mut back: Vec<f64> = (0..32).map(|k| k as f64 * 0.01).collect();
        back[20] = 0.05; // clock stepped backwards mid-scan
        assert!(ScanTiming::from_ticks(&ticks_at(&back)).is_none(), "non-monotonic");
    }

    #[test]
    fn regrid_series_follows_the_same_mapping() {
        let secs: Vec<f64> = (0..32).filter(|k| *k != 8).map(|k| k as f64 * 0.01).collect();
        let t = ScanTiming::from_ticks(&ticks_at(&secs)).unwrap();
        let src = t.source_columns();
        let per_frame: Vec<f64> = (0..31).map(|k| k as f64).collect();
        let out = regrid_series(&per_frame, &src);
        assert_eq!(out.len(), 32);
        assert!((out[30] - 29.0).abs() < 1e-6, "out[30] = {}", out[30]);
        assert!((out[5] - 5.0).abs() < 1e-6);
    }
}


