//! Experimental line-local structure metric, not a filament or velocity classifier.
//! Work on raw intensities, normalize each slit position by its continuum,
//! remove slowly varying line shape, and require coherent spectral residuals.
use ghostsun_core::mathutil::gaussian_smooth;
use std::{collections::VecDeque, time::{Duration, Instant}};

#[derive(Clone, Debug)]
pub struct Measurement {
    /// Noise-subtracted RMS along-slit gradient, percent continuum per camera pixel.
    pub clarity: f64,
    pub center: f64,
    /// Rectangles in normalized sensor coordinates (x0, y0, x1, y1).
    pub regions: Vec<[f32; 4]>,
    pub region_clarity: Vec<f64>,
}

#[derive(Default)]
pub struct MotionTracker {
    tracks: Vec<Track>,
}

struct Track {
    positions: VecDeque<(Instant, f64)>,
    width: f64,
}

impl MotionTracker {
    /// Evidence of motion in detector coordinates, not identification of solar material.
    /// Match each current streak once and require repeated observations over time.
    pub fn update(&mut self, at: Instant, regions: &[[f32; 4]], horizontal: bool, spatial: usize) -> Vec<usize> {
        let axis = if horizontal { 1 } else { 0 };
        let mut old: Vec<Option<Track>> = std::mem::take(&mut self.tracks).into_iter().map(Some).collect();
        let mut moving = Vec::new();
        for (index, r) in regions.iter().enumerate() {
            let center = (r[axis] + r[axis + 2]) as f64 * 0.5;
            let width = (r[axis + 2] - r[axis]) as f64;
            let best = old.iter().enumerate().filter_map(|(i, t)| {
                let t = t.as_ref()?;
                let (time, previous) = t.positions.back()?;
                let distance = (center - previous).abs();
                (at.checked_duration_since(*time).is_some_and(|age| age < Duration::from_millis(750))
                    && distance <= width.max(t.width).max(4.0 / spatial.max(1) as f64).min(0.025)
                    && width >= t.width * 0.5 && width <= t.width * 2.0)
                    .then_some((i, distance))
            }).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i);
            let mut track = best.and_then(|i| old[i].take()).unwrap_or(Track {
                positions: VecDeque::new(), width,
            });
            while track.positions.front().is_some_and(|(t, _)| at.saturating_duration_since(*t) >= Duration::from_secs(5)) {
                track.positions.pop_front();
            }
            track.positions.push_back((at, center));
            track.width = width;
            if track.positions.len() >= 5 && at.saturating_duration_since(track.positions[0].0) >= Duration::from_millis(300) {
                let mut positions: Vec<_> = track.positions.iter().map(|(_, p)| *p).collect();
                positions.sort_by(f64::total_cmp);
                let trim = positions.len() / 10;
                let excursion = positions[positions.len() - 1 - trim] - positions[trim];
                if excursion > (3.0 / spatial.max(1) as f64).max(width * 0.25) {
                    moving.push(index);
                }
            }
            self.tracks.push(track);
        }
        moving
    }
}

impl Measurement {
    pub fn moving_clarity(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() { return 0.0; }
        (indices.iter().map(|&i| self.region_clarity[i].powi(2)).sum::<f64>() / indices.len() as f64).sqrt()
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() { return 0.0; }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

pub fn measure(
    data: &[u16], w: usize, h: usize, dispersion_horizontal: bool,
    center: f64, fwhm: f64, continuum: &[f64],
) -> Option<Measurement> {
    let (spectral, spatial) = if dispersion_horizontal { (w, h) } else { (h, w) };
    if data.len() != w.checked_mul(h)? || spectral < 16 || spatial < 64
        || continuum.len() != spatial || !center.is_finite() || !fwhm.is_finite()
        || fwhm < 1.0 { return None; }
    let half = (fwhm * 1.5).ceil().clamp(6.0, 32.0) as usize;
    if center < (half + 2) as f64 || center + (half + 2) as f64 >= spectral as f64 {
        return None;
    }
    let start = center.round() as usize - half;
    let bands = 2 * half + 1;
    // Bound work independently of sensor size; bin rather than point-sample.
    let step = spatial.div_ceil(768);
    let n = spatial / step;
    let peak = continuum.iter().copied().fold(0.0_f64, f64::max);
    if peak < 128.0 { return None; }
    let mut valid = vec![false; n];
    let mut profiles = vec![vec![0.0; n]; bands];
    for i in 0..n {
        let lo = i * step;
        let c = continuum[lo..lo + step].iter().sum::<f64>() / step as f64;
        if !c.is_finite() || c < 0.4 * peak { continue; }
        valid[i] = true;
        for (b, profile) in profiles.iter_mut().enumerate() {
            profile[i] = (lo..lo + step).map(|s| {
                let index = if dispersion_horizontal { s * w + start + b } else { (start + b) * w + s };
                data[index] as f64
            }).sum::<f64>() / (step as f64 * c);
        }
    }
    let lo = valid.iter().position(|&v| v)?;
    let hi = valid.iter().rposition(|&v| v)? + 1;
    if hi - lo < 48 { return None; }
    // Missing illumination within the selected span invalidates the measure.
    if valid[lo..hi].iter().any(|&v| !v) { return None; }
    let len = hi - lo;
    let mut residual = Vec::with_capacity(bands);
    let mut differences = Vec::new();
    for profile in profiles {
        let p = &profile[lo..hi];
        let baseline = gaussian_smooth(p, (len as f64 / 24.0).max(4.0));
        let r: Vec<f64> = p.iter().zip(baseline).map(|(p, b)| p - b).collect();
        differences.extend(r.windows(2).map(|v| (v[1] - v[0]).abs()));
        residual.push(r);
    }
    let noise = (median(differences) / 0.9539).max(0.0005);
    // Coherence across adjacent wavelength pixels rejects isolated hot pixels.
    let coherent: Vec<Vec<f64>> = residual.windows(3).map(|p| {
        (0..len).map(|i| (p[0][i] + p[1][i] + p[2][i]) / 3.0).collect()
    }).collect();
    let strength: Vec<f64> = (0..len).map(|i| {
        coherent.iter().map(|p| p[i].abs()).fold(0.0_f64, f64::max)
    }).collect();
    let threshold = (5.0 * noise).max(0.01);
    let mut regions = Vec::new();
    let mut region_clarity = Vec::new();
    let mut i = 6;
    let mut gradient_energy = 0.0;
    let mut gradient_count = 0;
    while i + 6 < len {
        if strength[i] <= threshold { i += 1; continue; }
        let first = i;
        while i + 6 < len && strength[i] > threshold { i += 1; }
        if i - first < 2 { continue; }
        // Include shoulders when measuring sharpness, including smooth features.
        let a = first.saturating_sub(2).max(1);
        let z = (i + 2).min(len - 1);
        let previous_energy = gradient_energy;
        let previous_count = gradient_count;
        for p in &coherent {
            for j in a..z {
                gradient_energy += (p[j + 1] - p[j]).powi(2);
                gradient_count += 1;
            }
        }
        let s0 = ((lo + first) * step) as f32 / spatial as f32;
        region_clarity.push(((gradient_energy - previous_energy) / (gradient_count - previous_count) as f64
            - 2.0 * noise * noise / 3.0).max(0.0).sqrt() * 100.0 / step as f64);
        let s1 = ((lo + i) * step) as f32 / spatial as f32;
        let b0 = start as f32 / spectral as f32;
        let b1 = (start + bands) as f32 / spectral as f32;
        regions.push(if dispersion_horizontal { [b0, s0, b1, s1] } else { [s0, b0, s1, b1] });
    }
    let clarity = if gradient_count == 0 { 0.0 } else {
        ((gradient_energy / gradient_count as f64 - 2.0 * noise * noise / 3.0).max(0.0)).sqrt()
            * 100.0 / step as f64
    };
    Some(Measurement { clarity, center, regions, region_clarity })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn motion_evidence_rejects_stationary_dust_and_tracks_displacement() {
        let mut tracker = MotionTracker::default();
        let now = Instant::now();
        let mut moving = Vec::new();
        for i in 0..20 {
            let shift = i as f32 * 0.0008;
            // A fixed dust feature and a moving feature, both with constant widths.
            moving = tracker.update(now + Duration::from_millis(i * 50),
                &[[0.4, 0.2, 0.6, 0.22], [0.4, 0.5 + shift, 0.6, 0.52 + shift]], true, 1000);
        }
        assert_eq!(moving, vec![1]);
        assert!(tracker.update(now + Duration::from_secs(7),
            &[[0.4, 0.2, 0.6, 0.22], [0.4, 0.5152, 0.6, 0.5352]], true, 1000).is_empty());
    }

    #[test]
    fn subpixel_boundary_jitter_is_not_motion_evidence() {
        let mut tracker = MotionTracker::default();
        let now = Instant::now();
        for i in 0..20 {
            let jitter = (i % 3) as f32 * 0.0004;
            assert!(tracker.update(now + Duration::from_millis(i * 50),
                &[[0.4, 0.2 + jitter, 0.6, 0.22 + jitter]], true, 1000).is_empty());
        }
    }

    fn scene(blur: f64, feature: bool, dust: bool) -> (Vec<u16>, Vec<f64>) {
        let mut data = Vec::new();
        let mut continuum = Vec::new();
        for y in 0..256 {
            let c = 30000.0 * if dust && (120..126).contains(&y) { 0.65 } else { 1.0 };
            continuum.push(c);
            for x in 0..64 {
                let base = 1.0 - 0.45 * (-0.5 * ((x as f64 - 32.0) / 4.0).powi(2)).exp();
                let streak = if feature {
                    0.18 * (2.0 / blur) * (-0.5 * ((y as f64 - 120.0) / blur).powi(2)).exp()
                        * (-0.5 * ((x as f64 - 35.0) / 2.0).powi(2)).exp()
                } else { 0.0 };
                // Deterministic low-level sensor noise.
                let noise = ((x * 137 + y * 317 + x * y * 19) % 101) as f64 - 50.0;
                data.push((c * (base - streak) + noise) as u16);
            }
        }
        (data, continuum)
    }
    #[test]
    fn coherent_streak_detected_and_blur_reduces_clarity() {
        let (sharp, c) = scene(2.0, true, false);
        let (blurred, _) = scene(5.0, true, false);
        let a = measure(&sharp, 64, 256, true, 32.0, 9.4, &c).unwrap();
        let b = measure(&blurred, 64, 256, true, 32.0, 9.4, &c).unwrap();
        assert!(!a.regions.is_empty());
        assert!(a.clarity > b.clarity * 1.5, "{} vs {}", a.clarity, b.clarity);
        assert!(a.regions.iter().any(|r| r[1] < 120.0 / 256.0 && r[3] > 120.0 / 256.0));
    }
    #[test]
    fn plain_line_noise_and_continuum_dust_do_not_count_as_streaks() {
        for dust in [false, true] {
            let (data, c) = scene(2.0, false, dust);
            let m = measure(&data, 64, 256, true, 32.0, 9.4, &c).unwrap();
            assert!(m.regions.is_empty());
            assert_eq!(m.clarity, 0.0);
        }
    }
    #[test]
    fn transpose_preserves_score_and_overlay_coordinates() {
        let (data, c) = scene(2.0, true, false);
        let transposed: Vec<_> = (0..64).flat_map(|x| (0..256).map({ let data = &data; move |y| data[y * 64 + x] })).collect();
        let a = measure(&data, 64, 256, true, 32.0, 9.4, &c).unwrap();
        let b = measure(&transposed, 256, 64, false, 32.0, 9.4, &c).unwrap();
        assert_eq!(a.clarity, b.clarity);
        for (a, b) in a.regions.iter().zip(&b.regions) { assert_eq!(*a, [b[1], b[0], b[3], b[2]]); }
    }
    #[test]
    fn missing_illumination_and_cropped_line_are_unavailable() {
        let (data, c) = scene(2.0, true, false);
        assert!(measure(&data, 64, 256, true, 2.0, 9.4, &c).is_none());
        assert!(measure(&data, 64, 256, true, 32.0, 9.4, &vec![0.0; 256]).is_none());
    }
}
