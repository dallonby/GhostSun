//! Solar spectral-line identification (F12).
//!
//! Two-layer wavelength calibration for a sunlight source:
//!   1. **Geometric Å/px** from the grating + camera optics ([`geometric_dispersion`])
//!      — a good scale, but the absolute zero-point depends on the (unknown)
//!      grating rotation angle.
//!   2. **Absolute anchor from the spectrum itself** ([`calibrate`]) — match the
//!      detected line pattern against the embedded Fraunhofer catalog, seeded by
//!      the geometric scale and a target wavelength (Hα by default). Robust to
//!      the deepest line not being the target, and to the dispersion sign.
//!
//! Air wavelengths in Ångström.

/// One catalogued solar absorption line.
pub struct CatalogLine {
    pub wavelength: f64,
    pub element: &'static str,
    /// Qualitative strength 0..1 (for display / future weighting).
    pub strength: f32,
}

const fn line(wavelength: f64, element: &'static str, strength: f32) -> CatalogLine {
    CatalogLine { wavelength, element, strength }
}

/// Strong solar Fraunhofer lines plus the Hα-region tellurics. Deliberately
/// conservative — only lines whose air wavelengths are well established.
pub static SOLAR_LINES: &[CatalogLine] = &[
    line(3933.66, "Ca II K", 1.0),
    line(3968.47, "Ca II H", 1.0),
    line(4101.74, "Hδ", 0.8),
    line(4340.47, "Hγ", 0.8),
    line(4383.55, "Fe I", 0.5),
    line(4861.35, "Hβ", 0.9),
    line(5167.32, "Mg I", 0.7),
    line(5172.68, "Mg I b2", 0.8),
    line(5183.60, "Mg I b1", 0.8),
    line(5269.54, "Fe I E", 0.7),
    line(5328.04, "Fe I", 0.5),
    line(5889.95, "Na I D2", 1.0),
    line(5895.92, "Na I D1", 0.9),
    line(6122.22, "Ca I", 0.5),
    line(6162.17, "Ca I", 0.6),
    line(6173.33, "Fe I", 0.4),
    line(6302.49, "Fe I", 0.5),
    line(6494.98, "Fe I", 0.5),
    line(6546.24, "Fe I", 0.5),
    line(6561.10, "tell H₂O", 0.3),
    line(6562.79, "Hα", 1.0),
    line(6564.20, "tell H₂O", 0.3),
    line(6569.21, "Fe I", 0.4),
    line(6572.78, "Ca I", 0.4),
    line(6592.91, "Fe I", 0.5),
    line(6677.99, "Fe I", 0.5),
];

/// Reciprocal linear dispersion (Å/px) from grating + camera geometry, near
/// Littrow (Sol'Ex). `grating` in lines/mm, `focal_mm` the camera lens focal
/// length, `pixel_um` the sensor pixel pitch, `central_a` the working
/// wavelength. Returns `None` if the geometry is unphysical (|sin β| > 1).
pub fn geometric_dispersion(
    grating_l_per_mm: f64,
    order: u32,
    focal_mm: f64,
    pixel_um: f64,
    central_a: f64,
) -> Option<f64> {
    if grating_l_per_mm <= 0.0 || focal_mm <= 0.0 || order == 0 {
        return None;
    }
    let m = order as f64;
    let lambda_mm = central_a * 1e-7;
    let sin_beta = m * lambda_mm * grating_l_per_mm / 2.0; // Littrow: α ≈ β
    if sin_beta.abs() >= 1.0 {
        return None;
    }
    let cos_beta = (1.0 - sin_beta * sin_beta).sqrt();
    let pixel_mm = pixel_um * 1e-3;
    Some(pixel_mm * cos_beta / (m * grating_l_per_mm * focal_mm) * 1e7)
}

/// Linear pixel→wavelength solution `λ = a·x + b`.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    pub a: f64,
    pub b: f64,
    pub rms: f64,
    pub n_matched: usize,
}

impl Calibration {
    pub fn wavelength(&self, x: f64) -> f64 {
        self.a * x + self.b
    }
}

/// A detected line matched to the catalog.
#[derive(Clone, Copy, Debug)]
pub struct LabeledLine {
    pub x: f64,
    pub wavelength: f64,
    pub element: &'static str,
}

fn nearest_catalog(lambda: f64, tol: f64) -> Option<&'static CatalogLine> {
    SOLAR_LINES
        .iter()
        .filter(|c| (c.wavelength - lambda).abs() <= tol)
        .min_by(|a, b| {
            (a.wavelength - lambda)
                .abs()
                .partial_cmp(&(b.wavelength - lambda).abs())
                .unwrap()
        })
}

/// Least-squares fit of `λ = a·x + b` to matched pairs.
fn fit_ab(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = pairs.len() as f64;
    if pairs.len() < 2 {
        return None;
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for &(x, y) in pairs {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        return None;
    }
    let a = (n * sxy - sx * sy) / denom;
    let b = (sy - a * sx) / n;
    Some((a, b))
}

/// Solve the absolute wavelength calibration from detected line positions
/// (`centers`) and their relative depths, seeded by the geometric dispersion
/// `approx_a_per_px` and the target `central_wavelength` (assumed sunlight).
/// Tries both dispersion signs and the deepest few lines as anchors; refines by
/// iterated matching against the catalog. `None` if it can't lock ≥3 lines.
pub fn calibrate(
    centers: &[f64],
    depths: &[f64],
    approx_a_per_px: f64,
    central_wavelength: f64,
) -> Option<Calibration> {
    if centers.len() < 3 || approx_a_per_px.abs() < 1e-9 {
        return None;
    }
    // Anchor candidates: the deepest few lines (likely strong catalogued lines).
    let mut order: Vec<usize> = (0..centers.len()).collect();
    order.sort_by(|&i, &j| depths[j].partial_cmp(&depths[i]).unwrap());
    let anchors = &order[..order.len().min(3)];
    let a0 = approx_a_per_px.abs();

    let mut best: Option<Calibration> = None;
    for &anchor in anchors {
        for sign in [1.0f64, -1.0] {
            let mut a = sign * a0;
            let mut b = central_wavelength - a * centers[anchor];
            for it in 0..10 {
                let tol = (4.0 * a.abs()).max(0.5) * if it < 3 { 2.5 } else { 1.0 };
                let pairs: Vec<(f64, f64)> = centers
                    .iter()
                    .filter_map(|&x| nearest_catalog(a * x + b, tol).map(|c| (x, c.wavelength)))
                    .collect();
                if pairs.len() < 3 {
                    break;
                }
                match fit_ab(&pairs) {
                    Some((na, nb)) => {
                        a = na;
                        b = nb;
                    }
                    None => break,
                }
            }
            // Reject runaway solutions that wandered far from the geometry.
            if a.abs() < a0 * 0.4 || a.abs() > a0 * 2.5 {
                continue;
            }
            // Final tight-tolerance score.
            let tol = (2.0 * a.abs()).max(0.35);
            let matches: Vec<f64> = centers
                .iter()
                .filter_map(|&x| nearest_catalog(a * x + b, tol).map(|c| c.wavelength - (a * x + b)))
                .collect();
            if matches.len() >= 3 {
                let rms = (matches.iter().map(|d| d * d).sum::<f64>() / matches.len() as f64).sqrt();
                let cand = Calibration { a, b, rms, n_matched: matches.len() };
                best = Some(match best {
                    Some(bc)
                        if bc.n_matched > cand.n_matched
                            || (bc.n_matched == cand.n_matched && bc.rms <= cand.rms) =>
                    {
                        bc
                    }
                    _ => cand,
                });
            }
        }
    }
    best
}

/// Label each detected line with its nearest catalog match under `tol_a` (Å).
pub fn identify(centers: &[f64], cal: &Calibration, tol_a: f64) -> Vec<LabeledLine> {
    centers
        .iter()
        .filter_map(|&x| {
            let lambda = cal.wavelength(x);
            nearest_catalog(lambda, tol_a).map(|c| LabeledLine {
                x,
                wavelength: c.wavelength,
                element: c.element,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_matches_solex_halpha() {
        // 2400 l/mm, 125 mm lens, 2.0 µm pixel, order 1, at Hα ⇒ ~0.041 Å/px.
        let d = geometric_dispersion(2400.0, 1, 125.0, 2.0, 6562.79).unwrap();
        assert!((d - 0.041).abs() < 0.003, "got {d}");
    }

    #[test]
    fn calibrates_and_identifies_halpha() {
        // Ground-truth linear map a=0.05, Hα at x=300.
        let (a, b) = (0.05, 6562.79 - 0.05 * 300.0);
        let cat = [6546.24, 6562.79, 6569.21, 6572.78, 6592.91];
        let centers: Vec<f64> = cat.iter().map(|w| (w - b) / a).collect();
        // Hα (index 1) deepest.
        let depths = vec![0.4, 0.95, 0.35, 0.3, 0.4];
        let cal = calibrate(&centers, &depths, 0.045, 6562.79).expect("calibrated");
        assert!((cal.a.abs() - 0.05).abs() < 0.004, "a = {}", cal.a);
        let labels = identify(&centers, &cal, 0.3);
        assert!(
            labels.iter().any(|l| l.element == "Hα"),
            "labels: {:?}",
            labels.iter().map(|l| l.element).collect::<Vec<_>>()
        );
    }
}
