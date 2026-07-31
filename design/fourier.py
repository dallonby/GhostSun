#!/usr/bin/env python3
"""Fourier (wave-optics) pass for the all-reflective SHG, config A.

1D treatment in the dispersion plane (the slit-width axis), which is where
Fraunhofer diffraction from a few-um slit dominates the beam geometry.

Model
-----
Angular intensity leaving the slit = [top-hat of the geometric f/6.9 cone]
convolved with [single-slit sinc^2 of width w at wavelength lambda]
(incoherent extended solar source: angular spectra add in intensity).
After the collimator each angle theta maps to pupil height y = f1*theta and
propagates parallel, so aperture clipping is a simple window in y:

  collimator: |y| <= 0.9 * D_col / 2          (90% clear aperture)
  grating:    |y| <= (W_g / 2) * cos(alpha)   (projected groove width)

alpha/beta are taken from the exact raytraced geometry (raytrace.Design).

Outputs per (line, slit width, grating width):
  * enclosed energy at collimator and after grating (throughput)
  * delivered line-spread: geometric slit image (+ anamorphism) convolved
    with the diffraction PSF of the *clipped* pupil -> effective bandpass
    broadening
  * where the clipped energy lands (baffle sites)
"""

import math
import numpy as np
from raytrace import Design, CONFIGS, dot

from raytrace import CHOSEN
CFG = CONFIGS[CHOSEN["config"]]
F1, F2 = CFG["rfl1"], CFG["rfl2"]
FNUM = 6.9
D_COL_CA = 0.9 * CHOSEN.get("colD", 50.8)  # collimator CA (mm)
D_CAM_CA = 0.9 * CHOSEN.get("camD", 76.2)  # camera mirror CA (mm)
PIX = 3.76e-3               # mm

LINES = [
    ("CaK 393", 393.37, 2400.0),
    ("Ha 656", 656.28, 2400.0),
    ("He 1083", 1083.0, 1200.0),
]
SLITS_UM = [5.0, 7.0, 10.0]
GRATING_W = [50.0, 25.0]    # mm (candidate purchase vs existing 25 mm)


def geometry(lam_nm, lpmm):
    d = Design(lines_per_mm=lpmm, order=1, dev=CHOSEN["dev"], s2=CHOSEN["s2"], Lg=CHOSEN["Lg"], Lc=CHOSEN["Lc"], **CFG)
    d.build(lam_nm)
    n = d.gr.n
    ca = abs(dot(d.c1, n))
    cb = abs(dot(d.c2, n))
    return math.degrees(math.acos(ca)), math.degrees(math.acos(cb)), ca, cb


def angular_intensity(lam_mm, w_mm, ngrid=60001, span=0.7):
    """I(theta): tophat(geometric cone) conv sinc^2(slit)."""
    th = np.linspace(-span, span, ngrid)
    dth = th[1] - th[0]
    geo = (np.abs(th) <= 1.0 / (2 * FNUM)).astype(float)
    x = np.pi * w_mm * th / lam_mm
    s = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    s[nz] = (np.sin(x[nz]) / x[nz]) ** 2
    s /= s.sum() * dth
    I = np.convolve(geo, s, mode="same") * dth
    I /= I.sum() * dth
    return th, I, dth


def enclosed(th, I, dth, half_mm):
    m = np.abs(th * F1) <= half_mm
    return float(I[m].sum() * dth)


def run():
    print(f"config A: f1={F1} f2={F2} f/{FNUM}; collimator CA "
          f"{D_COL_CA:.1f} mm; camera CA {D_CAM_CA:.1f} mm\n")
    rows = []
    for (label, lam_nm, lpmm) in LINES:
        a_deg, b_deg, ca, cb = geometry(lam_nm, lpmm)
        lam = lam_nm * 1e-6
        anam = ca / cb  # dispersion-plane magnification factor cos(a)/cos(b)
        print(f"{label}: {lpmm:.0f} l/mm  alpha={a_deg:.1f} beta={b_deg:.1f} "
              f"deg  anamorphic cos(a)/cos(b)={anam:.3f}")
        for w_um in SLITS_UM:
            th, I, dth = angular_intensity(lam, w_um * 1e-3)
            e_col = enclosed(th, I, dth, D_COL_CA / 2)
            for Wg in GRATING_W:
                a_gr = (Wg / 2) * ca
                e_gr = enclosed(th, I, dth, min(D_COL_CA / 2, a_gr))
                # delivered PSF from clipped pupil (dispersion plane):
                d_eff = 2 * min(D_COL_CA / 2, a_gr)      # at grating, mm
                d_cam = d_eff * cb / ca                  # entering camera
                psf_fwhm = 1.03 * lam * F2 / d_cam       # mm at sensor
                w_geo = w_um * 1e-3 * (F2 / F1) * anam   # slit image, mm
                w_eff = math.sqrt(w_geo**2 + psf_fwhm**2)
                rows.append((label, w_um, Wg, e_col, e_gr,
                             w_geo * 1e3, psf_fwhm * 1e3, w_eff * 1e3))
                print(f"  slit {w_um:4.0f} um, grating {Wg:2.0f} mm: "
                      f"through collimator {e_col*100:5.1f}%%, delivered "
                      f"{e_gr*100:5.1f}%%; slit img {w_geo*1e3:5.1f} um "
                      f"+ psf {psf_fwhm*1e3:5.1f} um -> {w_eff*1e3:5.1f} um "
                      f"({(w_eff/w_geo-1)*100:4.1f}%% wider)")
        # baffle map for the 7 um slit, 50 mm grating case
        th7, I7, dth7 = angular_intensity(lam, 7e-3)
        m_col = np.abs(th7 * F1) > D_COL_CA / 2
        loss_col = float(I7[m_col].sum() * dth7)
        a_gr50 = 25.0 * ca
        m_gr = (np.abs(th7 * F1) <= D_COL_CA / 2) & (np.abs(th7 * F1) > a_gr50)
        loss_gr = float(I7[m_gr].sum() * dth7)
        print(f"  baffle sites (7um/50mm): {loss_col*100:.1f}%% lands around "
              f"OAP1 aperture (dispersion plane, |y| {D_COL_CA/2:.0f}-"
              f"{F1*0.5:.0f} mm); {loss_gr*100:.1f}%% overshoots grating "
              f"edges\n")
    return rows


if __name__ == "__main__":
    run()
