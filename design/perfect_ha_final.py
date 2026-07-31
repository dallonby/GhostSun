#!/usr/bin/env python3
"""PERFECT_HA step 4 — final geometry selection (fine grid around the
stage-2 winner) and the complete prescription printout for PERFECT_HA.md.

Winner family: Edmund #35-607 (RFL 81.79, 30 deg) collimator +
Edmund #35-588 (RFL 178.53, 45 deg) camera, GH50-24V 2400 l/mm,
dev ~ +16 deg off-Littrow, s2 = -1, IMX571.
"""
import math
import numpy as np
from perfect_ha_core import (build, alpha_beta, delivered_lsf, camera_exit,
                             oap1_return_clearance, slit_corridor_clearance,
                             oap2_clearances, trace_with_chief,
                             grating_capture, LAM_HA, SIGMA_2400, PIX_571)
from raytrace import stats, add, sub, mul, dot

PUPIL = 450.0
DISK = 4.2
RFL1, TH1, RFL2, TH2 = 81.79, 30.0, 178.53, 45.0
COLD, CAMD = 50.8, 76.2


def field_blur(d, xf, dl, fnum=6.9, nring=8, nseg=16):
    pts = trace_with_chief(d, xf, LAM_HA + dl, fnum, nring=nring,
                           nseg=nseg, pupil_dist=PUPIL)
    if not pts:
        return None
    cx, cy, rx, ry = stats(pts)
    return rx * 1e3, ry * 1e3, cx, cy


def pm_per_um(cb):
    """Plate factor at the sensor: pm of wavelength per um of x.
    dlambda/dx = sigma*cos(beta)/f2 [mm/mm] -> pm/um is x1e6."""
    return SIGMA_2400 * cb / RFL2 * 1e6


def mode_table(d, a, b, ca, cb, fields=(0.0, 2.1)):
    """Delivered FWHM incl. aberration at field centre & disk edge."""
    dldx_pm_per_um = pm_per_um(cb)
    rows = []
    for tag, w, fn in (("7um/65mm f6.9", 7.0, 6.9),
                       ("5um/65mm f6.9", 5.0, 6.9),
                       ("7um/85mm f5.3", 7.0, 5.3),
                       ("5um/85mm f5.3", 5.0, 5.3)):
        lsf = delivered_lsf(RFL1, RFL2, fn, w, 50.0, ca, cb, PIX_571,
                            colCA_mm=0.94 * COLD)
        per = {}
        for xf in fields:
            fb = field_blur(d, xf, 0.0, fnum=fn)
            sy_um = fb[1]
            ab_pm = 2.355 * sy_um * dldx_pm_per_um
            tot = math.hypot(lsf["fwhm_pm"], ab_pm)
            per[xf] = (tot, LAM_HA * 1e3 / tot)
        rows.append((tag, lsf, per))
    return rows, dldx_pm_per_um


def main():
    print("FINE GRID around stage-2 winner (full-res rays)")
    best = None
    for dev in (15.0, 16.0, 17.0, 18.0):
        for Lg in (165.0, 180.0, 195.0):
            for Lc in (270.0, 280.0, 290.0):
                try:
                    d = build(RFL1, TH1, RFL2, TH2, dev, -1.0, Lg, Lc)
                except (RuntimeError, AssertionError):
                    continue
                clr1 = oap1_return_clearance(d)
                if clr1 < 47.0:
                    continue
                axang, margin, F2, df = camera_exit(d)
                if margin < 15.0:
                    continue
                a, b, ca, cb = alpha_beta(d)
                fb = field_blur(d, 2.1, 0.0)
                if fb is None:
                    continue
                # deck extents gate: optics only (camera tail exits the
                # front wall beside the snout, off-deck per BODY.md)
                pys = [p[1] for p in (d.S, d.C1, d.G, d.C2, d.F2)]
                pzs = [p[2] for p in (d.S, d.C1, d.G, d.C2, d.F2)]
                sy = max(pys) - min(pys) + 70
                sz = max(pzs) - min(pzs) + 70
                if not (max(sy, sz) <= 350 and min(sy, sz) <= 300):
                    continue
                # merit: delivered R at disk edge (7um mode), then compact
                lsf = delivered_lsf(RFL1, RFL2, 6.9, 7.0, 50.0, ca, cb,
                                    PIX_571, colCA_mm=0.94 * COLD)
                ab_pm = 2.355 * fb[1] * pm_per_um(cb)
                tot = math.hypot(lsf["fwhm_pm"], ab_pm)
                key = (round(tot, 1), Lg + Lc)
                if best is None or key < best[0]:
                    best = (key, dev, Lg, Lc, d)
    (tot, _), dev, Lg, Lc, d = best
    a, b, ca, cb = alpha_beta(d)
    print(f"chosen: dev=+{dev:.0f} s2=-1 Lg={Lg:.0f} Lc={Lc:.0f}  "
          f"(disk-edge delivered {tot:.1f} pm)")

    print("\n=== PRESCRIPTION (world coords, slit=origin, feed +z, mm) ===")
    print(f"alpha={a:.2f} beta={b:.2f} deg   anamorphism cos(a)/cos(b)="
          f"{ca/cb:.3f}   grating tune gamma={d.gamma:.3f} deg")
    for name, p in (("slit S", d.S), ("OAP1 centre C1", d.C1),
                    ("grating pivot G", d.G), ("OAP2 centre C2", d.C2),
                    ("sensor F2", d.F2)):
        print(f"  {name:16s} ({p[0]:7.2f},{p[1]:7.2f},{p[2]:7.2f})")
    for name, v in (("slit->OAP1 c0", d.c0), ("OAP1->grating c1", d.c1),
                    ("grating->OAP2 c2", d.c2), ("OAP2->sensor df", d.df)):
        print(f"  {name:16s} ({v[0]:7.4f},{v[1]:7.4f},{v[2]:7.4f})")
    axang, margin, F2, df = camera_exit(d)
    print(f"  camera exit: {axang:.1f} deg off telescope axis, collision "
          f"margin {margin:.0f} mm; sensor lateral offset "
          f"{math.hypot(F2[0],F2[1]):.0f} mm")
    print(f"  clearances: OAP1-return {oap1_return_clearance(d):.0f} mm; "
          f"slit-corridor {slit_corridor_clearance(d):.0f} mm; OAP2 "
          f"{oap2_clearances(d)[0]:.0f} mm")
    # extents incl camera tail
    tail = add(F2, mul(df, 108.0))
    ys = [p[1] for p in (d.S, d.C1, d.G, d.C2, d.F2, tail)]
    zs = [p[2] for p in (d.S, d.C1, d.G, d.C2, d.F2, tail)]
    print(f"  fold-plane extents: y [{min(ys)-40:.0f},{max(ys)+40:.0f}] "
          f"z [{min(zs)-40:.0f},{max(zs)+40:.0f}]  (deck 350x300)")

    print("\n=== BLUR MAP (RMS um, spatial/dispersion), f/6.9, pupil 450 ===")
    print("field:      " + "".join(f"dl={dl:+5.2f}nm      "
                                   for dl in (0.0, 0.75, 1.5)))
    for xf in (0.0, 1.05, 2.1, 2.6, 3.0):
        row = f"  {xf:4.2f} mm  "
        for dl in (0.0, 0.75, 1.5):
            fb = field_blur(d, xf, dl)
            row += f"({fb[0]:5.1f},{fb[1]:5.1f})   "
        print(row + ("<- disk edge" if xf == 2.1 else
                     ("<- slit end (10mm slit would be 5.0)" if xf == 3.0
                      else "")))

    print("\n=== SMILE / TILT (line-centre position vs field) ===")
    cys = {}
    for xf in (-3.0, -2.1, 0.0, 2.1, 3.0):
        fb = field_blur(d, xf, 0.0)
        cys[xf] = fb[3]
    k = pm_per_um(cb)
    smile = (cys[2.1] + cys[-2.1]) / 2 - cys[0.0]
    tilt = (cys[2.1] - cys[-2.1]) / 2
    print(f"  smile over +-2.1 mm: {smile*1e3:.0f} um = "
          f"{smile*1e3*k:.0f} pm = {abs(smile)/PIX_571:.0f} px "
          f"(smooth quadratic; calibrated per column in recon, "
          f"Sol'Ex/INTI-style)")
    print(f"  tilt (odd term): {tilt*1e3:.1f} um across disk")

    print("\n=== MODE TABLE (delivered, incl. aberration) ===")
    rows, dldx = mode_table(d, a, b, ca, cb)
    for tag, lsf, per in rows:
        print(f"  {tag}: capture {lsf['cap']*100:.0f}%  slit-img "
              f"{lsf['w_img_um']:.1f} um  px {lsf['dl_px_pm']:.1f} pm  "
              f"wing {lsf['wing_frac']*100:.1f}%")
        print(f"      centre: FWHM {per[0.0][0]:5.1f} pm  R "
              f"{per[0.0][1]:7,.0f}   disk edge: {per[2.1][0]:5.1f} pm  "
              f"R {per[2.1][1]:7,.0f}")

    print("\n=== GRATING / OAP2 APERTURE USE ===")
    for w, fn in ((7.0, 6.9), (5.0, 6.9), (5.0, 5.3)):
        cap0, dclip = grating_capture(RFL1, fn, w, 50.0, ca)
        # field walk on grating: chief for field 2.1 with pupil at 450
        walk = Lg * 2.1 / RFL1 * (1 - RFL1 / PUPIL)
        capE, _ = grating_capture(RFL1, fn, w, 50.0, ca, walk_mm=2 * walk)
        print(f"  slit {w} um f/{fn}: centre capture {cap0*100:.1f}%, "
              f"disk-edge (walk {walk:.1f} mm) {capE*100:.1f}%")
    Dcam = dclip * cb / ca
    walk_cam = Lc * 2.1 / RFL1
    win_walk = 5e-6 / (SIGMA_2400 * cb) * Lc  # +-5 nm window walk, mm
    print(f"  camera-side beam {Dcam:.1f} mm + field walk "
          f"{2*walk_cam:.1f} + window(+-5nm) {2*win_walk:.1f} mm on "
          f"Ø{CAMD} OAP2 (CA ~{0.94*CAMD:.0f})")

    print("\n=== THROUGHPUT / SNR (from perfect_ha_req assumptions) ===")
    for coat, r in (("prot. Al pair", 0.88**2), ("prot. Ag pair", 0.975**2)):
        for geff, gl in ((0.60, "GH50-24V ~60%"),):
            net = 0.90 * r * geff
            print(f"  {coat}: prefilter 0.90 x mirrors {r:.2f} x "
                  f"grating {gl} -> slit-to-sensor {net*100:.0f}%")
    print("  continuum e-/px/ms at 65mm/7um ~ 20-36k (gain 0) -> expose "
          "0.7-1.2 ms;")
    print("  core SNR/resel/frame ~ 150-200; scan 600-1200 cols; IMX571 "
          "ROI 6248x256 ~15-25 fps -> 30-80 s/scan")


if __name__ == "__main__":
    main()
