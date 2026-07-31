#!/usr/bin/env python3
"""PERFECT_HA step 2 — sweep OAP-pair + plane-grating geometries at Ha with
the validated exact raytracer. Finds the aberration-optimal fold geometry
(th1, th2, dev, s2), then sizes arms for clearance and checks the camera
collision rule, delivered LSF, sampling and disk fit.

Stage 1: fold-geometry sweep at fixed RFL (aberrations scale weakly with
         RFL over the range of interest).
Stage 2: focal-length / arm refinement for the best folds + catalog alts.
"""
import math
import itertools
from perfect_ha_core import (build, blur_map, alpha_beta, delivered_lsf,
                             camera_exit, oap1_return_clearance,
                             slit_corridor_clearance, LAM_HA, PIX_571)

FIELDS = [0.0, 2.1, 2.6]
DLAMS = [0.0, 0.75]
PUPIL = 450.0          # telescope pupil distance behind slit (no field lens)


def eval_geom(rfl1, th1, rfl2, th2, dev, s2, Lg, Lc, fnum=6.9, w_um=7.0,
              pupil=PUPIL):
    try:
        d = build(rfl1, th1, rfl2, th2, dev, s2, Lg, Lc)
    except (RuntimeError, AssertionError):
        return None
    bm = blur_map(d, FIELDS, DLAMS, fnum=fnum, pupil_dist=pupil)
    if any(v is None for v in bm.values()):
        return None
    worst = max(max(v[0], v[1]) for v in bm.values())
    a, b, ca, cb = alpha_beta(d)
    lsf = delivered_lsf(rfl1, rfl2, fnum, w_um, 50.0, ca, cb, PIX_571)
    axang, margin, F2, df = camera_exit(d)
    clr1 = oap1_return_clearance(d)
    slitclr = slit_corridor_clearance(d)
    return dict(d=d, bm=bm, worst=worst, alpha=a, beta=b, lsf=lsf,
                axang=axang, cam_margin=margin, F2=F2, df=df,
                clr1=clr1, slitclr=slitclr,
                mag=rfl2 / rfl1, geo=(rfl1, th1, rfl2, th2, dev, s2, Lg, Lc))


def show(r, tag=""):
    g = r["geo"]
    F2 = r["F2"]
    lsf = r["lsf"]
    print(f"{tag}RFL {g[0]:.1f}/{g[2]:.1f} th {g[1]:.0f}/{g[3]:.0f} "
          f"dev {g[4]:+.0f} s2 {g[5]:+.0f} Lg {g[6]:.0f} Lc {g[7]:.0f} | "
          f"mag {r['mag']:.2f} a/b {r['alpha']:.1f}/{r['beta']:.1f} "
          f"anam {lsf['anam']:.2f}")
    rows = " ".join(f"{k[0]:.1f}/{k[1]:+.2f}:"
                    f"({v[0]:.1f},{v[1]:.1f})"
                    for k, v in sorted(r["bm"].items()))
    print(f"    blur um (field/dlam): {rows}")
    print(f"    LSF: slit img {lsf['w_img_um']:.1f} um, capture "
          f"{lsf['cap']*100:.0f}%, FWHM {lsf['fwhm_pm']:.1f} pm "
          f"(R {lsf['R']:,.0f}), {lsf['dl_px_pm']:.1f} pm/px, wing "
          f"{lsf['wing_frac']*100:.1f}%")
    print(f"    exits: df angle to scope axis {r['axang']:.1f} deg, "
          f"cam margin {r['cam_margin']:.0f} mm, F2 = "
          f"({F2[0]:.0f},{F2[1]:.0f},{F2[2]:.0f}); OAP1-return clr "
          f"{r['clr1']:.0f} mm; slit-corridor {r['slitclr']:.0f} mm")


if __name__ == "__main__":
    print("=" * 78)
    print("STAGE 1: fold-geometry sweep (RFL 80/130 fixed, Lg 150, Lc 170)")
    print("ranked by worst RMS blur over field 0-2.6 mm, dlam 0/+0.75 nm")
    print("=" * 78)
    results = []
    for th1, th2 in itertools.product((15.0, 20.0, 30.0),
                                      (15.0, 20.0, 30.0, 45.0)):
        for dev in (8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0,
                    -8.0, -10.0, -12.0, -14.0, -16.0, -20.0, -25.0):
            for s2 in (1.0, -1.0):
                r = eval_geom(80.0, th1, 130.0, th2, dev, s2, 150.0, 170.0)
                if r:
                    results.append(r)
    results.sort(key=lambda r: r["worst"])
    for r in results[:12]:
        show(r)
    print(f"\n(total viable: {len(results)})")

    print()
    print("=" * 78)
    print("STAGE 1b: same sweep, catalog angles th1=30 th2=45 (Edmund class)")
    print("=" * 78)
    cat = [r for r in results if r["geo"][1] == 30.0 and r["geo"][3] == 45.0]
    for r in cat[:4]:
        show(r)
