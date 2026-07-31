#!/usr/bin/env python3
"""PERFECT_HA step 3 — stage-2 refinement (fast version).
Order: build -> cheap clearance/collision filters -> coarse blur ->
cached delivered-LSF. Full-resolution numbers are re-derived for the
finalists only (perfect_ha_final.py)."""
import math
from perfect_ha_core import (build, alpha_beta, delivered_lsf, camera_exit,
                             oap1_return_clearance, slit_corridor_clearance,
                             oap2_clearances, trace_with_chief, LAM_HA,
                             PIX_571)
from raytrace import stats

FIELDS = [0.0, 2.1, 2.6]
DLAMS = [0.0, 0.75]
PUPIL = 450.0
DISK = 4.2
_LSF_CACHE = {}


def lsf_cached(f1, f2, fnum, w, ca, cb, colCA):
    key = (round(f1, 1), round(f2, 1), fnum, w, round(ca, 4), round(cb, 4),
           round(colCA, 1))
    if key not in _LSF_CACHE:
        _LSF_CACHE[key] = delivered_lsf(f1, f2, fnum, w, 50.0, ca, cb,
                                        PIX_571, colCA_mm=colCA)
    return _LSF_CACHE[key]


def coarse_blur(d, fnum=6.9):
    out = {}
    for xf in FIELDS:
        for dl in DLAMS:
            pts = trace_with_chief(d, xf, LAM_HA + dl, fnum, nring=4,
                                   nseg=8, pupil_dist=PUPIL)
            if not pts:
                return None
            cx, cy, rx, ry = stats(pts)
            out[(xf, dl)] = (rx * 1e3, ry * 1e3)
    return out


def full_eval(rfl1, th1, rfl2, th2, dev, s2, Lg, Lc, colD=40.0, camD=50.0):
    try:
        d = build(rfl1, th1, rfl2, th2, dev, s2, Lg, Lc)
    except (RuntimeError, AssertionError):
        return None
    # cheap geometry gates first
    beam = rfl1 * (1 / 6.9 + 2 * LAM_HA * 1e-6 / 7e-3)
    clr1 = oap1_return_clearance(d)
    need1 = colD / 2 + beam / 2 + 8.0
    if clr1 < need1:
        return None
    slitclr = slit_corridor_clearance(d)
    if slitclr < 18.0:
        return None
    c2a, c2b, f2g = oap2_clearances(d)
    need2 = camD / 2 + beam / 2 + 8.0
    if c2a < need2:
        return None
    axang, margin, F2, df = camera_exit(d)
    if margin < 10.0:          # +inf when camera never enters z<0
        return None
    if abs(F2[1]) > 175 or abs(F2[2]) > 175:
        return None
    bm = coarse_blur(d)
    if bm is None:
        return None
    a, b, ca, cb = alpha_beta(d)
    modes = {}
    for tag, w, fn in (("7/6.9", 7.0, 6.9), ("5/6.9", 5.0, 6.9),
                       ("7/5.3", 7.0, 5.3), ("5/5.3", 5.0, 5.3)):
        modes[tag] = lsf_cached(rfl1, rfl2, fn, w, ca, cb, 0.94 * colD * ca)
    worst_disk = max(max(v) for k, v in bm.items() if k[0] <= 2.1)
    worst_all = max(max(v) for v in bm.values())
    return dict(bm=bm, a=a, b=b, modes=modes, clr1=clr1, need1=need1,
                slitclr=slitclr, c2a=c2a, need2=need2, axang=axang,
                margin=margin, F2=F2, mag=rfl2 / rfl1,
                worst_disk=worst_disk, worst_all=worst_all,
                geo=(rfl1, th1, rfl2, th2, dev, s2, Lg, Lc))


def report(r, tag=""):
    g = r["geo"]
    print(f"{tag} RFL {g[0]:.2f}/{g[2]:.2f} th {g[1]:.0f}/{g[3]:.0f} "
          f"dev {g[4]:+.0f} Lg {g[6]:.0f} Lc {g[7]:.0f} "
          f"mag {r['mag']:.2f} a/b {r['a']:.1f}/{r['b']:.1f}")
    print(f"   blur: disk worst {r['worst_disk']:.1f} um, slit-end "
          f"{r['worst_all']:.1f} um (spatial slit img {7*r['mag']:.1f} um)")
    for k in ("7/6.9", "5/6.9", "7/5.3", "5/5.3"):
        v = r["modes"][k]
        print(f"   {k}: cap {v['cap']*100:4.0f}%  FWHM {v['fwhm_pm']:5.1f}pm"
              f"  R {v['R']:6,.0f}  {v['fwhm_pm']/v['dl_px_pm']:.1f}px/FWHM"
              f"  wing {v['wing_frac']*100:.1f}%")
    F2 = r["F2"]
    print(f"   disk {DISK*r['mag']:.1f}mm/15.7; axang {r['axang']:.0f} "
          f"margin {r['margin']:.0f} F2 ({F2[0]:.0f},{F2[1]:.0f},"
          f"{F2[2]:.0f}); clr1 {r['clr1']:.0f}/{r['need1']:.0f} "
          f"slit {r['slitclr']:.0f} oap2 {r['c2a']:.0f}/{r['need2']:.0f}")


def sweep(name, combos):
    print("=" * 96)
    print(name)
    print("=" * 96)
    best = []
    for args in combos:
        r = full_eval(*args[0], **args[1])
        if r:
            best.append(r)
    best.sort(key=lambda r: (r["worst_disk"]
                             - 0.6 * r["modes"]["7/6.9"]["R"] / 1e4))
    seen = set()
    shown = 0
    for r in best:
        key = r["geo"][:6]
        if key in seen:
            continue
        seen.add(key)
        report(r, name.split(":")[0])
        shown += 1
        if shown >= 5:
            break
    print(f"   ({len(best)} passing geometries)\n")
    return best


if __name__ == "__main__":
    f1c = []
    for th1, th2 in ((15.0, 20.0), (20.0, 20.0), (20.0, 30.0)):
        for f1 in (70.0, 80.0, 90.0):
            for f2 in (130.0, 150.0, 165.0):
                for dev in (12.0, 14.0, 16.0, 20.0):
                    for Lg in (150.0, 180.0, 210.0):
                        for Lc in (200.0, 240.0, 280.0):
                            f1c.append(((f1, th1, f2, th2, dev, -1.0, Lg,
                                         Lc), {}))
    sweep("F1: custom small-angle pair", f1c)

    f2c = []
    for f2, th2 in ((163.18, 30.0), (178.53, 45.0), (272.23, 30.0)):
        for dev in (14.0, 16.0, 18.0, 20.0, 22.0, 25.0):
            for Lg in (120.0, 150.0, 180.0):
                for Lc in (200.0, 240.0, 280.0):
                    f2c.append(((81.79, 30.0, f2, th2, dev, -1.0, Lg, Lc),
                                dict(colD=50.8, camD=76.2)))
    sweep("F2: Edmund 30deg collimator familes", f2c)

    f3c = []
    for dev in (12.0, 14.0, 16.0, 20.0):
        for Lg in (117.0, 150.0, 180.0):
            for Lc in (230.0, 260.0):
                f3c.append(((81.79, 30.0, 178.53, 45.0, dev, -1.0, Lg, Lc),
                            dict(colD=50.8, camD=76.2)))
    sweep("F3: production 30/45 B-config parts at Ha-optimal dev", f3c)
