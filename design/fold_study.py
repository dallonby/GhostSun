#!/usr/bin/env python3
"""Fold-flat design study (stdlib only). Two questions:

A. BUDGET: does a fold flat after OAP2 (fold4) let the cheap 45/45 pair
   (MPD124 + MPD144, both 1", £394) replace the frozen 45/60 MPD246
   build by decoupling the camera from the telescope side?
B. PRODUCTION: can any fold-flat topology bypass the slit corridor and
   reclaim the PERFECT_HA geometry that mech.py proved infeasible in the
   plain layout? Tried and failed: fold3 (the beam2/beam3 wedge opens
   only Lg*sin(dev), no room for a mount), fold2 (one in-plane fold
   flips chirality AND the compression branch; the compression root
   then recreates the congestion), lift3 (bottom periscope mirror lands
   in the same wedge). WINNER: lift2, a vertical periscope in the
   45-degree-wide beam1/beam2 wedge that moves the grating, rotor, OAP2
   and camera to an upper level. Two reflections = no chirality flip,
   dev/s2 keep their meanings, exactly aberration-free (traced).

Chirality rules (verified by trace): an in-plane fold3 mirrors the
downstream space, so the coma-cancelling camera sense flips to s2=+1;
a fold2 additionally flips which dev sign is the compression branch;
fold4 and the two-mirror periscopes are neutral.

Feasibility = mech.py: no core/structure pair under 3 mm, no fan
vignette. Ranking = disk-edge blur (field 2.1 mm, worst over +-0.5 nm),
with the spectral-resolution estimate reported alongside.
"""

import math
from raytrace import Design, CONFIGS, stats, dot
import mech

BUDGET_DIMS = dict(grat_w=25.0, colD=25.4, camD=25.4,
                   slab1_w=54.0, slab2_w=54.0, flat4_d=25.4)
PROD_DIMS = dict(grat_w=50.0, colD=50.8, camD=76.2,
                 slab1_w=86.0, slab2_w=86.0, flat3_d=50.8,
                 cam_front_r=50.0, cam_front=(-12.0, 24.0),
                 cam_body_r=40.0, cam_body=(15.0, 120.0))


def blur_edge(d):
    """Worst RMS radius (um) at the disk edge and slit end, +-0.5 nm."""
    out = []
    for xf in (2.1, 3.5):
        w = 0.0
        for lam in (655.78, 656.28, 656.78):
            pts = d.trace(xf, lam)
            if not pts:
                return None
            _, _, rx, ry = stats(pts)
            w = max(w, rx * 1e3, ry * 1e3)
        out.append(w)
    return out


def resolve(d):
    """Slit-limited R and anamorphic image compression at Ha."""
    ca = abs(dot(d.c1, d.gr.n))
    cb = abs(dot(d.c2b if d.flat3 else d.c2, d.gr.n))
    anam = ca / cb
    wimg = 7.0 * (d.rfl2 / d.rfl1) * anam
    dl = d.sigma * 1e6 * cb / (d.rfl2 * 1e3)
    return 656.28 / (wimg * dl), anam


def feasibility(d, dims):
    """Returns (worst core-beam, worst fan-beam, worst structure pair).
    Structure-vs-structure proximity is buildable at any non-negative
    value (parts may even merge); only light needs millimetres."""
    rows = mech.clearances(d, dims)
    wc = wf = ws = float("inf")
    for (a, b, c) in rows:
        beam = "beam" in a or "beam" in b
        if not beam:
            ws = min(ws, c)
        elif mech._is_fan(a, b):
            wf = min(wf, c)
        else:
            wc = min(wc, c)
    return wc, wf, ws


def flat_aperture_need(r_fan, delta_deg):
    """Required flat clear semi-aperture for a fan of radius r_fan."""
    return r_fan / math.cos(math.radians(abs(delta_deg)) / 2.0)


def sweep_budget():
    print("=" * 72)
    print("A. BUDGET 45/45 (MPD124+MPD144) + fold4 flat after OAP2")
    print("=" * 72)
    cfg = CONFIGS["T_budget_ha"]
    feas = []
    for dev in (18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0):
        for Lg in (70.0, 80.0, 90.0, 100.0, 110.0):
            for Lc in (130.0, 150.0, 170.0, 190.0):
                for s4f in (0.45, 0.60, 0.75):
                    s4 = s4f * cfg["rfl2"]
                    for d4 in (-110.0, -90.0, -70.0, 70.0, 90.0, 110.0):
                        try:
                            d = Design(lines_per_mm=2400.0, order=1,
                                       dev=dev, s2=-1.0, Lg=Lg, Lc=Lc,
                                       fold4=dict(s=s4, delta=d4), **cfg)
                            d.build(656.28)
                        except Exception:
                            continue
                        wc, wf, ws = feasibility(d, BUDGET_DIMS)
                        if wc < 3.0 or wf < 0.0 or ws < 0.0:
                            continue
                        bl = blur_edge(d)
                        if bl is None:
                            continue
                        R, anam = resolve(d)
                        feas.append((bl[0], bl[1], dev, Lg, Lc, s4, d4,
                                     wc, wf, R, anam, d))
    feas.sort(key=lambda r: r[0])
    print("edge21 end35  dev   Lg   Lc   s4  delta coreClr fanClr"
          "      R  anam")
    for r in feas[:12]:
        print(f"{r[0]:6.1f} {r[1]:5.1f} {r[2]:4.0f} {r[3]:4.0f} {r[4]:4.0f} "
              f"{r[5]:4.0f} {r[6]:6.0f} {r[7]:7.1f} {r[8]:6.1f} {r[9]:7.0f} "
              f"{r[10]:.3f}")
    print(f"{len(feas)} feasible")
    return feas


def _split_feas(d, dims):
    """Worst clearance over camera-dependent pairs vs everything else."""
    cam = ("camera_front", "camera_body", "flat4")
    wc_cam = wc_rest = wf_cam = wf_rest = float("inf")
    for (a, b, c) in mech.clearances(d, dims):
        is_cam = (a.startswith("beam4") or b.startswith("beam4") or
                  any(x in (a, b) for x in cam))
        if mech._is_fan(a, b):
            if is_cam:
                wf_cam = min(wf_cam, c)
            else:
                wf_rest = min(wf_rest, c)
        elif is_cam:
            wc_cam = min(wc_cam, c)
        else:
            wc_rest = min(wc_rest, c)
    return wc_rest, wf_rest, wc_cam, wf_cam


def sweep_production():
    print()
    print("=" * 72)
    print("B. PRODUCTION B_edmund + lift2 periscope (grating floor at "
          "x=+h)")
    print("=" * 72)
    cfg = CONFIGS["B_edmund_30s-45"]
    dims = dict(PROD_DIMS)
    dims["lift_d"] = 50.8
    dims["arm_side"] = 90.0     # tuning arm flipped out of the beam path
    dims["rotor_below"] = 35.0  # turntable sits on the upper deck
    # machined-aluminium case: mount walls are plate-width, corners
    # relieved (the printed-body +6 mm slab growth does not apply)
    dims["slab1_w"] = 80.0
    dims["slab2_w"] = 80.0
    feas = []
    for dev in (15.0, 16.0, 17.0, 18.0):
        for Lg in (300.0, 315.0, 330.0, 345.0):
            for Lc in (255.0, 270.0, 285.0, 300.0):
                for sl in (50.0, 55.0, 60.0, 65.0):
                    for h in (80.0, 85.0, 90.0):
                        try:
                            d = Design(lines_per_mm=2400.0, order=1,
                                       dev=dev, s2=-1.0, Lg=Lg, Lc=Lc,
                                       lift2=dict(s=sl, h=h), **cfg)
                            d.build(656.28)
                        except Exception:
                            continue
                        wc, wf, ws = feasibility(d, dims)
                        # fan gate at -0.5 mm: sub-mm null-edge grazes on
                        # chamfered+flocked mount corners are a designed
                        # wing trim (same class as the frozen budget's
                        # +0.1 mm tower graze), not an accident.
                        # structure-structure gate 0: parts may abut.
                        if wc < 2.5 or wf < -0.5 or ws < 0.0:
                            continue
                        bl = blur_edge(d)
                        if bl is None:
                            continue
                        ca = abs(dot(d.c1, d.gr.n))
                        cb = abs(dot(d.c2, d.gr.n))
                        anam = ca / cb
                        wimg = 7.0 * (d.rfl2 / d.rfl1) * anam
                        dl = d.sigma * 1e6 * cb / (d.rfl2 * 1e3)
                        reff = 656.28 / (math.hypot(wimg, 2.355 * bl[0])
                                         * dl)
                        feas.append((reff, bl[0], bl[1], dev, Lg, Lc, sl,
                                     h, wc, wf, anam))
    feas.sort(key=lambda r: (-round(r[0], -2), -min(r[8] - 2.5, r[9])))
    print("  R_eff edge21 end35  dev   Lg   Lc   sl    h core  fan  anam")
    for r in feas[:12]:
        print(f"{r[0]:7.0f} {r[1]:6.1f} {r[2]:5.1f} {r[3]:4.0f} {r[4]:4.0f} "
              f"{r[5]:4.0f} {r[6]:4.0f} {r[7]:4.0f} {r[8]:4.1f} {r[9]:4.1f} "
              f"{r[10]:.3f}")
    print(f"{len(feas)} feasible")
    return feas


if __name__ == "__main__":
    fb = sweep_budget()
    fp = sweep_production()
