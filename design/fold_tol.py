#!/usr/bin/env python3
"""Alignment-tolerance pass for the two fold-flat candidate designs.

Question: do the extra mirrors make the instrument harder for one person
to align on a bench? Perturb every adjustable DOF one at a time, allow
SENSOR REFOCUS as the only compensator (what you actually do), report
added blur and the magnitude that costs +3 um RMS.

Expectation to verify: flats in collimated or converging light steer the
image (retuned/repointed for free) but add almost no blur, so their
tolerances should come out 10-100x looser than the OAP pitches.
"""

import math
from raytrace import (Design, CONFIGS, stats, add, sub, mul, dot, cross,
                      norm, Paraboloid, Grating)
from tolerance import (rodrigues, rotate_paraboloid, rotate_grating,
                       translate_paraboloid)


def make_budget():
    d = Design(lines_per_mm=2400.0, order=1, dev=24.0, s2=-1.0,
               Lg=70.0, Lc=150.0, fold4=dict(s=46.0, delta=90.0),
               **CONFIGS["T_budget_ha"])
    d.build(656.28)
    return d


def make_production():
    d = Design(lines_per_mm=2400.0, order=1, dev=16.0, s2=-1.0,
               Lg=330.0, Lc=270.0, lift2=dict(s=60.0, h=90.0),
               **CONFIGS["B_edmund_30s-45"])
    d.build(656.28)
    return d


def worst_at_best_focus(d, fields=(0.0, 3.5), lam0=656.28):
    F2_nom = d.F2

    def worst(shift):
        d.F2 = add(F2_nom, mul(d.df, shift))
        w = 0.0
        for xf in fields:
            pts = d.trace(xf, lam0)
            if not pts:
                d.F2 = F2_nom
                return None
            _, _, rx, ry = stats(pts)
            w = max(w, rx, ry)
        d.F2 = F2_nom
        return w * 1e3

    gr = (math.sqrt(5) - 1) / 2
    a, b = -3.0, 3.0
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1, f2 = worst(c1), worst(c2)
    if f1 is None or f2 is None:
        return None
    for _ in range(50):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = worst(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = worst(c2)
        if f1 is None or f2 is None:
            return None
    return worst(0.5 * (a + b))


def rotate_flat(plane, axis, deg):
    p, n = plane
    return (p, rodrigues(n, axis, deg))


def perturb(maker, what, mag):
    d = maker()
    X = (1.0, 0.0, 0.0)
    if what == "oap1 pitch":
        d.oap1 = rotate_paraboloid(d.oap1, d.C1, X, mag)
    elif what == "oap2 pitch":
        d.oap2 = rotate_paraboloid(d.oap2, d.C2, X, mag)
    elif what == "grating clock":
        d.gr = rotate_grating(d.gr, d.gr.n, mag)
    elif what == "grating inplane yaw":
        d.gr = rotate_grating(d.gr, d.gr.t, mag)
    elif what == "slit despace":
        # slit fixed at the origin; equivalent: move OAP1 along +z
        d.oap1 = translate_paraboloid(d.oap1, (0.0, 0.0, mag))
    elif what.startswith("flat"):
        name, axis_name = what.split()[0], what.split()[1]
        plane = getattr(d, name)
        p, n = plane
        if axis_name == "pitch":       # about the vertical/slit axis? no:
            axis = X                   # about x (fold-plane tilt)
        else:                          # "yaw": about the in-plane transverse
            axis = norm(cross(n, X))
        setattr(d, name, rotate_flat(plane, axis, mag))
    return d


def run(label, maker, flat_names):
    print(f"\n=== {label} ===")
    d0 = maker()
    base = worst_at_best_focus(d0)
    print(f"nominal worst-field RMS at best focus: {base:.2f} um")
    perts = [("oap1 pitch", 0.05, "deg"),
             ("oap2 pitch", 0.05, "deg"),
             ("grating clock", 0.1, "deg"),
             ("grating inplane yaw", 0.1, "deg"),
             ("slit despace", 0.2, "mm")]
    for fn in flat_names:
        perts.append((f"{fn} pitch", 0.1, "deg"))
        perts.append((f"{fn} yaw", 0.1, "deg"))
    print(f"{'DOF':26s} {'blur@bf':>8s} {'added':>7s} {'for +3um':>12s}")
    for (what, mag, unit) in perts:
        try:
            dp = perturb(maker, what, mag)
        except Exception as e:
            print(f"{what:26s}  skip ({e})")
            continue
        w = worst_at_best_focus(dp)
        if w is None:
            print(f"{what:26s}  rays lost")
            continue
        added = math.sqrt(max(w * w - base * base, 0.0))
        tol = mag * 3.0 / added if added > 1e-6 else float("inf")
        tol_s = f"{tol:9.3f} {unit}" if tol < 1e3 else f"   >1000 {unit}"
        print(f"{what:26s} {w:8.2f} {added:7.2f} {tol_s:>12s}")


if __name__ == "__main__":
    run("BUDGET 45/45 + fold4 flat (dev24 Lg70 Lc150 s4=46 d4=+90)",
        make_budget, ["flat4"])
    run("PRODUCTION B_edmund + beam2 periscope (dev16 Lg330 Lc270 "
        "s=60 h=90)", make_production, ["lift2A", "lift2B"])
