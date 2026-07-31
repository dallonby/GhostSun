#!/usr/bin/env python3
"""Tolerance sweep for the winning all-reflective SHG (config A).

Perturbs one degree of freedom at a time, re-traces Ha at field 0 and the
slit end (3.5 mm), allows SENSOR REFOCUS along the chief ray as the only
compensator (that's what you actually do on the bench), and reports:
  * added blur (RMS, um) at best focus vs the nominal design
  * image centroid shift (um) — alignment/pointing, mostly re-tunable
  * the perturbation magnitude that would add ~3 um RMS (linear scale-out)

Rotations are about each element's mount point (segment center / grating
center). Grating pitch about the groove axis is excluded — that's the tuning
knob, compensated at every observation by construction.
"""

import math
from raytrace import (Design, CONFIGS, stats, add, sub, mul, dot, cross,
                      norm, Paraboloid, Grating)


def rodrigues(v, axis, deg):
    k = norm(axis)
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return add(add(mul(v, c), mul(cross(k, v), s)),
               mul(k, dot(k, v) * (1.0 - c)))


def rotate_paraboloid(p, center, axis, deg):
    new_axis = rodrigues(p.a, axis, deg)
    new_focus = add(center, rodrigues(sub(p.f, center), axis, deg))
    return Paraboloid(new_focus, new_axis, p.P)


def translate_paraboloid(p, dvec):
    return Paraboloid(add(p.f, dvec), p.a, p.P)


def rotate_grating(g, axis, deg):
    return Grating(g.p, rodrigues(g.n, axis, deg),
                   rodrigues(g.g, axis, deg), g.sigma, g.m)


def trace_at(d, xf, lam_nm, shift):
    """Trace with sensor plane moved by `shift` mm along the focused chief."""
    F2s = add(d.F2, mul(d.df, shift))
    lam = lam_nm * 1e-6
    pts = []
    na = 1.0 / (2.0 * 6.9)
    offs = [(0.0, 0.0)]
    for i in range(1, 7):
        r = na * i / 6
        for j in range(12):
            a = 2 * math.pi * j / 12
            offs.append((r * math.cos(a), r * math.sin(a)))
    for (ax, ay) in offs:
        dd0 = norm((ax, ay, 1.0))
        r1 = d.oap1.intersect_reflect((xf, 0.0, 0.0), dd0)
        if r1 is None:
            continue
        r2 = d.gr.intersect_diffract(r1[0], r1[1], lam)
        if r2 is None:
            continue
        r3 = d.oap2.intersect_reflect(r2[0], r2[1])
        if r3 is None:
            continue
        o, dd = r3
        dn = dot(dd, d.sens_n)
        if abs(dn) < 1e-15:
            continue
        t = dot(sub(F2s, o), d.sens_n) / dn
        hit = add(o, mul(dd, t))
        q = sub(hit, F2s)
        pts.append((dot(q, d.sens_x), dot(q, d.sens_y)))
    return pts


def worst_rms_at_best_focus(d, lam0):
    """Refocus sensor (golden search) to minimize the worst field RMS."""
    def worst(shift):
        w = 0.0
        for xf in (0.0, 3.5):
            pts = trace_at(d, xf, lam0, shift)
            if not pts:
                return None
            _, _, rx, ry = stats(pts)
            w = max(w, rx, ry)
        return w * 1e3  # um

    lo, hi = -3.0, 3.0
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1, f2 = worst(c1), worst(c2)
    if f1 is None or f2 is None:
        return None, None
    for _ in range(60):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = worst(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = worst(c2)
        if f1 is None or f2 is None:
            return None, None
    s = 0.5 * (a + b)
    return worst(s), s


def centroid_shift(d, d0, lam0):
    p = trace_at(d, 0.0, lam0, 0.0)
    p0 = trace_at(d0, 0.0, lam0, 0.0)
    if not p or not p0:
        return float("nan")
    cx, cy, _, _ = stats(p)
    cx0, cy0, _, _ = stats(p0)
    return math.hypot(cx - cx0, cy - cy0) * 1e3


def build_nominal():
    from raytrace import CHOSEN
    cfg = CONFIGS[CHOSEN["config"]]
    d = Design(lines_per_mm=2400.0, order=1, dev=CHOSEN["dev"], s2=CHOSEN["s2"], Lg=CHOSEN["Lg"], Lc=CHOSEN["Lc"], **cfg)
    d.build(656.28)
    return d


PERTS = [
    # (label, kind, target, axis-name, magnitude)
    ("slit despace 0.2 mm (z)",      "slit_z",   None, None, 0.2),
    ("OAP1 pitch 0.05 deg (x)",      "rot",  "oap1", "x", 0.05),
    ("OAP1 yaw 0.05 deg (t)",        "rot",  "oap1", "t", 0.05),
    ("OAP1 clock 0.2 deg (chief)",   "rot",  "oap1", "c", 0.2),
    ("OAP1 decenter 0.2 mm (y)",     "dec",  "oap1", "y", 0.2),
    ("OAP1 despace 0.2 mm (z)",      "dec",  "oap1", "z", 0.2),
    ("OAP2 pitch 0.05 deg (x)",      "rot",  "oap2", "x", 0.05),
    ("OAP2 yaw 0.05 deg (t)",        "rot",  "oap2", "t", 0.05),
    ("OAP2 clock 0.2 deg (chief)",   "rot",  "oap2", "c", 0.2),
    ("OAP2 decenter 0.2 mm (y)",     "dec",  "oap2", "y", 0.2),
    ("OAP2 despace 0.2 mm (z)",      "dec",  "oap2", "z", 0.2),
    ("grating yaw 0.1 deg (t: in-plane, perp grooves)", "rot", "gr", "t", 0.1),
    ("grating clock 0.1 deg (n: groove rotation)",      "rot", "gr", "n", 0.1),
]


def apply(d, kind, target, axname, mag):
    """Return a perturbed copy-ish design (mutates fresh nominal)."""
    p = build_nominal()
    if kind == "slit_z":
        # slit fixed at origin; equivalent: move OAP1 along -z by mag
        p.oap1 = translate_paraboloid(p.oap1, (0.0, 0.0, mag))
        return p
    if target == "oap1":
        C, chief = p.C1, p.c0
        axes = {"x": (1.0, 0.0, 0.0),
                "t": norm(cross(chief, (1.0, 0.0, 0.0))),
                "c": chief}
        if kind == "rot":
            p.oap1 = rotate_paraboloid(p.oap1, C, axes[axname], mag)
        else:
            dv = {"y": axes["t"], "z": chief}[axname]
            p.oap1 = translate_paraboloid(p.oap1, mul(dv, mag))
    elif target == "oap2":
        C, chief = p.C2, p.c2
        axes = {"x": (1.0, 0.0, 0.0),
                "t": norm(cross(chief, (1.0, 0.0, 0.0))),
                "c": chief}
        if kind == "rot":
            p.oap2 = rotate_paraboloid(p.oap2, C, axes[axname], mag)
        else:
            dv = {"y": axes["t"], "z": chief}[axname]
            p.oap2 = translate_paraboloid(p.oap2, mul(dv, mag))
    elif target == "gr":
        axes = {"t": p.gr.t, "n": p.gr.n}
        p.gr = rotate_grating(p.gr, axes[axname], mag)
    return p


def run():
    lam0 = 656.28
    d0 = build_nominal()
    base, s0 = worst_rms_at_best_focus(d0, lam0)
    print(f"nominal worst-field RMS at best focus: {base:.2f} um "
          f"(refocus {s0:+.3f} mm)\n")
    print(f"{'perturbation':44s} {'blur@bf':>8s} {'added':>7s} "
          f"{'shift':>8s} {'for +3um':>10s}")
    for (label, kind, target, axname, mag) in PERTS:
        p = apply(d0, kind, target, axname, mag)
        w, s = worst_rms_at_best_focus(p, lam0)
        if w is None:
            print(f"{label:44s}  rays lost — gross misalignment")
            continue
        added = math.sqrt(max(w * w - base * base, 0.0))
        cs = centroid_shift(p, d0, lam0)
        unit = "deg" if kind == "rot" else "mm"
        tol = mag * 3.0 / added if added > 1e-6 else float("inf")
        print(f"{label:44s} {w:8.2f} {added:7.2f} {cs:8.1f} "
              f"{tol:8.3f} {unit}")
    print("\nblur@bf: worst of field 0/3.5mm at Ha after sensor refocus (um "
          "RMS).  added: quadrature increase over nominal.  shift: image "
          "centroid motion (um), mostly absorbed by tuning/pointing.  "
          "for +3um: linearly-scaled magnitude of this DoF alone that adds "
          "3 um RMS blur.")


if __name__ == "__main__":
    run()
