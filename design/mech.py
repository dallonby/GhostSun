#!/usr/bin/env python3
"""Mechanical interference checker for the SHG design (stdlib only).

Every collision we have hit so far (diffracted beam through the OAP1 mount
slab, camera into the telescope, sensor on the snout) was caught by eyeball
after the optics were already "optimal". This module makes clearance a
first-class constraint: it builds real mechanical envelopes from the same
raytrace.Design object the optics use, models the BEAMS as solids at their
full Fraunhofer fan width (not chief rays, not the geometric cone), and
computes the minimum distance for every non-adjacent pair.

Use:
  * standalone:  python3 mech.py          -> clearance table for CHOSEN
  * as a guard:  mech.assert_clear(d)     -> SystemExit on any interference
  * in sweeps:   mech.ok(d, margin=5.0)   -> bool, cheap enough per candidate

Solids (world frame: slit at origin, +z toward OAP1, y-z = fold plane,
x = vertical/slit length):
  slit_tower, oap1_cell + oap1_mount, oap2_cell + oap2_mount, rotor
  (+tuning arm), camera_front, camera_body, snout, telescope, and
  beam1..beam4 each at two tiers (core / fan; see Beam).

Envelope dimensions default to shg_body.scad / mounts/kinematic_mount.scad
values; override via the dims argument for other configurations.

FAIL below 0 mm, WARN below the margin (default 5 mm).
"""

import math
from raytrace import (Design, CONFIGS, CHOSEN, add, sub, mul, dot, cross,
                      norm)

# ---------------------------------------------------------------- dims ----
DIMS = dict(
    slab1_w=86.0,       # OAP1 KM plate + 6 (shg_body slab1W)
    slab2_w=86.0,       # OAP2 KM plate + 6 (shg_body slab2W)
    slab_half_x=43.0,   # vertical half-extent of a mirror module
    module_depth=59.0,  # mirror face -> back of mount slab (mirrorStack+slabT)
    rotor_r=28.0,       # grating turntable radius (rotD/2)
    arm_r=9.0,          # tuning arm envelope radius
    arm_len=85.0,       # pivot -> past the tangent post (armR + 15)
    cam_front_r=29.0,   # tilt flange / focuser stack radius (camFocD/2)
    cam_front=(-12.0, 22.0),   # extent along df relative to the sensor plane
    cam_body_r=20.0,    # camera barrel radius (camBodyD/2)
    cam_body=(15.0, 85.0),     # extent along df relative to the sensor plane
    snout_r=28.0,       # scopeFlangeD/2
    snout=(-88.0, -38.0),      # snout extent along z (wall face to tip)
    # FSQ-85 OTA in two segments behind the M48 face at the snout tip:
    # focuser drawtube + adapter stack first, then the main tube.
    scope_r=35.0,       # focuser/drawtube radius
    scope=(-190.0, -90.0),     # drawtube extent along z
    scope2_r=50.0,      # main tube radius (focuser knobs excluded)
    scope2=(-540.0, -190.0),   # main tube extent along z
    # Slit tower, CAD-true asymmetric footprint (shg_body.scad): the
    # downstream face is a thin blade wall at z = tower_z[1] (the beam3
    # side), the cartridge structure extends upstream to tower_z[0].
    tower_y_half=10.0,  # across-beam half width (cartridge + walls)
    tower_z=(-10.0, 4.0),  # upstream/downstream faces (blade at +4)
    tower_x_half=37.5,  # vertical half extent
    cell_depth=22.0,    # mirror + cell depth at the module face
    tower_cx=-22.5,     # tower box center height (deck at -beamH)
    slit_w_um=7.0,      # slit width for the Fraunhofer fan
    fnum=6.9,
    lam_nm=656.28,
    band_nm=5.0,        # half-band surviving the prefilter (10 nm FWHM)
    trim=22.0,          # ignore this much beam near its own end elements
    grat_w=25.0,        # grating ruled width in the dispersion plane (mm)
    ca_frac=0.9,        # clear-aperture fraction of mirror diameters
)


# ------------------------------------------------------------ geometry ----
def _seg_point(a, b, p):
    ab = sub(b, a)
    t = dot(sub(p, a), ab) / max(dot(ab, ab), 1e-12)
    t = max(0.0, min(1.0, t))
    q = add(a, mul(ab, t))
    return math.sqrt(dot(sub(p, q), sub(p, q)))


def _seg_seg(p1, q1, p2, q2):
    """Min distance between segments p1q1 and p2q2."""
    d1, d2, r = sub(q1, p1), sub(q2, p2), sub(p1, p2)
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return math.sqrt(dot(r, r))
    if a < 1e-12:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    else:
        c = dot(d1, r)
        if e < 1e-12:
            t, s = 0.0, max(0.0, min(1.0, -c / a))
        else:
            b = dot(d1, d2)
            den = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / den)) if den > 1e-12 \
                else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, max(0.0, min(1.0, (b - c) / a))
    c1 = add(p1, mul(d1, s))
    c2 = add(p2, mul(d2, t))
    return math.sqrt(dot(sub(c1, c2), sub(c1, c2)))


class Box:
    """Oriented box: center, three unit axes, three half sizes."""

    def __init__(self, name, center, axes, half):
        self.name, self.c, self.axes, self.h = name, center, axes, half

    def dist_point(self, p):
        q = sub(p, self.c)
        loc = [dot(q, a) for a in self.axes]
        ex = [max(abs(loc[i]) - self.h[i], 0.0) for i in range(3)]
        return math.sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)

    def surface_points(self, step=8.0):
        pts = []
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            nu = max(2, int(2 * self.h[u] / step) + 1)
            nv = max(2, int(2 * self.h[v] / step) + 1)
            for sgn in (-1.0, 1.0):
                for i in range(nu):
                    for j in range(nv):
                        cu = -self.h[u] + 2 * self.h[u] * i / (nu - 1)
                        cv = -self.h[v] + 2 * self.h[v] * j / (nv - 1)
                        p = add(self.c, mul(self.axes[axis],
                                            sgn * self.h[axis]))
                        p = add(p, mul(self.axes[u], cu))
                        pts.append(add(p, mul(self.axes[v], cv)))
        return pts


class Capsule:
    """Sphere-swept segment."""

    def __init__(self, name, p0, p1, r):
        self.name, self.p0, self.p1, self.r = name, p0, p1, r

    def dist_point(self, p):
        return _seg_point(self.p0, self.p1, p) - self.r

    def axis_points(self, step=3.0):
        L = math.sqrt(dot(sub(self.p1, self.p0), sub(self.p1, self.p0)))
        n = max(2, int(L / step) + 1)
        return [add(self.p0, mul(sub(self.p1, self.p0), i / (n - 1)))
                for i in range(n)]


class Beam:
    """Chain of (point, radius) stations along a traced chief ray.

    tier is "core" (geometric f/# cone: carries ~94% of the energy, any
    interference is a hard FAIL) or "fan" (full Fraunhofer fan edge: the
    outer wings carry a few percent, interference means vignetting plus a
    scatter source at that spot, reported as VIGNETTE)."""

    def __init__(self, name, tier, p0, p1, r0, r1, adjacent, step=2.0):
        self.name = f"{name}[{tier}]"
        self.tier = tier
        self.adjacent = set(adjacent)
        L = math.sqrt(dot(sub(p1, p0), sub(p1, p0)))
        n = max(2, int(L / step) + 1)
        self.stations = []
        for i in range(n):
            t = i / (n - 1)
            p = add(p0, mul(sub(p1, p0), t))
            self.stations.append((p, r0 + (r1 - r0) * t, t * L, L))


def _dist(a, b):
    """Min clearance between two solids (Beam/Box/Capsule)."""
    if isinstance(a, Beam):
        trim = DIMS["trim"] if b.name in a.adjacent else 0.0
        best = float("inf")
        for (p, r, s, L) in a.stations:
            if s < trim or (L - s) < trim:
                continue
            best = min(best, b.dist_point(p) - r)
        return best
    if isinstance(b, Beam):
        return _dist(b, a)
    if isinstance(a, Capsule) and isinstance(b, Capsule):
        return _seg_seg(a.p0, a.p1, b.p0, b.p1) - a.r - b.r
    if isinstance(a, Capsule):
        return min(b.dist_point(p) for p in a.axis_points()) - a.r \
            if isinstance(b, Box) else _dist(b, a)
    if isinstance(b, Capsule):
        return _dist(b, a)
    return min(b.dist_point(p) for p in a.surface_points())


# -------------------------------------------------------------- solids ----
def build_solids(d, dims=None):
    """Mechanical envelopes + beam solids for a built Design."""
    dm = dict(DIMS)
    if dims:
        dm.update(dims)
    X = (1.0, 0.0, 0.0)

    def module(name, face_center, back_dir, mirror_d, slab_w):
        """Two boxes: the mirror in its cell at the face, and the KM
        plate/slab stack behind it (only the slab is slab_w wide; beside
        the cell, the first cell_depth mm is free air)."""
        a1 = norm(back_dir)
        a3 = norm(cross(a1, X))
        cell_d = dm["cell_depth"]
        cell_half = (mirror_d + 6.0) / 2.0
        cell = Box(name + "_cell",
                   add(face_center, mul(a1, cell_d / 2.0 - 2.0)),
                   [a1, X, a3], [cell_d / 2.0 + 2.0, cell_half, cell_half])
        slab_d = dm["module_depth"] - cell_d
        slab = Box(name + "_mount",
                   add(face_center, mul(a1, cell_d + slab_d / 2.0)),
                   [a1, X, a3],
                   [slab_d / 2.0, dm["slab_half_x"], slab_w / 2.0])
        return [cell, slab]

    z0, z1 = dm["tower_z"]
    solids = [
        Box("slit_tower", (dm["tower_cx"], 0.0, (z0 + z1) / 2.0),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            [dm["tower_x_half"], dm["tower_y_half"], (z1 - z0) / 2.0]),
        *module("oap1", d.C1, mul(d.c1, -1.0), dm.get("colD", 50.8),
                dm["slab1_w"]),
        *module("oap2", d.C2, d.c2, dm.get("camD", 76.2), dm["slab2_w"]),
        Capsule("rotor", (-60.0, d.G[1], d.G[2]), (40.0, d.G[1], d.G[2]),
                dm["rotor_r"]),
        Capsule("snout", (0.0, 0.0, dm["snout"][0]),
                (0.0, 0.0, dm["snout"][1]), dm["snout_r"]),
        Capsule("telescope", (0.0, 0.0, dm["scope"][0]),
                (0.0, 0.0, dm["scope"][1]), dm["scope_r"]),
        Capsule("telescope_ota", (0.0, 0.0, dm["scope2"][0]),
                (0.0, 0.0, dm["scope2"][1]), dm["scope2_r"]),
        Capsule("camera_front", add(d.F2, mul(d.df, dm["cam_front"][0])),
                add(d.F2, mul(d.df, dm["cam_front"][1])), dm["cam_front_r"]),
        Capsule("camera_body", add(d.F2, mul(d.df, dm["cam_body"][0])),
                add(d.F2, mul(d.df, dm["cam_body"][1])), dm["cam_body_r"]),
    ]
    # stray-light vane (optional): vertical strip at y = vane[2],
    # from world z = vane[0] to vane[1], thickness vane[3]. body_export
    # computes vane[0] so the diffracted fan clears it.
    if dm.get("vane"):
        x0, x1, vy, vt = dm["vane"]
        solids.append(Box("vane", (2.5, vy + vt / 2.0, (x0 + x1) / 2.0),
                          [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                           (0.0, 0.0, 1.0)],
                          [62.5, vt / 2.0, (x1 - x0) / 2.0]))

    # grating tuning arm: body angle = ang(gr normal) + 270 (shg_body.scad)
    th = math.atan2(d.gr.n[1], d.gr.n[2]) + math.radians(270.0)
    arm_dir = (0.0, math.sin(th), math.cos(th))
    solids.append(Capsule("arm", d.G, add(d.G, mul(arm_dir, dm["arm_len"])),
                          dm["arm_r"]))

    # Beams at two tiers. Half-angles from the slit:
    #   core: 1/(2 f#)             geometric cone, ~94% of the energy
    #   fan:  1/(2 f#) + lambda/w  out to the first Fraunhofer nulls
    # Each is then clamped by the apertures the light actually passes:
    # collimator CA, grating projected ruled width, camera CA.
    lam = dm["lam_nm"] * 1e-6
    w = dm["slit_w_um"] * 1e-3
    col_ca = dm["ca_frac"] * dm.get("colD", d.rfl1 / 2.0) / 2.0
    cam_ca = dm["ca_frac"] * dm.get("camD", d.rfl2 / 2.0) / 2.0
    ca, cb = abs(dot(d.c1, d.gr.n)), abs(dot(d.c2, d.gr.n))
    dbeta = dm["band_nm"] * 1e-6 / (d.sigma * cb)   # band spread half-angle
    beams = []
    for tier, u in (("core", 1.0 / (2.0 * dm["fnum"])),
                    ("fan", 1.0 / (2.0 * dm["fnum"]) + lam / w)):
        r_col = min(d.rfl1 * u, col_ca)     # radius at the collimator
        foot = min(r_col / ca, dm["grat_w"] / 2.0)  # grating footprint
        r_g2 = foot * cb                    # re-projected after diffraction
        r_c2 = min(r_g2 + d.Lc * dbeta, cam_ca)     # at the camera mirror
        beams += [
            Beam("beam1_slit_oap1", tier, d.S, d.C1, 0.1, r_col,
                 ["slit_tower", "oap1_cell"]),
            Beam("beam2_oap1_grating", tier, d.C1, d.G, r_col, r_col,
                 ["oap1_cell", "rotor", "arm"]),
            Beam("beam3_grating_oap2", tier, d.G, d.C2, r_g2, r_c2,
                 ["rotor", "arm", "oap2_cell"]),
            Beam("beam4_oap2_sensor", tier, d.C2, d.F2, r_c2,
                 0.5 + d.rfl2 * dbeta,
                 ["oap2_cell", "camera_front", "camera_body"]),
        ]
    return solids, beams


_SKIP = {  # rigidly connected pairs: contact is by construction
    frozenset(("rotor", "arm")),
    frozenset(("oap1_cell", "oap1_mount")),
    frozenset(("oap2_cell", "oap2_mount")),
    frozenset(("camera_front", "camera_body")),
    frozenset(("snout", "telescope")),
    frozenset(("snout", "telescope_ota")),
    frozenset(("telescope", "telescope_ota")),
    frozenset(("slit_tower", "snout")),
}


def clearances(d, dims=None):
    """All pairwise clearances, sorted worst first: [(a, b, mm), ...]."""
    solids, beams = build_solids(d, dims)
    out = []
    for i, a in enumerate(solids):
        for b in solids[i + 1:]:
            if frozenset((a.name, b.name)) in _SKIP:
                continue
            out.append((a.name, b.name, _dist(a, b)))
    for bm in beams:
        for s in solids:
            if s.name in bm.adjacent:
                continue
            out.append((bm.name, s.name, _dist(bm, s)))
    out.sort(key=lambda r: r[2])
    return out


def _is_fan(name_a, name_b):
    return "[fan]" in name_a or "[fan]" in name_b


def ok(d, margin=5.0, dims=None, fan_margin=0.0):
    """True if no core/structure pair is within margin and no fan is
    vignetted. Use as the accept test inside geometry sweeps."""
    for (a, b, c) in clearances(d, dims):
        if _is_fan(a, b):
            if c < fan_margin:
                return False
        elif c < margin:
            return False
    return True


def assert_clear(d, margin=5.0, dims=None):
    """Print violations; SystemExit on any hard interference (structure or
    core beam). Fan interferences print as VIGNETTE warnings only."""
    import sys
    fatal = False
    for (a, b, c) in clearances(d, dims):
        if c >= margin:
            continue
        if _is_fan(a, b):
            if c < 0:
                print(f"// VIGNETTE: {a} vs {b}: {c:+.1f} mm "
                      f"(fan wings clipped: throughput loss + scatter here)",
                      file=sys.stderr)
        elif c < 0:
            print(f"// FATAL collision: {a} vs {b}: {c:+.1f} mm",
                  file=sys.stderr)
            fatal = True
        else:
            print(f"// WARNING clearance: {a} vs {b}: {c:+.1f} mm",
                  file=sys.stderr)
    if fatal:
        raise SystemExit("FATAL: mechanical interference (see above)")


def build_chosen(lam_nm=656.28):
    """Returns (design, dims) for the CHOSEN geometry."""
    cfg = CONFIGS[CHOSEN["config"]]
    d = Design(lines_per_mm=2400.0, order=1, dev=CHOSEN["dev"],
               s2=CHOSEN["s2"], Lg=CHOSEN["Lg"], Lc=CHOSEN["Lc"], **cfg)
    d.build(lam_nm)
    dims = {k: CHOSEN[k] for k in ("colD", "camD", "grat_w",
                                   "slab1_w", "slab2_w") if k in CHOSEN}
    return d, dims


if __name__ == "__main__":
    d, dims = build_chosen()
    print(f"CHOSEN = {CHOSEN['config']} dev={CHOSEN['dev']} "
          f"Lg={CHOSEN['Lg']} Lc={CHOSEN['Lc']}  (all clearances in mm; "
          f"beams at full diffraction-fan width)\n")
    rows = clearances(d, dims)
    wide = max(len(a) + len(b) for (a, b, _) in rows) + 6
    shown = 0
    for (a, b, c) in rows:
        if _is_fan(a, b):
            flag = "VIGNETTE" if c < 0 else ("fan-tight" if c < 5 else "ok")
        else:
            flag = "FAIL" if c < 0 else ("WARN" if c < 5.0 else "ok")
        if c < 25.0:
            print(f"  {(a + ' / ' + b):{wide}s} {c:8.1f}   {flag}")
            shown += 1
    n_fail = sum(1 for (a, b, c) in rows if c < 0 and not _is_fan(a, b))
    n_vig = sum(1 for (a, b, c) in rows if c < 0 and _is_fan(a, b))
    print(f"\n{n_fail} hard collision(s), {n_vig} fan vignette(s), "
          f"{len(rows)} pairs checked ({len(rows) - shown} clear pairs "
          f">25 mm not shown)")
