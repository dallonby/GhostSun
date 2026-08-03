#!/usr/bin/env python3
"""Exact vector raytracer for an all-reflective (OAP + plane grating + OAP)
spectroheliograph — stdlib only.

Geometry conventions
--------------------
* World frame: slit center S at origin. Chief ray slit -> collimator along +z.
* Fold plane (dispersion plane) = y-z plane. Slit length & grating grooves = x.
* Catalog "off-axis angle" theta_c is the angle between the focused and
  collimated BEAMS at the mirror (Thorlabs/Edmund convention). A 90 deg OAP
  turns the beam by 90 deg; a 45 deg OAP turns it by 135 deg.
  RFL = 2*P/(1+cos(theta_c)), P = parent focal length.
* Paraboloid local frame: origin at FOCUS, +z along parabola axis (toward
  opening). Surface: x^2 + y^2 = 4 P (z + P). Rays from the focus reflect
  into +z (collimated); collimated rays traveling -z focus onto the origin.
* Grating: plane, grooves along x. Vector diffraction: component along
  grooves conserved; in-plane tangential component gets +/- m*lambda/d.

Validation: with a perfectly collimated chain, the on-axis field at the
tuned wavelength must focus to a numerical point (<1e-9 mm RMS). Aberrations
appear only for slit-end fields and window-edge wavelengths.
"""

import math
import sys

V = tuple  # 3-vectors as tuples


def add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def mul(a, s): return (a[0]*s, a[1]*s, a[2]*s)
def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a):
    n = math.sqrt(dot(a, a))
    return (a[0]/n, a[1]/n, a[2]/n)
def ang(a, b):
    c = max(-1.0, min(1.0, dot(norm(a), norm(b))))
    return math.degrees(math.acos(c))
def rot_x(v, deg):
    """Rotate v about +x axis by deg (fold-plane rotations)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (v[0], c*v[1] - s*v[2], s*v[1] + c*v[2])
def reflect(d, n):
    return sub(d, mul(n, 2.0*dot(d, n)))


class Paraboloid:
    """Parent paraboloid defined by focus point (world), axis unit vector
    (world, toward opening), and parent focal length P (mm)."""

    def __init__(self, focus, axis, P):
        self.f = focus
        self.a = norm(axis)
        self.P = P
        # build local frame: z = axis; x = world x orthogonalized; y = z cross x
        zl = self.a
        xl = sub((1.0, 0.0, 0.0), mul(zl, zl[0]))
        self.xl = norm(xl)
        self.yl = cross(zl, self.xl)
        self.zl = zl

    def w2l(self, p):
        q = sub(p, self.f)
        return (dot(q, self.xl), dot(q, self.yl), dot(q, self.zl))

    def w2l_dir(self, d):
        return (dot(d, self.xl), dot(d, self.yl), dot(d, self.zl))

    def l2w_dir(self, d):
        return add(add(mul(self.xl, d[0]), mul(self.yl, d[1])), mul(self.zl, d[2]))

    def intersect_reflect(self, o_w, d_w):
        """Return (point_world, dir_world) after reflection, or None."""
        o = self.w2l(o_w)
        d = self.w2l_dir(d_w)
        P = self.P
        a = d[0]*d[0] + d[1]*d[1]
        b = 2.0*(o[0]*d[0] + o[1]*d[1]) - 4.0*P*d[2]
        c = o[0]*o[0] + o[1]*o[1] - 4.0*P*o[2] - 4.0*P*P
        if abs(a) < 1e-15:
            if abs(b) < 1e-15:
                return None
            ts = [-c/b]
        else:
            disc = b*b - 4*a*c
            if disc < 0:
                return None
            r = math.sqrt(disc)
            ts = [(-b - r)/(2*a), (-b + r)/(2*a)]
        ts = [t for t in ts if t > 1e-9]
        if not ts:
            return None
        t = min(ts)
        p = (o[0]+t*d[0], o[1]+t*d[1], o[2]+t*d[2])
        n = norm((2.0*p[0], 2.0*p[1], -4.0*P))
        dr = reflect(d, n)
        p_w = add(add(mul(self.xl, p[0]), mul(self.yl, p[1])),
                  add(mul(self.zl, p[2]), self.f))
        return p_w, self.l2w_dir(dr)


class Grating:
    """Plane reflection grating. point: on plane. normal: unit, facing the
    incoming beam. grooves along gvec (unit, in plane). sigma = period mm.
    order m (signed)."""

    def __init__(self, point, normal, gvec, sigma, m):
        self.p = point
        self.n = norm(normal)
        self.g = norm(gvec)
        self.t = norm(cross(self.g, self.n))  # in-plane dispersion direction
        self.sigma = sigma
        self.m = m

    def intersect_diffract(self, o, d, lam_mm):
        dn = dot(d, self.n)
        if abs(dn) < 1e-15:
            return None
        t = dot(sub(self.p, o), self.n) / dn
        if t < 1e-9:
            return None
        hit = add(o, mul(d, t))
        kg = dot(d, self.g)
        kt = dot(d, self.t) + self.m * lam_mm / self.sigma
        rem = 1.0 - kg*kg - kt*kt
        if rem < 0:
            return None  # evanescent
        # reflected side: leave along +n if arrived against n
        kn = math.sqrt(rem) if dn < 0 else -math.sqrt(rem)
        dd = add(add(mul(self.g, kg), mul(self.t, kt)), mul(self.n, kn))
        return hit, norm(dd)


class Design:
    """Assembled spectrograph. All lengths mm, angles deg (catalog OAP angles).
    s2: +1/-1 fold sense of camera OAP relative to collimator fold.
    dev: grating chief deviation angle (between incident and diffracted chief),
         signed: + folds back toward the collimator side."""

    def __init__(self, rfl1, th1, rfl2, th2, lines_per_mm, order,
                 dev=35.0, s2=1.0, Lg=90.0, Lc=90.0,
                 fold2=None, fold3=None, fold4=None, lift3=None,
                 lift2=None):
        """fold2/fold3/fold4: optional in-plane fold FLATS,
        dict(s=..., delta=...). fold2 sits in the collimated beam s mm
        after OAP1 (relocates the grating and everything downstream);
        fold3 sits in the collimated diffracted beam s mm after the
        grating; fold4 sits in the converging beam s mm after OAP2.
        delta = in-plane change of the chief direction (deg, +x rotation
        sense). Plane mirrors add NO aberration; they only relocate
        everything downstream (Lc and rfl2 are PATH lengths)."""
        self.rfl1, self.th1 = rfl1, th1
        self.rfl2, self.th2 = rfl2, th2
        self.sigma = 1.0 / lines_per_mm
        self.m = order
        self.dev = dev
        self.s2 = s2
        self.Lg, self.Lc = Lg, Lc
        self.fold2, self.fold3, self.fold4 = fold2, fold3, fold4
        self.lift2 = lift2   # dict(s=..., h=...): periscope in beam2:
        # grating and everything downstream move to level x=+h
        self.lift3 = lift3   # dict(s=..., h=...): vertical periscope
        # in the diffracted beam: two 45-degree flats lift the beam +h in
        # x (along the slit axis). Two reflections: no chirality flip,
        # dev/s2 keep their unfolded meanings; exactly aberration-free.

    def build(self, lam0_nm):
        """Position everything; tune grating so the chief at lam0 leaves at
        self.dev degrees from the incident collimated axis."""
        lam = lam0_nm * 1e-6  # mm
        S = (0.0, 0.0, 0.0)
        c0 = (0.0, 0.0, 1.0)                      # slit -> OAP1
        th1 = self.th1
        # beam turns by (180 - th1): outgoing collimated direction
        c1 = rot_x(c0, (180.0 - th1))             # fold sense s1 = +1 fixed
        P1 = self.rfl1 * (1.0 + math.cos(math.radians(th1))) / 2.0
        oap1 = Paraboloid(S, c1, P1)
        C1 = mul(c0, self.rfl1)
        assert abs(ang(sub(C1, S), c0)) < 1e-9

        # optional vertical periscope in the collimated beam: the
        # grating and everything after it live at level x = +h
        if self.lift2:
            sl2, hl2 = self.lift2["s"], self.lift2["h"]
            up2 = (1.0, 0.0, 0.0)
            QA = add(C1, mul(c1, sl2))
            QB = add(QA, mul(up2, hl2))
            self.lift2A = (QA, norm(sub(up2, c1)))
            self.lift2B = (QB, norm(sub(c1, up2)))
        else:
            self.lift2A = self.lift2B = None
        # optional fold flat in the collimated beam before the grating
        if self.fold2:
            s2f, d2f = self.fold2["s"], self.fold2["delta"]
            P2f = add(C1, mul(c1, s2f))
            c1b = rot_x(c1, d2f)
            self.flat2 = (P2f, norm(sub(c1b, c1)))
            G = add(P2f, mul(c1b, self.Lg - s2f))
        else:
            c1b, self.flat2 = c1, None
            if self.lift2:
                G = add(QB, mul(c1, self.Lg - sl2 - hl2))
            else:
                G = add(C1, mul(c1, self.Lg))
        gvec = (1.0, 0.0, 0.0)

        # Find grating rotation gamma (about x) so the diffracted chief
        # departs from RETRO (Littrow) by exactly `dev` degrees, on the side
        # given by dev's sign. Signed off-retro angle:
        #   psi = sign(cross(c1,c2).x) * (180 - ang(c1, c2))
        # psi = 0 at exact Littrow; we solve psi = dev. (A previous version
        # bisected on atan2 of the raw angle and converged to a spurious
        # root at exact retro -> OAP2 coincided with OAP1. See RESULTS.md.)
        target = self.dev

        def diff_dir(gamma):
            n = rot_x(mul(c1b, -1.0), gamma)      # normal facing the beam
            gr = Grating(G, n, gvec, self.sigma, self.m)
            r = gr.intersect_diffract(sub(G, mul(c1b, 10.0)), c1b, lam)
            return (gr, r[1]) if r else (gr, None)

        def psi_of(d2):
            s = dot(cross(c1b, d2), (1.0, 0.0, 0.0))
            return math.copysign(180.0 - ang(c1b, d2),
                                 s if s != 0 else 1.0)

        best = None
        prev = None
        for i in range(1601):
            g = -80.0 + 160.0 * i / 1600.0
            gr, d2 = diff_dir(g)
            if d2 is None:
                prev = None
                continue
            f = psi_of(d2) - target
            if prev is not None and prev[1] * f < 0 and abs(f) < 30:
                a, fa = prev
                b, fb = g, f
                for _ in range(80):
                    mid = 0.5 * (a + b)
                    _, dm = diff_dir(mid)
                    if dm is None:
                        break
                    fm = psi_of(dm) - target
                    if fa * fm <= 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                best = 0.5 * (a + b)
                break
            prev = (g, f)
        if best is None:
            raise RuntimeError("grating tune failed (no solution for dev)")
        self.gr, c2 = diff_dir(best)
        self.gamma = best
        got = psi_of(c2)
        if abs(got - target) > 0.01:
            raise RuntimeError(f"tune landed at psi={got:.2f}, want {target}")

        th2 = self.th2
        # optional vertical periscope in the diffracted beam
        if self.lift3:
            sl, hl = self.lift3["s"], self.lift3["h"]
            up = (1.0, 0.0, 0.0)
            PA = add(G, mul(c2, sl))
            PB = add(PA, mul(up, hl))
            self.liftA = (PA, norm(sub(up, c2)))
            self.liftB = (PB, norm(sub(c2, up)))
        else:
            self.liftA = self.liftB = None
        # optional fold flat in the collimated diffracted beam
        if self.fold3:
            s3, d3 = self.fold3["s"], self.fold3["delta"]
            P3 = add(G, mul(c2, s3))
            c2b = rot_x(c2, d3)
            self.flat3 = (P3, norm(sub(c2b, c2)))
            C2 = add(P3, mul(c2b, self.Lc - s3))
        else:
            c2b, self.flat3 = c2, None
            if self.lift3:
                C2 = add(PB, mul(c2, self.Lc - sl - hl))
            else:
                C2 = add(G, mul(c2, self.Lc))
        df0 = rot_x(c2b, self.s2 * (180.0 - th2))  # focused chief direction
        assert abs(ang(c2b, df0) - (180.0 - th2)) < 1e-6
        # the paraboloid's own focus is the UNFOLDED focal point; a fold4
        # flat relocates the physical focus F2 without touching the shape
        F2u = add(C2, mul(df0, self.rfl2))
        P2 = self.rfl2 * (1.0 + math.cos(math.radians(th2))) / 2.0
        oap2 = Paraboloid(F2u, mul(c2b, -1.0), P2)
        if self.fold4:
            s4, d4 = self.fold4["s"], self.fold4["delta"]
            P4 = add(C2, mul(df0, s4))
            df = rot_x(df0, d4)
            self.flat4 = (P4, norm(sub(df, df0)))
            F2 = add(P4, mul(df, self.rfl2 - s4))
        else:
            df, F2, self.flat4 = df0, F2u, None

        self.oap1, self.oap2 = oap1, oap2
        self.S, self.C1, self.G, self.C2, self.F2 = S, C1, G, C2, F2
        self.c0, self.c1, self.c2, self.df = c0, c1, c2, df
        self.c1b, self.c2b, self.df0 = c1b, c2b, df0
        # sensor frame: origin F2, normal df; xs ~ world x, ys completes
        ns = mul(df, -1.0)
        xs = sub((1.0, 0.0, 0.0), mul(ns, ns[0]))
        self.sens_x = norm(xs)
        self.sens_y = norm(cross(ns, self.sens_x))
        self.sens_n = ns

    def trace(self, xf, lam_nm, fnum=6.9, nring=6, nseg=12):
        """Trace a cone from slit point (xf,0,0). Returns list of sensor
        (x, y) mm. Cone axis +z (telecentric feed approx)."""
        lam = lam_nm * 1e-6
        pts = []
        na = 1.0 / (2.0 * fnum)
        offs = [(0.0, 0.0)]
        for i in range(1, nring+1):
            r = na * i / nring
            for j in range(nseg):
                a = 2*math.pi*j/nseg
                offs.append((r*math.cos(a), r*math.sin(a)))
        o0 = (xf, 0.0, 0.0)
        for (ax, ay) in offs:
            d = norm((ax, ay, 1.0))
            r1 = self.oap1.intersect_reflect(o0, d)
            if r1 is None:
                continue
            if self.lift2A:
                r1 = plane_reflect(r1[0], r1[1], self.lift2A)
                if r1 is None:
                    continue
                r1 = plane_reflect(r1[0], r1[1], self.lift2B)
                if r1 is None:
                    continue
            if self.flat2:
                r1 = plane_reflect(r1[0], r1[1], self.flat2)
                if r1 is None:
                    continue
            r2 = self.gr.intersect_diffract(r1[0], r1[1], lam)
            if r2 is None:
                continue
            if self.liftA:
                r2 = plane_reflect(r2[0], r2[1], self.liftA)
                if r2 is None:
                    continue
                r2 = plane_reflect(r2[0], r2[1], self.liftB)
                if r2 is None:
                    continue
            if self.flat3:
                r2 = plane_reflect(r2[0], r2[1], self.flat3)
                if r2 is None:
                    continue
            r3 = self.oap2.intersect_reflect(r2[0], r2[1])
            if r3 is None:
                continue
            if self.flat4:
                r3 = plane_reflect(r3[0], r3[1], self.flat4)
                if r3 is None:
                    continue
            o, dd = r3
            dn = dot(dd, self.sens_n)
            if abs(dn) < 1e-15:
                continue
            t = dot(sub(self.F2, o), self.sens_n) / dn
            hit = add(o, mul(dd, t))
            q = sub(hit, self.F2)
            pts.append((dot(q, self.sens_x), dot(q, self.sens_y)))
        return pts


def plane_reflect(o, d, plane):
    """Reflect ray (o, d) off an infinite plane mirror (point, normal)."""
    p, n = plane
    dn = dot(d, n)
    if abs(dn) < 1e-15:
        return None
    t = dot(sub(p, o), n) / dn
    if t < 1e-9:
        return None
    hit = add(o, mul(d, t))
    return hit, reflect(d, n)


def stats(pts):
    n = len(pts)
    cx = sum(p[0] for p in pts)/n
    cy = sum(p[1] for p in pts)/n
    rx = math.sqrt(sum((p[0]-cx)**2 for p in pts)/n)
    ry = math.sqrt(sum((p[1]-cy)**2 for p in pts)/n)
    return cx, cy, rx, ry


CONFIGS = {
    "A_edmund_30-45": dict(rfl1=108.89, th1=30.0, rfl2=178.53, th2=45.0),
    "B_edmund_30s-45": dict(rfl1=81.79, th1=30.0, rfl2=178.53, th2=45.0),
    "C_thor_ag_45-45": dict(rfl1=152.4, th1=45.0, rfl2=304.8, th2=45.0),
    "D_thor_al_90-90": dict(rfl1=101.6, th1=90.0, rfl2=152.4, th2=90.0),
    "E_edmund_30-30": dict(rfl1=108.89, th1=30.0, rfl2=272.23, th2=30.0),
    # Ha-only budget prototype: Thorlabs protected silver, in stock, £394/pair
    "T_budget_ha": dict(rfl1=50.8, th1=45.0, rfl2=101.6, th2=45.0),
    # v2: MPD124 collimator + MPD246 60 deg camera (£353). The 60 deg fold
    # exits the camera parallel to the telescope (kills the 45/45 pair's
    # convergent-exit collision family). Frozen via mech.py sweep.
    "T_budget_ha_v2": dict(rfl1=50.8, th1=45.0, rfl2=101.6, th2=60.0),
}

# Chosen geometry for the CURRENT BUILD. Production broadband design:
#   two-floor periscope layout, FOLDS.md (dev 17, Lg 315, Lc 255,
#   lift2 s=65 h=80) -- CAD started in prod_case.scad, not yet frozen.
# Active: Ha-only budget prototype v3 (FROZEN 2026-08-02, FOLDS.md):
# Thorlabs silver MPD124 + MPD144 (45/45) + one O25.4 protected-silver
# flat 46 mm after OAP2 folding the focused beam +90 deg away from the
# telescope. Worst core clearance +4.4 mm, fan +1.7 mm, edge blur
# 9.3 um (+-0.5 nm), R~27.7k slit-limited, ~L467 optics. The flat sits
# in a FIXED printed seat: fold_tol.py shows its tilt tolerance is
# unbounded (pure image steering), so it gets no adjusters.
# Extra keys are the mechanical envelope inputs mech/body_export verify.
CHOSEN = dict(config="T_budget_ha", dev=24.0, s2=-1.0,
              Lg=70.0, Lc=130.0, fold4=dict(s=46.0, delta=90.0),
              colD=25.4, camD=25.4, grat_w=25.0,
              km100_modules=True, km100_scallop_half=19.0,
              slab1_w=56.0, slab2_w=56.0, flat4_d=25.4)

LINES = [  # (label, lambda nm, lines/mm, order)
    ("CaK 393", 393.37, 2400.0, 1),
    ("Ha 656", 656.28, 2400.0, 1),
    ("He 1083", 1083.0, 1200.0, 1),
]

FIELDS = [0.0, 2.1, 3.5]         # mm along slit (disk edge, slit end)
DLAM = [0.0, 0.5, 10.0]          # nm offsets: line, fast-mode edge, rich-mode edge
DEVS = [10.0, 15.0, 20.0, 25.0, 30.0]   # off-Littrow angles (deg)


def run():
    fnum = 6.9
    print(f"feed f/{fnum}; fields (mm along slit): {FIELDS}; "
          f"window offsets (nm): {DLAM}")
    print("spot values are RMS radii in um at the sensor: "
          "sx = spatial (along slit), sy = spectral (dispersion)")
    for name, cfg in CONFIGS.items():
        mag = cfg["rfl2"]/cfg["rfl1"]
        print(f"\n=== {name}  (RFL {cfg['rfl1']}/{cfg['rfl2']} mm, "
              f"angles {cfg['th1']}/{cfg['th2']} deg, mag {mag:.2f}x) ===")
        for (label, lam0, lpmm, order) in LINES:
            best = None
            for dev in [d for v in DEVS for d in (v, -v)]:
                for s2 in (1.0, -1.0):
                    try:
                        d = Design(lines_per_mm=lpmm, order=order,
                                   dev=dev, s2=s2, **cfg)
                        d.build(lam0)
                    except (RuntimeError, AssertionError):
                        continue
                    # on-axis sanity: should be a numerical point
                    p0 = d.trace(0.0, lam0, fnum)
                    if not p0:
                        continue
                    _, _, rx0, ry0 = stats(p0)
                    worst = 0.0
                    rows = []
                    for xf in FIELDS:
                        for dl in DLAM:
                            pts = d.trace(xf, lam0+dl, fnum)
                            if not pts:
                                rows = None
                                break
                            cx, cy, rx, ry = stats(pts)
                            rows.append((xf, dl, rx*1e3, ry*1e3))
                            if dl <= 0.5:  # optimize for fast-mode window
                                worst = max(worst, rx*1e3, ry*1e3)
                        if rows is None:
                            break
                    if rows is None:
                        continue
                    key = (worst, dev, s2)
                    if best is None or key[0] < best[0][0]:
                        best = (key, rows, rx0*1e3, ry0*1e3)
            if best is None:
                print(f"  {label}: NO GEOMETRY (evanescent or tune fail)")
                continue
            (worst, dev, s2), rows, rx0, ry0 = best
            # mechanical clearance: perpendicular distance from OAP1 center
            # to the diffracted chief line (must exceed OAP1 half-size +
            # beam half-width, ~40 mm, for the return beam to miss OAP1)
            dchk = Design(lines_per_mm=lpmm, order=order, dev=dev, s2=s2,
                          **cfg)
            dchk.build(lam0)
            off = sub(dchk.C1, dchk.G)
            clr = math.sqrt(max(dot(off, off) -
                                dot(off, dchk.c2) ** 2, 0.0))
            print(f"  {label}: dev={dev:+.0f} s2={s2:+.0f}  "
                  f"OAP1-to-return-beam clearance {clr:.0f} mm  "
                  f"on-axis sanity rms=({rx0:.2e},{ry0:.2e}) um")
            for (xf, dl, rx, ry) in rows:
                flag = " <-- worst" if max(rx, ry) == worst else ""
                print(f"    field {xf:4.1f} mm, dlam {dl:+5.1f} nm: "
                      f"sx={rx:7.2f} sy={ry:7.2f}{flag}")
    print("\nbudget: slit image ~10.5-17.5 um wide depending on mag; "
          "pixel 3.76 um. blur RMS < ~3 um is negligible, < ~8 um usable.")


if __name__ == "__main__":
    sys.exit(run())
