#!/usr/bin/env python3
"""One-element SHG study: exact trace of a CONCAVE grating (stdlib only).

slit -> concave grating -> sensor. The grating is a spherical cap whose
groove pattern is either CLASSICAL (equally spaced along the chord, the
ruled-engine case) or HOLOGRAPHIC (interference of two recording point
sources; the aberration-corrected "Type IV" case). Diffraction is exact:
at each surface point the local grating vector K(P) is the tangential
gradient of the groove phase, and the outgoing ray satisfies
  d_out_tang = d_in_tang + m*lambda*K(P),  |d_out| = 1  (reflection).

Frame: slit center at origin, slit length along x. Grating pole at
distance LA along +z. Dispersion plane y-z.
"""

import math
from raytrace import add, sub, mul, dot, cross, norm, stats

LAM = 656.28e-6           # mm
SIGMA = 1.0 / 2400.0      # mm
LAM_REC = 441.6e-6        # HeCd recording line (typical), mm


class ConcaveGrating:
    def __init__(self, pole, axis, Rc, phase_grad):
        """pole: center of ruled area (on sphere). axis: unit inward
        normal at the pole (toward the incoming light). Rc: radius.
        phase_grad(P) -> grating vector K (cycles/mm, 3-vector; its
        tangential part is used)."""
        self.pole = pole
        self.axis = norm(axis)
        self.Rc = Rc
        self.center = sub(pole, mul(self.axis, -Rc))  # sphere center is
        # BEHIND the concave face: pole + axis*Rc with axis toward light
        self.center = add(pole, mul(self.axis, Rc))
        self.K = phase_grad

    def intersect(self, o, d):
        oc = sub(o, self.center)
        b = 2.0 * dot(d, oc)
        c = dot(oc, oc) - self.Rc * self.Rc
        disc = b * b - 4 * c
        if disc < 0:
            return None
        r = math.sqrt(disc)
        # want the intersection on the concave (near) side
        ts = sorted(t for t in ((-b - r) / 2, (-b + r) / 2) if t > 1e-9)
        for t in ts:
            p = add(o, mul(d, t))
            n = norm(sub(self.center, p))     # inward normal (toward light)
            if dot(n, self.axis) > 0.5:       # correct cap, not far side
                return p, n
        return None

    def diffract(self, o, d, lam, m=1):
        hit = self.intersect(o, d)
        if hit is None:
            return None
        p, n = hit
        K = self.K(p)
        Kt = sub(K, mul(n, dot(K, n)))        # tangential grating vector
        dt = sub(d, mul(n, dot(d, n)))        # tangential incoming
        ot = add(dt, mul(Kt, m * lam))        # tangential outgoing
        rem = 1.0 - dot(ot, ot)
        if rem <= 0:
            return None
        # reflection: outgoing normal component opposite the incoming's
        sgn = -1.0 if dot(d, n) > 0 else 1.0
        return p, add(ot, mul(n, sgn * math.sqrt(rem)))


def classical_K(sigma, e_hat):
    """Ruled: grooves are intersections with planes e.P = k*sigma."""
    def K(P):
        return mul(e_hat, 1.0 / sigma)
    return K


def holographic_K(C, D, lam_rec):
    """Recorded: grooves are fringes of two point sources at C and D
    (both diverging)."""
    def K(P):
        uc = norm(sub(P, C))
        ud = norm(sub(P, D))
        return mul(sub(uc, ud), 1.0 / lam_rec)
    return K


def conjugate_K(C, D, lam):
    """The stigmatic hologram for conjugates C (source) and D (real
    image) at wavelength lam: rays from C leave toward D exactly. Made
    physically with one diverging and one CONVERGING recording beam, or
    written directly by e-beam. K = (unit(D-P) - unit(P-C)) / lam."""
    def K(P):
        return mul(sub(norm(sub(D, P)), norm(sub(P, C))), 1.0 / lam)
    return K


class OneElementSHG:
    """Geometry: slit at origin; grating pole at (0,0,LA); grating tilted
    so the incidence angle at the pole is alpha (rotation about x).
    Sensor plane found from the diffracted chief at 656."""

    def __init__(self, Rc, LA, alpha_deg, LB, K_factory, sensor_tilt=0.0):
        self.Rc, self.LA, self.LB = Rc, LA, LB
        a = math.radians(alpha_deg)
        self.pole = (0.0, 0.0, LA)
        # inward normal at pole, in y-z plane, making alpha with the
        # incoming chief (which travels +z):
        self.n0 = (0.0, math.sin(a), -math.cos(a))
        self.gr = ConcaveGrating(self.pole, self.n0, Rc, None)
        # dispersion direction at the pole: in-plane, perpendicular to
        # n0, signed so first order folds back toward the slit side
        self.e0 = norm(cross(self.n0, (1.0, 0.0, 0.0)))
        self.gr.K = K_factory(self)
        # chief ray at 656 defines the sensor
        r = self.gr.diffract((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), LAM)
        if r is None:
            raise RuntimeError("chief lost")
        p, dch = r
        self.F = add(p, mul(dch, LB))
        self.dch = dch
        ns = mul(dch, -1.0)
        st = math.radians(sensor_tilt)
        # tilt sensor about x
        c, s = math.cos(st), math.sin(st)
        ns = (ns[0], c * ns[1] - s * ns[2], s * ns[1] + c * ns[2])
        self.sens_n = norm(ns)
        self.sens_x = norm(sub((1.0, 0.0, 0.0),
                               mul(self.sens_n, self.sens_n[0])))
        self.sens_y = norm(cross(self.sens_n, self.sens_x))

    def trace(self, xf, lam_nm, fnum=6.9, nring=5, nseg=10, focus=0.0):
        lam = lam_nm * 1e-6
        na = 1.0 / (2.0 * fnum)
        offs = [(0.0, 0.0)] + [
            (na * i / nring * math.cos(2 * math.pi * j / nseg),
             na * i / nring * math.sin(2 * math.pi * j / nseg))
            for i in range(1, nring + 1) for j in range(nseg)]
        F = add(self.F, mul(self.dch, focus))
        pts = []
        for (ax, ay) in offs:
            d = norm((ax, ay, 1.0))
            r = self.gr.diffract((xf, 0.0, 0.0), d, lam)
            if r is None:
                continue
            o, dd = r
            dn = dot(dd, self.sens_n)
            if abs(dn) < 1e-12:
                continue
            t = dot(sub(F, o), self.sens_n) / dn
            hit = add(o, mul(dd, t))
            q = sub(hit, F)
            pts.append((dot(q, self.sens_x), dot(q, self.sens_y)))
        return pts

    def worst(self, fields=(0.0, 2.1, 3.5), lams=(655.78, 656.28, 656.78),
              focus=0.0):
        w = 0.0
        for xf in fields:
            for lam in lams:
                pts = self.trace(xf, lam, focus=focus)
                if len(pts) < 10:
                    return 1e9
                _, _, rx, ry = stats(pts)
                w = max(w, rx * 1e3, ry * 1e3)
        return w


def classical_factory(sys):
    return classical_K(SIGMA, sys.e0)


def make_holo_factory(rc_mm, gam_deg, lam_rec=LAM_REC):
    """Recording points in the dispersion plane, distances from the pole.
    D's angle solves the pole line density = 1/SIGMA."""
    def factory(sys):
        g = math.radians(gam_deg)
        s_delta = math.sin(g) - lam_rec / SIGMA
        if abs(s_delta) > 1:
            raise RuntimeError("no recording solution")
        dlt = math.asin(s_delta)
        e0, n0, pole = sys.e0, sys.n0, sys.pole
        def pt(r, ang):
            dirv = add(mul(n0, math.cos(ang)), mul(e0, math.sin(ang)))
            return add(pole, mul(dirv, r))
        C = pt(rc_mm[0], g)
        D = pt(rc_mm[1], dlt)
        # K = grad(|P-C| - |P-D|)/lam; unit(P-C) at the pole is MINUS the
        # pole->C direction, so swap roles to make the pole density +1/sigma
        return holographic_K(D, C, lam_rec)
    return factory


def optimize():
    fixed_sigma_note = "2400 l/mm at pole"

    def build(p):
        return OneElementSHG(p["Rc"], p["LA"], p["alpha"], p["LB"],
                             make_holo_factory((p["rC"], p["rD"]),
                                               p["gam"]),
                             sensor_tilt=p["tilt"])

    def cost(p):
        try:
            return build(p).worst()
        except Exception:
            return 1e9

    def descend(params, steps, iters=120):
        cur = cost(params)
        for _ in range(iters):
            improved = False
            for k in params:
                for sgn in (1.0, -1.0):
                    trial = dict(params)
                    trial[k] = params[k] + sgn * steps[k]
                    c = cost(trial)
                    if c < cur:
                        params, cur = trial, c
                        improved = True
            if not improved:
                for k in steps:
                    steps[k] *= 0.5
                if max(steps.values()) < 5e-4:
                    break
        return params, cur

    # classical baseline for the record
    base = OneElementSHG(400.0, 165.0, 65.6, 299.0, classical_factory)
    best_cl = min(base.worst(focus=f) for f in
                  [x * 0.5 for x in range(-20, 21)])
    print(f"classical ruled concave: worst RMS {best_cl:.0f} um "
          f"(the astigmatism a Type IV recording must correct)")

    best = (None, 1e9)
    starts = []
    for Rc in (350.0, 450.0):
        for alpha in (55.0, 65.6):
            for gam in (50.0, 70.0, 82.0):
                starts.append(dict(Rc=Rc, LA=Rc * 0.42, alpha=alpha,
                                   LB=Rc * 0.72, tilt=0.0, gam=gam,
                                   rC=Rc * 0.6, rD=Rc * 1.1))
    for st in starts:
        steps = dict(Rc=40.0, LA=25.0, alpha=5.0, LB=30.0, tilt=6.0,
                     gam=6.0, rC=60.0, rD=60.0)
        p, c = descend(dict(st), steps)
        if c < best[1]:
            best = (p, c)
    p, c = best
    print("best design: " +
          ", ".join(f"{k}={v:.2f}" for k, v in sorted(p.items())))
    d = build(p)
    print(f"corrected concave ({fixed_sigma_note}): worst RMS {c:.1f} um "
          f"over slit +-3.5 mm, +-0.5 nm")
    for xf in (0.0, 2.1, 3.5):
        pts = d.trace(xf, 656.28)
        _, _, rx, ry = stats(pts)
        print(f"  field {xf} mm: {rx*1e3:.1f} x {ry*1e3:.1f} um")
    p0 = d.trace(0.0, 656.28)
    p1 = d.trace(1.0, 656.28)
    mag = abs(stats(p1)[0] - stats(p0)[0])
    dl0 = d.trace(0.0, 656.48)
    disp = abs(stats(dl0)[1] - stats(p0)[1]) / 0.2
    wimg = 7e-3 * mag
    print(f"  spatial mag {mag:.2f}; dispersion {disp*1e2:.1f} um/0.1nm; "
          f"slit image {wimg*1e3:.1f} um -> slit-limited R ~ "
          f"{656.28/(wimg/disp):.0f}")
    return d, p, c


if __name__ == "__main__":
    optimize()
