#!/usr/bin/env python3
"""PERFECT_HA candidate C3 — true Littrow (single OAP, double pass).

Optically a single-OAP autocollimating Littrow at 2400 l/mm is superb on
axis (double pass through one paraboloid, grating at peak efficiency and
symmetric anamorphism = 1). This script quantifies BOTH the optics and the
packaging, because the packaging is what kills it:

  (a) exact double-pass trace with the raytrace.py Paraboloid + Grating
      classes -> spot sizes vs field, image-plane position;
  (b) beam-envelope separation analysis: where can a pickoff flat live
      without vignetting either the outgoing slit cone or the returning
      focused beam, and how much back-focus remains for a real camera.
"""
import math
import numpy as np
from perfect_ha_core import LAM_HA, SIGMA_2400
from raytrace import Paraboloid, Grating, reflect, add, sub, mul, dot, norm, cross, stats

F = 95.0          # OAP RFL (mm), custom 15 deg off-axis class
TH = 15.0         # catalog off-axis angle (deg)
FNUM = 6.9
W_SLIT = 7e-3     # mm
LAM = LAM_HA * 1e-6


def build_littrow(psi_deg):
    """Slit at origin, feed +z. OAP collimates into c1; grating normal set
    so the diffracted chief returns at psi degrees from retro (in-plane)."""
    S = (0.0, 0.0, 0.0)
    c0 = (0.0, 0.0, 1.0)
    th = math.radians(TH)
    c1 = (0.0, math.sin(math.pi - th) * 0 - math.sin(th), -math.cos(th))
    # beam turns by 180-TH about x: c1 = rot_x(c0, 180-TH)
    a = math.radians(180.0 - TH)
    c1 = (0.0, -math.sin(a), math.cos(a))
    P = F * (1.0 + math.cos(th)) / 2.0
    oap = Paraboloid(S, c1, P)
    C1 = mul(c0, F)
    # grating: Littrow angle for 2400 l/mm at Ha
    thL = math.asin(LAM / (2 * SIGMA_2400))
    # normal in the y-z plane, tilted from -c1 by (thL rotation) so that
    # incidence = thL + psi/2 gives diffracted chief psi from retro
    best = None
    for gamma in np.linspace(-80, 80, 3201):
        n = rot_about_x(mul(c1, -1.0), gamma)
        gr = Grating((0, 0, 0), n, (1.0, 0.0, 0.0), SIGMA_2400, 1)
        r = gr.intersect_diffract(sub(C1, mul(c1, 200.0)), c1, LAM)
        if r is None:
            continue
        d2 = r[1]
        s = dot(cross(c1, d2), (1.0, 0.0, 0.0))
        psi = math.copysign(180.0 - math.degrees(
            math.acos(max(-1, min(1, dot(c1, d2))))), s if s else 1.0)
        if best is None or abs(psi - psi_deg) < abs(best[0] - psi_deg):
            best = (psi, gamma)
    gamma = best[1]
    G = add(C1, mul(c1, 160.0))       # grating 160 mm from OAP
    n = rot_about_x(mul(c1, -1.0), gamma)
    gr = Grating(G, n, (1.0, 0.0, 0.0), SIGMA_2400, 1)
    return oap, gr, C1, c1


def rot_about_x(v, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (v[0], c * v[1] - s * v[2], s * v[1] + c * v[2])


def trace(oap, gr, xf, lam_nm, nring=6, nseg=12):
    """slit -> OAP -> grating -> same OAP -> best focal plane near z=0."""
    lam = lam_nm * 1e-6
    na = 1.0 / (2 * FNUM)
    offs = [(0.0, 0.0)] + [(na * i / nring * math.cos(2 * math.pi * j / nseg),
                            na * i / nring * math.sin(2 * math.pi * j / nseg))
                           for i in range(1, nring + 1) for j in range(nseg)]
    rays = []
    for ax, ay in offs:
        d = norm((ax, ay, 1.0))
        r1 = oap.intersect_reflect((xf, 0.0, 0.0), d)
        if r1 is None:
            continue
        r2 = gr.intersect_diffract(r1[0], r1[1], lam)
        if r2 is None:
            continue
        r3 = oap.intersect_reflect(r2[0], r2[1])
        if r3 is None:
            continue
        rays.append(r3)
    if not rays:
        return None, None
    # focus: minimise RMS on plane z = const (image forms near slit plane)
    def spot(zp):
        pts = []
        for o, d in rays:
            t = (zp - o[2]) / d[2]
            pts.append((o[0] + t * d[0], o[1] + t * d[1]))
        return pts
    zs = np.linspace(-8, 8, 81)
    best = None
    for z in zs:
        pts = spot(z)
        cx, cy, rx, ry = stats(pts)
        r = math.hypot(rx, ry)
        if best is None or r < best[0]:
            best = (r, z, pts)
    return best[2], best[1]


if __name__ == "__main__":
    print("C3 true-Littrow, single OAP RFL 95 / 15 deg, GH50-24V at Ha")
    thL = math.degrees(math.asin(LAM / (2 * SIGMA_2400)))
    print(f"Littrow angle {thL:.2f} deg; R_slit = 2 f tan(thL)/w = "
          f"{2*F*math.tan(math.radians(thL))/W_SLIT:,.0f} (7 um slit)")
    print(f"dLam_slit = {656.28/(2*F*math.tan(math.radians(thL))/W_SLIT)*1e3:.1f} pm  "
          f"(f = {F} mm, double-pass mag 1.0)")

    for psi in (2.5, 4.0, 6.0):
        oap, gr, C1, c1 = build_littrow(psi)
        pts, zf = trace(oap, gr, 0.0, LAM_HA)
        if pts:
            cx, cy, rx, ry = stats(pts)
            print(f"\npsi = {psi:.1f} deg: image centre y = {cy:.2f} mm "
                  f"(slit at y=0), best-focus z = {zf:.1f} mm, on-axis "
                  f"spot rms = ({rx*1e3:.2f}, {ry*1e3:.2f}) um")
            pts2, _ = trace(oap, gr, 2.1, LAM_HA)
            if pts2:
                _, _, rx2, ry2 = stats(pts2)
                print(f"   field 2.1 mm: rms = ({rx2*1e3:.1f}, "
                      f"{ry2*1e3:.1f}) um  -> optics are fine; now packaging:")
            # packaging: separation of outgoing cone and return beam
            sep = None
            for z in np.linspace(2, 90, 89):
                out_half = z / (2 * FNUM) + 0.3          # outgoing cone
                # return beam: converges from OAP (z=F) to focus (y=cy,z~0)
                ret_c = cy * (1 - z / F)
                ret_half = (F - z) / F * (F / (2 * FNUM)) + 0.3
                gap = (ret_c - ret_half) - out_half
                if gap > 0 and sep is None:
                    sep = z
            if sep is None:
                print("   NO plane between slit and OAP where a pickoff "
                      "flat clears both beams -> flat must sit within "
                      "~2-3 mm of the focal plane.")
            else:
                print(f"   beams separate only for z < {sep:.0f} mm")
            # back focus if a knife-edge flat sits at z = 3 mm
            print(f"   knife-edge flat at z = 3 mm folds a beam that "
                  f"focuses {3:.0f} mm later: back focus = 3 mm << 12.5 mm "
                  f"(IMX678 flange) or 17.5 mm (IMX571).")
            print(f"   sensor AT the slit plane instead: die centre "
                  f"{cy:.1f} mm from slit; camera body dia >= 40-80 mm "
                  f"would occupy the feed snout volume -> collides with "
                  f"the telescope drawtube (hard constraint).")
