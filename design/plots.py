#!/usr/bin/env python3
"""Charts for the all-reflective SHG design (chosen config B).

layout.png — fold-plane (y-z) beam layout with element footprints:
             the mechanical clearance check at the 20 deg deviation.
spots.png  — spot-diagram grid: rows = lines, cols = slit fields, with the
             slit-image width and one sensor pixel for scale.
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from raytrace import Design, CONFIGS, stats, add, mul, norm, dot, sub, cross

from raytrace import CHOSEN
CFG = CONFIGS[CHOSEN["config"]]
NA = 1.0 / (2 * 6.9)


def build(lam_nm, lpmm):
    d = Design(lines_per_mm=lpmm, order=1, dev=CHOSEN["dev"], s2=CHOSEN["s2"], Lg=CHOSEN["Lg"], Lc=CHOSEN["Lc"], **CFG)
    d.build(lam_nm)
    return d


def yz(p):
    return p[2], p[1]  # plot z horizontal, y vertical


def trace_path(d, ay, lam_nm):
    """Trace one ray in the fold plane (pupil offset ay); return polyline."""
    lam = lam_nm * 1e-6
    o = (0.0, 0.0, 0.0)
    dd = norm((0.0, ay, 1.0))
    pts = [o]
    r1 = d.oap1.intersect_reflect(o, dd)
    pts.append(r1[0])
    r2 = d.gr.intersect_diffract(r1[0], r1[1], lam)
    pts.append(r2[0])
    r3 = d.oap2.intersect_reflect(r2[0], r2[1])
    pts.append(r3[0])
    o3, d3 = r3
    t = dot(sub(d.F2, o3), d.sens_n) / dot(d3, d.sens_n)
    pts.append(add(o3, mul(d3, t)))
    return pts


def element_span(center, tangent, half):
    a = add(center, mul(tangent, half))
    b = add(center, mul(tangent, -half))
    return [yz(a), yz(b)]


def layout():
    lam0 = 656.28
    d = build(lam0, 2400.0)
    fig, ax = plt.subplots(figsize=(9, 7))
    rays = [(-NA, "#87b5e5"), (0.0, "#1f77b4"), (NA, "#87b5e5")]
    for ay, col in rays:
        pts = [yz(p) for p in trace_path(d, ay, lam0)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=col, lw=1.2 if ay == 0 else 0.8, zorder=2)
    # element footprints (fold-plane tangents)
    x = (1.0, 0.0, 0.0)
    for center, chief, half, name, col in [
        (d.C1, d.c0, 0.9 * 50.8 / 2, "OAP1 collimator\n#35-607 Ø50.8 30°", "#d62728"),
        (d.G, d.c1, 25.0, "grating 50 mm\n(Shelyak 25 fits)", "#2ca02c"),
        (d.C2, d.c2, 0.9 * 76.2 / 2, "OAP2 camera\n#35-588 Ø76.2 45°", "#d62728"),
    ]:
        tang = norm(cross(x, chief))
        seg = element_span(center, tang, half)
        ax.plot([s[0] for s in seg], [s[1] for s in seg],
                color=col, lw=4, solid_capstyle="butt", zorder=3)
        ax.annotate(name, yz(center), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=col)
    # slit + sensor
    ax.plot(*yz((0.0, 0.0, 0.0)), "ks", ms=6, zorder=4)
    ax.annotate("slit", (0, 0), textcoords="offset points", xytext=(-24, -4),
                fontsize=9)
    tang = norm(cross(x, d.df))
    seg = element_span(d.F2, tang, 8.0)
    ax.plot([s[0] for s in seg], [s[1] for s in seg], color="k", lw=4,
            zorder=3)
    ax.annotate("sensor (IMX571)", yz(d.F2), textcoords="offset points",
                xytext=(8, -14), fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)  — fold / dispersion plane")
    ax.set_title("All-reflective SHG, chosen config B — Hα layout "
                 "(20° off-Littrow, opposite OAP folds)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("layout.png", dpi=160)
    print("layout.png written")


def spots():
    lines = [("Ca K 393 nm (2400/mm)", 393.37, 2400.0),
             ("Hα 656 nm (2400/mm)", 656.28, 2400.0),
             ("He I 1083 nm (1200/mm)", 1083.0, 1200.0)]
    fields = [0.0, 2.1, 3.5]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, (label, lam0, lpmm) in enumerate(lines):
        d = build(lam0, lpmm)
        for j, xf in enumerate(fields):
            ax = axes[i][j]
            for dl, col, m in [(0.0, "#1f77b4", "o"), (0.5, "#d62728", "x")]:
                pts = d.trace(xf, lam0 + dl)
                cx, cy, _, _ = stats(pts)
                ax.scatter([(p[0] - cx) * 1e3 for p in pts],
                           [(p[1] - cy) * 1e3 for p in pts],
                           s=7, c=col, marker=m, linewidths=0.8,
                           label=f"Δλ = {dl:+.1f} nm")
            # slit image width band (dispersion axis) and one pixel
            w_img = 7.0 * CFG["rfl2"] / CFG["rfl1"]
            ax.axhspan(-w_img / 2, w_img / 2, color="0.85", zorder=0)
            ax.add_patch(plt.Rectangle((-3.76 / 2, -3.76 / 2), 3.76, 3.76,
                                       fill=False, ec="0.4", ls=":"))
            lim = 30
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"field {xf:.1f} mm"
                             + (" (disk edge)" if xf == 2.1 else
                                " (slit end)" if xf == 3.5 else " (center)"),
                             fontsize=10)
            if j == 0:
                ax.set_ylabel(label + "\ndispersion (µm)", fontsize=9)
            if i == 2:
                ax.set_xlabel("along slit (µm)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Spot diagrams, config B — grey band = 7 µm slit image "
                 f"({7*CFG['rfl2']/CFG['rfl1']:.1f} µm); dotted square = one 3.76 µm pixel",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("spots.png", dpi=160)
    print("spots.png written")


if __name__ == "__main__":
    layout()
    spots()
