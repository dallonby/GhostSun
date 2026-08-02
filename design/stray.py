#!/usr/bin/env python3
"""Stray-light pass for the SHG: deterministic sources, ranked (stdlib only).

The science target is dark structure inside a dark line core (Ha core is
~16% of continuum). Any diffuse pedestal fills the core first, so this
budget is about VEILING, not throughput.

What this computes:
 1. Which grating orders propagate besides the imaging order (m=0 always;
    others per the grating equation), where each one lands inside the box
    (first hit against the mech.py solids or the housing walls), and how
    much in-band energy it carries. These are the trap/baffle sites.
 2. Mirror micro-roughness total integrated scatter (TIS) vs the roughness
    spec, and what it does to line-core contrast.
 3. A line-core veiling budget combining both with the prefilter.

Energy assumptions (state-of-knowledge, replace with vendor curves):
holographic 2400 l/mm at 656 nm: m=+1 55%, m=0 25%, other propagating
orders ~8%, remainder absorbed/diffuse.
"""

import math
from raytrace import Design, CONFIGS, Grating, add, sub, mul, dot
import mech

EFF = {1: 0.55, 0: 0.25}     # fraction of in-band light per order
EFF_OTHER = 0.08
WALL = dict(x=(-120.0, 120.0), y=(-160.0, 200.0), z=(-100.0, 220.0))


def propagating_orders(d, lam_nm):
    """All orders the grating equation allows for the tuned geometry."""
    lam = lam_nm * 1e-6
    out = []
    for m in range(-4, 5):
        if m == 1:
            continue
        kg = dot(d.c1, d.gr.g)
        kt = dot(d.c1, d.gr.t) + m * lam / d.sigma
        if 1.0 - kg * kg - kt * kt >= 0.0:
            out.append(m)
    return out


def first_hit(o, dd, solids, skip=()):
    """March a ray; return (name, point, dist) of the first solid or wall
    it meets."""
    step = 1.0
    for i in range(1, 500):
        p = add(o, mul(dd, i * step))
        for s in solids:
            if s.name in skip:
                continue
            if s.dist_point(p) <= 0.5:
                return s.name, p, i * step
        if not (WALL["x"][0] < p[0] < WALL["x"][1] and
                WALL["y"][0] < p[1] < WALL["y"][1] and
                WALL["z"][0] < p[2] < WALL["z"][1]):
            return "housing wall", p, i * step
    return "lost", p, 500.0


def order_map(d, dims, label):
    print(f"\n=== {label}: grating-order landing map (in-band light after "
          f"the prefilter) ===")
    solids, _ = mech.build_solids(d, dims)
    lam = 656.28e-6
    orders = propagating_orders(d, 656.28)
    print(f"propagating orders besides m=+1: {orders}")
    for m in orders:
        gr = Grating(d.gr.p, d.gr.n, d.gr.g, d.sigma, m)
        r = gr.intersect_diffract(d.C1, d.c1, lam)
        if r is None:
            continue
        hit, dd = r
        name, p, dist = first_hit(hit, dd, solids,
                                  skip=("rotor", "arm"))
        frac = EFF.get(m, EFF_OTHER)
        bx, by = p[2], p[1]
        # risk: how close does this stray beam pass to the sensor?
        toF2 = sub(d.F2, hit)
        along = dot(toF2, dd)
        if along > 0:
            perp = sub(toF2, mul(dd, along))
            miss = math.sqrt(dot(perp, perp))
        else:
            miss = float("inf")
        print(f"  m={m:+d}  ~{frac*100:.0f}% of in-band light -> {name} at "
              f"body ({bx:.0f}, {by:.0f}), {dist:.0f} mm from grating; "
              f"passes {miss:.0f} mm from the sensor")
    print("  (each landing site needs a matte trap/baffle; bare printed or "
          "aluminum walls are NOT enough, see STRAYLIGHT.md)")


def tis_table():
    print("\n=== mirror micro-roughness: total integrated scatter at "
          "656 nm ===")
    print("sigma (nm RMS)   per mirror   two mirrors")
    for s_nm in (1.0, 2.0, 5.0, 10.0):
        tis = (4.0 * math.pi * s_nm / 656.28) ** 2
        print(f"  {s_nm:4.0f}          {tis*100:7.2f}%     {2*tis*100:7.2f}%")
    print("RFQ spec is 2 nm RMS (amended 2026-08-02; was 10 nm).")


def veiling_budget():
    """Line-core veiling: pedestal relative to the Ha core signal.

    Assumptions (deliberately simple, all stated):
      * prefilter passes a 10 nm band; mean band intensity ~0.85 I_cont
        (line core + wings + continuum)
      * Ha core sits at 0.16 I_cont
      * half of mirror-scattered light stays near-specular and lands on
        the sensor region (conservative for smooth-surface scatter, which
        is strongly forward-peaked)
      * grating diffuse scatter (holographic): 0.3% of in-band into the
        hemisphere, 5% of that reaching the sensor solid angle
      * zero order: 25% of in-band onto its wall spot; matte black trap
        ~5% Lambertian; sensor subtends ~1% of the hemisphere from there
    """
    core = 0.16
    band = 0.85
    print("\n=== line-core veiling budget (pedestal / core signal) ===")
    print("source                                  veiling V")
    for s_nm in (2.0, 10.0):
        tis = (4.0 * math.pi * s_nm / 656.28) ** 2
        v = 2 * tis * 0.5 * band / core
        print(f"  mirror scatter, sigma = {s_nm:2.0f} nm         "
              f"{v*100:6.2f}%")
    v_gr = 0.003 * 0.05 * band / core
    print(f"  grating diffuse scatter (holographic) {v_gr*100:6.2f}%")
    v_z_trap = 0.25 * 0.05 * 0.01 * band / core
    v_z_bare = 0.25 * 0.60 * 0.01 * band / core
    print(f"  zero order into matte trap             {v_z_trap*100:6.2f}%")
    print(f"  zero order onto bare/printed wall      {v_z_bare*100:6.2f}%")
    print("\nfilament contrast multiplier = 1/(1+V): a 16% veil costs 14% "
          "of the contrast; 1% veil costs 1%.")


def run():
    # production geometry (PERFECT_HA numbers; mech.py flags its corridor
    # interference separately, the order landing map is unaffected)
    cfg = CONFIGS["B_edmund_30s-45"]
    d = Design(lines_per_mm=2400.0, order=1, dev=16.0, s2=-1.0,
               Lg=180.0, Lc=290.0, **cfg)
    d.build(656.28)
    dims = dict(grat_w=50.0, colD=50.8, camD=76.2, cam_front_r=50.0,
                cam_front=(-12.0, 24.0), cam_body_r=40.0,
                cam_body=(15.0, 120.0))
    order_map(d, dims, "production B_edmund dev+16 Lg180 Lc290")

    db, dimsb = mech.build_chosen()
    from raytrace import CHOSEN
    order_map(db, dimsb, f"budget {CHOSEN['config']} dev+{CHOSEN['dev']:.0f} "
              f"Lg{CHOSEN['Lg']:.0f} Lc{CHOSEN['Lc']:.0f}")

    tis_table()
    veiling_budget()


if __name__ == "__main__":
    run()
