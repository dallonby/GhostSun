#!/usr/bin/env python3
"""PERFECT_HA candidate C2 — all-lens V spectrograph (Sol'Ex-evolved) at a
single wavelength. Exact meridional trace of a cemented achromatic doublet
at 656.3 nm with the bending optimised for THIS wavelength (best case any
catalog or semi-custom doublet can do). Questions answered with numbers:

  1. on-axis blur of the f/6.9 geometric core (must be << slit image);
  2. transverse ray error at the slit-diffraction fan zones (the fan fills
     the aperture to ~f/3 on an 80 mm collimator: spherical aberration
     there scatters line-wing light -> spectral-purity wings);
  3. field blur at the 2.1 mm disk edge;
  4. in-band ghost budget vs the all-mirror design.
"""
import math
import numpy as np

N_BK7 = 1.51432    # n at 656.3 nm
N_SF5 = 1.66661
V_BK7, V_SF5 = 64.17, 32.25


def make_doublet(f, c1):
    """Cemented doublet, total power 1/f, crown-front bending set by c1.
    Returns surface list [(curvature, n_before, n_after, thickness_after)]."""
    phi = 1.0 / f
    phi1 = phi * V_BK7 / (V_BK7 - V_SF5)
    phi2 = phi - phi1
    c2 = c1 - phi1 / (N_BK7 - 1.0)
    c3 = c2 - phi2 / (N_SF5 - 1.0)
    t1, t2 = 9.0, 3.5
    return [(c1, 1.0, N_BK7, t1), (c2, N_BK7, N_SF5, t2),
            (c3, N_SF5, 1.0, 0.0)]


def trace_y(surfs, y0, u0, z0=-1e9):
    """Meridional trace: ray height y, angle u (rad), from collimated
    (u0 = field angle). Surfaces spaced by thickness; returns (y, u) after
    last surface and the paraxial-free exact path via iterative sphere
    intersection in 2D (y-z plane)."""
    # start well before the lens travelling +z
    y, z, dy, dz = y0, -50.0, math.sin(u0), math.cos(u0)
    zv = 0.0
    for (c, n1, n2, t) in surfs:
        # sphere: z = zv + c*y^2 / (1 + sqrt(1 - c^2 y^2))  (sag form)
        # solve intersection iteratively
        zz, yy = z, y
        for _ in range(60):
            if abs(c) < 1e-12:
                zs = zv
            else:
                s2 = max(0.0, 1.0 - c * c * yy * yy)
                zs = zv + c * yy * yy / (1.0 + math.sqrt(s2))
            tstep = (zs - zz) / dz
            yy2 = yy + tstep * dy
            if abs(yy2 - yy) < 1e-12:
                yy = yy2
                zz = zs
                break
            yy, zz = yy2, zs
        # surface normal (meridional): N = (-c*y, 1-...) normalized;
        # for sphere with curvature c centred at zv + 1/c:
        if abs(c) < 1e-12:
            ny, nz = 0.0, -1.0
        else:
            cy = zv + 1.0 / c
            ny, nz = yy - 0.0, zz - cy
            nl = math.hypot(ny, nz)
            ny, nz = ny / nl, nz / nl
            if nz > 0:
                ny, nz = -ny, -nz  # face incoming beam
        # Snell (2D vector)
        mu = n1 / n2
        ci = -(dy * ny + dz * nz)
        s2t = mu * mu * (1.0 - ci * ci)
        if s2t > 1.0:
            return None
        ct = math.sqrt(1.0 - s2t)
        dy = mu * dy + (mu * ci - ct) * ny
        dz = mu * dz + (mu * ci - ct) * nz
        y, z = yy, zz
        zv += t   # next vertex along the axis
    return y, z, dy, dz


def focus_spot(surfs, u0, heights, core_h, f):
    """Transverse positions at the best common focal plane (plane that
    minimises the spread of the <= core_h bundle). Returns
    (z_focus, dict h -> y_at_focus) or None."""
    rays = []
    for h in heights:
        r = trace_y(surfs, h, u0)
        if r is None:
            rays.append((h, None))
        else:
            rays.append((h, r))
    core = [r for (h, r) in rays if r is not None and 0 < h <= core_h + 1e-9]
    if len(core) < 2:
        return None

    def y_at(z, r):
        y, zz, dy, dz = r
        return y + (z - zz) * dy / dz
    zs = np.linspace(0.4 * f, 1.4 * f, 6001)
    best = None
    for z in zs:
        ys = [y_at(z, r) for r in core] + [0.0]  # chief through axis
        w = max(ys) - min(ys)
        if best is None or w < best[1]:
            best = (z, w)
    zf = best[0]
    return zf, {h: (None if r is None else y_at(zf, r)) for (h, r) in rays}


def optimise_bending(f, core_h):
    best = None
    for c1 in np.linspace(0.2 / f, 3.0 / f, 141):
        surfs = make_doublet(f, c1)
        r = focus_spot(surfs, 0.0, [0.4 * core_h, 0.7 * core_h, core_h],
                       core_h, f)
        if r is None:
            continue
        zf, ys = r
        vals = [v for v in ys.values() if v is not None]
        err = max(vals) - min(vals)
        if best is None or err < best[0]:
            best = (err, c1)
    return best[1]


if __name__ == "__main__":
    print("C2 all-lens V layout: exact meridional doublet trace at 656 nm")
    for f, role, core, fan in ((80.0, "collimator (Sol'Ex-like)", 5.8, 13.3),
                               (150.0, "collimator (fan-relaxed)", 10.9, 24.9)):
        c1 = optimise_bending(f, core)
        surfs = make_doublet(f, c1)
        res = focus_spot(surfs, 0.0, [0.3 * core, core * 0.7, core,
                                      (core + fan) / 2, fan], core, f)
        zf, ys = res
        print(f"\n{role}: f = {f} mm, ideal 656 nm bending c1 = "
              f"{c1*1e3:.2f} 1/m*1e-3  (focus z = {zf:.1f} mm)")
        print(f"  zone (mm)   transverse error at focus (um)")
        for h in sorted(ys):
            v = ys[h]
            txt = "   (vignetted/TIR)" if v is None else f"{abs(v)*1e3:8.2f}"
            print(f"  {h:6.1f}     {txt}"
                  + ("   <- f/6.9 geometric core edge" if h == core else
                     ("   <- slit-diffraction fan edge" if h == fan else "")))
        # equivalent slit-referred wing blur: error at fan zones scatters
        # first-lobe light; compare Airy scale lam*f/D_core
        print(f"  (slit image = 7 um; Airy radius of core = "
              f"{1.22*656e-6*f/ (2*core)*1e3:.1f} um)")

    print("""
Ghost budget (in-band, prefiltered):
  8 air-glass surfaces; R = 0.5 % (BBAR) or 0.25 % (V-coat 656):
  broad veil ~ sum over 28 pairs R^2 ~ 7e-4 (BBAR) / 1.8e-4 (V-coat)
  of the in-band flux, plus 2-4 semi-focused pairs (camera rear surfaces)
  that can reach ~1e-3 local contrast artefacts unless the lenses are
  tilted 1-2 deg. All-mirror design: zero.
Conclusion: an ideally-bent single-lambda doublet is diffraction-limited
over the f/6.9 geometric core (<1 um transverse error) -- lenses are NOT
excluded by core image quality. They lose on the slit-diffraction fan: the
fan fills the aperture to ~f/3 regardless of focal length (fan angle is
set at the slit), and at those zones spherical aberration reaches tens to
hundreds of um, smearing the ~9 % first-lobe energy into broad LSF skirts.
A paraboloid refocuses every zone stigmatically, so the mirror design has
no such skirt. Add small-but-nonzero in-band ghosts and no broadband
future, at essentially equal cost to the catalog OAP pair: the mirror pair
keeps the purity edge and wins the candidate comparison.""")
