#!/usr/bin/env python3
"""Raytrace-driven render of the one-element concave SHG: actual traced
rays plus the true spherical-cap grating, emitted as OpenSCAD."""
import math
from concave import OneElementSHG, conjugate_K
from raytrace import add, sub, mul, dot, cross, norm

P = dict(Rc=308.75, LA=302.75, alpha=57.88, LB=260.0, LBi=260.0,
         dy=0.0, dz=0.0, tilt=-12.81)

def factory(sys):
    a = math.radians(P["alpha"])
    b = math.asin(1.575 - math.sin(a))
    dirD = add(mul(sys.n0, math.cos(b)), mul(sys.e0, math.sin(b)))
    D = add(sys.pole, mul(dirD, P["LBi"]))
    D = (D[0], D[1] + P["dy"], D[2] + P["dz"])
    return conjugate_K((0.0, 0.0, 0.0), D, 656.28e-6)

d = OneElementSHG(P["Rc"], P["LA"], P["alpha"], P["LB"], factory,
                  sensor_tilt=P["tilt"])

out = ["$fn = 64;",
       "module ray(a, b, r, col, al) { color(col, al) hull() {",
       "  translate(a) sphere(r); translate(b) sphere(r); } }"]

def V(p):  # world (x=slit axis, y, z) -> scad (X=z, Y=y, Z=x)
    return f"[{p[2]:.2f}, {p[1]:.2f}, {p[0]:.2f}]"

# traced rays: 3 fields x 3 wavelengths, chief + ring
COLS = {655.78: "orange", 656.28: "red", 656.78: "crimson"}
for xf in (-3.5, 0.0, 3.5):
    for lam, col in COLS.items():
        lam_mm = lam * 1e-6
        na = 1.0 / (2 * 6.9)
        dirs = [(0.0, 0.0)] + [(na * math.cos(a), na * math.sin(a))
                               for a in [k * math.pi / 3 for k in range(6)]]
        for (ax, ay) in dirs:
            dd = norm((ax, ay, 1.0))
            r = d.gr.diffract((xf, 0.0, 0.0), dd, lam_mm)
            if r is None:
                continue
            hit, dout = r
            dn = dot(dout, d.sens_n)
            t = dot(sub(d.F, hit), d.sens_n) / dn
            send = add(hit, mul(dout, t))
            rr = 0.12 if (ax, ay) != (0.0, 0.0) else 0.25
            out.append(f"ray({V((xf,0.0,0.0))}, {V(hit)}, {rr}, \"{col}\", 0.75);")
            out.append(f"ray({V(hit)}, {V(send)}, {rr}, \"{col}\", 0.75);")

# grating: true spherical cap, 50 x 40 aperture about the pole
C = d.gr.center
pole = d.pole
n0 = d.n0
e0 = d.e0
x_ax = (1.0, 0.0, 0.0)
# scad: sphere shell at C, intersect an oriented box at the pole
def ang_of(v):
    return math.degrees(math.atan2(v[1], v[2]))
out.append(f"""
color("mediumblue", 0.95) intersection() {{
    translate({V(C)}) difference() {{
        sphere(r = {P['Rc']:.1f}, $fn = 200);
        sphere(r = {P['Rc'] - 8:.1f}, $fn = 200);
    }}
    translate({V(pole)}) rotate([{-ang_of(n0):.2f}, 0, 0])
        rotate([0, 90, 0]) cube([48, 72, 56], center = true);
}}""")
# slit plate (green) and sensor (dark) with tilt
out.append(f"color(\"green\") translate({V((0,0,0))}) "
           f"cube([2, 14, 24], center = true);")
sy = d.sens_y
sang = ang_of(mul(d.sens_n, -1.0))
out.append(f"color(\"dimgray\") translate({V(d.F)}) "
           f"rotate([{-sang:.2f}, 0, 0]) rotate([0, 90, 0]) "
           f"cube([32, 3, 26], center = true);")
open("concave_layout.scad", "w").write("\n".join(out) + "\n")
print("concave_layout.scad written; pole", [round(v,1) for v in pole],
      "F", [round(v,1) for v in d.F])
