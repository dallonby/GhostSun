#!/usr/bin/env python3
"""Regenerate ghostsun_shg_all.scad from the split sources.
Run body_export.py first if the optics changed."""
geom = open("body_geom.scad").read()
body = open("shg_body.scad").read().replace("include <body_geom.scad>;\n", "")
km = open("mounts/kinematic_mount.scad").read()
km = km.replace('part = "both";       // both | base | platform\n', "")
km = km.replace('/* [Render] */\n', "")
km = km.replace('if (part == "base" || part == "both") base();',
                'if (part == "km_base") km_base();')
km = km.replace('''if (part == "platform" || part == "both")
    translate([0, 0, part == "both" ? plateT + gap : 0])
        mirror([0, 0, part == "both" ? 1 : 0])
            translate([0, 0, -plateT]) platform();''',
                'if (part == "km_platform") km_platform();')
km = km.replace("module base() {", "module km_base() {")
km = km.replace("module platform() {", "module km_platform() {")
for a, b in [("module plate()", "module km_plate()"), ("plate();", "km_plate();"),
             ("module at(", "module km_at("), ("at(pivot)", "km_at(pivot)"),
             ("at(adjA)", "km_at(adjA)"), ("at(a)", "km_at(a)"),
             ("at(s)", "km_at(s)"), ("at(c)", "km_at(c)"),
             ("at([centroid", "km_at([centroid")]:
    km = km.replace(a, b)
header = open("ghostsun_shg_all.scad").read().split("// ---- raytrace geometry")[0]
out = (header + "// ---- raytrace geometry (auto-generated) ----\n" + geom +
       "\n// ---- instrument body ----\n" + body +
       "\n// ---- printable kinematic OAP mount ----\n" + km)
open("ghostsun_shg_all.scad", "w").write(out)
print(f"ghostsun_shg_all.scad regenerated ({len(out)} bytes)")
