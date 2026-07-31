// GhostSun SHG — printable kinematic OAP mount (OpenSCAD twin of the
// Fusion 360 script; same parameter names and geometry).
//
// Kinematics: cone / vee / flat. Fixed pivot ball glued into the platform
// pocket rides the base CONE; adjuster screw at adjA lands in the VEE
// (pitch); adjuster screw at adjB lands on the FLAT (yaw). Two steel
// extension springs preload the plates together.
//
// Angular resolution: deg per turn = atan(screw_pitch / leverArm).
//   leverArm=120, M4x0.7:  0.33 deg/turn (20 arcmin) -> 1/20 turn = 1'
//   leverArm=120, 100TPI:  0.12 deg/turn ( 7 arcmin) -> 1/20 turn = 0.36'
// OAP yaw tolerance from raytrace: 1.5 arcmin. Lock after alignment.
//
// Print ASA/PC/annealed PETG, 6+ perimeters, plates flat on the bed.

/* [Kinematics] */
leverArm = 120;      // pivot-to-adjuster distance
gap = 10;            // assembled plate separation
ballD = 6;           // steel balls (pivot + screw tips)

/* [Plates] */
plateW = 150;
plateH = 150;
plateT = 8;
bossMargin = 22;     // corner margin / boss diameter

/* [Hardware] */
insertD = 5.6;       // M4 heat-set insert hole
insertL = 8;
springHoleD = 3.2;
mirrorBCD = 40;      // backing-plate bolt circle (measure your mirror!)
mirrorOffX = 0;      // mirror center offset along X on the platform.
                     // OAP1 platform: print with mirrorOffX = -34 (the
                     // slab+mount are shifted +34 so the diffracted beam
                     // clears the plate edge; mirror stays on-axis).
mirrorBoltD = 4.4;
mirrorBoltN = 3;
baseBoltD = 5.4;     // M5 to housing
baseCboreD = 10;
baseCboreZ = 4;

/* [Render] */
part = "both";       // both | base | platform

m = bossMargin;
pivot = [m, m];
adjA  = [m, m + leverArm];   // vee -> pitch
adjB  = [m + leverArm, m];   // flat -> yaw
springs = [[m + 0.35*leverArm, m + 0.35*leverArm],
           [m + 0.60*leverArm, m + 0.60*leverArm]];
centroid = [(pivot[0]+adjA[0]+adjB[0])/3 + mirrorOffX, (pivot[1]+adjA[1]+adjB[1])/3];

module plate() { cube([plateW, plateH, plateT]); }

module at(p) { translate([p[0], p[1], 0]) children(); }

module cone_seat() {            // 120-deg included cone, mouth 0.9*ballD
    mouth = 0.9 * ballD;
    depth = (mouth/2) / tan(60);
    translate([0, 0, plateT - depth + 0.01])
        cylinder(h = depth, d1 = 0, d2 = mouth, $fn = 64);
}

module vee_groove() {           // 90-deg vee aimed at the cone (along Y)
    w = 0.8 * ballD; l = bossMargin;
    translate([0, 0, plateT])
        rotate([0, 45, 0])      // square prism on its edge
            cube([w/sqrt(2), l, w/sqrt(2)], center = true);
}

module base() {
    difference() {
        plate();
        at(pivot) cone_seat();
        at(adjA) vee_groove();
        // flat at adjB: untouched surface
        for (s = springs) at(s) cylinder(h=3*plateT, d=springHoleD, center=true, $fn=32);
        for (c = [[plateW-m/2, plateH-m/2], [plateW-m/2, m/2], [m/2, plateH-m/2]]) {
            at(c) cylinder(h=3*plateT, d=baseBoltD, center=true, $fn=32);
            at(c) cylinder(h=baseCboreZ, d=baseCboreD, $fn=32);
        }
    }
}

module platform() {
    difference() {
        plate();
        // pivot ball press-fit pocket (glue), half-ball deep
        at(pivot) translate([0,0,plateT-ballD/2])
            cylinder(h=ballD/2+0.01, d=0.98*ballD, $fn=64);
        // adjuster heat-set insert holes (screws pass through to base side)
        for (a = [adjA, adjB]) {
            at(a) cylinder(h=3*plateT, d=4.4, center=true, $fn=32);
            at(a) translate([0,0,plateT-insertL])
                cylinder(h=insertL+0.01, d=insertD, $fn=32);
        }
        for (s = springs) at(s) cylinder(h=3*plateT, d=springHoleD, center=true, $fn=32);
        // mirror backing-plate bolt circle at support-triangle centroid
        for (k = [0 : mirrorBoltN-1])
            at([centroid[0] + mirrorBCD/2*cos(360*k/mirrorBoltN),
                centroid[1] + mirrorBCD/2*sin(360*k/mirrorBoltN)])
                cylinder(h=3*plateT, d=mirrorBoltD, center=true, $fn=32);
    }
}

if (part == "base" || part == "both") base();
if (part == "platform" || part == "both")
    translate([0, 0, part == "both" ? plateT + gap : 0])
        mirror([0, 0, part == "both" ? 1 : 0])
            translate([0, 0, -plateT]) platform();
