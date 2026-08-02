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
// Plates are MIRROR-SIZED; fine resolution comes from 100 TPI adjusters,
// not lever length. deg/turn = atan(pitch / leverArm):
//   OAP1 plate 80 (lever 48):  100TPI 0.30 deg/turn (18'/turn)
//   OAP2 plate 105 (lever 73): 100TPI 0.20 deg/turn (12'/turn)
// vs 1.5 arcmin yaw tolerance: 1/12 - 1/8 turn per tolerance step; use
// lock nuts. M4x0.7 is NOT sufficient at these levers.
leverArm = 73;       // pivot-to-adjuster distance (= plateW - 2*bossMargin)
gap = 10;            // assembled plate separation
ballD = 6;           // steel balls (pivot + screw tips)

/* [Plates] */
plateW = 105;        // print OAP1 pair at 80, OAP2 pair at 105
plateH = 105;
plateT = 8;
bossMargin = 16;     // corner margin / boss diameter

/* [Hardware] */
insertD = 5.6;       // M4 heat-set insert hole
insertL = 8;
springHoleD = 3.2;
mirrorBCD = 40;      // backing-plate bolt circle (measure your mirror!)
mirrorOffX = 0;      // mirror center offset along X (normally 0)
mirrorBoltD = 4.4;
mirrorBoltN = 3;
baseBoltD = 5.4;     // M5 to housing
baseCboreD = 10;
baseCboreZ = 4;

/* [Swappability] */
label = "OAP1-AU";   // embossed on the platform edge (station + coating)
liftTabW = 22;       // finger tab on the platform's free corner
keyFence = true;     // corner fence on the base: platform seats one way only

/* [Render] */
part = "both";       // both | base | platform | base_cap

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
    // corner fence: the platform seats in one orientation only
    if (keyFence)
        for (f = [[plateW-3, -3, 3, plateH*0.4], [-3, plateH-3, plateW*0.4, 3]])
            translate([f[0], f[1], 0]) cube([f[2], f[3], plateT + gap*0.6]);
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
        union() {
            plate();
            // lift tab on the free corner: tool-free module removal
            translate([plateW - 2, plateH - liftTabW - 4, 0])
                cube([14, liftTabW, plateT]);
        }
        // grip grooves on the tab
        for (g = [0:2])
            translate([plateW + 4 + g*3, plateH - liftTabW - 5, plateT - 1.2])
                cube([1.4, liftTabW + 2, 2]);
        // embossed module label on the front edge (station + coating)
        translate([plateW*0.32, 1.2, plateT*0.25]) rotate([90,0,0])
            linear_extrude(1.4) text(label, size=plateT*0.5,
                font="Liberation Sans:style=Bold");
        // pivot ball press-fit pocket (glue), half-ball deep
        at(pivot) translate([0,0,plateT-ballD/2])
            cylinder(h=ballD/2+0.01, d=0.98*ballD, $fn=64);
        // adjuster heat-set insert holes (screws pass through to base side)
        for (a = [adjA, adjB]) {
            at(a) cylinder(h=3*plateT, d=4.4, center=true, $fn=32);
            at(a) translate([0,0,plateT-insertL])
                cylinder(h=insertL+0.01, d=insertD, $fn=32);
        }
        // OPEN spring hooks: keyhole slots run to the plate edge so the
        // springs unhook sideways without tools (fast module swap)
        for (s = springs) {
            at(s) cylinder(h=3*plateT, d=springHoleD, center=true, $fn=32);
            at(s) translate([-springHoleD/2, -plateH, -plateT])
                cube([springHoleD, plateH, 3*plateT]);
        }
        // mirror backing-plate bolt circle at support-triangle centroid
        for (k = [0 : mirrorBoltN-1])
            at([centroid[0] + mirrorBCD/2*cos(360*k/mirrorBoltN),
                centroid[1] + mirrorBCD/2*sin(360*k/mirrorBoltN)])
                cylinder(h=3*plateT, d=mirrorBoltD, center=true, $fn=32);
    }
}

// snap-on dust cap protecting the cone/vee/flat seats while a module
// is off the instrument (grit in the vee costs arcminutes)
module base_cap() {
    difference() {
        translate([-2, -2, 0]) cube([plateW*0.75, plateH*0.75, 3]);
        translate([2, 2, -1]) cube([plateW*0.75-8, plateH*0.75-8, 3]);
    }
    for (p = [pivot, adjA, adjB])
        at(p) cylinder(h=6, d=ballD*2.2, $fn=48);
}

if (part == "base_cap") base_cap();
if (part == "base" || part == "both") base();
if (part == "platform" || part == "both")
    translate([0, 0, part == "both" ? plateT + gap : 0])
        mirror([0, 0, part == "both" ? 1 : 0])
            translate([0, 0, -plateT]) platform();
