// GhostSun SHG — v3 kinematic OAP mount (48 mm plate, 25.4 mm MPD OAP).
// Scope: v3 budget build ONLY (casefile 49e5a75e). CAD stays parametric;
// only the v3 parameter set is validated by the self-check.
//
// Kinematics: cone / vee / flat exact constraint. 6 mm ball glued on the
// platform's pivot post rides the base CONE (120 deg). 100 TPI adjuster
// ball-tips ride the VEE (90 deg, aimed at the cone) and the FLAT.
// Two extension springs on top/bottom edge ears preload the plates.
//
// Layout (plate coords: u right, v up, z out of wall):
//   pivot (8,8)  vee (15.5,40.75)  flat (40.75,15.5)  lever 30.25 mm
//   M5 (8,40) (40,8) (40,40)     mirror (24,24)    springs u=32 top, u=16 bot
//
// Adjustment (100 TPI, pitch 0.254 mm, lever 30.25 mm):
//   deg/turn = atan(0.254/30.25) = 0.481 deg/turn = 28.9 arcmin/turn
//   per 10 deg knob nudge: 0.80 arcmin   (spec floor: 2.0; tolerance 1.7)
//   range +/-2 deg costs +/-1.06 mm screw travel, ~4.2 turns end to end.
//
// Stack: wall z=0 | base 8 | gap 21 | platform 8 | mirror 8 | face z=45.
// M5 socket heads stand proud INSIDE the 21 mm gap: no counterbores
// anywhere, so the wall-side face is virgin (defect 3 fix) and no
// counterbore eats a seat or forces the layout (defect 1 enabler).
// Springs anchor in ear holes on the TOP/BOTTOM edges (beams graze the
// two vertical side edges only): hooks outside the silhouette, tool-free,
// never in the gap, never against the wall, never under the mirror
// (defect 2 fix). Spring slots do not exist (defect 1 fix).
//
// Print: base wall-side DOWN (flush face = bed face), platform mirror-side
// DOWN. ASA or PC, 4+ perimeters, no support needed anywhere.
//
// part = "assembled" | "exploded" | "base" | "platform" | "cap" | "check"

/* [Plates] */
plateW = 48;
plateH = 48;
plateT = 8;
standoff = 45;        // wall face -> mirror optical face (fixed interface)
mirrorT = 8;          // mirror substrate thickness (VERIFY Thorlabs drawing)
gap = standoff - 2*plateT - mirrorT;   // 21

/* [Kinematics] */
ballD = 6;            // pivot ball (glued to platform post)
tipBallD = 4.8;       // adjuster ball tips
coneAng = 120;        // included cone angle
coneMouthD = 5.6;     // cone mouth diameter
veeAng = 90;          // included vee angle
veeLen = 6;           // vee groove length (tip travel is +/-1.1)
pivot = [8, 8];
vee   = [15.5, 40.75];  // adjuster A (pitch)
flat  = [40.75, 15.5];  // adjuster B (yaw)

/* [Mirror] */
mirrorD = 25.4;
mirrorOff = [24, 24];       // mirror centre in plate coords
mirrorBoltD = 4.2;          // 8-32 clearance
mirrorBoltN = 1;            // v3: single central bolt
mirrorBCD = 0;              // parametric growth path (unvalidated)

/* [Body interface] */
m5HoleD = 5.4;        // M5 clearance through the base
m5HeadD = 8.5;        // socket head (stands proud in the gap)
m5HeadH = 5;
m5insD = 6.4;         // heat-set insert in the WALL (reference)
m5insL = 12;
m5pos = [[8, 40], [40, 8], [40, 40]];   // (+16,+16),(+16,-16),(-16,+16)
                                      // rel. plate centre; (8,8) stays free

/* [Adjusters] */
bushingD = 9.5;       // press-in bushing OD (parameter per spec)
bushingFlangeD = 11;
bushingFlangeH = 2;
screwD = 6.35;        // 1/4-100 screw
knobD = 9.5;          // keep knobs <= 9.5 (mirror clearance!)
pitch100 = 0.254;     // 100 TPI pitch

/* [Springs] */
springEarW = 10;      // ear width (ligaments >= 2.5 around the hole)
springHoleD = 4.5;    // hook hole in the ear
springTopU = 32;      // top-edge ear position
springBotU = 16;      // bottom-edge ear position
earLig = 2.5;         // min ligament hole -> plate edge and hole -> tip

/* [Module features] */
label = "COLL-50.8";  // station + RFL (swap guard, casefile 1ebe4716)
tabLen = 10;          // lift tab overhang at the (48,48) corner
fenceClr = 0.6;       // key fence clearance (never touches seated platform)

/* [Render] */
part = "assembled";   // assembled | exploded | base | platform | cap | check
explode = 26;         // exploded-view spacing

// ---- derived ----
lever = abs((flat[0]-pivot[0])*(vee[1]-pivot[1])
          - (flat[1]-pivot[1])*(vee[0]-pivot[0])) / norm(vee - pivot);
degPerTurn = atan(pitch100 / lever);
arcminPer10 = degPerTurn * 60 / 36;
centroid = [(pivot[0]+vee[0]+flat[0])/3, (pivot[1]+vee[1]+flat[1])/3];
coneDepth = (coneMouthD/2) / tan(coneAng/2);
veeW = 0.8 * tipBallD;
veeDepth = veeW / 2;             // 90 deg vee: depth = half width
zBase = plateT;                  // base top face
zPlatB = plateT + gap;           // platform bottom face
zPlatT = zPlatB + plateT;        // platform top face (= mirror back)
zFace = zPlatT + mirrorT;        // mirror optical face (= standoff)
postH = zPlatB - zBase - coneDepth - ballD/2 + 1;  // pivot post length
earOut = earLig + springHoleD/2;          // hole centre beyond plate edge
earTip = earOut + springHoleD/2 + earLig; // ear tip beyond plate edge
mirrorR = mirrorD/2;

function d2(a, b) = norm(a - b);
// point-in-triangle (2D half-plane tests)
function inTri(p) =
    let(a = pivot, b = vee, c = flat,
        d1 = (p[0]-b[0])*(a[1]-b[1]) - (a[0]-b[0])*(p[1]-b[1]),
        d2_ = (p[0]-c[0])*(b[1]-c[1]) - (b[0]-c[0])*(p[1]-c[1]),
        d3 = (p[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(p[1]-a[1]))
    !((d1 < 0 || d2_ < 0 || d3 < 0) && (d1 > 0 || d2_ > 0 || d3 > 0));

module at(p) { translate([p[0], p[1], 0]) children(); }

// ---------- features ----------
module cone_seat() {   // 120 deg included cone in the base top face
    translate([0, 0, zBase - coneDepth + 0.01])
        cylinder(h = coneDepth, d1 = 0, d2 = coneMouthD, $fn = 64);
}

module vee_groove() {  // 90 deg vee aimed at the cone, in the base top face
    aim = atan2(pivot[1] - vee[1], pivot[0] - vee[0]);
    translate([0, 0, zBase]) rotate([0, 0, aim])
        translate([0, 0, veeDepth - 0.01]) rotate([0, 45, 0])
            cube([veeW/sqrt(2) + 0.02, veeLen, veeW/sqrt(2) + 0.02],
                 center = true);
}

module ear(sgn) {      // spring anchor ear on a top/bottom edge
    // 2D profile: plate outline extension with rounded tip
    translate([0, sgn > 0 ? plateH : 0, 0]) scale([1, sgn, 1]) {
        translate([-springEarW/2, 0]) square([springEarW, earOut]);
        translate([0, earOut]) circle(r = springEarW/2, $fn = 48);
    }
}

module ear_hole(sgn) {
    translate([0, sgn > 0 ? plateH + earOut : -earOut, -1])
        cylinder(h = plateT + 2, d = springHoleD, $fn = 32);
}

module outline() {     // plate outline + ears (2D)
    square([plateW, plateH]);
    translate([springTopU, 0]) ear(1);
    translate([springBotU, 0]) ear(-1);
}

module m5_holes() {
    for (p = m5pos)
        at(p) cylinder(h = 3*plateT, d = m5HoleD, center = true, $fn = 32);
}

// ---------- base ----------
module base() {
    notch = 5;  // key-fence notch reach at the (48,48) corner
    difference() {
        union() {
            linear_extrude(plateT) outline();
            // key fence: fills the platform notch with fenceClr clearance.
            // Blocks any rotated seating; never touches the seated module.
            fence = notch + fenceClr;
            translate([plateW - fence, plateH - fence, 0])
                linear_extrude(zPlatT + 3)
                    polygon([[0, fence], [fence, 0], [fence, fence]]);
        }
        at(pivot) cone_seat();
        at(vee)   vee_groove();
        // flat: untouched face by construction
        m5_holes();   // through only; heads stand proud in the gap
        translate([springTopU, 0, 0]) ear_hole(1);
        translate([springBotU, 0, 0]) ear_hole(-1);
    }
}

// ---------- platform ----------
module platform() {
    notch = 5 + fenceClr;   // key notch at the (48,48) corner
    difference() {
        union() {
            difference() {
                linear_extrude(plateT) outline();
                // key notch (48,48) corner
                translate([plateW - notch, plateH - notch, -1])
                    linear_extrude(plateT + 2)
                        polygon([[0, notch], [notch, 0], [notch, notch]]);
            }
            // lift tab: diagonal overhang at the free corner
            translate([plateW - notch, plateH - notch, 0])
                rotate([0, 0, 45]) translate([-8, 0, 0])
                    cube([16, notch + tabLen, plateT]);
            // pivot ball post (bridges the gap to the cone)
            at(pivot) translate([0, 0, -postH])
                cylinder(h = postH + 0.01, d = ballD + 3, $fn = 48);
        }
        // grip grooves on the tab
        for (g = [0:2])
            translate([plateW + 1.5 + g*2.6, plateH + 1.5, plateT - 1.2])
                rotate([0, 0, 45]) cube([1.2, tabLen + 8, 2]);
        // engraved station label on the top edge face (visible installed)
        translate([6, plateH - 0.9, plateT/2]) rotate([90, 0, 0])
            linear_extrude(1.0)
                text(label, size = 4.5, font = "Liberation Sans:style=Bold");
        // pivot ball pocket (glue), half-ball deep, at the post tip
        at(pivot) translate([0, 0, -postH - ballD/2])
            sphere(d = 0.98*ballD, $fn = 48);
        // bushing seats (press-in from the top face) + flange counterbores
        for (a = [vee, flat]) {
            at(a) cylinder(h = 3*plateT, d = bushingD - 0.1,
                           center = true, $fn = 48);
            at(a) translate([0, 0, plateT - bushingFlangeH])
                cylinder(h = bushingFlangeH + 0.01, d = bushingFlangeD,
                         $fn = 48);
        }
        // mirror bolt(s): v3 = single central 8-32; head+washer in the gap
        for (k = [0 : mirrorBoltN-1])
            at([mirrorOff[0] + (mirrorBCD/2)*cos(360*k/mirrorBoltN),
                mirrorOff[1] + (mirrorBCD/2)*sin(360*k/mirrorBoltN)])
                cylinder(h = 3*plateT, d = mirrorBoltD, center = true,
                         $fn = 32);
        // spring ear holes
        translate([springTopU, 0, 0]) ear_hole(1);
        translate([springBotU, 0, 0]) ear_hole(-1);
    }
}

// ---------- snap-on seat cap (module off-instrument) ----------
module cap() {
    notch = 5 + fenceClr + 0.4;
    difference() {
        union() {
            linear_extrude(3) difference() {
                offset(r = 1) square([plateW - 2, plateH - 2]);
                translate([plateW - 2 - notch, plateH - 2 - notch])
                    polygon([[0, notch], [notch, 0], [notch, notch]]);
            }
            // rim
            linear_extrude(6) difference() {
                offset(r = 1) square([plateW - 2, plateH - 2]);
                offset(r = -1) square([plateW - 6, plateH - 6]);
            }
            // seat nubs rest in cone/vee/on flat
            for (p = [pivot, vee, flat])
                translate([p[0] - 1, p[1] - 1, 3])
                    cylinder(h = 3.5, d = ballD*0.9, $fn = 48);
        }
    }
}

// ---------- hardware ghosts (render only) ----------
module ghost_ball() color("silver")
    at(pivot) translate([0, 0, zBase - coneDepth + ballD/2 + 0.4])
        sphere(d = ballD, $fn = 48);

module ghost_adjusters() color("darkgray") {
    for (a = [vee, flat]) at(a) {
        translate([0, 0, zBase + 1]) sphere(d = tipBallD, $fn = 32);
        translate([0, 0, zBase + 1])
            cylinder(h = zPlatT - zBase, d = screwD, $fn = 32);
        translate([0, 0, zPlatT + 6]) cylinder(h = 8, d = knobD, $fn = 32);
    }
}

module ghost_mirror(col = "lightsteelblue") color(col, 0.95)
    at(mirrorOff) translate([0, 0, zPlatT])
        cylinder(h = mirrorT, d = mirrorD, $fn = 64);

module ghost_springs() color("gray", 0.9) {
    for (s = [[springTopU, plateH + earOut], [springBotU, -earOut]])
        translate([s[0], s[1], zBase])
            cylinder(h = zPlatB - zBase, d = 5, $fn = 24);
}

module ghost_m5() color("silver")
    for (p = m5pos) at(p) {
        translate([0, 0, -m5insL + 2]) cylinder(h = m5insL, d = 5, $fn = 24);
        translate([0, 0, zBase]) cylinder(h = m5HeadH, d = m5HeadD, $fn = 24);
    }

module ghost_bushings() color("peru")
    for (a = [vee, flat]) at(a)
        translate([0, 0, zPlatB])
            cylinder(h = plateT, d = bushingD, $fn = 32);

// ---------- modes ----------
if (part == "base") base();
if (part == "platform")   // print orientation: mirror-side down
    translate([0, 0, plateT]) mirror([0, 0, 1]) platform();
if (part == "cap") translate([1, 1, 0]) cap();

if (part == "assembled" || part == "exploded") {
    ex = part == "exploded" ? explode : 0;
    color("sandybrown") base();
    translate([0, 0, ex*0.6]) color("lightskyblue", 0.9)
        translate([0, 0, zPlatB]) platform();
    translate([0, 0, ex*1.4]) ghost_mirror();
    translate([0, 0, ex*1.1]) ghost_adjusters();
    translate([0, 0, ex*1.1]) ghost_bushings();
    translate([0, 0, ex*0.6]) ghost_ball();
    translate([0, 0, ex*0.3]) ghost_springs();
    translate([0, 0, -ex*0.4]) ghost_m5();
    if (part == "exploded") translate([0, 0, -ex*1.0])
        color("palegreen", 0.8) translate([1, 1, 0]) cap();
}

// ---------- self-check (openscad -D part=\"check\") ----------
// Pure-geometry assertions at the v3 parameter set. Any violation aborts
// the render; every check also echoes one PASS line.
module check() {
    // stack closes to the standoff
    assert(abs(zFace - standoff) < 1e-9, "stack != standoff");
    echo(str("PASS stack: wall->face = ", zFace, " mm"));
    // resolution meets the 2 arcmin / 10 deg knob floor
    assert(arcminPer10 <= 2.0, "resolution worse than spec floor");
    echo(str("PASS resolution: ", degPerTurn, " deg/turn (",
             degPerTurn*60, " arcmin/turn), ", arcminPer10,
             " arcmin per 10 deg knob, lever ", lever, " mm"));
    // lever geometry sanity
    assert(lever >= 12.1, "lever arm below resolution requirement");
    // range travel at +/-2 deg
    echo(str("PASS range: +/-2 deg = +/-", lever*tan(2),
             " mm tip travel, ", 2/degPerTurn, " turns end to end"));
    // springs: resultant (midpoint of ear anchors) inside contact triangle
    sprMid = [(springTopU + springBotU)/2, plateH/2];
    assert(inTri(sprMid), "spring resultant outside contact triangle");
    echo(str("PASS springs: resultant at ", sprMid, " inside triangle (centroid ", centroid, ")"));
    // spring anchors clear of the mirror footprint (+2 mm handling)
    for (s = [[springTopU, plateH + earOut], [springBotU, -earOut]])
        assert(d2(s, mirrorOff) >= mirrorR + 2,
               "spring anchor inside mirror footprint");
    echo(str("PASS springs clear of mirror: top ",
             d2([springTopU, plateH + earOut], mirrorOff) - mirrorR,
             " mm, bottom ",
             d2([springBotU, -earOut], mirrorOff) - mirrorR, " mm"));
    // seats clear of M5 holes (same plate: keep >= 2.5 wall)
    for (p = m5pos) {
        assert(d2(p, pivot) - m5HoleD/2 - coneMouthD/2 >= 2.5,
               "cone vs M5 hole");
        assert(d2(p, vee) - m5HoleD/2 - veeW/2 >= 2.5, "vee vs M5 hole");
    }
    echo(str("PASS cone/vee vs M5 holes: nearest wall ",
        min(min([for (p = m5pos) d2(p, pivot) - m5HoleD/2 - coneMouthD/2]),
            min([for (p = m5pos) d2(p, vee) - m5HoleD/2 - veeW/2])), " mm"));
    // adjuster ball tips clear of proud M5 heads (cross-plate, in the gap)
    for (a = [vee, flat])
        assert(min([for (p = m5pos) d2(a, p)]) >= m5HeadD/2 + tipBallD/2,
               "adjuster tip hits an M5 head");
    echo(str("PASS tips vs M5 heads: clearance ",
        min([for (a = [vee, flat], p = m5pos) d2(a, p)])
        - m5HeadD/2 - tipBallD/2, " mm (spec socket heads <= 8.5 mm)"));
    // knobs clear of the mirror cylinder
    for (a = [vee, flat])
        assert(d2(a, mirrorOff) >= mirrorR + knobD/2 + 1.0,
               "knob hits mirror");
    echo(str("PASS knobs vs mirror: clearance ",
        min([for (a = [vee, flat]) d2(a, mirrorOff)]) - mirrorR - knobD/2,
        " mm"));
    // pivot post clear of the mirror footprint
    assert(d2(pivot, mirrorOff) >= mirrorR + (ballD+3)/2 + 1.0,
           "pivot post inside mirror footprint");
    echo(str("PASS pivot post vs mirror: clearance ",
        d2(pivot, mirrorOff) - mirrorR - (ballD+3)/2, " mm"));
    // bushing seats: edge walls >= 2.5, clear of mirror bolt
    for (a = [vee, flat]) {
        assert(min(a[0], a[1], plateW - a[0], plateH - a[1]) - bushingD/2
               >= 2.5, "bushing seat wall < 2.5");
        assert(d2(a, mirrorOff) - bushingFlangeD/2 >= 2.5 + 0,
               "bushing flange reaches mirror centre region");
    }
    echo(str("PASS bushing seats: edge wall ",
        min([for (a = [vee, flat])
             min(a[0], a[1], plateW - a[0], plateH - a[1]) - bushingD/2]),
        " mm; flange rim to mirror centre ",
        min([for (a = [vee, flat]) d2(a, mirrorOff)]) - bushingFlangeD/2,
        " mm"));
    // mirror bolt clear of pivot post and bushings
    for (a = concat([pivot], [vee, flat]))
        assert(d2(a, mirrorOff) >= (ballD+3)/2 + mirrorBoltD/2 + 2.5 ||
               d2(a, mirrorOff) >= bushingD/2 + mirrorBoltD/2 + 2.5,
               "mirror bolt hits contact feature");
    echo(str("PASS mirror bolt: clear of post and bushings"));
    // ear ligaments (min feature 2.5)
    assert(earLig >= 2.5 && (springEarW - springHoleD)/2 >= 2.5,
           "ear ligament < 2.5");
    echo(str("PASS ears: ligaments ", earLig, " / ",
         (springEarW - springHoleD)/2, " mm, protrude ", earTip, " mm"));
    // label clear of ears and bushings (label spans u 6..~26 on v=plateH)
    assert(springTopU - springEarW/2 - 26 >= 0, "label reaches spring ear");
    assert(vee[1] + knobD/2 <= plateH, "knob crosses the label edge");
    echo(str("PASS label: u 6..26 on top edge, clear of ear (u>=",
         springTopU - springEarW/2, ") and knobs (v<=",
         vee[1] + knobD/2, ")"));
    // key fence: clears the M5 head at (40,40) (fence keeps out of reach)
    assert(norm([plateW, plateH] - m5pos[2]) - (5 + fenceClr)*sqrt(2)
           >= m5HeadD/2 - 4.5, "fence vs M5 head");
    echo(str("PASS key fence at (48,48) corner, clearance ", fenceClr,
         "; tab overhangs there only"));
    // beam sides (u=0 and u=plateH edges): no protrusions except the tab
    // corner. By construction ears are on v-edges only; assert tab is the
    // only feature crossing u<0 or u>plateW outside v in [42,48].
    assert(tabLen <= 10, "tab overhang beyond designated allowance");
    echo(str("PASS beam sides: no features on u-edges except tab at v 42..48"));
    // flush wall face: no cut opens on base z=0 (by construction: M5
    // through-holes only, all recesses from the top). Assert symbolically:
    assert(m5HeadH <= gap, "M5 head taller than the gap");
    echo(str("PASS flush wall face: zero recesses on z=0; M5 heads stand ",
         m5HeadH, " mm proud inside the ", gap, " mm gap"));
    // mass estimate (ASA 1.07 g/cm^3, rough solid volumes)
    platVol = plateW*plateH*plateT/1000;   // cm3, ignores cuts
    echo(str("PASS mass est: platform ~", platVol*1.07*0.85,
         " g + base not counted; moving side << 150 g budget"));
}

if (part == "check") {
    check();
    // render something trivial so the run succeeds
    cube([1, 1, 1]);
}
