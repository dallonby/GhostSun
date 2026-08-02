# Mechanical interference findings (mech.py, 2026-08-02)

`mech.py` makes clearance a first-class design constraint. It builds
mechanical envelopes (mirror cell + mount slab, grating rotor + tuning
arm, camera front stack + barrel, slit tower, snout, telescope OTA) from
the same raytrace.Design object the optics use, and models each beam leg
as a solid at two tiers:

* core: the geometric f/6.9 cone (~94% of the energy). Interference is a
  hard FAIL.
* fan: out to the first Fraunhofer nulls of the 7 um slit (the wings,
  a few percent). Interference means vignetting plus a scatter source at
  that spot (VIGNETTE), which stray-light budgeting then owns.

Beam widths are clamped by every aperture the light passes (collimator
CA, grating projected ruled width, camera CA). `body_export.py` now runs
`mech.assert_clear` instead of the old single-plane slab guard, and
`mech.ok()` is cheap enough to sit inside dev/Lg/Lc sweeps.

Run `python3 mech.py` for the CHOSEN table.

## Headline finding: the beam3 corridor

In this folded architecture the diffracted beam (grating -> OAP2,
"beam3") must thread the corridor between the slit tower and the OAP1
module on its way across the box. Every geometry on the books fails it
once the beam is given its real width; every earlier "verified" clearance
was chief-ray arithmetic:

| geometry | core verdict | fan verdict |
|---|---|---|
| budget CHOSEN (T_budget dev+25 Lg85 Lc175) | tower -7.8, OAP1 mount -2.4 | tower -10.0 |
| PERFECT_HA (dev+16 Lg180 Lc290) | tower -10.6, snout -1.6 | tower -19.8, snout -10.7 |
| pre-expert production (dev+20 Lg117 Lc240) | OAP1 mount -11.6, tower -1.9 | OAP1 mount -20.3 |

In PERFECT_HA the beam3 chief passes 9.2 mm from the slit CENTER with a
core radius of 8.6 mm: 0.6 mm of air. No slit holder fits in 0.6 mm.

The budget CHOSEN additionally has the known camera-vs-telescope
interference family (worst -44.5 mm), pending the MPD246 60-degree
camera fix.

## Constrained sweeps: no drop-in fix

A full dev 16-34 / Lg 105-200 / Lc 240-290 sweep of the production config
with `mech.ok()` as the accept test finds ZERO feasible geometries, even
with the OAP1 mount plate narrowed from 80 to 68 mm. The corridor needs
roughly 2 x (tower half-width + core radius + margins) ~ 40 mm and has
~35 mm at best. Negative dev moves beam3 away from the slit entirely but
was checked and rejected: anamorphism flips to 1.45-1.87x stretch
(halving spectral resolution), coma cancellation breaks (blur 50-79 um at
the slit end), and the camera lands on the OAP1 mount (-16 to -45 mm).

Least-bad positive-dev point: dev 23.5, Lg 105, Lc 290 with a 74 mm OAP1
slab: core is -1.4 mm into the (assumed 10 mm half-width) slit tower and
+2.0 mm off the OAP1 cell; the fan shaves both sides regardless.

## Paths to feasibility (decision needed)

1. Asymmetric slit tower: put all holder structure (slide, clamp) on the
   side away from beam3, leaving a ~3 mm blade wall on the beam3 side.
   Buys ~7 mm; combined with the 74 mm OAP1 slab this makes the least-bad
   point feasible for the CORE (~+5 mm) with the FAN still grazing the
   tower blade and OAP1 cell edge: both faces flocked/black, budgeted in
   STRAYLIGHT.md. Cheapest path, stays 2D, costs a tower redesign and a
   deliberate few-percent wing clip near the slit.
2. Shorter collimator: beam3 core radius scales linearly with rfl1
   (budget 50.8 mm collimator gives 6.8 mm core vs 8.6 mm at 81.8 mm).
   Costs grating fill and therefore the resolution ceiling.
3. Out-of-plane lift: tilt the grating grooves so beam3 passes above the
   tower. Clearing +27 mm at the crossing needs ~14 degrees of conical
   angle: too much (line curvature, astigmatism). Rejected at this size.
4. Re-open the fold-angle family (th1/th2 other than 30/45) with mech.ok
   inside the sweep from day one. Larger redesign, unexplored.

Recommendation: path 1. It is a mechanical change only (tower + OAP1
plate), keeps PERFECT_HA-class optics, and turns the fan clip into a
designed, flocked baffle edge instead of an accident.

## Re-freeze record (2026-08-02)

The budget prototype was re-frozen with path 1 plus the camera swap:

* config T_budget_ha_v2: MPD124 45 deg collimator + MPD246 60 deg camera
  (the 45/45 pair's exit converges 25 deg onto the OTA; no placement
  clears it. The 60 deg fold exits 6 deg off parallel and clears the
  two-segment FSQ-85 envelope: drawtube r35 to -190, tube r50 beyond).
* CHOSEN = dev 21.0, s2 -1, Lg 80, Lc 150 from the constrained sweep
  (both fold senses swept; s2=+1 feasible but always blurrier). Worst
  core clearance +3.2 mm, fan +0.1 mm (grazes the tower blade: flocked,
  budgeted in STRAYLIGHT.md). Edge blur 12.9 um, center 2.6 um.
* Slit tower slimmed to the CAD-true asymmetric footprint: bx -10..+4
  with the +4 blade wall, across-beam width +-10 (was +-17).
* OAP1 KM plate 48 mm (slab 54, km1Hole 16): lever 16 mm -> 0.9 deg/turn
  on 100 TPI, ~7x finer than the arcminute tolerances need.
* NEW CATCH while re-freezing: the stray-light vane at by=25 spanned the
  full box and blocked beam3 in the rev-2 committed body. body_export.py
  now computes vaneX0 from the traced fan crossing (+4 mm margin), emits
  it into body_geom.scad, and the vane is a mech.py solid in the guard.
* Production (B_edmund) remains unfrozen: same recipe applies but the
  corridor is tighter (bigger fan); re-run the constrained sweep with
  production dims before ordering Edmund glass.

## Checker limitations

* Envelope dimensions default to shg_body.scad values (86 mm slabs,
  10 mm tower half-width); override via dims= for other hardware.
  Conclusions above were checked against plausible slimmer envelopes.
* Solid-solid distances use surface sampling (8 mm grid) or capsule
  analytics: accurate to ~1 mm, fine for 5 mm margins.
* The telescope is a 100 mm cylinder; focuser knobs are not modeled.
* Beams are straight-chief swept spheres; field spread along the slit
  (vertical) is not added because every solid spans the beam vertically.
