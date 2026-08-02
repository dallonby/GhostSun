# Fold-flat study: two candidate architectures (2026-08-02)

Tools: `fold_study.py` (constrained sweeps), `fold_tol.py` (alignment),
`thermal.py` (sunlight/aluminium case). raytrace.py now supports
`fold2/fold3/fold4` (in-plane flats) and `lift2/lift3` (two-flat
vertical periscopes); mech.py models their mounts and split beams.

Physics used throughout: a plane mirror adds ZERO aberration anywhere
in the system (verified to machine precision in the tracer); it costs
one silver reflection (~2.5%), one 2 nm-roughness scatter surface
(~0.15% veil), and one mount.

## Chirality rules (verified by exact trace)

* fold3 (one in-plane flat after the grating): mirrors the downstream
  space; the coma-cancelling camera sense flips to s2=+1.
* fold2 (one in-plane flat before the grating): flips s2 AND which dev
  sign is the anamorphic-compression branch.
* fold4 and two-mirror periscopes: neutral; everything keeps its
  unfolded meaning.

## Dead ends (why, quantitatively)

* fold3: the wedge between the incident and diffracted beams opens only
  Lg*sin(dev) at the grating exit (~20 mm at useful devs); a flat plus
  mount needs ~35 mm. No placement exists.
* fold2 in-plane: the compression branch re-creates the same congestion
  around OAP1 that it was meant to fix.
* lift3 (periscope after the grating): the bottom mirror sits in the
  same starved wedge as fold3.

## A. BUDGET winner: 45/45 pair + one fold4 flat

`config T_budget_ha (MPD124 + MPD144) + Ø25.4 silver flat 46 mm after
OAP2, folding +90 deg; dev 24, s2 -1, Lg 70, Lc 130.`

| metric | frozen v2 (MPD246 60 deg) | fold4 candidate |
|---|---|---|
| disk-edge blur (+-0.5 nm) | 12.9 um | **9.3 um** |
| slit-end blur | ~30 um | ~20 um |
| R (slit-limited) | 25.8k | **27.7k** (anam 0.552) |
| worst core clearance | +3.2 mm | **+4.4 mm** |
| worst fan clearance | +0.1 mm | **+1.7 mm** |
| optics cost | £550 (353+197) | **~£467** (197+197+~£73 PF10-03-P01) |
| surfaces | 3 | 4 (-2.5% light, +0.15% veil) |

The flat fits: fan footprint on it is 8.8 mm semi vs 11.4 mm CA. The
camera decouples from the telescope entirely (sensor behind the box's
far corner). Strictly better than the frozen v2 on every metric except
one extra reflection. RE-FREEZE DECISION PENDING: adopting this as v3
means regenerating the body (new camera tunnel direction, flat seat)
and buying MPD144 + PF10 instead of MPD246.

## B. PRODUCTION winner: two-floor periscope (lift2)

`config B_edmund_30s-45 + Ø50.8 silver periscope pair in the collimated
beam (s=65, h=80): slit, OAP1 and telescope on the ground floor; the
grating, rotor, OAP2 and camera on a deck 80 mm up. dev 17, s2 -1,
Lg 315, Lc 255.`

* Disk-edge blur 8.6 um, slit end 19 um (+-0.5 nm): full
  PERFECT_HA-class optics, now in a geometry that actually exists.
* Anamorphism 0.671, slit-limited R ~38k class (PERFECT_HA convention).
* Worst core-beam clearance +5.2 mm; two fan-edge grazes at -0.2 mm on
  mount corners: chamfered and flocked, a designed wing trim of the
  outermost null edge (same class as the budget tower graze).
* Structure abutments (periscope mount near the slit tower) are
  deliberate; parts may merge in the machined case.
* Needs: machined two-level aluminium case (~450 x 260 x 230), the
  50 mm grating on the upper deck, tuning arm flipped to the far side
  (arm_side 90), OAP mount walls at plate width with relieved corners.
* Periscope flats: Ø50.8 lambda/10 protected silver (PF20-03-P01 class,
  ~£120 each); fan semi-footprint 19.2 mm vs 22.9 mm CA.

## Alignment (fold_tol.py): the flats are free

Worst-field RMS at best focus, sensor refocus as the only compensator;
"for +3 um" = single-DOF magnitude that adds 3 um RMS:

| DOF | budget+fold4 | production periscope |
|---|---|---|
| OAP1 pitch | insensitive | 0.050 deg (3 arcmin) |
| OAP2 pitch | 0.027 deg (1.6') | 0.028 deg (1.7') |
| grating clock | 0.045 deg (2.7') | insensitive |
| grating in-plane yaw | insensitive | 0.060 deg (3.6') |
| slit despace | >1000 mm (refocus) | >1000 mm |
| every flat, both axes | **>1000 deg** | **>1000 deg** |

Flat tilts steer the image and change nothing else; steering is
absorbed by pointing and the grating tune. The instrument has the SAME
three arcminute-class critical adjustments it had with no flats: two
OAP pitches and one grating axis, each visible live on the spectrum.
Adding mirrors did not add alignment burden. Bench procedure stays:
set spacings to +-0.5 mm, autocollimate the zero order, tune the line,
touch two OAP screws while watching line width, done.

## Thermal (thermal.py, machined aluminium case in sunlight)

* Uniform soak: all-Al bench + Al OAPs is athermal (focus and angles
  invariant). The grating tune drifts 4.7 pm/K (BK7 substrate; 0.34 on
  fused silica): one knob touch per session, and a free thermometer.
* Flats on fused-silica in Al cells drift in pointing only, which costs
  nothing (see table above). Thermal motion and flats are a good match.
* Side-to-side gradients are the real budget: 19-33 K of sustained
  difference eats +3 um on the critical DOFs. Untreated dark boxes in
  sun run 5-15 K; white-painted or bare metal <3 K. Margin ~6x.
* Slit jaws absorb ~2.4 W -> ~3 K local, ~4 um despace vs 210 um
  tolerance: fine (jaws already tilted 10 deg; an ERF halves it).
* Rules: white/bare exterior, no steel standoffs in the train, G10
  thermal break at the telescope flange, 15-20 min soak, watch the
  line-position telemetry in GhostSun as the thermometer.

## Recommendation

Budget: adopt A as v3 when ordering (cheaper AND sharper than the
frozen v2); until then v2 remains the frozen buildable state.
Production: B is the first geometrically legal full-performance layout;
it needs the two-level case CAD and a freeze pass of its own before any
glass is ordered.
