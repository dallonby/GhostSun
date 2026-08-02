# Stray light and line-core SNR budget (2026-08-02)

Tool: `stray.py` (deterministic order tracing against the mech.py solids,
TIS arithmetic, veiling budget). Companion: `fourier.py` (slit-fan
overshoot, already quantified) and `mech.py` (fan vignetting sites).

## Why this budget exists

The science target is dark structure inside a dark line core. The Ha core
sits at roughly 16% of continuum; filaments are contrast WITHIN that. A
diffuse pedestal of scattered in-band light fills the core first, so the
metric throughout is veiling V = pedestal / core signal, and the filament
contrast multiplier is 1/(1+V).

The single biggest lever is already in the design: the 656/10 nm
prefilter keeps the full solar spectrum out of the box entirely, so every
number below is about the surviving 10 nm band (mean level ~0.85 of
continuum). The current lens Sol'Ex has no prefilter and refractive
surfaces in the beam; its known purity problem is exactly this budget
running unmanaged.

## Source ranking (production geometry, numbers from stray.py)

| source | veiling V | note |
|---|---|---|
| mirror scatter at 10 nm RMS roughness | ~19% | the original RFQ spec; would cost ~16% of filament contrast on its own |
| mirror scatter at 2 nm RMS roughness | 0.8% | amended RFQ spec (2026-08-02); catalog Thorlabs/Edmund parts already meet this |
| zero order onto a bare aluminum/printed wall | 0.8% | 25% of the in-band light, specular off the grating |
| zero order into a matte black trap | 0.07% | one part, one glue joint |
| grating diffuse scatter (holographic) | ~0.08% | why holographic was chosen; ruled ghosts would add discrete false lines instead |
| slit-fan overshoot on structure | 2-6% of light, small V if baffled matte | sites mapped by fourier.py and mech.py fan tier |

Bottom line: with the 2 nm roughness spec, a matte zero-order trap, and
matte baffles at the fan-overshoot sites, the total veil is ~1-2% and
filament contrast is essentially untouched. Skip those three and the
budget blows out to ~20%+, which is visible-by-eye contrast loss over the
whole disk.

## Grating orders (stray.py landing map)

Only m=0 propagates besides the imaging m=+1 at 2400 l/mm / 656 nm
(higher orders are evanescent at these angles; m=-1 does not fit the
grating equation). Landing sites (body coordinates, bx along entry axis,
by across):

* production dev+16, Lg180: zero order leaves the grating toward the
  entry-side wall, first structure ~30 mm from the grating on the wall
  beyond it. Trap: matte black wedge behind/beside the grating turntable.
* budget dev+25, Lg85: same topology, wall hit ~91 mm out, ~120 mm from
  the sensor. Same trap concept.

The zero order is also the alignment lamp: make the trap removable or
hinged, because zero-order autocollimation sets slit/OAP1 despace.

## Mirror roughness (TIS at 656 nm)

TIS = (4 pi sigma / lambda)^2 per surface:

| sigma RMS | per mirror | two mirrors |
|---|---|---|
| 1 nm | 0.04% | 0.07% |
| 2 nm | 0.15% | 0.29% |
| 5 nm | 0.9% | 1.8% |
| 10 nm | 3.7% | 7.3% |

The RFQ has been amended from 10 nm to 2 nm RMS. When quotes come back,
this is the spec NOT to trade away; wavefront figure affects blur (we
have 3 um of quadrature room) but roughness attacks contrast directly.

## What is bounded, not computed

* Grating BSDF: 0.3% diffuse into the hemisphere is a typical holographic
  figure; ask the vendor for measured scatter or a witness scan.
* Multi-bounce wall paths: single-bounce arithmetic here. A matte black
  (~5% Lambertian) interior makes second bounces negligible (0.05^2).
  If we ever want the real number, this is the one place a physically
  based renderer (Mitsuba) is the right tool, since it is a radiometric
  question, not a wavefront one.
* Sensor-window ghost: the camera cover glass retro-reflects a defocused
  ghost back through OAP2. Standard AR glass ~0.5%/surface; the ghost is
  strongly defocused at the sensor, so V contribution is <0.1%. A slight
  camera tilt (the tilt flange exists) kills the symmetric return.

## Action list

1. RFQ roughness spec 2 nm RMS: DONE (RFQ_shanghai_optics.md).
2. Matte black zero-order trap behind the grating (removable for
   zero-order alignment): add to shg_body.scad when the geometry is
   re-frozen after the mech.py corridor findings (see MECH.md).
3. Matte baffle ring at the OAP1 aperture and grating-edge trap (fourier
   sites); the mech.py fan tier now prints exactly where the wings land
   for any candidate geometry.
4. Flock or black-anodize the tower face and OAP1-cell edge that the fan
   grazes in the corridor (MECH.md); these are the two scatter sources
   closest to the image path.
5. Ask the grating vendor for a measured scatter/ghost figure with the
   efficiency curve (RFQ already requests the curve).
