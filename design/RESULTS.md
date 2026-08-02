# All-reflective SHG — design results (2026-07-31, rev 2)

**MECH NOTE (2026-08-02):** mech.py (full-width beam interference checker) finds the diffracted-beam corridor between the slit tower and the OAP1 module infeasible for THIS GEOMETRY at real beam width; earlier clearances were chief-ray only. Do not freeze hardware from these numbers until the corridor decision in MECH.md is made. Stray-light budget: STRAYLIGHT.md.

Tools (run under `design/.venv`): `raytrace.py` (exact vector trace),
`tolerance.py` (perturbation sweep), `fourier.py` (slit-diffraction wave
optics), `plots.py` (layout.png, spots.png).

**REVISION NOTE:** rev 1 of this file reported config A at "20 deg
deviation" with 3-6 um field blur. That geometry was the product of a
solver bug: the deviation tuner converged to a spurious root at exact
Littrow retro-reflection, placing OAP2 coincident with OAP1 — optically
flattering, mechanically impossible (caught by the layout chart; the
telltale was anamorphism = exactly 1.000). The tuner now solves the signed
off-Littrow angle directly, and all results below are for buildable
geometry. Casefile ids: bug d64467a2/753eea86/abfd14b4 superseded via
correction entries.

## Chosen design: config B, 20 deg off-Littrow

| part | role | spec | price |
|---|---|---|---|
| Edmund #35-607 | collimator | Ø50.8 mm, 30 deg, RFL 81.79 | $599 |
| Edmund #35-588 | camera | Ø76.2 mm, 45 deg, RFL 178.53 | $649 |

Geometry (`CHOSEN` in raytrace.py): dev = +20 deg off-Littrow, camera fold
OPPOSITE the collimator fold (s2 = -1), Lg = 117 mm (slit-side clearance:
return beam passes ~38 mm from OAP1), Lc = 230 mm (sensor at z = -50,
y = +102 mm: camera axis 102 mm from the telescope axis, clearing an
Ø80 cooled IMX571 body beside the snout extension by ~20 mm). Magnification 2.18x:
7 um slit -> 15.3 um spatial image (4.1 px on IMX571); disk image 9.2 mm
(needs the IMX571-class sensor; IMX678 too small at this mag).

Why B over A (108.9 mm collimator): with ONE deviation angle shared by all
lines (the arms are a housing constant), A only beats B at dev <= 10 deg,
which needs a 230 mm grating arm for mechanical clearance. B peaks at
dev = 20 (coma balance), which is buildable at Lg = 117. B's shorter
collimator also throws a tighter diffraction fan -> less clipping on small
gratings.

## Performance (RMS spot radii, um; slit image is 15.3 um)

field 0 = disk center, 2.1 mm = disk edge, 3.5 mm = slit end; at line
center / +0.5 nm the numbers are essentially identical:

| line | center | disk edge | slit end |
|---|---|---|---|
| Ca K 393 | 0 | 7.2 / 4.8 | 12.9 / 7.9 |
| Ha 656 | 0 | 7.1 / 4.5 | 12.2 / 9.7 |
| He 1083 | 0 | 4.4 / 1.9 | 8.5 / 4.3 |

(spatial / dispersion). Verdict: excellent over the disk (blur < half the
slit image), marginal at the extreme slit ends — which only matters if the
full 7 mm slit length is used; the 4.2 mm solar disk lives inside the
good field. Rich-mode window: ~40-60 um at +/-10 nm (unchanged story);
usable co-registered window remains ~+/-1.5-2 nm.

Anamorphism is now real: cos(a)/cos(b) = 0.82 (CaK) / 0.62 (Ha) / 0.73
(He). The spectral image of the slit is compressed vs the spatial axis —
higher spectral resolution than the geometric slit suggests; recon must
not assume square pixels across axes.

## Tolerances (Ha, sensor refocus as sole compensator, +3 um budget)

| DoF | tolerance | note |
|---|---|---|
| OAP1 / OAP2 yaw | 0.025-0.026 deg (~1.5 arcmin) | TIGHTEST; adjust while watching live spot |
| grating in-plane yaw | 0.033 deg (~2 arcmin) | shows as focus/tilt on live spectrum |
| OAP2 pitch | 0.055 deg | kinematic mount |
| OAP1 pitch | 0.088 deg | kinematic mount |
| OAP2 decenter | 0.45 mm | easy |
| slit despace, OAP clocking, grating clocking, OAP2 despace | insensitive at tested magnitudes | shifts only |

Tighter than rev 1 (yaws now matter at ~1.5 arcmin) but still
kinematic-mount + live-spectrum territory. A GhostSun live alignment
readout (spot FWHM + line tilt) remains the enabling build tool.

## Fourier pass (throughput / delivered line spread, 50 mm grating)

| line / slit | delivered | LSF broadening |
|---|---|---|
| CaK / 5 um | 98% | +1% |
| Ha / 7 um | 94% | +6% |
| He / 10 um | 95% | +4% |

Existing Shelyak 25 mm grating: 74-78% delivered at Ha/He, 22-30% LSF
broadening — usable (comparable clipping already exists in the lens
Sol'Ex), staged-build plan stands: first light on the Shelyak grating,
upgrade to Thorlabs GH50-24V (£313, in stock) when wanted. Baffle sites:
1-4% in a ring around OAP1 (dispersion plane) and up to 4% past the
grating edges.

## Thermal / mechanical notes

- Mirrors see ~7 mW total: no cooling. Slit absorbs ~1 W: tilt it; ERF
  optional. Aluminum mirrors + aluminum housing = athermal.
- Envelope roughly 300 x 200 mm in the fold plane (slit at origin, OAP1 at
  z=+82, grating at (-20, -58), OAP2 at (+109, +95), sensor at (-69, +79)).
- Grating cartridge must accept 25x25x6 (Shelyak) and 50x50x9.5 (Thorlabs).

## Open items

- Mechanical CAD: housing, cartridge mounts (slit + grating), baffle ring,
  camera mount clearance vs telescope drawtube (verified in layout.png at
  chief-ray level; needs solid-model check with real camera body).
- He I 1083 grating choice: GH50-12V is visible-optimized; compare 1083 nm
  efficiency vs ruled blazed 1200/mm before purchase.
- Optiland cross-check of the OAP relay (optional second opinion).
