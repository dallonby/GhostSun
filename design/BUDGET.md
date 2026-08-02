# Ha-only budget prototype ("sanity build") — rev 3, 2026-08-02

**GEOMETRY RE-FROZEN 2026-08-02** after the mech.py corridor findings
(MECH.md): camera mirror swapped MPD144 -> MPD246 (60 deg), asymmetric
slit tower, 48 mm OAP1 KM plate, auto-placed stray vane. body_export.py
now passes its interference guard with zero collisions and zero fan
vignettes. Stray-light budget: STRAYLIGHT.md.

Goal: cheapest build that validates the all-reflective concept end-to-end
(geometry, printed mounts, rotator, alignment flow, GhostSun integration)
using owned parts wherever possible. Active CHOSEN config: T_budget_ha_v2.

## New purchases

| item | part | price |
|---|---|---|
| collimator | Thorlabs MPD124-P01 (Ø25.4, 45°, RFL 50.8, prot. silver) | £197 |
| camera mirror | Thorlabs MPD246-P01 (Ø50.8, 60°, RFL 101.6, prot. silver) | £353 |
| 4x 100 TPI fine adjusters | Thorlabs | ~£80 |
| balls, springs, dowel, inserts, screws | — | ~£25 |
| M42 helical focuser | generic | ~£40 |
| **total new spend** | | **~£695 (~$890)** |

(The MPD144 45/45 pairing was £156 cheaper and 6 um sharper at the disk
edge, but no placement of it clears the telescope: its camera exit
converges 25 deg onto the OTA. The 60 deg MPD246 exits 6 deg off
parallel at 70 mm lateral separation and passes the interference check.)

Owned/printed: Shelyak 2400 l/mm 25 mm grating, 7 um slit, ToupTek
G3M678M (IMX678), FSQ-85 @ 65 mm, printed body + mounts + rotor (ASA).

## Geometry (CHOSEN, frozen by mech.py sweep 2026-08-02)

MPD124 (45 deg) + MPD246 (60 deg), dev = 21 deg off-Littrow, s2 = -1,
Lg = 80, Lc = 150. alpha 63.7 / beta 42.7, anamorphism 0.603. Mag 2.0:
slit image 14.0 um spatial / 8.4 um dispersion; disk image 8.4 mm on the
7.68 mm IMX678 axis -> poles clip ~9% (unchanged; IMX571 path un-clips).
Worst core-beam clearance +3.2 mm (beam3 vs slit tower); diffraction-fan
wings graze the tower blade at +0.1 mm by design -> flock that face
(STRAYLIGHT.md). Camera exits at -174 deg (6 deg off telescope-parallel),
sensor at body (-46, 70).

## Performance at Ha (+-0.5 nm window, RMS radii)

- Blur: 2.6 um (center) / 12.9 um (disk edge) / ~30 um (slit end) vs
  14 um slit image. The 45/60 pair gives up ~4 um at the edge vs the
  collision-impossible 45/45; center of disk is unaffected.
- R ~ 26,000 slit-limited (0.025 nm) — resolves the 0.05 nm core.
- Throughput: silver^2 (~95%) x grating 88% ~ 84% net — HIGHER than the
  production Al design at Ha (81% x 94% ~ 76%). The short collimator's
  tight fan is why the 25 mm grating stops mattering.
- Tolerances: same class as production (OAP yaw ~2 arcmin critical);
  same printed mounts, 100 TPI required.
- Sensor: dispersion image 3.7 px, spatial 7 px on 2 um pixels.

## Limits vs production design

- Ha only: silver kills Ca K (~25% net at 393). He I 1083 remains
  possible later (silver ~98% there; needs the 1200 l/mm grating and the
  10 um slit; collimator fan is marginal at Ø25.4 — check before buying).
- Slit ends soft (~30 um) — full-disk work lives inside the good field.
- Disk pole clip ~9% on IMX678.

## Upgrade path

Same body concept: production = Edmund Al pair (#35-494/#35-588 class),
IMX571, 50 mm grating. CHOSEN in raytrace.py holds the production values
in a comment; flip config + re-run body_export.py + make_all_scad.py to
regenerate the production CAD.
