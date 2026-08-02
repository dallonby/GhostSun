# Ha-only budget prototype ("sanity build") — rev 4 (v3), 2026-08-02

**v3 FROZEN 2026-08-02** (FOLDS.md study): back to the cheap 45/45 pair
(MPD124 + MPD144) plus one O25.4 protected-silver fold flat 46 mm after
OAP2, folding the focused beam +90 deg away from the telescope. The
flat sits in a FIXED printed seat with three nylon-tip grub screws: its
tilt tolerance is unbounded (fold_tol.py), so no adjusters. Supersedes
the interim MPD246 60-deg freeze (rev 3), which cost more and imaged
worse. body_export.py guard: zero collisions, zero fan vignettes.
Stray-light budget: STRAYLIGHT.md. Thermal: THERMAL.md.

Goal: cheapest build that validates the all-reflective concept end-to-end
(geometry, printed mounts, rotator, alignment flow, GhostSun integration)
using owned parts wherever possible. Active CHOSEN config: T_budget_ha_v2.

## New purchases

| item | part | price |
|---|---|---|
| collimator | Thorlabs MPD124-P01 (Ø25.4, 45°, RFL 50.8, prot. silver) | £197 |
| camera mirror | Thorlabs MPD144-P01 (Ø25.4, 45°, RFL 101.6, prot. silver) | £197 |
| fold flat | Thorlabs PF10-03-P01 (Ø25.4, lambda/10, prot. silver) | £73 |
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

## Geometry (CHOSEN v3, frozen by fold_study.py sweep 2026-08-02)

MPD124 + MPD144 (45/45), dev = 24 deg off-Littrow, s2 = -1, Lg = 70,
Lc = 130, fold flat at 46 mm after OAP2 folding +90 deg. alpha 65.6 /
beta 41.6, anamorphism 0.552. Mag 2.0: slit image 14.0 um spatial /
7.7 um dispersion; disk image 8.4 mm on the 7.68 mm IMX678 axis ->
poles clip ~9% (unchanged; IMX571 production path un-clips).
Worst core-beam clearance +4.4 mm; worst fan +1.7 mm (no grazes this
time; flock the tower blade anyway). Sensor at body (-17, 104), camera
exits through the top back corner at 114 deg: fully decoupled from the
telescope side.

## Performance at Ha (+-0.5 nm window, RMS radii)

- Blur: 2.0 um (center) / 9.3 um (disk edge) / ~20 um (slit end) vs
  14 um slit image.
- R ~ 27,700 slit-limited (0.024 nm) — resolves the 0.05 nm core.
- One extra silver reflection: ~2.5% light, ~0.15% veil (2 nm flat).
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
- Slit ends soft (~20 um) — full-disk work lives inside the good field.
- Disk pole clip ~9% on IMX678.

## Upgrade path

Same body concept: production = Edmund Al pair (#35-494/#35-588 class),
IMX571, 50 mm grating. CHOSEN in raytrace.py holds the production values
in a comment; flip config + re-run body_export.py + make_all_scad.py to
regenerate the production CAD.
