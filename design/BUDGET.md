# Ha-only budget prototype ("sanity build") — 2026-07-31

Goal: cheapest build that validates the all-reflective concept end-to-end
(geometry, printed mounts, rotator, alignment flow, GhostSun integration)
using owned parts wherever possible. Active CHOSEN config: T_budget_ha.

## New purchases

| item | part | price |
|---|---|---|
| collimator | Thorlabs MPD124-P01 (Ø25.4, 45°, RFL 50.8, prot. silver) | £197 |
| camera mirror | Thorlabs MPD144-P01 (Ø25.4, 45°, RFL 101.6, prot. silver) | £197 |
| 4x 100 TPI fine adjusters | Thorlabs | ~£80 |
| balls, springs, dowel, inserts, screws | — | ~£25 |
| M42 helical focuser | generic | ~£40 |
| **total new spend** | | **~£540 (~$690)** |

Owned/printed: Shelyak 2400 l/mm 25 mm grating, 7 um slit, ToupTek
G3M678M (IMX678), FSQ-85 @ 65 mm, printed body + mounts + rotor (ASA).

## Geometry (CHOSEN)

45/45 silver pair, dev = 25 deg off-Littrow, s2 = -1, Lg = 85, Lc = 175.
Mag 2.0: slit image 14.0 um spatial / 7.5 um dispersion (anamorphism
0.535); disk image 8.4 mm on the 7.68 mm IMX678 axis -> poles clip ~9%
(acceptable for prototype; IMX571 production path un-clips).

## Performance at Ha

- Blur: 0 (center) / 9.2 um (disk edge) / ~19 um (slit end) vs 14 um image.
- R ~ 27,000 slit-limited (0.024 nm) — resolves the 0.05 nm core.
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
- Slit ends soft (~19 um) — full-disk work lives inside the good field.
- Disk pole clip ~9% on IMX678.

## Upgrade path

Same body concept: production = Edmund Al pair (#35-494/#35-588 class),
IMX571, 50 mm grating. CHOSEN in raytrace.py holds the production values
in a comment; flip config + re-run body_export.py + make_all_scad.py to
regenerate the production CAD.
