# Printable kinematic OAP mounts

Two identical mounts (one per OAP), parametric, derived from the rev-2
tolerance analysis in ../RESULTS.md.

## Files

- `kinematic_mount_fusion.py` — **primary**: Fusion 360 script. Utilities >
  Add-Ins > Scripts and Add-Ins > add + Run. Builds `KM_Base` and
  `KM_Platform` components driven by `km_*` user parameters (Modify >
  Change Parameters). Edit parameters after the first run; no re-run needed.
- `kinematic_mount.scad` — OpenSCAD twin for quick STL export / preview.
  Same parameter names, same geometry. `part = "base" | "platform" | "both"`.

## Kinematic scheme (cone / vee / flat)

- Fixed steel ball glued into the platform pocket rides the base **cone**
  (the pivot — no adjustment).
- Adjuster screw at `adjA` (lever arm along Y) lands in the **vee** whose
  groove AIMS AT the cone → pure **pitch**.
- Adjuster screw at `adjB` (lever arm along X) lands on the **flat** →
  **yaw** about the cone–vee line.
- Two steel extension springs preload the plates.
- Mirror backing plate bolts to the platform at the support-triangle
  centroid (keeps the mirror's mass inside the ball triangle).

## Resolution / budget (from raytrace tolerances: OAP yaw 1.5 arcmin)

deg per turn = atan(screw pitch / leverArm):

| leverArm | M4 x 0.7 | 100 TPI (0.254 mm) |
|---|---|---|
| 120 mm | 20 arcmin/turn | 7.3 arcmin/turn |
| 150 mm | 16 arcmin/turn | 5.8 arcmin/turn |

A controlled 1/20 turn ≈ 1.0 / 0.36 arcmin — inside budget either way;
100 TPI adjusters make it comfortable.

## Hardware per mount

- 3× Ø6 mm steel balls (bearing balls; one glued in platform, two epoxied
  to adjuster screw tips — or buy ball-tip 100 TPI adjusters)
- 2× M4 fine screws + heat-set inserts, or 2× 100 TPI adjuster + bushing
- 2× steel extension springs, ~8–10 N at working length, span `gap`
- 3× M4 screws into the mirror's backing plate; 3× M5 to the housing
- 2× M4 jam nuts (lock after alignment) + epoxy tack

## Printing

- **ASA, PC, or annealed PETG — not PLA** (solar enclosure heat + creep).
- Plates flat on the bed, 6+ perimeters, ≥40% infill, 0.2 mm layers.
- The cone and vee print facing up on the base plate — no supports needed
  at these shallow depths. Ream the pivot pocket to a snug ball fit.

## Per-mirror print settings

- OAP2: defaults (mirrorOffX = 0).
- OAP1: platform printed with **mirrorOffX = -34** — the body's OAP1 slab
  is shifted +34 mm along its face so the diffracted beam clears the
  plate edge; the offset keeps the mirror at the optical position.

## Before printing: measure two things

1. The actual bolt pattern on your Edmund mirror backs (set `mirrorBCD`,
   `mirrorBoltD`, `mirrorBoltN`) — the plate can double as the backing
   plate (skip Edmund's #47-112/#63-375 entirely) if you match the
   mirror's own tapped holes.
2. The mirror clear-swing envelope: platform edge must not vignette the
   beam at ±3° of adjustment.

## Alignment procedure (from RESULTS.md)

LED slit → live spectrum → camera focus at field center → OAP2 pitch/yaw
then OAP1 pitch/yaw minimizing line FWHM at the spatial-axis ends →
grating yaw set-screw for symmetric sharpness → lock jam nuts, epoxy tack.
Re-trim seasonally if the plastic creeps; the live readout makes that a
five-minute job.
