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
| 48 mm (OAP1) | 50 arcmin/turn (too coarse) | 18 arcmin/turn |
| 73 mm (OAP2) | 33 arcmin/turn (too coarse) | 12 arcmin/turn |

With 100 TPI, a controlled 1/12 turn = the full 1.5 arcmin tolerance —
workable with lock nuts; the plates stay mirror-sized.

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

## Per-mirror print settings (mirror-sized plates, 100 TPI adjusters)

- OAP1 pair: plateW = plateH = 80, leverArm = 48 -> 18 arcmin/turn.
- OAP2 pair: plateW = plateH = 105, leverArm = 73 -> 12 arcmin/turn.
- 100 TPI adjusters are REQUIRED at these levers (M4x0.7 gives ~1 deg/turn
  at 48 mm — too coarse vs the 1.5 arcmin yaw tolerance). Lock after
  alignment as before.

## Module swap (engineered-in swappability)

The platform + mirror + adjusters form a married MODULE that retains its
alignment. The cone/vee/flat coupling is a repeatable connector: unhook
the two springs (open keyhole slots, no tools), lift by the tab, seat the
next module, re-hook. Re-seat repeatability is far inside the 1.5 arcmin
tolerance; verify on the live spectrum in seconds after each swap.

- Print one platform per mirror module (e.g. gold pair + aluminum pair =
  4 platforms, 2 bases). Set `label` per module ("OAP1-AU", "OAP1-AL"...).
- Bases are keyed (corner fence) so a module seats one way only; the two
  stations self-key by size (80 vs 105 plate).
- Print a `base_cap` per base: it covers the seats whenever a module is
  off. Grit in the vee costs arcminutes. Blower-puff seats before
  re-seating.
- Modules store face-down in a box with the cap on the ball side.

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
