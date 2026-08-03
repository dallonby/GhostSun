# Kinematic OAP mount — v3 (48 mm plate, 25.4 mm MPD)

Scope: v3 budget build only (casefile 49e5a75e). CAD is parametric; only
the v3 parameter set is validated. CAD: `kinematic_mount_v3.scad`.
Views: `km_v3_assembled.png`, `km_v3_exploded.png`.

Layout (plate coords, u/v in the plate plane, z = plate normal off the wall):

- pivot cone (8,8) · vee (15.5,40.75) · flat (40.75,15.5) — lever 30.25 mm
- M5 pattern (8,40),(40,8),(40,40) = ±16 mm about plate centre, (8,8) free
- mirror centre (24,24); springs on top edge u=32 and bottom edge u=16
- stack: wall z=0 | base 8 | gap 21 | platform 8 | mirror 8 | face z=45.0

Adjustment, achieved (100 TPI, lever 30.25 mm):

- **0.481 deg/turn = 28.9 arcmin/turn**, both axes
- **0.80 arcmin per 10° knob nudge** (spec floor 2.0; tolerance 1.7)
- ±2° range costs ±1.06 mm tip travel, ~4.2 turns end to end

## BOM (per mount; two mounts needed: COLL-50.8, CAM-101.6)

| qty | part | spec |
|---|---|---|
| 1 | base plate | printed ASA or PC, wall-side down on the bed |
| 1 | platform | printed ASA or PC, mirror-side down on the bed |
| 1 | seat cap | printed, any filament |
| 1 | OAP mirror | Thorlabs MPD124-P01 (collimator) / MPD144-P01 (camera) |
| 1 | pivot ball | 6 mm steel ball, glued (epoxy) |
| 2 | fine adjusters | 1/4″-100 (100 TPI) × ~2″, ball tip, hex/knob ≤ Ø9.5 (Thorlabs F25SS-class) |
| 2 | lock nuts | 1/4″-100 (Thorlabs LN24100-class) |
| 2 | press-in bushings | 9.5 mm OD (Thorlabs F25SSB-class; verify OD = bushingD before printing) |
| 2 | extension springs | stainless, **free length ~15 mm, rate ~0.5 N/mm, max extension ≥15 mm**; working length 26 mm → ~5 N each, ~10 N total (band 8–16 N) |
| 3 | M5 × 16 socket head screws | head Ø ≤ 8.5 mm (0.89 mm design clearance at the yaw tip) |
| 3 | M5 heat-set inserts | Ø6.4 × 12, set in the body wall (reference; not part of the mount) |
| 1 | 8-32 × 3/4″ screw + washer | mirror clamp, head lives in the gap |

Moving-side mass ≈ 56 g (platform ~17 g, mirror ~11 g, adjusters ~20 g,
rest ~8 g) vs 150 g budget.

## Assembly + alignment (10 lines)

1. Press both bushings into the platform seats until the flanges seat; set the 3 M5 inserts in the wall.
2. Glue the 6 mm ball into the pivot-post pocket; cure fully before loading.
3. Bolt the mirror on (8-32 + washer); clock the off-axis mark per station, then snug.
4. Thread both adjusters to mid-travel; start the lock nuts loosely.
5. Screw the base to the wall, 3× M5 socket heads — heads stand proud in the gap by design.
6. Seat the platform: notch over the fence, ball into the cone, tips to vee/flat — one orientation only.
7. Hook each spring: bottom hook down through the base ear hole, stretch, top hook up through the platform ear hole.
8. Align: vee adjuster = pitch, flat adjuster = yaw; 0.48°/turn, work in ≤10° nudges (0.8 arcmin).
9. Lock: hold the adjuster hex, snug the lock nut, re-check pointing (< 0.5 arcmin shift).
10. Removal: unhook the two top loops, lift by the tab, snap the cap onto the base.

## Notes / watch items

- **M5 heads proud in the gap** (no counterbores anywhere): wall-side face
  is virgin (defect 3 fix), and removing counterbores is what freed the
  seat layout (defect 1 enabler). Yaw-tip-to-head clearance is 0.89 mm —
  use standard Ø8.5 socket heads, not flange heads.
- **Springs anchor on top/bottom edge ears** (the beams graze the two
  vertical side edges only; body renders confirm). Hooks are outside the
  silhouette, tool-free, never in the gap / against the wall / under the
  mirror (defect 2 fix). There are no spring slots at all (defect 1 fix).
- **mirrorT = 8 mm is the body CAD's substrate proxy** — verify MPD124/144
  thickness from the Thorlabs solid models; `gap` follows automatically
  (`gap = 45 − 2·plateT − mirrorT`). Per-mirror thickness differences go
  in `mirrorT` per station, not in new parts (casefile 1ebe4716).
- Knob-to-mirror clearance is 1.33 mm at Ø9.5 knobs — do not fit larger
  knobs.
- Machining route: same topology; the pivot post (≈18 mm) and key fence
  (to z≈40) are the two features that need stock thicker than 8 mm or a
  brazed-on boss — everything else mills from plate.
- Self-check: `openscad -o /tmp/x.stl -D 'part="check"'
  kinematic_mount_v3.scad` — 17 assertions over the v3 set (stack,
  resolution, spring resultant inside the triangle, anchors vs mirror,
  seats vs M5, tips vs heads, knobs vs mirror, ligaments, label, fence,
  beam sides, flush wall face). Any violation aborts the render.
