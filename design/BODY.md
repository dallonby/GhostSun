# Instrument body (v1 blockout)

`shg_body.scad` + `body_geom.scad` (AUTO-GENERATED — run
`.venv/bin/python body_export.py` after any change to `CHOSEN` in
raytrace.py; never hand-edit positions).

Single source of truth: every optical anchor (slit, OAP faces, grating
pivot, sensor plane, all beam angles, per-line grating tuning angles) is
exported from the raytraced CHOSEN geometry. The printed body IS the
raytrace.

## Architecture

- Flat optical deck parallel to the fold plane; beams at `beamH` (75 mm —
  chosen so the 150 mm KM plates from mounts/ sit on the deck).
- Two angled wall slabs carry the printed kinematic bases; face normals
  are the mirror substrate axes; M5 heat-set pilot holes match the KM
  base bolt triangle (kmHole = 64 mm half-pitch).
- `mirrorStack` (45 mm default) = optical face -> wall face. MEASURE the
  real stack (substrate + backing plate + KM mount + gap) before printing.
- Slit: cartridge pocket in the tower at the origin, tilted `slitTilt`
  (10 deg) about vertical so the jaw reflection dumps onto the interior
  wall, not back up the snout. Pocket sized for a 12 x 22 x 3.2 slide —
  measure the actual Shelyak slit holder and adjust.
- Enclosed snout from the telescope flange (front wall) to the slit; the
  flange bore is 38 mm, adapt to T2/2" with a threaded insert or bonded
  adapter.
- Camera tunnel from the sensor plane through the front wall along the
  focused chief (5 deg off the deck axis); bore 34 mm; bolt a helical
  focuser + ToupTek adapter to the exterior face. Sensor plane sits at
  the interior end of the tunnel — set back-focus with the focuser.
- Grating rotator: deck turntable, vertical steel dowel pivot (8 mm), the
  rotor registers the grating FRONT face on the pivot-axis plane
  (thickness-agnostic: 50x50x9.5 direct, Shelyak 25x25x6 via
  `shelyak_adapter` sleeve). Tangent arm + arc clamp slot.
- Per-line tuning detents (from body_geom.scad): CaK 68.6 deg / He I
  81.3 deg / Ha 93.1 deg grating normal -> three tangent-screw posts on
  the deck; coarse-position the arm at a post, clamp the arc screw, trim
  with the M4 tangent screw (0.57 deg/turn at armR = 70).
- Stray-light vane at by = +25 with a single aperture where the
  diffracted beam crosses; separates the slit/snout region from the
  camera corridor. Flock or matte-black the interior, especially the
  dispersion-plane ring around OAP1 (Fourier pass: 2-4% lands there).

## Known v1 gaps (by design, finish in Fusion)

- No lid (flat plate + lip + M3s; add a labyrinth edge for light
  tightness).
- Telescope flange is a plain bore + 4 bolts — needs the real T2 adapter
  decision.
- Camera flange bolt pattern generic; match the chosen helical focuser.
- Slit pocket and grating cell dims need verification against the
  physical parts (slit slide, grating thickness tolerances).
- Motorized tuning (stepper on a worm ring replacing the tangent posts)
  is the intended v2 — GhostSun already drives hardware; software line
  selection is the payoff.

## Workflow

1. Edit optics -> `raytrace.py` (`CHOSEN`), re-run `body_export.py`.
2. Preview: `openscad shg_body.scad` (part = "preview").
3. Export STEP/STL per part (`part = "body" | "rotor" | "shelyak_adapter"`)
   and refine cosmetics in Fusion; keep anchor dimensions from
   body_geom.scad.
