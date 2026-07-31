# All-reflective SHG — raytrace results (2026-07-31)

Tracer: `raytrace.py` (exact vector trace, stdlib; run under `design/.venv`).
Validation: on-axis field at tuned wavelength focuses to machine precision
(~1e-12 um RMS) through the full slit -> OAP1 -> grating -> OAP2 -> sensor
chain; catalog RFL/PFL ratios reproduced for both vendors' angle conventions.

Feed: f/6.9 (FSQ-85 @ 65 mm stop, 450 mm FL). Slit 7 um x 7 mm along x.
Fields: 0 / 2.1 (disk edge) / 3.5 mm (slit end). Gratings: 2400 l/mm
(CaK, Ha), 1200 l/mm (He 1083). Sensor budget: slit image ~11.5 um,
pixel 3.76 um -> blur RMS <3 um negligible, <8 um usable.

## Winner: config A — Edmund protected aluminum

| part | role | spec | price |
|---|---|---|---|
| #35-494 | collimator | Ø50.8, 30°, RFL 108.9 | $499 |
| #35-588 | camera | Ø76.2, 45°, RFL 178.5 | $649 |

Geometry: grating deviation ~20°, camera fold sense OPPOSITE to collimator
(s2=-1) — this asymmetric pairing near-cancels field coma (verified: matched
30/30 config E is 10x worse at field). Magnification 1.64x.

Fast mode (±0.5 nm window), worst RMS radii:
- CaK 393: 4.9 um (slit end), 3.4 um (disk edge)
- Ha 656: 5.8 um (slit end), 4.2 um (disk edge)
- He 1083: 4.8 um (slit end), 3.0 um (disk edge)
All usable; disk-edge performance near-negligible vs 11.5 um slit image.

Rich mode: blur grows ~linearly with window offset; ~6 um at ±1 nm,
~12 um at ±2 nm, 40-60 um at ±10 nm. Usable co-registered window is
therefore ±1.5-2 nm (line core + wings + local continuum — covers the
velocity-min composite use case), NOT ±15 nm panoramas.

## Eliminated

- B (#35-607 81.8 mm collimator): 20 um field blur — collimator field angle
  too large at short RFL.
- C (Thorlabs silver 45/45, MPD364+MPD3124): 48 um at slit end even on-line.
  45° collimator is the killer, not the camera.
- D (Thorlabs aluminum 90/90, MPD249+MPD269): 57-70 um at slit end.
  Confirms 90° OAPs unusable for slit-field imaging.
- E (#35-494 + #35-580 30/30): 47 um at slit end — breaking the 30/45
  asymmetry un-cancels coma. Angle asymmetry is load-bearing.

## Tolerance sweep (tolerance.py, Ha, worst of field 0/3.5 mm, sensor
refocus allowed as sole compensator)

Nominal at best focus: 4.7 um RMS. Tolerances quoted = magnitude of that
single DoF adding +3 um RMS blur:

| DoF | tolerance | note |
|---|---|---|
| grating clocking (about normal) | 0.047 deg (~3 arcmin) | TIGHTEST; visible as line tilt (3-5 px over slit) -> align live on spectrum |
| OAP2 pitch (fold plane) | 0.064 deg (~4 arcmin) | kinematic mount territory |
| OAP1 pitch (fold plane) | 0.075 deg (~4.5 arcmin) | kinematic mount territory |
| OAP1 clocking (about chief) | 0.13 deg (~8 arcmin) | mount alignment pin + registration |
| slit/OAP1 despace | 0.21 mm | set by zero-order autocollimation (resolves ~10s of um) |
| OAP yaw, decenters, OAP2 clock/despace, grating in-plane yaw | insensitive | image shifts only (100-500 um) -> absorbed by tuning/pointing |

Verdict: amateur-buildable. All sensitive DoFs are arcminute-class, each
observable on the live spectrum itself (line tilt, spot focus) — an
in-GhostSun live alignment readout would make assembly procedural.

## Fourier pass (fourier.py — slit sinc^2 fan conv geometric cone,
propagated through the raytraced aperture stack)

Better than the back-of-envelope estimates. With the 50 mm grating,
delivered energy and line-spread broadening:

| line / slit | delivered | LSF broadening |
|---|---|---|
| CaK, 5 um | 97% | +2% |
| Ha, 7 um | 94% | +6% |
| He 1083, 10 um | 95% | +5% |
| He 1083, 7 um | 93% | +10% |

The short 108.9 mm collimator keeps the whole fan compact — the earlier
"15-25% He clipping" fear was pessimistic. Bonus: at 20 deg deviation the
geometry is symmetric (alpha = beta, anamorphism 1.000) at all three lines —
no anamorphic distortion, square pixels preserved.

The existing Shelyak ~25 mm grating delivers ~75% at Ha / 74% at He with
21-35% bandpass broadening. IMPORTANT NUANCE: the current lens Sol'Ex has
the same clipping physics (80 mm collimator, same fan vs same grating), so
the mirror build with the existing grating is roughly AT PARITY with the
instrument in use today — fully usable, just not the upgrade's full value.
Staged plan: build with the Shelyak grating (free), drop in a 50 mm later.
Grating cartridge must accept both form factors from day one
(Shelyak 25x25x6 mm vs Thorlabs 50x50x9.5 mm).

Verified source (2026-07-31): Thorlabs GH50-24V, visible reflective
holographic, 2400/mm, 50x50x9.5 mm, £313.18, in stock (ships Bergkirchen).
Companion GH50-12V (1200/mm, same price/size) exists for the He I cartridge,
but it is VISIBLE-optimized — check its efficiency at 1083 nm vs a ruled
blazed 1200/mm (Richardson/Newport, Edmund, Optometrics) before buying the
IR grating. Holographic = low ghosting/stray light, 45-65% peak efficiency.

Baffle sites (small but worth catching in a dark-line-core instrument):
2-6% of the light lands in a ring around the collimator aperture in the
dispersion plane (|y| = 23-54 mm at OAP1) -> matte baffle ring there;
0.1-2.6% overshoots the grating edges -> trap behind/beside the grating.

## Open items
- Mechanical: 20° grating deviation needs clearance check; all-aluminum
  housing makes instrument athermal (mirrors are Al substrate).
- Thermal: mirrors see ~7 mW total — no cooling; slit absorbs ~1 W
  (tilt + optional ERF).
