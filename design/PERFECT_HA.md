# PERFECT_HA — uncompromising Hα spectroheliograph design study (2026-07-31)

Clean-sheet Hα-only design for the FSQ-85ED + GhostSun printed platform,
derived from first principles with all prior assumptions re-examined.
Analysis tools (new files, run under `design/.venv`): `perfect_ha_req.py`
(requirements & photon budget), `perfect_ha_core.py` (shared physics:
delivered-LSF, capture, collision model), `perfect_ha_sweep.py` +
`perfect_ha_stage2.py` (exact-raytrace architecture sweeps),
`perfect_ha_littrow.py` (C3 disproof), `perfect_ha_lens.py` (C2 exact
doublet trace), `perfect_ha_final.py` (final prescription printout).
Existing `raytrace.py` used read-only; CHOSEN untouched.

## 1. Derived requirements

| requirement | value | derivation |
|---|---|---|
| delivered spectral FWHM | ≤ 30 pm req, ≤ 25 pm goal | Hα chromospheric core ≈ 50 pm FWHM; filament/fibril contrast saturates once instrument ≤ core/2. 0.25 Å-class etalons are the imaging gold standard |
| Doppler range mapped | ± 50 km/s (± 109 pm) | active filaments/surges 20–50 km/s; sub-resel centroiding gives ~1 km/s velocimetry at SNR ≥ 100 |
| spectral purity | far-wing + scatter ≤ ~1 % of continuum into core | core residual intensity is 0.16 I_c; a few % veil visibly greys filaments |
| spatial resolution | optics blur ≪ seeing image everywhere on disk | seeing 2″ (1.5″ best) ⊕ diffraction: 2.9″/2.6″ at 65 mm; 2.5″/2.2″ at 85 mm = 5.4–6.2 µm FWHM at the slit (scale 2.18 µm/″); r0(656) = 70–93 mm, so ms-class frames are near-diffraction-limited with software destretch |
| field | full 4.2 mm disk + prominence margin; slit ends ≥ ±2.6 mm clean | |
| photometry | line-core SNR ≥ 100 per 2″ resel per frame | achieved trivially (Sec. 2) |
| operations | refocus-free, repeatable line selection, 30–80 s scans | GhostSun motorised rotator v2 |
| budget | ≤ $3000 optics | actual: ~$2.0k |

## 2. First-principles findings (the priors, challenged)

**Photons are not the currency.** Ground irradiance ~1.4 W/m²/nm at 656 nm
through a 65 mm stop and a 7 µm slit delivers ~2.5×10⁸ e⁻/s per 2″×20 pm
resel after 42 % instrument transmission. The IMX571 well (51 ke⁻) fills
at continuum in **1.4 ms**. Exposure is full-well-limited near 1 ms with
core SNR ≈ 200 per frame; even prominences (1–5 % of disk) reach SNR
15–40. Throughput therefore buys nothing on-disk — the design currencies
are delivered FWHM, purity, blur, and scan cadence. (`perfect_ha_req.py`)

**Grating-limited resolution law.** With the collimator sized to pass the
full first diffraction lobe of the slit (aperture rule
D = f₁(1/f# + 2λ/w)), the slit-limited bandwidth is

  Δλ_slit = σ(w/f# + 2λ)/W_g  — independent of incidence angle α.

For the 50 mm GH50-24V: **19.4 pm (R 33.8k)** at 7 µm/f6.9, 17.0 pm
(R 38.7k) at 5 µm/f6.9. The owned Shelyak 25 mm caps at 38.8 pm (R 16.9k)
— the 50 mm grating is mandatory for the mission. A 68 mm grating would
reach R 46k, but no catalog 2400 l/mm visible holographic exists at that
size (Newport/Richardson 2400 masters are UV-peaked; HORIBA = quote).
Accepting ~5 % fan clip (capture 95 %) buys the anamorphic α-boost above
the law's full-capture value — the Sol'Ex trick in moderation: Sol'Ex
runs α = 72.4° with heavy Hα vignetting on a small grating; we run
α = 60.7° with 95 % capture on a 50 mm one.

**The 65 mm stop stays (with an 85 mm mode).** Opening to 85 mm (f/5.3)
improves combined seeing+diffraction blur only 2.9″→2.5″ typical, and the
fatter cone eats grating width: R ceiling drops 33.8k→29.9k at 7 µm and
disk-edge capture falls to 80–88 %. Since the stop is an external mask,
keep 65 mm as the purity/resolution baseline and open to 85 mm on
sub-1.5″ days (with the 5 µm slit) for spatial gain — quantified in the
mode table below. The stop is a per-session choice, not a design commitment.

**Slit: 7 µm baseline, 5 µm high-R mode.** 7 µm = 3.2″ (matched to 2″
seeing scan sampling with 2-frame-per-slit-width stepping); 5 µm = 2.3″
lifts R to 52k delivered at 6 % less capture and slightly more wing. Both
Shelyak, £~95 each. 10 µm only lowers R (28.5k ceiling) — rejected.

**Scan cadence is camera-limited, not photon-limited.** IMX571 USB3 strip
ROI (6248×256) ≈ 15–25 fps → 30–80 s per disk. The IMX678 does >200 fps
strips → 3–6 s scans, but at mag 2.18 clips the disk 9 % — retained as a
prominence-movie/fast-seeing secondary camera, not the primary.

## 3. Architecture candidates

| # | architecture | delivered FWHM @7 µm (centre/disk-edge) | capture | purity/ghosts | camera exit | cost (new optics) | verdict |
|---|---|---|---|---|---|---|---|
| C1 | **OAP pair + GH50-24V, dev +16°, s2=−1** (recommended) | **17.2 / 19.2 pm (R 38.2k / 34.1k)** | 95 % | mirror = zero ghosts; wing 1.2 % | 1° off axis, 115 mm lateral, +32 mm margin | ~$1.9k | **winner** |
| C2 | all-lens V (656-optimised doublets) | core optics diffraction-limited (<1 µm) — same LSF as C1 in the core | 95 % | fan zones at ~f/3 smear 3–60 µm → skirt on ~9 % of light; in-band ghosts 2–7×10⁻⁴ + semi-focused pairs | same V-fold needed | ~$1.6-2.0k (custom 656 doublets) | loses on purity; no broadband future |
| C3 | true Littrow, single OAP double-pass | on-axis spots **16–36 µm RMS** (OAP used 4–10 mm off its stigmatic field) | — | grating at blaze peak | **fails**: beams separate only <2–3 mm from focus; back focus 3 mm ≪ 12.5 mm flange; sensor-at-slit puts camera in the snout | ~$0.9k | dead twice (optics + packaging) |
| C4 | VPH transmission 2400 l/mm | same grating-limited law ⇒ no R gain | — | low scatter, but λ/Λ=1.57 ⇒ strong s/p split risk | 104° deviation (not straight-through) | no stock part at 2400/656 (Wasatch = custom, $2–4k, 50 mm CA) | rejected: cost/availability, no ceiling gain |
| C5 | echelle 79 g/mm m=34 (GE2550-0863, $398) | ceiling only +12 % (σ_eff 0.372 vs 0.417 µm at fixed 50 mm) | worse (63.4° footprint) | ruled scatter + inter-order; FSR 19 nm forces the same 10 nm prefilter | similar | ~$1.7k | rejected: purity regression for marginal R |
| C6 | stretch: 68 mm 2400 holographic + 85 mm stop | ceiling R 46k @7 µm | full | as C1 | as C1 | +$1.5–3k (quote) | future upgrade slot only |

Notes.
- C1 vs custom small-angle OAPs: stage-2 sweep shows 15–20° customs give
  the best raw field blur, but **every** such geometry fails the camera
  collision rule (short throw angles cannot put the Ø80/Ø100 IMX571 body
  ≥60 mm off the telescope axis; margins −37…−66 mm). The catalog 30°/45°
  pair is what buys the 115 mm lateral exit. Packaging, not aberration,
  selects the mirrors. (`perfect_ha_stage2.py`)
- C2: exact meridional trace of ideally-bent N-BK7/N-SF5 cemented
  doublets at 656.3 nm: f/6.9 core error 0.3–0.9 µm (validates Sol'Ex-class
  optics!), but the slit-diffraction fan fills the aperture to ~f/3
  *regardless of focal length* and picks up 59 µm (f=80) / 157 µm (f=150)
  transverse error at the fan edge — a permanent LSF skirt the paraboloids
  simply do not have. (`perfect_ha_lens.py`)
- C3: with ψ = 2.5° the return image lands 4.0 mm from the slit; outgoing
  cone and returning beam overlap everywhere except within ~2–3 mm of the
  focal plane, so no pickoff flat fits and no camera flange reaches.
  Out-of-plane (γ) variants displace along the slit axis and collide with
  the slit-length field instead. (`perfect_ha_littrow.py`)

## 4. Recommended design (full prescription)

### 4.1 Layout (world frame: slit at origin, feed along +z, fold plane y–z)

Config: `rfl1=81.79, th1=30, rfl2=178.53, th2=45, 2400 l/mm, m=1,
dev=+16.0, s2=−1, Lg=180, Lc=290` — build at 656.28 nm.

| element | position (x,y,z) mm | notes |
|---|---|---|
| slit S | (0, 0, 0) | Shelyak 7 µm (+5 µm), 10 mm, tilted 10° per BODY.md |
| OAP1 centre C1 | (0, 0, 81.79) | Edmund #35-607, Ø50.8, 30°, RFL 81.79 |
| grating pivot G | (0, −90.00, −74.09) | GH50-24V front face on pivot plane |
| OAP2 centre C2 | (0, 118.61, 127.36) | Edmund #35-588, Ø76.2, 45°, RFL 178.53 |
| sensor F2 | (0, 115.49, −51.15) | IMX571, 3.76 µm px |

Chief directions: c1 = (0, −0.500, −0.866); c2 = (0, 0.719, 0.695);
df = (0, −0.0175, −0.9998). α = 60.68°, β = 44.68°, anamorphism
cosα/cosβ = 0.689, grating tune γ = −60.68° (raytrace convention;
body_export regenerates the detent). Magnification: spatial ×2.18,
dispersion ×1.50. Plate: 6.2 pm/px; 0.603 mm/nm.

Camera exit df is **1.0° off the telescope axis** with the sensor 115 mm
lateral — collision margin **+32 mm** against the snout/drawtube/OTA chain
(z-dependent obstacle model mirroring BODY.md; production design cleared
by ~20 mm). Clearances: OAP1-to-return-beam 50 mm (need 47), slit-to-
collimated-corridor 41 mm, OAP2-to-corridor 80 mm (need 60). Optics
bounding box ~290 × 210 mm on the 350 × 300 deck; camera tail exits the
front wall beside the snout exactly as in the production body.

### 4.2 Bill of materials

| item | part | price | status |
|---|---|---|---|
| collimator | Edmund #35-607 OAP, Ø50.8, 30°, RFL 81.79, prot. Al | $599 | = production part |
| camera mirror | Edmund #35-588 OAP, Ø76.2, 45°, RFL 178.53, prot. Al | $649 | = production part |
| grating | Thorlabs GH50-24V, 2400 l/mm, 50×50×9.5 | £313 ($568 US list) | known buy |
| Hα prefilter | Edmund #19-820, 656/10 nm, OD4, Ø25, hard-coated | $272 | new — mounts ≥70 mm before slit in the snout |
| slit (high-R mode) | Shelyak 5 µm × 10 mm | ~£95 | optional |
| camera | cooled IMX571 | owned | Ø80 body, Ø100×10 flange |
| fast/secondary camera | ToupTek IMX678 | owned | prominence movies |
| slit 7 µm, 25 mm grating, printed body/mounts/rotator | — | owned | Shelyak 25 mm = alignment spare |
| **total new spend** | | **≈ $2.0k** | ceiling $3k |

Alternative (Hα-max variant): replace the Edmund pair with custom
protected-silver clones of the same RFL/angle (Shanghai Optics class,
$250–600 each) → slit-to-sensor throughput 42 % → 51 % and ~$650 saved,
at the cost of catalog-grade certainty and any future Ca K use. Photon
budget says the 18 % gain is not needed on disk; take catalog Al unless
prominence work dominates.

### 4.3 Delivered performance (raytraced + wave LSF, aberration included)

Mode table (LSF = slit ⊗ clipped-pupil diffraction ⊗ 3.76 µm pixel,
quadrature with traced dispersion-axis blur; centre / disk edge):

| mode (slit/stop) | capture | FWHM centre | R centre | FWHM disk-edge | R disk-edge | px/FWHM | wing >2×FWHM |
|---|---|---|---|---|---|---|---|
| **7 µm / 65 mm** (baseline) | 95 % | **17.2 pm** | **38,200** | 19.2 pm | 34,100 | 2.8 | 1.2 % |
| 5 µm / 65 mm (high-R) | 94 % | **12.5 pm** | **52,600** | 15.2 pm | 43,200 | 2.0 | 1.9 % |
| 7 µm / 85 mm (sharp) | 95 % | 17.2 pm | 38,200 | 20.9 pm | 31,300 | 2.8 | 1.2 % |
| 5 µm / 85 mm (best-seeing) | 93 % | 12.5 pm | 52,600 | 17.3 pm | 37,900 | 2.0 | 1.9 % |

All modes beat the 25 pm goal with margin; the baseline resolves the
50 pm Hα core ~3× and the 5 µm mode reaches 0.125 Å — beyond any amateur
etalon. One geometry serves all four modes: stop and slit are swappable
without realignment (slit cartridge, external mask).

Blur map (RMS spot radii µm, spatial/dispersion, f/6.9, real 450 mm pupil;
slit spatial image = 15.3 µm):

| field | Δλ=0 | +0.75 nm | +1.5 nm |
|---|---|---|---|
| centre | 0.0 / 0.0 | 2.8 / 4.0 | 5.5 / 8.2 |
| 1.05 mm | 2.1 / 0.7 | 2.7 / 3.6 | 5.0 / 7.7 |
| disk edge 2.1 | 5.2 / 2.2 | 3.9 / 2.6 | 4.4 / 6.4 |
| 2.6 mm | 7.3 / 3.3 | 5.5 / 2.1 | 4.8 / 5.6 |
| slit end 3.0 | 9.2 / 4.3 | 7.1 / 2.1 | 5.8 / 4.8 |

Optics blur at the disk edge (5.2 µm RMS at the sensor = 2.4 µm at the
slit) is ~40 % of the seeing image — spatial resolution is seeing/
diffraction-limited everywhere on the disk, both axes. The fast window
(±0.75 nm) stays clean; ±1.5 nm soft-focuses mildly (rich-mode unchanged
story from RESULTS.md).

Field-dependent capture (fan-wing vignette from footprint walk on the
grating, no field lens): 95 % centre → 90 % disk edge (7 µm). Smooth,
flat-fielded; optional slit field lens (Ø12.7 f≈165 plano-convex,
reimages the FSQ pupil onto the grating) recovers it if edge purity ever
shows in data — deliberately NOT in the baseline (adds 2 surfaces + a
near-slit ghost path).

Line geometry: smile −258 µm (69 px, −428 pm equivalent) across ±2.1 mm,
zero odd tilt — smooth quadratic, calibrated per column in recon
(INTI-standard practice; GhostSun recon must fit line centre per column).

Throughput (slit→sensor): prefilter 0.90 × Al pair 0.77 × grating ~0.60 =
**42 %** (silver variant 51 %). Upstream: FSQ ~0.95 × stop. Exposure
0.7–1.2 ms at gain 0 (full-well limited); core SNR/resel ≈ 150–200 per
frame; prominences SNR 15–40 (stack 2–4 scans). Scan: 600 cols (7 µm
steps) in 30–40 s, 1200 cols (3.5 µm) in 50–80 s at 15–25 fps ROI.

Purity budget (core veil, fraction of continuum): fan-clip wing ~1.2 %
inherent and symmetric (stable, calibratable); holographic grating
in-band scatter ~0.1–0.3 % over the ±5 nm prefiltered window (no Rowland
ghosts — the reason GH50-24V over any ruled option); mirror
micro-roughness TIS ~0.3 % near-angle; out-of-band blocked at OD4.
Total ≲ 1–2 % ⇒ measured core intensity ~0.17–0.18 vs true 0.16.
The 10 nm prefilter is the single biggest purity lever vs the Sol'Ex
heritage (which admits the full spectrum to scatter) — and it also cuts
slit-plane heating ~10× (≤0.1 W on the jaws).

### 4.4 Alignment & tolerances

Same optics family, fold sense and angle class as the production design →
RESULTS.md rev-2 tolerance table applies: OAP yaws 1.5–1.6 arcmin
(tightest), grating in-plane yaw 2 arcmin, OAP2 pitch 3.3 arcmin,
decenters ≥0.45 mm, sensor refocus as sole compensator. Printed kinematic
mounts + 100 TPI adjusters + GhostSun live spot-FWHM/line-tilt readout is
the validated flow. On-axis build sanity: traced on-axis spot is a
numerical point (<10⁻⁹ mm), matching raytrace.py validation.

### 4.5 Risk register

| risk | exposure | mitigation |
|---|---|---|
| GH50-24V efficiency at 656 nm unpublished (blurb: 45–65 % at peak; s/p split unknown at λ/σ=1.57) | scan SNR only (photon-rich) | measure on arrival (flat + photodiode); silver-clone variant regains 18 % if needed; Shelyak 25 mm proves geometry meanwhile |
| IMX571 strip-ROI fps (est. 15–25) | scan time 30–80 s | verify owned ToupTek; IMX678 fast mode 3–6 s as fallback; drive-rate scanning tolerates any fps |
| printed-body thermal focus drift (ASA ~90 ppm/K over ~470 mm path ≈ 0.35 mm/8 °C) | refocus-free goal | helical focuser + live focus metric; carbon-fibre rod stiffening or the production aluminum body close it out |
| prefilter thermal load (~2 W/cm² at Ø14 mm footprint) | CWL drift/stress | mount ≥70 mm before slit, reflective side sunward; optional 2″ UV/IR cut at the drawtube |
| smile 69 px | recon complexity | per-column line-centre fit (quadratic) — required GhostSun recon feature |
| 5 µm slit width ripple → transversalium stripes | cosmetic bands | standard slit-flat calibration; 7 µm baseline less sensitive |
| disk-edge fan vignette (95→90 %) | edge LSF wings | flat-field absorbs photometry; field-lens upgrade slot reserved in snout |

### 4.6 Reuse map

Unchanged from the existing programme: printed body concept + kinematic
mounts + 100 TPI adjusters, grating rotator (GH50 fits the 50×50×9.5
cartridge directly; Shelyak via existing sleeve), slit cartridge/tower,
M48 snout + M42 camera tunnel, body_export.py→OpenSCAD regeneration
(feed it this geometry verbatim), GhostSun live-alignment and scan
control, tolerance methodology. New printed parts only: prefilter cell in
the snout and the re-generated walls for Lg=180/Lc=290. Both mirrors and
the grating are the **same purchase list as the production broadband
design** — the perfect-Hα build and the production build share hardware;
only geometry (arms + detents) differs.

## 5. What the existing configs should learn

**T_budget_ha (active build):** (1) its true ceiling is the 25 mm grating
— Δλ_slit ≥ 38.8 pm (R ≤ 16.9k) by the grating-limited law, not the
~24 pm/R 27k in BUDGET.md which ignored fan clipping (74–78 % capture);
treat it as the geometry/software validator it is. (2) BUDGET's "grating
88 %" conflates geometric capture with diffraction efficiency — net is
~0.6× lower; photon budget absorbs it. (3) Add the $272 prefilter NOW —
it transfers to every later build and transforms budget-build purity too.
(4) Exposure should be ~1 ms full-well-limited; don't chase throughput.

**B_edmund (production broadband):** (1) For Hα sessions the SAME parts
re-tuned to dev +16, Lg 180, Lc 290 deliver R 38k (vs ~27k at the
broadband-compromise dev 20/Lg 117/Lc 240 geometry) with a larger
collision margin (32 vs ~20 mm) — since the body is regenerated CAD,
consider a second wall-set print ("Hα deck") or making Lg/Lc arms
relocatable between two bolt patterns. (2) dev sign/sense s2=−1 and the
rotator detent scheme carry over; add the +16° Hα detent. (3) The
anamorphic direction matters: positive dev (α > Littrow) compresses the
slit image — the prior A-vs-B dev sweep optimised blur only and left
~40 % of the available resolution unclaimed at Hα. (4) Prefilter slot in
the snout should be a standard feature of the production body.

## 6. Sources

- Thorlabs GH50-24V (2400 l/mm, 50×50×9.5, Borofloat, "45–65 % efficiency
  at peak", $568): thorlabs.com/item/GH50-24V; reseller spec mirror:
  fiberoptics4sale.com (T-GH50-24V)
- Edmund 656/10 nm OD4 hard-coated Ø25 bandpass, $272: edmundoptics.com
  /p/656nm-cwl-25mm-dia-hard-coated-od-4-10nm-bandpass-filter/19820/
- Newport/Richardson 2400 g/mm plane holographic masters (410H/420H/430H,
  UV-peaked 250–300 nm — no visible-optimised ≥50 mm stock):
  newport.com/c/plane-holographic-diffraction-gratings
- Wasatch Photonics VPH capabilities (150–5000 l/mm, custom; no stock
  2400@656): wasatchphotonics.com/product-category/gratings-and-
  diffractive-optics/
- Thorlabs echelle GE2550-0863 (79 g/mm, 63°, 25×50, $398):
  thorlabs.com/thorproduct.cfm?partnumber=GE2550-0863
- ZWO ASI2600MM (IMX571) ROI rates ~14 fps class at large ROI, USB3:
  astronomics.com product page + ZWO ASI2600 manual
- Sol'Ex optical theory (2400 l/mm, α=72.4°, β=38.4°, f 80/125, 10 µm,
  R≈40k, anamorphism 0.386): solex.astrosurf.com/solex-theory-en.html
- Solar irradiance ~1.4 W/m²/nm at 656 nm (AM1.5G); Hα core residual
  ~0.16 I_c (standard solar atlas values)
