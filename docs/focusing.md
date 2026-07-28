# Focusing a spectroheliograph with GhostSun

The **Focus** tab measures focus numerically instead of by eye, and walks you
through the three adjustments in an order that has a single correct answer.

## Why there are stages

A spectroheliograph has three focus adjustments — telescope, collimator, camera
lens — and only the camera lens has an obvious reference, which on many builds
you cannot reach because the lens and camera shoulder will not come off as a
unit. The instinctive approach, optimising solar sharpness against all three, is
a shallow valley with no unique answer: every adjustment feels like a trade
against the others, because it is.

It is not actually degenerate. Look at which measurement responds to what:

| measurement                | telescope | collimator | camera |
|----------------------------|-----------|------------|--------|
| slit-jaw dust sharpness    | **no**    | yes        | yes    |
| spectral line width        | **no**    | yes        | yes    |
| solar detail along the slit| yes       | yes        | yes    |

Dust sits physically in the slit plane, and a spectral line is the dispersed
image of that same plane. Neither cares where the telescope's focal plane is.

So **Stage A** uses the first two rows to solve collimator and camera with the
telescope completely out of the picture — you can run it indoors, at the bench,
with a lamp on the slit. **Stage B** then has one unknown left.

Order matters. Focus the telescope against a mis-collimated spectrograph and it
settles wherever best hides the blur, while being unable to touch the real
problem — the telescope has no leverage in the dispersion plane at all.

## What you need

- Both micrometers readable, and a habit of always approaching a setting from
  the same direction (backlash).
- **A source with absorption lines.** GhostSun detects absorption lines only —
  local minima, fitted with an inverted Gaussian. **An emission source such as a
  neon lamp will not register at all**, however sharp its lines are.

  So for bench work, put **daylight** on the slit: sky through a window, or a
  white card in sunlight. That gives the full solar absorption spectrum, and it
  needs no clear view of the Sun itself, so Stage A still works indoors and
  under cloud.
- Pick a **telluric** line as the target, not Hα itself. Hα is about 1 Å wide
  and its shape changes with whatever is on the slit, so its width is not a
  clean focus signal.

---

## Setup (once per session)

1. Open the **Focus** tab. Click **⟳ Scan** if your camera is not already
   selected — GhostSun picks hardware over the synthetic source automatically.
2. Click **▶ Start**.
3. Set **dispersion axis** to match your camera's orientation: whichever way
   wavelength runs across the frame. This only decides which readout is labelled
   *spectral* and which *slit*, so if the two look swapped, flip it.
4. Set exposure so the frame is bright but not clipped. **auto-exposure** is
   fine for setup; turn it off before capturing so samples are comparable.
5. Check the two readouts are alive:
   - **Spectral line (dispersion)** — a number in px and Å
   - **Slit jaws / dust (spatial)** — a number in px

If the slit readout says *no line*, your slit jaws are too clean to give a
signal. Tape a single hair across a jaw, or use the slit ends — any hard edge in
the slit plane works.

### Telling the two families apart

They are perpendicular, and which one *looks* vertical depends only on how your
camera is rotated:

- **Slit-jaw dust** — a speck in the slit plane blocks one position *along the
  slit*, at every wavelength, so it draws a line along the **dispersion** axis.
- **Spectral lines** — one wavelength, dark at every position along the slit,
  so they run along the **slit** axis.

The quickest check that **dispersion axis** is set correctly is to look at the
two profile plots. The spectral one should obviously be a spectrum: many narrow
dips of varying depth. The slit one should be mostly flat with a scatter of
shallower dips. If those two look swapped, flip the toggle.

If you want certainty, change something and see what moves. Dust is fixed in the
frame and identical from frame to frame — it does not care where the telescope
points or whether it is focused, because it lives in the slit plane. Solar
structure moves; rotate the grating and the spectral lines slide along the
dispersion axis while the dust does not budge.

That immovability is exactly why Stage A works at the bench: dust is
telescope-blind, which is what makes it a clean measurement of the spectrograph
alone.

### Choosing which line each family measures

There are two independent selectors, **spectral target** and **slit target**,
each offering `narrowest`, `deepest` and `picked`. Click any dip in the
corresponding profile plot to lock onto that specific feature; the mode changes
to `picked` and the locked position is marked on the plot.

**Lock both before you start a sweep.** Stage A is only valid if the *same*
feature is measured at every camera position. Left on `narrowest`, the selection
can hop between features from frame to frame — most easily on the slit side,
where several dust specks of differing width compete — and that scatter lands
straight in the V-curve and therefore in Δ.

Choose a line that is **deep and well isolated**, not necessarily the narrowest
one on offer. A feature with clear space either side keeps the fitter from
wandering onto a neighbour as it broadens with defocus, which is precisely when
the sweep depends on it staying put. A dust line of around 10% depth is a good
slit target; on the spectral side any strong, unblended line will do.

Changing either selector resets the min-hold, because a held minimum measured on
a different feature means nothing.

---

## Stage A — collimator and camera

Select **A · spectrograph**. This is the bench stage; no telescope, no sun.

### You are not focusing on dust *or* on the spectral line

A natural question at this point is which of the two you are supposed to be
sharpening. The answer is neither: **you measure both at once, and what you are
after is whether they disagree.**

Every capture records both numbers, giving two V-curves with two minima. The gap
between them is what matters:

- **Δ ≠ 0** — the two are sharpest at *different* camera positions, so the
  **collimator** is wrong. Move the collimator, not the camera.
- **Δ = 0** — both are sharpest at the same camera position, and that position
  is your camera setting.

They can disagree because each metric probes a different plane:

| metric | width measured across | plane it probes |
|---|---|---|
| dust line width | the slit axis | slit-length plane — the grating does nothing here |
| spectral line width | the dispersion axis | dispersion plane — the grating expands the beam ~2–2.6× |

If the beam reaching the grating is not collimated, the grating gives those two
planes *different* wavefront curvature. That is astigmatism, and it vanishes
only when the slit sits exactly at the collimator's focal length. Driving Δ to
zero is therefore what replaces the infinity reference you cannot set
mechanically.

**Why not just focus the spectral line?** Because you can always make it sharp
by using the camera to compensate a mis-collimated beam — and that leaves the
along-slit direction defocused. That direction is the *vertical* resolution of
your reconstructed disk. You would be trading image sharpness for spectral
sharpness permanently, with nothing on screen to tell you.

### The two readings

Both fields want a number off a micrometer. GhostSun never interprets them — it
needs no units, no zero reference and no calibration, only numbers that
consistently identify where each adjuster is set.

| field | what you type | how it is used |
|---|---|---|
| **camera reading** | the camera micrometer, **stepped** every capture | x-axis of the two V-curves; Δ comes out in these units |
| **collimator reading** | the collimator micrometer, **unchanged** all sweep | x-axis of the null solve; one value per completed sweep |

During a sweep you only touch the camera adjuster. The collimator sits still, so
its reading is simply a label saying *this whole sweep was taken with the
collimator here* — it earns its keep at **bank Δ at this collimator**, below.

If your collimator has no marked scale, count turns or divisions from any
repeatable stop. The solve only needs the numbers to be linear in real
displacement, not to mean anything absolute. Whatever you type in is what
`set collimator to …` gives back.

Approach every setting from the same direction. Backlash between the reading and
the true position is the one thing that will quietly corrupt this, and it shows
up as scatter that looks like measurement noise.

### Where to put the camera micrometer

You do not know where focus is yet, so find the centre by hand first. Move the
camera micrometer while watching the live **Spectral line (dispersion)** readout
and find roughly where it bottoms out. The **min-hold** value helps: it
remembers the best reading you have passed through, so you can tell when you
have gone past the minimum and come back. Call that position **C**, click
**reset min-hold**, and sample around it. Precision here does not matter — you
only need to be close enough that the sweep straddles the true minimum.

Sample **symmetrically about C**, dense in the middle and wide at the ends:

```
C−0.25   C−0.10   C−0.05   C   C+0.05   C+0.10   C+0.25
```

Symmetry matters more than the exact spacing, for a non-obvious reason. The two
curves have very different widths:

| direction | effective f-ratio | depth of focus |
|---|---|---|
| slit / spatial | ~f/15 | ±0.32 mm |
| spectral / dispersion | ~f/6 (anamorphic) | ±0.05 mm |

That roughly six-fold difference is the same anamorphic factor that makes the
null test work at all, and it means no single step size suits both curves: fine
enough for the spectral one leaves the slit curve flat across the whole sweep,
coarse enough for the slit one steps straight over the spectral minimum. Hence
the dense core plus wings.

A parabola is only a local approximation, so at the outer points it does not fit
the spectral curve well — but a *symmetric* misfit leaves the vertex where it
is. Lopsided sampling is what actually biases the answer.

Treat the millimetre figures as a starting guess; they assume a Sol'Ex-like f/10
feed. The first sweep is reconnaissance — read the real curve shapes off the
plot and set the spacing from those:

- **Slit curve flat across the sweep** → widen the outer points, try ±0.4 mm.
- **Spectral curve has no clear bottom**, or it reports *curve bends the wrong
  way* → you stepped over it; tighten the inner points to ±0.02 / ±0.04.
- **Both roughly double their FWHM at the outer points** → about right.

It is worth settling this once, because every later sweep reuses the spacing.

### One sweep

1. Type your current **collimator reading** in. It stays fixed for the whole
   sweep.
2. Set the camera micrometer to the first position. Type the value into
   **camera reading**.
3. Click **◉ capture**. It averages **frames 40** frames (adjustable) and adds
   one point to both curves.
4. Move to the next position and repeat. **At least 3 positions, and they must
   straddle each minimum**; seven as above is comfortable.

Watch the **Stage A V-curves** plot below the profiles: two U-shapes with their
fitted parabolas. Watch **Δ** in the panel:

```
spectral min: 8.1420 ± 0.0031   (n=5, rms 0.019)
slit min:     8.0910 ± 0.0028   (n=5, rms 0.014)
Δ = +0.0510 ± 0.0042
astigmatic — move the collimator
```

`undo last` drops a bad sample. `clear sweep` starts over.

### Finding the collimator setting

Do **not** guess which way to move the collimator — the direction depends on
your instrument's geometry, and guessing sends you the wrong way. Let the app
measure it:

1. With Δ solved, click **bank Δ at this collimator**. This stores the
   (collimator, Δ) pair and clears the sweep.
2. Move the collimator a deliberate amount — 0.2 mm is a good first step. Update
   **collimator reading**.
3. Run a second full sweep, then bank it.

The panel now prints:

```
set collimator to 12.4180
dΔ/dcollimator = +0.2140 from 2 point(s)
```

That is a straight-line solve through the two Δ values. It measures the sign and
the sensitivity together, so nothing is assumed. Set the collimator there, run
one confirming sweep, and you should see:

```
collimated — Δ is zero within 1σ
```

The camera micrometer's correct position is now the common minimum — either
vertex, they agree.

4. Click **💾 save converged settings**. They are written to
   `~/Library/Application Support/GhostSun/focus.txt` so you can verify against
   them later instead of re-deriving them.

If the solve says *extrapolated, expect one more iteration*, the root is outside
the two collimator settings you tried — set the collimator to the suggested
value, bank a third point, and it will bracket.

---

## Stage B — telescope

Select **B · telescope**. Only run this once Stage A is closed.

### Pick a metric

- **limb edge** — the solar limb crossing the slit, used as a knife edge. A limb
  is a full disk-to-sky step, so slit dust is negligible against it. **Trust
  this one.** It needs the limb on the slit, so point at the limb.
- **contrast** — high-passed detail along the slit. Always available, but slit
  dust adds a constant offset, so the curve is shallower than it looks like it
  should be. The peak is still in the right place. This is the only metric that
  offers the top/bottom check below.

Both are measured on **continuum** columns, never the line core — the core is
low-contrast chromosphere and scattered light flattens its curve. GhostSun picks
the continuum automatically; the **Continuum cut along the slit** plot shows
exactly what it is measuring.

### Sweep the focuser

The big number is a **best decile** over the last ~90 frames — it tracks your
optics rather than the seeing, so let it settle for a second or two before
capturing.

1. Type the focuser position into **focuser reading**.
2. **◉ capture**.
3. Step the focuser and repeat, again straddling the extremum with 3+ points.

**best focus** reports the vertex. Save when you are happy.

### Field curvature check (contrast metric only)

With **contrast** selected, GhostSun also fits the top and bottom thirds of the
slit separately. If it reports:

> top and bottom focus at different positions: field curvature or slit tilt

then no single focuser setting can be right for the whole slit. At around
1200 mm focal length a 7 mm slit spans roughly 20 arcmin — most of the solar
diameter — so this is worth checking once. The fix is a flattener, or accepting
a compromise focused for mid-disk radius. If the two agree, you can forget it
permanently.

---

## Reading the messages

| Message | Meaning |
|---|---|
| `not solved yet` | Fewer than 4 usable samples, or the curves are not yet parabolic |
| `curve bends the wrong way` | Your range is too small to see the extremum, or the metric has no signal |
| `extremum is extrapolated beyond the samples` | Step past the sharpest point and capture again — the answer is outside what you sampled |
| `a 4th sample gives an uncertainty` | It works with 3, but you get no ± until 4 |
| `no usable line in that burst` | Check exposure and that a line is actually detected |
| `no limb on the slit in that burst` | Point at the limb, or switch to **contrast** |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Δ will not go to zero | Collimator | Bank two sweeps and use the solved value |
| Both minima coincide but everything is soft | Camera lens | Camera micrometer — the common vertex |
| Sharp at one end of the frame, soft at the other | Tilted spectral focal plane | Camera tilt, or just optimise at your working wavelength |
| Lines sharp, reconstruction smeared **vertically** | Telescope focus | Stage B |
| Lines sharp, reconstruction smeared **horizontally** only | Scan rate, tracking, seeing | Not a focus problem |

## Routine

Stage A drifts far more slowly than Stage B — a short metal path with no tube.
So on a normal morning you only re-run Stage B, and check Stage A against your
saved readings occasionally. Do it at operating temperature, and re-check around
midday.
