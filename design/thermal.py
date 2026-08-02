#!/usr/bin/env python3
"""Thermal model for the SHG in sunlight, machined aluminium case.

Three effect classes, in order of what actually matters:
 1. Uniform soak (whole case at ambient + N kelvin): an all-aluminium
    bench holding aluminium-substrate OAPs is ATHERMAL to first order,
    because every spacing and every focal length scales by the same
    (1 + alpha dT). What survives is the grating: groove density scales
    with the SUBSTRATE CTE, so the tuned wavelength drifts.
 2. Gradients (one side hotter): differential rail expansion tilts
    elements against each other. This is the real risk; compare against
    the +3 um tolerance magnitudes from tolerance.py / fold_tol.py.
 3. Local loads: the slit jaws absorb the rejected solar image.

Assumptions: CTE aluminium 23e-6/K, BK7 7.1e-6/K, fused silica
0.52e-6/K; FSQ-85 at 65 mm stop delivers ~2.4 W into the solar image at
the slit plane (no front ERF; halve with one).
"""

import math

ALPHA_AL = 23e-6
ALPHA_BK7 = 7.1e-6
ALPHA_FS = 0.52e-6
LAM = 656.28e-9


def uniform_soak():
    print("=== 1. uniform soak ===")
    print("all-Al bench + Al mirrors: focus and alignment invariant "
          "(spacings and focal lengths scale together).")
    for name, a in (("BK7 (Shelyak 2400)", ALPHA_BK7),
                    ("fused silica (holographic)", ALPHA_FS)):
        d = 656.28e3 * a  # pm per K (lambda in pm times alpha)
        print(f"  grating on {name}: tuned line drifts {d:.2f} pm/K "
              f"({50/d:.0f} K per core width). GhostSun already "
              f"tracks the line: retune is a knob touch, and the drift "
              f"doubles as a free case thermometer.")
    print("  flats (fused silica substrate in Al cells): pointing drift "
          "only; fold_tol.py shows pointing DOFs cost no blur.\n")


def gradients():
    print("=== 2. side-to-side gradients (the real budget) ===")
    print("tilt between elements L apart on a case W wide, per kelvin of "
          "side-to-side difference: alpha * L / W")
    cases = [
        # (label, rail length mm, case width mm, tolerance deg for +3um)
        ("BUDGET  OAP2 pitch (Lc rail 150)", 150.0, 200.0, 0.027),
        ("BUDGET  grating clock (rotor seat)", 85.0, 200.0, 0.045),
        ("PROD    OAP1 pitch (collimated arm 330)", 330.0, 240.0, 0.050),
        ("PROD    OAP2 pitch (Lc rail 270)", 270.0, 240.0, 0.028),
        ("PROD    grating in-plane yaw (arm 330)", 330.0, 240.0, 0.060),
    ]
    print(f"{'DOF':42s} {'deg/K':>9s} {'allowable dT':>13s}")
    for (label, L, W, tol) in cases:
        dpk = math.degrees(ALPHA_AL * L / W)
        print(f"{label:42s} {dpk:9.4f} {tol/dpk:11.1f} K")
    print("Verdict: ~10-25 K of SUSTAINED side-to-side difference eats "
          "the +3 um budget. A dark case in full sun develops 5-15 K "
          "untreated, <3 K painted white / foil-wrapped. So: white or "
          "bare-metal exterior, no black anodize outside, and let it "
          "soak 15-20 min after slewing.\n")


def local_loads():
    print("=== 3. local loads ===")
    P_img = 2.4       # W into the solar image at the slit plane
    frac_slit = 0.01  # slit passes ~1% (7 um of a 4.2 mm disk scan zone)
    P_jaw = P_img * (1 - frac_slit)
    # slit tower: Al, ~14x20 mm section, 60 mm to the deck
    Rth = 0.060 / (170.0 * 14e-3 * 20e-3)
    dT = P_jaw * Rth
    despace = 60.0 * ALPHA_AL * dT * 1e3  # um
    print(f"  slit jaws absorb ~{P_jaw:.1f} W -> tower rises ~{dT:.1f} K "
          f"-> slit despace ~{despace:.0f} um (tolerance 210 um): fine.")
    print("  polished/tilted jaws (already 10 deg) reflect most of it "
          "back out the snout; an ERF halves everything.")
    print("  mirrors see mW after the slit: figure distortion is nm-class "
          "on aluminium substrates. Ignore.")
    print("  periscope column (production): a vertical gradient tilts the "
          "lift flats -> pure image shift (fold_tol: no blur). Benign.\n")


def recommendations():
    print("=== recommendations ===")
    for r in ("1. Machined Al case, all-Al mounts and OAPs: keeps the "
              "athermal property. Avoid steel standoffs in the beam "
              "train (CTE mismatch turns soak into tilt).",
              "2. Exterior: white paint or bare metal; never black.",
              "3. Thermal break (G10 washers or a printed collar) "
              "between the snout/telescope flange and the case front "
              "wall: the OTA is the hottest thing you touch.",
              "4. Use GhostSun's live line-position readout as the "
              "thermometer: 4.7 pm/K (BK7) means a 10-pm line drift "
              "flags a 2 K soak change before any blur is visible.",
              "5. Soak 15-20 min after setup; expect to touch the "
              "grating tune knob once per session, nothing else."):
        print("  " + r)


if __name__ == "__main__":
    uniform_soak()
    gradients()
    local_loads()
    recommendations()
