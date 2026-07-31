#!/usr/bin/env python3
"""PERFECT_HA step 1 — derive the quantitative requirements from first
principles: seeing/diffraction, slit trade, grating-limited resolution
ceiling, photon/full-well/scan budget. Pure calculation, no raytrace."""
import math

LAM = 656.28e-9            # m
FL = 450.0                 # mm telescope focal length
DISK = 4.2                 # mm solar image diameter
ARCSEC = math.pi / 180 / 3600
SCALE = FL * ARCSEC * 1e3  # um per arcsec at the slit  (450mm -> 2.182 um)

print("=== spatial: seeing + diffraction at the slit ===")
print(f"plate scale: {SCALE:.3f} um/arcsec ; disk {DISK} mm = "
      f"{DISK*1e3/SCALE:.0f} arcsec")
for D in (65.0, 85.0):
    dif = 1.02 * LAM / (D * 1e-3) / ARCSEC   # FWHM arcsec
    for see500 in (2.0, 1.5):
        # r0 scaling 500 -> 656 nm: seeing FWHM ~ lambda^-0.2
        see = see500 * (656.28 / 500.0) ** -0.2
        comb = math.hypot(see, dif)
        print(f"  D={D:3.0f} mm: diffr {dif:4.2f}\" ; seeing(500nm "
              f"{see500}\")->656nm {see:4.2f}\" ; combined {comb:4.2f}\" "
              f"= {comb*SCALE:4.1f} um at slit")
print("  short-exposure (ms frames): blur ~ diffraction when D <~ r0;")
for see500 in (2.0, 1.5):
    r0_500 = 0.98 * 500e-9 / (see500 * ARCSEC)
    r0 = r0_500 * (656.28 / 500.0) ** 1.2
    print(f"    seeing {see500}\" @500nm: r0(656nm) = {r0*1e3:.0f} mm")

print("\n=== slit width trade (scan-axis sampling & spectral) ===")
SIG = 1.0 / 2400.0         # mm
for w in (5.0, 7.0, 10.0):
    ang = w / SCALE
    print(f"  slit {w:4.1f} um = {ang:4.2f} arcsec on sky")

print("\n=== grating-limited resolution ceiling ===")
print("with f_col sized to capture the FULL first diffraction lobe")
print("(D_col = f_col*(1/f# + 2*lam/w)), the slit-limited bandwidth is")
print("dLam = sigma*(w/f# + 2*lam)/W_g   -- independent of alpha:")
lam_mm = LAM * 1e3
for Wg, label in ((50.0, "GH50-24V 50mm"), (25.0, "Shelyak 25mm"),
                  (68.0, "custom 68mm")):
    for fnum, D in ((6.9, 65), (5.3, 85)):
        for w_um in (5.0, 7.0, 10.0):
            w = w_um * 1e-3
            dl = SIG * (w / fnum + 2 * lam_mm) / Wg   # mm
            dl_pm = dl * 1e9
            print(f"  {label:15s} f/{fnum} ({D}mm) slit {w_um:4.1f}um: "
                  f"dLam_slit >= {dl_pm:5.1f} pm  (R <= "
                  f"{656.28e3/dl_pm:,.0f})")

print("\n=== target derivation ===")
print("Ha chromospheric core FWHM ~ 50 pm; Doppler 20-50 km/s = 44-109 pm.")
print("Contrast saturates once delivered FWHM <~ core/2 = 25 pm (R ~ 26k).")
print("=> requirement: delivered FWHM <= 30 pm (R >= 22k); goal <= 25 pm")
print("   (R >= 26k); stretch 17 pm (R 38k) via 5 um slit mode.")

print("\n=== photon / full-well / scan budget ===")
E656 = 1.4                 # W/m^2/nm ground-level irradiance near 656 nm
QE = 0.80
EPH = 6.626e-34 * 2.998e8 / LAM  # J
CORE = 0.16                # Ha core residual intensity vs continuum
T_TEL, T_FILT, T_SPEC = 0.95, 0.90, 0.55  # telescope, prefilter, slit->sensor
for D, fnum in ((65.0, 6.9), (85.0, 5.3)):
    A = math.pi * (D * 1e-3 / 2) ** 2
    for w_um in (5.0, 7.0):
        P_nm = E656 * A * (w_um * 1e-3 * DISK) / (math.pi * (DISK/2)**2)
        ph_nm = P_nm / EPH                     # photons/s/nm through slit
        # per spatial resel (2" = 4.36 um along slit), per 20 pm
        resel = 2.0 * SCALE * 1e-3             # mm
        ph = ph_nm * (resel / DISK) * 0.020
        sysT = T_TEL * T_FILT * T_SPEC * QE
        e_res = ph * sysT
        # per pixel: resel ~ 3.2 px spatial x 2.2 px spectral (IMX571 map)
        e_px = e_res / 7.0
        t_fw = 51000 / (e_px / CORE**0)        # continuum-limited (v=1)
        print(f"  D={D:3.0f} slit {w_um:3.1f}um: continuum "
              f"{e_res:,.3e} e-/s/resel(2\"x20pm); e-/px/ms "
              f"{e_px*1e-3:,.0f}; full-well(51ke) at continuum in "
              f"{51000/(e_px)*1e3:.2f} ms; core SNR/resel in 1 ms "
              f"{math.sqrt(e_res*CORE*1e-3):,.0f}")
print("  prominences ~1-5% of disk continuum -> SNR 15-40 per 1 ms resel;")
print("  scan: 4.2 mm disk / 7 um steps = 600 frames; / 3.5 um = 1200.")
print("  IMX571 ROI ~6248x256 @ ~15-25 fps (USB3) -> 25-80 s per scan.")
print("  IMX678 ROI ~3856x200 @ >200 fps -> 3-6 s per scan.")
print("=> photons are NOT the limiting currency on-disk; exposure is")
print("   full-well limited at ~1 ms. Optimize for delivered resolution,")
print("   purity and blur; throughput matters only for prominence SNR.")
