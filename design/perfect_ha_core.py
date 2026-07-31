#!/usr/bin/env python3
"""Shared physics helpers for the PERFECT_HA design study (new file; does
not modify any existing tool). Import raytrace from the same directory.

Conventions match raytrace.py: slit at origin, feed along +z (telescope at
z<0), fold/dispersion plane = y-z, slit length along x. All mm / deg.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from raytrace import Design, stats, dot, sub, add, mul, norm, cross

LAM_HA = 656.28          # nm
SIGMA_2400 = 1.0 / 2400.0  # mm
PIX_571 = 3.76e-3        # mm
PIX_678 = 2.00e-3        # mm


# ---------------------------------------------------------------- geometry
def alpha_beta(d):
    """Incidence/diffraction angles (deg) and cosines from a built Design."""
    n = d.gr.n
    ca = abs(dot(d.c1, n))
    cb = abs(dot(d.c2, n))
    return math.degrees(math.acos(ca)), math.degrees(math.acos(cb)), ca, cb


def grating_capture(f1, fnum, w_um, Wg_mm, ca, lam_nm=LAM_HA,
                    ngrid=40001, span=0.45, walk_mm=0.0):
    """1-D dispersion-plane wave pass (same model as fourier.py):
    angular intensity = tophat(f/#) conv sinc^2(slit w), clipped by the
    grating's projected width Wg*cos(alpha) (optionally reduced by a
    field-dependent footprint walk). Returns (captured fraction,
    effective clipped pupil width at grating, mm)."""
    lam = lam_nm * 1e-6
    w = w_um * 1e-3
    th = np.linspace(-span, span, ngrid)
    dth = th[1] - th[0]
    geo = (np.abs(th) <= 1.0 / (2 * fnum)).astype(float)
    x = np.pi * w * th / lam
    s = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    s[nz] = (np.sin(x[nz]) / x[nz]) ** 2
    s /= s.sum() * dth
    I = np.convolve(geo, s, mode="same") * dth
    I /= I.sum() * dth
    half = max(0.0, (Wg_mm * ca - walk_mm) / 2.0)
    m = np.abs(th * f1) <= half
    cap = float(I[m].sum() * dth)
    return cap, 2.0 * half


def delivered_lsf(f1, f2, fnum, w_um, Wg_mm, ca, cb, pix_mm,
                  lam_nm=LAM_HA, colCA_mm=None):
    """Delivered line-spread in the dispersion axis at the sensor.
    LSF = tophat(geometric slit image, anamorphic) conv PSF(clipped pupil)
          conv tophat(pixel).
    PSF of the clipped pupil is modelled as sinc^2 of the effective camera-
    side pupil D_cam = D_clip * cb/ca (fourier.py model, full convolution
    instead of quadrature). Returns dict with widths (um and pm), capture.
    """
    lam = lam_nm * 1e-6
    anam = ca / cb
    w_img = w_um * 1e-3 * (f2 / f1) * anam        # geometric slit image, mm
    cap, d_clip = grating_capture(f1, fnum, w_um, Wg_mm, ca, lam_nm)
    if colCA_mm is not None:
        d_clip = min(d_clip, colCA_mm)
    d_cam = d_clip * cb / ca
    # grid (capped for speed; resolution still ~w_img/12)
    dx = min(w_img, pix_mm) / 12.0
    span = max(6 * w_img, 25 * lam * f2 / max(d_cam, 1.0), 6 * pix_mm)
    if 2 * span / dx > 12000:
        dx = 2 * span / 12000.0
    x = np.arange(-span, span, dx)
    slit = (np.abs(x) <= w_img / 2).astype(float)
    u = np.pi * d_cam * x / (lam * f2)
    psf = np.ones_like(x)
    nz = np.abs(u) > 1e-12
    psf[nz] = (np.sin(u[nz]) / u[nz]) ** 2
    pixk = (np.abs(x) <= pix_mm / 2).astype(float)
    lsf = np.convolve(np.convolve(slit, psf, mode="same"), pixk, mode="same")
    lsf /= lsf.max()
    # FWHM
    above = np.where(lsf >= 0.5)[0]
    fwhm_mm = x[above[-1]] - x[above[0]]
    # wavelength scale: dλ/dx = σ cosβ / f2  (m=1, 2400 l/mm)
    dldx = SIGMA_2400 * cb / f2                    # mm(λ)/mm(x)
    fwhm_pm = fwhm_mm * dldx * 1e9                 # mm -> pm
    # energy fraction of LSF beyond ±2×FWHM (far-wing / purity proxy)
    e = lsf / lsf.sum()
    wing = float(e[np.abs(x) > 2.0 * fwhm_mm].sum())
    return dict(w_img_um=w_img * 1e3, cap=cap, d_clip=d_clip, d_cam=d_cam,
                fwhm_um=fwhm_mm * 1e3, fwhm_pm=fwhm_pm,
                dl_px_pm=pix_mm * dldx * 1e9, R=LAM_HA * 1000.0 / fwhm_pm,
                wing_frac=wing, anam=anam)


# ------------------------------------------------------------- collisions
def scope_obstacle_radius(z):
    """Radius of the telescope-side exclusion cylinder vs z (mm).
    Mirrors BODY.md: printed snout near the slit, then drawtube/focuser,
    then the Ø95 OTA further out. z=0 is the slit; telescope at z<0."""
    if z > -70.0:
        return 32.0     # snout tube region
    if z > -170.0:
        return 42.0     # drawtube + focuser barrel
    return 48.0         # Ø95 OTA

def camera_exit(d, cam_dia=80.0, cam_len=108.0, flange_dia=100.0):
    """Camera-vs-telescope collision figure of merit.
    Camera body = cylinder cam_dia (flange_dia for first 12 mm) from the
    sensor F2 extending along +df (behind the sensor plane) by cam_len.
    Returns (df angle to telescope axis deg, min radial margin mm over the
    part of the camera that overlaps z<0, F2, df). Margin is +inf if the
    camera never enters z<0."""
    F2, df = d.F2, d.df
    axis_ang = math.degrees(math.acos(max(-1, min(1, abs(df[2])))))
    worst = float("inf")
    for t in np.linspace(0.0, cam_len, 60):
        p = add(F2, mul(df, float(t)))
        if p[2] < 0.0:
            r = math.hypot(p[0], p[1])
            body_r = flange_dia / 2 if t < 12.0 else cam_dia / 2
            worst = min(worst, r - body_r - scope_obstacle_radius(p[2]))
    return axis_ang, worst, F2, df


def oap1_return_clearance(d):
    """Perpendicular distance from OAP1 centre to the diffracted chief line
    (raytrace.run() convention)."""
    off = sub(d.C1, d.G)
    return math.sqrt(max(dot(off, off) - dot(off, d.c2) ** 2, 0.0))


def slit_corridor_clearance(d):
    """Distance from the slit (origin) to the collimated corridor line
    C1 + t*c1 (how far the OAP1->grating beam passes from the slit)."""
    off = mul(d.C1, -1.0)  # origin - C1
    return math.sqrt(max(dot(off, off) - dot(off, d.c1) ** 2, 0.0))


def _pt_line(p, a, u):
    """Distance of point p from line a + t*u (u unit)."""
    off = sub(p, a)
    return math.sqrt(max(dot(off, off) - dot(off, u) ** 2, 0.0))


def oap2_clearances(d):
    """(OAP2 centre vs collimator->grating corridor,
        OAP2 centre vs slit->OAP1 cone axis,
        sensor F2 vs grating->OAP2 beam corridor)."""
    return (_pt_line(d.C2, d.C1, d.c1),
            _pt_line(d.C2, d.S, d.c0),
            _pt_line(d.F2, d.G, d.c2))


def trace_with_chief(d, xf, lam_nm, fnum=6.9, nring=8, nseg=16,
                     pupil_dist=None):
    """Like Design.trace but the cone axis tilts toward an entrance-pupil
    image at distance pupil_dist behind the slit (None = telecentric).
    pupil_dist>0 models the real telescope pupil 450 mm upstream, or the
    field-lens-corrected pupil position (negative of image side)."""
    lam = lam_nm * 1e-6
    pts = []
    na = 1.0 / (2.0 * fnum)
    offs = [(0.0, 0.0)]
    for i in range(1, nring + 1):
        r = na * i / nring
        for j in range(nseg):
            a = 2 * math.pi * j / nseg
            offs.append((r * math.cos(a), r * math.sin(a)))
    o0 = (xf, 0.0, 0.0)
    tx = 0.0 if pupil_dist is None else -xf / pupil_dist
    for (ax, ay) in offs:
        v = (ax + tx, ay, 1.0)
        dvec = norm(v)
        r1 = d.oap1.intersect_reflect(o0, dvec)
        if r1 is None:
            continue
        r2 = d.gr.intersect_diffract(r1[0], r1[1], lam)
        if r2 is None:
            continue
        r3 = d.oap2.intersect_reflect(r2[0], r2[1])
        if r3 is None:
            continue
        o, dd = r3
        dn = dot(dd, d.sens_n)
        if abs(dn) < 1e-15:
            continue
        t = dot(sub(d.F2, o), d.sens_n) / dn
        hit = add(o, mul(dd, t))
        q = sub(hit, d.F2)
        pts.append((dot(q, d.sens_x), dot(q, d.sens_y)))
    return pts


def blur_map(d, fields, dlams, lam0=LAM_HA, fnum=6.9, pupil_dist=None):
    """RMS spot radii table: {(field, dlam): (sx_um, sy_um, cx, cy)}."""
    out = {}
    for xf in fields:
        for dl in dlams:
            pts = trace_with_chief(d, xf, lam0 + dl, fnum,
                                   pupil_dist=pupil_dist)
            if not pts:
                out[(xf, dl)] = None
                continue
            cx, cy, rx, ry = stats(pts)
            out[(xf, dl)] = (rx * 1e3, ry * 1e3, cx, cy)
    return out


def build(rfl1, th1, rfl2, th2, dev, s2, Lg, Lc, lam0=LAM_HA,
          lines_per_mm=2400.0, order=1):
    d = Design(rfl1=rfl1, th1=th1, rfl2=rfl2, th2=th2,
               lines_per_mm=lines_per_mm, order=order,
               dev=dev, s2=s2, Lg=Lg, Lc=Lc)
    d.build(lam0)
    return d
