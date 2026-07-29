# GhostSun for Windows 11

GhostSun runs natively on 64-bit Intel/AMD editions of Windows 11. The desktop
app uses Direct3D 12 or Vulkan through `wgpu`. Profile extraction, temporal NLM,
and geometric warping fall back to CPU implementations if compatible compute
is unavailable; the optional residual column-state stage is GPU-only and is
skipped.

## Run the packaged app

1. Download the `GhostSun-Windows-x64` artifact from the repository's latest
   **Desktop builds** workflow run, or download the ZIP from a tagged release.
2. Extract the downloaded ZIP completely.
3. Double-click `GhostSun.exe`.
4. Open or drag in a `.ser`, `.fits`, `.fit`, or `.png` file.
5. For SER data, review the pipeline settings and click **Process**.

After a timestamped SER has been processed, **Orient from GONG** can download
the nearest public GONG H-alpha reference and feature-match the result to solar
north-up, east-left. This optional action needs an internet connection and a
valid UTC timestamp in the SER header. References are cached under the current
Windows temporary directory.

The executable is currently unsigned, so Windows SmartScreen may show an
"unrecognized app" warning. Verify `GhostSun.exe` against `SHA256SUMS.txt`
before choosing **More info > Run anyway**.

The app does not need an installer or the Rust toolchain. Keep it anywhere you
can write files; loaded scans and exported images can be located elsewhere.

## ZWO mount control

The **Mount** tab controls ZWO AM-series mounts directly over their
LX200-compatible USB serial connection. Power on the mount and connect its
USB-B control port to the computer with a USB 2.0 data cable. Select the
resulting `COM` port, click **Connect**, and use the press-and-hold jog buttons.
Releasing a direction, changing rate, leaving the tab, disconnecting, or
closing GhostSun sends a stop command.

GhostSun scans Windows SetupAPI COM ports every three seconds and prioritizes
ZWO's USB vendor ID. If no ZWO port is found, expand **Mount not detected** for
a cable/power checklist and **Open ASI Mount**. Confirm that ASI Mount detects
the hardware, then disconnect or close it so the serial port is free for
GhostSun.

**Slew to the Sun** requires a separate confirmation and then requests solar
tracking. Before confirming, securely fit a suitable solar filter and check the
entire slew path. The mount rejects GoTo with error **e7** until time and site
coordinates are set: use the **Observing site** panel (latitude/longitude, UTC
offset, optional OpenStreetMap place search) and **Sync now**, or enable
**Sync time & site on connect**. Site settings are saved under
`%APPDATA%\GhostSun\mount_site.json`. Mechanical home position and
polar/alignment setup still require ASI Mount / the hand controller.

Camera-assisted centering offers two separately confirmed starts:
**GoTo + center** first slews to the calculated Sun, while **Center from here**
keeps the current pointing as the search origin. Both modes sample a bounded
0.2° coarse spiral, use the mean of the brightest one percent of camera pixels
as a hot-pixel-resistant peak signal, refine the strongest point on a 0.1°
grid, and return to that refined maximum. Set the search radius so every
possible nudge remains mechanically and optically safe. The Mount tab displays
the selected camera's live spectrum and can start or stop its preview directly;
during auto-center the Stop camera control is locked, while Cancel auto-center
stops mount motion and restores the previous camera state.

## Guided acquisition

The **Acquire** tab integrates the camera, ZWO mount, SER recording, default
high-quality reconstruction, and optional multi-scan stacking. It deliberately
starts from a prepared observing state: the filtered solar disc must already
be reasonably centred and the telescope and SHG focused. GhostSun displays
this requirement and will not start a scan until it is acknowledged.

Choose a detected horizontal spectral-line anchor and a vertical capture
height. GhostSun writes the resulting fixed crop as lossless mono16 SER; the
live preview remains bounded to the newest frame and cannot build a backlog or
cause recording frames to be discarded. A manual N/S/E/W scan direction can be
used, or **Auto-detect scan axis** makes small, reversible N/S and E/W probes
and compares slit-profile motion. It recommends the more nearly
slit-perpendicular axis and estimates sensor off-axis angle. An estimate above
10° produces a warning that must be acknowledged before acquisition.

At the default 60× sidereal rate, GhostSun pre-positions half the selected
span from the current (assumed centred) point, records pre-roll, scans across
the disc, and records post-roll. Multi-scan mode alternates direction without
an unnecessary return slew. Each SER is reconstructed independently; reverse
passes are flipped consistently, then registered and robust-stacked with
evolution compensation. The session folder retains every SER, individual
16-bit PNG/FITS reconstruction, and the final PNG/FITS product.

All automated motion requires both the prepared-observation and safe-motion
confirmations. **STOP**, leaving Acquire, a mount error, or a disconnect sends
a mount stop and closes any active SER.

## ToupTek cameras

The Focus view uses ToupTek's 64-bit SDK at runtime. GhostSun searches beside
`GhostSun.exe`, the system DLL paths, and common 64-bit N.I.N.A., SharpCap, and
ASCOM installations. Click **Refresh** in the Focus view after connecting the
camera; the status line now distinguishes a missing SDK from an SDK that loaded
but detected no hardware.

For another SDK location, set `GHOSTSUN_TOUPCAM_LIB` to the full path of the
64-bit `toupcam.dll` before starting GhostSun. Do not point it at the x86 DLL.
The standalone package does not redistribute the vendor SDK.

## QHYCCD cameras

QHY cameras (including **QHY5III678M**) use the official `qhyccd.dll` at
runtime with the Windows `__stdcall` ABI. GhostSun looks beside
`GhostSun.exe`, `GHOSTSUN_QHY_LIB`, and common
`%ProgramFiles%\QHYCCD\SDK\` (and `x64`) install paths. Install the [QHYCCD
SDK / drivers](https://www.qhyccd.com/download/), then **Refresh** cameras in
Focus or Mount. The Windows package does not redistribute the QHY SDK unless
you add `qhyccd.dll` next to the exe yourself under the vendor licence.

## Build on Windows

Install these prerequisites:

- Windows 11 x64
- [Rust through rustup](https://rustup.rs/)
- Visual Studio 2022 Build Tools with the **Desktop development with C++**
  workload and a Windows SDK

From PowerShell in the repository root, run:

```powershell
.\scripts\package-windows.ps1
```

The standalone package is written to `dist\GhostSun-Windows-x64.zip`. For a
developer build with a visible diagnostic console, use:

```powershell
cargo run --package ghostsun-app
```

## Graphics troubleshooting

Install the latest graphics driver from Intel, AMD, or NVIDIA if the window
cannot start or rendering is corrupted. Processing stages with CPU
implementations fall back when compute acceleration is unavailable; GPU-only
residual column-state correction is skipped. The desktop window itself still
requires a Direct3D 12- or Vulkan-capable driver.
