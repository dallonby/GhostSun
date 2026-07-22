# GhostSun for Windows 11

GhostSun runs natively on 64-bit Intel/AMD editions of Windows 11. The desktop
app uses Direct3D 12 or Vulkan through `wgpu`. Profile extraction, temporal NLM,
and geometric warping fall back to CPU implementations if compatible compute
is unavailable; the optional residual column-state stage is GPU-only and is
skipped.

## Run the packaged app

1. Download the `GhostSun-Windows-x64` artifact from the repository's latest
   **Windows 11 build** workflow run.
2. Extract the downloaded ZIP completely.
3. Double-click `GhostSun.exe`.
4. Open or drag in a `.ser`, `.fits`, `.fit`, or `.png` file.

The executable is currently unsigned, so Windows SmartScreen may show an
"unrecognized app" warning. Verify `GhostSun.exe` against `SHA256SUMS.txt`
before choosing **More info > Run anyway**.

The app does not need an installer or the Rust toolchain. Keep it anywhere you
can write files; loaded scans and exported images can be located elsewhere.

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
