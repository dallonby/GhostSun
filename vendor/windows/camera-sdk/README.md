# Bundled Windows camera SDKs

- `x64/toupcam.dll` — ToupTek ToupCam SDK 59.30701.20260128.

Only the x64 build is vendored: `scripts/package-windows.ps1` builds
`x86_64-pc-windows-msvc` exclusively, and offering the wrong architecture is
worse than offering none — loading an x86 DLL into an x64 process fails at
`dlopen` with a misleading error. The SDK also ships `arm64` and `x86` builds;
add them here alongside a matching packaging change if those targets are ever
built.

ZWO's `ASICamera2.dll` is **not** vendored. ToupTek's redistribution permission
(see `NOTICE-ToupTek.txt`) does not extend to ZWO, and no equivalent permission
has been recorded — on Windows the ZWO backend still resolves the DLL from a
system install. See `vendor/macos/camera-sdk/LICENSE-ZWO.txt` for the terms
that apply to the macOS bundle.
