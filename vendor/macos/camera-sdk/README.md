# Bundled macOS camera SDKs

GhostSun bundles these runtime-loaded camera SDK libraries in
`GhostSun.app/Contents/Frameworks`:

- `libASICamera2.dylib` — ZWO ASI SDK 1.41, universal `arm64` + `x86_64`.
- `libtoupcam.dylib` — ToupTek ToupCam SDK 59.30701.20260128, universal
  `arm64` + `x86_64`.

The packaging script extracts only the release target's architecture. ZWO also
requires a matching `libusb-1.0.0.dylib`, sourced from the build machine's
Homebrew installation. All nested code is signed before the application.

ZWO's SDK licence is in `LICENSE-ZWO.txt`. ToupTek redistribution permission
was confirmed to the project owner on 2026-07-23 and is recorded in
`NOTICE-ToupTek.txt`.
