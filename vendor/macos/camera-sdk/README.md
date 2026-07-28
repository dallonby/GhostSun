# Bundled macOS camera SDKs

GhostSun bundles these runtime-loaded camera SDK libraries in
`GhostSun.app/Contents/Frameworks`:

- `libASICamera2.dylib` — ZWO ASI SDK 1.41, universal `arm64` + `x86_64`.
- `libtoupcam.dylib` — ToupTek ToupCam SDK 59.30701.20260128, universal
  `arm64` + `x86_64`.

**Optional — QHYCCD:** place `libqhyccd.dylib` (from the [QHYCCD SDK](
https://www.qhyccd.com/download/)) in this directory or install the SDK
system-wide. Packaging copies it when present; otherwise the QHY backend is
compiled in but reports no devices until the library is found at runtime via
`GHOSTSUN_QHY_LIB`, `Contents/Frameworks/libqhyccd.dylib`, or
`/usr/local/lib` / `/opt/homebrew/lib`. Verified target: **QHY5III678M**
(mono live stream, 16-bit).

The packaging script extracts only the release target's architecture. ZWO also
requires a matching `libusb-1.0.0.dylib`, sourced from the build machine's
Homebrew installation. All nested code is signed before the application.

ZWO's SDK licence is in `LICENSE-ZWO.txt`. ToupTek redistribution permission
was confirmed to the project owner on 2026-07-23 and is recorded in
`NOTICE-ToupTek.txt`. QHY SDK redistribution is governed by QHYCCD's own
licence — do not commit proprietary QHY dylibs unless redistribution is
explicitly allowed.

## Code signatures are load-bearing — do not commit a stripped dylib

macOS refuses to load an unsigned arm64 library into a process, failing with
`Trying to load an unsigned library`. These dylibs must therefore carry a valid
signature **in the repository**, not merely after packaging.

`lipo` silently drops code signatures. Both bundled dylibs were committed in
that state for a while, which made them loadable only from a packaged,
re-signed `.app`; a plain `cargo run` or `target/release/ghostsun-app`
reported the library as missing — with the developer's own machine appearing to
have no camera at all.

Verify before committing:

```sh
codesign -dv vendor/macos/camera-sdk/libtoupcam.dylib
```

It must report `Signature=adhoc` (as shipped by the vendor) or a real identity —
never `code object is not signed at all`. If a dylib has been thinned or
recombined, re-sign it before committing:

```sh
codesign -s - -f vendor/macos/camera-sdk/libASICamera2.dylib
```

Prefer re-copying the vendor's pristine file over re-signing: ToupTek ships
theirs already ad-hoc/linker-signed, so a straight copy keeps it byte-identical
to the upstream release.
