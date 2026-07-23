# GhostSun for macOS

GhostSun uses Metal through `wgpu`. The release workflow publishes separate
packages for Apple Silicon and Intel Macs.

## Run the packaged app

1. Download the package matching the Mac: **Apple Silicon** for M-series Macs,
   or **Intel** for older Intel Macs.
2. Extract the ZIP and move `GhostSun.app` to `Applications`.
3. Control-click `GhostSun.app`, choose **Open**, then confirm **Open**.

Open a SER, review the pipeline controls, and click **Process**. Once processing
finishes, **Orient from GONG** can use the SER's UTC timestamp and a downloaded
GONG H-alpha reference to feature-match the image into solar north-up,
east-left orientation. The optional match requires internet access; references
are cached in the macOS temporary directory.

CI packages are ad-hoc signed so their bundle is internally consistent, but
they are not Apple-notarized. The first launch can therefore show a Gatekeeper
warning. A notarized public distribution would additionally require an Apple
Developer ID certificate and notarization credentials stored as repository
secrets.

## Camera SDKs

The Focus view loads camera SDKs at runtime. The package does not redistribute
ToupTek's `libtoupcam.dylib` or ZWO's `libASICamera2.dylib`. GhostSun searches
the app's `Contents/Frameworks` directory, normal dynamic-library search paths,
and the KStars frameworks directory. Developers can also set
`GHOSTSUN_TOUPCAM_LIB` or `GHOSTSUN_ASI_LIB` to a full library path before
launching from a terminal.

## Build locally

Install Rust through [rustup](https://rustup.rs/), then run:

```sh
bash scripts/package-macos.sh aarch64-apple-darwin
```

Use `x86_64-apple-darwin` instead when building on an Intel Mac. Packages are
written to `dist/`.
