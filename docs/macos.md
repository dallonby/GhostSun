# GhostSun for macOS

GhostSun uses Metal through `wgpu`. The release workflow publishes separate
packages for Apple Silicon and Intel Macs.

## Run the packaged app

1. Download the package matching the Mac: **Apple Silicon** for M-series Macs,
   or **Intel** for older Intel Macs.
2. Extract the ZIP and move `GhostSun.app` to `Applications`.
3. Open `GhostSun.app`.

Open a SER, review the pipeline controls, and click **Process**. Once processing
finishes, **Orient from GONG** can use the SER's UTC timestamp and a downloaded
GONG H-alpha reference to feature-match the image into solar north-up,
east-left orientation. The optional match requires internet access; references
are cached in the macOS temporary directory.

Tagged releases are signed with GhostSun's Developer ID Application identity,
submitted to Apple's notary service, and distributed with the notarization
ticket stapled to the app. Ordinary branch and pull-request artifacts use an
ad-hoc signature because release credentials are deliberately unavailable to
untrusted builds; Gatekeeper may warn when opening those CI-only artifacts.

### Release signing setup

The repository needs these GitHub Actions secrets before a version tag is
pushed:

- `MACOS_CERTIFICATE_P12_BASE64`: the Developer ID Application certificate and
  private key exported from Keychain Access as a password-protected `.p12`,
  then base64-encoded.
- `MACOS_CERTIFICATE_PASSWORD`: the password chosen during the `.p12` export.
- `APPLE_NOTARY_KEY_P8_BASE64`: an App Store Connect API private key,
  base64-encoded.
- `APPLE_NOTARY_KEY_ID`: the API key ID.
- `APPLE_NOTARY_ISSUER_ID`: the App Store Connect issuer ID.

The workflow imports the signing identity into a temporary keychain, signs the
libraries before the app with the hardened runtime enabled, submits each ZIP to
Apple, staples the returned ticket, and rebuilds the downloadable archive.

## Camera SDKs

The packaged app includes the ToupTek and ZWO camera SDKs in
`Contents/Frameworks`, together with the `libusb` dependency required by ZWO.
Each library is reduced to the package's target architecture and signed before
the app itself. GhostSun can still use an explicitly selected development
library by setting `GHOSTSUN_TOUPCAM_LIB` or `GHOSTSUN_ASI_LIB` to its full path
before launching from a terminal.

## Build locally

Install Rust through [rustup](https://rustup.rs/), then run:

```sh
bash scripts/package-macos.sh aarch64-apple-darwin
```

Use `x86_64-apple-darwin` instead when building on an Intel Mac. Packages are
written to `dist/`.
