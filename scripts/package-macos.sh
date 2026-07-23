#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-$(rustc -vV | sed -n 's/^host: //p')}"
destination="${2:-dist}"
vendor_camera_dir="$repo_root/vendor/macos/camera-sdk"

case "$target" in
  aarch64-apple-darwin)
    default_package="GhostSun-macOS-Apple-Silicon"
    expected_arch="arm64"
    ;;
  x86_64-apple-darwin)
    default_package="GhostSun-macOS-Intel"
    expected_arch="x86_64"
    ;;
  *)
    echo "Unsupported macOS target: $target" >&2
    exit 2
    ;;
esac

package_name="${3:-$default_package}"
if [[ ! "$package_name" =~ ^GhostSun-[A-Za-z0-9._-]+$ ]]; then
  echo "Unsafe package name: $package_name" >&2
  exit 2
fi
if [[ "$destination" != /* ]]; then
  destination="$repo_root/$destination"
fi
stage="$destination/$package_name"
app="$stage/GhostSun.app"
archive="$destination/$package_name.zip"
frameworks="$app/Contents/Frameworks"
third_party_licenses="$app/Contents/Resources/ThirdPartyLicenses"
executable="$repo_root/target/$target/release/ghostsun-app"
version="$(
  awk '
    /^\[workspace.package\]$/ { in_package = 1; next }
    in_package && /^version = / {
      gsub(/"/, "", $3)
      print $3
      exit
    }
  ' "$repo_root/Cargo.toml"
)"

rustup target add "$target"
cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  --release \
  --locked \
  --package ghostsun-app \
  --target "$target"

if [[ ! -x "$executable" ]]; then
  echo "macOS executable not found at $executable" >&2
  exit 1
fi

actual_archs="$(lipo -archs "$executable")"
if [[ " $actual_archs " != *" $expected_arch "* ]]; then
  echo "Expected $expected_arch executable, found: $actual_archs" >&2
  exit 1
fi

mkdir -p "$destination"
rm -rf "$stage"
rm -f "$archive"
mkdir -p "$app/Contents/MacOS" "$frameworks" "$third_party_licenses"
cp "$executable" "$app/Contents/MacOS/GhostSun"
chmod 755 "$app/Contents/MacOS/GhostSun"
cp "$repo_root/docs/macos.md" "$stage/README-macOS.md"

zwo_source="$vendor_camera_dir/libASICamera2.dylib"
touptek_source="$vendor_camera_dir/libtoupcam.dylib"
for sdk in "$zwo_source" "$touptek_source"; do
  if [[ ! -f "$sdk" ]]; then
    echo "Bundled camera SDK is missing: $sdk" >&2
    exit 1
  fi
done

# Keep release packages architecture-specific even though the checked-in SDK
# dylibs are universal. ZWO also needs a matching libusb alongside it.
lipo "$zwo_source" -thin "$expected_arch" -output "$frameworks/libASICamera2.dylib"
lipo "$touptek_source" -thin "$expected_arch" -output "$frameworks/libtoupcam.dylib"

libusb_source="${MACOS_LIBUSB_DYLIB:-}"
if [[ -z "$libusb_source" ]] && command -v brew >/dev/null 2>&1; then
  libusb_prefix="$(brew --prefix libusb 2>/dev/null || true)"
  if [[ -n "$libusb_prefix" ]]; then
    libusb_source="$libusb_prefix/lib/libusb-1.0.0.dylib"
  fi
fi
if [[ ! -f "$libusb_source" ]]; then
  echo "libusb dylib not found; install libusb or set MACOS_LIBUSB_DYLIB" >&2
  exit 1
fi
if [[ " $(lipo -archs "$libusb_source") " != *" $expected_arch "* ]]; then
  echo "libusb does not contain $expected_arch: $libusb_source" >&2
  exit 1
fi
cp "$libusb_source" "$frameworks/libusb-1.0.0.dylib"

chmod 755 "$frameworks"/*.dylib
install_name_tool -id @rpath/libASICamera2.dylib "$frameworks/libASICamera2.dylib"
install_name_tool -id @rpath/libtoupcam.dylib "$frameworks/libtoupcam.dylib"
install_name_tool -id @rpath/libusb-1.0.0.dylib "$frameworks/libusb-1.0.0.dylib"

cp "$vendor_camera_dir/LICENSE-ZWO.txt" "$third_party_licenses/ZWO-ASI-SDK.txt"
cp "$vendor_camera_dir/NOTICE-ToupTek.txt" "$third_party_licenses/ToupTek-ToupCam-SDK.txt"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>GhostSun</string>
  <key>CFBundleExecutable</key>
  <string>GhostSun</string>
  <key>CFBundleIdentifier</key>
  <string>io.github.dallonby.ghostsun</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>GhostSun</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>$version</string>
  <key>CFBundleVersion</key>
  <string>$version</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.photography</string>
  <key>LSMinimumSystemVersion</key>
  <string>11.0</string>
  <key>NSCameraUsageDescription</key>
  <string>GhostSun uses the selected astronomy camera for its live focus assistant.</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

plutil -lint "$app/Contents/Info.plist"

# Downloads and copied vendor SDKs can carry Finder or quarantine metadata.
# Such extended attributes are forbidden in a Developer ID-signed bundle.
xattr -cr "$app"

signing_identity="${MACOS_SIGNING_IDENTITY:--}"
if [[ "$signing_identity" == "-" ]]; then
  nested_sign_args=(--force --sign -)
  app_sign_args=(--force --sign -)
else
  nested_sign_args=(--force --sign "$signing_identity" --options runtime --timestamp)
  app_sign_args=(--force --sign "$signing_identity" --options runtime --timestamp)
fi

# Sign nested code from the inside out. Do not use --deep for signing: it can
# conceal missing or incorrectly signed bundle components.
for code in \
  "$frameworks/libusb-1.0.0.dylib" \
  "$frameworks/libASICamera2.dylib" \
  "$frameworks/libtoupcam.dylib"
do
  codesign "${nested_sign_args[@]}" "$code"
  codesign --verify --strict --verbose=2 "$code"
done

codesign "${app_sign_args[@]}" "$app"
codesign --verify --deep --strict "$app"

if [[ "$signing_identity" != "-" ]]; then
  signature_details="$(codesign -dv --verbose=4 "$app" 2>&1)"
  if [[ "$signature_details" != *"Authority=Developer ID Application:"* ]]; then
    echo "Expected a Developer ID Application signature" >&2
    exit 1
  fi
  if [[ "$signature_details" != *"flags=0x10000(runtime)"* ]]; then
    echo "Expected hardened runtime signing" >&2
    exit 1
  fi
fi

hash="$(shasum -a 256 "$app/Contents/MacOS/GhostSun" | awk '{print $1}')"
printf '%s  %s\n' "$hash" "GhostSun.app/Contents/MacOS/GhostSun" > "$stage/SHA256SUMS.txt"
ditto -c -k --sequesterRsrc --keepParent "$stage" "$archive"

echo "macOS package: $archive"
echo "Executable architecture: $actual_archs"
echo "Executable SHA-256: $hash"
