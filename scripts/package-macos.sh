#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-$(rustc -vV | sed -n 's/^host: //p')}"
destination="${2:-dist}"

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
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$executable" "$app/Contents/MacOS/GhostSun"
chmod 755 "$app/Contents/MacOS/GhostSun"
cp "$repo_root/docs/macos.md" "$stage/README-macOS.md"

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
codesign --force --deep --sign - "$app"
codesign --verify --deep --strict "$app"

hash="$(shasum -a 256 "$app/Contents/MacOS/GhostSun" | awk '{print $1}')"
printf '%s  %s\n' "$hash" "GhostSun.app/Contents/MacOS/GhostSun" > "$stage/SHA256SUMS.txt"
ditto -c -k --sequesterRsrc --keepParent "$stage" "$archive"

echo "macOS package: $archive"
echo "Executable architecture: $actual_archs"
echo "Executable SHA-256: $hash"
