#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <GhostSun.app> <archive.zip>" >&2
  exit 2
fi

: "${APPLE_NOTARY_KEY_P8_BASE64:?Set APPLE_NOTARY_KEY_P8_BASE64}"
: "${APPLE_NOTARY_KEY_ID:?Set APPLE_NOTARY_KEY_ID}"
: "${APPLE_NOTARY_ISSUER_ID:?Set APPLE_NOTARY_ISSUER_ID}"

app="$1"
archive="$2"
stage="$(dirname "$app")"
notary_key="$(mktemp "${RUNNER_TEMP:-/tmp}/ghostsun-notary-key.XXXXXX.p8")"
trap 'rm -f "$notary_key"' EXIT

printf '%s' "$APPLE_NOTARY_KEY_P8_BASE64" \
  | openssl base64 -d -A \
  > "$notary_key"
chmod 600 "$notary_key"

xcrun notarytool submit "$archive" \
  --key "$notary_key" \
  --key-id "$APPLE_NOTARY_KEY_ID" \
  --issuer "$APPLE_NOTARY_ISSUER_ID" \
  --wait

xcrun stapler staple "$app"
xcrun stapler validate "$app"
codesign --verify --deep --strict --verbose=2 "$app"
spctl --assess --type execute --verbose=4 "$app"

# The ticket is stapled to the app after the original archive was submitted,
# so recreate the distributable ZIP with the stapled bundle.
rm -f "$archive"
ditto -c -k --sequesterRsrc --keepParent "$stage" "$archive"

echo "Notarized and stapled: $archive"
