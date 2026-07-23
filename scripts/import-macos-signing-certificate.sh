#!/usr/bin/env bash
set -euo pipefail

: "${MACOS_CERTIFICATE_P12_BASE64:?Set MACOS_CERTIFICATE_P12_BASE64}"
: "${MACOS_CERTIFICATE_PASSWORD:?Set MACOS_CERTIFICATE_PASSWORD}"
: "${GITHUB_ENV:?This script is intended for GitHub Actions}"

keychain="$RUNNER_TEMP/ghostsun-signing.keychain-db"
certificate="$RUNNER_TEMP/ghostsun-developer-id.p12"
keychain_password="$(openssl rand -hex 24)"
trap 'rm -f "$certificate"' EXIT

printf '%s' "$MACOS_CERTIFICATE_P12_BASE64" \
  | openssl base64 -d -A \
  > "$certificate"

security create-keychain -p "$keychain_password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$keychain_password" "$keychain"
security import "$certificate" \
  -k "$keychain" \
  -P "$MACOS_CERTIFICATE_PASSWORD" \
  -T /usr/bin/codesign \
  -T /usr/bin/security
security set-key-partition-list \
  -S apple-tool:,apple: \
  -s \
  -k "$keychain_password" \
  "$keychain"
security list-keychains -d user -s "$keychain" login.keychain-db

identity="$(
  security find-identity -v -p codesigning "$keychain" \
    | sed -n 's/.*"\(Developer ID Application:.*\)"/\1/p' \
    | head -n 1
)"
if [[ -z "$identity" ]]; then
  echo "Developer ID Application identity was not found in the imported certificate" >&2
  exit 1
fi

{
  printf 'MACOS_SIGNING_IDENTITY=%s\n' "$identity"
  printf 'MACOS_SIGNING_KEYCHAIN=%s\n' "$keychain"
} >> "$GITHUB_ENV"

echo "Imported: $identity"
