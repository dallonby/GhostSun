#!/usr/bin/env bash
# One-command iteration loop against the field MacBook:
#   build + package -> rsync GhostSun.app (and CLI) to the field machine
#   -> optionally relaunch the app -> pull field data (scans/logs) back here.
#
# Usage:
#   FIELD_HOST=user@field-mbp.local ./scripts/deploy-field.sh
# Or put config in ~/.ghostsun-field (sourced if present):
#   FIELD_HOST=user@field-mbp.local     # ssh destination (LAN .local or Tailscale name)
#   FIELD_APP_DIR=Applications          # remote dir for GhostSun.app (relative to remote $HOME)
#   FIELD_DATA_DIR=GhostSunField        # remote dir whose contents rsync back to ./field-data
#   RELAUNCH=1                          # kill + reopen the app after push (default 0:
#                                       #   never yank the app mid-scan by accident)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$HOME/.ghostsun-field" ]] && source "$HOME/.ghostsun-field"

FIELD_HOST="${FIELD_HOST:?set FIELD_HOST (ssh destination for the field MacBook)}"
FIELD_APP_DIR="${FIELD_APP_DIR:-Applications}"
FIELD_DATA_DIR="${FIELD_DATA_DIR:-GhostSunField}"
RELAUNCH="${RELAUNCH:-0}"
target="aarch64-apple-darwin"

echo "==> build + package ($target)"
"$repo_root/scripts/package-macos.sh" "$target"

echo "==> build CLI"
cargo build --manifest-path "$repo_root/Cargo.toml" \
  --release --locked --package ghostsun-cli --target "$target"

app="$repo_root/dist/GhostSun-macOS-Apple-Silicon/GhostSun.app"
cli="$repo_root/target/$target/release/ghostsun"
[[ -d "$app" ]] || { echo "missing $app" >&2; exit 1; }

echo "==> push to $FIELD_HOST"
ssh "$FIELD_HOST" "mkdir -p '$FIELD_APP_DIR' '$FIELD_DATA_DIR/bin'"
rsync -a --delete "$app/" "$FIELD_HOST:$FIELD_APP_DIR/GhostSun.app/"
[[ -x "$cli" ]] && rsync -a "$cli" "$FIELD_HOST:$FIELD_DATA_DIR/bin/ghostsun"

if [[ "$RELAUNCH" == "1" ]]; then
  echo "==> relaunch on field machine"
  ssh "$FIELD_HOST" "pkill -x ghostsun-app 2>/dev/null || true; open \"\$HOME/$FIELD_APP_DIR/GhostSun.app\""
else
  echo "==> not relaunching (set RELAUNCH=1 to kill + reopen remotely)"
fi

echo "==> pull field data -> field-data/"
mkdir -p "$repo_root/field-data"
rsync -a --exclude '*.tmp' "$FIELD_HOST:$FIELD_DATA_DIR/" "$repo_root/field-data/" || true

echo "done: $(date '+%H:%M:%S')"
