# GhostSun macOS feature handover

Continue work on the GhostSun ZWO mount-control feature from this pushed WIP checkpoint:

- Repository: `https://github.com/dallonby/GhostSun.git`
- Branch: `feature/zwo-mount-control`
- Initial WIP commit: `3c50db0` (`WIP: add cross-platform ZWO mount controls`)

On the Mac:

```bash
git clone https://github.com/dallonby/GhostSun.git
cd GhostSun
git switch feature/zwo-mount-control
```

This is explicitly a **work-in-progress checkpoint, not a finished or currently compiling feature**. Do not restart the implementation from scratch and do not assume the auto-centering code is complete.

## Work already present

- A new Mount tab and cross-platform direct USB serial control for ZWO AM-series mounts, avoiding an ASCOM dependency.
- Windows serial discovery and macOS `/dev/cu.*` discovery, plus platform-specific troubleshooting prompts and launch helpers.
- Connection/status display, press-and-hold N/S/E/W jogging, ten ZWO protocol speed settings, STOP, Go Home, Park, Unpark, and a confirmed Sun GoTo.
- The beginning of camera-assisted Sun centering: selection of a hardware camera, temporary exposure control (default 250 ms), signal sampling, timed mount nudges, and WIP scan-state types.
- Updated Windows and macOS documentation.

## Hardware and protocol findings

- The connected mount enumerated on Windows as `COM3`, USB VID `03C3`, PID `4001`.
- `:GVP#` returned `AM5N#`.
- Read-only RA, Dec, altitude, azimuth, and status queries worked at 9600 baud.
- No live movement commands were issued during validation.
- Go Home, Park, Unpark, Sun GoTo, jogging, and timed nudges still require careful physical testing. Do not move the mount unless the user explicitly confirms it is safe.

## Files to inspect first

- `crates/ghostsun-app/src/mount.rs`
- `crates/ghostsun-app/src/focus.rs`
- `crates/ghostsun-app/src/main.rs`
- `docs/macos.md`
- `docs/windows.md`

The immediate compile failure is intentional WIP: `MountState` calls three methods that have not yet been implemented:

- `cancel_auto_center`
- `auto_center_nudge_done`
- `advance_auto_center`

## Finish the camera-assisted centering workflow

It should:

1. Require an explicit safety confirmation before starting.
2. Raise the selected hardware camera exposure to the chosen value (default 250 ms), remembering all settings that need restoration.
3. Slew to the calculated Sun position, wait for the slew and settling state, then sample image brightness.
4. Perform a bounded expanding square-spiral scan using timed mount nudges, recording the signal at each point.
5. Return to the strongest sampled position and stop.
6. Show clear phase, progress, and status in the UI.
7. Cancel immediately on STOP, disconnect, tab exit, camera failure, or mount error; stop all movement and restore the camera state.
8. Enforce a maximum radius and duration, and never continue blindly after an error.

Also check the N/S and W/STOP/E layout alignment, and preserve the two-stage Sun GoTo confirmation.

## Validation

Validate incrementally:

```bash
cargo fmt --check
cargo check -p ghostsun-app --locked
cargo test -p ghostsun-app --locked
```

Then validate the native Mac target and serial discovery. Package for the machine's architecture using the repository's macOS packaging script. Test `/dev/cu.*` detection with the AM5N attached, but do not send physical movement commands until the user explicitly approves a safe hardware test.

Keep the work on `feature/zwo-mount-control`. Before committing further changes, report what compiles, what remains untested on hardware, and any protocol assumptions that need confirmation.
