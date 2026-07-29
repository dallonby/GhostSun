//! Protocol probe for the ZWO mount.
//!
//! Answers questions the documentation does not. The AM5 manual lists only the
//! coarse 0.5x-1440x rate table, while INDI's driver uses a continuous
//! `:Rv<x.xx>#`; rather than trust either, ask the firmware.
//!
//! The read-only pass turns out to be inconclusive on rates, and that is worth
//! knowing: LX200 *set* commands reply with nothing at all, including the
//! known-good `:Rg`. Silence therefore does not distinguish "accepted" from
//! "ignored", and there is no query for the slew rate. The only way to tell
//! whether a rate command was honoured is to move a known axis for a known time
//! and measure the angle covered.
//!
//! ```sh
//! cargo run --release -p ghostsun-app --example mount_probe -- <dev>          # no motion
//! cargo run --release -p ghostsun-app --example mount_probe -- <dev> home     # MOVES
//! cargo run --release -p ghostsun-app --example mount_probe -- <dev> rate     # MOVES
//! cargo run --release -p ghostsun-app --example mount_probe -- <dev> hybrid   # MOVES
//! ```


use std::time::{Duration, Instant};

/// Seconds of motion per rate measurement. Long enough to swamp command
/// latency, short enough that the axis moves only a fraction of a degree.
const MEASURE_SECS: f64 = 3.0;
const SIDEREAL_ARCSEC_PER_SEC: f64 = 15.041;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: mount_probe <serial-device> [home|rate|hybrid]");
        std::process::exit(2);
    });
    let mode = args.next().unwrap_or_else(|| "readonly".to_owned());

    // Same parameters the app uses, so behaviour here predicts behaviour there.
    let mut port = match serialport::new(&path, 9_600)
        .timeout(Duration::from_millis(300))
        .open()
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("open {path}: {e}");
            std::process::exit(1);
        }
    };
    let port = &mut *port;

    match mode.as_str() {
        "readonly" => readonly(port),
        "home" => home(port),
        "rate" => rate(port),
        "hybrid" => hybrid(port),
        "peraxis" => peraxis(port),
        "transient" => transient(port),
        other => {
            eprintln!("unknown mode {other:?}");
            std::process::exit(2);
        }
    }
}

fn readonly(port: &mut dyn serialport::SerialPort) {
    println!("== identity and state ==");
    for (cmd, note) in [
        (":GVP#", "product name"),
        (":GVN#", "firmware version"),
        (":GU#", "status flags"),
        (":GR#", "right ascension"),
        (":GD#", "declination"),
        (":GT#", "tracking mode"),
        (":Ggr#", "guide rate"),
        // Recovered from firmware v1.8.8 strings; undocumented. Gets only --
        // the :S* siblings write settings and are deliberately not sent.
        (":GVE#", "firmware/version?"),
        (":GSN#", "serial number?"),
        (":GPT#", "?"),
        (":GCAL#", "calibration?"),
        (":GADC#", "ADC / voltage?"),
        (":GWF#", "wifi?"),
        (":GKEY#", "?"),
    ] {
        probe(port, cmd, note);
    }
    println!();
    println!("NOTE: set-commands reply with nothing, so acceptance cannot be");
    println!("read from silence. Use `rate` to measure whether :Rv is honoured.");
}

fn home(port: &mut dyn serialport::SerialPort) {
    println!("== homing (the mount will move) ==");
    probe(port, ":GU#", "status before");
    let _ = send(port, ":hC#");
    let started = Instant::now();
    // Poll until the status flags report home, rather than guessing a duration.
    while started.elapsed() < Duration::from_secs(180) {
        std::thread::sleep(Duration::from_millis(1000));
        let flags = exchange(port, ":GU#").unwrap_or_default();
        let coords = read_coords(port);
        println!(
            "  {:>3}s  {flags:<18} RA {} Dec {}",
            started.elapsed().as_secs(),
            coords.0,
            coords.1
        );
        if flags.contains('H') {
            println!("HOMED after {:.0}s", started.elapsed().as_secs_f64());
            return;
        }
    }
    println!("timed out waiting for the home flag; check the mount");
}

/// Measure the sky rate each command actually produces.
///
/// The table rates bracket the interesting value: if `:Rv` is ignored, the axis
/// keeps whatever the preceding `:R<n>#` selected, so alternating table and
/// variable commands makes an ignored `:Rv` obvious -- it simply reproduces the
/// table rate before it. Two `:Rv` measurements agreeing with each other while
/// differing from both neighbours is the positive result.
fn rate(port: &mut dyn serialport::SerialPort) {
    println!("== measuring actual sky rate per command (the mount will move) ==");
    println!("   each test drives Dec south for {MEASURE_SECS:.0}s\n");
    for (setup, label, expect) in [
        (":R4#", "table 4x", Some(4.0)),
        (":Rv8.51#", "variable 8.51x", None),
        (":R6#", "table 20x", Some(20.0)),
        (":Rv8.51#", "variable 8.51x (repeat)", None),
    ] {
        let _ = send(port, setup);
        std::thread::sleep(Duration::from_millis(250));
        match measure_dec_rate(port) {
            Some(multiple) => {
                let verdict = match expect {
                    Some(e) if (multiple - e).abs() < e * 0.25 => "matches table",
                    Some(_) => "DOES NOT match table",
                    None => "compare with neighbours",
                };
                println!("  {setup:<12} {label:<24} -> {multiple:6.2}x sidereal  ({verdict})");
            }
            None => println!("  {setup:<12} {label:<24} -> could not read Dec"),
        }
    }
    println!();
    println!("If both :Rv rows read ~8.5x, variable rates are honoured and the");
    println!("scan can hit any target time exactly. If they echo the preceding");
    println!("table row instead, :Rv is ignored and the ten steps are the limit.");
}

fn hybrid(port: &mut dyn serialport::SerialPort) {
    println!("== do :Mn and :Me drive BOTH axes at once? (the mount will move) ==");
    let _ = send(port, ":R4#");
    std::thread::sleep(Duration::from_millis(250));
    let (ra0, dec0) = read_coords(port);
    println!("  before: RA {ra0} Dec {dec0}");
    let _ = send(port, ":Ms#");
    let _ = send(port, ":Me#");
    std::thread::sleep(Duration::from_secs_f64(2.0));
    let _ = send(port, ":Q#");
    std::thread::sleep(Duration::from_millis(500));
    let (ra1, dec1) = read_coords(port);
    println!("  after:  RA {ra1} Dec {dec1}");
    let moved_ra = ra0 != ra1;
    let moved_dec = dec0 != dec1;
    println!();
    match (moved_ra, moved_dec) {
        (true, true) => println!("BOTH axes moved: a 45-degree hybrid mode exists."),
        (false, true) => println!("Only Dec moved: :Me was ignored or replaced by :Ms."),
        (true, false) => println!("Only RA moved: :Ms was ignored or replaced by :Me."),
        (false, false) => println!("Neither moved -- check the mount is unparked and homed."),
    }
}

/// Can the two axes be driven at DIFFERENT rates?
///
/// This is the question that decides whether hybrid scanning is possible at
/// all. Setting a sky angle theta at velocity v needs Dec = v*cos(theta) and
/// RA = v*sin(theta)/cos(Dec) -- two independent rates. `:RA`/`:RE` replied
/// with silence, but so does every set-command including the known-good `:Rg`,
/// so silence is not evidence. Only measured motion settles it.
fn peraxis(port: &mut dyn serialport::SerialPort) {
    println!("== can the axes take different rates? (the mount will move) ==\n");
    // Guide/rate settings may only bite with tracking active, and a rate change
    // may only latch when motion restarts -- every measurement below issues a
    // fresh :M.../:Q# pair, so the restart is inherent.
    let _ = send(port, ":Te#");
    std::thread::sleep(Duration::from_millis(500));
    probe(port, ":GU#", "status with tracking on");

    let _ = send(port, ":Rv4.00#");
    std::thread::sleep(Duration::from_millis(250));
    let dec_a = measure_dec_rate(port);
    let ra_a = measure_ra_rate(port);
    println!("  after :Rv4.00#            Dec {:?}  RA {:?}", fmt(dec_a), fmt(ra_a));

    // If these are honoured the axes diverge; if ignored both stay at 4x.
    let _ = send(port, ":RA08.0#");
    let _ = send(port, ":RE02.0#");
    std::thread::sleep(Duration::from_millis(250));
    let dec_b = measure_dec_rate(port);
    let ra_b = measure_ra_rate(port);
    println!("  after :RA08.0# :RE02.0#   Dec {:?}  RA {:?}", fmt(dec_b), fmt(ra_b));

    println!();
    match (dec_a, ra_a, dec_b, ra_b) {
        (Some(_), Some(ra0), Some(_), Some(ra1)) if (ra1 - ra0).abs() > ra0 * 0.3 => {
            println!("RA rate CHANGED -> per-axis rates exist; arbitrary hybrid angles possible.")
        }
        (Some(_), Some(_), Some(_), Some(_)) => {
            println!("Both axes unchanged -> :RA/:RE ignored; only one global rate is available,")
        }
        _ => println!("measurement incomplete"),
    }
}

/// Does the 8x post-homing anomaly follow homing specifically, or any state
/// change? The mitigation differs: one throwaway move after homing, versus a
/// verification leg before every timed sequence.
fn transient(port: &mut dyn serialport::SerialPort) {
    println!("== does the 8x rate anomaly follow state changes? ==\n");
    let _ = send(port, ":Rv4.00#");
    std::thread::sleep(Duration::from_millis(250));
    println!("  baseline                  -> {:?}", fmt(measure_dec_rate(port)));

    for (cmd, label, settle) in [
        (":Te#", "after tracking ON", 1),
        (":Td#", "after tracking OFF", 1),
        (":Q#", "after a stop", 1),
    ] {
        let _ = send(port, cmd);
        std::thread::sleep(Duration::from_secs(settle));
        let _ = send(port, ":Rv4.00#");
        std::thread::sleep(Duration::from_millis(250));
        println!("  {label:<25} -> {:?}", fmt(measure_dec_rate(port)));
    }

    println!("  idling 45s ...");
    std::thread::sleep(Duration::from_secs(45));
    let _ = send(port, ":Rv4.00#");
    std::thread::sleep(Duration::from_millis(250));
    println!("  {:<25} -> {:?}", "after a long idle", fmt(measure_dec_rate(port)));
    println!();
    println!("All ~4x means the anomaly is homing-specific: one throwaway move");
    println!("after homing suffices. Any ~32x row means it is broader.");
}

fn fmt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.2}x")).unwrap_or_else(|| "?".into())
}

/// RA-axis rate as a multiple of sidereal. The `:GR#` readout is in time units,
/// so one second of RA is 15 arcsec of AXIS rotation -- not of sky angle, which
/// is smaller by cos(Dec). Axis rate is what the rate commands control, so that
/// is what is compared.
fn measure_ra_rate(port: &mut dyn serialport::SerialPort) -> Option<f64> {
    let before = parse_ra_arcsec(&exchange(port, ":GR#")?)?;
    let t0 = Instant::now();
    send(port, ":Me#").ok()?;
    std::thread::sleep(Duration::from_secs_f64(MEASURE_SECS));
    send(port, ":Q#").ok()?;
    let elapsed = t0.elapsed().as_secs_f64();
    std::thread::sleep(Duration::from_millis(600));
    let after = parse_ra_arcsec(&exchange(port, ":GR#")?)?;
    Some(((after - before).abs() / elapsed) / SIDEREAL_ARCSEC_PER_SEC)
}

/// `HH:MM:SS#` to arcseconds of axis rotation (1 s of RA = 15 arcsec).
fn parse_ra_arcsec(text: &str) -> Option<f64> {
    let t = text.trim().trim_end_matches('#').trim();
    let mut parts = t.split(':');
    let h: f64 = parts.next()?.trim().parse().ok()?;
    let m: f64 = parts.next()?.trim().parse().ok()?;
    let s: f64 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
    Some((h * 3600.0 + m * 60.0 + s) * 15.0)
}

/// Drive Dec south briefly and return the rate as a multiple of sidereal.
///
/// Dec rather than RA because RA motion composes with whatever tracking is
/// doing, which would contaminate the measurement.
fn measure_dec_rate(port: &mut dyn serialport::SerialPort) -> Option<f64> {
    let before = parse_dec_arcsec(&exchange(port, ":GD#")?)?;
    let t0 = Instant::now();
    send(port, ":Ms#").ok()?;
    std::thread::sleep(Duration::from_secs_f64(MEASURE_SECS));
    send(port, ":Q#").ok()?;
    let elapsed = t0.elapsed().as_secs_f64();
    std::thread::sleep(Duration::from_millis(600));
    let after = parse_dec_arcsec(&exchange(port, ":GD#")?)?;
    let arcsec_per_sec = (before - after).abs() / elapsed;
    Some(arcsec_per_sec / SIDEREAL_ARCSEC_PER_SEC)
}

/// `sDD*MM#` or `sDD*MM:SS#` to arcseconds. The sign applies to the whole
/// value, not just the degree field.
fn parse_dec_arcsec(text: &str) -> Option<f64> {
    let t = text.trim().trim_end_matches('#').trim();
    let negative = t.starts_with('-');
    let body = t.trim_start_matches(['+', '-']);
    let idx = body.find(['*', '\u{b0}', ':'])?;
    let (d, rest) = body.split_at(idx);
    let rest = &rest[rest.chars().next()?.len_utf8()..];
    let d: f64 = d.trim().parse().ok()?;
    let (m, s) = match rest.split_once(':') {
        Some((m, s)) => (m.trim().parse::<f64>().ok()?, s.trim().parse::<f64>().ok()?),
        None => (rest.trim().parse::<f64>().ok()?, 0.0),
    };
    let total = d.abs() * 3600.0 + m * 60.0 + s;
    Some(if negative { -total } else { total })
}

fn read_coords(port: &mut dyn serialport::SerialPort) -> (String, String) {
    (
        exchange(port, ":GR#").unwrap_or_default(),
        exchange(port, ":GD#").unwrap_or_default(),
    )
}

fn probe(port: &mut dyn serialport::SerialPort, cmd: &str, note: &str) {
    match exchange(port, cmd) {
        Some(reply) if reply.is_empty() => {
            println!("  {cmd:<12} {note:<32} -> (no reply)")
        }
        Some(reply) => println!("  {cmd:<12} {note:<32} -> {reply:?}"),
        None => println!("  {cmd:<12} {note:<32} -> WRITE FAILED"),
    }
}

fn send(port: &mut dyn serialport::SerialPort, cmd: &str) -> std::io::Result<()> {
    port.write_all(cmd.as_bytes())?;
    port.flush()
}

/// Send and collect whatever arrives before the timeout. Deliberately not
/// parsed: an unknown command's reply shape is what is being measured.
fn exchange(port: &mut dyn serialport::SerialPort, cmd: &str) -> Option<String> {
    let _ = port.clear(serialport::ClearBuffer::Input);
    send(port, cmd).ok()?;
    let mut out = Vec::new();
    let mut byte = [0u8; 1];
    for _ in 0..64 {
        match port.read(&mut byte) {
            Ok(1) => {
                out.push(byte[0]);
                if byte[0] == b'#' {
                    break;
                }
            }
            _ => break,
        }
    }
    Some(String::from_utf8_lossy(&out).into_owned())
}
