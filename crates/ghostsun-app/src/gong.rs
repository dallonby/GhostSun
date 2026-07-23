//! Timestamp selection and download of calibrated GONG H-alpha references.

use ghostsun_core::image2d::Image;
use ghostsun_core::metrics::DiskFit;
use ghostsun_core::ser::SerReader;
use std::path::{Path, PathBuf};

const DOTNET_UNIX_EPOCH_SECONDS: i64 = 62_135_596_800;
const GONG_ROOT: &str = "https://gong2.nso.edu/ftp/HA/hag";
const MAX_REFERENCE_DELTA_SECONDS: i64 = 6 * 60 * 60;

pub struct GongReference {
    pub image: Image,
    pub disk: DiskFit,
    pub filename: String,
    pub url: String,
    pub delta_seconds: i64,
}

pub fn download_nearest(ser_path: &Path) -> Result<GongReference, String> {
    let reader =
        SerReader::open(ser_path).map_err(|e| format!("cannot read SER timestamp: {e}"))?;
    let ticks = reader.header.date_time_utc;
    if ticks <= DOTNET_UNIX_EPOCH_SECONDS * 10_000_000 {
        return Err(
            "this SER has no valid UTC timestamp; GONG feature matching needs acquisition time"
                .into(),
        );
    }
    let target_seconds = ticks / 10_000_000 - DOTNET_UNIX_EPOCH_SECONDS;
    let target_day = target_seconds.div_euclid(86_400);
    let agent = native_tls_agent();

    let mut nearest: Option<(i64, String, String)> = None;
    for day_offset in -1..=1 {
        let (year, month, day) = civil_from_days(target_day + day_offset);
        let month_dir = format!("{year:04}{month:02}");
        let day_dir = format!("{year:04}{month:02}{day:02}");
        let index_url = format!("{GONG_ROOT}/{month_dir}/{day_dir}/");
        let index = match get_text(&agent, &index_url) {
            Ok(body) => body,
            Err(_) => continue,
        };
        for filename in parse_filenames(&index) {
            let Some(seconds) = filename_unix_seconds(&filename) else {
                continue;
            };
            let delta = (seconds - target_seconds).abs();
            if nearest.as_ref().map(|n| delta < n.0).unwrap_or(true) {
                nearest = Some((delta, filename, index_url.clone()));
            }
        }
    }

    let Some((delta_seconds, filename, directory_url)) = nearest else {
        return Err("no GONG H-alpha reference was available near the SER timestamp".into());
    };
    if delta_seconds > MAX_REFERENCE_DELTA_SECONDS {
        return Err(format!(
            "nearest GONG H-alpha reference is {:.1} hours away; refusing an unreliable match",
            delta_seconds as f64 / 3600.0
        ));
    }

    let url = format!("{directory_url}{filename}");
    let bytes = load_cached_or_download(&agent, &filename, &url)?;
    let decoded =
        image::load_from_memory(&bytes).map_err(|e| format!("cannot decode GONG JPEG: {e}"))?;
    let gray = decoded.to_luma8();
    let (width, height) = gray.dimensions();
    if width < 512 || height < 512 {
        return Err(format!(
            "GONG reference is unexpectedly small ({width}x{height}); full resolution is required"
        ));
    }
    let mut image = Image::new(width as usize, height as usize);
    for (dst, src) in image.data.iter_mut().zip(gray.as_raw()) {
        *dst = *src as f32 * 257.0;
    }

    // GONG's calibrated full-disk product is centred in a 2048-square frame
    // at approximately 900 px radius. Scale this for future archive variants.
    let short_side = width.min(height) as f64;
    let disk = DiskFit {
        xc: (width as f64 - 1.0) * 0.5,
        yc: (height as f64 - 1.0) * 0.5,
        r: short_side * (900.0 / 2048.0),
    };
    Ok(GongReference {
        image,
        disk,
        filename,
        url,
        delta_seconds,
    })
}

fn native_tls_agent() -> ureq::Agent {
    use ureq::tls::{RootCerts, TlsConfig, TlsProvider};
    let config = ureq::config::Config::builder()
        .tls_config(
            TlsConfig::builder()
                .provider(TlsProvider::NativeTls)
                .root_certs(RootCerts::PlatformVerifier)
                .build(),
        )
        .build();
    config.new_agent()
}

fn get_text(agent: &ureq::Agent, url: &str) -> Result<String, String> {
    let mut response = agent
        .get(url)
        .header("User-Agent", "GhostSun/0.2 GONG orientation matcher")
        .call()
        .map_err(|e| format!("GONG request failed: {e}"))?;
    response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("cannot read GONG archive index: {e}"))
}

fn load_cached_or_download(
    agent: &ureq::Agent,
    filename: &str,
    url: &str,
) -> Result<Vec<u8>, String> {
    let cache = gong_cache_dir()?.join(filename);
    if let Ok(bytes) = std::fs::read(&cache) {
        if !bytes.is_empty() {
            return Ok(bytes);
        }
    }

    let mut response = agent
        .get(url)
        .header("User-Agent", "GhostSun/0.2 GONG orientation matcher")
        .call()
        .map_err(|e| format!("GONG image download failed: {e}"))?;
    let bytes = response
        .body_mut()
        .read_to_vec()
        .map_err(|e| format!("cannot read GONG image: {e}"))?;
    if bytes.is_empty() {
        return Err("GONG returned an empty image".into());
    }
    std::fs::write(&cache, &bytes)
        .map_err(|e| format!("cannot cache GONG image {}: {e}", cache.display()))?;
    Ok(bytes)
}

fn gong_cache_dir() -> Result<PathBuf, String> {
    let path = std::env::temp_dir().join("GhostSun").join("gong");
    std::fs::create_dir_all(&path)
        .map_err(|e| format!("cannot create GONG cache {}: {e}", path.display()))?;
    Ok(path)
}

fn parse_filenames(index: &str) -> Vec<String> {
    let bytes = index.as_bytes();
    let mut result = Vec::new();
    if bytes.len() < 20 {
        return result;
    }
    for start in 0..=bytes.len() - 20 {
        let candidate = &bytes[start..start + 20];
        if candidate[..14].iter().all(u8::is_ascii_digit)
            && matches!(candidate[14], b'B' | b'C' | b'L' | b'M' | b'T' | b'U')
            && candidate[15] == b'h'
            && &candidate[16..] == b".jpg"
        {
            let filename = String::from_utf8_lossy(candidate).into_owned();
            if result.last() != Some(&filename) {
                result.push(filename);
            }
        }
    }
    result
}

fn filename_unix_seconds(filename: &str) -> Option<i64> {
    if filename.len() < 14 || !filename.as_bytes()[..14].iter().all(u8::is_ascii_digit) {
        return None;
    }
    let year = filename[0..4].parse::<i32>().ok()?;
    let month = filename[4..6].parse::<u32>().ok()?;
    let day = filename[6..8].parse::<u32>().ok()?;
    let hour = filename[8..10].parse::<i64>().ok()?;
    let minute = filename[10..12].parse::<i64>().ok()?;
    let second = filename[12..14].parse::<i64>().ok()?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second)
}

// Gregorian calendar conversion algorithms adapted from Howard Hinnant's
// public-domain civil calendar formulae.
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = (adjusted_year as i64).div_euclid(400);
    let year_of_era = adjusted_year as i64 - era * 400;
    let shifted_month = month as i64 + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i64 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days_since_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year as i32, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_science_grade_halpha_files() {
        let html = r#"<a href="20260723160722Th.jpg">x</a>
            <a href="20260723160722Th.jpg">x</a>
            <a href="20260723160822Ub.jpg">bad product</a>
            <a href="20260723160922Uh.jpg">x</a>"#;
        assert_eq!(
            parse_filenames(html),
            vec!["20260723160722Th.jpg", "20260723160922Uh.jpg"]
        );
    }

    #[test]
    fn converts_gong_timestamp_round_trip() {
        let seconds = filename_unix_seconds("20260723160722Th.jpg").unwrap();
        let day = seconds.div_euclid(86_400);
        assert_eq!(civil_from_days(day), (2026, 7, 23));
        assert_eq!(seconds.rem_euclid(86_400), 16 * 3600 + 7 * 60 + 22);
    }

    /// Optional real archive smoke test:
    /// GHOSTSUN_TEST_SER=scan.ser GHOSTSUN_TEST_IMAGE=reconstruction.png
    /// cargo test -p ghostsun-app live_gong_feature_match -- --ignored --nocapture
    #[test]
    #[ignore = "requires a timestamped SER, its reconstruction, and network access"]
    fn live_gong_feature_match() {
        let ser = std::env::var_os("GHOSTSUN_TEST_SER").expect("GHOSTSUN_TEST_SER");
        let image_path = std::env::var_os("GHOSTSUN_TEST_IMAGE").expect("GHOSTSUN_TEST_IMAGE");
        let reference = download_nearest(Path::new(&ser)).unwrap();
        let image = ghostsun_core::output::read_png16(Path::new(&image_path)).unwrap();
        let prep = ghostsun_core::render::prepare(&image).unwrap();
        let matched = ghostsun_core::orientation::match_to_reference(
            &image,
            &prep.disk,
            &reference.image,
            &reference.disk,
        )
        .unwrap();
        eprintln!(
            "{}: mirror={}, rotation={:+.1}, score={:.3}, margin={:.3}",
            reference.filename,
            matched.mirrored,
            matched.rotation_deg,
            matched.score,
            matched.confidence_margin()
        );
        assert!(matched.is_confident());
        if let Some(output) = std::env::var_os("GHOSTSUN_TEST_OUTPUT") {
            let transformed = ghostsun_core::orientation::apply_orientation(
                &image,
                &prep.disk,
                matched.mirrored,
                matched.rotation_deg,
            );
            ghostsun_core::output::write_png16(Path::new(&output), &transformed, None).unwrap();
        }
    }
}
