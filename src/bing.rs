//! Bing's image of the day as a background. One worker downloads the 4K (UHD) original, derives
//! the same 1080p shaded/unshaded/thumbnail set as uploads plus `@2x` variants for 4K drawables,
//! and sleeps until Bing's next daily change. No timer runs while the background is unused.
use crate::wallpapers::Wallpaper;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::Duration,
};

/// Background ID of the daily image. It is not a `custom-` upload and never reaches the catalog.
pub const ID: &str = "bing";
const ORIGIN: &str = "https://www.bing.com";
const MARKET: &str = "en-US";
const DIRECTORY_PREFIX: &str = "bing-";
const JSON_LIMIT: u64 = 64 * 1024;
const IMAGE_LIMIT: u64 = 16 * 1024 * 1024;
const DAY: u64 = 24 * 60 * 60;
/// Bing publishes slightly after its nominal start; ask a few minutes later.
const PUBLISH_GRACE: u64 = 5 * 60;
const RETRY: [u64; 5] = [60, 5 * 60, 15 * 60, 60 * 60, 3 * 60 * 60];

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Metadata {
    /// Bing's UTC start, `YYYYMMDDHHMM`.
    pub start: String,
    pub hash: String,
    pub title: String,
    pub copyright: String,
}

#[derive(Debug, PartialEq, Eq)]
struct Image {
    metadata: Metadata,
    /// `/th?id=OHR.Name_EN-US0123456789`
    base: String,
}

pub enum Event {
    Ready(Wallpaper, Metadata),
}

enum Command {
    Wanted(bool),
    Wake,
}

pub struct Service {
    commands: mpsc::Sender<Command>,
    pub receiver: mpsc::Receiver<Event>,
    wanted: bool,
}

impl Service {
    /// `online` is false for scripted/simulated previews: they show a stored image but never
    /// contact Bing.
    pub fn new(preferences: &Path, online: bool) -> Result<Self, String> {
        let directory = preferences
            .parent()
            .unwrap_or(Path::new("."))
            .join("wallpapers");
        let (sender, receiver) = mpsc::sync_channel(2);
        let (commands, updates) = mpsc::channel();
        thread::Builder::new()
            .name("luma-bing".into())
            .spawn(move || worker(&directory, online, &sender, &updates))
            .map_err(|e| format!("Could not start Bing worker: {e}"))?;
        Ok(Self {
            commands,
            receiver,
            wanted: false,
        })
    }

    /// Download only while the daily image is selected, rotating or on screen in settings.
    pub fn set_wanted(&mut self, wanted: bool) {
        if self.wanted != wanted {
            self.wanted = wanted;
            let _ = self.commands.send(Command::Wanted(wanted));
        }
    }

    /// Re-check the schedule, e.g. after the TV resumes: monotonic sleeps stop in standby.
    pub fn wake(&self) {
        let _ = self.commands.send(Command::Wake);
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `YYYYMMDDHHMM` (UTC) to Unix seconds.
fn parse_start(start: &str) -> Option<u64> {
    if start.len() != 12 || !start.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let number = |range: std::ops::Range<usize>| start[range].parse::<u32>().ok();
    let month = time::Month::try_from(number(4..6)? as u8).ok()?;
    let date =
        time::Date::from_calendar_date(number(0..4)? as i32, month, number(6..8)? as u8).ok()?;
    let time = time::Time::from_hms(number(8..10)? as u8, number(10..12)? as u8, 0).ok()?;
    u64::try_from(date.with_time(time).assume_utc().unix_timestamp()).ok()
}

/// When the image after `metadata` is expected.
fn next_due(metadata: &Metadata) -> u64 {
    parse_start(&metadata.start).map_or(0, |start| start + DAY + PUBLISH_GRACE)
}

fn valid_base(base: &str) -> bool {
    base.strip_prefix("/th?id=").is_some_and(|id| {
        !id.is_empty()
            && id.len() <= 160
            && id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
    })
}

fn parse_archive(bytes: &[u8]) -> Result<Image, String> {
    #[derive(Deserialize)]
    struct Archive {
        images: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        fullstartdate: String,
        urlbase: String,
        #[serde(default)]
        hsh: String,
        #[serde(default)]
        title: String,
        #[serde(default)]
        copyright: String,
    }
    let archive: Archive =
        serde_json::from_slice(bytes).map_err(|e| format!("Invalid Bing reply: {e}"))?;
    let entry = archive
        .images
        .into_iter()
        .next()
        .ok_or("Bing returned no image")?;
    if parse_start(&entry.fullstartdate).is_none() || !valid_base(&entry.urlbase) {
        return Err("Unexpected Bing image description".into());
    }
    let clean = |text: String, limit: usize| -> String {
        text.chars()
            .filter(|c| !c.is_control())
            .take(limit)
            .collect()
    };
    Ok(Image {
        metadata: Metadata {
            start: entry.fullstartdate,
            hash: clean(entry.hsh, 64),
            title: clean(entry.title, 200),
            copyright: clean(entry.copyright, 300),
        },
        base: entry.urlbase,
    })
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(90)))
        .max_redirects(2)
        .https_only(true)
        .proxy(None)
        .build()
        .new_agent()
}

fn get(agent: &ureq::Agent, url: &str, limit: u64) -> Result<Vec<u8>, String> {
    let mut response = agent
        .get(url)
        .header("User-Agent", "LumaNative")
        .call()
        .map_err(|e| format!("Bing request failed: {e}"))?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Bing download failed: {e}"))?;
    if bytes.len() as u64 > limit {
        return Err("Bing reply is too large".into());
    }
    Ok(bytes)
}

fn image_directory(directory: &Path, metadata: &Metadata) -> PathBuf {
    // A new path per day: the renderer caches textures by path.
    directory.join(format!("{DIRECTORY_PREFIX}{}", metadata.start))
}

fn stored(path: &Path) -> Option<(Wallpaper, Metadata)> {
    let metadata: Metadata = serde_json::from_slice(&fs::read(path.join("metadata.json")).ok()?)
        .ok()
        .filter(|metadata: &Metadata| parse_start(&metadata.start).is_some())?;
    let complete = ["wallpaper.jpg", "original.jpg", "thumbnail.jpg"]
        .iter()
        .all(|file| {
            fs::symlink_metadata(path.join(file)).is_ok_and(|m| m.is_file() && m.len() > 0)
        });
    complete.then(|| {
        (
            Wallpaper {
                id: ID.into(),
                directory: path.to_owned(),
                has_original: true,
            },
            metadata,
        )
    })
}

/// The newest complete download, if any.
fn latest(directory: &Path) -> Option<(Wallpaper, Metadata)> {
    let mut paths: Vec<_> = fs::read_dir(directory)
        .ok()?
        .take(256)
        .flatten()
        .filter(|entry| {
            entry.file_name().to_str().is_some_and(|name| {
                name.strip_prefix(DIRECTORY_PREFIX)
                    .is_some_and(|start| parse_start(start).is_some())
            })
        })
        .map(|entry| entry.path())
        .collect();
    paths.sort();
    paths.iter().rev().find_map(|path| stored(path))
}

/// Remove every daily directory except `keep`, including interrupted staging directories.
fn prune(directory: &Path, keep: &Path) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.take(256).flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let daily = name.strip_prefix(DIRECTORY_PREFIX).or_else(|| {
            name.strip_prefix(".")
                .and_then(|name| name.strip_prefix(DIRECTORY_PREFIX))
        });
        let path = entry.path();
        if daily.is_some_and(|start| parse_start(start).is_some())
            && path != keep
            && fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir())
        {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn fill(image: image::RgbImage, width: u32, height: u32) -> image::RgbImage {
    if image.dimensions() == (width, height) {
        return image;
    }
    image::DynamicImage::ImageRgb8(image)
        .resize_to_fill(width, height, image::imageops::FilterType::Lanczos3)
        .to_rgb8()
}

fn decode(bytes: &[u8]) -> Result<image::RgbImage, String> {
    let mut reader = image::ImageReader::with_format(
        std::io::Cursor::new(bytes),
        image::guess_format(bytes).map_err(|_| "Bing did not return an image")?,
    );
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(160 * 1024 * 1024);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|e| format!("Invalid Bing image: {e}"))?;
    if image.width() < 1920 || image.height() < 1080 {
        return Err("Bing image is smaller than 1080p".into());
    }
    Ok(image.to_rgb8())
}

/// Builds the files in a hidden staging directory, then renames it into place.
fn store(directory: &Path, bytes: &[u8], metadata: &Metadata) -> Result<Wallpaper, String> {
    let uhd = fill(decode(bytes)?, 3840, 2160);
    let original = image::imageops::resize(&uhd, 1920, 1080, image::imageops::FilterType::Lanczos3);
    let thumbnail =
        image::imageops::resize(&original, 480, 270, image::imageops::FilterType::Triangle);
    let mut shaded = original.clone();
    crate::wallpapers::shade(&mut shaded);
    let mut shaded_uhd = uhd.clone();
    crate::wallpapers::shade(&mut shaded_uhd);
    let jpeg = crate::wallpapers::jpeg;
    let files = [
        ("wallpaper.jpg", jpeg(&shaded, 94)?),
        ("original.jpg", jpeg(&original, 94)?),
        ("thumbnail.jpg", jpeg(&thumbnail, 82)?),
        ("wallpaper@2x.jpg", jpeg(&shaded_uhd, 92)?),
        ("original@2x.jpg", jpeg(&uhd, 92)?),
    ];
    if let Some(parent) = directory.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    crate::wallpapers::create_directory(directory)?;
    let target = image_directory(directory, metadata);
    let staging = directory.join(format!(".{DIRECTORY_PREFIX}{}", metadata.start));
    let _ = fs::remove_dir_all(&staging);
    crate::wallpapers::create_directory(&staging)?;
    let result = (|| {
        for (name, bytes) in &files {
            crate::wallpapers::write_new(&staging.join(name), bytes)?;
        }
        crate::wallpapers::write_new(
            &staging.join("metadata.json"),
            &serde_json::to_vec(metadata).map_err(|e| e.to_string())?,
        )?;
        let _ = fs::remove_dir_all(&target);
        fs::rename(&staging, &target).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result?;
    Ok(Wallpaper {
        id: ID.into(),
        directory: target,
        has_original: true,
    })
}

enum Outcome {
    New(Wallpaper, Metadata),
    /// Bing still shows the image we have.
    Unchanged,
}

fn refresh(
    agent: &ureq::Agent,
    directory: &Path,
    current: Option<&Metadata>,
) -> Result<Outcome, String> {
    let archive = get(
        agent,
        &format!("{ORIGIN}/HPImageArchive.aspx?format=js&idx=0&n=1&mkt={MARKET}"),
        JSON_LIMIT,
    )?;
    let image = parse_archive(&archive)?;
    if current.is_some_and(|current| {
        current.start == image.metadata.start
            || (!current.hash.is_empty() && current.hash == image.metadata.hash)
    }) {
        return Ok(Outcome::Unchanged);
    }
    let bytes = get(
        agent,
        &format!("{ORIGIN}{}_UHD.jpg", image.base),
        IMAGE_LIMIT,
    )?;
    let wallpaper = store(directory, &bytes, &image.metadata)?;
    prune(directory, &wallpaper.directory);
    Ok(Outcome::New(wallpaper, image.metadata))
}

fn worker(
    directory: &Path,
    online: bool,
    sender: &mpsc::SyncSender<Event>,
    commands: &mpsc::Receiver<Command>,
) {
    let mut current = latest(directory);
    if let Some((wallpaper, metadata)) = current.clone()
        && sender.send(Event::Ready(wallpaper, metadata)).is_err()
    {
        return;
    }
    let agent = agent();
    let mut wanted = false;
    let mut failures = 0usize;
    let mut retry_at = 0u64;
    loop {
        let now = unix_now();
        let due = current
            .as_ref()
            .map_or(0, |(_, metadata)| next_due(metadata))
            .max(retry_at);
        let command = if !online || !wanted {
            commands.recv().map_err(|_| ())
        } else if now >= due {
            match refresh(&agent, directory, current.as_ref().map(|(_, m)| m)) {
                Ok(Outcome::New(wallpaper, metadata)) => {
                    failures = 0;
                    retry_at = 0;
                    current = Some((wallpaper.clone(), metadata.clone()));
                    if sender.send(Event::Ready(wallpaper, metadata)).is_err() {
                        return;
                    }
                }
                Ok(Outcome::Unchanged) => {
                    // Published late: ask again in 15 minutes, not in a tight loop.
                    retry_at = now + 15 * 60;
                }
                Err(error) => {
                    eprintln!("Luma Bing wallpaper: {error}");
                    retry_at = now + RETRY[failures.min(RETRY.len() - 1)];
                    failures += 1;
                }
            }
            continue;
        } else {
            match commands.recv_timeout(Duration::from_secs(due - now)) {
                Ok(command) => Ok(command),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(()),
            }
        };
        match command {
            Ok(Command::Wanted(value)) => wanted = value,
            Ok(Command::Wake) => {}
            Err(()) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCHIVE: &str = r#"{"images":[{"startdate":"20261106","fullstartdate":"202611060800",
        "enddate":"20261107","url":"/th?id=OHR.PipiwaiTrail_EN-US0631882545_1920x1080.jpg&rf=x&pid=hp",
        "urlbase":"/th?id=OHR.PipiwaiTrail_EN-US0631882545","copyright":"Bamboo forest (© MNStudio/Getty Images)",
        "title":"A green tunnel through Maui","hsh":"2b8827c13484500ffaf00dd7575ed652"}]}"#;

    #[test]
    fn parses_the_daily_archive_reply() {
        let image = parse_archive(ARCHIVE.as_bytes()).unwrap();
        assert_eq!(image.base, "/th?id=OHR.PipiwaiTrail_EN-US0631882545");
        assert_eq!(image.metadata.start, "202611060800");
        assert_eq!(image.metadata.title, "A green tunnel through Maui");
        assert_eq!(image.metadata.hash, "2b8827c13484500ffaf00dd7575ed652");
    }

    #[test]
    fn rejects_unexpected_image_paths() {
        for base in [
            "https://evil.example/x",
            "/th?id=",
            "/th?id=../../etc",
            "/th?id=a&b=c",
            "/th?id=a/b",
            "/other?id=OHR.X",
        ] {
            assert!(!valid_base(base), "{base}");
            let reply = ARCHIVE.replace(
                "/th?id=OHR.PipiwaiTrail_EN-US0631882545\"",
                &format!("{base}\""),
            );
            assert!(parse_archive(reply.as_bytes()).is_err(), "{base}");
        }
        assert!(parse_archive(br#"{"images":[]}"#).is_err());
        assert!(parse_archive(b"<html>").is_err());
    }

    #[test]
    fn next_image_is_due_a_day_after_the_current_start() {
        assert_eq!(parse_start("202611060800"), Some(1_793_952_000));
        for invalid in [
            "",
            "20261106",
            "202613060800",
            "2026110608xx",
            "202611062500",
        ] {
            assert_eq!(parse_start(invalid), None, "{invalid}");
        }
        let metadata = Metadata {
            start: "202611060800".into(),
            ..Metadata::default()
        };
        assert_eq!(next_due(&metadata), 1_793_952_000 + DAY + PUBLISH_GRACE);
        assert_eq!(next_due(&Metadata::default()), 0);
    }

    #[test]
    fn stores_1080p_and_4k_variants_and_keeps_only_the_newest_day() {
        let root = std::env::temp_dir().join(format!("luma-bing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let directory = root.join("wallpapers");
        let source = image::RgbImage::from_fn(3840, 2160, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 90])
        });
        let bytes = crate::wallpapers::jpeg(&source, 80).unwrap();
        let mut days = vec![];
        for start in ["202611050800", "202611060800"] {
            let metadata = Metadata {
                start: start.into(),
                ..Metadata::default()
            };
            let wallpaper = store(&directory, &bytes, &metadata).unwrap();
            prune(&directory, &wallpaper.directory);
            days.push(wallpaper);
        }
        let (wallpaper, metadata) = latest(&directory).unwrap();
        assert_eq!(metadata.start, "202611060800");
        assert_eq!(wallpaper.directory, days[1].directory);
        assert!(!days[0].directory.exists());
        let size = |file: &str| image::image_dimensions(wallpaper.directory.join(file)).unwrap();
        assert_eq!(size("wallpaper.jpg"), (1920, 1080));
        assert_eq!(size("original.jpg"), (1920, 1080));
        assert_eq!(size("thumbnail.jpg"), (480, 270));
        assert_eq!(size("wallpaper@2x.jpg"), (3840, 2160));
        assert_eq!(size("original@2x.jpg"), (3840, 2160));
        // Uploaded-wallpaper discovery never lists the daily image.
        assert!(crate::wallpapers::catalog(&directory).is_empty());
        let _ = fs::remove_dir_all(root);
    }
}
