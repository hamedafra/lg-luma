//! LG's own interface typeface, LG Smart UI, read from the TV's firmware at runtime.
//! It is never bundled: the files belong to LG and only exist on the TV itself.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// Where webOS keeps its system fonts.
pub const DIRECTORY: &str = "/usr/share/fonts";

const FACES: [&str; 4] = [
    "LG_Smart_UI-Light.ttf",
    "LG_Smart_UI-Regular.ttf",
    "LG_Smart_UI-SemiBold.ttf",
    "LG_Smart_UI-Bold.ttf",
];

/// Advance widths in ems per character, one map per face, for layout measurement.
pub type Metrics = [BTreeMap<char, f32>; 4];

/// LG Smart UI has four static weights. Luma's weights map to the nearest one, leaning
/// heavier as LG's own Home does, which sets most of its text in SemiBold.
pub fn face(weight: u16) -> usize {
    match weight {
        0..=375 => 0,
        376..=475 => 1,
        476..=675 => 2,
        _ => 3,
    }
}

pub fn path(directory: &Path, weight: u16) -> PathBuf {
    directory.join(FACES[face(weight)])
}

/// The directory with all four faces, when this device has them. `LUMA_SYSTEM_FONTS`
/// points elsewhere, for desktop previews with the files copied from a TV.
pub fn directory() -> Option<PathBuf> {
    let directory = std::env::var_os("LUMA_SYSTEM_FONTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| DIRECTORY.into());
    FACES
        .iter()
        .all(|face| directory.join(face).is_file())
        .then_some(directory)
}

pub fn metrics(directory: &Path) -> Result<Metrics, String> {
    let mut out: Metrics = Default::default();
    for (index, name) in FACES.iter().enumerate() {
        let path = directory.join(name);
        let bytes = std::fs::read(&path).map_err(|e| format!("Font {}: {e}", path.display()))?;
        let face = owned_ttf_parser::Face::parse(&bytes, 0)
            .map_err(|e| format!("Font {}: {e}", path.display()))?;
        let units = f32::from(face.units_per_em());
        let advances = &mut out[index];
        for table in face
            .tables()
            .cmap
            .into_iter()
            .flat_map(|cmap| cmap.subtables)
        {
            if !table.is_unicode() {
                continue;
            }
            table.codepoints(|codepoint| {
                if let Some(c) = char::from_u32(codepoint)
                    && let Some(glyph) = table.glyph_index(codepoint)
                    && glyph.0 != 0
                    && let Some(advance) = face.glyph_hor_advance(glyph)
                {
                    advances.entry(c).or_insert(f32::from(advance) / units);
                }
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_lean_towards_lg_homes_semibold() {
        assert_eq!(face(350), 0);
        assert_eq!(face(400), 1);
        assert_eq!(face(450), 1);
        assert_eq!(face(500), 2);
        assert_eq!(face(550), 2);
        assert_eq!(face(650), 2);
        assert_eq!(face(800), 3);
        assert!(path(Path::new("/x"), 600).ends_with("LG_Smart_UI-SemiBold.ttf"));
    }

    #[test]
    fn a_directory_without_every_face_is_not_used() {
        let directory = std::env::temp_dir().join(format!("luma-sysfont-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(FACES[1]), b"").unwrap();
        assert!(FACES.iter().any(|face| !directory.join(face).is_file()));
        assert!(metrics(&directory).is_err());
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
