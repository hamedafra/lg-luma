//! Lazy TTF metrics and individual outlines; never preprocess an entire font.
use geometry::lyon_tessellation::{
    FillRule,
    math::point,
    path::{Path, path::Builder},
};
use luma_vector_geometry::{self as geometry, Vertex};
use owned_ttf_parser::{AsFaceRef, GlyphId, OutlineBuilder, OwnedFace};
use std::{cell::RefCell, collections::HashMap};

/// Grid fitting for the TV's 1080p graphics plane, which the panel then scales up. Unhinted
/// stems land between pixels and turn grey; hinting snaps them to the pixel grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Hinting {
    Off,
    /// Vertical only (FreeType `LIGHT`): baselines, x-height and horizontal stems. Glyphs keep
    /// their designed widths, so spacing stays even, as in Chrome and LG's own Home.
    #[default]
    Light,
    /// Both axes (FreeType `NORMAL`): vertical stems as well. Moving stems sideways without
    /// correcting the advances leaves uneven gaps ("Fon t hin ting"), so it is opt-in only.
    Normal,
}

impl Hinting {
    /// `LUMA_HINTING=off|light|normal` overrides the saved choice, for hardware comparisons.
    pub fn from_env(enabled: bool, value: Option<&str>) -> Self {
        match value {
            Some("off") => Self::Off,
            Some("light") => Self::Light,
            Some("normal") => Self::Normal,
            _ if enabled => Self::default(),
            _ => Self::Off,
        }
    }
}

pub struct Font {
    face: OwnedFace,
    /// One autohinter instance per (physical size in 1/64 px, mode); built on first use.
    hinters: RefCell<HashMap<(u32, Hinting), Option<skrifa::outline::HintingInstance>>>,
}

pub struct LineMetrics {
    pub ascent: f32,
    pub descent: f32,
}

pub struct Metrics {
    pub advance_width: f32,
}

impl Font {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        OwnedFace::from_vec(bytes, 0)
            .map(|face| Self {
                face,
                hinters: RefCell::new(HashMap::new()),
            })
            .map_err(|e| e.to_string())
    }

    pub fn lookup_glyph_index(&self, character: char) -> u16 {
        self.face
            .as_face_ref()
            .glyph_index(character)
            .map_or(0, |id| id.0)
    }

    pub fn horizontal_line_metrics(&self, size: f32) -> Option<LineMetrics> {
        let face = self.face.as_face_ref();
        let scale = size / f32::from(face.units_per_em());
        Some(LineMetrics {
            ascent: f32::from(face.ascender()) * scale,
            descent: f32::from(face.descender()) * scale,
        })
    }

    pub fn metrics(&self, character: char, size: f32) -> Metrics {
        let face = self.face.as_face_ref();
        let glyph = GlyphId(self.lookup_glyph_index(character));
        Metrics {
            advance_width: f32::from(face.glyph_hor_advance(glyph).unwrap_or(0)) * size
                / f32::from(face.units_per_em()),
        }
    }

    pub fn horizontal_kern(&self, left: char, right: char, size: f32) -> Option<f32> {
        let face = self.face.as_face_ref();
        let left = face.glyph_index(left)?;
        let right = face.glyph_index(right)?;
        for table in face.tables().kern?.subtables {
            if table.horizontal
                && !table.has_cross_stream
                && let Some(value) = table.glyphs_kerning(left, right)
            {
                return Some(f32::from(value) * size / f32::from(face.units_per_em()));
            }
        }
        None
    }

    /// Outline in em units, grid-fitted for `ppem` physical pixels when hinting is on. Falls
    /// back to the unhinted outline if the hinter cannot handle the glyph.
    pub fn geometry_at(
        &self,
        character: char,
        ppem: f32,
        hinting: Hinting,
    ) -> Result<Vec<Vertex>, String> {
        if hinting == Hinting::Off || !(4.0..=256.).contains(&ppem) {
            return self.geometry(character);
        }
        match self.hinted(character, ppem, hinting) {
            Some(path) => geometry::tessellate(&path, 0.15 / 256., FillRule::NonZero),
            None => self.geometry(character),
        }
    }

    fn hinted(&self, character: char, ppem: f32, hinting: Hinting) -> Option<Path> {
        use skrifa::{
            MetadataProvider,
            instance::{LocationRef, Size},
            outline::{DrawSettings, Engine, HintingInstance, HintingOptions, SmoothMode},
        };
        let font = skrifa::FontRef::new(self.face.as_slice()).ok()?;
        let outlines = font.outline_glyphs();
        let glyph = outlines.get(skrifa::GlyphId::new(u32::from(
            self.lookup_glyph_index(character),
        )))?;
        let key = ((ppem * 64.).round() as u32, hinting);
        let mut hinters = self.hinters.borrow_mut();
        let instance = hinters.entry(key).or_insert_with(|| {
            let mode = if hinting == Hinting::Light {
                SmoothMode::Light
            } else {
                SmoothMode::Normal
            };
            // The bundled fonts carry no bytecode; always use the automatic hinter.
            let options = HintingOptions {
                engine: Engine::Auto(None),
                target: mode.into(),
            };
            HintingInstance::new(&outlines, Size::new(ppem), LocationRef::default(), options)
                .map_err(|error| eprintln!("Luma font hinting at {ppem} px: {error}"))
                .ok()
        });
        let instance = instance.as_ref()?;
        let mut pen = HintedOutline {
            builder: Path::builder(),
            scale: 1. / ppem,
            open: false,
        };
        glyph
            .draw(DrawSettings::hinted(instance, false), &mut pen)
            .ok()?;
        if pen.open {
            pen.builder.end(true);
        }
        Some(pen.builder.build())
    }

    pub fn geometry(&self, character: char) -> Result<Vec<Vertex>, String> {
        let face = self.face.as_face_ref();
        let mut outline = GlyphOutline {
            builder: Path::builder(),
            scale: 1. / f32::from(face.units_per_em()),
            open: false,
        };
        let glyph = GlyphId(self.lookup_glyph_index(character));
        if face.outline_glyph(glyph, &mut outline).is_none() {
            return Ok(Vec::new());
        }
        if outline.open {
            outline.builder.end(true);
        }
        // Flatten at subpixel precision for glyphs up to 256 physical px/em,
        // covering the current 1080p and 4K typography with the same geometry.
        geometry::tessellate(&outline.builder.build(), 0.15 / 256., FillRule::NonZero)
    }
}

/// Receives a hinted outline in pixels at the hinting size and stores it in em units.
struct HintedOutline {
    builder: Builder,
    scale: f32,
    open: bool,
}

impl skrifa::outline::OutlinePen for HintedOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.open {
            self.builder.end(true);
        }
        self.builder.begin(point(x * self.scale, -y * self.scale));
        self.open = true;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(point(x * self.scale, -y * self.scale));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(
            point(x1 * self.scale, -y1 * self.scale),
            point(x * self.scale, -y * self.scale),
        );
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(
            point(x1 * self.scale, -y1 * self.scale),
            point(x2 * self.scale, -y2 * self.scale),
            point(x * self.scale, -y * self.scale),
        );
    }
    fn close(&mut self) {
        if self.open {
            self.builder.end(true);
            self.open = false;
        }
    }
}

struct GlyphOutline {
    builder: Builder,
    scale: f32,
    open: bool,
}

impl OutlineBuilder for GlyphOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.open {
            self.builder.end(true);
        }
        self.builder.begin(point(x * self.scale, -y * self.scale));
        self.open = true;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(point(x * self.scale, -y * self.scale));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(
            point(x1 * self.scale, -y1 * self.scale),
            point(x * self.scale, -y * self.scale),
        );
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(
            point(x1 * self.scale, -y1 * self.scale),
            point(x2 * self.scale, -y2 * self.scale),
            point(x * self.scale, -y * self.scale),
        );
    }
    fn close(&mut self) {
        if self.open {
            self.builder.end(true);
            self.open = false;
        }
    }
}
