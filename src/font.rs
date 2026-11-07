//! Lazy TTF metrics and per-glyph coverage bitmaps; never preprocess an entire font.
use owned_ttf_parser::{AsFaceRef, GlyphId, OwnedFace};
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

    /// A coverage bitmap of `character` at `ppem` physical pixels, its outline shifted right by
    /// `dx` pixels (subpixel positioning). Coverage is the exact area of each pixel inside the
    /// outline, as FreeType's and Skia's grayscale rasterizers compute it.
    pub fn rasterize(&self, character: char, ppem: f32, hinting: Hinting, dx: f32) -> Bitmap {
        use skrifa::{
            MetadataProvider,
            instance::{LocationRef, Size},
            outline::DrawSettings,
        };
        let empty = Bitmap::default();
        let Ok(font) = skrifa::FontRef::new(self.face.as_slice()) else {
            return empty;
        };
        let outlines = font.outline_glyphs();
        let Some(glyph) = outlines.get(skrifa::GlyphId::new(u32::from(
            self.lookup_glyph_index(character),
        ))) else {
            return empty;
        };
        let mut pen = PixelOutline::default();
        let hinted = hinting != Hinting::Off
            && (4.0..=256.).contains(&ppem)
            && self.with_hinter(ppem, hinting, |instance| {
                glyph
                    .draw(DrawSettings::hinted(instance, false), &mut pen)
                    .is_ok()
            }) == Some(true);
        if !hinted {
            pen = PixelOutline::default();
            let settings = DrawSettings::unhinted(Size::new(ppem), LocationRef::default());
            if glyph.draw(settings, &mut pen).is_err() {
                return empty;
            }
        }
        pen.rasterize(dx)
    }

    fn with_hinter<T>(
        &self,
        ppem: f32,
        hinting: Hinting,
        draw: impl FnOnce(&skrifa::outline::HintingInstance) -> T,
    ) -> Option<T> {
        use skrifa::{
            MetadataProvider,
            instance::{LocationRef, Size},
            outline::{Engine, HintingInstance, HintingOptions, SmoothMode},
        };
        let font = skrifa::FontRef::new(self.face.as_slice()).ok()?;
        let outlines = font.outline_glyphs();
        let key = ((ppem * 64.).round() as u32, hinting);
        let mut hinters = self.hinters.borrow_mut();
        let instance = hinters.entry(key).or_insert_with(|| {
            let mode = if hinting == Hinting::Light {
                SmoothMode::Light
            } else {
                SmoothMode::Normal
            };
            // Neither Inter nor LG Smart UI carries bytecode; always use the automatic hinter.
            let options = HintingOptions {
                engine: Engine::Auto(None),
                target: mode.into(),
            };
            HintingInstance::new(&outlines, Size::new(ppem), LocationRef::default(), options)
                .map_err(|error| eprintln!("Luma font hinting at {ppem} px: {error}"))
                .ok()
        });
        Some(draw(instance.as_ref()?))
    }
}

/// A glyph's coverage, one byte per pixel, placed relative to its pen position on the
/// baseline: `left` pixels right and `top` pixels down (negative above the baseline).
#[derive(Default)]
pub struct Bitmap {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub coverage: Vec<u8>,
}

enum Segment {
    Move([f32; 2]),
    Line([f32; 2]),
    Quad([f32; 2], [f32; 2]),
    Cubic([f32; 2], [f32; 2], [f32; 2]),
}

/// Collects an outline in physical pixels, y down, for the coverage rasterizer.
#[derive(Default)]
struct PixelOutline {
    segments: Vec<Segment>,
}

impl PixelOutline {
    fn rasterize(&self, dx: f32) -> Bitmap {
        let points = self.segments.iter().flat_map(|segment| match segment {
            Segment::Move(p) | Segment::Line(p) => vec![*p],
            Segment::Quad(a, b) => vec![*a, *b],
            Segment::Cubic(a, b, c) => vec![*a, *b, *c],
        });
        let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
        for [x, y] in points {
            min = [min[0].min(x + dx), min[1].min(y)];
            max = [max[0].max(x + dx), max[1].max(y)];
        }
        if min[0] >= max[0] || min[1] >= max[1] {
            return Bitmap::default();
        }
        let (left, top) = (min[0].floor() as i32, min[1].floor() as i32);
        let width = (max[0].ceil() as i32 - left).max(1) as u32;
        let height = (max[1].ceil() as i32 - top).max(1) as u32;
        let shift =
            |[x, y]: [f32; 2]| ab_glyph_rasterizer::point(x + dx - left as f32, y - top as f32);
        let mut raster = ab_glyph_rasterizer::Rasterizer::new(width as usize, height as usize);
        let (mut start, mut last) = ([0.; 2], [0.; 2]);
        let close = |raster: &mut ab_glyph_rasterizer::Rasterizer, last: [f32; 2], start| {
            if last != start {
                raster.draw_line(shift(last), shift(start));
            }
        };
        for segment in &self.segments {
            match *segment {
                Segment::Move(p) => {
                    close(&mut raster, last, start);
                    (start, last) = (p, p);
                }
                Segment::Line(p) => {
                    raster.draw_line(shift(last), shift(p));
                    last = p;
                }
                Segment::Quad(c, p) => {
                    raster.draw_quad(shift(last), shift(c), shift(p));
                    last = p;
                }
                Segment::Cubic(c1, c2, p) => {
                    raster.draw_cubic(shift(last), shift(c1), shift(c2), shift(p));
                    last = p;
                }
            }
        }
        close(&mut raster, last, start);
        let mut coverage = vec![0; width as usize * height as usize];
        raster.for_each_pixel(|index, alpha| {
            coverage[index] = (alpha.clamp(0., 1.) * 255. + 0.5) as u8;
        });
        Bitmap {
            left,
            top,
            width,
            height,
            coverage,
        }
    }
}

impl skrifa::outline::OutlinePen for PixelOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.segments.push(Segment::Move([x, -y]));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.segments.push(Segment::Line([x, -y]));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.segments.push(Segment::Quad([x1, -y1], [x, -y]));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.segments
            .push(Segment::Cubic([x1, -y1], [x2, -y2], [x, -y]));
    }
    fn close(&mut self) {}
}
