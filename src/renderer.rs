//! A small GLES2-compatible GPU painter. Rust owns the glyph atlas, vector logos and the draw
//! list; SDL only supplies the context. No browser, LVGL or Qt runtime.
use crate::{
    font::Font,
    raster::{Size, Viewport},
    state::Rect,
    vector,
};
use gl::types::*;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, CString},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

pub type Color = [f32; 4];
pub const WHITE: Color = [0.96, 0.96, 0.96, 1.];
const TEXT_RUN_BUDGET: usize = 4 * 1024 * 1024;
/// Glyph atlas pages, RGBA with coverage in alpha (GLES2 and core GL share no one-channel
/// format). Past `ATLAS_PAGES` the atlas starts over at the next frame.
const ATLAS_SIZE: u32 = 1024;
const ATLAS_PAGES: usize = 4;
/// Horizontal glyph positions per pixel, as Skia's grayscale text uses: letters keep their
/// designed spacing instead of each snapping to a whole pixel.
const SUBPIXEL_STEPS: f32 = 4.;
const TEXTURE_BUDGET: usize = 48 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct TextStyle {
    pub size: f32,
    pub weight: u16,
    /// CSS letter-spacing in em, not pixels.
    pub tracking: f32,
    pub line_height: f32,
    pub align_right: bool,
}
impl TextStyle {
    pub const fn new(size: f32) -> Self {
        Self {
            size,
            weight: 400,
            tracking: 0.,
            line_height: 1.2,
            align_right: false,
        }
    }
    pub const fn weight(mut self, weight: u16) -> Self {
        self.weight = weight;
        self
    }
    pub const fn tracking(mut self, tracking: f32) -> Self {
        self.tracking = tracking;
        self
    }
    pub const fn leading(mut self, height: f32) -> Self {
        self.line_height = height;
        self
    }
    pub const fn right(mut self) -> Self {
        self.align_right = true;
        self
    }
}

#[derive(Clone)]
pub enum Draw {
    Shape(Rect, f32, f32, Color),
    RoundedGradient(Rect, f32, Color, Color),
    Shadow(Rect, f32, f32, Color),
    Text(String, Rect, TextStyle, Color, bool),
    Image(String, Rect, f32, bool),
    TintedImage(String, Rect, f32, bool, Color),
    /// Installed icon and its backdrop, composited before opacity and edge fading.
    AppImage(Arc<crate::app_artwork::Artwork>, Rect, f32, Color),
    Clip(Option<Rect>),
    EdgeFade(Option<[f32; 4]>),
    /// Horizontal counterpart: [left fade-in start, end, right fade-out start, end].
    EdgeFadeX(Option<[f32; 4]>),
}

struct Texture {
    id: GLuint,
    width: u32,
    height: u32,
    frame: u64,
}
impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.id);
        }
    }
}

/// One rasterized glyph: character, font, physical size in 1/64 px and subpixel offset.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    c: char,
    font: u16,
    ppem: u32,
    subpixel: u8,
}

/// Where a glyph's coverage sits in the atlas; `width == 0` for blank glyphs.
#[derive(Clone, Copy)]
struct AtlasGlyph {
    page: usize,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    left: i32,
    top: i32,
}

/// One atlas texture filled shelf by shelf.
struct AtlasPage {
    texture: Texture,
    x: u32,
    y: u32,
    row: u32,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct TextKey {
    text: String,
    weight: u16,
    // Logical size, tracking, line height, available width and height.
    layout: [u32; 5],
    flags: u8,
}

/// A laid-out run: each glyph's pen position in logical pixels from the run's origin. Layout
/// is density-independent; glyphs are rasterized for the drawable when the run is drawn.
struct Placed {
    c: char,
    font: u16,
    x: f32,
    baseline: f32,
}

struct TextRun {
    glyphs: Vec<Placed>,
    missing_glyphs: usize,
    failures: usize,
    frame: u64,
}

#[derive(Clone, Serialize)]
pub struct GraphicsInfo {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
    pub max_texture_size: u32,
    pub max_viewport: [i32; 2],
    pub fragment_highp: bool,
}

#[derive(Clone, Serialize)]
pub struct RenderStats {
    pub target: Size,
    pub viewport: Viewport,
    pub font_antialiasing: &'static str,
    pub multisample_samples: u32,
    pub max_logical_font_px: f32,
    pub max_physical_font_px: f32,
    pub cached_glyphs: usize,
    pub glyph_tessellations: usize,
    /// Milliseconds this frame spent on first-use work: glyph outlines, logo meshes, images.
    pub glyph_ms: f64,
    pub vector_load_ms: f64,
    pub image_load_ms: f64,
    pub glyph_failures: usize,
    pub missing_glyphs: usize,
    pub missing_images: usize,
    pub glyph_geometry_bytes: usize,
    pub cached_text_runs: usize,
    pub text_geometry_bytes: usize,
    pub text_triangles: usize,
    pub artwork_bytes: usize,
    pub vector_meshes: usize,
    pub vector_bytes: usize,
    pub vector_triangles: usize,
    pub high_resolution_wallpaper: bool,
    pub draw_calls: usize,
    pub cpu_submission_ms: f64,
    /// Share of the drawable redrawn in the last frame (1 without partial redraw).
    pub redrawn_fraction: f64,
}
impl RenderStats {
    fn new(target: Size) -> Self {
        Self {
            target,
            viewport: Viewport::new(target),
            font_antialiasing: "grayscale glyph atlas",
            multisample_samples: 0,
            max_logical_font_px: 0.,
            max_physical_font_px: 0.,
            cached_glyphs: 0,
            glyph_tessellations: 0,
            glyph_ms: 0.,
            vector_load_ms: 0.,
            image_load_ms: 0.,
            glyph_failures: 0,
            missing_glyphs: 0,
            missing_images: 0,
            glyph_geometry_bytes: 0,
            cached_text_runs: 0,
            text_geometry_bytes: 0,
            text_triangles: 0,
            artwork_bytes: 0,
            vector_meshes: 0,
            vector_bytes: 0,
            vector_triangles: 0,
            high_resolution_wallpaper: false,
            draw_calls: 0,
            cpu_submission_ms: 0.,
            redrawn_fraction: 1.,
        }
    }
}

enum BatchKind {
    Paint {
        texture: GLuint,
        start: usize,
        count: usize,
        backdrop: Option<[[u8; 3]; 2]>,
    },
    Vector {
        path: String,
        rect: Rect,
        tint: Color,
    },
}

struct Batch {
    kind: BatchKind,
    clip: Option<Rect>,
    // [top fade-in, top end, bottom start, bottom fade-out, left ×2, right ×2]; 0 = off.
    fade: Option<[f32; 8]>,
}

fn full_screen(rect: &Rect) -> bool {
    rect.x <= 0. && rect.y <= 0. && rect.x + rect.w >= 1920. && rect.y + rect.h >= 1080.
}

/// An opaque, untinted, square full-screen image: the wallpaper that scrims can fold into.
fn plain_wallpaper(command: &Draw) -> Option<&str> {
    match command {
        Draw::Image(path, rect, radius, false) if full_screen(rect) && *radius <= 0. => Some(path),
        Draw::TintedImage(path, rect, radius, false, tint)
            if full_screen(rect) && *radius <= 0. && *tint == [1.; 4] =>
        {
            Some(path)
        }
        _ => None,
    }
}

/// A full-screen translucent fill without corners or border: a flat colour or a vertical
/// gradient (straight alpha, interpolated like the paint shader's vertex colours).
enum Scrim {
    Flat(Color),
    Gradient(Color, Color),
}
impl Scrim {
    fn of(command: &Draw) -> Option<Self> {
        match command {
            Draw::Shape(rect, radius, border, color)
                if full_screen(rect) && *radius <= 0. && *border <= 0. =>
            {
                Some(Self::Flat(*color))
            }
            Draw::RoundedGradient(rect, radius, top, bottom)
                if full_screen(rect) && *radius <= 0. =>
            {
                Some(Self::Gradient(*top, *bottom))
            }
            _ => None,
        }
    }
    fn at(&self, t: f32) -> Color {
        match self {
            Self::Flat(color) => *color,
            Self::Gradient(top, bottom) => {
                std::array::from_fn(|i| top[i] + (bottom[i] - top[i]) * t)
            }
        }
    }
}

fn intersect(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    let right = (a[0] + a[2]).min(b[0] + b[2]);
    let top = (a[1] + a[3]).min(b[1] + b[3]);
    [x, y, (right - x).max(0), (top - y).max(0)]
}

/// Square images skip the rounded-rectangle distance field: a negative radius selects the
/// shader's plain texture path. Per pixel that field cost as much as the rest of the Home scene
/// on a webOS 5 Mali for the full-screen wallpaper.
fn plain_radius(radius: f32) -> f32 {
    if radius > 0. { radius } else { -1. }
}

/// Measurement aid: LUMA_PROFILE_SKIP=wallpaper,image,app,shape,gradient,shadow,text or all
/// leaves those draws out, to find which part of the scene costs the GPU time.
/// Coverage → alpha. `LUMA_TEXT_GAMMA` above 1 lifts partial coverage, which thickens light
/// text on dark backgrounds the way Skia's contrast boost does; 1 is FreeType's linear coverage.
fn coverage_curve() -> &'static [u8; 256] {
    static CURVE: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    CURVE.get_or_init(|| {
        let gamma = std::env::var("LUMA_TEXT_GAMMA")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|gamma| (0.25..=4.).contains(gamma))
            .unwrap_or(1.);
        std::array::from_fn(|i| ((i as f32 / 255.).powf(1. / gamma) * 255. + 0.5) as u8)
    })
}

fn profile_skip(command: &Draw) -> bool {
    static SKIP: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let skip = SKIP.get_or_init(|| {
        std::env::var("LUMA_PROFILE_SKIP")
            .map(|value| value.split(',').map(str::to_owned).collect())
            .unwrap_or_default()
    });
    if skip.is_empty() {
        return false;
    }
    let kind = match command {
        Draw::Image(_, rect, ..) | Draw::TintedImage(_, rect, ..)
            if rect.w >= 1920. && rect.h >= 1080. =>
        {
            "wallpaper"
        }
        Draw::Image(..) | Draw::TintedImage(..) => "image",
        Draw::AppImage(..) => "app",
        Draw::Shape(..) => "shape",
        Draw::RoundedGradient(..) => "gradient",
        Draw::Shadow(..) => "shadow",
        Draw::Text(..) => "text",
        Draw::Clip(_) | Draw::EdgeFade(_) | Draw::EdgeFadeX(_) => return false,
    };
    skip.iter().any(|entry| entry == kind || entry == "all")
}

fn snap_rect(rect: Rect, scale: f32) -> Rect {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("LUMA_SNAP").as_deref() != Ok("0")) {
        return rect;
    }
    let scale = scale.max(0.01);
    let snap = |value: f32| (value * scale).round() / scale;
    let (x0, y0) = (snap(rect.x), snap(rect.y));
    // Never collapse a visible hairline.
    let x1 = snap(rect.x + rect.w).max(x0 + 1. / scale);
    let y1 = snap(rect.y + rect.h).max(y0 + 1. / scale);
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

pub struct Renderer {
    program: GLuint,
    vector_program: GLuint,
    buffer: GLuint,
    vao: GLuint,
    white: Texture,
    font_directory: PathBuf,
    /// LG Smart UI from the TV's firmware in place of the bundled Inter, when chosen.
    system_fonts: Option<PathBuf>,
    fonts: HashMap<u16, Font>,
    failed_fonts: HashSet<u16>,
    atlas: HashMap<GlyphKey, AtlasGlyph>,
    atlas_pages: Vec<AtlasPage>,
    hinting: crate::font::Hinting,
    text_runs: HashMap<TextKey, TextRun>,
    textures: HashMap<String, Option<Texture>>,
    vectors: HashMap<String, Option<vector::Mesh>>,
    artwork_variants: HashMap<String, String>,
    vertices: Vec<f32>,
    batches: Vec<Batch>,
    clip: Option<Rect>,
    // [top fade-in, top end, bottom start, bottom fade-out, left ×2, right ×2]; 0 = off.
    fade: Option<[f32; 8]>,
    frame: u64,
    stats: RenderStats,
    pub info: GraphicsInfo,
}

impl Renderer {
    fn set_fade(&mut self, offset: usize, fade: Option<[f32; 4]>) {
        let mut next = self.fade.unwrap_or([0.; 8]);
        next[offset..offset + 4].copy_from_slice(&fade.unwrap_or([0.; 4]));
        self.fade = (next != [0.; 8]).then_some(next);
    }

    pub fn new(font: &Path, gles: bool) -> Result<Self, String> {
        let directory = font.parent().ok_or("Invalid font path")?;
        let mut fonts = HashMap::new();
        // Open only the common face. Metrics and outlines stay lazy even when
        // the much larger CJK fallback is needed later.
        fonts.insert(400, load_font(directory, 400)?);
        let mut max_texture_size = 0;
        let mut max_viewport = [0; 2];
        let mut precision = 0;
        unsafe {
            gl::GetIntegerv(gl::MAX_TEXTURE_SIZE, &mut max_texture_size);
            gl::GetIntegerv(gl::MAX_VIEWPORT_DIMS, max_viewport.as_mut_ptr());
            if gles && gl::GetShaderPrecisionFormat::is_loaded() {
                let mut range = [0; 2];
                gl::GetShaderPrecisionFormat(
                    gl::FRAGMENT_SHADER,
                    gl::HIGH_FLOAT,
                    range.as_mut_ptr(),
                    &mut precision,
                );
            }
        }
        let fragment_highp = !gles || precision > 0;
        let qualifier = if fragment_highp { "highp" } else { "mediump" };
        let vertex_prefix = if gles {
            format!("#version 100\nprecision highp float;\n#define PAINT_PRECISION {qualifier}\n")
        } else {
            "#version 150\n#define attribute in\n#define varying out\n#define PAINT_PRECISION\n"
                .into()
        };
        let fragment_prefix = if gles {
            format!(
                "#version 100\nprecision {qualifier} float;\n#define PAINT_PRECISION {qualifier}\n"
            )
        } else {
            "#version 150\n#define varying in\n#define texture2D texture\nout vec4 out_color;\n#define gl_FragColor out_color\n#define PAINT_PRECISION\n".into()
        };
        let program = link(
            &(vertex_prefix.clone() + include_str!("shaders/paint.vert")),
            &(fragment_prefix.clone() + include_str!("shaders/paint.frag")),
            &[c"a_position", c"a_uv", c"a_color", c"a_shape", c"a_params"],
        )?;
        let vector_program = match link(
            &(vertex_prefix.clone() + include_str!("shaders/vector.vert")),
            &(fragment_prefix.clone() + include_str!("shaders/vector.frag")),
            &[c"a_position", c"a_offset", c"a_color", c"a_coverage"],
        ) {
            Ok(program) => program,
            Err(error) => {
                unsafe { gl::DeleteProgram(program) };
                return Err(error);
            }
        };
        let mut buffer = 0;
        let mut vao = 0;
        unsafe {
            if !gles {
                gl::GenVertexArrays(1, &mut vao);
                gl::BindVertexArray(vao);
            }
            gl::GenBuffers(1, &mut buffer);
            gl::Enable(gl::BLEND);
            // Preserve an opaque destination alpha. Using BlendFunc for both
            // channels makes translucent UI darken again in PNG/window compositing.
            gl::BlendFuncSeparate(
                gl::SRC_ALPHA,
                gl::ONE_MINUS_SRC_ALPHA,
                gl::ONE,
                gl::ONE_MINUS_SRC_ALPHA,
            );
            gl::Disable(gl::DEPTH_TEST);
            gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
        }
        let white = texture(1, 1, &[255; 4]);
        Ok(Self {
            program,
            vector_program,
            buffer,
            vao,
            white,
            font_directory: directory.into(),
            system_fonts: None,
            fonts,
            failed_fonts: HashSet::new(),
            atlas: HashMap::new(),
            atlas_pages: Vec::new(),
            hinting: crate::font::Hinting::Off,
            text_runs: HashMap::new(),
            textures: HashMap::new(),
            vectors: HashMap::new(),
            artwork_variants: HashMap::new(),
            vertices: Vec::with_capacity(65536),
            batches: Vec::new(),
            clip: None,
            fade: None,
            frame: 0,
            stats: RenderStats::new(Size::HD),
            info: GraphicsInfo {
                vendor: gl_string(gl::VENDOR),
                renderer: gl_string(gl::RENDERER),
                version: gl_string(gl::VERSION),
                max_texture_size: max_texture_size.max(0) as u32,
                max_viewport,
                fragment_highp,
            },
        })
    }

    fn ensure_font(&mut self, id: u16) -> bool {
        if self.fonts.contains_key(&id) {
            return true;
        }
        if self.failed_fonts.contains(&id) {
            return false;
        }
        let system = self
            .system_fonts
            .as_deref()
            .filter(|_| id < 2000)
            .map(|directory| crate::system_font::path(directory, id % 1000));
        let loaded = match system {
            Some(path) => load_font_file(&path).or_else(|error| {
                eprintln!("{error}; using the bundled font");
                load_font(&self.font_directory, id)
            }),
            None => load_font(&self.font_directory, id),
        };
        match loaded {
            Ok(font) => {
                self.fonts.insert(id, font);
                true
            }
            Err(error) => {
                eprintln!("{error}");
                self.failed_fonts.insert(id);
                false
            }
        }
    }

    /// Switching between the bundled and the TV's own typeface rebuilds every text run.
    pub fn set_system_fonts(&mut self, directory: Option<PathBuf>) -> bool {
        if self.system_fonts == directory {
            return false;
        }
        self.system_fonts = directory;
        // The CJK and Hangul fallbacks (ids 2000+) are bundled either way.
        self.fonts.retain(|id, _| *id >= 2000);
        self.failed_fonts.clear();
        self.clear_atlas();
        self.text_runs.clear();
        true
    }

    /// Switching grid fitting rebuilds every glyph on next use; layout is unchanged.
    pub fn set_hinting(&mut self, hinting: crate::font::Hinting) -> bool {
        if self.hinting == hinting {
            return false;
        }
        self.hinting = hinting;
        self.clear_atlas();
        true
    }

    fn clear_atlas(&mut self) {
        self.atlas.clear();
        self.atlas_pages.clear();
    }

    /// The glyph's place in the atlas, rasterizing and uploading it on first use.
    fn atlas_glyph(&mut self, key: GlyphKey) -> Option<AtlasGlyph> {
        if let Some(glyph) = self.atlas.get(&key) {
            return Some(*glyph);
        }
        let started = Instant::now();
        let bitmap = self.fonts.get(&key.font)?.rasterize(
            key.c,
            key.ppem as f32 / 64.,
            self.hinting,
            f32::from(key.subpixel) / SUBPIXEL_STEPS,
        );
        self.stats.glyph_tessellations += 1;
        let mut glyph = AtlasGlyph {
            page: 0,
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            left: bitmap.left,
            top: bitmap.top,
        };
        if bitmap.width > 0 {
            // A transparent pixel on every side keeps neighbours out of filtered samples.
            let (width, height) = (bitmap.width + 2, bitmap.height + 2);
            if width > ATLAS_SIZE || height > ATLAS_SIZE {
                return None;
            }
            let (page, x, y) = self.atlas_allocate(width, height);
            let curve = coverage_curve();
            let mut pixels = vec![0u8; (width * height * 4) as usize];
            for row in 0..bitmap.height {
                for column in 0..bitmap.width {
                    let coverage = bitmap.coverage[(row * bitmap.width + column) as usize];
                    let at = (((row + 1) * width + column + 1) * 4) as usize;
                    pixels[at..at + 4].copy_from_slice(&[255, 255, 255, curve[coverage as usize]]);
                }
            }
            unsafe {
                gl::BindTexture(gl::TEXTURE_2D, self.atlas_pages[page].texture.id);
                gl::TexSubImage2D(
                    gl::TEXTURE_2D,
                    0,
                    x as i32,
                    y as i32,
                    width as i32,
                    height as i32,
                    gl::RGBA,
                    gl::UNSIGNED_BYTE,
                    pixels.as_ptr().cast(),
                );
            }
            glyph = AtlasGlyph {
                page,
                x: x + 1,
                y: y + 1,
                width: bitmap.width,
                height: bitmap.height,
                ..glyph
            };
        }
        self.stats.glyph_ms += started.elapsed().as_secs_f64() * 1000.;
        self.atlas.insert(key, glyph);
        Some(glyph)
    }

    /// Shelf packing: fill the current row, then start a row below, then a new page.
    fn atlas_allocate(&mut self, width: u32, height: u32) -> (usize, u32, u32) {
        if let Some(page) = self.atlas_pages.last_mut() {
            if page.x + width > ATLAS_SIZE {
                page.x = 0;
                page.y += page.row;
                page.row = 0;
            }
            if page.y + height <= ATLAS_SIZE {
                let position = (page.x, page.y);
                page.x += width;
                page.row = page.row.max(height);
                return (self.atlas_pages.len() - 1, position.0, position.1);
            }
        }
        let texture = texture(
            ATLAS_SIZE,
            ATLAS_SIZE,
            &vec![0; (ATLAS_SIZE * ATLAS_SIZE * 4) as usize],
        );
        unsafe {
            // Glyph quads map one texel to one drawable pixel; never blend neighbours.
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        }
        self.atlas_pages.push(AtlasPage {
            texture,
            x: width,
            y: 0,
            row: height,
        });
        (self.atlas_pages.len() - 1, 0, 0)
    }

    fn trim_text_geometry(&mut self) {
        let run_bytes = |run: &TextRun| run.glyphs.len() * size_of::<Placed>();
        let mut bytes: usize = self.text_runs.values().map(run_bytes).sum();
        if bytes > TEXT_RUN_BUDGET || self.text_runs.len() > 512 {
            let mut old: Vec<_> = self
                .text_runs
                .iter()
                .filter(|(_, run)| run.frame != self.frame)
                .map(|(key, run)| (key.clone(), run.frame))
                .collect();
            old.sort_by_key(|(_, frame)| *frame);
            for (key, _) in old {
                if bytes <= TEXT_RUN_BUDGET && self.text_runs.len() <= 512 {
                    break;
                }
                if let Some(run) = self.text_runs.remove(&key) {
                    bytes -= run_bytes(&run);
                }
            }
        }
        self.stats.cached_glyphs = self.atlas.len();
        self.stats.glyph_geometry_bytes =
            self.atlas_pages.len() * (ATLAS_SIZE * ATLAS_SIZE * 4) as usize;
        self.stats.cached_text_runs = self.text_runs.len();
        self.stats.text_geometry_bytes = self.text_runs.values().map(run_bytes).sum();
    }

    fn quad(
        &mut self,
        rect: Rect,
        uv: [f32; 4],
        radius: f32,
        border: f32,
        color: Color,
        texture: GLuint,
    ) {
        self.gradient_quad(rect, uv, radius, border, [color, color], texture, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn gradient_quad(
        &mut self,
        rect: Rect,
        uv: [f32; 4],
        radius: f32,
        border: f32,
        colors: [Color; 2],
        texture: GLuint,
        backdrop: Option<[[u8; 3]; 2]>,
    ) {
        if rect.w <= 0. || rect.h <= 0. {
            return;
        }
        let rect = self.snap(rect);
        let start = self.vertices.len() / 14;
        for (x, y) in [(0., 0.), (1., 0.), (0., 1.), (0., 1.), (1., 0.), (1., 1.)] {
            let color = colors[y as usize];
            self.vertices.extend_from_slice(&[
                rect.x + x * rect.w,
                rect.y + y * rect.h,
                uv[0] + x * (uv[2] - uv[0]),
                uv[1] + y * (uv[3] - uv[1]),
                color[0],
                color[1],
                color[2],
                color[3],
                (x - 0.5) * rect.w,
                (y - 0.5) * rect.h,
                rect.w / 2.,
                rect.h / 2.,
                radius,
                border,
            ]);
        }
        self.paint_batch(start, 6, texture, backdrop);
    }

    /// Appends `count` paint vertices from `start`, merged into the previous batch when it uses
    /// the same texture, clip and fade.
    fn paint_batch(
        &mut self,
        start: usize,
        count: usize,
        texture: GLuint,
        backdrop: Option<[[u8; 3]; 2]>,
    ) {
        if let Some(batch) = self.batches.last_mut() {
            let same_clip = match (batch.clip, self.clip) {
                (None, None) => true,
                (Some(a), Some(b)) => a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h,
                _ => false,
            };
            if same_clip
                && batch.fade == self.fade
                && let BatchKind::Paint {
                    texture: previous,
                    count: batch_count,
                    backdrop: previous_backdrop,
                    ..
                } = &mut batch.kind
                && *previous == texture
                && *previous_backdrop == backdrop
            {
                *batch_count += count;
                return;
            }
        }
        self.batches.push(Batch {
            kind: BatchKind::Paint {
                texture,
                start,
                count,
                backdrop,
            },
            clip: self.clip,
            fade: self.fade,
        });
    }

    /// The wallpaper with full-screen translucent fills folded in: each fill maps a pixel p to
    /// p * (1 - a) + c * a, so together they are p * m + k, drawn in one pass instead of one
    /// full-screen layer each. Strips keep gradients' products of colour and alpha exact.
    fn folded_quad(&mut self, rect: Rect, texture: GLuint, scrims: &[Scrim]) {
        const STRIPS: usize = 16;
        let fold = |t: f32| {
            let (mut m, mut k) = ([1f32; 3], [0f32; 3]);
            for scrim in scrims {
                let color = scrim.at(t);
                for i in 0..3 {
                    m[i] *= 1. - color[3];
                    k[i] = k[i] * (1. - color[3]) + color[i] * color[3];
                }
            }
            (m, k)
        };
        let start = self.vertices.len() / 14;
        for strip in 0..STRIPS {
            let rows = [
                strip as f32 / STRIPS as f32,
                (strip + 1) as f32 / STRIPS as f32,
            ];
            let folds = rows.map(fold);
            for (x, row) in [(0., 0), (1., 0), (0., 1), (0., 1), (1., 0), (1., 1)] {
                let (m, k) = folds[row];
                let t = rows[row];
                self.vertices.extend_from_slice(&[
                    rect.x + x * rect.w,
                    rect.y + t * rect.h,
                    x,
                    t,
                    m[0],
                    m[1],
                    m[2],
                    1.,
                    k[0],
                    k[1],
                    k[2],
                    0.,
                    -2.,
                    0.,
                ]);
            }
        }
        self.paint_batch(start, STRIPS * 6, texture, None);
    }

    fn text(&mut self, text: &str, rect: Rect, style: TextStyle, color: Color, centered: bool) {
        // Layout remains logical and is reused at every density; glyph bitmaps are made for
        // the drawable's pixels when the run is drawn.
        let size_f = style.size;
        let viewport = self.stats.viewport;
        self.stats.max_logical_font_px = self.stats.max_logical_font_px.max(size_f);
        self.stats.max_physical_font_px = self
            .stats
            .max_physical_font_px
            .max(size_f * viewport.text_scale());
        let padded = rect.inset(-size_f);
        if padded.x + padded.w < 0.
            || padded.y + padded.h < 0.
            || padded.x > 1920.
            || padded.y > 1080.
            || self.clip.is_some_and(|clip| {
                padded.x + padded.w < clip.x
                    || padded.y + padded.h < clip.y
                    || padded.x > clip.x + clip.w
                    || padded.y > clip.y + clip.h
            })
        {
            return;
        }
        let origin = [rect.x, rect.y];
        let key = TextKey {
            text: text.into(),
            weight: style.weight,
            layout: [
                size_f.to_bits(),
                style.tracking.to_bits(),
                style.line_height.to_bits(),
                rect.w.to_bits(),
                rect.h.to_bits(),
            ],
            flags: u8::from(style.align_right) | (u8::from(centered) << 1),
        };
        if self.text_runs.contains_key(&key) {
            self.submit_text(key, origin, color);
            return;
        }
        let rect = Rect::new(0., 0., rect.w, rect.h);
        let font_id = style.weight + if style.size >= 32. { 1000 } else { 0 };
        let cjk_id = if style.weight >= 500 { 2500 } else { 2400 };
        let hangul_id = cjk_id + 1000;
        if !self.ensure_font(font_id) {
            self.stats.glyph_failures += text.chars().count();
            return;
        }
        let (mut needs_cjk, mut needs_hangul) = (false, false);
        for c in text.chars() {
            if !c.is_ascii_control() && self.fonts[&font_id].lookup_glyph_index(c) == 0 {
                if is_hangul(c) {
                    needs_hangul = true;
                } else {
                    needs_cjk = true;
                }
            }
        }
        // Each fallback loads only when a run needs it, so Korean text never
        // pulls in the larger Chinese font and vice versa.
        let cjk_ready = needs_cjk && self.ensure_font(cjk_id);
        let hangul_ready = needs_hangul && self.ensure_font(hangul_id);
        let fonts = &self.fonts;
        let choose_font = |c: char| {
            if fonts[&font_id].lookup_glyph_index(c) != 0 {
                font_id
            } else if is_hangul(c) {
                if hangul_ready { hangul_id } else { font_id }
            } else if cjk_ready {
                cjk_id
            } else {
                font_id
            }
        };
        let font = &fonts[&font_id];
        let line_height = size_f * style.line_height;
        let baseline = font.horizontal_line_metrics(size_f).map_or(size_f, |m| {
            (line_height - m.ascent + m.descent) / 2. + m.ascent
        });
        // LG Smart UI is spaced for TV distance; drop Inter's negative tracking (Assets::tracking).
        let tracking = if self.system_fonts.is_some() {
            style.tracking.max(0.)
        } else {
            style.tracking
        } * size_f;
        let advance = |c| {
            if c == '\n' {
                0.
            } else {
                fonts[&choose_font(c)].metrics(c, size_f).advance_width + tracking
            }
        };
        let mut lines: Vec<String> = Vec::new();
        let mut line = String::new();
        let mut width = 0.;
        // Word wrapping, with character fallback for a title wider than the tile.
        for word in text.split_inclusive(char::is_whitespace) {
            let word_width: f32 = word.chars().map(advance).sum();
            if width > 0. && width + word_width > rect.w {
                lines.push(line.trim_end().into());
                line.clear();
                width = 0.;
            }
            for c in word.chars() {
                if c == '\n' {
                    lines.push(line.trim_end().into());
                    line.clear();
                    width = 0.;
                    continue;
                }
                let w = advance(c);
                if width > 0. && width + w > rect.w {
                    lines.push(line.trim_end().into());
                    line.clear();
                    width = 0.;
                }
                line.push(c);
                width += w;
            }
        }
        if !line.is_empty() {
            lines.push(line.trim_end().into());
        }
        let mut placed = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            let top = rect.y + index as f32 * line_height;
            if index > 0 && top + line_height > rect.y + rect.h + 1. {
                break;
            }
            let mut previous = None;
            let width: f32 = line
                .chars()
                .map(|c| {
                    let kern = previous
                        .and_then(|p| font.horizontal_kern(p, c, size_f))
                        .unwrap_or(0.);
                    previous = Some(c);
                    advance(c) + kern
                })
                .sum();
            let mut x = rect.x
                + if style.align_right {
                    rect.w - width
                } else if centered {
                    (rect.w - width) / 2.
                } else {
                    0.
                };
            let mut previous = None;
            for c in line.chars() {
                if let Some(previous) = previous {
                    x += font.horizontal_kern(previous, c, size_f).unwrap_or(0.);
                }
                let selected = choose_font(c);
                let baseline = if selected == font_id {
                    baseline
                } else {
                    fonts[&selected]
                        .horizontal_line_metrics(size_f)
                        .map_or(baseline, |m| {
                            (line_height - m.ascent + m.descent) / 2. + m.ascent
                        })
                };
                placed.push((c, x, top + baseline, selected));
                x += advance(c);
                previous = Some(c);
            }
        }
        let mut missing_glyphs = 0;
        let glyphs = placed
            .into_iter()
            .filter_map(|(c, x, baseline, font)| {
                if self.fonts[&font].lookup_glyph_index(c) == 0 && !c.is_whitespace() {
                    missing_glyphs += 1;
                }
                (!c.is_whitespace()).then_some(Placed {
                    c,
                    font,
                    x,
                    baseline,
                })
            })
            .collect();
        self.text_runs.insert(
            key.clone(),
            TextRun {
                glyphs,
                missing_glyphs,
                failures: 0,
                frame: self.frame,
            },
        );
        self.submit_text(key, origin, color);
    }

    /// Draws a run's glyphs from the atlas: each quad covers whole drawable pixels, with the
    /// pen position quantized to a quarter pixel horizontally and the baseline to a pixel.
    fn submit_text(&mut self, key: TextKey, origin: [f32; 2], tint: Color) {
        let run = self.text_runs.get_mut(&key).unwrap();
        run.frame = self.frame;
        self.stats.missing_glyphs += run.missing_glyphs;
        self.stats.glyph_failures += run.failures;
        let glyphs = std::mem::take(&mut run.glyphs);
        let viewport = self.stats.viewport;
        let (sx, sy) = (
            viewport.width as f32 / 1920.,
            viewport.height as f32 / 1080.,
        );
        let ppem = (f32::from_bits(key.layout[0]) * viewport.text_scale() * 64.).round() as u32;
        let subpixel = self.hinting != crate::font::Hinting::Normal;
        let mut failures = 0;
        for placed in &glyphs {
            let pen = (origin[0] + placed.x) * sx;
            let (mut pixel, mut step) = (pen.floor(), 0.);
            if subpixel {
                step = ((pen - pixel) * SUBPIXEL_STEPS).round();
                if step >= SUBPIXEL_STEPS {
                    (pixel, step) = (pixel + 1., 0.);
                }
            } else {
                pixel = pen.round();
            }
            let baseline = ((origin[1] + placed.baseline) * sy).round();
            let Some(glyph) = self.atlas_glyph(GlyphKey {
                c: placed.c,
                font: placed.font,
                ppem,
                subpixel: step as u8,
            }) else {
                failures += 1;
                continue;
            };
            if glyph.width == 0 {
                continue;
            }
            let size = ATLAS_SIZE as f32;
            self.quad(
                Rect::new(
                    (pixel + glyph.left as f32) / sx,
                    (baseline + glyph.top as f32) / sy,
                    glyph.width as f32 / sx,
                    glyph.height as f32 / sy,
                ),
                [
                    glyph.x as f32 / size,
                    glyph.y as f32 / size,
                    (glyph.x + glyph.width) as f32 / size,
                    (glyph.y + glyph.height) as f32 / size,
                ],
                -1.,
                0.,
                tint,
                self.atlas_pages[glyph.page].texture.id,
            );
            self.stats.text_triangles += 2;
        }
        self.stats.glyph_failures += failures;
        if let Some(run) = self.text_runs.get_mut(&key) {
            run.glyphs = glyphs;
        }
    }

    fn image(&mut self, path: &str, rect: Rect, radius: f32, contain: bool, tint: Color) {
        self.folded_image(path, rect, radius, contain, tint, &[]);
    }

    fn folded_image(
        &mut self,
        path: &str,
        mut rect: Rect,
        radius: f32,
        contain: bool,
        tint: Color,
        scrims: &[Scrim],
    ) {
        if path.ends_with(".lvec") {
            self.vector_image(path, rect, contain, tint);
            return;
        }
        let path = if self.stats.viewport.scale > 1.
            && self.info.max_texture_size >= 3840
            && Path::new(path)
                .ancestors()
                .skip(1)
                .take(2)
                .any(|p| p.file_name().is_some_and(|p| p == "wallpapers"))
        {
            self.artwork_variants
                .entry(path.into())
                .or_insert_with(|| {
                    let original = Path::new(path);
                    let variant = original.with_file_name(format!(
                        "{}@2x.jpg",
                        original.file_stem().unwrap_or_default().to_string_lossy()
                    ));
                    if variant.is_file() {
                        variant.to_string_lossy().into_owned()
                    } else {
                        path.into()
                    }
                })
                .clone()
        } else {
            path.into()
        };
        if path.ends_with("@2x.jpg") {
            self.stats.high_resolution_wallpaper = true;
        }
        let frame = self.frame;
        let started = Instant::now();
        let loaded = !self.textures.contains_key(&path);
        let entry = self
            .textures
            .entry(path.clone())
            .or_insert_with(|| match image::open(&path) {
                Ok(image) => {
                    let image = image.into_rgba8();
                    Some(texture(image.width(), image.height(), image.as_raw()))
                }
                Err(error) => {
                    eprintln!("Artwork {path}: {error}");
                    None
                }
            });
        if loaded {
            self.stats.image_load_ms += started.elapsed().as_secs_f64() * 1000.;
        }
        let Some(texture) = entry else {
            self.stats.missing_images += 1;
            return;
        };
        texture.frame = frame;
        let id = texture.id;
        if contain {
            let scale = (rect.w / texture.width as f32).min(rect.h / texture.height as f32);
            let w = texture.width as f32 * scale;
            let h = texture.height as f32 * scale;
            rect = Rect::new(rect.x + (rect.w - w) / 2., rect.y + (rect.h - h) / 2., w, h);
        }
        if scrims.is_empty() {
            self.quad(rect, [0., 0., 1., 1.], plain_radius(radius), 0., tint, id);
        } else {
            self.folded_quad(rect, id, scrims);
        }
    }

    /// Edges on whole drawable pixels. The TV scales its 1080p graphics plane up 2x, which
    /// turns a half-covered edge pixel into a soft 2-pixel band; whole pixels stay crisp.
    /// LUMA_SNAP=0 keeps logical positions (for comparison).
    fn snap(&self, rect: Rect) -> Rect {
        snap_rect(rect, self.stats.viewport.scale)
    }

    fn vector_image(&mut self, path: &str, mut rect: Rect, contain: bool, tint: Color) {
        if rect.w <= 0. || rect.h <= 0. {
            return;
        }
        let scale = self.stats.viewport.scale;
        let started = Instant::now();
        let loaded = !self.vectors.contains_key(path);
        let mesh = self.vectors.entry(path.into()).or_insert_with(|| {
            match vector::Mesh::load(Path::new(path)) {
                Ok(mesh) => Some(mesh),
                Err(error) => {
                    eprintln!("{error}");
                    None
                }
            }
        });
        if loaded {
            self.stats.vector_load_ms += started.elapsed().as_secs_f64() * 1000.;
        }
        let Some(mesh) = mesh else {
            self.stats.missing_images += 1;
            return;
        };
        if contain {
            let scale = (rect.w / mesh.width).min(rect.h / mesh.height);
            let width = mesh.width * scale;
            let height = mesh.height * scale;
            rect = Rect::new(
                rect.x + (rect.w - width) / 2.,
                rect.y + (rect.h - height) / 2.,
                width,
                height,
            );
        }
        // Logos keep their size; only their placement moves onto the pixel grid.
        let snapped = snap_rect(rect, scale);
        rect = Rect::new(snapped.x, snapped.y, rect.w, rect.h);
        let clip = if let Some(clip) = self.clip {
            let x = rect.x.max(clip.x);
            let y = rect.y.max(clip.y);
            Rect::new(
                x,
                y,
                (rect.x + rect.w).min(clip.x + clip.w).max(x) - x,
                (rect.y + rect.h).min(clip.y + clip.h).max(y) - y,
            )
        } else {
            rect
        };
        if clip.w <= 0. || clip.h <= 0. {
            return;
        }
        self.stats.vector_triangles += mesh.vertices as usize / 3;
        self.batches.push(Batch {
            kind: BatchKind::Vector {
                path: path.into(),
                rect,
                tint,
            },
            clip: Some(clip),
            fade: self.fade,
        });
    }

    fn app_image(
        &mut self,
        image: &crate::app_artwork::Artwork,
        rect: Rect,
        radius: f32,
        tint: Color,
    ) {
        let entry = self.textures.entry(image.key.clone()).or_insert_with(|| {
            Some(texture(
                image.pixels.width(),
                image.pixels.height(),
                image.pixels.as_raw(),
            ))
        });
        let Some(texture) = entry else {
            return;
        };
        texture.frame = self.frame;
        let scale =
            (rect.w * 0.68 / texture.width as f32).min(rect.h * 0.68 / texture.height as f32);
        let w = texture.width as f32 * scale;
        let h = texture.height as f32 * scale;
        let id = texture.id;
        self.gradient_quad(
            rect,
            [
                0.5 - rect.w / (2. * w),
                0.5 - rect.h / (2. * h),
                0.5 + rect.w / (2. * w),
                0.5 + rect.h / (2. * h),
            ],
            radius,
            0.,
            [tint; 2],
            id,
            Some(image.backdrop),
        );
    }

    /// A logical damage rect in drawable pixels ([x, y, width, height], origin bottom-left),
    /// grown to whole pixels.
    pub fn damage_pixels(rect: Rect, width: u32, height: u32) -> [i32; 4] {
        let viewport = Viewport::new(Size::new(width, height));
        let sx = viewport.width as f32 / 1920.;
        let sy = viewport.height as f32 / 1080.;
        let x0 = (viewport.x as f32 + rect.x.max(0.) * sx).floor() as i32;
        let x1 = (viewport.x as f32 + (rect.x + rect.w).min(1920.) * sx).ceil() as i32;
        let y0 = (viewport.y as f32 + (1080. - (rect.y + rect.h).min(1080.)) * sy).floor() as i32;
        let y1 = (viewport.y as f32 + (1080. - rect.y.max(0.)) * sy).ceil() as i32;
        if x1 <= x0 || y1 <= y0 {
            // No change still names, clears and redraws one pixel inside the drawable. An empty
            // or off-surface region left the driver nothing to preserve: Mali presented black.
            return [viewport.x, viewport.y, 1, 1];
        }
        [x0, y0, x1 - x0, y1 - y0]
    }

    /// The logical area drawable pixels from `damage_pixels` cover. Grown to whole pixels it
    /// can exceed the original logical damage; commands are culled against this area, so none
    /// that paints into the cleared pixels is skipped.
    pub fn damage_area(pixels: [i32; 4], width: u32, height: u32) -> Rect {
        let viewport = Viewport::new(Size::new(width, height));
        let sx = viewport.width as f32 / 1920.;
        let sy = viewport.height as f32 / 1080.;
        let [x, y, w, h] = pixels;
        Rect::new(
            (x - viewport.x) as f32 / sx,
            1080. - (y + h - viewport.y) as f32 / sy,
            w as f32 / sx,
            h as f32 / sy,
        )
    }

    /// Prepares first-use work for a scene without drawing it (text layout and glyph
    /// outlines, logo meshes, images), e.g. another page while the screen is idle, so its
    /// first appearance does not pay for it. The visible frame's statistics stay unchanged.
    pub fn prepare(&mut self, commands: &[Draw], width: u32, height: u32) {
        let stats = self.stats.clone();
        self.build(commands, width, height, None);
        self.vertices.clear();
        self.batches.clear();
        self.stats = stats;
    }

    /// Lays out the scene into vertices and batches; commands entirely outside `area` are
    /// skipped.
    fn build(&mut self, commands: &[Draw], width: u32, height: u32, area: Option<Rect>) {
        self.stats = RenderStats::new(Size::new(width, height));
        // Captures can bind a single-sample FBO even when the actual TV window
        // is multisampled, so query the active target rather than assume.
        let mut samples = 0;
        unsafe { gl::GetIntegerv(gl::SAMPLES, &mut samples) };
        self.stats.multisample_samples = samples.max(0) as u32;
        if self.atlas_pages.len() > ATLAS_PAGES {
            self.clear_atlas();
        }
        self.frame += 1;
        self.vertices.clear();
        self.batches.clear();
        self.clip = None;
        self.fade = None;
        let mut index = 0;
        while index < commands.len() {
            let command = &commands[index];
            index += 1;
            if profile_skip(command) {
                continue;
            }
            if let Some(path) = plain_wallpaper(command)
                && self.clip.is_none()
                && self.fade.is_none()
            {
                let mut scrims = Vec::new();
                while let Some(next) = commands.get(index) {
                    if profile_skip(next) {
                        index += 1;
                    } else if let Some(scrim) = Scrim::of(next) {
                        scrims.push(scrim);
                        index += 1;
                    } else {
                        break;
                    }
                }
                if !scrims.is_empty() {
                    let screen = Rect::new(0., 0., 1920., 1080.);
                    self.folded_image(path, screen, 0., false, [1.; 4], &scrims);
                    continue;
                }
            }
            // Outside the damage the scissor would discard it anyway; skip its layout too.
            if let (Some(area), Some(bounds)) = (area, crate::damage::bounds(command))
                && !crate::damage::intersects(area, bounds)
            {
                continue;
            }
            match command {
                Draw::Shape(rect, radius, border, color) => self.quad(
                    *rect,
                    [0., 0., 1., 1.],
                    *radius,
                    *border,
                    *color,
                    self.white.id,
                ),
                Draw::RoundedGradient(rect, radius, top, bottom) => self.gradient_quad(
                    *rect,
                    [0., 0., 1., 1.],
                    *radius,
                    0.,
                    [*top, *bottom],
                    self.white.id,
                    None,
                ),
                Draw::Shadow(rect, radius, blur, color) => self.quad(
                    rect.inset(-*blur),
                    [0., 0., 1., 1.],
                    *radius,
                    -*blur,
                    *color,
                    self.white.id,
                ),
                Draw::Text(text, rect, size, color, centered) => {
                    self.text(text, *rect, *size, *color, *centered)
                }
                Draw::Image(path, rect, radius, contain) => {
                    self.image(path, *rect, *radius, *contain, [1.; 4])
                }
                Draw::TintedImage(path, rect, radius, contain, tint) => {
                    self.image(path, *rect, *radius, *contain, *tint)
                }
                Draw::AppImage(image, rect, radius, tint) => {
                    self.app_image(image, *rect, *radius, *tint)
                }
                Draw::Clip(clip) => self.clip = *clip,
                Draw::EdgeFade(fade) => self.set_fade(0, *fade),
                Draw::EdgeFadeX(fade) => self.set_fade(4, *fade),
            }
        }
    }

    /// Draws the scene. With `damage` (drawable pixels from `damage_pixels`), only those pixels
    /// are cleared and drawn; commands entirely outside them are skipped, and the rest of the
    /// back buffer keeps its earlier contents.
    pub fn render(&mut self, commands: &[Draw], width: u32, height: u32, damage: Option<[i32; 4]>) {
        let area = damage.map(|pixels| Self::damage_area(pixels, width, height));
        let started = Instant::now();
        self.build(commands, width, height, area);
        let Viewport {
            x: vx,
            y: vy,
            width: vw,
            height: vh,
            scale,
        } = self.stats.viewport;
        if let Some([_, _, w, h]) = damage {
            self.stats.redrawn_fraction = f64::from(w * h) / f64::from((width * height).max(1));
        }
        let scissor = |area: Option<[i32; 4]>| unsafe {
            match area {
                Some([x, y, w, h]) => {
                    gl::Enable(gl::SCISSOR_TEST);
                    gl::Scissor(x, y, w.max(0), h.max(0));
                }
                None => gl::Disable(gl::SCISSOR_TEST),
            }
        };
        unsafe {
            gl::Viewport(0, 0, width as i32, height as i32);
            scissor(damage);
            gl::ClearColor(0.04, 0.04, 0.05, 1.);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::Viewport(vx, vy, vw, vh);
            if self.vao != 0 {
                gl::BindVertexArray(self.vao);
            }
            gl::BindBuffer(gl::ARRAY_BUFFER, self.buffer);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (self.vertices.len() * size_of::<f32>()) as isize,
                self.vertices.as_ptr().cast(),
                gl::DYNAMIC_DRAW,
            );
            gl::UseProgram(self.program);
            // One drawable pixel in logical units: the paint shader's edge filter width.
            gl::Uniform1f(
                gl::GetUniformLocation(self.program, c"u_aa".as_ptr()),
                1. / scale.max(0.01),
            );
            gl::Uniform1i(
                gl::GetUniformLocation(self.program, c"u_texture".as_ptr()),
                0,
            );
            gl::ActiveTexture(gl::TEXTURE0);
            let mut paint_bound = false;
            let mut current_fade = None;
            for batch in &self.batches {
                if current_fade != Some(batch.fade) {
                    // The same cheap alpha mask applies to shapes, textures,
                    // icons and text; the unmasked wallpaper remains beneath.
                    let fade = batch.fade.unwrap_or([0.; 8]);
                    for program in [self.program, self.vector_program] {
                        gl::UseProgram(program);
                        gl::Uniform4f(
                            gl::GetUniformLocation(program, c"u_edge_fade".as_ptr()),
                            fade[0],
                            fade[1],
                            fade[2],
                            fade[3],
                        );
                        gl::Uniform4f(
                            gl::GetUniformLocation(program, c"u_edge_fade_x".as_ptr()),
                            fade[4],
                            fade[5],
                            fade[6],
                            fade[7],
                        );
                    }
                    current_fade = Some(batch.fade);
                    paint_bound = false;
                }
                let clip = batch.clip.map(|clip| {
                    [
                        vx + (clip.x * vw as f32 / 1920.).round() as i32,
                        vy + ((1080. - clip.y - clip.h) * vh as f32 / 1080.).round() as i32,
                        (clip.w * vw as f32 / 1920.).round() as i32,
                        (clip.h * vh as f32 / 1080.).round() as i32,
                    ]
                });
                scissor(match (clip, damage) {
                    (Some(a), Some(b)) => Some(intersect(a, b)),
                    (a, b) => a.or(b),
                });
                match &batch.kind {
                    BatchKind::Paint {
                        texture,
                        start,
                        count,
                        backdrop,
                    } => {
                        if !paint_bound {
                            gl::UseProgram(self.program);
                            gl::BindBuffer(gl::ARRAY_BUFFER, self.buffer);
                            let mut offset = 0;
                            for (index, size) in [2, 2, 4, 4, 2].into_iter().enumerate() {
                                gl::EnableVertexAttribArray(index as u32);
                                gl::VertexAttribPointer(
                                    index as u32,
                                    size,
                                    gl::FLOAT,
                                    gl::FALSE,
                                    56,
                                    (offset * size_of::<f32>()) as *const _,
                                );
                                offset += size as usize;
                            }
                            paint_bound = true;
                        }
                        gl::BindTexture(gl::TEXTURE_2D, *texture);
                        gl::Uniform1i(
                            gl::GetUniformLocation(self.program, c"u_app".as_ptr()),
                            i32::from(backdrop.is_some()),
                        );
                        if let Some(stops) = backdrop {
                            for (name, rgb) in
                                [(c"u_app_top", stops[0]), (c"u_app_bottom", stops[1])]
                            {
                                gl::Uniform3f(
                                    gl::GetUniformLocation(self.program, name.as_ptr()),
                                    rgb[0] as f32 / 255.,
                                    rgb[1] as f32 / 255.,
                                    rgb[2] as f32 / 255.,
                                );
                            }
                        }
                        gl::DrawArrays(gl::TRIANGLES, *start as i32, *count as i32);
                    }
                    BatchKind::Vector { path, rect, tint } => {
                        let Some(mesh) = self.vectors.get(path).and_then(Option::as_ref) else {
                            continue;
                        };
                        self.draw_mesh(
                            self.vector_program,
                            mesh,
                            [rect.w / mesh.width, rect.h / mesh.height, rect.x, rect.y],
                            *tint,
                        );
                        paint_bound = false;
                    }
                }
            }
            gl::Disable(gl::SCISSOR_TEST);
        }
        // Keep the current scene, evict old full-resolution wallpapers before memory grows.
        let mut bytes: usize = self
            .textures
            .values()
            .flatten()
            .map(|t| t.width as usize * t.height as usize * 4)
            .sum();
        if bytes > TEXTURE_BUDGET {
            let mut old: Vec<_> = self
                .textures
                .iter()
                .filter_map(|(path, t)| {
                    t.as_ref()
                        .filter(|t| t.frame != self.frame)
                        .map(|t| (path.clone(), t.frame))
                })
                .collect();
            old.sort_by_key(|(_, frame)| *frame);
            for (path, _) in old {
                if bytes <= TEXTURE_BUDGET {
                    break;
                }
                if let Some(Some(t)) = self.textures.remove(&path) {
                    bytes -= t.width as usize * t.height as usize * 4;
                }
            }
        }
        self.trim_text_geometry();
        self.stats.vector_meshes = self.vectors.values().flatten().count();
        self.stats.vector_bytes = self.vectors.values().flatten().map(|mesh| mesh.bytes).sum();
        self.stats.artwork_bytes = bytes + self.stats.vector_bytes;
        self.stats.draw_calls = self.batches.len();
        self.stats.cpu_submission_ms = started.elapsed().as_secs_f64() * 1000.;
    }

    fn draw_mesh(&self, program: GLuint, mesh: &vector::Mesh, transform: [f32; 4], tint: Color) {
        let viewport = self.stats.viewport;
        unsafe {
            gl::UseProgram(program);
            gl::BindBuffer(gl::ARRAY_BUFFER, mesh.buffer);
            gl::DisableVertexAttribArray(4);
            let mut offset = 0;
            for (index, size) in [2, 2, 4, 1].into_iter().enumerate() {
                gl::EnableVertexAttribArray(index as u32);
                gl::VertexAttribPointer(
                    index as u32,
                    size,
                    gl::FLOAT,
                    gl::FALSE,
                    36,
                    (offset * size_of::<f32>()) as *const _,
                );
                offset += size as usize;
            }
            gl::Uniform4f(
                gl::GetUniformLocation(program, c"u_transform".as_ptr()),
                transform[0],
                transform[1],
                transform[2],
                transform[3],
            );
            gl::Uniform2f(
                gl::GetUniformLocation(program, c"u_pixel_scale".as_ptr()),
                viewport.width as f32 / 1920.,
                viewport.height as f32 / 1080.,
            );
            gl::Uniform4f(
                gl::GetUniformLocation(program, c"u_tint".as_ptr()),
                tint[0],
                tint[1],
                tint[2],
                tint[3],
            );
            gl::DrawArrays(gl::TRIANGLES, 0, mesh.vertices);
        }
    }

    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    pub fn screenshot(&self, path: &Path, width: u32, height: u32) -> Result<(), String> {
        let mut pixels = vec![0; width as usize * height as usize * 4];
        unsafe {
            gl::ReadPixels(
                0,
                0,
                width as i32,
                height as i32,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                pixels.as_mut_ptr().cast(),
            );
        }
        let mut image = image::RgbaImage::from_raw(width, height, pixels)
            .ok_or("Invalid framebuffer dimensions")?;
        image::imageops::flip_vertical_in_place(&mut image);
        image.save(path).map_err(|e| e.to_string())
    }

    /// Capture the authored TV surface, independent of desktop HiDPI/window limits.
    pub fn capture(
        &mut self,
        commands: &[Draw],
        path: &Path,
        size: Size,
    ) -> Result<RenderStats, String> {
        if size.width > self.info.max_texture_size
            || size.height > self.info.max_texture_size
            || size.width > self.info.max_viewport[0] as u32
            || size.height > self.info.max_viewport[1] as u32
        {
            return Err(format!(
                "Capture {}x{} exceeds GPU limits: {}px texture, {:?} viewport",
                size.width, size.height, self.info.max_texture_size, self.info.max_viewport
            ));
        }
        let image = texture(size.width, size.height, &[]);
        let mut framebuffer = 0;
        unsafe {
            gl::GenFramebuffers(1, &mut framebuffer);
            gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffer);
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                image.id,
                0,
            );
            if gl::CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::DeleteFramebuffers(1, &framebuffer);
                return Err("Screenshot framebuffer is incomplete".into());
            }
        }
        self.render(commands, size.width, size.height, None);
        let stats = self.stats.clone();
        let result = self.screenshot(path, size.width, size.height);
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::DeleteFramebuffers(1, &framebuffer);
        }
        result.map(|_| stats)
    }

    /// Verification aid (LUMA_DAMAGE_VERIFY=1): compares the window's back buffer after a
    /// partial redraw with the same scene drawn in full offscreen at the window's sample count.
    /// Returns the pixels differing by more than 16 in a channel, and the largest difference.
    pub fn verify_window(
        &mut self,
        commands: &[Draw],
        width: u32,
        height: u32,
        save: Option<&Path>,
    ) -> (usize, u8) {
        let (w, h) = (width as i32, height as i32);
        let length = width as usize * height as usize * 4;
        let mut window = vec![0u8; length];
        let mut full = vec![0u8; length];
        let samples = self.stats.multisample_samples as i32;
        let stats = self.stats.clone();
        let mut framebuffers = [0; 2];
        let mut renderbuffers = [0; 2];
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::ReadPixels(
                0,
                0,
                w,
                h,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                window.as_mut_ptr().cast(),
            );
            gl::GenFramebuffers(2, framebuffers.as_mut_ptr());
            gl::GenRenderbuffers(2, renderbuffers.as_mut_ptr());
            gl::BindRenderbuffer(gl::RENDERBUFFER, renderbuffers[0]);
            if samples > 1 {
                gl::RenderbufferStorageMultisample(gl::RENDERBUFFER, samples, gl::RGBA8, w, h);
            } else {
                gl::RenderbufferStorage(gl::RENDERBUFFER, gl::RGBA8, w, h);
            }
            gl::BindRenderbuffer(gl::RENDERBUFFER, renderbuffers[1]);
            gl::RenderbufferStorage(gl::RENDERBUFFER, gl::RGBA8, w, h);
            for (framebuffer, renderbuffer) in framebuffers.into_iter().zip(renderbuffers) {
                gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffer);
                gl::FramebufferRenderbuffer(
                    gl::FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::RENDERBUFFER,
                    renderbuffer,
                );
            }
            gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffers[0]);
        }
        self.render(commands, width, height, None);
        unsafe {
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, framebuffers[0]);
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, framebuffers[1]);
            gl::BlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl::COLOR_BUFFER_BIT, gl::NEAREST);
            gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffers[1]);
            gl::ReadPixels(
                0,
                0,
                w,
                h,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                full.as_mut_ptr().cast(),
            );
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::DeleteFramebuffers(2, framebuffers.as_ptr());
            gl::DeleteRenderbuffers(2, renderbuffers.as_ptr());
        }
        self.stats = stats;
        let mut differing = 0;
        let mut largest = 0;
        for (a, b) in window.chunks_exact(4).zip(full.chunks_exact(4)) {
            let difference = (0..3).map(|i| a[i].abs_diff(b[i])).max().unwrap_or(0);
            largest = largest.max(difference);
            differing += usize::from(difference > 16);
        }
        if let Some(stem) = save.filter(|_| differing > 0) {
            for (name, pixels) in [("window", window), ("full", full)] {
                if let Some(image) = image::RgbaImage::from_raw(width, height, pixels) {
                    let image = image::imageops::flip_vertical(&image);
                    let _ = image.save(stem.with_extension(format!("{name}.png")));
                }
            }
        }
        (differing, largest)
    }
}

// Noto Sans SC (font-cjk) has no Hangul syllables; these go to font-kr instead.
// Keep in sync with is_hangul() in tools/font-weights.py, which builds font-kr.
fn is_hangul(c: char) -> bool {
    matches!(
        c,
        '\u{1100}'..='\u{11FF}'
            | '\u{3130}'..='\u{318F}'
            | '\u{A960}'..='\u{A97F}'
            | '\u{AC00}'..='\u{D7A3}'
            | '\u{D7B0}'..='\u{D7FF}'
    )
}

fn load_font(directory: &Path, id: u16) -> Result<Font, String> {
    let name = if id >= 3000 {
        format!("font-kr-{}.ttf", id - 3000)
    } else if id >= 2000 {
        format!("font-cjk-{}.ttf", id - 2000)
    } else if id >= 1000 {
        format!("font-display-{}.ttf", id - 1000)
    } else {
        format!("font-{id}.ttf")
    };
    load_font_file(&directory.join(&name)).map_err(|e| format!("{e}. Regenerate native assets."))
}

fn load_font_file(path: &Path) -> Result<Font, String> {
    let started = Instant::now();
    let bytes = std::fs::read(path).map_err(|e| format!("Font {}: {e}", path.display()))?;
    let font = Font::from_bytes(bytes).map_err(|e| format!("Font {}: {e}", path.display()))?;
    if cfg!(feature = "webos") {
        eprintln!(
            "Luma font: {} loaded in {} ms",
            path.display(),
            started.elapsed().as_millis()
        );
    }
    Ok(font)
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.buffer);
            gl::DeleteProgram(self.program);
            gl::DeleteProgram(self.vector_program);
            if self.vao != 0 {
                gl::DeleteVertexArrays(1, &self.vao);
            }
        }
    }
}

fn texture(width: u32, height: u32, pixels: &[u8]) -> Texture {
    let mut id = 0;
    unsafe {
        gl::GenTextures(1, &mut id);
        gl::BindTexture(gl::TEXTURE_2D, id);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGBA as i32,
            width as i32,
            height as i32,
            0,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            if pixels.is_empty() {
                std::ptr::null()
            } else {
                pixels.as_ptr().cast()
            },
        );
    }
    Texture {
        id,
        width,
        height,
        frame: 0,
    }
}

fn gl_string(name: GLenum) -> String {
    unsafe {
        let value = gl::GetString(name);
        if value.is_null() {
            "unknown".into()
        } else {
            CStr::from_ptr(value.cast()).to_string_lossy().into_owned()
        }
    }
}

fn compile(kind: GLenum, source: &str) -> Result<GLuint, String> {
    let source = CString::new(source).map_err(|e| e.to_string())?;
    unsafe {
        let shader = gl::CreateShader(kind);
        gl::ShaderSource(shader, 1, &source.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);
        let mut success = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success == 0 {
            let mut log = vec![0u8; 4096];
            let mut len = 0;
            gl::GetShaderInfoLog(shader, log.len() as i32, &mut len, log.as_mut_ptr().cast());
            gl::DeleteShader(shader);
            return Err(format!(
                "Shader compilation: {}",
                String::from_utf8_lossy(&log[..len as usize])
            ));
        }
        Ok(shader)
    }
}

fn link(vertex: &str, fragment: &str, attributes: &[&CStr]) -> Result<GLuint, String> {
    let vertex = compile(gl::VERTEX_SHADER, vertex)?;
    let fragment = match compile(gl::FRAGMENT_SHADER, fragment) {
        Ok(fragment) => fragment,
        Err(error) => {
            unsafe {
                gl::DeleteShader(vertex);
            }
            return Err(error);
        }
    };
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vertex);
        gl::AttachShader(program, fragment);
        for (index, name) in attributes.iter().enumerate() {
            gl::BindAttribLocation(program, index as u32, name.as_ptr());
        }
        gl::LinkProgram(program);
        gl::DeleteShader(vertex);
        gl::DeleteShader(fragment);
        let mut success = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        if success == 0 {
            let mut log = vec![0u8; 4096];
            let mut len = 0;
            gl::GetProgramInfoLog(program, log.len() as i32, &mut len, log.as_mut_ptr().cast());
            gl::DeleteProgram(program);
            return Err(format!(
                "Shader linking: {}",
                String::from_utf8_lossy(&log[..len as usize])
            ));
        }
        Ok(program)
    }
}
