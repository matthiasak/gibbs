//! Media preprocessing and prompt expansion, model-agnostic: every token id, string and size comes
//! from the spec's `runtime.media.<modality>` object ([`MediaSpec`]).
//!
//! Per request:
//! 1. [`measure`] each attached item: decode (image) or read the WAV, size math only, giving its
//!    [`Grid`] (token count, token grid, tower instance vars). Cheap: prompt-length checks use it.
//! 2. [`expand`] the prompt: each placeholder token becomes its item's expansion. Returns the
//!    expanded ids, the rows the tower's output rows scatter into (`media_index`), 3-axis rotary
//!    positions when a modality uses `positions: mrope_grid`, and bidirectional block spans.
//! 3. [`preprocess`] each item into its tower input ([`Prepared`]); the tower instance is built
//!    with [`Grid::vars`] and its output row `k` (items in prompt order) lands on
//!    `media_index[k]`.
//!
//! Spec shape (strings inside the object are expressions, see [`resolve`]; literal strings are
//! quoted: `"'<image>'"`):
//!
//! ```json5
//! media: { image: {
//!   component: "vision",
//!   preprocess: {kind: "smart_resize", patch: 16, merge: 2, temporal: 2,
//!                min_pixels: "json.preprocessor_config.size.shortest_edge",
//!                max_pixels: "json.preprocessor_config.size.longest_edge",
//!                norm: {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}},
//!   expand: {kind: "merged_grid", placeholder: "cfg.image_token_id"},
//!   positions: "mrope_grid",          // default "plain"
//!   bidirectional: false,             // media rows of one item attend each other both ways
//! } }
//! ```

use crate::arch::build::{build, Instance, ShapeCache};
use crate::arch::expr::{eval_json, Env, Val};
use crate::arch::gen::{generate, GenData};
use crate::arch::load::Model;
use crate::graph::program::GraphProgram;
use crate::graph::Tensor;
use crate::hardware::PlacementMode;
use crate::model::audio::{read_wav, resample, resampled_len};
use crate::model::image::ImageBuffer;
use parking_lot::Mutex;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;

/// `runtime.media`: modality name (`image`, `audio`, …) → its spec.
pub type Media = BTreeMap<String, MediaSpec>;

/// Evaluates the raw `runtime.media` object against the model `env` (strings are expressions:
/// `cfg.*`, `json.*`, config vars; bare undefined identifiers are strings) and deserializes it.
/// A modality whose `when` expression is false or undefined is dropped first.
pub fn resolve(raw: &serde_json::Value, env: &dyn Env) -> Result<Media, String> {
    let mut kept = serde_json::Map::new();
    for (name, m) in raw.as_object().into_iter().flatten() {
        let mut m = m.clone();
        if let Some(w) = m.as_object_mut().and_then(|o| o.remove("when")) {
            if !eval_json(&w, env)?.truthy() {
                continue;
            }
        }
        kept.insert(name.clone(), m);
    }
    serde_json::from_value(eval_json(&serde_json::Value::Object(kept), env)?.to_json())
        .map_err(|e| format!("runtime.media: {e}"))
}

/// One modality: how an item is preprocessed for its tower and how its placeholder expands.
#[derive(Debug, Clone, Deserialize)]
pub struct MediaSpec {
    /// Tower component each item runs through.
    pub component: String,
    pub preprocess: Preprocess,
    pub expand: Expand,
    #[serde(default)]
    pub positions: Positions,
    /// An item's media rows form one block attended bidirectionally ([`Expanded::blocks`]).
    #[serde(default)]
    pub bidirectional: bool,
}

/// Rotary positions of an item's media rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Positions {
    /// Like text: one position per row.
    #[default]
    Plain,
    /// `(t, h, w)` = `(p, p + row, p + col)` over the item's `rows x cols` grid from the item's
    /// start position `p`; text resumes at `p + max(rows, cols)`.
    MropeGrid,
}

/// Pixel normalization: `(v * rescale - mean[c]) / std[c]` for an 8-bit channel value `v`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
pub struct Norm {
    pub rescale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for Norm {
    fn default() -> Self {
        Self { rescale: 1.0 / 255.0, mean: [0.0; 3], std: [1.0; 3] }
    }
}

impl Norm {
    fn px(&self, v: u8, c: usize) -> f32 {
        (v as f32 * self.rescale - self.mean[c]) / self.std[c]
    }
}

/// Resampling filter of [`Preprocess::TileLongestEdge`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    #[default]
    Bicubic,
    Lanczos,
}

/// How one item becomes its tower input. Image kinds resize with [`ImageBuffer`]; audio kinds
/// read WAV ([`crate::model::audio`]).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Preprocess {
    /// Longest side to `longest_edge` (aspect kept, even, sides capped at `max_side`); with
    /// `split`, sides then grow to multiples of `tile` and the image is cut into `rows x cols`
    /// row-major crops plus a global view (a single `tile x tile` view when it fits one tile).
    /// Each tile gives `tokens_per_tile` rows.
    /// Data `[tiles, 3, tile, tile]` (channel-planar). Grid: `rows x cols` crops (`0 x 0` =
    /// single tile); vars `tiles`.
    TileLongestEdge {
        longest_edge: usize,
        tile: usize,
        max_side: usize,
        tokens_per_tile: usize,
        #[serde(default = "yes")]
        split: bool,
        #[serde(default)]
        filter: Filter,
        #[serde(default)]
        norm: Norm,
    },
    /// `smart_resize`: sides rounded to multiples of `patch * merge` (half to even), pixel count
    /// clamped into `[min_pixels, max_pixels]`, bicubic. Data `[gh * gw, 3 * temporal * patch^2]`:
    /// patches in merge-block order (block row, block col, row in block, col in block), each
    /// `[channel, frame, y, x]` with the still image repeated over `temporal` frames. Grid:
    /// `gh/merge x gw/merge` merged tokens; vars `grid_h = gh`, `grid_w = gw` (patch grid).
    SmartResize {
        patch: usize,
        merge: usize,
        temporal: usize,
        min_pixels: usize,
        max_pixels: usize,
        #[serde(default)]
        norm: Norm,
    },
    /// Largest aspect-preserving size, sides multiples of `pool * patch`, within
    /// `max_soft_tokens * pool^2` patches (`get_aspect_ratio_preserving_size`), bicubic.
    /// Data `[ph * pw, patch * patch * 3]`: raster patches, `(y, x, c)` inside. Grid:
    /// `ph/pool x pw/pool` pooled tokens; vars `grid_h = ph`, `grid_w = pw` (patch grid).
    SoftTokenBudget {
        patch: usize,
        pool: usize,
        max_soft_tokens: usize,
        #[serde(default)]
        norm: Norm,
    },
    /// The [`Self::SoftTokenBudget`] size, cut directly into `patch * pool` px model patches (one
    /// token each). Data `[gh * gw, (patch * pool)^2 * 3]` raster, `(y, x, c)` inside. Grid and
    /// vars: `grid_h = gh`, `grid_w = gw`; `max_grid` (position-table size) bounds both sides.
    PatchGrid {
        patch: usize,
        pool: usize,
        max_soft_tokens: usize,
        #[serde(default)]
        max_grid: Option<usize>,
        #[serde(default)]
        norm: Norm,
    },
    /// Log-mel front-end framing: the clip resampled to `sampling_rate` (at most `max_samples`),
    /// right-padded to a multiple of `pad_multiple`, `frame_length / 2` zeros prepended
    /// (semicausal), cut into frames of `frame_length` every `hop_length`. The windowed DFT, mel
    /// filters and `ln` run in the tower graph. Data `[frames, frame_length]`; valid frames form
    /// a prefix. Grid: `tokens x 1`, `tokens` = valid frames among every `token_stride`-th;
    /// vars `frames`, `valid_frames`, `tokens`.
    LogMel {
        sampling_rate: u32,
        frame_length: usize,
        hop_length: usize,
        pad_multiple: usize,
        max_samples: usize,
        token_stride: usize,
    },
    /// Raw sample frames: the clip resampled to `sampling_rate`, zero-padded to whole frames of
    /// `frame` samples, at most `max_tokens` frames. Data `[tokens, frame]`. Grid `tokens x 1`;
    /// vars `tokens`.
    RawFrames {
        sampling_rate: u32,
        frame: usize,
        max_tokens: usize,
    },
}

fn yes() -> bool {
    true
}

/// How a placeholder token in the prompt expands. `placeholder` is the prompt token one item
/// replaces; media rows carry it in the expanded ids.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Expand {
    /// `open`, `tokens` media rows, `close`.
    RepeatBetween { placeholder: u32, open: u32, close: u32 },
    /// Tiled-image text, tokenized: per crop row `r`, per crop col `c` (1-based):
    /// `around + cell{r,c} + token * per_tile`, then `\n`; an extra `\n` after the crops; then
    /// `around + global + token * per_tile + around`. `cell` holds `{r}`/`{c}`; `per_tile` =
    /// tokens / tiles. Media rows are the tokenized ids equal to `placeholder`.
    GridText { placeholder: u32, around: String, cell: String, global: String, token: String },
    /// `tokens` media rows (the merged grid, row-major).
    MergedGrid { placeholder: u32 },
}

impl Expand {
    pub fn placeholder(&self) -> u32 {
        match self {
            Expand::RepeatBetween { placeholder, .. }
            | Expand::GridText { placeholder, .. }
            | Expand::MergedGrid { placeholder } => *placeholder,
        }
    }
}

/// One item's size, from [`measure`] (and inside [`Prepared`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    /// Token grid the prompt sees (per kind: see [`Preprocess`]).
    pub rows: usize,
    pub cols: usize,
    /// Media rows the tower emits for this item (= rows scattered into the prompt).
    pub tokens: usize,
    /// Tower instance variables.
    pub vars: Vec<(&'static str, usize)>,
}

impl Grid {
    pub fn var(&self, name: &str) -> Option<usize> {
        self.vars.iter().find(|(k, _)| *k == name).map(|&(_, v)| v)
    }
}

/// One item ready for its tower: `data` row-major with dims `shape` (layout per [`Preprocess`]).
#[derive(Debug, Clone)]
pub struct Prepared {
    pub grid: Grid,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// The item's [`Grid`] from file bytes, without building the tower input.
pub fn measure(p: &Preprocess, bytes: &[u8]) -> Result<Grid, String> {
    match p {
        Preprocess::LogMel { .. } | Preprocess::RawFrames { .. } => {
            let (pcm, rate) = read_wav(bytes)?;
            Ok(audio_plan(p, pcm.len(), rate)?.0)
        }
        _ => {
            let img = ImageBuffer::decode(bytes)?;
            image_grid(p, img.width, img.height)
        }
    }
}

/// The item's tower input from file bytes.
pub fn preprocess(p: &Preprocess, bytes: &[u8]) -> Result<Prepared, String> {
    match p {
        Preprocess::LogMel { .. } | Preprocess::RawFrames { .. } => {
            let (pcm, rate) = read_wav(bytes)?;
            audio_input(p, &pcm, rate)
        }
        _ => image_input(p, &ImageBuffer::decode(bytes)?),
    }
}

/// A small synthetic item of `p`'s kind (64 x 64 grey image; silence: one frame for raw frames,
/// 1 s at 16 kHz for log-mel): what a tower is probed with at load (its weights, whether it
/// builds).
pub fn probe_item(p: &Preprocess) -> Vec<u8> {
    let (rate, n) = match *p {
        Preprocess::RawFrames { sampling_rate, frame, .. } => (sampling_rate, frame as u32),
        Preprocess::LogMel { .. } => (16_000, 16_000),
        _ => {
            let mut img = ImageBuffer::new(64, 64);
            img.data.fill(128);
            return img.to_png();
        }
    };
    let mut b = b"RIFF".to_vec();
    b.extend((36 + 2 * n).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend([1, 0, 1, 0]);
    b.extend(rate.to_le_bytes());
    b.extend((2 * rate).to_le_bytes());
    b.extend([2, 0, 16, 0]);
    b.extend(b"data");
    b.extend((2 * n).to_le_bytes());
    b.resize(b.len() + 2 * n as usize, 0);
    b
}


/// Media tower components run on one device: per component, one program compiled from its first
/// instance, every other item shape run reshaped on it.
pub struct Towers {
    mode: PlacementMode,
    towers: Mutex<HashMap<String, Tower>>,
}

struct Tower {
    shapes: Arc<ShapeCache>,
    /// The compiled program and the instance key it was compiled for.
    program: Option<(String, GraphProgram)>,
}

impl Towers {
    pub fn new(mode: PlacementMode) -> Self {
        Self { mode, towers: Mutex::new(HashMap::new()) }
    }

    /// `component` of `m` built for `grid` (its vars are the instance vars).
    pub fn instance(&self, m: &Model, component: &str, grid: &Grid) -> Result<Arc<Instance>, String> {
        let vars: Vec<(&str, Val)> = grid.vars.iter().map(|&(k, v)| (k, Val::Num(v as f64))).collect();
        let shapes = self
            .towers
            .lock()
            .entry(component.to_string())
            .or_insert_with(|| Tower { shapes: Arc::new(ShapeCache::new(component)), program: None })
            .shapes
            .clone();
        shapes.get(&vars, || build(&m.spec, component, &m.env, &m.store, &vars))
    }

    /// Runs `component` of `m` on one preprocessed item: input 0 is `item.data`, every other input
    /// is generated. Returns the outputs by name, in declaration order.
    pub fn run(&self, m: &Model, component: &str, item: &Prepared) -> Result<Vec<(String, Vec<f32>)>, String> {
        let inst = self.instance(m, component, &item.grid)?;
        let mut inputs = Vec::with_capacity(inst.inputs.len());
        for (i, d) in inst.inputs.iter().enumerate() {
            let want = d.shape.iter().product::<usize>();
            let t = match (&d.gen, i) {
                (None, 0) if item.data.len() == want => Tensor::F32(item.data.clone()),
                (None, 0) => {
                    return Err(format!(
                        "{component}: input `{}` {:?} does not hold the item's {:?}",
                        d.name, d.shape, item.shape
                    ))
                }
                (None, _) => return Err(format!("{component}: input `{}` needs a `gen`", d.name)),
                (Some((g, args)), _) => match generate(g, args).map_err(|e| format!("{component}: input `{}`: {e}", d.name))?.data {
                    GenData::F32(v) => Tensor::F32(v),
                    GenData::U32(v) => Tensor::U32(v),
                },
            };
            inputs.push(t);
        }
        let key = ShapeCache::key(&item.grid.vars.iter().map(|&(k, v)| (k, Val::Num(v as f64))).collect::<Vec<_>>());
        let mut towers = self.towers.lock();
        let tower = towers.get_mut(component).expect("instance() registered the tower");
        if tower.program.is_none() {
            let w = m.store.materialize(&inst.weights)?;
            let p = GraphProgram::compile_reshapable(inst.graph.clone(), w, self.mode)
                .map_err(|e| format!("{component}: {e}"))?;
            tower.program = Some((key.clone(), p));
        }
        let (k, p) = tower.program.as_mut().unwrap();
        let out = if *k == key { p.run(inputs)? } else { p.run_reshaped(&inst.graph, inputs, true)? };
        inst.outputs
            .iter()
            .cloned()
            .zip(out)
            .map(|(n, t)| Ok((n, t.into_f32()?)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

fn image_grid(p: &Preprocess, w: usize, h: usize) -> Result<Grid, String> {
    let grid = |rows, cols, tokens, gh, gw| Grid {
        rows,
        cols,
        tokens,
        vars: vec![("grid_h", gh), ("grid_w", gw)],
    };
    Ok(match *p {
        Preprocess::TileLongestEdge { longest_edge, tile, max_side, tokens_per_tile, split, .. } => {
            let (rows, cols) = tile_grid(w, h, longest_edge, tile, max_side, split);
            let tiles = if rows == 0 { 1 } else { rows * cols + 1 };
            Grid { rows, cols, tokens: tiles * tokens_per_tile, vars: vec![("tiles", tiles)] }
        }
        Preprocess::SmartResize { patch, merge, min_pixels, max_pixels, .. } => {
            let (rh, rw) = smart_resize(h, w, patch * merge, min_pixels, max_pixels)?;
            let (gh, gw) = (rh / patch, rw / patch);
            let (rows, cols) = (gh / merge, gw / merge);
            grid(rows, cols, rows * cols, gh, gw)
        }
        Preprocess::SoftTokenBudget { patch, pool, max_soft_tokens, .. } => {
            let (th, tw) = target_size(h, w, patch, max_soft_tokens * pool * pool, pool)?;
            let (ph, pw) = (th / patch, tw / patch);
            let (rows, cols) = (ph / pool, pw / pool);
            grid(rows, cols, rows * cols, ph, pw)
        }
        Preprocess::PatchGrid { patch, pool, max_soft_tokens, max_grid, .. } => {
            let (th, tw) = target_size(h, w, patch, max_soft_tokens * pool * pool, pool)?;
            let (gh, gw) = (th / (patch * pool), tw / (patch * pool));
            if let Some(m) = max_grid.filter(|&m| gh > m || gw > m) {
                return Err(format!("image grid {gw}x{gh} exceeds the {m} position embeddings"));
            }
            grid(gh, gw, gh * gw, gh, gw)
        }
        _ => unreachable!("audio kinds are not images"),
    })
}

fn image_input(p: &Preprocess, img: &ImageBuffer) -> Result<Prepared, String> {
    let grid = image_grid(p, img.width, img.height)?;
    let (data, shape) = match *p {
        Preprocess::TileLongestEdge { longest_edge, tile, max_side, split, filter, norm, .. } => {
            let tiles = tiles(img, longest_edge, tile, max_side, split, filter);
            let n = tile * tile;
            let mut out = vec![0.0f32; tiles.len() * 3 * n];
            for (t, dst) in tiles.iter().zip(out.chunks_exact_mut(3 * n)) {
                for (i, px) in t.data.chunks_exact(3).enumerate() {
                    for c in 0..3 {
                        dst[c * n + i] = norm.px(px[c], c);
                    }
                }
            }
            (out, vec![tiles.len(), 3, tile, tile])
        }
        Preprocess::SmartResize { patch: p, merge: m, temporal, norm, .. } => {
            let (gh, gw) = (grid.var("grid_h").unwrap(), grid.var("grid_w").unwrap());
            let (rh, rw) = (gh * p, gw * p);
            let img = fit(img, rw, rh);
            let k = 3 * temporal * p * p;
            let mut coords = Vec::with_capacity(gh * gw);
            for br in 0..gh / m {
                for bc in 0..gw / m {
                    for ir in 0..m {
                        for ic in 0..m {
                            coords.push((br * m + ir, bc * m + ic));
                        }
                    }
                }
            }
            let mut out = vec![0.0f32; coords.len() * k];
            crate::model::par::par_rows(&mut out, k, &|i, row| {
                let (r, c) = coords[i];
                let mut o = 0;
                for ch in 0..3 {
                    for _ in 0..temporal {
                        for y in 0..p {
                            for x in 0..p {
                                row[o] = norm.px(img.data[((r * p + y) * rw + c * p + x) * 3 + ch], ch);
                                o += 1;
                            }
                        }
                    }
                }
            });
            (out, vec![gh * gw, k])
        }
        Preprocess::SoftTokenBudget { patch, norm, .. } => raster_patches(img, &grid, patch, norm),
        Preprocess::PatchGrid { patch, pool, norm, .. } => {
            raster_patches(img, &grid, patch * pool, norm)
        }
        _ => unreachable!("audio kinds are not images"),
    };
    Ok(Prepared { grid, data, shape })
}

/// `img` itself when already `w x h`, else its bicubic resize.
fn fit(img: &ImageBuffer, w: usize, h: usize) -> std::borrow::Cow<'_, ImageBuffer> {
    if (img.width, img.height) == (w, h) {
        std::borrow::Cow::Borrowed(img)
    } else {
        std::borrow::Cow::Owned(img.resize(w, h))
    }
}

/// `side x side` raster patches over the `grid_h x grid_w` patch grid of the resized image.
fn raster_patches(img: &ImageBuffer, grid: &Grid, side: usize, norm: Norm) -> (Vec<f32>, Vec<usize>) {
    let (gh, gw) = (grid.var("grid_h").unwrap(), grid.var("grid_w").unwrap());
    let tw = gw * side;
    let img = fit(img, tw, gh * side);
    let pd = side * side * 3;
    let mut out = vec![0.0f32; gh * gw * pd];
    for (i, patch) in out.chunks_exact_mut(pd).enumerate() {
        let (gy, gx) = (i / gw, i % gw);
        for y in 0..side {
            let src = ((gy * side + y) * tw + gx * side) * 3;
            for (j, d) in patch[y * side * 3..(y + 1) * side * 3].iter_mut().enumerate() {
                *d = norm.px(img.data[src + j], j % 3);
            }
        }
    }
    (out, vec![gh * gw, pd])
}

/// Sizes a `w x h` image is resized to: longest edge to `longest_edge` (Python `int()` on the
/// aspect ratio, odd sides made even, sides capped at `max_side`), then, when splitting, sides up
/// to multiples of `tile` keeping the aspect ratio.
fn tile_sizes(
    (w, h): (usize, usize),
    longest_edge: usize,
    tile: usize,
    max_side: usize,
    split: bool,
) -> ((usize, usize), Option<(usize, usize)>) {
    let t = tile;
    let (mut w, mut h) = (w, h);
    let ar = w as f64 / h as f64;
    if w >= h {
        w = longest_edge;
        h = (w as f64 / ar) as usize;
        h += h % 2;
    } else {
        h = longest_edge;
        w = (h as f64 * ar) as usize;
        w += w % 2;
    }
    let ar = w.max(1) as f64 / h.max(1) as f64;
    if w >= h && w > max_side {
        (w, h) = (max_side, (max_side as f64 / ar) as usize);
    } else if h > w && h > max_side {
        (w, h) = ((max_side as f64 * ar) as usize, max_side);
    }
    let (w, h) = (w.max(1), h.max(1));
    if !split {
        return ((w, h), None);
    }
    let ar = w as f64 / h as f64;
    let split = if w >= h {
        let w = w.div_ceil(t) * t;
        (w, ((w as f64 / ar) as usize).div_ceil(t).max(1) * t)
    } else {
        let h = h.div_ceil(t) * t;
        (((h as f64 * ar) as usize).div_ceil(t).max(1) * t, h)
    };
    ((w, h), Some(split))
}

/// `(rows, cols)` of crops; `(0, 0)` when the image is a single tile.
fn tile_grid(w: usize, h: usize, longest_edge: usize, tile: usize, max_side: usize, split: bool) -> (usize, usize) {
    match tile_sizes((w, h), longest_edge, tile, max_side, split).1 {
        Some((w, h)) if w > tile || h > tile => (h / tile, w / tile),
        _ => (0, 0),
    }
}

/// The `rows x cols` row-major crops then the global view, or the single view; all `tile x tile`.
fn tiles(img: &ImageBuffer, longest_edge: usize, t: usize, max_side: usize, split: bool, filter: Filter) -> Vec<ImageBuffer> {
    let resize = |img: &ImageBuffer, w, h| match filter {
        Filter::Lanczos => img.resize_lanczos(w, h),
        Filter::Bicubic => img.resize(w, h),
    };
    let (rows, cols) = tile_grid(img.width, img.height, longest_edge, t, max_side, split);
    let ((w, h), split) = tile_sizes((img.width, img.height), longest_edge, t, max_side, split);
    let img = resize(img, w, h);
    let Some((w, h)) = split else {
        return vec![resize(&img, t, t)];
    };
    let img = resize(&img, w, h);
    if rows == 0 {
        return vec![img];
    }
    let mut out = Vec::with_capacity(rows * cols + 1);
    for r in 0..rows {
        for c in 0..cols {
            let data = (0..t)
                .flat_map(|y| {
                    let start = ((r * t + y) * w + c * t) * 3;
                    img.data[start..start + t * 3].iter().copied()
                })
                .collect();
            out.push(ImageBuffer::from_rgb(t, t, data).expect("tile is t*t*3"));
        }
    }
    out.push(resize(&img, t, t));
    out
}

/// `smart_resize`: (height, width) divisible by `factor`, aspect kept, pixel count in
/// `[min, max]`. Python's `round` is half-to-even.
fn smart_resize(h: usize, w: usize, factor: usize, min: usize, max: usize) -> Result<(usize, usize), String> {
    let (hf, wf, f) = (h as f64, w as f64, factor as f64);
    if h == 0 || w == 0 || hf.max(wf) / hf.min(wf) > 200.0 {
        return Err(format!("image {w}x{h}: aspect ratio must be below 200"));
    }
    let mut hb = (hf / f).round_ties_even() * f;
    let mut wb = (wf / f).round_ties_even() * f;
    if hb * wb > max as f64 {
        let beta = (hf * wf / max as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wb = f.max((wf / beta / f).floor() * f);
    } else if hb * wb < min as f64 {
        let beta = (min as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Largest `(h, w)` keeping the aspect ratio, both multiples of `pool * patch`, within
/// `max_patches` patches.
fn target_size(height: usize, width: usize, patch: usize, max_patches: usize, pool: usize) -> Result<(usize, usize), String> {
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = pool * patch;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    let max_side = (max_patches / (pool * pool)) * side;
    match (th, tw) {
        (0, 0) => return Err(format!("image {width}x{height} resizes to 0x0")),
        (0, _) => {
            th = side;
            tw = ((width / height) * side).min(max_side);
        }
        (_, 0) => {
            tw = side;
            th = ((height / width) * side).min(max_side);
        }
        _ => {}
    }
    if (th * tw) as f64 > target_px {
        return Err(format!("image {width}x{height} exceeds {max_patches} patches"));
    }
    Ok((th, tw))
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

/// The grid of a clip of `len` samples at `rate` Hz, and (log-mel) each frame's validity;
/// checked against the limits before anything is resampled.
fn audio_plan(p: &Preprocess, len: usize, rate: u32) -> Result<(Grid, Vec<bool>), String> {
    let secs = len as f64 / rate as f64;
    match *p {
        Preprocess::LogMel { sampling_rate, frame_length, hop_length, pad_multiple, max_samples, token_stride } => {
            let real = resampled_len(len, rate, sampling_rate);
            if real > max_samples {
                return Err(format!(
                    "audio is {secs:.1} s; the model takes at most {:.0} s per clip",
                    max_samples as f64 / sampling_rate as f64
                ));
            }
            let left = frame_length / 2;
            let padded = left + real.div_ceil(pad_multiple) * pad_multiple;
            let span = frame_length + 1;
            if padded < span {
                return Err("audio clip is too short".into());
            }
            let t = (padded - span) / hop_length + 1;
            let mask: Vec<bool> = (0..t)
                .map(|i| (left..left + real).contains(&(i * hop_length + span - 1)))
                .collect();
            let n = mask.iter().step_by(token_stride.max(1)).filter(|&&ok| ok).count();
            if n == 0 {
                return Err("audio clip is too short".into());
            }
            let valid = mask.iter().filter(|&&ok| ok).count();
            let vars = vec![("frames", t), ("valid_frames", valid), ("tokens", n)];
            Ok((Grid { rows: n, cols: 1, tokens: n, vars }, mask))
        }
        Preprocess::RawFrames { sampling_rate, frame, max_tokens } => {
            let n = resampled_len(len, rate, sampling_rate).div_ceil(frame);
            if n == 0 {
                return Err("audio clip is empty".into());
            }
            if n > max_tokens {
                return Err(format!(
                    "audio clip is {secs:.1} s; the model takes at most {:.0} s",
                    (max_tokens * frame) as f64 / sampling_rate as f64
                ));
            }
            Ok((Grid { rows: n, cols: 1, tokens: n, vars: vec![("tokens", n)] }, Vec::new()))
        }
        _ => unreachable!("image kinds are not audio"),
    }
}

fn audio_input(p: &Preprocess, pcm: &[f32], rate: u32) -> Result<Prepared, String> {
    let (grid, mask) = audio_plan(p, pcm.len(), rate)?;
    let (data, shape) = match *p {
        Preprocess::LogMel { sampling_rate, frame_length, hop_length, pad_multiple, .. } => {
            let mut x = resample(pcm, rate, sampling_rate);
            x.resize(x.len().div_ceil(pad_multiple) * pad_multiple, 0.0);
            let padded: Vec<f32> = std::iter::repeat_n(0.0, frame_length / 2).chain(x).collect();
            let t = mask.len();
            let frames = (0..t)
                .flat_map(|i| &padded[i * hop_length..i * hop_length + frame_length])
                .copied()
                .collect();
            (frames, vec![t, frame_length])
        }
        Preprocess::RawFrames { sampling_rate, frame, .. } => {
            let mut x = resample(pcm, rate, sampling_rate);
            x.resize(grid.tokens * frame, 0.0); // zero-pad the last frame
            (x, vec![grid.tokens, frame])
        }
        _ => unreachable!("image kinds are not audio"),
    };
    Ok(Prepared { grid, data, shape })
}

// ---------------------------------------------------------------------------
// Prompt expansion
// ---------------------------------------------------------------------------

/// A prompt with every placeholder expanded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Expanded {
    pub ids: Vec<u32>,
    /// Row of `ids` each media row fills, in media-row order (items in prompt order, each item's
    /// tower rows in order): the `media_index` input of the text instance.
    pub media_index: Vec<u32>,
    /// `(t, h, w)` rotary position per row of `ids`; empty unless some modality of `media` uses
    /// [`Positions::MropeGrid`].
    pub pos3: Vec<[u32; 3]>,
    /// Position offset of rows past the prompt (`next position - ids.len()`; 0 without grids).
    pub delta: i64,
    /// Row spans of items whose modality is `bidirectional`, ascending.
    pub blocks: Vec<Range<usize>>,
}

impl Expanded {
    /// A text-like row at the next position.
    fn text(&mut self, tok: u32, cur: &mut u32, mrope: bool) {
        self.ids.push(tok);
        if mrope {
            self.pos3.push([*cur; 3]);
        }
        *cur += 1;
    }

    /// Media row `k` of an item.
    fn media(&mut self, tok: u32, k: usize, grid: &Grid, on_grid: bool, cur: &mut u32, mrope: bool) {
        self.media_index.push(self.ids.len() as u32);
        if !on_grid {
            return self.text(tok, cur, mrope);
        }
        self.ids.push(tok);
        let (r, c) = ((k / grid.cols) as u32, (k % grid.cols) as u32);
        self.pos3.push([*cur, *cur + r, *cur + c]);
        if k + 1 == grid.tokens {
            *cur += grid.rows.max(grid.cols) as u32;
        }
    }
}

/// Expands `ids`: each token equal to a modality's placeholder takes the next of `items`
/// (`(modality, grid)`, prompt order). `tokenize` encodes text expansions
/// ([`Expand::GridText`]; special tokens as single ids). Errs when the placeholders and the items
/// disagree in count or modality.
pub fn expand(
    ids: &[u32],
    media: &Media,
    items: &[(&str, &Grid)],
    tokenize: &dyn Fn(&str) -> Result<Vec<u32>, String>,
) -> Result<Expanded, String> {
    let slot = |t: u32| media.iter().find(|(_, s)| s.expand.placeholder() == t);
    let slots = ids.iter().filter(|&&t| slot(t).is_some()).count();
    if slots != items.len() {
        return Err(format!(
            "prompt has {slots} media placeholder(s) for {} media item(s)",
            items.len()
        ));
    }
    let mrope = media.values().any(|s| s.positions == Positions::MropeGrid);
    let mut e = Expanded { ids: Vec::with_capacity(ids.len()), ..Default::default() };
    let mut cur = 0u32;
    let mut items = items.iter().enumerate();
    for &t in ids {
        let Some((name, spec)) = slot(t) else {
            e.text(t, &mut cur, mrope);
            continue;
        };
        let (i, &(item, grid)) = items.next().expect("placeholders counted");
        if item != name {
            return Err(format!(
                "media item {} is `{item}` but its placeholder is `{name}`",
                i + 1
            ));
        }
        let on_grid = spec.positions == Positions::MropeGrid;
        if on_grid && (grid.rows * grid.cols != grid.tokens || matches!(spec.expand, Expand::GridText { .. })) {
            return Err(format!(
                "`{name}`: positions mrope_grid needs contiguous rows x cols media rows"
            ));
        }
        let first = e.media_index.len();
        match &spec.expand {
            Expand::RepeatBetween { open, close, .. } => {
                e.text(*open, &mut cur, mrope);
                for k in 0..grid.tokens {
                    e.media(t, k, grid, on_grid, &mut cur, mrope);
                }
                e.text(*close, &mut cur, mrope);
            }
            Expand::MergedGrid { .. } => {
                for k in 0..grid.tokens {
                    e.media(t, k, grid, on_grid, &mut cur, mrope);
                }
            }
            Expand::GridText { around, cell, global, token, .. } => {
                let tiles = if grid.rows == 0 { 1 } else { grid.rows * grid.cols + 1 };
                if grid.tokens % tiles != 0 {
                    return Err(format!("`{name}`: {} tokens over {tiles} tiles", grid.tokens));
                }
                let text = grid_text(around, cell, global, token, grid.rows, grid.cols, grid.tokens / tiles);
                for (k, tok) in tokenize(&text)?.into_iter().enumerate() {
                    if tok == t {
                        e.media(tok, k, grid, false, &mut cur, mrope);
                    } else {
                        e.text(tok, &mut cur, mrope);
                    }
                }
            }
        }
        let n = e.media_index.len() - first;
        if n != grid.tokens {
            return Err(format!(
                "`{name}` expansion has {n} media rows; the item has {}",
                grid.tokens
            ));
        }
        if spec.bidirectional && n > 0 {
            e.blocks.push(e.media_index[first] as usize..*e.media_index.last().unwrap() as usize + 1);
        }
    }
    e.delta = cur as i64 - e.ids.len() as i64;
    Ok(e)
}

/// The [`Expand::GridText`] text of a `rows x cols` split.
fn grid_text(around: &str, cell: &str, global: &str, token: &str, rows: usize, cols: usize, per_tile: usize) -> String {
    let tokens = token.repeat(per_tile);
    let mut s = String::new();
    for r in 1..=rows {
        for c in 1..=cols {
            s += around;
            s += &cell.replace("{r}", &r.to_string()).replace("{c}", &c.to_string());
            s += &tokens;
        }
        s.push('\n');
    }
    if rows > 0 {
        s.push('\n');
    }
    s + around + global + &tokens + around
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn png(w: usize, h: usize) -> Vec<u8> {
        let mut img = ImageBuffer::new(w, h);
        for (i, px) in img.data.iter_mut().enumerate() {
            *px = ((i * 37 + (i / 7) * 11) % 251) as u8;
        }
        img.to_png()
    }

    /// 16-bit mono WAV of a 440 Hz tone.
    fn wav(rate: u32, samples: usize) -> Vec<u8> {
        let pcm: Vec<u8> = (0..samples)
            .flat_map(|i| {
                let v = (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin();
                ((v * 12000.0) as i16).to_le_bytes()
            })
            .collect();
        crate::model::audio::wav_bytes(1, 16, 1, rate, &pcm)
    }

    /// Position-weighted sum: a pinned fingerprint of a tensor.
    fn sum(v: &[f32]) -> f64 {
        v.iter().enumerate().map(|(i, &x)| x as f64 * (i % 7 + 1) as f64).sum()
    }

    /// Position-weighted sum (relative 1e-9) and exact elements at `len/3 + 1`, `len/2`, `len - 1`.
    fn pinned(v: &[f32], (want_sum, want_at): (f64, [f32; 3])) {
        let got = sum(v);
        let at = [v[v.len() / 3 + 1], v[v.len() / 2], v[v.len() - 1]];
        assert!((got - want_sum).abs() <= 1e-9 * want_sum.abs().max(1.0), "fingerprint {got} != {want_sum}");
        assert_eq!(at, want_at);
    }

    fn spec(j: serde_json::Value) -> MediaSpec {
        serde_json::from_value(j).unwrap()
    }

    fn half_norm() -> Norm {
        Norm { rescale: 1.0 / 255.0, mean: [0.5; 3], std: [0.5; 3] }
    }

    fn no_text(_: &str) -> Result<Vec<u32>, String> {
        Err("no text expansion".into())
    }

    fn tile(split: bool) -> Preprocess {
        Preprocess::TileLongestEdge {
            longest_edge: 96,
            tile: 32,
            max_side: 4096,
            tokens_per_tile: 4,
            split,
            filter: Filter::Lanczos,
            norm: half_norm(),
        }
    }

    #[test]
    fn tile_longest_edge_pinned() {
        // 300x200 → 96x64 → crops of 32: 2 rows x 3 cols + global = 7 tiles.
        let p = preprocess(&tile(true), &png(300, 200)).unwrap();
        assert_eq!((p.grid.rows, p.grid.cols, p.grid.tokens), (2, 3, 28));
        assert_eq!(p.grid.vars, vec![("tiles", 7)]);
        assert_eq!(p.shape, vec![7, 3, 32, 32]);
        pinned(&p.data, PIN_TILE);
        // Portrait 100x700 → 14x96 → 32x96: 3 x 1 + global; unsplit: one view.
        assert_eq!(measure(&tile(true), &png(100, 700)).unwrap().tokens, 16);
        let one = preprocess(&tile(false), &png(300, 200)).unwrap();
        assert_eq!((one.grid.rows, one.grid.tokens, one.shape[0]), (0, 4, 1));
    }

    fn smart() -> Preprocess {
        Preprocess::SmartResize {
            patch: 2,
            merge: 2,
            temporal: 2,
            min_pixels: 64,
            max_pixels: 100_000,
            norm: half_norm(),
        }
    }

    #[test]
    fn smart_resize_pinned() {
        // 16x24 is already a multiple of patch*merge = 4: 12x8 patches, 6x4 merged tokens.
        let p = preprocess(&smart(), &png(16, 24)).unwrap();
        assert_eq!((p.grid.rows, p.grid.cols, p.grid.tokens), (6, 4, 24));
        assert_eq!(p.grid.vars, vec![("grid_h", 12), ("grid_w", 8)]);
        assert_eq!(p.shape, vec![96, 24]);
        pinned(&p.data, PIN_SMART);
        // Over the pixel budget: 400x300 shrinks to 364x272 (multiples of 4).
        let g = measure(&smart(), &png(400, 300)).unwrap();
        assert_eq!((g.var("grid_h"), g.var("grid_w")), (Some(136), Some(182)));
        // Under it: 5x5 grows to 8x8.
        assert_eq!(measure(&smart(), &png(5, 5)).unwrap().tokens, 4);
        assert!(measure(&smart(), &png(1, 300)).unwrap_err().contains("aspect"));
    }

    fn budget() -> Preprocess {
        Preprocess::SoftTokenBudget { patch: 4, pool: 2, max_soft_tokens: 4, norm: half_norm() }
    }

    #[test]
    fn soft_token_budget_pinned() {
        // 40x30 within 16 patches of 4 px, sides multiples of 8: 16x8 → 2x4 patches, 1x2 tokens.
        let p = preprocess(&budget(), &png(40, 30)).unwrap();
        assert_eq!((p.grid.rows, p.grid.cols, p.grid.tokens), (1, 2, 2));
        assert_eq!(p.grid.vars, vec![("grid_h", 2), ("grid_w", 4)]);
        assert_eq!(p.shape, vec![8, 48]);
        pinned(&p.data, PIN_BUDGET);
        // Real-size budget: 280 tokens of 3x3 16-px patches.
        let real = Preprocess::SoftTokenBudget { patch: 16, pool: 3, max_soft_tokens: 280, norm: half_norm() };
        let g = measure(&real, &png(640, 480)).unwrap();
        assert_eq!((g.rows, g.cols, g.tokens), (PIN_REAL.0, PIN_REAL.1, PIN_REAL.0 * PIN_REAL.1));
    }

    fn patch_grid() -> Preprocess {
        Preprocess::PatchGrid {
            patch: 16,
            pool: 3,
            max_soft_tokens: 280,
            max_grid: Some(1120),
            norm: Norm::default(),
        }
    }

    #[test]
    fn patch_grid_pinned() {
        let p = preprocess(&patch_grid(), &png(200, 150)).unwrap();
        assert_eq!((p.grid.rows, p.grid.cols, p.grid.tokens), (PIN_GRID.0, PIN_GRID.1, PIN_GRID.0 * PIN_GRID.1));
        assert_eq!(p.shape, vec![p.grid.tokens, 48 * 48 * 3]);
        pinned(&p.data, PIN_PATCH);
        let tight = Preprocess::PatchGrid {
            patch: 16,
            pool: 3,
            max_soft_tokens: 280,
            max_grid: Some(4),
            norm: Norm::default(),
        };
        assert!(measure(&tight, &png(200, 150)).unwrap_err().contains("position embeddings"));
    }

    fn mel(frame_length: usize, hop_length: usize) -> Preprocess {
        Preprocess::LogMel {
            sampling_rate: 16000,
            frame_length,
            hop_length,
            pad_multiple: 128,
            max_samples: 480_000,
            token_stride: 4,
        }
    }

    #[test]
    fn log_mel_pinned() {
        // 0.5 s at 16 kHz: 8000 samples padded to 8064, 160 + 8064 = 8224 → 50 frames of 321
        // (the last one past the clip); every 4th of the 49 valid → 13 tokens.
        let p = preprocess(&mel(320, 160), &wav(16000, 8000)).unwrap();
        assert_eq!(p.grid.vars, vec![("frames", 50), ("valid_frames", 49), ("tokens", 13)]);
        assert_eq!(p.shape, vec![50, 320]);
        pinned(&p.data, PIN_MEL);
        // A 22.05 kHz clip is resampled first.
        assert_eq!(measure(&mel(320, 160), &wav(22050, 30000)).unwrap().tokens, PIN_MEL_TOKENS);
        let err = measure(&mel(320, 160), &wav(8000, 31 * 8000)).unwrap_err();
        assert!(err.contains("at most 30 s"), "{err}");
    }

    #[test]
    fn raw_frames_pinned() {
        let p = Preprocess::RawFrames { sampling_rate: 16000, frame: 640, max_tokens: 750 };
        // 30000 samples at 22.05 kHz → 21769 at 16 kHz → 35 frames of 640 (last zero-padded).
        let r = preprocess(&p, &wav(22050, 30000)).unwrap();
        assert_eq!((r.grid.tokens, r.shape.clone()), (35, vec![35, 640]));
        assert!(r.data[21769..].iter().all(|&v| v == 0.0));
        assert!(measure(&p, &wav(16000, 640 * 751)).unwrap_err().contains("at most"));
    }

    fn gemma_like() -> Media {
        let mut m = Media::new();
        m.insert("image".into(), spec(json!({"component": "vision",
            "preprocess": {"kind": "patch_grid", "patch": 16, "pool": 3, "max_soft_tokens": 280, "max_grid": 1120},
            "expand": {"kind": "repeat_between", "placeholder": 60, "open": 62, "close": 63},
            "bidirectional": true})));
        m.insert("audio".into(), spec(json!({"component": "audio",
            "preprocess": {"kind": "raw_frames", "sampling_rate": 16000, "frame": 640, "max_tokens": 750},
            "expand": {"kind": "repeat_between", "placeholder": 61, "open": 58, "close": 59}})));
        m
    }

    fn grid(rows: usize, cols: usize) -> Grid {
        Grid { rows, cols, tokens: rows * cols, vars: Vec::new() }
    }

    #[test]
    fn repeat_between_pinned() {
        let m = gemma_like();
        let (img, clip) = (grid(1, 2), grid(3, 1));
        let e = expand(&[2, 60, 5, 61, 7], &m, &[("image", &img), ("audio", &clip)], &no_text).unwrap();
        assert_eq!(e.ids, vec![2, 62, 60, 60, 63, 5, 58, 61, 61, 61, 59, 7]);
        assert_eq!(e.media_index, vec![2, 3, 7, 8, 9]);
        assert_eq!(e.blocks, vec![2..4]);
        assert!(e.pos3.is_empty() && e.delta == 0);
        let err = expand(&[60, 61], &m, &[("audio", &clip), ("image", &img)], &no_text).unwrap_err();
        assert!(err.contains("placeholder is `image`"), "{err}");
        assert!(expand(&[60], &m, &[], &no_text).unwrap_err().contains("1 media placeholder"));
    }

    const IMG: u32 = 5;
    const SPECIALS: [(&str, u32); 5] =
        [("<fake>", 3), ("<global-img>", 4), ("<image>", IMG), ("<row_1_col_1>", 6), ("<row_1_col_2>", 7)];

    /// Specials as single ids, other characters as `1000 + char`.
    fn tokenize(s: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        let mut rest = s;
        while let Some(ch) = rest.chars().next() {
            match SPECIALS.iter().find(|(t, _)| rest.starts_with(t)) {
                Some((t, id)) => {
                    out.push(*id);
                    rest = &rest[t.len()..];
                }
                None => {
                    out.push(1000 + ch as u32);
                    rest = &rest[ch.len_utf8()..];
                }
            }
        }
        Ok(out)
    }

    fn tiled() -> Media {
        let mut m = Media::new();
        m.insert("image".into(), spec(json!({"component": "vision",
            "preprocess": {"kind": "tile_longest_edge", "longest_edge": 96, "tile": 32, "max_side": 4096, "tokens_per_tile": 2},
            "expand": {"kind": "grid_text", "placeholder": IMG, "around": "<fake>",
                       "cell": "<row_{r}_col_{c}>", "global": "<global-img>", "token": "<image>"}})));
        m
    }

    #[test]
    fn grid_text_pinned() {
        let g = Grid { rows: 1, cols: 2, tokens: 6, vars: Vec::new() };
        let e = expand(&[1, IMG, 2], &tiled(), &[("image", &g)], &tokenize).unwrap();
        let nl = 1000 + '\n' as u32;
        assert_eq!(e.ids, vec![1, 3, 6, 5, 5, 3, 7, 5, 5, nl, nl, 3, 4, 5, 5, 3, 2]);
        assert_eq!(e.media_index, vec![3, 4, 7, 8, 13, 14]);
        let single = Grid { rows: 0, cols: 0, tokens: 2, vars: Vec::new() };
        let e = expand(&[IMG], &tiled(), &[("image", &single)], &tokenize).unwrap();
        assert_eq!(e.ids, vec![3, 4, 5, 5, 3]);
        assert_eq!(e.media_index, vec![2, 3]);
    }

    fn mrope() -> Media {
        let mut m = Media::new();
        m.insert("image".into(), spec(json!({"component": "vision",
            "preprocess": {"kind": "smart_resize", "patch": 2, "merge": 2, "temporal": 2,
                           "min_pixels": 64, "max_pixels": 100000, "norm": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
            "expand": {"kind": "merged_grid", "placeholder": 90}, "positions": "mrope_grid"})));
        m
    }

    #[test]
    fn merged_grid_pinned() {
        let g = grid(2, 3);
        let e = expand(&[3, 90, 11], &mrope(), &[("image", &g)], &no_text).unwrap();
        assert_eq!(e.ids, vec![3, 90, 90, 90, 90, 90, 90, 11]);
        assert_eq!(e.media_index, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(
            e.pos3,
            vec![[0; 3], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 1], [1, 2, 2], [1, 2, 3], [4; 3]]
        );
        assert_eq!(e.delta, 5 - 8);
        // Text-only prompts still get positions.
        let e = expand(&[3, 4], &mrope(), &[], &no_text).unwrap();
        assert_eq!((e.pos3, e.delta), (vec![[0; 3], [1; 3]], 0));
    }

    /// Token ids and sizes come from the model scope; quoted strings stay literal.
    #[test]
    fn resolve_evaluates_spec_expressions() {
        struct Cfg(serde_json::Value);
        impl Env for Cfg {
            fn lookup(&self, root: &str, path: &[crate::arch::expr::Val]) -> Option<crate::arch::expr::Val> {
                match root {
                    "cfg" => crate::arch::expr::walk(crate::arch::expr::Val::from_json(&self.0), path),
                    "patch" => Some(crate::arch::expr::Val::Num(16.0)),
                    _ => None,
                }
            }
        }
        let env = Cfg(json!({"image_token_id": 90, "min": 64, "bidir": "vision"}));
        let raw = json!({"image": {"component": "vision",
            "preprocess": {"kind": "smart_resize", "patch": "patch", "merge": 2, "temporal": 2,
                           "min_pixels": "cfg.min", "max_pixels": "cfg.min * 1000"},
            "expand": {"kind": "grid_text", "placeholder": "cfg.image_token_id", "around": "'<fake>'",
                       "cell": "'<row_{r}_col_{c}>'", "global": "'<g>'", "token": "'<image>'"},
            "positions": "mrope_grid", "bidirectional": "cfg.bidir == 'vision'"}});
        let m = resolve(&raw, &env).unwrap();
        let s = &m["image"];
        assert!(matches!(s.preprocess,
            Preprocess::SmartResize { patch: 16, min_pixels: 64, max_pixels: 64000, .. }));
        assert!(matches!(&s.expand, Expand::GridText { placeholder: 90, cell, .. } if cell == "<row_{r}_col_{c}>"));
        assert_eq!((s.positions, s.bidirectional), (Positions::MropeGrid, true));
    }

    const PIN_TILE: (f64, [f32; 3]) = (-1685.3751647472382, [-0.07450974, -0.09803915, -0.0666666]);
    const PIN_SMART: (f64, [f32; 3]) = (-164.63871252536774, [-0.09803915, -0.011764646, 0.68627465]);
    const PIN_BUDGET: (f64, [f32; 3]) = (-30.058731615543365, [0.17647064, -0.003921509, -0.18431365]);
    const PIN_REAL: (usize, usize) = (14, 19);
    const PIN_GRID: (usize, usize) = (14, 19);
    const PIN_PATCH: (f64, [f32; 3]) = (3605089.2276314674, [0.23137257, 0.427451, 0.19607845]);
    const PIN_MEL: (f64, [f32; 3]) = (12.474334716796875, [-0.24215698, -0.21524048, -0.062927246]);
    const PIN_MEL_TOKENS: usize = 34;
}
