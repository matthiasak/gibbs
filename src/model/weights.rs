//! Memory-mapped safetensors checkpoints for the native models: bf16 matrices are borrowed from
//! the mapping (no copy), so resident memory tracks the tensors actually touched and tensors a
//! model never reads (e.g. multi-token-prediction heads) are never paged in.

use crate::model::par::{par_for, OutPtr};
use crate::model::prequant::{glq, mlx};
use crate::model::{bf16_to_f32, f32_to_bf16, ModelManifest, UniversalModelLoader, VirtualTensor};
use memmap2::Mmap;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
enum Buf {
    Map(Arc<Mmap>, usize),
    /// Decoded at load (MLX-quantized or re-laid-out tensors).
    Owned(Arc<[u16]>),
    /// Zero-copy GGUF blocks mapped directly.
    Packed(Packed),
}

/// A bf16 tensor borrowed from a mapped checkpoint (or owned once decoded), viewed as
/// `[rows, cols]` (1-D: one row).
#[derive(Clone)]
pub struct Mat {
    buf: Buf,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    /// A `[rows, cols]` view of packed GGUF blocks (no copy).
    pub fn from_packed(p: Packed, rows: usize, cols: usize) -> Self {
        Self {
            buf: Buf::Packed(p),
            rows,
            cols,
        }
    }
    /// Owned `bytes` of `format` blocks viewed as `[rows, cols]`.
    pub fn from_packed_bytes(
        format: crate::hardware::QuantFormat,
        bytes: Vec<u8>,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self::from_packed(
            Packed {
                map: Arc::new(crate::model::loaders::gguf::GgufData::Buf(bytes)),
                off: 0,
                format,
                shape: vec![rows, cols],
            },
            rows,
            cols,
        )
    }
    pub fn packed(&self) -> Option<&Packed> {
        match &self.buf {
            Buf::Packed(p) => Some(p),
            _ => None,
        }
    }
    /// Owned bf16 values viewed as `[rows, cols]`.
    pub fn from_bf16(v: Vec<u16>, rows: usize, cols: usize) -> Self {
        Self {
            buf: Buf::Owned(v.into()),
            rows,
            cols,
        }
    }
    /// Whether the Metal executor gathers rows of this table on device (`Op::Embedding`).
    pub fn gathers_on_device(&self) -> bool {
        self.packed().is_none_or(|p| gathers_on_device(p.format))
    }
    pub fn to_graph_tensor(&self) -> crate::graph::Tensor {
        crate::graph::Tensor::Mat(self.clone())
    }
    pub fn data(&self) -> &[u16] {
        match &self.buf {
            // SAFETY: `Weights::mat` checked bounds and 2-byte alignment; the map lives in `self.buf`.
            Buf::Map(map, off) => unsafe {
                std::slice::from_raw_parts(
                    map.as_ptr().add(*off) as *const u16,
                    self.rows * self.cols,
                )
            },
            Buf::Owned(v) => v,
            Buf::Packed(_) => {
                panic!("Mat::data() called on Packed GGUF buffer; use to_f32() or packed()")
            }
        }
    }
    pub fn row(&self, r: usize) -> &[u16] {
        &self.data()[r * self.cols..(r + 1) * self.cols]
    }
    /// Decodes the first `out.len()` (at most `cols`) values of row `r` into `out`.
    pub fn decode_row_f32(&self, r: usize, out: &mut [f32]) {
        let n = out.len().min(self.cols);
        self.decode_slice_f32(r, 0, &mut out[..n]);
    }
    /// Decodes `dst.len()` values starting at column `p0` of row `r` into `dst`, block by block
    /// straight from the stored encoding (blocks straddling the range decode on the stack).
    pub fn decode_slice_f32(&self, r: usize, p0: usize, dst: &mut [f32]) {
        if let Buf::Packed(p) = &self.buf {
            let row_off = p.off + r * p.format.tensor_bytes(self.cols);
            let n = dst.len();
            if p.format == crate::hardware::QuantFormat::FP16 {
                let start = row_off + p0 * 2;
                for (o, c) in dst
                    .iter_mut()
                    .zip(p.map[start..start + n * 2].chunks_exact(2))
                {
                    *o = crate::model::f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                }
                return;
            }
            if p.format == crate::hardware::QuantFormat::BF16 {
                let start = row_off + p0 * 2;
                for (o, c) in dst
                    .iter_mut()
                    .zip(p.map[start..start + n * 2].chunks_exact(2))
                {
                    *o = crate::model::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                }
                return;
            }
            if p.format == crate::hardware::QuantFormat::FP32 {
                let start = row_off + p0 * 4;
                for (o, c) in dst
                    .iter_mut()
                    .zip(p.map[start..start + n * 4].chunks_exact(4))
                {
                    *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                return;
            }
            let Some((elems, block_bytes, decode)) =
                crate::model::layers::gguf_block_codec(p.format)
            else {
                panic!("cannot decode {:?} blocks", p.format);
            };
            let (b0, b1) = (p0 / elems, (p0 + n).div_ceil(elems));
            let blocks = &p.map[row_off + b0 * block_bytes..row_off + b1 * block_bytes];
            if p0 % elems == 0 && n % elems == 0 {
                for (b, out_blk) in blocks
                    .chunks_exact(block_bytes)
                    .zip(dst.chunks_exact_mut(elems))
                {
                    decode(b, out_blk);
                }
                return;
            }
            let mut t = [0.0f32; 256];
            for (i, b) in blocks.chunks_exact(block_bytes).enumerate() {
                let s = (b0 + i) * elems;
                decode(b, &mut t[..elems]);
                let (lo, hi) = (s.max(p0), (s + elems).min(p0 + n));
                dst[lo - p0..hi - p0].copy_from_slice(&t[lo - s..hi - s]);
            }
            return;
        }
        for (o, &v) in dst.iter_mut().zip(&self.row(r)[p0..]) {
            *o = bf16_to_f32(v);
        }
    }
    pub fn to_f32(&self) -> Vec<f32> {
        if let Buf::Packed(p) = &self.buf {
            let len = p.format.tensor_bytes(self.rows * self.cols);
            let bytes = &p.map[p.off..p.off + len];
            if p.format == crate::hardware::QuantFormat::FP16 {
                return bytes
                    .chunks_exact(2)
                    .map(|c| crate::model::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .take(self.rows * self.cols)
                    .collect();
            }
            if p.format == crate::hardware::QuantFormat::BF16 {
                return bytes
                    .chunks_exact(2)
                    .map(|c| crate::model::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .take(self.rows * self.cols)
                    .collect();
            }
            if p.format == crate::hardware::QuantFormat::FP32 {
                return bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .take(self.rows * self.cols)
                    .collect();
            }
            if let Some(mut v) = crate::model::layers::dequantize_gguf_blocks(p.format, bytes) {
                v.truncate(self.rows * self.cols);
                return v;
            } else {
                panic!("cannot dequantize {:?} blocks in Mat::to_f32", p.format);
            }
        }
        let mut out = vec![0.0f32; self.rows * self.cols];
        let src = self.data();
        let o = OutPtr::new(&mut out);
        const CH: usize = 1 << 16;
        par_for(src.len().div_ceil(CH), &|t| {
            let s = &src[t * CH..((t + 1) * CH).min(src.len())];
            // SAFETY: chunk t is this task's.
            let d = unsafe { o.slice(t * CH, s.len()) };
            for (d, &s) in d.iter_mut().zip(s) {
                *d = bf16_to_f32(s);
            }
        });
        out
    }
}

impl PartialEq for Mat {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        match (&self.buf, &other.buf) {
            (Buf::Packed(a), Buf::Packed(b)) => {
                a.off == b.off && a.shape == b.shape && a.format == b.format
            }
            (Buf::Packed(_), _) | (_, Buf::Packed(_)) => false,
            _ => self.data() == other.data(),
        }
    }
}

impl std::fmt::Debug for Mat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mat({}x{})", self.rows, self.cols)
    }
}

/// Whether the Metal executor gathers `Op::Embedding` rows from a table stored as `f`: the core
/// f32 kernel, the half-float and `graph::codec` gathers of `metal/ext/moe.rs`.
pub fn gathers_on_device(f: crate::hardware::QuantFormat) -> bool {
    use crate::hardware::QuantFormat as Q;
    matches!(f, Q::FP32 | Q::FP16 | Q::BF16) || crate::graph::codec::codec(f).is_some()
}

/// A GGUF tensor served straight from the file data (the mapped file, or a split set's buffer), still in its block format: mixture-of-
/// experts banks, whose dequantized size would not fit in memory. `shape` is `[experts, rows,
/// cols]` (rows of `cols` weights, blocks along `cols`).
#[derive(Clone)]
pub struct Packed {
    pub map: Arc<crate::model::loaders::gguf::GgufData>,
    pub off: usize,
    pub format: crate::hardware::QuantFormat,
    pub shape: Vec<usize>,
}

/// A safetensors checkpoint, memory-mapped: one `*.safetensors` file, or every shard a
/// `*.safetensors.index.json` weight map names (same directory). MLX checkpoints (a
/// `quantization` block in the config.json beside them, mlx-vlm tensor names) are served under
/// their HF names and layouts: packed weights decode to bf16 when read.
pub struct Weights {
    maps: Vec<Arc<Mmap>>,
    /// Tensor -> (index into `maps`, extent within that shard).
    tensors: HashMap<String, (usize, VirtualTensor)>,
    /// MLX-packed weights (stored name -> spec); their `shape` is already the logical one.
    packed: HashMap<String, mlx::Spec>,
    /// GLQ shards (index into `maps` -> that shard's prepared manifest); their aliased
    /// `X.weight`s decode to owned bf16 when read.
    glq: HashMap<usize, ModelManifest>,
    /// GGUF expert banks left in the file mapping, by HF name ([`Self::with_gguf_packed`]).
    gguf_packed: HashMap<String, Packed>,
    /// mlx-vlm checkpoint: names lack HF's `model.` nesting, convolutions are channels-last.
    vlm: bool,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let map_file = |p: &Path| -> Result<Mmap, String> {
            let file = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            // SAFETY: read-only mapping of a cache file nothing rewrites while the model is loaded.
            unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: {e}", p.display()))
        };
        let mut w = Self {
            maps: Vec::new(),
            tensors: HashMap::new(),
            packed: HashMap::new(),
            glq: HashMap::new(),
            gguf_packed: HashMap::new(),
            vlm: false,
        };
        // `model.safetensors.index.json`, or a diffusers variant index (`...index.fp16.json`).
        if !path.to_string_lossy().ends_with(".json") {
            let map = map_file(path)?;
            let manifest = UniversalModelLoader::parse_safetensors(&map)?;
            w.maps.push(Arc::new(map));
            w.tensors = manifest
                .tensors
                .into_iter()
                .map(|(n, t)| (n, (0, t)))
                .collect();
        } else {
            let json = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let index = UniversalModelLoader::parse_safetensors_index(&json)?;
            if index.is_empty() {
                return Err(format!("{}: empty or missing weight_map", path.display()));
            }
            let dir = path.parent().unwrap_or(Path::new("."));
            for shard in index.values().map(String::as_str).collect::<BTreeSet<_>>() {
                let map = map_file(&dir.join(shard))?;
                let manifest = UniversalModelLoader::parse_safetensors(&map)
                    .map_err(|e| format!("shard {shard}: {e}"))?;
                let s = w.maps.len();
                w.tensors
                    .extend(manifest.tensors.into_iter().map(|(n, t)| (n, (s, t))));
                w.maps.push(Arc::new(map));
            }
            if let Some((name, shard)) = index.iter().find(|(n, _)| !w.tensors.contains_key(*n)) {
                return Err(format!(
                    "tensor {name} listed in the index is absent from shard {shard}"
                ));
            }
        }
        if let Some(q) = mlx::config_beside(path)? {
            let mut found = Vec::new();
            for name in w.tensors.keys() {
                if let Some(p) = mlx::packed(&q, name, |n| w.tensors.get(n).map(|t| &t.1))? {
                    found.push((name.clone(), p));
                }
            }
            for (name, (spec, shape)) in found {
                w.tensors.get_mut(&name).unwrap().1.shape = shape;
                w.packed.insert(name, spec);
            }
        }
        w.attach_glq(path)?;
        w.vlm = w
            .tensors
            .keys()
            .any(|n| n.starts_with("language_model.model."));
        Ok(w)
    }

    /// Prepares each GLQ shard ([`glq::prepare`]: true shapes, shipped codebook) with the
    /// config.json beside `path`.
    fn attach_glq(&mut self, path: &Path) -> Result<(), String> {
        let config = path.parent().unwrap_or(Path::new(".")).join("config.json");
        for s in 0..self.maps.len() {
            let mut man = ModelManifest::default();
            man.tensors.extend(
                self.tensors
                    .iter()
                    .filter(|(_, (i, _))| *i == s)
                    .map(|(n, (_, t))| (n.clone(), t.clone())),
            );
            if !man.tensors.keys().any(|n| n.ends_with(".SV")) {
                continue;
            }
            glq::prepare(&mut man, Some(&config))?;
            for (n, t) in &man.tensors {
                self.tensors.get_mut(n).unwrap().1.shape = t.shape.clone();
            }
            self.glq.insert(s, man);
        }
        Ok(())
    }

    /// The shard manifest when `stored` is a GLQ-aliased weight.
    fn glq_of(&self, stored: &str, shard: usize) -> Option<&ModelManifest> {
        let man = self.glq.get(&shard)?;
        let stem = stored.strip_suffix(".weight")?;
        man.tensors
            .contains_key(&format!("{stem}.SV"))
            .then_some(man)
    }

    /// True for mlx-vlm checkpoints (`language_model.model.*`), whose converter may have rewritten
    /// values (e.g. some models store their `1 + w` RMSNorm weights pre-shifted).
    pub fn mlx_vlm(&self) -> bool {
        self.vlm
    }

    /// bf16 copies of the tensors `specs` names (`[rows, cols, ...]` shapes), each read as
    /// row-major f32 by `fetch`, held in an anonymous mapping: converted checkpoints (GGUF) run
    /// through the same [`Mat`] code as mapped safetensors.
    pub fn from_f32(
        specs: Vec<(String, Vec<usize>)>,
        fetch: impl Fn(&str) -> Result<Vec<f32>, String>,
    ) -> Result<Self, String> {
        let total: usize = specs
            .iter()
            .map(|(_, s)| s.iter().product::<usize>() * 2)
            .sum();
        let mut map = memmap2::MmapMut::map_anon(total.max(1)).map_err(|e| e.to_string())?;
        let (mut off, mut tensors) = (0, HashMap::new());
        for (name, shape) in specs {
            let data = fetch(&name)?;
            let len = shape.iter().product::<usize>() * 2;
            if data.len() * 2 != len {
                return Err(format!(
                    "tensor '{name}': {} values for shape {shape:?}",
                    data.len()
                ));
            }
            for (d, &v) in map[off..off + len].chunks_exact_mut(2).zip(&data) {
                d.copy_from_slice(&crate::model::f32_to_bf16(v).to_le_bytes());
            }
            let t = VirtualTensor {
                name: name.clone(),
                shape,
                quant_format: crate::hardware::QuantFormat::BF16,
                byte_offset: off,
                byte_len: len,
                dtype: "BF16".into(),
                shard_index: 0,
            };
            tensors.insert(name, (0, t));
            off += len;
        }
        let map = map.make_read_only().map_err(|e| e.to_string())?;
        Ok(Self {
            maps: vec![Arc::new(map)],
            tensors,
            packed: HashMap::new(),
            glq: HashMap::new(),
            gguf_packed: HashMap::new(),
            vlm: false,
        })
    }

    /// Serves `tensors` (HF name, byte offset in `map`, GGUF format, `[experts, rows, cols]`)
    /// packed from `map` ([`Self::packed_gguf`]).
    pub fn with_gguf_packed(
        mut self,
        map: &Arc<crate::model::loaders::gguf::GgufData>,
        tensors: impl IntoIterator<Item = (String, usize, crate::hardware::QuantFormat, Vec<usize>)>,
    ) -> Self {
        let cfg = crate::model::weight_format::ReencodeConfig::effective();
        let tensors: Vec<_> = tensors.into_iter().collect();
        if let Some(p) = &cfg.progress {
            p.total
                .fetch_add(tensors.len(), std::sync::atomic::Ordering::Relaxed);
        }
        for (name, off, format, shape) in tensors {
            let mut p = Packed {
                map: map.clone(),
                off,
                format,
                shape,
            };
            // A 2-D projection under a non-bit-exact policy may be re-stored in a faster format;
            // the result lives in memory instead of the file mapping.
            if let [rows, cols] = p.shape[..] {
                let len = rows * format.tensor_bytes(cols);
                if let Some(raw) = map.get(off..off + len) {
                    if let Some((f, blocks)) = crate::model::reencode::reencode_native(
                        &cfg, &name, format, rows, cols, raw,
                    ) {
                        p = Packed {
                            map: Arc::new(crate::model::loaders::gguf::GgufData::Buf(blocks)),
                            off: 0,
                            format: f,
                            shape: p.shape,
                        };
                    }
                }
            }
            self.gguf_packed.insert(name, p);
            if let Some(pr) = &cfg.progress {
                pr.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self
    }
    pub fn insert_packed(&mut self, name: String, p: Packed) {
        self.gguf_packed.insert(name, p);
    }
    /// Loads weights from a GGUF checkpoint, packing eligible Q4_0 and Q8_0 2D projection
    /// tensors into `gguf_packed` without dequantization, while dequantizing 1D vectors and
    /// transformed tensors into memory via `from_f32`.
    pub fn from_gguf<F, P, T>(
        data: &Arc<crate::model::loaders::gguf::GgufData>,
        m: &crate::model::ModelManifest,
        to_hf: F,
        should_transform: P,
        transform: T,
    ) -> Result<Self, String>
    where
        F: Fn(&str) -> Result<Option<String>, String>,
        P: Fn(&str) -> bool,
        T: Fn(&str, &mut [f32]),
    {
        let mut specs = Vec::new();
        let mut packed = Vec::new();
        let mut name_map = HashMap::new();

        for (name, t) in &m.tensors {
            let Some(hf) = to_hf(name)? else {
                continue;
            };
            let shape: Vec<usize> = t.shape.iter().rev().copied().collect();
            let can_pack = (shape.len() == 2 || hf.contains(".mlp.experts."))
                && (crate::graph::codec::codec(t.quant_format).is_some()
                    || matches!(
                        t.quant_format,
                        crate::hardware::QuantFormat::GGUF_Q4_0
                            | crate::hardware::QuantFormat::GGUF_Q8_0
                            | crate::hardware::QuantFormat::GGUF_IQ4_XS
                            | crate::hardware::QuantFormat::GGUF_IQ2_S
                            | crate::hardware::QuantFormat::GGUF_Q4_K_M
                            | crate::hardware::QuantFormat::GGUF_Q2_K
                            | crate::hardware::QuantFormat::GGUF_IQ3_S
                            | crate::hardware::QuantFormat::GGUF_TQ2_0
                    ))
                && !should_transform(&hf);

            if can_pack {
                packed.push((hf, t.byte_offset, t.quant_format, shape));
            } else {
                specs.push((hf.clone(), shape));
                name_map.insert(hf, name.as_str());
            }
        }

        let ck = Self::from_f32(specs, |hf| {
            let mut v = m.extract_tensor_f32(data, name_map[hf])?;
            transform(hf, &mut v);
            Ok(v)
        })?
        .with_gguf_packed(data, packed);

        Ok(ck)
    }

    pub fn packed_gguf(&self, name: &str) -> Option<&Packed> {
        self.gguf_packed.get(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.resolve(name).is_some() || self.gguf_packed.contains_key(name)
    }
    /// Every stored tensor name (checkpoint names, before any alias), packed GGUF banks included.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .keys()
            .chain(self.gguf_packed.keys())
            .map(String::as_str)
    }

    /// The stored dtype of `name` (`BF16`, `F32`, `U32`, ...).
    pub fn stored_dtype(&self, name: &str) -> Option<&str> {
        self.resolve(name).map(|(_, (_, t))| t.dtype.as_str())
    }

    /// Whether [`Self::mat`] serves `name` as bf16 by decoding (MLX-packed, GLQ, EXL3 trellis)
    /// or straight from the mapping (plain bf16), rather than refusing a non-float dtype.
    pub fn decodes_to_bf16(&self, name: &str) -> bool {
        let Some((stored, (s, t))) = self.resolve(name) else {
            return false;
        };
        t.dtype == "BF16"
            || self.packed.contains_key(&stored)
            || self.glq_of(&stored, *s).is_some()
            || name
                .strip_suffix(".weight")
                .is_some_and(|s| t.dtype == "I16" && self.has(&format!("{s}.trellis")))
    }

    /// Whether `name` is stored as plain bf16, which [`Self::mat`] serves without rounding.
    pub fn is_bf16(&self, name: &str) -> bool {
        self.resolve(name)
            .is_some_and(|(s, (_, t))| t.dtype == "BF16" && !self.packed.contains_key(&s))
    }

    /// `(stored name, (shard, extent))` of HF tensor `name`.
    fn resolve(&self, name: &str) -> Option<(String, &(usize, VirtualTensor))> {
        if let Some(t) = self.tensors.get(name) {
            return Some((name.to_string(), t));
        }
        let alt = mlx::vlm_name(name).filter(|_| self.vlm)?;
        self.tensors.get(&alt).map(|t| (alt, t))
    }

    fn get(&self, name: &str) -> Result<(String, &(usize, VirtualTensor)), String> {
        self.resolve(name)
            .ok_or_else(|| format!("tensor '{name}' missing"))
    }

    /// mlx-vlm convolutions are channels-last (`[out, *kernel, in]`); HF's are `[out, in, *kernel]`.
    /// Only convolutions: other 3-D tensors (e.g. a `[2, size, hidden]` position table) are
    /// stored as HF has them.
    fn channels_last(&self, stored: &str, t: &VirtualTensor) -> bool {
        self.vlm
            && t.shape.len() >= 3
            && !self.packed.contains_key(stored)
            && (stored.contains("conv") || stored.ends_with("patch_embed.proj.weight"))
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        let (stored, (_, t)) = self.get(name)?;
        let mut s = t.shape.clone();
        if self.channels_last(&stored, t) {
            s[1..].rotate_right(1);
        }
        Ok(s)
    }

    /// `name` as a bf16 matrix (trailing dims flattened into `cols`). Non-bf16 matrices (F16/F32,
    /// EXL3 trellis weights) are expanded and rounded to bf16 in an anonymous map.
    pub fn mat(&self, name: &str) -> Result<Mat, String> {
        if let Some(p) = self.gguf_packed.get(name) {
            let rows = p.shape.first().copied().unwrap_or(1);
            let cols = p.shape.iter().skip(1).product::<usize>();
            return Ok(Mat {
                buf: Buf::Packed(p.clone()),
                rows,
                cols,
            });
        }
        let (stored, (s, t)) = self.get(name)?;
        let rows = t.shape.first().copied().unwrap_or(1);
        let cols = t.shape.iter().skip(1).product::<usize>();
        if let Some(man) = self.glq_of(&stored, *s) {
            let v = glq::dequantize_bf16(man, &self.maps[*s], &stored)?
                .ok_or_else(|| format!("tensor '{name}': GLQ payload missing"))?;
            return Ok(Mat {
                buf: Buf::Owned(v.into()),
                rows,
                cols,
            });
        }
        if let Some(p) = self.mlx_packed(&stored, t)? {
            let v = p
                .decode_bf16()
                .map_err(|e| format!("tensor '{name}': {e}"))?;
            return Ok(Mat {
                buf: Buf::Owned(v.into()),
                rows,
                cols,
            });
        }
        // MLX writes unpadded headers, so its bf16 tensors may sit at odd offsets: copy those.
        let odd = t.dtype == "BF16" && t.byte_offset % 2 != 0;
        if self.channels_last(&stored, t) || odd {
            let v: Vec<u16> = self.vec(name)?.into_iter().map(f32_to_bf16).collect();
            return Ok(Mat {
                buf: Buf::Owned(v.into()),
                rows,
                cols,
            });
        }
        if t.dtype != "BF16" {
            return self.converted(name, t);
        }
        let map = &self.maps[*s];
        if t.byte_len != rows * cols * 2 || t.byte_offset + t.byte_len > map.len() {
            return Err(format!("tensor '{name}': bad bf16 extent"));
        }
        Ok(Mat {
            buf: Buf::Map(map.clone(), t.byte_offset),
            rows,
            cols,
        })
    }

    fn raw(&self, stored: &str) -> Option<(&[u8], &str)> {
        let (s, t) = self.tensors.get(stored)?;
        Some((
            self.maps[*s].get(t.byte_offset..t.byte_offset + t.byte_len)?,
            &t.dtype,
        ))
    }

    /// MLX-packed `stored` with its scales / biases.
    fn mlx_packed(
        &self,
        stored: &str,
        t: &VirtualTensor,
    ) -> Result<Option<mlx::Packed<'_>>, String> {
        let Some(&spec) = self.packed.get(stored) else {
            return Ok(None);
        };
        let stem = &stored[..stored.len() - ".weight".len()];
        let bad = || format!("tensor '{stored}': bad extent");
        let cols = *t.shape.last().unwrap();
        Ok(Some(mlx::Packed {
            spec,
            rows: t.shape.iter().product::<usize>() / cols,
            cols,
            w: self.raw(stored).ok_or_else(bad)?.0,
            scales: self.raw(&format!("{stem}.scales")).ok_or_else(bad)?,
            biases: self.raw(&format!("{stem}.biases")),
        }))
    }

    /// A non-bf16 matrix (F16/F32, EXL3 trellis) expanded and rounded to owned bf16.
    fn converted(&self, name: &str, t: &VirtualTensor) -> Result<Mat, String> {
        let exl3 = name
            .strip_suffix(".weight")
            .filter(|s| t.dtype == "I16" && self.has(&format!("{s}.trellis")));
        let get = |n: &str| {
            let (s, t) = self.tensors.get(n)?;
            Some((
                t,
                self.maps[*s].get(t.byte_offset..t.byte_offset + t.byte_len)?,
            ))
        };
        let v = match exl3 {
            Some(stem) => crate::model::prequant::exl3::dequantize(&get, stem)?,
            None if matches!(t.dtype.as_str(), "F16" | "F32") => self.vec(name)?,
            None => {
                return Err(format!(
                    "tensor '{name}' is {}; only bf16/f16/f32, MLX-packed or EXL3 weight matrices are supported",
                    t.dtype
                ))
            }
        };
        Ok(Mat {
            buf: Buf::Owned(v.into_iter().map(f32_to_bf16).collect()),
            rows: t.shape.first().copied().unwrap_or(1),
            cols: t.shape.iter().skip(1).product::<usize>(),
        })
    }

    /// `name` decoded to f32 (small tensors: norms, biases, scalars).
    pub fn vec(&self, name: &str) -> Result<Vec<f32>, String> {
        if let Some(p) = self.gguf_packed.get(name) {
            let bytes = &p.map[p.off..];
            if let Some(v) = crate::model::layers::dequantize_gguf_blocks(p.format, bytes) {
                return Ok(v);
            }
        }
        let (stored, (s, t)) = self.get(name)?;
        let err = |e: String| format!("tensor '{name}': {e}");
        if let Some(man) = self.glq_of(&stored, *s) {
            return glq::dequantize(man, &self.maps[*s], &stored)?
                .ok_or_else(|| err("GLQ payload missing".into()));
        }
        if let Some(p) = self.mlx_packed(&stored, t)? {
            return p.decode().map_err(err);
        }
        let (raw, dtype) = self.raw(&stored).ok_or_else(|| err("bad extent".into()))?;
        let v = crate::model::decode_dtype(raw, dtype).map_err(err)?;
        if !self.channels_last(&stored, t) {
            return Ok(v);
        }
        // [out, k, in] -> [out, in, k] with k the flattened kernel.
        let (o, i) = (t.shape[0], *t.shape.last().unwrap());
        let k = v.len() / (o * i).max(1);
        let mut out = vec![0f32; v.len()];
        for (a, src) in out.chunks_exact_mut(i * k).zip(v.chunks_exact(i * k)) {
            for (ki, row) in src.chunks_exact(i).enumerate() {
                for (ci, &x) in row.iter().enumerate() {
                    a[ci * k + ki] = x;
                }
            }
        }
        Ok(out)
    }

    /// An `I64` tensor's values (non-negative: hash constants too wide for [`Self::vec`]'s f32).
    pub fn u64s(&self, name: &str) -> Result<Vec<u64>, String> {
        let (stored, _) = self.get(name)?;
        match self.raw(&stored) {
            Some((raw, "I64")) => Ok(raw
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
                .collect()),
            _ => Err(format!("tensor '{name}' is not an I64 tensor")),
        }
    }

    pub fn scalar(&self, name: &str) -> Result<f32, String> {
        self.vec(name)?
            .first()
            .copied()
            .ok_or_else(|| format!("tensor '{name}' is empty"))
    }

    /// `vec` with its length checked.
    pub fn vec_n(&self, name: &str, n: usize) -> Result<Vec<f32>, String> {
        let v = self.vec(name)?;
        if v.len() != n {
            return Err(format!(
                "tensor '{name}' has {} values, expected {n}",
                v.len()
            ));
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::Weights;
    use std::path::PathBuf;

    fn snapshot(repo: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "disk/models--{}/snapshots/main",
            repo.replace('/', "--")
        ))
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        d / (na.sqrt() * nb.sqrt())
    }

    /// mlx-community Qwen3.5 checkpoints (mlx-vlm names, packed language model, channels-last
    /// patch embed, pre-shifted norms) read under HF names vs Qwen/Qwen3.5-0.8B: cosine of every
    /// packed tensor, exact vision weights, then a greedy completion through the qwen3_5 spec.
    #[test]
    fn mlx_qwen3_5_matches_hf_original() {
        let orig = snapshot("Qwen/Qwen3.5-0.8B").join("model.safetensors.index.json");
        if !orig.exists() {
            eprintln!("SKIP: Qwen/Qwen3.5-0.8B not in disk/");
            return;
        }
        let r = Weights::open(&orig).unwrap();
        for (variant, min_cos) in [
            ("3bit", 0.95),
            ("4bit", 0.99),
            ("mixed_3_6", 0.95),
            ("6bit", 0.998),
            ("8bit", 0.999),
            ("mxfp4", 0.98),
            ("nvfp4", 0.99),
            ("mxfp8", 0.995),
        ] {
            let repo = format!("mlx-community/Qwen3.5-0.8B-{variant}");
            let dir = snapshot(&repo);
            if !dir.join("model.safetensors").exists() {
                eprintln!("SKIP: {repo} not in disk/");
                continue;
            }
            let w = Weights::open(&dir.join("model.safetensors")).unwrap();
            assert!(w.mlx_vlm() && !w.packed.is_empty(), "{repo}");
            // Layer 3 (full attention), 14 and 19 (linear attention) and the tied embedding cover
            // every packed kind.
            let sample: Vec<&String> = w
                .packed
                .keys()
                .filter(|n| {
                    [".3.", ".14.", ".19.", "embed"]
                        .iter()
                        .any(|l| n.contains(l))
                })
                .collect();
            let (mut worst, mut sum) = ((1.0f64, String::new()), 0f64);
            for stored in &sample {
                let hf = stored.replacen("language_model.model.", "model.language_model.", 1);
                let (a, b) = (w.vec(&hf).unwrap(), r.vec(&hf).unwrap());
                assert_eq!(a.len(), b.len(), "{hf}");
                let c = cosine(&a, &b);
                sum += c;
                if c < worst.0 {
                    worst = (c, hf);
                }
            }
            println!(
                "{repo}: {} of {} packed tensors, mean cos {:.5}, min cos {:.5} ({})",
                sample.len(),
                w.packed.len(),
                sum / sample.len() as f64,
                worst.0,
                worst.1
            );
            assert!(worst.0 >= min_cos, "{repo}: {} < {min_cos}", worst.0);
            for name in [
                "model.visual.patch_embed.proj.weight",
                "model.visual.blocks.0.attn.qkv.weight",
            ] {
                assert_eq!(w.shape(name).unwrap(), r.shape(name).unwrap(), "{name}");
                assert_eq!(
                    w.mat(name).unwrap().data(),
                    r.mat(name).unwrap().data(),
                    "{name}"
                );
            }
            // mlx-vlm stores `1 + w`, rounded to bf16.
            let n = "model.language_model.layers.0.input_layernorm.weight";
            let shifted: Vec<f32> = r.vec(n).unwrap().iter().map(|v| v + 1.0).collect();
            assert!(cosine(&w.vec(n).unwrap(), &shifted) > 0.9999);
        }

        let dir = snapshot("mlx-community/Qwen3.5-0.8B-4bit");
        if !dir.join("model.safetensors").exists() {
            eprintln!("SKIP: mlx-community/Qwen3.5-0.8B-4bit not in disk/");
            return;
        }
        let artifacts = crate::model::hub::ModelArtifacts {
            model_id: "mlx-community/Qwen3.5-0.8B-4bit".into(),
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors")),
            ..Default::default()
        };
        let reg = crate::arch::registry().unwrap();
        let cpu = crate::hardware::Device::Cpu.into();
        let lm = crate::runtime::lm::load(&artifacts, reg, None, cpu).unwrap();
        let ids = lm.tokenizer().encode("The capital of France is");
        let (mut ids, mut state) = lm.prepare(ids, &[]).unwrap();
        let mut pos = 0;
        let mut out = Vec::new();
        for _ in 0..4 {
            let logits = lm.forward(&ids[pos..], pos, &mut state).unwrap();
            pos = ids.len();
            let next = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap() as u32;
            ids.push(next);
            out.push(next);
        }
        let text = lm.tokenizer().decode(&out);
        println!("Qwen3.5-0.8B-4bit: The capital of France is{text:?}");
        assert!(text.contains("Paris"), "{text:?}");
    }

    #[test]
    fn test_packed_quant_gemv_synthetic() {
        use crate::hardware::QuantFormat;
        // 2 rows, 64 cols (2 blocks of 32 per row)
        let rows = 2;
        let cols = 64;
        let mut raw_bytes = Vec::new();

        for r in 0..rows {
            for b in 0..2 {
                let scale = 0.5f32 * (r as f32 + 1.0) * (b as f32 + 1.0);
                let d_f16 = crate::model::f32_to_f16(scale);
                raw_bytes.extend(d_f16.to_le_bytes());
                // 16 bytes of nibbles (32 values)
                for i in 0..16 {
                    let lo = (i % 8) as u8;
                    let hi = ((i + 1) % 8) as u8;
                    raw_bytes.push(lo | (hi << 4));
                }
            }
        }

        let mut map = memmap2::MmapMut::map_anon(raw_bytes.len().max(1)).unwrap();
        map[..raw_bytes.len()].copy_from_slice(&raw_bytes);
        let map = std::sync::Arc::new(crate::model::loaders::gguf::GgufData::Map(
            map.make_read_only().unwrap(),
        ));
        let weights = Weights::from_f32(vec![], |_| Ok(vec![]))
            .unwrap()
            .with_gguf_packed(
                &map,
                vec![(
                    "layer.weight".into(),
                    0,
                    QuantFormat::GGUF_Q4_0,
                    vec![rows, cols],
                )],
            );

        let mat = weights.mat("layer.weight").unwrap();
        assert!(mat.packed().is_some());
        assert_eq!(mat.rows, rows);
        assert_eq!(mat.cols, cols);

        // Test row decoding
        let mut row0 = vec![0.0f32; cols];
        mat.decode_row_f32(0, &mut row0);
        assert_eq!(row0.len(), cols);

        // Test GEMV
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut out_gemv = vec![0.0f32; rows];
        crate::model::qdot::linear_q(&mat, &x, 1, None, &mut out_gemv);

        // Reference GEMV using full to_f32
        let w = mat.to_f32();
        let mut ref_out = vec![0.0f32; rows];
        crate::model::nn::matmul_nt(&x, &w, None, 1, cols, rows, &mut ref_out);

        for (j, (a, b)) in out_gemv.iter().zip(&ref_out).enumerate() {
            // Int8 activations err by at most half a step (max|x| / 254 <= 1 / 254) per element.
            let tol = w[j * cols..][..cols].iter().map(|v| v.abs()).sum::<f32>() / 250.0 + 1e-4;
            assert!((a - b).abs() < tol, "gemv {a} vs ref {b}");
        }
    }
}
