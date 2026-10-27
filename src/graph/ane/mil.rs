//! Lowering of a graph fragment to ANE MIL text (`program(1.3)`, `func main<ios18>`) plus one
//! `weight.bin`.
//!
//! Every program input and output is an fp32 `[1, C, 1, S]` surface (channel-major, each channel
//! row padded to 64 bytes); the program casts to fp16 on entry and back to fp32 on exit, since the
//! ANE computes in fp16 only. A row-major `[S, C]` activation (tokens × features) lives in MIL as
//! `[1, C, 1, S]` ("token form", so Linear is a 1×1 `conv`); every other value keeps its row-major
//! element order under any 4-D shape ("row form", so reshapes are free). Weights are `BLOBFILE`
//! constants in `weight.bin`: a 64-byte file header, then per tensor a 64-byte chunk header (magic
//! `DEADBEEF`, blob dtype at +4, byte size at +8, absolute data offset at +16) and its data.

use super::AneInfo;
use crate::graph::cpu::{attention_key_range, channel_layout, rope_position, upsample_nsp};
use crate::graph::{
    BinaryKind, DType, Graph, MegakernelKind, Op, PairLayout, ReduceKind, RopeVariant, Tensor,
    UnaryKind, UpsampleMode, ValueId, ValueKind, Weights,
};
use crate::hardware::QuantFormat;
use std::collections::HashMap;
use std::fmt::Write;

const HEADER: &str = "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n";
const BLOB_PATH: &str = "@model_path/weights/weight.bin";
const FP16_MAX: f32 = 65504.0;
/// MILBlob storage dtypes.
const BLOB_FP16: u32 = 1;
const BLOB_INT8: u32 = 4;
const BLOB_INT4: u32 = 8;

/// One program input or output surface.
#[derive(Debug, Clone, PartialEq)]
pub struct IoBinding {
    pub source: IoSource,
    /// `C` of the fp32 `[1, C, 1, S]` surface.
    pub channels: usize,
    /// `S` of the surface.
    pub spatial: usize,
    /// The graph tensor is row-major `[S, C]` and is transposed on the way in or out; otherwise
    /// its row-major order already is `[C, S]`.
    pub transposed: bool,
}

/// What the host writes into an input surface (outputs are always [`IoSource::Value`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoSource {
    /// A graph input (U32 inputs are converted to f32) or graph output.
    Value(ValueId),
    /// The `[cos; sin]` table (`C = rotary_dim`, `S = rows`) of the `Rope` node at this index in
    /// `graph.nodes`, computed on the host in f32 from its positions and `inv_freq`
    /// ([`rope_table`]).
    RopeTable(usize),
}

impl IoBinding {
    /// Bytes per channel row (rows are 64-byte aligned).
    pub fn row_bytes(&self) -> usize {
        (self.spatial * 4).next_multiple_of(64)
    }

    pub fn bytes(&self) -> usize {
        self.channels * self.row_bytes()
    }

    /// Writes graph row-major `data` into a surface's bytes.
    pub fn pack(&self, data: &[f32], dst: &mut [u8]) {
        let (c, s, rb) = (self.channels, self.spatial, self.row_bytes());
        for ci in 0..c {
            let row = &mut dst[ci * rb..][..s * 4];
            for (si, o) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let v = if self.transposed {
                    data[si * c + ci]
                } else {
                    data[ci * s + si]
                };
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    /// Reads a surface's bytes back in graph row-major order.
    pub fn unpack(&self, src: &[u8]) -> Vec<f32> {
        let (c, s, rb) = (self.channels, self.spatial, self.row_bytes());
        let mut out = vec![0.0f32; c * s];
        for ci in 0..c {
            for (si, b) in src[ci * rb..][..s * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
            {
                let v = f32::from_le_bytes(*b);
                out[if self.transposed {
                    si * c + ci
                } else {
                    ci * s + si
                }] = v;
            }
        }
        out
    }
}

/// Program inputs (`in_00…`) and outputs (`out_00…`, in `graph.outputs` order), in MIL argument
/// order (which is also alphabetical, the order the framework binds surfaces in).
#[derive(Debug, Clone, PartialEq)]
pub struct IoLayout {
    pub inputs: Vec<IoBinding>,
    pub outputs: Vec<IoBinding>,
}

impl IoLayout {
    /// Size of every input surface (one request needs equal input sizes).
    pub fn in_bytes(&self) -> usize {
        self.inputs.iter().map(IoBinding::bytes).max().unwrap_or(64)
    }

    /// Size of every output surface.
    pub fn out_bytes(&self) -> usize {
        self.outputs
            .iter()
            .map(IoBinding::bytes)
            .max()
            .unwrap_or(64)
    }
}

/// `[cos; sin]` (`rotary_dim × rows`, channel-major) of a `Rope` op, scaled by `mscale`.
pub fn rope_table(
    op: &Op,
    positions: &[u32],
    inv_freq: &[f32],
    rows: usize,
) -> Result<Vec<f32>, String> {
    let Op::Rope {
        rotary_dim,
        variant,
        mscale,
        ..
    } = op
    else {
        return Err(format!("{} is not a Rope", op.name()));
    };
    let half = rotary_dim / 2;
    if inv_freq.len() != half {
        return Err(format!(
            "Rope inv_freq has {} entries, expected {half}",
            inv_freq.len()
        ));
    }
    let mut t = vec![0.0f32; rotary_dim * rows];
    for (j, &f) in inv_freq.iter().enumerate() {
        for s in 0..rows {
            let theta = rope_position(variant, positions, s, j) as f32 * f;
            t[j * rows + s] = theta.cos() * mscale;
            t[(half + j) * rows + s] = theta.sin() * mscale;
        }
    }
    Ok(t)
}

/// Whether [`lower`] handles `op`; `dtype` is the placement dtype (the weight's for `Linear` and
/// `Megakernel`, else the output's) and `shapes` the operand shapes. State ops, gathers,
/// scatters, `TopK`, `MoeLinear`, `Conv3d`, `Concat`, `Ln`, `RoundEven` and `Sign` stay on Metal.
pub fn supports_op(op: &Op, dtype: DType, shapes: &[&[usize]]) -> bool {
    if shapes.iter().any(|s| s.len() > 4) {
        return false;
    }
    let float = matches!(dtype, DType::F32 | DType::F16 | DType::BF16);
    match op {
        Op::Linear | Op::Megakernel(_) => dtype != DType::U32,
        Op::MatMul { .. } => float && shapes.iter().all(|s| s.len() >= 2),
        Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::Rope { .. }
        | Op::Attention { .. }
        | Op::SwiGlu
        | Op::GeGlu
        | Op::Softmax { .. }
        | Op::Reduce { .. }
        | Op::Transpose { .. }
        | Op::Slice { .. }
        | Op::Squeeze { .. }
        | Op::Unsqueeze { .. }
        | Op::Tile { .. }
        | Op::Pad { .. }
        | Op::Conv1d { .. }
        | Op::Conv2d { .. }
        | Op::MeanPool
        | Op::L2Normalize => float,
        Op::AvgPool2d { .. } => float && shapes.first().is_some_and(|s| s.len() >= 3),
        Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
            float
                && match shapes {
                    [x, g, ..] => channel_layout(x, g.iter().product()).is_ok_and(|(n, _)| n == 1),
                    _ => false,
                }
        }
        Op::Reshape { shape } => float && shape.is_some(),
        Op::Unary { kind } => {
            float && !matches!(kind, UnaryKind::Ln | UnaryKind::RoundEven | UnaryKind::Sign)
        }
        Op::Binary { .. } => {
            float
                && match shapes {
                    [a, b] => broadcast(a, b).is_some(),
                    _ => false,
                }
        }
        Op::Upsample { mode, .. } => {
            float
                && matches!(mode, UpsampleMode::Nearest | UpsampleMode::Bilinear)
                && shapes.first().is_some_and(|s| s.len() >= 2)
        }
        _ => false,
    }
}

/// Lowers every node of `graph` (whose inputs and outputs are the program's) to MIL text and
/// `weight.bin` (empty for weight-free programs). Linear weights use the most compact form the
/// probe passed: `Q4_0` as int4 and `Q8_0` as int8 with per-32 fp16 scales
/// (`constexpr_blockwise_shift_scale`), other quantized formats as per-output-channel int8
/// (`constexpr_affine_dequantize`, absmax / 127 of the decoded row), float weights as fp16.
pub fn lower(
    graph: &Graph,
    weights: &Weights,
    info: &AneInfo,
) -> Result<(String, Vec<u8>, IoLayout), String> {
    let mut l = Lower {
        g: graph,
        w: weights,
        info,
        body: String::new(),
        blob: vec![0; 64],
        blobs: 0,
        vals: HashMap::new(),
        views: HashMap::new(),
        shapes: HashMap::new(),
        ins: Vec::new(),
        n: 0,
    };
    for i in 0..graph.nodes.len() {
        l.node(i)
            .map_err(|e| format!("ANE lowering of node '{}': {e}", graph.nodes[i].name))?;
    }
    let to32 = l.string("fp32");
    let mut outs = Vec::new();
    let mut tail = String::new();
    for (j, &v) in graph.outputs.iter().enumerate() {
        let shape = graph.value(v).shape.clone();
        let (s, c) = flat2(&shape);
        let m = l.mv(v)?;
        let (b, var) = if m.tok && s > 1 && c > 1 {
            (io(v, c, s, true), m.var)
        } else {
            let (ch, sp) = row_cs(&shape);
            (
                io(v, ch, sp, false),
                l.operand(v, Want::Row([1, ch, 1, sp]))?,
            )
        };
        writeln!(
            tail,
            "        tensor<fp32, [1, {}, 1, {}]> out_{j:02} = cast(dtype = {to32}, x = {var})[name = string(\"out_{j:02}\")];",
            b.channels, b.spatial
        )
        .unwrap();
        outs.push(b);
    }
    let args: Vec<String> = l
        .ins
        .iter()
        .enumerate()
        .map(|(i, b)| {
            format!(
                "tensor<fp32, [1, {}, 1, {}]> in_{i:02}",
                b.channels, b.spatial
            )
        })
        .collect();
    let names: Vec<String> = (0..outs.len()).map(|j| format!("out_{j:02}")).collect();
    let mil = format!(
        "{HEADER}    func main<ios18>({}) {{\n{}{tail}    }} -> ({});\n}}\n",
        args.join(", "),
        l.body,
        names.join(", ")
    );
    let weights = if l.blobs == 0 {
        Vec::new()
    } else {
        l.blob[0..4].copy_from_slice(&(l.blobs as u32).to_le_bytes());
        l.blob[4..8].copy_from_slice(&2u32.to_le_bytes());
        l.blob
    };
    Ok((
        mil,
        weights,
        IoLayout {
            inputs: l.ins,
            outputs: outs,
        },
    ))
}

fn io(v: ValueId, channels: usize, spatial: usize, transposed: bool) -> IoBinding {
    IoBinding {
        source: IoSource::Value(v),
        channels,
        spatial,
        transposed,
    }
}

/// `(S, C)` of a value viewed as `[S, C]` (C the last dim).
fn flat2(shape: &[usize]) -> (usize, usize) {
    let c = shape.last().copied().unwrap_or(1).max(1);
    (shape.iter().product::<usize>() / c, c)
}

/// `(C, S)` of a row-form surface: the leading dim by the rest (rank ≤ 1: one channel).
fn row_cs(shape: &[usize]) -> (usize, usize) {
    let n: usize = shape.iter().product();
    match shape {
        [c, _, ..] if *c > 0 => (*c, n / c),
        _ => (1, n),
    }
}

fn pad4(shape: &[usize]) -> Result<[usize; 4], String> {
    if shape.len() > 4 {
        return Err(format!("rank {} > 4", shape.len()));
    }
    let mut p = [1; 4];
    p[4 - shape.len()..].copy_from_slice(shape);
    Ok(p)
}

/// Numpy broadcast of two shapes, `None` when incompatible.
fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let at = |s: &[usize], i: usize| {
        if i + s.len() >= r {
            s[i + s.len() - r]
        } else {
            1
        }
    };
    (0..r)
        .map(|i| match (at(a, i), at(b, i)) {
            (x, y) if x == y || y == 1 => Some(x),
            (1, y) => Some(y),
            _ => None,
        })
        .collect()
}

fn dims(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// MIL fp16 literal (clamped to the finite fp16 range).
fn lit(x: f32) -> String {
    format!("{}", x.clamp(-FP16_MAX, FP16_MAX))
}

/// A lowered value: `var` holds it in token form (`[1, C, 1, S]`) or in row-major order.
#[derive(Clone)]
struct Mv {
    var: String,
    shape: [usize; 4],
    tok: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Want {
    Tok,
    Row([usize; 4]),
}

struct Lower<'a> {
    g: &'a Graph,
    w: &'a Weights,
    info: &'a AneInfo,
    body: String,
    blob: Vec<u8>,
    blobs: usize,
    vals: HashMap<ValueId, Mv>,
    views: HashMap<(ValueId, Want), String>,
    shapes: HashMap<String, [usize; 4]>,
    ins: Vec<IoBinding>,
    n: usize,
}

impl<'a> Lower<'a> {
    fn shape(&self, v: ValueId) -> Vec<usize> {
        self.g.value(v).shape.clone()
    }

    fn fresh(&mut self) -> String {
        self.n += 1;
        format!("v{}", self.n)
    }

    /// Emits `tensor<ty, shape> v = call[name = "v"]`.
    fn emit(&mut self, ty: &str, shape: &[usize], call: String) -> String {
        let v = self.fresh();
        writeln!(
            self.body,
            "        tensor<{ty}, [{}]> {v} = {call}[name = string(\"{v}\")];",
            dims(shape)
        )
        .unwrap();
        if let Ok(s) = <[usize; 4]>::try_from(shape) {
            self.shapes.insert(v.clone(), s);
        }
        v
    }

    fn op(&mut self, shape: [usize; 4], call: String) -> String {
        self.emit("fp16", &shape, call)
    }

    fn konst(&mut self, ty: &str, val: String) -> String {
        let v = self.fresh();
        writeln!(
            self.body,
            "        {ty} {v} = const()[name = string(\"{v}\"), val = {val}];"
        )
        .unwrap();
        v
    }

    fn ints(&mut self, xs: &[usize]) -> String {
        let ty = format!("tensor<int32, [{}]>", xs.len());
        self.konst(&ty, format!("{ty}([{}])", dims(xs)))
    }

    fn int(&mut self, x: usize) -> String {
        self.konst("int32", format!("int32({x})"))
    }

    fn boolean(&mut self, b: bool) -> String {
        self.konst("bool", format!("bool({b})"))
    }

    fn string(&mut self, s: &str) -> String {
        self.konst("string", format!("string(\"{s}\")"))
    }

    fn scalar(&mut self, x: f32) -> String {
        self.konst("fp16", format!("fp16({})", lit(x)))
    }

    fn sh(&self, v: &str) -> [usize; 4] {
        self.shapes.get(v).copied().unwrap_or([1; 4])
    }

    /// Appends one `weight.bin` chunk; returns its `BLOBFILE` reference.
    fn chunk(&mut self, dtype: u32, data: &[u8]) -> String {
        let off = self.blob.len();
        let mut h = [0u8; 64];
        h[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        h[4..8].copy_from_slice(&dtype.to_le_bytes());
        h[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
        h[16..24].copy_from_slice(&((off + 64) as u64).to_le_bytes());
        self.blob.extend_from_slice(&h);
        self.blob.extend_from_slice(data);
        self.blob.resize(self.blob.len().next_multiple_of(64), 0);
        self.blobs += 1;
        format!("BLOBFILE(path = string(\"{BLOB_PATH}\"), offset = uint64({off}))")
    }

    fn const_f16(&mut self, shape: &[usize], data: &[f32]) -> String {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&x| crate::model::f32_to_f16(x.clamp(-FP16_MAX, FP16_MAX)).to_le_bytes())
            .collect();
        let file = self.chunk(BLOB_FP16, &bytes);
        let ty = format!("tensor<fp16, [{}]>", dims(shape));
        let v = self.konst(&ty, format!("{ty}({file})"));
        if let Ok(s) = <[usize; 4]>::try_from(shape) {
            self.shapes.insert(v.clone(), s);
        }
        v
    }

    fn weight(&self, v: ValueId) -> Result<&'a Tensor, String> {
        let val = self.g.value(v);
        if val.kind != ValueKind::Weight {
            return Err(format!("'{}' must be a weight", val.name));
        }
        self.w
            .get(&val.name)
            .ok_or_else(|| format!("weight '{}' is not bound", val.name))
    }

    /// The lowered value `v`: node outputs as produced, graph inputs bound to the next input
    /// surface on first use, weights as row-form constants.
    fn mv(&mut self, v: ValueId) -> Result<Mv, String> {
        if let Some(m) = self.vals.get(&v) {
            return Ok(m.clone());
        }
        let val = self.g.value(v);
        let shape = val.shape.clone();
        let m = match val.kind {
            ValueKind::Input => {
                let (s, c) = flat2(&shape);
                let tok = s > 1 && c > 1 && self.prefers_tok(v);
                let (ch, sp) = if tok { (c, s) } else { row_cs(&shape) };
                let var = self.input(io(v, ch, sp, tok));
                Mv {
                    var,
                    shape: [1, ch, 1, sp],
                    tok,
                }
            }
            ValueKind::Weight => {
                let p = pad4(&shape)?;
                Mv {
                    var: self.operand(v, Want::Row(p))?,
                    shape: p,
                    tok: false,
                }
            }
            _ => {
                return Err(format!(
                    "'{}' is neither produced in nor an input of the fragment",
                    val.name
                ))
            }
        };
        self.vals.insert(v, m.clone());
        Ok(m)
    }

    /// Binds the next input surface and casts it to fp16.
    fn input(&mut self, b: IoBinding) -> String {
        let name = format!("in_{:02}", self.ins.len());
        let shape = [1, b.channels, 1, b.spatial];
        self.ins.push(b);
        let to16 = self.string("fp16");
        self.op(shape, format!("cast(dtype = {to16}, x = {name})"))
    }

    /// Whether a direct consumer reads `v` as a `[tokens, features]` activation.
    fn prefers_tok(&self, v: ValueId) -> bool {
        self.g.consumers(v).any(|n| {
            let slot = n.inputs.iter().position(|&x| x == v);
            match &n.op {
                Op::SwiGlu | Op::GeGlu => true,
                Op::Attention { .. } => slot.is_some_and(|s| s < 3),
                Op::Softmax { axis } => *axis == -1 || *axis as usize + 1 == self.shape(v).len(),
                Op::Linear
                | Op::RmsNorm { .. }
                | Op::LayerNorm { .. }
                | Op::Rope { .. }
                | Op::MeanPool
                | Op::Megakernel(_) => slot == Some(0),
                _ => false,
            }
        })
    }

    /// `v` in token form or as row-major `[usize; 4]`, converting (and caching) as needed.
    fn operand(&mut self, v: ValueId, want: Want) -> Result<String, String> {
        if let Some(s) = self.views.get(&(v, want)) {
            return Ok(s.clone());
        }
        let shape = self.shape(v);
        let (s, c) = flat2(&shape);
        let r = if self.g.value(v).kind == ValueKind::Weight {
            let data = self.weight(v)?.to_f32();
            if data.len() == 1 {
                self.scalar(data[0])
            } else {
                match want {
                    Want::Row(p) => self.const_f16(&p, &data),
                    Want::Tok => {
                        let t: Vec<f32> = (0..c * s).map(|i| data[(i % s) * c + i / s]).collect();
                        self.const_f16(&[1, c, 1, s], &t)
                    }
                }
            }
        } else {
            let m = self.mv(v)?;
            let trivial = s == 1 || c == 1;
            match want {
                Want::Tok if m.tok => m.var,
                Want::Tok if trivial => self.reshape(&m.var, [1, c, 1, s]),
                Want::Tok => {
                    let r = self.reshape(&m.var, [1, 1, s, c]);
                    self.transpose(&r, [0, 3, 1, 2])
                }
                Want::Row(p) => {
                    let var = if m.tok && !trivial {
                        self.transpose(&m.var, [0, 2, 3, 1])
                    } else {
                        m.var
                    };
                    self.reshape(&var, p)
                }
            }
        };
        self.views.insert((v, want), r.clone());
        Ok(r)
    }

    fn tok(&mut self, v: ValueId) -> Result<String, String> {
        self.operand(v, Want::Tok)
    }

    /// `v` in row form with its own (padded) graph shape.
    fn row(&mut self, v: ValueId) -> Result<(String, [usize; 4]), String> {
        let p = pad4(&self.shape(v))?;
        Ok((self.operand(v, Want::Row(p))?, p))
    }

    /// `v` in whatever form it already has (weights in row form); returns the form flag.
    fn like(&mut self, v: ValueId) -> Result<(String, bool), String> {
        let m = self.mv(v)?;
        let want = if m.tok { Want::Tok } else { Want::Row(m.shape) };
        Ok((self.operand(v, want)?, m.tok))
    }

    fn set(&mut self, out: ValueId, var: String, tok: bool) {
        let shape = self.sh(&var);
        self.vals.insert(out, Mv { var, shape, tok });
    }

    fn reshape(&mut self, x: &str, to: [usize; 4]) -> String {
        if self.sh(x) == to {
            return x.to_string();
        }
        let s = self.ints(&to);
        self.op(to, format!("reshape(shape = {s}, x = {x})"))
    }

    fn transpose(&mut self, x: &str, perm: [usize; 4]) -> String {
        let from = self.sh(x);
        let p = self.ints(&perm);
        self.op(
            perm.map(|i| from[i]),
            format!("transpose(perm = {p}, x = {x})"),
        )
    }

    fn un(&mut self, f: &str, x: &str) -> String {
        let s = self.sh(x);
        self.op(s, format!("{f}(x = {x})"))
    }

    fn bin(&mut self, f: &str, x: &str, y: &str) -> String {
        let (a, b) = (self.sh(x), self.sh(y));
        let s = std::array::from_fn(|i| a[i].max(b[i]));
        self.op(s, format!("{f}(x = {x}, y = {y})"))
    }

    fn bin_s(&mut self, f: &str, x: &str, k: f32) -> String {
        let k = self.scalar(k);
        self.bin(f, x, &k)
    }

    /// `reduce_*` over `axes`, keeping dims.
    fn reduce(&mut self, f: &str, x: &str, axes: &[usize]) -> String {
        let mut s = self.sh(x);
        axes.iter().for_each(|&a| s[a] = 1);
        let (ax, kd) = (self.ints(axes), self.boolean(true));
        self.op(s, format!("{f}(axes = {ax}, keep_dims = {kd}, x = {x})"))
    }

    fn slice(&mut self, x: &str, begin: [usize; 4], end: [usize; 4]) -> String {
        let (b, e) = (self.ints(&begin), self.ints(&end));
        self.op(
            std::array::from_fn(|i| end[i] - begin[i]),
            format!("slice_by_index(begin = {b}, end = {e}, x = {x})"),
        )
    }

    fn concat(&mut self, axis: usize, interleave: bool, parts: &[String]) -> String {
        let mut s = self.sh(&parts[0]);
        s[axis] = parts.iter().map(|p| self.sh(p)[axis]).sum();
        let (ax, il) = (self.int(axis), self.boolean(interleave));
        self.op(
            s,
            format!(
                "concat(axis = {ax}, interleave = {il}, values = ({}))",
                parts.join(", ")
            ),
        )
    }

    fn matmul(&mut self, x: &str, y: &str, tx: bool, ty: bool, out: [usize; 4]) -> String {
        let (a, b) = (self.boolean(tx), self.boolean(ty));
        self.op(
            out,
            format!("matmul(transpose_x = {a}, transpose_y = {b}, x = {x}, y = {y})"),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: &str,
        w: &str,
        out: [usize; 4],
        strides: [usize; 2],
        pad: [usize; 4],
        dilations: [usize; 2],
        groups: usize,
    ) -> String {
        let pt = self.string(if pad == [0; 4] { "valid" } else { "custom" });
        let (st, pd, dl, gr) = (
            self.ints(&strides),
            self.ints(&pad),
            self.ints(&dilations),
            self.int(groups),
        );
        self.op(out, format!("conv(dilations = {dl}, groups = {gr}, pad = {pd}, pad_type = {pt}, strides = {st}, weight = {w}, x = {x})"))
    }

    /// `x · Wᵀ` for a token-form `x` (`[1, in, 1, S]`) and Linear weight `w` (`[out, in]`).
    fn linear(&mut self, x: &str, w: ValueId) -> Result<String, String> {
        let (o, s) = (self.shape(w)[0], self.sh(x)[3]);
        let wv = self.linear_weight(w)?;
        Ok(self.conv(x, &wv, [1, o, 1, s], [1, 1], [0; 4], [1, 1], 1))
    }

    /// A Linear weight `[out, in]` as an fp16 `[out, in, 1, 1]` conv weight in the most compact
    /// form the probe passed.
    fn linear_weight(&mut self, v: ValueId) -> Result<String, String> {
        let t = self.weight(v)?;
        let [o, k] = self.shape(v)[..] else {
            return Err(format!(
                "Linear weight {:?} is not [out, in]",
                self.shape(v)
            ));
        };
        let packed = match t {
            Tensor::Mat(m) => m.packed().map(|p| (p.format, &p.map[p.off..])),
            _ => None,
        };
        let blockwise = self.info.int4.is_ok() && k % 32 == 0;
        Ok(match packed {
            Some((QuantFormat::GGUF_Q4_0, raw)) if blockwise => self.blockwise(raw, o, k, 4)?,
            Some((QuantFormat::GGUF_Q8_0, raw)) if blockwise => self.blockwise(raw, o, k, 8)?,
            Some((f, _))
                if self.info.int8.is_ok()
                    && !matches!(f, QuantFormat::FP32 | QuantFormat::FP16 | QuantFormat::BF16) =>
            {
                self.affine_int8(&t.to_f32(), o, k)
            }
            _ => self.const_f16(&[o, k, 1, 1], &t.to_f32()),
        })
    }

    /// `Q4_0` / `Q8_0` blocks as int4 / int8 data with their fp16 block scales. The 2-D form
    /// compiles on `h17`; the same op on `[out, in, 1, 1]` is rejected, so it is reshaped after.
    fn blockwise(&mut self, raw: &[u8], o: usize, k: usize, bits: usize) -> Result<String, String> {
        let nb = k / 32;
        let block = if bits == 4 { 18 } else { 34 };
        if raw.len() < o * nb * block {
            return Err(format!("{bits}-bit blocks of [{o}, {k}] are truncated"));
        }
        let mut data = vec![0u8; o * k * bits / 8];
        let mut scale = Vec::with_capacity(o * nb * 2);
        for (bi, blk) in raw[..o * nb * block].chunks_exact(block).enumerate() {
            scale.extend_from_slice(&blk[..2]);
            let base = bi * 32;
            for j in 0..32 {
                let e = base + j;
                if bits == 4 {
                    let q = if j < 16 {
                        blk[2 + j] & 15
                    } else {
                        blk[2 + j - 16] >> 4
                    };
                    data[e / 2] |= (q.wrapping_sub(8) & 15) << (4 * (e % 2));
                } else {
                    data[e] = blk[2 + j];
                }
            }
        }
        let (ty, dt) = if bits == 4 {
            ("int4", BLOB_INT4)
        } else {
            ("int8", BLOB_INT8)
        };
        let df = self.chunk(dt, &data);
        let sf = self.chunk(BLOB_FP16, &scale);
        let w2 = self.emit(
            "fp16",
            &[o, k],
            format!("constexpr_blockwise_shift_scale(data = tensor<{ty}, [{o}, {k}]>({df}), scale = tensor<fp16, [{o}, {nb}]>({sf}))"),
        );
        let s = self.ints(&[o, k, 1, 1]);
        Ok(self.op([o, k, 1, 1], format!("reshape(shape = {s}, x = {w2})")))
    }

    /// Per-output-channel int8 (`absmax / 127` of each decoded row).
    fn affine_int8(&mut self, w: &[f32], o: usize, k: usize) -> String {
        let mut q = vec![0u8; o * k];
        let mut scales = Vec::with_capacity(o);
        for (r, row) in w.chunks_exact(k).enumerate() {
            let s = row.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30) / 127.0;
            for (j, x) in row.iter().enumerate() {
                q[r * k + j] = (x / s).round().clamp(-127.0, 127.0) as i8 as u8;
            }
            scales.push(lit(s));
        }
        let file = self.chunk(BLOB_INT8, &q);
        let v = self.fresh();
        writeln!(
            self.body,
            "        tensor<fp16, [{o}, {k}, 1, 1]> {v} = constexpr_affine_dequantize()[axis = int32(0), name = string(\"{v}\"), quantized_data = tensor<int8, [{o}, {k}, 1, 1]>({file}), scale = tensor<fp16, [{o}]>([{}]), zero_point = int8(0)];",
            scales.join(", ")
        )
        .unwrap();
        self.shapes.insert(v.clone(), [o, k, 1, 1]);
        v
    }

    /// `x · rsqrt(mean(x²) + eps)` over the channel axis of a token-form `x`; squares are taken
    /// after scaling by `1/sqrt(d)` so fp16 holds them.
    fn rms(&mut self, x: &str, eps: f32) -> String {
        let d = self.sh(x)[1];
        let xs = self.bin_s("mul", x, 1.0 / (d as f32).sqrt());
        let sq = self.bin("mul", &xs, &xs);
        let ss = self.reduce("reduce_sum", &sq, &[1]);
        let e = self.bin_s("add", &ss, eps);
        let r = self.bin_s("pow", &e, -0.5);
        self.bin("mul", x, &r)
    }

    fn gelu_tanh(&mut self, x: &str) -> String {
        let x2 = self.bin("mul", x, x);
        let c = self.bin_s("mul", &x2, 0.044715);
        let x3 = self.bin("mul", &c, x);
        let inner = self.bin("add", x, &x3);
        let a = self.bin_s("mul", &inner, 0.797_884_6);
        let t = self.un("tanh", &a);
        let t1 = self.bin_s("add", &t, 1.0);
        let h = self.bin_s("mul", x, 0.5);
        self.bin("mul", &h, &t1)
    }

    fn glu(&mut self, gate: &str, up: &str, geglu: bool) -> String {
        let a = if geglu {
            self.gelu_tanh(gate)
        } else {
            self.un("silu", gate)
        };
        self.bin("mul", &a, up)
    }

    fn node(&mut self, i: usize) -> Result<(), String> {
        let g = self.g;
        let node = &g.nodes[i];
        let ins = node.inputs.clone();
        let out = node.outputs[0];
        let in_shape = ins.first().map(|&v| self.shape(v)).unwrap_or_default();
        match &node.op {
            Op::Linear => {
                let x = self.tok(ins[0])?;
                let mut y = self.linear(&x, ins[1])?;
                if let Some(&b) = ins.get(2) {
                    let b = self.tok(b)?;
                    y = self.bin("add", &y, &b);
                }
                self.set(out, y, true);
            }
            Op::MatMul {
                transpose_a,
                transpose_b,
            } => {
                let (a, _) = self.row(ins[0])?;
                let (b, _) = self.row(ins[1])?;
                let o = pad4(&self.shape(out))?;
                let y = self.matmul(&a, &b, *transpose_a, *transpose_b, o);
                self.set(out, y, false);
            }
            Op::RmsNorm { eps } => {
                let x = self.tok(ins[0])?;
                let mut y = self.rms(&x, *eps);
                if let Some(&g) = ins.get(1) {
                    let g = self.tok(g)?;
                    y = self.bin("mul", &y, &g);
                }
                self.set(out, y, true);
            }
            Op::LayerNorm { eps } => {
                let x = self.tok(ins[0])?;
                let d = self.sh(&x)[1];
                let m = self.reduce("reduce_mean", &x, &[1]);
                let xc = self.bin("sub", &x, &m);
                let xs = self.bin_s("mul", &xc, 1.0 / (d as f32).sqrt());
                let sq = self.bin("mul", &xs, &xs);
                let var = self.reduce("reduce_sum", &sq, &[1]);
                let e = self.bin_s("add", &var, *eps);
                let r = self.bin_s("pow", &e, -0.5);
                let mut y = self.bin("mul", &xc, &r);
                let g = self.tok(ins[1])?;
                y = self.bin("mul", &y, &g);
                if let Some(&b) = ins.get(2) {
                    let b = self.tok(b)?;
                    y = self.bin("add", &y, &b);
                }
                self.set(out, y, true);
            }
            Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
                let c: usize = self.shape(ins[1]).iter().product();
                let (groups, eps) = match node.op {
                    Op::GroupNorm { groups, eps } => (groups, eps),
                    Op::InstanceNorm { eps } => (c, eps),
                    _ => unreachable!(),
                };
                let (n, hw) = channel_layout(&in_shape, c)?;
                if n != 1 || groups == 0 || !c.is_multiple_of(groups) {
                    return Err(format!("{in_shape:?} in {groups} groups is not one sample"));
                }
                let x = self.operand(ins[0], Want::Row([1, groups, c / groups, hw]))?;
                let m = self.reduce("reduce_mean", &x, &[2, 3]);
                let xc = self.bin("sub", &x, &m);
                let xs = self.bin_s("mul", &xc, 1.0 / ((c / groups * hw) as f32).sqrt());
                let sq = self.bin("mul", &xs, &xs);
                let var = self.reduce("reduce_sum", &sq, &[2, 3]);
                let e = self.bin_s("add", &var, eps);
                let r = self.bin_s("pow", &e, -0.5);
                let y = self.bin("mul", &xc, &r);
                let y = self.reshape(&y, [1, c, 1, hw]);
                let g = self.operand(ins[1], Want::Row([1, c, 1, 1]))?;
                let b = self.operand(ins[2], Want::Row([1, c, 1, 1]))?;
                let y = self.bin("mul", &y, &g);
                let y = self.bin("add", &y, &b);
                self.set(out, y, false);
            }
            Op::Rope {
                head_dim,
                rotary_dim,
                variant,
                ..
            } => {
                let (d, r) = (*head_dim, *rotary_dim);
                let (n, width) = flat2(&in_shape);
                let heads = width / d.max(1);
                if d == 0 || r > d || r % 2 != 0 || heads * d != width {
                    return Err(format!("rotary {r} of head {d} does not fit {in_shape:?}"));
                }
                if self.g.value(ins[1]).kind != ValueKind::Input {
                    return Err("Rope positions must be a fragment input".into());
                }
                self.weight(ins[2])?;
                let x = self.tok(ins[0])?;
                let x4 = self.reshape(&x, [heads, d, 1, n]);
                let t = self.input(IoBinding {
                    source: IoSource::RopeTable(i),
                    channels: r,
                    spatial: n,
                    transposed: false,
                });
                let h = r / 2;
                let cos = self.slice(&t, [0; 4], [1, h, 1, n]);
                let sin = self.slice(&t, [0, h, 0, 0], [1, r, 1, n]);
                let half = matches!(
                    variant,
                    RopeVariant::HalfSplit
                        | RopeVariant::MRope {
                            pairs: PairLayout::Half,
                            ..
                        }
                );
                // Each pair (a, b) rotates to (a·cos − b·sin, a·sin + b·cos).
                let (a, b) = if half {
                    (
                        self.slice(&x4, [0; 4], [heads, h, 1, n]),
                        self.slice(&x4, [0, h, 0, 0], [heads, r, 1, n]),
                    )
                } else {
                    let xr = if r < d {
                        self.slice(&x4, [0; 4], [heads, r, 1, n])
                    } else {
                        x4.clone()
                    };
                    let p = self.reshape(&xr, [heads, h, 2, n]);
                    (
                        self.slice(&p, [0; 4], [heads, h, 1, n]),
                        self.slice(&p, [0, 0, 1, 0], [heads, h, 2, n]),
                    )
                };
                let (ac, bs) = (self.bin("mul", &a, &cos), self.bin("mul", &b, &sin));
                let (as_, bc) = (self.bin("mul", &a, &sin), self.bin("mul", &b, &cos));
                let ra = self.bin("sub", &ac, &bs);
                let rb = self.bin("add", &as_, &bc);
                let mut y = self.concat(1, !half, &[ra, rb]);
                if r < d {
                    let rest = self.slice(&x4, [0, r, 0, 0], [heads, d, 1, n]);
                    y = self.concat(1, false, &[y, rest]);
                }
                let y = self.reshape(&y, [1, width, 1, n]);
                self.set(out, y, true);
            }
            Op::Attention {
                heads,
                kv_heads,
                causal,
                window,
                soft_cap,
                scale,
            } => {
                let (h, kv) = (*heads, *kv_heads);
                let (nq, width) = flat2(&in_shape);
                let d = width / h.max(1);
                let (nk, kw) = flat2(&self.shape(ins[1]));
                if kv == 0 || h % kv != 0 || d * h != width || kw != kv * d {
                    return Err(format!(
                        "{h} heads over {kv} kv heads do not fit {in_shape:?}"
                    ));
                }
                let g = h / kv;
                let q = self.tok(ins[0])?;
                let q = self.reshape(&q, [1, h, d, nq]);
                let q = self.transpose(&q, [0, 1, 3, 2]);
                let q = self.reshape(&q, [1, kv, g * nq, d]);
                let k = self.tok(ins[1])?;
                let k = self.reshape(&k, [1, kv, d, nk]);
                let s = self.matmul(&q, &k, false, false, [1, kv, g * nq, nk]);
                let s = self.reshape(&s, [1, h, nq, nk]);
                let sc = scale.unwrap_or(1.0 / (d as f32).sqrt());
                let mut s = match soft_cap {
                    Some(cap) => {
                        let a = self.bin_s("mul", &s, sc / cap);
                        let t = self.un("tanh", &a);
                        self.bin_s("mul", &t, *cap)
                    }
                    None => self.bin_s("mul", &s, sc),
                };
                if *causal || window.is_some() {
                    let mut m = vec![0.0f32; nq * nk];
                    for p in 0..nq {
                        let (lo, hi) = attention_key_range(p, nk, *causal, *window);
                        for (kk, x) in m[p * nk..][..nk].iter_mut().enumerate() {
                            if kk < lo || kk >= hi {
                                *x = -FP16_MAX;
                            }
                        }
                    }
                    let mask = self.const_f16(&[1, 1, nq, nk], &m);
                    s = self.bin("add", &s, &mask);
                }
                if let Some(&b) = ins.get(3) {
                    let bl: usize = self.shape(b).iter().product();
                    let bs = if bl == nk {
                        [1, 1, 1, nk]
                    } else if bl == nq * nk {
                        [1, 1, nq, nk]
                    } else {
                        [1, h, nq, nk]
                    };
                    let b = self.operand(b, Want::Row(bs))?;
                    s = self.bin("add", &s, &b);
                }
                let ax = self.int(3);
                let p = self.op([1, h, nq, nk], format!("softmax(axis = {ax}, x = {s})"));
                let p = self.reshape(&p, [1, kv, g * nq, nk]);
                let v = self.tok(ins[2])?;
                let v = self.reshape(&v, [1, kv, d, nk]);
                let o = self.matmul(&p, &v, false, true, [1, kv, g * nq, d]);
                let o = self.reshape(&o, [1, h, nq, d]);
                let o = self.transpose(&o, [0, 1, 3, 2]);
                let o = self.reshape(&o, [1, h * d, 1, nq]);
                self.set(out, o, true);
            }
            Op::SwiGlu | Op::GeGlu => {
                let (gate, tok) = self.like(ins[0])?;
                let want = if tok {
                    Want::Tok
                } else {
                    Want::Row(self.sh(&gate))
                };
                let up = self.operand(ins[1], want)?;
                let y = self.glu(&gate, &up, matches!(node.op, Op::GeGlu));
                self.set(out, y, tok);
            }
            Op::Unary { kind } => {
                let (x, tok) = self.like(ins[0])?;
                let y = match kind {
                    UnaryKind::Exp => self.un("exp", &x),
                    UnaryKind::Sigmoid => self.un("sigmoid", &x),
                    UnaryKind::Tanh => self.un("tanh", &x),
                    UnaryKind::Softplus => self.un("softplus", &x),
                    UnaryKind::Neg => self.bin_s("mul", &x, -1.0),
                    UnaryKind::Sqrt => self.un("sqrt", &x),
                    UnaryKind::Rsqrt => self.bin_s("pow", &x, -0.5),
                    UnaryKind::Abs => self.un("abs", &x),
                    UnaryKind::Relu => self.un("relu", &x),
                    UnaryKind::Relu2 => {
                        let r = self.un("relu", &x);
                        self.bin("mul", &r, &r)
                    }
                    UnaryKind::Silu => self.un("silu", &x),
                    UnaryKind::GeluErf => {
                        let a = self.bin_s("mul", &x, std::f32::consts::FRAC_1_SQRT_2);
                        let e = self.un("erf", &a);
                        let e1 = self.bin_s("add", &e, 1.0);
                        let h = self.bin_s("mul", &x, 0.5);
                        self.bin("mul", &h, &e1)
                    }
                    UnaryKind::GeluTanh => self.gelu_tanh(&x),
                    UnaryKind::Scale(s) => self.bin_s("mul", &x, *s),
                    UnaryKind::Clamp(lo, hi) => {
                        let (a, b) = (self.scalar(*lo), self.scalar(*hi));
                        let s = self.sh(&x);
                        self.op(s, format!("clip(alpha = {a}, beta = {b}, x = {x})"))
                    }
                    UnaryKind::SoftCap(cap) => {
                        let a = self.bin_s("mul", &x, 1.0 / cap);
                        let t = self.un("tanh", &a);
                        self.bin_s("mul", &t, *cap)
                    }
                    UnaryKind::Ln | UnaryKind::RoundEven | UnaryKind::Sign => {
                        return Err(format!("Unary {kind:?} does not compile for the ANE"))
                    }
                };
                self.set(out, y, tok);
            }
            Op::Binary { kind } => {
                let f = match kind {
                    BinaryKind::Add => "add",
                    BinaryKind::Sub => "sub",
                    BinaryKind::Mul => "mul",
                    BinaryKind::Div => "real_div",
                    BinaryKind::Max => "maximum",
                    BinaryKind::Min => "minimum",
                };
                let (a, b) = (ins[0], ins[1]);
                let (sa, sb) = (self.shape(a), self.shape(b));
                let lead = if self.g.value(a).kind == ValueKind::Weight {
                    b
                } else {
                    a
                };
                let other = if lead == a { b } else { a };
                let (so, sl) = (self.shape(other), self.shape(lead));
                // Token form when the other operand broadcasts over `[S, C]` without leading dims.
                let ((s, c), (os, oc)) = (flat2(&sl), flat2(&so));
                let tok = self.mv(lead)?.tok
                    && (so.len() <= 1 || (sl.len() <= 2 && so.len() <= 2))
                    && (os == s || os == 1)
                    && (oc == c || oc == 1);
                let (xa, xb) = if tok {
                    (self.tok(a)?, self.tok(b)?)
                } else {
                    (
                        self.operand(a, Want::Row(pad4(&sa)?))?,
                        self.operand(b, Want::Row(pad4(&sb)?))?,
                    )
                };
                let y = self.bin(f, &xa, &xb);
                self.set(out, y, tok);
            }
            Op::Softmax { axis } => {
                let r = in_shape.len();
                let ax = if *axis < 0 { *axis + r as isize } else { *axis } as usize;
                let (x, axm, tok) = if ax + 1 == r && self.mv(ins[0])?.tok {
                    (self.tok(ins[0])?, 1, true)
                } else {
                    (self.row(ins[0])?.0, ax + 4 - r, false)
                };
                let a = self.int(axm);
                let s = self.sh(&x);
                let y = self.op(s, format!("softmax(axis = {a}, x = {x})"));
                self.set(out, y, tok);
            }
            Op::Reduce { kind, axes, .. } => {
                let r = in_shape.len();
                let (x, _) = self.row(ins[0])?;
                let axes: Vec<usize> = axes
                    .iter()
                    .map(|&a| (if a < 0 { a + r as isize } else { a }) as usize + 4 - r)
                    .collect();
                let f = match kind {
                    ReduceKind::Sum => "reduce_sum",
                    ReduceKind::Mean => "reduce_mean",
                    ReduceKind::Max => "reduce_max",
                    ReduceKind::Min => "reduce_min",
                };
                let y = self.reduce(f, &x, &axes);
                self.set(out, y, false);
            }
            Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
                let m = self.mv(ins[0])?;
                let (s0, c0) = flat2(&in_shape);
                if !m.tok || flat2(&self.shape(out)).1 == c0 {
                    self.vals.insert(out, m);
                } else {
                    let x = self.operand(ins[0], Want::Row([1, 1, s0, c0]))?;
                    self.set(out, x, false);
                }
            }
            Op::Transpose { perm } => {
                let m = self.mv(ins[0])?;
                if in_shape.len() == 2 && perm[..] == [1, 0] {
                    // A token-form `[S, C]` is row-major `[C, S]` and vice versa.
                    let (s, c) = (in_shape[0], in_shape[1]);
                    let x = if m.tok {
                        m.var
                    } else {
                        self.operand(ins[0], Want::Row([1, s, 1, c]))?
                    };
                    self.set(out, x, !m.tok);
                } else {
                    let (x, _) = self.row(ins[0])?;
                    let off = 4 - in_shape.len();
                    let mut p = [0, 1, 2, 3];
                    for (j, &q) in perm.iter().enumerate() {
                        p[off + j] = q + off;
                    }
                    let y = self.transpose(&x, p);
                    self.set(out, y, false);
                }
            }
            Op::Slice { axis, start, end } => {
                let r = in_shape.len();
                let ax = if *axis < 0 { *axis + r as isize } else { *axis } as usize;
                let dim = in_shape[ax] as i64;
                let at = |v: i64| (if v < 0 { dim + v } else { v }).clamp(0, dim) as usize;
                let (s0, e0) = (at(*start), at(*end).max(at(*start)));
                if ax + 1 == r && self.mv(ins[0])?.tok {
                    let x = self.tok(ins[0])?;
                    let n = self.sh(&x)[3];
                    let y = self.slice(&x, [0, s0, 0, 0], [1, e0, 1, n]);
                    self.set(out, y, true);
                } else {
                    let (x, p) = self.row(ins[0])?;
                    let (mut b, mut e) = ([0; 4], p);
                    b[ax + 4 - r] = s0;
                    e[ax + 4 - r] = e0;
                    let y = self.slice(&x, b, e);
                    self.set(out, y, false);
                }
            }
            Op::Tile { reps } => {
                let (x, p) = self.row(ins[0])?;
                let rp = pad4(reps)?;
                let o = std::array::from_fn(|i| p[i] * rp[i]);
                let r = self.ints(&rp);
                let y = self.op(o, format!("tile(reps = {r}, x = {x})"));
                self.set(out, y, false);
            }
            Op::Pad { padding, value } => {
                let (x, p) = self.row(ins[0])?;
                let k = padding.len() / 2;
                let mut pads = [0usize; 8];
                let mut o = p;
                for j in 0..k {
                    let d = 4 - k + j;
                    pads[2 * d] = padding[2 * j];
                    pads[2 * d + 1] = padding[2 * j + 1];
                    o[d] += padding[2 * j] + padding[2 * j + 1];
                }
                let (pd, md, cv) = (
                    self.ints(&pads),
                    self.string("constant"),
                    self.scalar(*value),
                );
                let y = self.op(
                    o,
                    format!("pad(constant_val = {cv}, mode = {md}, pad = {pd}, x = {x})"),
                );
                self.set(out, y, false);
            }
            Op::Conv2d {
                stride,
                padding: p,
                dilation,
                groups,
            } => {
                let (ci, h, w) = match in_shape[..] {
                    [ci, h, w] | [1, ci, h, w] => (ci, h, w),
                    _ => return Err(format!("Conv2d input {in_shape:?} is not [c, h, w]")),
                };
                let ws = self.shape(ins[1]);
                let os = self.shape(out);
                let (co, oh, ow) = (ws[0], os[os.len() - 2], os[os.len() - 1]);
                let x = self.operand(ins[0], Want::Row([1, ci, h, w]))?;
                let wt = self.weight(ins[1])?.to_f32();
                let wv = self.const_f16(&ws, &wt);
                let mut y = self.conv(
                    &x,
                    &wv,
                    [1, co, oh, ow],
                    *stride,
                    [p[0], p[2], p[1], p[3]],
                    *dilation,
                    *groups,
                );
                if let Some(&b) = ins.get(2) {
                    let b = self.operand(b, Want::Row([1, co, 1, 1]))?;
                    y = self.bin("add", &y, &b);
                }
                self.set(out, y, false);
            }
            Op::Conv1d {
                stride,
                padding: p,
                dilation,
                groups,
            } => {
                let (ci, len) = match in_shape[..] {
                    [ci, l] | [1, ci, l] => (ci, l),
                    _ => return Err(format!("Conv1d input {in_shape:?} is not [c, l]")),
                };
                let ws = self.shape(ins[1]);
                let lo = *self.shape(out).last().unwrap_or(&0);
                let x = self.operand(ins[0], Want::Row([1, ci, 1, len]))?;
                let wt = self.weight(ins[1])?.to_f32();
                let wv = self.const_f16(&[ws[0], ws[1], 1, ws[2]], &wt);
                let mut y = self.conv(
                    &x,
                    &wv,
                    [1, ws[0], 1, lo],
                    [1, *stride],
                    [0, 0, p[0], p[1]],
                    [1, *dilation],
                    *groups,
                );
                if let Some(&b) = ins.get(2) {
                    let b = self.operand(b, Want::Row([1, ws[0], 1, 1]))?;
                    y = self.bin("add", &y, &b);
                }
                self.set(out, y, false);
            }
            Op::AvgPool2d { kernel, stride } => {
                let r = in_shape.len();
                let lead = in_shape[..r - 2].iter().product();
                let x = self.operand(
                    ins[0],
                    Want::Row([1, lead, in_shape[r - 2], in_shape[r - 1]]),
                )?;
                let os = self.shape(out);
                let (k, s, pd) = (self.ints(kernel), self.ints(stride), self.ints(&[0; 4]));
                let (pt, ex, cm) = (
                    self.string("valid"),
                    self.boolean(false),
                    self.boolean(false),
                );
                let y = self.op(
                    [1, lead, os[r - 2], os[r - 1]],
                    format!("avg_pool(ceil_mode = {cm}, exclude_padding_from_average = {ex}, kernel_sizes = {k}, pad = {pd}, pad_type = {pt}, strides = {s}, x = {x})"),
                );
                self.set(out, y, false);
            }
            Op::Upsample { scale, mode } => {
                let os = self.shape(out);
                if upsample_nsp(&in_shape, &os, *scale)? != 2 {
                    return Err("only 2-D upsampling lowers to the ANE".into());
                }
                let r = in_shape.len();
                let lead = in_shape[..r - 2].iter().product();
                let x = self.operand(
                    ins[0],
                    Want::Row([1, lead, in_shape[r - 2], in_shape[r - 1]]),
                )?;
                let sf = self.konst("fp32", format!("fp32({scale})"));
                let o = [1, lead, os[r - 2], os[r - 1]];
                let y = match mode {
                    UpsampleMode::Nearest => self.op(o, format!("upsample_nearest_neighbor(scale_factor_height = {sf}, scale_factor_width = {sf}, x = {x})")),
                    UpsampleMode::Bilinear => {
                        let ac = self.boolean(false);
                        self.op(o, format!("upsample_bilinear(align_corners = {ac}, scale_factor_height = {sf}, scale_factor_width = {sf}, x = {x})"))
                    }
                    UpsampleMode::Bicubic => return Err("bicubic upsampling does not lower to the ANE".into()),
                };
                self.set(out, y, false);
            }
            Op::MeanPool => {
                let (n, d) = flat2(&in_shape);
                let x = self.tok(ins[0])?;
                let m = self.operand(ins[1], Want::Row([1, 1, 1, n]))?;
                let m = self.bin_s("minimum", &m, 1.0);
                let xm = self.bin("mul", &x, &m);
                // mean(x·m) / max(mean(m), 1/n) = sum(x·m) / max(count, 1), without fp16 overflow.
                let a = self.reduce("reduce_mean", &xm, &[3]);
                let f = self.reduce("reduce_mean", &m, &[3]);
                let f = self.bin_s("maximum", &f, 1.0 / n as f32);
                let y = self.bin("real_div", &a, &f);
                debug_assert_eq!(self.sh(&y), [1, d, 1, 1]);
                self.set(out, y, false);
            }
            Op::L2Normalize => {
                let (x, tok) = self.like(ins[0])?;
                let all = [0, 1, 2, 3];
                // Scale by 1 / max|x| first so the squares stay in fp16 range.
                let ab = self.un("abs", &x);
                let mx = self.reduce("reduce_max", &ab, &all);
                let mx = self.bin_s("maximum", &mx, 1e-4);
                let inv = self.bin_s("pow", &mx, -1.0);
                let y = self.bin("mul", &x, &inv);
                let sq = self.bin("mul", &y, &y);
                let ss = self.reduce("reduce_sum", &sq, &all);
                let e = self.bin_s("add", &ss, 1e-7);
                let r = self.bin_s("pow", &e, -0.5);
                let y = self.bin("mul", &y, &r);
                self.set(out, y, tok);
            }
            Op::Megakernel(desc) => {
                let MegakernelKind::Mlp {
                    eps,
                    is_geglu,
                    has_post_norm,
                    ..
                } = desc.kind;
                let x = self.tok(ins[0])?;
                let nw = self.tok(ins[1])?;
                let n = self.rms(&x, eps);
                let n = self.bin("mul", &n, &nw);
                let gate = self.linear(&n, ins[2])?;
                let up = self.linear(&n, ins[3])?;
                let a = self.glu(&gate, &up, is_geglu);
                let mut dn = self.linear(&a, ins[4])?;
                if has_post_norm {
                    let pw = self.tok(ins[5])?;
                    dn = self.rms(&dn, eps);
                    dn = self.bin("mul", &dn, &pw);
                }
                let y = self.bin("add", &x, &dn);
                self.set(out, y, true);
            }
            other => return Err(format!("{} does not lower to the ANE", other.name())),
        }
        Ok(())
    }
}
