//! Gibbs graph IR: one model description shared by every executor (CPU and Metal).
//!
//! A [`Graph`] is pure structure in SSA form. Every value is a graph input, a named weight, or the
//! output of a node, and nodes are appended in topological order (a node can only reference values
//! that already exist). Weights are ordinary operands, never op fields, and their data lives
//! outside the graph in [`Weights`], so one graph can be fingerprinted or run.

pub mod avenues;
pub mod codec;
pub mod codegen;
pub mod cpu;
pub mod cuda;
pub mod egraph;
pub mod extropic;
pub mod intel_xpu;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod passes;
pub mod pattern;
pub mod placement;
pub mod plan;
pub mod program;
pub mod rocm;
pub mod shape;
#[cfg(test)]
pub(crate) mod synth_decoder;
pub mod tune;
pub mod xla;
use crate::hardware::{Device, QuantFormat};
use std::collections::HashMap;

/// Whether `item` is in `GIBBS_DISABLE` (comma list, read once): per-item kill switch next to
/// `metal::fusion_enabled` (here because the e-graph builds without Metal). Items: `deltanet` and
/// `pipeline` (greedy decode streams, `program::GraphProgram::stream_next`). Tests set
/// [`TEST_DISABLE`] per thread.
pub(crate) fn disabled(item: &str) -> bool {
    #[cfg(test)]
    let list = TEST_DISABLE.with(|d| d.get());
    #[cfg(not(test))]
    let list = {
        static D: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| std::env::var("GIBBS_DISABLE").unwrap_or_default());
        D.as_str()
    };
    list.split(',').any(|s| s.trim() == item)
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_DISABLE: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Element type of a value. Quantized weights carry their block format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
    Quant(QuantFormat),
}

/// Physical layout of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Row-major in `shape` order; CPU images `[c, h*w]` are NCHW.
    #[default]
    RowMajor,
    /// Channels-last images (Metal `convolution2d`).
    Nhwc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueKind {
    Input,
    /// Bound by name to [`Weights`] at execution.
    Weight,
    /// Persistent per-sequence buffer (e.g. KV cache slots, Gated DeltaNet recurrent state, causal conv state).
    /// `recurrent`: the buffer is a running summary of every position so far (DeltaNet state, conv
    /// window), so it cannot rewind by position; a KV cache (row per position) can.
    State {
        name: String,
        recurrent: bool,
    },
    Node(NodeId),
}
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: String,
    /// Empty when unknown (e.g. per-node ONNX lowering).
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub layout: Layout,
    pub kind: ValueKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PairLayout {
    Adjacent,
    Half,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RopeVariant {
    /// Rotates adjacent pairs `(2i, 2i+1)`.
    Interleaved,
    /// Rotates `(i, i + rotary_dim/2)` (half-split layout).
    HalfSplit,
    /// Multimodal RoPE with temporal, height, width sections.
    MRope {
        sections: [u32; 3],
        interleaved: bool,
        pairs: PairLayout,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
    Min,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UnaryKind {
    Exp,
    Sigmoid,
    Tanh,
    Softplus,
    Neg,
    Sqrt,
    Rsqrt,
    Abs,
    Relu,
    Relu2,
    Silu,
    GeluErf,
    GeluTanh,
    Scale(f32),
    Clamp(f32, f32),
    SoftCap(f32),
    /// Natural logarithm.
    Ln,
    /// Round half to even.
    RoundEven,
    /// `1` for positive, `-1` for negative, `0` otherwise (zero, NaN).
    Sign,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UpsampleMode {
    Nearest,
    Bilinear,
    Bicubic,
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MegakernelKind {
    /// Fused MLP: RmsNorm -> Linear(Gate) & Linear(Up) (or Linear(GateUp)) -> SwiGLU / GeGLU -> Linear(Down) -> Add
    Mlp {
        hidden_dim: usize,
        intermediate_dim: usize,
        eps: f32,
        is_geglu: bool,
        has_post_norm: bool,
    },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MegakernelDescriptor {
    pub kind: MegakernelKind,
    pub name: String,
}

/// Operations. Operand order is the node's `inputs` order, documented per variant. Serde uses the
/// externally tagged form (`"Linear"`, `{"RmsNorm": {"eps": 1e-6}}`); optional fields may be omitted.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Op {
    /// `[table [rows, d], ids [n] (U32)]` → `[n, d]`.
    Embedding,
    /// `[x [n, in], w [out, in], bias? [out]]` → `[n, out]`.
    Linear,
    /// `[a, b]` → `op(a)·op(b)` for activation × activation products.
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
    },
    /// `[x [n, d], gamma [d], beta? [d]]`, over the last dim.
    LayerNorm {
        eps: f32,
    },
    /// `[x [n, d], gamma [d]]`.
    RmsNorm {
        eps: f32,
    },
    /// `[x [c, hw], gamma [c], beta [c]]`.
    GroupNorm {
        groups: usize,
        eps: f32,
    },
    /// `[x, gamma, beta]`, per instance.
    InstanceNorm {
        eps: f32,
    },
    /// `[x [n, heads*head_dim], positions [n] (U32)]`; rotates the first `rotary_dim` of each head.
    /// `[x [n, heads*head_dim], positions, inv_freq [rotary_dim/2] F32 weight]`.
    Rope {
        head_dim: usize,
        rotary_dim: usize,
        variant: RopeVariant,
        mscale: f32,
    },
    /// `[q [nq, heads*head_dim], k [nk, kv_heads*head_dim], v [nk, kv_heads*head_dim], bias?]` →
    /// `[nq, heads*head_dim]`, with `head_dim` taken from `q`'s shape; query head `h` reads kv head
    /// `h / (heads / kv_heads)`. Cross-attention is this op with `k`/`v` from another sequence.
    /// Scores are `q·k * scale` (`scale: None` = `1 / sqrt(head_dim)`), then `soft_cap * tanh(s / soft_cap)`, then the optional
    /// additive `bias` (`[heads, nq, nk]`, `[nq, nk]` or `[nk]`; `-inf` masks, e.g. padding).
    /// `causal`: query `p` sees keys `k <= p` (top-left aligned when `nq != nk`). With `window = Some(w)`:
    /// causal sees `p - w < k <= p` (w keys), non-causal sees `|p - k| <= w / 2`. A query with no
    /// visible key outputs zeros.
    Attention {
        heads: usize,
        kv_heads: usize,
        causal: bool,
        #[serde(default)]
        window: Option<usize>,
        #[serde(default)]
        soft_cap: Option<f32>,
        #[serde(default)]
        scale: Option<f32>,
    },
    /// `[gate, up]` → `silu(gate) * up`.
    SwiGlu,
    /// `[gate, up]` → `gelu_pytorch_tanh(gate) * up`.
    GeGlu,
    /// `[x]` element-wise unary transformation.
    Unary {
        kind: UnaryKind,
    },
    /// `[a, b]` element-wise binary transformation with broadcasting.
    Binary {
        kind: BinaryKind,
    },
    /// `[q, k_new, v_new, k_state, v_state, positions, visible_end?]` evaluating causal attention and appending
    /// to state. The optional `visible_end` (`[m]` U32, each >= its row's position + 1) lets row `i` see keys
    /// `< visible_end[i]`; the sliding window still starts at the row's own position.
    CachedAttention {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        scale: f32,
        causal: bool,
        #[serde(default)]
        window: Option<usize>,
        #[serde(default)]
        soft_cap: Option<f32>,
        write: bool,
    },
    /// `[x, weight, bias?, conv_state]` 1D causal convolution updating state.
    CausalConv1dCached {
        kernel: usize,
        dilation: usize,
    },
    /// `[q, k, v, beta, log_decay, ssm_state]` Gated DeltaNet recurrence rule.
    DeltaRule {
        key_heads: usize,
        val_heads: usize,
        key_dim: usize,
        val_dim: usize,
    },
    Gather {
        axis: isize,
    },
    Scatter {
        axis: isize,
    },
    Pad {
        padding: Vec<usize>,
        value: f32,
    },
    AvgPool2d {
        kernel: [usize; 2],
        stride: [usize; 2],
    },
    Upsample {
        scale: usize,
        mode: UpsampleMode,
    },
    Conv3d {
        stride: [usize; 3],
        padding: [usize; 6],
        dilation: [usize; 3],
        groups: usize,
    },
    /// `[x [rows, n] f32]` → `[rows, k]` U32 indices of the `k` largest values per row, sorted by
    /// value descending; equal values keep the lower index first.
    TopK {
        k: usize,
    },
    /// `[x [rows, n] f32, idx [rows, k] U32]` → `[rows, k]` f32, `out[r, j] = x[r, idx[r, j]]`; a
    /// single-row `x` is shared by every row of `idx`.
    GatherCols,
    /// Expert-indexed linear `[x, bank, ids]`: `x` is `[m, in]` (every slot of a row reads the same
    /// row) or `[m * top_k, in]` (row `r` is slot `r`); `bank` is the `[experts * out, in]` stack of
    /// every expert's `[out, in]` weight (a GGUF `*_exps` tensor viewed as rows); `ids` is `[m, top_k]`
    /// U32 expert numbers. Output `[m * top_k, out]`: slot `s` is `x_s · bank[ids[s] * out ..][..out]ᵀ`.
    MoeLinear {
        experts: usize,
        top_k: usize,
    },
    Softmax {
        axis: isize,
    },
    /// `[table [buckets, heads], q_pos [n] U32, k_pos [m] U32]` → `[heads, n, m]`, the relative-position bias added
    /// to attention scores: `table[bucket(k_pos - q_pos)]` with HF `_relative_position_bucket`.
    RelPosBias {
        buckets: usize,
        max_distance: usize,
        bidirectional: bool,
        heads: usize,
    },
    /// `[state, parent [batch] U32]` permuting batch rows in-place.
    StateReorder,
    /// `[state [cap, w], rows [m, w], pos0 [1] U32]` → `[1]` U32 holding `pos0 + m`. Writes
    /// `state[pos0..pos0 + m] = rows` in place; `state` must be a graph `State` value, so the rows
    /// persist into later runs. The write is a side effect (ops are single-output): the returned token
    /// is what orders it, so a node that must observe the write depends on the token, and dead-node
    /// elimination keeps every `StateWrite`. Within one graph, nodes after it reading the same state
    /// see the new rows. Writing past `cap` is an error on both executors (Metal needs `pos0` to be a
    /// graph input and checks it on the host).
    StateWrite,
    /// `[score [n, B] F32, positions [n] U32]` → `[n, K]` U32 key ids with
    /// `K = top_blocks * ratio + ratio - 1` (sparse block attention key selection). Row `t` sees
    /// `visible = positions[t] + 1` keys: `blocks = visible / ratio` complete blocks of `ratio` keys
    /// (`blocks <= B` required; `score[t, b]` for `b >= blocks` is ignored) and an incomplete tail.
    /// With `blocks <= top_blocks` the ids are every visible key `0..visible`; otherwise the
    /// `top_blocks` best complete blocks (score descending in `f32::total_cmp` order, equal scores
    /// lower block first), in ascending block order, each expanded to its `ratio` keys, then the tail
    /// `blocks * ratio..visible`. Unused trailing slots hold 0. Ops are single-output, so the matching
    /// mask is the separate [`Op::QsaBias`].
    QsaSelect {
        ratio: usize,
        top_blocks: usize,
    },
    /// [`Op::QsaSelect`]'s operands → `[n, K]` F32 additive bias for the gathered keys: 0 where
    /// `QsaSelect`'s slot holds a selected key, `-inf` in unused slots.
    QsaBias {
        ratio: usize,
        top_blocks: usize,
    },
    /// `[x [c_in, h, w], w [c_out, c_in/groups, kh, kw], bias? [c_out]]`; padding is
    /// `[top, left, bottom, right]`.
    Conv2d {
        stride: [usize; 2],
        padding: [usize; 4],
        dilation: [usize; 2],
        groups: usize,
    },
    /// `[x [c_in, l], w [c_out, c_in/groups, k], bias? [c_out]]`; padding is `[begin, end]`.
    Conv1d {
        stride: usize,
        padding: [usize; 2],
        dilation: usize,
        groups: usize,
    },
    /// `[x]` with a static target shape, or `[x, shape]` when the shape is a runtime tensor (ONNX).
    Reshape {
        #[serde(default)]
        shape: Option<Vec<usize>>,
    },
    Transpose {
        perm: Vec<usize>,
    },
    Concat {
        axis: isize,
    },
    Slice {
        axis: isize,
        start: i64,
        end: i64,
    },
    Squeeze {
        axes: Vec<isize>,
    },
    Unsqueeze {
        axes: Vec<isize>,
    },
    Tile {
        reps: Vec<usize>,
    },
    Reduce {
        kind: ReduceKind,
        axes: Vec<isize>,
        keep_dims: bool,
    },
    /// `[x [n, d], mask [n] (U32)]` → mean of the rows whose mask is non-zero → `[d]`.
    MeanPool,
    /// `[x]` → `x / ‖x‖₂`.
    L2Normalize,
    /// Fused transformer block Megakernel single-dispatch operator.
    Megakernel(MegakernelDescriptor),
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::Embedding => "Embedding",
            Op::Linear => "Linear",
            Op::MatMul { .. } => "MatMul",
            Op::LayerNorm { .. } => "LayerNorm",
            Op::RmsNorm { .. } => "RmsNorm",
            Op::GroupNorm { .. } => "GroupNorm",
            Op::InstanceNorm { .. } => "InstanceNorm",
            Op::Rope { .. } => "Rope",
            Op::Attention { .. } => "Attention",
            Op::SwiGlu => "SwiGlu",
            Op::TopK { .. } => "TopK",
            Op::GatherCols => "GatherCols",
            Op::MoeLinear { .. } => "MoeLinear",
            Op::Softmax { .. } => "Softmax",
            Op::RelPosBias { .. } => "RelPosBias",
            Op::StateReorder => "StateReorder",
            Op::StateWrite => "StateWrite",
            Op::QsaSelect { .. } => "QsaSelect",
            Op::QsaBias { .. } => "QsaBias",
            Op::Conv2d { .. } => "Conv2d",
            Op::Conv1d { .. } => "Conv1d",
            Op::Reshape { .. } => "Reshape",
            Op::Transpose { .. } => "Transpose",
            Op::Concat { .. } => "Concat",
            Op::Slice { .. } => "Slice",
            Op::Squeeze { .. } => "Squeeze",
            Op::Unsqueeze { .. } => "Unsqueeze",
            Op::Tile { .. } => "Tile",
            Op::Reduce { .. } => "Reduce",
            Op::MeanPool => "MeanPool",
            Op::L2Normalize => "L2Normalize",
            Op::GeGlu => "GeGlu",
            Op::Unary { .. } => "Unary",
            Op::Binary { .. } => "Binary",
            Op::CachedAttention { .. } => "CachedAttention",
            Op::CausalConv1dCached { .. } => "CausalConv1dCached",
            Op::DeltaRule { .. } => "DeltaRule",
            Op::Gather { .. } => "Gather",
            Op::Scatter { .. } => "Scatter",
            Op::Pad { .. } => "Pad",
            Op::AvgPool2d { .. } => "AvgPool2d",
            Op::Upsample { .. } => "Upsample",
            Op::Conv3d { .. } => "Conv3d",
            Op::Megakernel(_) => "Megakernel",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub name: String,
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    /// Placement chosen by the placement pass; builders leave it on the CPU.
    pub device: Device,
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub values: Vec<Value>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push_value(
        &mut self,
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        kind: ValueKind,
    ) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(Value {
            name,
            shape,
            dtype,
            layout: Layout::RowMajor,
            kind,
        });
        id
    }

    /// Declares a graph input; inputs are bound by declaration order at execution.
    pub fn input(&mut self, name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> ValueId {
        let id = self.push_value(name.into(), shape, dtype, ValueKind::Input);
        self.inputs.push(id);
        id
    }

    /// Declares a weight, bound by `name` to [`Weights`] at execution.
    pub fn weight(&mut self, name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> ValueId {
        self.push_value(name.into(), shape, dtype, ValueKind::Weight)
    }

    /// Declares a persistent state tensor, bound to [`StateStore`] at execution. `recurrent`: see
    /// [`ValueKind::State`]; [`program::GraphProgram::recurrent_states`] lists those.
    pub fn state(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DType,
        recurrent: bool,
    ) -> ValueId {
        let name = name.into();
        let kind = ValueKind::State {
            name: name.clone(),
            recurrent,
        };
        self.push_value(name, shape, dtype, kind)
    }

    /// Names of the graph's recurrent states (sorted, deduplicated).
    pub fn recurrent_states(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .values
            .iter()
            .filter_map(|v| match &v.kind {
                ValueKind::State {
                    name,
                    recurrent: true,
                } => Some(name.clone()),
                _ => None,
            })
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Appends a single-output node on the CPU and returns its output value.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        op: Op,
        inputs: &[ValueId],
        shape: Vec<usize>,
        dtype: DType,
    ) -> ValueId {
        let n = self.values.len() as u32;
        assert!(
            inputs.iter().all(|v| v.0 < n),
            "graph node references a value that does not exist yet"
        );
        let name = name.into();
        let node = NodeId(self.nodes.len() as u32);
        let out = self.push_value(name.clone(), shape, dtype, ValueKind::Node(node));
        self.nodes.push(Node {
            name,
            op,
            inputs: inputs.to_vec(),
            outputs: vec![out],
            device: Device::Cpu,
        });
        out
    }

    /// Appends a single-output node whose shape and dtype are inferred from `inputs`
    /// ([`shape::infer`]).
    pub fn push(
        &mut self,
        name: impl Into<String>,
        op: Op,
        inputs: &[ValueId],
    ) -> Result<ValueId, String> {
        let name = name.into();
        let operands = inputs
            .iter()
            .map(|v| {
                let val = self
                    .values
                    .get(v.0 as usize)
                    .ok_or_else(|| format!("node '{name}': operand {v:?} does not exist yet"))?;
                Ok((val.shape.as_slice(), val.dtype))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let (shape, dtype) =
            shape::infer(&op, &operands).map_err(|e| format!("node '{name}': {e}"))?;
        Ok(self.add(name, op, inputs, shape, dtype))
    }

    /// Marks `v` as a graph output; outputs are returned in marking order.
    pub fn output(&mut self, v: ValueId) {
        self.outputs.push(v);
    }

    pub fn value(&self, v: ValueId) -> &Value {
        &self.values[v.0 as usize]
    }

    pub fn node(&self, n: NodeId) -> &Node {
        &self.nodes[n.0 as usize]
    }

    /// The node that produces `v`, if `v` is a node output.
    pub fn producer(&self, v: ValueId) -> Option<&Node> {
        match &self.value(v).kind {
            ValueKind::Node(n) => Some(self.node(*n)),
            _ => None,
        }
    }

    /// Nodes that read `v` (linear scan).
    pub fn consumers(&self, v: ValueId) -> impl Iterator<Item = &Node> + '_ {
        self.nodes.iter().filter(move |n| n.inputs.contains(&v))
    }
}

/// Tensor data for weights, graph inputs, and graph outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    F32(Vec<f32>),
    U32(Vec<u32>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
    Mat(crate::model::weights::Mat),
}

impl Tensor {
    pub fn len(&self) -> usize {
        match self {
            Tensor::F32(v) => v.len(),
            Tensor::U32(v) => v.len(),
            Tensor::F16(v) => v.len(),
            Tensor::BF16(v) => v.len(),
            Tensor::Mat(m) => m.rows * m.cols,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Result<&[f32], String> {
        match self {
            Tensor::F32(v) => Ok(v.as_slice()),
            Tensor::U32(_) => Err("expected an f32 tensor, got u32".into()),
            Tensor::F16(_) => Err("expected an f32 tensor, got f16".into()),
            Tensor::BF16(_) => Err("expected an f32 tensor, got bf16".into()),
            Tensor::Mat(_) => Err("expected an f32 tensor, got Mat".into()),
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Tensor::F32(v) => v.clone(),
            Tensor::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Tensor::F16(v) => v.iter().map(|&x| crate::model::f16_to_f32(x)).collect(),
            Tensor::BF16(v) => v.iter().map(|&x| crate::model::bf16_to_f32(x)).collect(),
            Tensor::Mat(m) => m.to_f32(),
        }
    }

    pub fn into_f32(self) -> Result<Vec<f32>, String> {
        match self {
            Tensor::F32(v) => Ok(v),
            other => Ok(other.to_f32()),
        }
    }

    pub fn as_u32(&self) -> Result<&[u32], String> {
        match self {
            Tensor::U32(v) => Ok(v.as_slice()),
            _ => Err("expected a u32 tensor".into()),
        }
    }
}

/// A paged KV state ([`crate::kv`]): `[rows, row_width]` rows stored in pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedState {
    pub name: String,
    pub row_width: usize,
    /// The attention window of every reader (`None`: full attention, or readers disagree).
    pub window: Option<usize>,
}

/// The graph's paged KV states: non-recurrent 2-D states every reader of which is the k or v
/// state operand of a `CachedAttention`, the two states of each such node both qualifying.
/// Everything else (recurrent conv/ssm, `StateWrite` targets, states another op reads such as
/// cross-attention `xk`/`xv`, graph outputs) stays a contiguous per-sequence buffer.
pub fn paged_states(graph: &Graph) -> Vec<PagedState> {
    use std::collections::{BTreeMap, HashSet};
    let state_name = |v: ValueId| match &graph.value(v).kind {
        ValueKind::State { name, recurrent } => Some((name.as_str(), *recurrent)),
        _ => None,
    };
    // name -> (row width, windows of its readers)
    let mut found: BTreeMap<&str, (usize, Vec<Option<usize>>)> = BTreeMap::new();
    let mut bad: HashSet<&str> = HashSet::new();
    let mut pairs: Vec<(&str, &str)> = Vec::new();
    for node in &graph.nodes {
        let attn = match node.op {
            Op::CachedAttention { window, .. } if node.inputs.len() >= 6 => Some(window),
            _ => None,
        };
        if attn.is_some() {
            if let (Some((k, _)), Some((v, _))) =
                (state_name(node.inputs[3]), state_name(node.inputs[4]))
            {
                pairs.push((k, v));
            }
        }
        for (i, &x) in node.inputs.iter().enumerate() {
            let Some((name, recurrent)) = state_name(x) else {
                continue;
            };
            let shape = &graph.value(x).shape;
            match attn {
                Some(window) if (i == 3 || i == 4) && !recurrent && shape.len() == 2 => {
                    let e = found.entry(name).or_insert((shape[1], Vec::new()));
                    if e.0 != shape[1] {
                        bad.insert(name);
                    }
                    e.1.push(window);
                }
                _ => {
                    bad.insert(name);
                }
            }
        }
    }
    for &o in &graph.outputs {
        if let Some((name, _)) = state_name(o) {
            bad.insert(name);
        }
    }
    let window_of = |ws: &[Option<usize>]| match ws.first() {
        Some(&w) if ws.iter().all(|&x| x == w) => w,
        _ => None,
    };
    let mut window: BTreeMap<&str, Option<usize>> =
        found.iter().map(|(n, (_, ws))| (*n, window_of(ws))).collect();
    // A node's two states are read by one kernel: both paged with one window, or neither.
    loop {
        let mut changed = false;
        for &(k, v) in &pairs {
            if bad.contains(k) != bad.contains(v) {
                bad.insert(k);
                bad.insert(v);
                changed = true;
            }
            if window.get(k) != window.get(v) {
                for n in [k, v] {
                    if let Some(w) = window.get_mut(n) {
                        changed |= w.take().is_some();
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    found
        .into_iter()
        .filter(|(n, _)| !bad.contains(n))
        .map(|(n, (row_width, _))| PagedState {
            name: n.to_string(),
            row_width,
            window: window[n],
        })
        .collect()
}

/// Persistent state buffers mapped by name for sequential / autoregressive decoding.
#[derive(Debug, Default)]
pub struct StateStore {
    pub buffers: HashMap<String, Tensor>,
    #[cfg(target_os = "macos")]
    pub metal_buffers: HashMap<String, (std::sync::Arc<::metal::Buffer>, usize)>,
    /// Names in `metal_buffers` whose elements are f16 (attention KV caches); the rest are f32. The
    /// element count in `metal_buffers` counts elements of either type.
    #[cfg(target_os = "macos")]
    pub metal_f16: std::collections::HashSet<String>,
    /// The greedy decode stream in flight on this state, if any (a clone starts without one).
    #[cfg(target_os = "macos")]
    pub stream: crate::graph::metal::StreamSlot,
    /// The paged KV states' page tables ([`crate::kv`]); the program attaches it on first run.
    pub kv: Option<crate::kv::KvView>,
}

impl StateStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.buffers.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.buffers.get_mut(name)
    }

    pub fn insert(&mut self, name: impl Into<String>, t: Tensor) {
        self.buffers.insert(name.into(), t);
    }

    /// Rewinds to `pos` positions. Full KV states need nothing (attention reads only the rows
    /// below the position it is given, and later writes overwrite the rest); a held state named in
    /// `recurrent` (e.g. [`program::GraphProgram::recurrent_states`]) cannot rewind and errs:
    /// `restore` a [`Self::snapshot`] taken at `pos` instead.
    pub fn truncate(&mut self, pos: usize, recurrent: &[String]) -> Result<(), String> {
        let names = self.buffers.keys().chain(self.metal_names());
        match names.into_iter().find(|n| recurrent.contains(n)) {
            Some(n) => Err(format!(
                "cannot rewind recurrent state {n} to position {pos}: restore a snapshot"
            )),
            None => Ok(()),
        }
    }

    #[cfg(target_os = "macos")]
    fn metal_names(&self) -> impl Iterator<Item = &String> {
        self.metal_buffers.keys()
    }

    #[cfg(not(target_os = "macos"))]
    fn metal_names(&self) -> impl Iterator<Item = &String> {
        std::iter::empty()
    }

    /// Device buffer `name` as f32 (f16 buffers are widened); a paged state gathers its stored
    /// rows through its page table.
    #[cfg(target_os = "macos")]
    pub fn read_metal(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(v) = self.kv.as_ref().filter(|v| v.has(name)) {
            return v.gather(name);
        }
        let (buf, n) = self.metal_buffers.get(name)?;
        // SAFETY: shared-storage buffer of `n` elements of the recorded type; programs run
        // synchronously, so the GPU is idle.
        Some(unsafe {
            if self.metal_f16.contains(name) {
                std::slice::from_raw_parts(buf.contents() as *const u16, *n)
                    .iter()
                    .map(|&h| crate::model::f16_to_f32(h))
                    .collect()
            } else {
                std::slice::from_raw_parts(buf.contents() as *const f32, *n).to_vec()
            }
        })
    }

    /// Copies of the named states (device-resident ones are read back from their buffers);
    /// paged KV states are skipped (they fork by page instead, [`Self::fork`]).
    pub fn snapshot(&self, names: &[&str]) -> HashMap<String, Tensor> {
        let mut snap = HashMap::new();
        for &name in names {
            if self.kv.as_ref().is_some_and(|v| v.has(name)) {
                continue;
            }
            #[cfg(target_os = "macos")]
            if let Some(v) = self.read_metal(name) {
                snap.insert(name.to_string(), Tensor::F32(v));
                continue;
            }
            if let Some(t) = self.buffers.get(name) {
                snap.insert(name.to_string(), t.clone());
            }
        }
        snap
    }

    /// Writes `snapshot` back: into the device buffer of a device-resident state.
    pub fn restore(&mut self, snapshot: HashMap<String, Tensor>) {
        for (name, t) in snapshot {
            #[cfg(target_os = "macos")]
            if let Some((buf, n)) = self.metal_buffers.get(&name) {
                if let Tensor::F32(v) = &t {
                    // The device buffer may have grown past the snapshot (states grow with the context).
                    assert!(
                        v.len() <= *n,
                        "snapshot of state {name} is larger than its buffer"
                    );
                    // SAFETY: as in `read_metal`; `v` has v.len() <= n elements.
                    unsafe {
                        if self.metal_f16.contains(&name) {
                            let dst =
                                std::slice::from_raw_parts_mut(buf.contents() as *mut u16, v.len());
                            for (d, &x) in dst.iter_mut().zip(v) {
                                *d = crate::model::f32_to_f16(x.clamp(-65504.0, 65504.0));
                            }
                        } else {
                            std::ptr::copy_nonoverlapping(
                                v.as_ptr(),
                                buf.contents() as *mut f32,
                                v.len(),
                            )
                        }
                    };
                    self.buffers.remove(&name);
                    continue;
                }
            }
            self.buffers.insert(name, t);
        }
    }

    #[cfg(target_os = "macos")]
    pub fn sync_metal_to_cpu(&mut self) {
        let synced: Vec<_> = self
            .metal_buffers
            .keys()
            .filter_map(|name| Some((name.clone(), self.read_metal(name)?)))
            .collect();
        for (name, v) in synced {
            self.buffers.insert(name, Tensor::F32(v));
        }
    }

    /// An independent copy holding positions `..len`: contiguous states (host and device) are
    /// copied whole, paged KV shares this store's pages ([`crate::kv::KvView::fork`]).
    pub fn fork(&self, len: usize) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        let metal_buffers = {
            let mut mb = HashMap::with_capacity(self.metal_buffers.len());
            if !self.metal_buffers.is_empty() {
                let dev = crate::graph::metal::system_device()?;
                for (name, (src, n)) in &self.metal_buffers {
                    let bytes = src.length() as usize;
                    let dst = crate::graph::metal::new_shared_buffer(&dev, bytes)?;
                    // SAFETY: two shared buffers of `bytes` bytes; programs run synchronously.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.contents() as *const u8,
                            dst.contents() as *mut u8,
                            bytes,
                        )
                    };
                    mb.insert(name.clone(), (std::sync::Arc::new(dst), *n));
                }
            }
            mb
        };
        Ok(Self {
            buffers: self.buffers.clone(),
            #[cfg(target_os = "macos")]
            metal_buffers,
            #[cfg(target_os = "macos")]
            metal_f16: self.metal_f16.clone(),
            #[cfg(target_os = "macos")]
            stream: crate::graph::metal::StreamSlot::default(),
            kv: self.kv.as_ref().map(|v| v.fork(len)).transpose()?,
        })
    }
}

/// Weight data by name, owned outside the graph.
#[derive(Debug, Clone, Default)]
pub struct Weights(HashMap<String, Tensor>);

impl Weights {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, data: Tensor) {
        self.0.insert(name.into(), data);
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.0.get(name)
    }
    pub fn contains_key(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        self.0.remove(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Total element count across all weights.
    pub fn parameter_count(&self) -> usize {
        self.0.values().map(Tensor::len).sum()
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    /// Full KV rewinds by position; a recurrent state (by the graph's flag, not its name) refuses
    /// and comes back from a snapshot; a device-resident state is read from and written back into
    /// its buffer.
    #[test]
    fn truncate_snapshot_restore() {
        let mut g = Graph::new();
        g.state("layers.3.k_cache", vec![4, 2], DType::F32, false);
        g.state("layers.0.mixer", vec![2], DType::F32, true);
        g.state("layers.0.mixer", vec![2], DType::F32, true);
        let recurrent = g.recurrent_states();
        assert_eq!(recurrent, ["layers.0.mixer"]);
        let mut st = StateStore::new();
        st.insert("layers.3.k_cache", Tensor::F32(vec![1.0; 8]));
        assert!(st.truncate(2, &recurrent).is_ok());
        st.insert("layers.0.mixer", Tensor::F32(vec![1.0, 2.0]));
        assert!(st.truncate(2, &recurrent).unwrap_err().contains("mixer"));
        assert!(st.truncate(2, &[]).is_ok());
        st.insert("layers.0.ssm_state", Tensor::F32(vec![1.0, 2.0]));
        let snap = st.snapshot(&["layers.0.ssm_state", "missing"]);
        assert_eq!(snap.len(), 1);
        st.insert("layers.0.ssm_state", Tensor::F32(vec![9.0, 9.0]));
        st.restore(snap);
        assert_eq!(
            st.get("layers.0.ssm_state").unwrap().as_f32().unwrap(),
            &[1.0, 2.0]
        );
        #[cfg(target_os = "macos")]
        if let Some(dev) = ::metal::Device::system_default() {
            let buf = dev.new_buffer(8, ::metal::MTLResourceOptions::StorageModeShared);
            // SAFETY: shared 8-byte buffer = 2 f32.
            unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, 2) }
                .copy_from_slice(&[5.0, 6.0]);
            st.metal_buffers
                .insert("layers.1.conv_state".into(), (std::sync::Arc::new(buf), 2));
            let snap = st.snapshot(&["layers.1.conv_state"]);
            let live = st.metal_buffers["layers.1.conv_state"].0.contents() as *mut f32;
            // SAFETY: as above.
            unsafe { live.write(0.0) };
            st.restore(snap);
            assert_eq!(unsafe { *live }, 5.0);
        }
    }

    /// `Op` deserializes from its externally tagged JSON (unit variants as strings, optional fields
    /// omitted) and every op round-trips.
    #[test]
    fn op_serde_round_trips() {
        use serde_json::json;
        let parse = |v: serde_json::Value| serde_json::from_value::<Op>(v).unwrap();
        assert_eq!(
            parse(json!({"RmsNorm": {"eps": 1e-6}})),
            Op::RmsNorm { eps: 1e-6 }
        );
        assert_eq!(parse(json!("Linear")), Op::Linear);
        assert_eq!(
            parse(json!({"Attention": {"heads": 8, "kv_heads": 2, "causal": true}})),
            Op::Attention {
                heads: 8,
                kv_heads: 2,
                causal: true,
                window: None,
                soft_cap: None,
                scale: None,
            }
        );
        assert_eq!(
            parse(json!({"Unary": {"kind": {"Clamp": [-1.0, 1.0]}}})),
            Op::Unary {
                kind: UnaryKind::Clamp(-1.0, 1.0)
            }
        );
        let ops = [
            Op::Attention {
                heads: 4,
                kv_heads: 4,
                causal: false,
                window: Some(8),
                soft_cap: Some(30.0),
                scale: Some(1.0),
            },
            Op::Unary {
                kind: UnaryKind::RoundEven,
            },
            Op::Binary {
                kind: BinaryKind::Max,
            },
            Op::Rope {
                head_dim: 64,
                rotary_dim: 32,
                variant: RopeVariant::MRope {
                    sections: [8, 4, 4],
                    interleaved: false,
                    pairs: PairLayout::Half,
                },
                mscale: 1.0,
            },
            Op::Reduce {
                kind: ReduceKind::Mean,
                axes: vec![-1],
                keep_dims: true,
            },
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Bilinear,
            },
            Op::Reshape { shape: None },
            Op::StateWrite,
            Op::QsaSelect {
                ratio: 4,
                top_blocks: 2,
            },
            Op::Megakernel(MegakernelDescriptor {
                kind: MegakernelKind::Mlp {
                    hidden_dim: 8,
                    intermediate_dim: 16,
                    eps: 1e-5,
                    is_geglu: true,
                    has_post_norm: false,
                },
                name: "mlp".into(),
            }),
        ];
        for op in ops {
            let v = serde_json::to_value(&op).unwrap();
            assert_eq!(parse(v), op);
        }
    }
}
