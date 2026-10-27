//! Metal lowering group `moe`: planning, encoding and kernels for its ops.
//! Kernels live in `moe.metal` (appended to the shared library, names listed in `KERNELS`).
//! Hooks called from `ext.rs`; every function returns "not mine" for ops it does not claim.
//!
//! Owns: `TopK` (row top-k indices), `QsaSelect` / `QsaBias` (sparse block key selection and its
//! mask), `GatherCols`, `MoeLinear` (expert-indexed GEMV over a stacked
//! weight bank, generated per codec by `codegen::gemv_indexed`, f32/f16/bf16 banks through
//! pseudo-codecs), `Embedding` over F16/BF16/codec-quantized tables, and shape guards for
//! `CausalConv1dCached` / `DeltaRule` forms the core kernels would read out of bounds.

use super::super::{new_shared_buffer, numel, set_bytes, u32_of, Planned, Session, TG};
use super::{Enc, PlanOut};
use crate::graph::codec::{codec, Codec};
use crate::graph::codegen::{
    block_ref, device_msl, gemm_common, gemm_packed, gemm_packed_name, gemv_indexed,
    gemv_indexed_name, gemv_rows_per_tg, GemvParams,
};
use crate::graph::{DType, Graph, Node, Op, ValueId};
use crate::hardware::QuantFormat;
use metal::{CompileOptions, ComputePipelineState, MTLSize};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::LazyLock;

pub(super) const MSL: &str = include_str!("moe.metal");
/// Compute functions in `moe.metal`.
pub(super) const KERNELS: &[&str] = &[
    "moe_topk_f32",
    "moe_argmax_part_f32",
    "moe_argmax_reduce_f32",
    "moe_topk_softmax_f32",
    "moe_gather_cols_f32",
    "moe_embed_f16",
    "moe_embed_bf16",
    "moe_qsa_select_f32",
    "moe_qsa_bias_f32",
    "moe_map_rows",
];

/// A single-row `TopK { k: 1 }` at least this wide runs the two-pass argmax (one threadgroup per
/// `ARGMAX_SPAN` columns, then one reducing threadgroup) instead of one threadgroup over the row.
const ARGMAX_WIDE: usize = 8192;
const ARGMAX_SPAN: usize = 4096;
/// Minimum routed slot count (`m * top_k`) at which `Step::ExpertLinear` uses grouped expert GEMM.
const MOE_GEMM_MIN_SLOTS: usize = 32;

/// Source encoding of an embedding table.
#[derive(Debug, Clone, Copy)]
pub enum Table {
    F16,
    BF16,
    Quant(QuantFormat),
}

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    TopK {
        x: ValueId,
        rows: usize,
        cols: usize,
        k: usize,
        weights_out: Option<ValueId>,
        scale: Option<ValueId>,
    },
    GatherCols {
        x: ValueId,
        idx: ValueId,
        total: usize,
        cols: usize,
        k: usize,
        /// `x` is one row shared by every row of `idx`.
        shared: bool,
    },
    ExpertLinear {
        x: ValueId,
        bank: ValueId,
        ids: ValueId,
        /// Storage dtype of the bank (`bank_codec` reads it).
        dtype: DType,
        /// Rows of one expert's `[rows, cols]` matrix.
        rows: usize,
        cols: usize,
        slots: usize,
        top_k: usize,
        experts: usize,
        x_per_slot: bool,
    },
    Embed {
        table: ValueId,
        ids: ValueId,
        n: usize,
        d: usize,
        rows: usize,
        src: Table,
    },
    /// `QsaSelect` (`select`) or `QsaBias` over `score [n, b]` and `pos [n]`; output `[n, k]`.
    Qsa {
        score: ValueId,
        pos: ValueId,
        select: bool,
        n: usize,
        b: usize,
        ratio: usize,
        top_blocks: usize,
        k: usize,
    },
}

/// Float banks read through the same `decode8` contract as the quantized codecs: one 8-element block.
static CODEC_F32: Codec = Codec {
    format: QuantFormat::FP32,
    block_elems: 8,
    block_bytes: 32,
    msl: "inline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n    device const float* f = (device const float*)blk;\n    for (uint i = 0; i < 8; ++i) out8[i] = f[i];\n}\n",
};

/// How the expert GEMV reads a bank stored as `dtype`: every codec format, plus f32/f16/bf16.
pub(super) fn bank_codec(dtype: DType) -> Option<&'static Codec> {
    match dtype {
        DType::F32 => Some(&CODEC_F32),
        DType::F16 => Some(&crate::graph::codegen::CODEC_F16),
        DType::BF16 => Some(&crate::graph::codegen::CODEC_BF16),
        DType::Quant(f) => codec(f),
        DType::U32 => None,
    }
}

/// Whether the Metal executor lowers `op` for `dtype`: the activation dtype for `TopK`/`GatherCols`,
/// the bank dtype for `MoeLinear`, the table dtype for `Embedding`.
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    match op {
        Op::TopK { .. } | Op::GatherCols | Op::QsaSelect { .. } | Op::QsaBias { .. } => {
            dtype == DType::F32
        }
        Op::MoeLinear { .. } => bank_codec(dtype).is_some(),
        Op::Embedding => match dtype {
            DType::F16 | DType::BF16 => true,
            DType::Quant(f) => codec(f).is_some(),
            _ => false,
        },
        _ => false,
    }
}

fn shape2(g: &Graph, v: ValueId) -> Option<[usize; 2]> {
    match g.value(v).shape[..] {
        [a, b] => Some([a, b]),
        _ => None,
    }
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    _aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<Step>, String>> {
    let name = &node.name;
    let arg = |k: usize| {
        node.inputs
            .get(k)
            .copied()
            .ok_or_else(|| format!("node '{name}': {} needs operand {k}", node.op.name()))
    };
    match &node.op {
        Op::TopK { k } => Some((|| {
            let x = arg(0)?;
            let cols = g.value(x).shape.last().copied().unwrap_or(0);
            if *k == 0 || *k > cols {
                return Err(format!("node '{name}': TopK needs 1 <= k <= the row width {cols}, got k={k}"));
            }
            let n = numel(g, x)?;
            u32_of(*k)?;
            Ok(PlanOut::Step(
                Step::TopK {
                    x,
                    rows: n / cols,
                    cols,
                    k: *k,
                    weights_out: None,
                    scale: None,
                },
                n / cols * k,
            ))
        })()),
        Op::GatherCols => Some((|| {
            let (x, idx) = (arg(0)?, arg(1)?);
            let (cols, k) = (
                g.value(x).shape.last().copied().unwrap_or(0),
                g.value(idx).shape.last().copied().unwrap_or(0),
            );
            let total = numel(g, idx)?;
            let x_rows = numel(g, x)?.checked_div(cols).unwrap_or(0);
            if cols == 0 || k == 0 || (x_rows != total / k && x_rows != 1) || g.value(idx).dtype != DType::U32 {
                return Err(format!(
                    "node '{name}': GatherCols needs x [rows, n] (or one shared row) and u32 idx [rows, k]"
                ));
            }
            u32_of(total)?;
            Ok(PlanOut::Step(Step::GatherCols { x, idx, total, cols, k, shared: x_rows == 1 }, total))
        })()),
        Op::QsaSelect { ratio, top_blocks } | Op::QsaBias { ratio, top_blocks } => Some((|| {
            let (score, pos) = (arg(0)?, arg(1)?);
            let Some([n, b]) = shape2(g, score) else {
                return Err(format!("node '{name}': QSA score is not [n, B]"));
            };
            if *ratio == 0 || numel(g, pos)? != n || g.value(pos).dtype != DType::U32 {
                return Err(format!(
                    "node '{name}': QSA needs ratio >= 1 and u32 positions [{n}], got ratio {ratio}"
                ));
            }
            let k = top_blocks * ratio + ratio - 1;
            u32_of(n * k.max(b))?;
            let select = matches!(node.op, Op::QsaSelect { .. });
            let (ratio, top_blocks) = (*ratio, *top_blocks);
            let step = Step::Qsa { score, pos, select, n, b, ratio, top_blocks, k };
            Ok(PlanOut::Step(step, n * k))
        })()),
        Op::MoeLinear { experts, top_k } => Some((|| {
            if node.inputs.len() != 3 {
                return Err(format!("node '{name}': MoeLinear needs [x, bank, ids], got {} operands", node.inputs.len()));
            }
            let (x, bank, ids) = (arg(0)?, arg(1)?, arg(2)?);
            if *experts == 0 || *top_k == 0 {
                return Err(format!("node '{name}': MoeLinear needs experts >= 1 and top_k >= 1"));
            }
            let (Some([x_rows, cols]), Some([bank_rows, bank_cols])) = (shape2(g, x), shape2(g, bank)) else {
                return Err(format!("node '{name}': MoeLinear needs x [m, in] and a bank [experts * out, in]"));
            };
            if bank_cols != cols || bank_rows % experts != 0 || bank_rows == 0 {
                return Err(format!(
                    "node '{name}': bank [{bank_rows}, {bank_cols}] is not {experts} stacked [out, {cols}] matrices"
                ));
            }
            let slots = numel(g, ids)?;
            if g.value(ids).dtype != DType::U32 || slots % top_k != 0 {
                return Err(format!("node '{name}': ids must be u32 [m, {top_k}]"));
            }
            if x_rows != slots && x_rows * top_k != slots {
                return Err(format!(
                    "node '{name}': x has {x_rows} rows, neither m nor m * {top_k} for {slots} slots"
                ));
            }
            let rows = bank_rows / experts;
            if numel(g, node.outputs[0])? != slots * rows {
                return Err(format!("node '{name}': output must be [{slots}, {rows}]"));
            }
            let dtype = g.value(bank).dtype;
            let c = bank_codec(dtype).ok_or_else(|| {
                format!("node '{name}': bank '{}' ({dtype:?}) has no GEMV codec", g.value(bank).name)
            })?;
            if cols % c.block_elems as usize != 0 {
                return Err(format!(
                    "node '{name}': {cols} columns are not a multiple of the {:?} block ({})",
                    c.format, c.block_elems
                ));
            }
            for n in [rows, cols, slots, *top_k, *experts] {
                u32_of(n)?;
            }
            Ok(PlanOut::Step(
                Step::ExpertLinear {
                    x,
                    bank,
                    ids,
                    dtype,
                    rows,
                    cols,
                    slots,
                    top_k: *top_k,
                    experts: *experts,
                    x_per_slot: x_rows == slots,
                },
                slots * rows,
            ))
        })()),
        Op::Embedding => {
            let table = node.inputs.first().copied()?;
            let src = match g.value(table).dtype {
                DType::F16 => Table::F16,
                DType::BF16 => Table::BF16,
                DType::Quant(f) if codec(f).is_some() => Table::Quant(f),
                _ => return None,
            };
            Some((|| {
                let ids = arg(1)?;
                let Some([rows, d]) = shape2(g, table) else {
                    return Err(format!("node '{name}': embedding table is not [rows, d]"));
                };
                if g.value(ids).dtype != DType::U32 || !g.inputs.contains(&ids) {
                    return Err(format!(
                        "node '{name}': Metal embedding ids must be a u32 graph input (range-checked on the host)"
                    ));
                }
                if let Table::Quant(f) = src {
                    let be = codec(f).map_or(1, |c| c.block_elems as usize);
                    if d % be != 0 {
                        return Err(format!("node '{name}': width {d} is not a multiple of the {f:?} block ({be})"));
                    }
                }
                let n = numel(g, ids)?;
                Ok(PlanOut::Step(Step::Embed { table, ids, n, d, rows, src }, n * d))
            })())
        }
        // The core kernels index state and per-head inputs unchecked; the CPU executor resets a
        // mis-sized state to zeros and defaults short gates. Reject those forms instead of reading OOB.
        Op::CausalConv1dCached { kernel, dilation } => (|| {
            let (k, d) = (*kernel, *dilation);
            let w = arg(1)?;
            if k == 0 {
                return Err(format!("node '{name}': kernel must be >= 1"));
            }
            let channels = numel(g, w)? / k;
            match node.inputs.get(2) {
                Some(&s) if numel(g, s)? != (k - 1) * d * channels => Err(format!(
                    "node '{name}': conv state must hold (kernel-1)*dilation = {} rows of {channels} channels",
                    (k - 1) * d
                )),
                _ => Ok(()),
            }
        })()
        .err()
        .map(Err),
        Op::DeltaRule { key_heads: kh, val_heads: vh, key_dim: kd, val_dim: vd } => (|| {
            let (q, beta, ld) = (arg(0)?, arg(3)?, arg(4)?);
            if *kh == 0 || *vh % *kh != 0 || *kd == 0 || *vd == 0 {
                return Err(format!("node '{name}': val_heads must be a positive multiple of key_heads"));
            }
            let n = numel(g, q)? / (kh * kd);
            if numel(g, arg(1)?)? != n * kh * kd
                || numel(g, arg(2)?)? != n * vh * vd
                || numel(g, beta)? < n * vh
                || numel(g, ld)? < n * vh
            {
                return Err(format!("node '{name}': q/k/v/beta/log_decay sizes disagree for {n} tokens"));
            }
            match node.inputs.get(5) {
                Some(&s) if numel(g, s)? != vh * kd * vd => {
                    Err(format!("node '{name}': ssm state must hold {} values", vh * kd * vd))
                }
                _ => Ok(()),
            }
        })()
        .err()
        .map(Err),
        _ => None,
    }
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match st {
        Step::TopK { x, scale, .. } => {
            let mut v = vec![*x];
            if let Some(sc) = *scale {
                v.push(sc);
            }
            v
        }
        Step::GatherCols { x, idx, .. } => vec![*x, *idx],
        Step::ExpertLinear { x, bank, ids, .. } => vec![*x, *bank, *ids],
        Step::Embed { table, ids, .. } => vec![*table, *ids],
        Step::Qsa { score, pos, .. } => vec![*score, *pos],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, v: ValueId) -> bool {
    match st {
        Step::TopK { .. } => false,
        Step::GatherCols { idx, .. } => v == *idx,
        Step::ExpertLinear { x, .. } => v != *x,
        Step::Embed { .. } => true,
        Step::Qsa { pos, .. } => v == *pos,
    }
}

/// Generated gather for one codec: one thread per 8-element sub-block of one token's row.
fn quant_embed_src(c: &Codec) -> String {
    let be = c.block_elems;
    let br = block_ref(
        c,
        "table",
        &format!("(ulong)min(ids[gid.y], p.z - 1) * (d / {be}u) + blk"),
    );
    let (decl, sub) = (&br.decl, &br.sub);
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
{msl}
kernel void embed_quant(device const uchar* table [[buffer(0)]],
                        device const uint* ids [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint4& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {{
    uint n = p.x, d = p.y;
    if (gid.x * 8 >= d || gid.y >= n) return;
    uint blk = (gid.x * 8) / {be}u;
    uint sub = ((gid.x * 8) % {be}u) / 8;
    {decl}
    float v[8];
    decode8(b, {sub}sub, v);
    for (uint i = 0; i < 8; ++i) out[(ulong)gid.y * d + gid.x * 8 + i] = v[i];
}}
"#,
        msl = device_msl(c)
    )
}

/// Per (device, codec) pipelines of [`quant_embed_src`], compiled on first use.
static QUANT_EMBED: LazyLock<Mutex<HashMap<(u64, QuantFormat), ComputePipelineState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn quant_embed_pso(sess: &Session, f: QuantFormat) -> Result<ComputePipelineState, String> {
    let device = &sess.rt.device;
    let key = (device.registry_id(), f);
    if let Some(p) = QUANT_EMBED.lock().get(&key) {
        return Ok(p.clone());
    }
    let c = codec(f).ok_or_else(|| format!("no codec for {f:?}"))?;
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let lib = device
        .new_library_with_source(&quant_embed_src(c), &opts)
        .map_err(|e| format!("generated {f:?} embedding gather failed to compile: {e}"))?;
    let func = lib.get_function("embed_quant", None)?;
    let pso = device.new_compute_pipeline_state_with_function(&func)?;
    QUANT_EMBED.lock().insert(key, pso.clone());
    Ok(pso)
}

#[cfg(test)]
thread_local! {
    static TEST_ROWS_PER_SG: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_PSO_OVERRIDE: std::cell::RefCell<Option<ComputePipelineState>> =
        const { std::cell::RefCell::new(None) };
}

fn moe_rows_per_sg(_c: &Codec) -> usize {
    #[cfg(test)]
    {
        let ov = TEST_ROWS_PER_SG.get();
        if ov != 0 {
            return ov;
        }
    }
    // Conservative default 2 until a format wins >= 6/7 ABBA rounds with worst_ratio > 1.0:
    // match c.format {
    //     QuantFormat::GGUF_IQ4_NL | QuantFormat::GGUF_Q4_0 | QuantFormat::GGUF_IQ2_M => 4,
    //     _ => 2,
    // }
    2
}
/// Per (device, codec, rows_per_sg) pipelines of [`gemv_indexed`], compiled on first use.
static MOE_GEMV: LazyLock<Mutex<HashMap<(u64, QuantFormat, usize), ComputePipelineState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn moe_gemv_pso(sess: &Session, c: &Codec, r: usize) -> Result<ComputePipelineState, String> {
    #[cfg(test)]
    if let Some(pso) = TEST_PSO_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return Ok(pso);
    }
    let device = &sess.rt.device;
    let key = (device.registry_id(), c.format, r);
    if let Some(p) = MOE_GEMV.lock().get(&key) {
        return Ok(p.clone());
    }
    let params = GemvParams {
        rows_per_sg: r,
        ..Default::default()
    };
    let src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}",
        gemv_indexed(c, params)
    );
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).map_err(|e| {
        format!(
            "generated {:?} expert GEMV failed to compile: {e}",
            c.format
        )
    })?;
    let func = lib.get_function(&gemv_indexed_name(c, params), None)?;
    let pso = device.new_compute_pipeline_state_with_function(&func)?;
    MOE_GEMV.lock().insert(key, pso.clone());
    Ok(pso)
}

/// Per (device, codec, epi) pipelines of indexed [`gemm_packed`], compiled on first use.
static MOE_GEMM: LazyLock<Mutex<HashMap<(u64, QuantFormat, u32), ComputePipelineState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn moe_gemm_pso(sess: &Session, c: &Codec, epi: u32) -> Result<ComputePipelineState, String> {
    let device = &sess.rt.device;
    let key = (device.registry_id(), c.format, epi);
    if let Some(p) = MOE_GEMM.lock().get(&key) {
        return Ok(p.clone());
    }
    let mut src = String::from(
        "#define GIBBS_MPP 1\n#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n",
    );
    src.push_str(gemm_common(&super::super::KERNELS_MSL));
    src.push_str(&gemm_packed(c, epi, true));
    let name = gemm_packed_name(c, epi, true);
    let pso = sess.rt.compile_generated(&src, &name)?;
    MOE_GEMM.lock().insert(key, pso.clone());
    Ok(pso)
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    let (enc, rt) = (e.enc, &sess.rt);
    match st {
        Step::TopK {
            x,
            rows,
            cols,
            k,
            weights_out,
            scale,
        } => {
            if let Some(w_out) = *weights_out {
                enc.set_compute_pipeline_state(rt.pso("moe_topk_softmax_f32")?);
                e.bind(0, *x);
                e.bind(1, s.out);
                e.bind(2, w_out);
                e.bind(3, scale.unwrap_or(*x));
                set_bytes(
                    enc,
                    4,
                    &[
                        u32_of(*rows)?,
                        u32_of(*cols)?,
                        u32_of(*k)?,
                        scale.is_some() as u32,
                    ],
                );
                enc.dispatch_thread_groups(
                    MTLSize::new(*rows as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            } else if *rows == 1 && *k == 1 && *cols >= ARGMAX_WIDE {
                let parts = cols.div_ceil(ARGMAX_SPAN);
                let mut guard = sess.argmax_scratch.lock();
                if guard
                    .as_ref()
                    .is_none_or(|b| (b.length() as usize) < parts * 8)
                {
                    *guard = Some(new_shared_buffer(&rt.device, parts * 8)?);
                }
                let part = guard.as_ref().expect("allocated above");
                let idx_off = (parts * 4) as u64;
                enc.set_compute_pipeline_state(rt.pso("moe_argmax_part_f32")?);
                e.bind(0, *x);
                enc.set_buffer(1, Some(part), 0);
                enc.set_buffer(2, Some(part), idx_off);
                set_bytes(enc, 3, &[u32_of(*cols)?, ARGMAX_SPAN as u32]);
                enc.dispatch_thread_groups(
                    MTLSize::new(parts as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.set_compute_pipeline_state(rt.pso("moe_argmax_reduce_f32")?);
                enc.set_buffer(0, Some(part), 0);
                enc.set_buffer(1, Some(part), idx_off);
                e.bind(2, s.out);
                set_bytes(enc, 3, &u32_of(parts)?);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            } else {
                enc.set_compute_pipeline_state(rt.pso("moe_topk_f32")?);
                e.bind(0, *x);
                e.bind(1, s.out);
                set_bytes(enc, 2, &[u32_of(*rows)?, u32_of(*cols)?, u32_of(*k)?]);
                enc.dispatch_thread_groups(
                    MTLSize::new(*rows as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
        }
        Step::GatherCols {
            x,
            idx,
            total,
            cols,
            k,
            shared,
        } => {
            enc.set_compute_pipeline_state(rt.pso("moe_gather_cols_f32")?);
            e.bind(0, *x);
            e.bind(1, *idx);
            e.bind(2, s.out);
            set_bytes(
                enc,
                3,
                &[u32_of(*total)?, u32_of(*cols)?, u32_of(*k)?, *shared as u32],
            );
            e.grid1(*total);
        }
        Step::ExpertLinear {
            x,
            bank,
            ids,
            dtype,
            rows,
            cols,
            slots,
            top_k,
            experts,
            x_per_slot,
        } => {
            let c = bank_codec(*dtype)
                .ok_or_else(|| format!("node '{}': no GEMV codec for {dtype:?}", s.name))?;
            if *slots >= MOE_GEMM_MIN_SLOTS && rt.profile.mpp.is_ok() {
                let need = (*experts + *experts * *slots) * 4;
                let mut guard = sess.argmax_scratch.lock();
                if guard.as_ref().is_none_or(|b| (b.length() as usize) < need) {
                    *guard = Some(new_shared_buffer(&rt.device, need)?);
                }
                let scratch = guard.as_ref().expect("allocated above");
                let list_off = (*experts * 4) as u64;
                enc.set_compute_pipeline_state(rt.pso("moe_map_rows")?);
                e.bind(0, *ids);
                enc.set_buffer(1, Some(scratch), 0);
                enc.set_buffer(2, Some(scratch), list_off);
                e.bind(3, s.out);
                set_bytes(
                    enc,
                    4,
                    &[u32_of(*slots)?, u32_of(*experts)?, u32_of(*rows)?],
                );
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));

                let pso = moe_gemm_pso(sess, c, 0)?;
                enc.set_compute_pipeline_state(&pso);
                e.bind(0, *x);
                e.bind(1, *bank);
                enc.set_buffer(2, Some(scratch), 0);
                enc.set_buffer(3, Some(scratch), list_off);
                e.bind(4, s.out);
                set_bytes(
                    enc,
                    5,
                    &[
                        u32_of(*rows)?,
                        u32_of(*cols)?,
                        u32_of(*slots)?,
                        u32_of(*top_k)?,
                        u32_of(*experts)?,
                        *x_per_slot as u32,
                    ],
                );
                let shmem = 24576;
                enc.set_threadgroup_memory_length(0, shmem);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (*slots as u64).div_ceil(128),
                        (*rows as u64).div_ceil(64),
                        *experts as u64,
                    ),
                    MTLSize::new(super::super::TG4, 1, 1),
                );
            } else {
                let r = moe_rows_per_sg(c);
                let pso = moe_gemv_pso(sess, c, r)?;
                enc.set_compute_pipeline_state(&pso);
                e.bind(0, *bank);
                e.bind(1, *x);
                e.bind(2, s.out);
                set_bytes(
                    enc,
                    3,
                    &[
                        u32_of(*rows)?,
                        u32_of(*cols)?,
                        u32_of(*slots)?,
                        u32_of(*top_k)?,
                        u32_of(*experts)?,
                        *x_per_slot as u32,
                    ],
                );
                e.bind(4, *ids);
                let rows_per_tg = gemv_rows_per_tg(GemvParams {
                    rows_per_sg: r,
                    ..Default::default()
                });
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (*rows as u64).div_ceil(rows_per_tg as u64),
                        *slots as u64,
                        1,
                    ),
                    MTLSize::new(256, 1, 1),
                );
            }
        }
        Step::Embed {
            table,
            ids,
            n,
            d,
            rows,
            src,
        } => {
            // Ids the host wrote into the arena are range-checked here; ids a queued token
            // computes on the device (`Session::submit_fed`) are clamped in the kernels.
            if !sess.fed_ids.load(std::sync::atomic::Ordering::Relaxed) {
                let (ib, io) = (e.loc)(*ids);
                // SAFETY: `plan` requires `ids` to be a graph input, which the host copies into this
                // shared buffer (n u32) before any step is encoded.
                let idv = unsafe {
                    std::slice::from_raw_parts(
                        (ib.contents() as *const u8).add(io) as *const u32,
                        *n,
                    )
                };
                if let Some((r, &id)) = idv
                    .iter()
                    .enumerate()
                    .find(|&(_, &id)| id as usize >= *rows)
                {
                    return Err(format!(
                        "node '{}': row {r} id {id} out of range for table of {rows} rows",
                        s.name
                    ));
                }
            }
            let (pso, cols_per_thread) = match src {
                Table::F16 => (rt.pso("moe_embed_f16")?.clone(), 1),
                Table::BF16 => (rt.pso("moe_embed_bf16")?.clone(), 1),
                Table::Quant(f) => (quant_embed_pso(sess, *f)?, 8),
            };
            enc.set_compute_pipeline_state(&pso);
            e.bind(0, *table);
            e.bind(1, *ids);
            e.bind(2, s.out);
            set_bytes(enc, 3, &[u32_of(*n)?, u32_of(*d)?, u32_of(*rows)?, 0]);
            enc.dispatch_thread_groups(
                MTLSize::new(((*d / cols_per_thread) as u64).div_ceil(TG), *n as u64, 1),
                MTLSize::new(TG, 1, 1),
            );
        }
        Step::Qsa {
            score,
            pos,
            select,
            n,
            b,
            ratio,
            top_blocks,
            k,
        } => {
            if *n * *k > 0 {
                let params = [
                    u32_of(*n)?,
                    u32_of(*b)?,
                    u32_of(*ratio)?,
                    u32_of(*top_blocks)?,
                    u32_of(*k)?,
                ];
                if *select {
                    enc.set_compute_pipeline_state(rt.pso("moe_qsa_select_f32")?);
                    e.bind(0, *score);
                    e.bind(1, *pos);
                    e.bind(2, s.out);
                    set_bytes(enc, 3, &params);
                    enc.dispatch_thread_groups(
                        MTLSize::new(*n as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                } else {
                    enc.set_compute_pipeline_state(rt.pso("moe_qsa_bias_f32")?);
                    e.bind(0, *pos);
                    e.bind(1, s.out);
                    set_bytes(enc, 2, &params);
                    e.grid1(*n * *k);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::testutil::{both, rand, rel_err, rt};
    use super::super::super::Session;
    use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, Weights};
    use crate::hardware::QuantFormat as Q;
    use crate::model::{f32_to_bf16, f32_to_f16};

    /// Storage format of a test weight.
    #[derive(Clone, Copy, Debug)]
    enum Fmt {
        F32,
        F16,
        BF16,
        Q(Q),
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Byte offsets of the f16 fields a block dequantizer reads (mirrors `tests/codec_parity.rs`).
    fn f16_offsets(f: Q) -> &'static [usize] {
        match f {
            Q::GGUF_Q4_1 | Q::GGUF_Q5_1 | Q::GGUF_Q4_K_M | Q::GGUF_Q5_K_M => &[0, 2],
            Q::GGUF_Q6_K => &[208],
            Q::GGUF_Q2_K => &[80, 82],
            Q::GGUF_Q3_K => &[108],
            Q::GGUF_TQ1_0 => &[52],
            Q::GGUF_TQ2_0 => &[64],
            Q::GGUF_IQ1_M | Q::MXFP4 => &[],
            _ => &[0],
        }
    }

    /// Random finite blocks. `small` keeps scales in 2^-5..2^1 so GEMV sums stay well conditioned.
    fn random_blocks(f: Q, n_blocks: usize, small: bool, seed: u64) -> Vec<u8> {
        let bb = crate::graph::codec::codec(f).expect("codec").block_bytes as usize;
        let mut r = Lcg(seed);
        let f16 = |r: &mut Lcg| -> u16 {
            let sign = (r.next() & 1) as u16;
            let exp = if small {
                10 + (r.next() % 6) as u16
            } else if r.next() % 8 == 0 {
                0
            } else {
                1 + (r.next() % 25) as u16
            };
            (sign << 15) | (exp << 10) | (r.next() & 0x3FF) as u16
        };
        let mut raw: Vec<u8> = (0..n_blocks * bb).map(|_| (r.next() >> 5) as u8).collect();
        for b in raw.chunks_exact_mut(bb) {
            for &o in f16_offsets(f) {
                b[o..o + 2].copy_from_slice(&f16(&mut r).to_le_bytes());
            }
            match f {
                Q::GGUF_IQ1_M => {
                    let h = f16(&mut r);
                    for k in 0..4 {
                        let at = 48 + 2 * k;
                        let w = u16::from_le_bytes([b[at], b[at + 1]]);
                        let w = (w & 0x0FFF) | (((h >> (4 * k)) & 0xF) << 12);
                        b[at..at + 2].copy_from_slice(&w.to_le_bytes());
                    }
                }
                Q::MXFP4 => b[0] = 100 + (r.next() % 50) as u8,
                _ => {}
            }
        }
        raw
    }

    /// Random blocks of `f` that decode to finite, moderate values on the CPU (formats with unknown
    /// scale fields, like the repacked ones, get bad blocks redrawn).
    fn sane_blocks(f: Q, n_blocks: usize, seed: u64) -> Vec<u8> {
        let (_, _, dequant) = crate::model::layers::gguf_block_codec(f).expect("CPU dequantizer");
        let be = crate::graph::codec::codec(f).expect("codec").block_elems as usize;
        let mut out = Vec::new();
        for b in 0..n_blocks {
            for attempt in 0..1000u64 {
                let blk = random_blocks(f, 1, true, seed * 1_000_003 + b as u64 * 1009 + attempt);
                let mut y = vec![0.0f32; be];
                dequant(&blk, &mut y);
                if y.iter().all(|v| v.is_finite() && v.abs() < 4096.0) {
                    out.extend(blk);
                    break;
                }
                assert!(attempt < 999, "{f:?}: no sane random block found");
            }
        }
        out
    }

    /// A `[rows, cols]` weight in `fmt` with its graph dtype.
    fn weight(fmt: Fmt, rows: usize, cols: usize, seed: u64) -> (DType, Tensor) {
        let vals = rand(rows * cols, seed);
        match fmt {
            Fmt::F32 => (DType::F32, Tensor::F32(vals)),
            Fmt::F16 => (
                DType::F16,
                Tensor::F16(vals.iter().map(|&v| f32_to_f16(v)).collect()),
            ),
            Fmt::BF16 => (
                DType::BF16,
                Tensor::BF16(vals.iter().map(|&v| f32_to_bf16(v)).collect()),
            ),
            Fmt::Q(f) => {
                let be = crate::graph::codec::codec(f).expect("codec").block_elems as usize;
                assert_eq!(cols % be, 0);
                let raw = sane_blocks(f, rows * cols / be, seed);
                (
                    DType::Quant(f),
                    Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                        f, raw, rows, cols,
                    )),
                )
            }
        }
    }

    // ---- TopK / GatherCols ----

    /// (CPU reference, Metal) indices; the Metal output carries u32 bits in its f32 slots.
    fn topk_both(x: Vec<f32>, rows: usize, cols: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
        let mut g = Graph::new();
        let xv = g.input("x", vec![rows, cols], DType::F32);
        let y = g.add("tk", Op::TopK { k }, &[xv], vec![rows, k], DType::U32);
        g.output(y);
        let inputs = vec![Tensor::F32(x)];
        let want = match cpu::run_stateless(&g, &Weights::new(), inputs.clone())
            .unwrap()
            .remove(0)
        {
            Tensor::U32(v) => v,
            t => panic!("TopK must produce u32 indices, got {t:?}"),
        };
        let got = Session::new(rt(), &Weights::new())
            .unwrap()
            .run(&g, inputs)
            .unwrap()
            .remove(0)
            .as_u32()
            .unwrap()
            .to_vec();
        (want, got)
    }

    #[test]
    fn topk_orders_by_value_then_lower_index() {
        let (want, got) = topk_both(vec![1.0, 3.0, 3.0, 2.0, -0.0, 0.0, f32::NAN, 5.0], 2, 4, 3);
        assert_eq!(
            want,
            vec![1, 2, 3, 3, 0, 1],
            "ties and signed zeros keep the lower index first"
        );
        assert_eq!(got, want);
        // NaN ranks below every number: it is picked last even when k covers the whole row.
        let (want, got) = topk_both(vec![f32::NAN, 1.0, f32::NAN, 2.0], 1, 4, 4);
        assert_eq!(want, vec![3, 1, 0, 2]);
        assert_eq!(got, want);
    }

    /// The two-pass wide-row argmax (`TopK { k: 1 }` over one row >= `ARGMAX_WIDE`) equals the CPU
    /// TopK, and the decode argmax head over a valid prefix (`program::argmax_head`, a Slice view
    /// then TopK) equals the host greedy pick, with the maximum tied across threadgroup spans.
    #[test]
    fn wide_argmax_matches_cpu_and_host_greedy() {
        for cols in [
            super::ARGMAX_WIDE,
            super::ARGMAX_WIDE + 1,
            3 * super::ARGMAX_SPAN + 17,
            262_144,
        ] {
            let distinct = rand(cols, cols as u64);
            let mut tied: Vec<f32> = distinct.iter().map(|v| (v * 3.0).round()).collect();
            // The same maximum in several spans and at both span edges; the first must win.
            for i in [
                super::ARGMAX_SPAN - 1,
                super::ARGMAX_SPAN,
                cols / 2,
                cols - 1,
            ] {
                tied[i] = 100.0;
            }
            let mut nan = distinct.clone();
            nan.iter_mut().step_by(7).for_each(|v| *v = f32::NAN);
            let late = {
                let mut v = vec![-1.0f32; cols];
                v[cols - 1] = 0.0;
                v
            };
            for (name, x) in [
                ("distinct", distinct),
                ("tied", tied),
                ("nan", nan),
                ("late", late),
                ("constant", vec![0.5f32; cols]),
                ("-inf", vec![f32::NEG_INFINITY; cols]),
            ] {
                let (want, got) = topk_both(x.clone(), 1, cols, 1);
                assert_eq!(got, want, "argmax {name} cols={cols}");
                if name == "nan" {
                    continue; // the host scan skips NaN instead of ranking it lowest
                }
                let mut g = Graph::new();
                let xv = g.input("x", vec![1, cols], DType::F32);
                g.output(xv);
                for valid in [cols, cols - cols / 20] {
                    let head = crate::graph::program::argmax_head(&g, valid).unwrap();
                    let got = Session::new(rt(), &Weights::new())
                        .unwrap()
                        .run(&head, vec![Tensor::F32(x.clone())])
                        .unwrap()
                        .remove(0)
                        .as_u32()
                        .unwrap()[0];
                    let want = crate::runtime::lm::sample_logits(&x[..valid], &[], 0.0, 50, 1.0);
                    assert_eq!(got, want, "head {name} cols={cols} valid={valid}");
                }
            }
        }
    }

    #[test]
    fn topk_matches_cpu_across_shapes_and_ties() {
        for (rows, cols, k) in [
            (1, 1, 1),
            (1, 2, 2),
            (3, 8, 3),
            (4, 128, 8),
            (2, 129, 129),
            (1, 1000, 64),
            (5, 300, 1),
            (2, 2048, 8),
            (7, 37, 37),
        ] {
            let seed = (rows * 31 + cols) as u64;
            let distinct = rand(rows * cols, seed);
            // A handful of levels: most values tie.
            let tied: Vec<f32> = distinct.iter().map(|v| (v * 3.0).round()).collect();
            let constant = vec![0.5f32; rows * cols];
            for (name, x) in [
                ("distinct", distinct),
                ("tied", tied),
                ("constant", constant),
            ] {
                let (want, got) = topk_both(x, rows, cols, k);
                assert_eq!(got, want, "topk {name} {rows}x{cols} k={k}");
            }
        }
    }

    #[test]
    fn topk_rejects_bad_k() {
        for k in [0, 5] {
            let mut g = Graph::new();
            let x = g.input("x", vec![2, 4], DType::F32);
            let y = g.add("tk", Op::TopK { k }, &[x], vec![2, k.max(1)], DType::U32);
            g.output(y);
            let inputs = vec![Tensor::F32(vec![0.0; 8])];
            let e = Session::new(rt(), &Weights::new())
                .unwrap()
                .run(&g, inputs.clone())
                .unwrap_err();
            assert!(e.contains("tk"), "{e}");
            assert!(cpu::run_stateless(&g, &Weights::new(), inputs)
                .unwrap_err()
                .contains("tk"));
        }
    }

    #[test]
    fn gather_cols_follows_topk_indices() {
        for (rows, cols, k) in [(1, 8, 3), (4, 100, 8), (3, 37, 37)] {
            let mut g = Graph::new();
            let x = g.input("x", vec![rows, cols], DType::F32);
            let ids = g.add("ids", Op::TopK { k }, &[x], vec![rows, k], DType::U32);
            let y = g.add("top", Op::GatherCols, &[x, ids], vec![rows, k], DType::F32);
            g.output(y);
            let xs = rand(rows * cols, 99 + cols as u64);
            let (want, got) = both(&g, &Weights::new(), vec![Tensor::F32(xs.clone())]);
            assert_eq!(want, got);
            for (r, row) in want.chunks(k).enumerate() {
                let mut sorted = xs[r * cols..(r + 1) * cols].to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a));
                assert_eq!(
                    row,
                    &sorted[..k],
                    "row {r} holds the k largest values, descending"
                );
            }
        }
    }

    #[test]
    fn gather_cols_shares_a_single_row() {
        // One row of per-expert scales indexed by the ids of three tokens.
        let (n, e, k) = (3, 8, 2);
        let mut g = Graph::new();
        let table = g.input("table", vec![1, e], DType::F32);
        let ids = g.input("ids", vec![n, k], DType::U32);
        let y = g.add(
            "scale",
            Op::GatherCols,
            &[table, ids],
            vec![n, k],
            DType::F32,
        );
        g.output(y);
        let table = rand(e, 5);
        let ids = vec![0u32, 7, 3, 3, 6, 1];
        let inputs = vec![Tensor::F32(table.clone()), Tensor::U32(ids.clone())];
        let (want, got) = both(&g, &Weights::new(), inputs);
        assert_eq!(
            want,
            ids.iter().map(|&i| table[i as usize]).collect::<Vec<_>>()
        );
        assert_eq!(got, want);
    }

    // ---- Embedding ----

    fn embed_run(
        fmt: Fmt,
        rows: usize,
        d: usize,
        ids: Vec<u32>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (dt, t) = weight(fmt, rows, d, (rows * 7 + d) as u64);
        let mut g = Graph::new();
        let table = g.weight("t", vec![rows, d], dt);
        let idv = g.input("ids", vec![ids.len()], DType::U32);
        let y = g.add(
            "emb",
            Op::Embedding,
            &[table, idv],
            vec![ids.len(), d],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("t", t);
        let inputs = vec![Tensor::U32(ids)];
        let want = cpu::run_stateless(&g, &w, inputs.clone())?
            .remove(0)
            .into_f32()?;
        let got = Session::new(rt(), &w)?
            .run(&g, inputs)?
            .remove(0)
            .into_f32()?;
        Ok((want, got))
    }

    #[test]
    fn embedding_half_tables_match_cpu() {
        for fmt in [Fmt::F16, Fmt::BF16] {
            for (rows, d) in [(1, 1), (3, 3), (37, 37), (129, 64), (5, 129)] {
                for n in [1usize, 5, 37] {
                    let ids: Vec<u32> = (0..n)
                        .map(|i| ((i * 7 + 3) % rows) as u32)
                        .chain([0, rows as u32 - 1])
                        .collect();
                    let (want, got) = embed_run(fmt, rows, d, ids).unwrap();
                    assert!(rel_err(&want, &got) < 1e-6, "{fmt:?} {rows}x{d} n={n}");
                }
            }
        }
    }

    #[test]
    fn embedding_quantized_tables_match_cpu() {
        let formats = [
            Q::GGUF_Q4_0,
            Q::GGUF_Q4_1,
            Q::GGUF_Q5_0,
            Q::GGUF_Q5_1,
            Q::GGUF_Q8_0,
            Q::GGUF_Q2_K,
            Q::GGUF_Q3_K,
            Q::GGUF_Q4_K_M,
            Q::GGUF_Q5_K_M,
            Q::GGUF_Q6_K,
            Q::GGUF_IQ4_NL,
            Q::GGUF_IQ4_XS,
            Q::GGUF_IQ3_XXS,
            Q::GGUF_IQ3_S,
            Q::GGUF_IQ2_XXS,
            Q::GGUF_IQ2_XS,
            Q::GGUF_IQ2_S,
            Q::GGUF_IQ2_M,
            Q::GGUF_IQ1_S,
            Q::GGUF_IQ1_M,
            Q::GGUF_TQ1_0,
            Q::GGUF_TQ2_0,
            Q::MXFP4,
        ];
        let mut checked = 0;
        for f in formats {
            let Some(c) = crate::graph::codec::codec(f) else {
                continue;
            };
            let be = c.block_elems as usize;
            for (rows, blocks) in [(1usize, 1usize), (7, 3), (37, 2)] {
                let ids: Vec<u32> = vec![0, rows as u32 - 1, (rows / 2) as u32, 0];
                let (want, got) = embed_run(Fmt::Q(f), rows, be * blocks, ids).unwrap();
                let bad = want
                    .iter()
                    .zip(&got)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    bad.is_none(),
                    "{f:?} {rows} rows x {} cols: element {bad:?} differs",
                    be * blocks
                );
            }
            checked += 1;
        }
        assert!(checked >= 10, "only {checked} codecs registered");
    }

    #[test]
    fn embedding_rejects_bad_ids_and_widths() {
        let e = embed_run(Fmt::F16, 4, 8, vec![0, 4]).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let e = embed_run(Fmt::BF16, 4, 8, vec![9]).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let (dt, t) = weight(Fmt::Q(Q::GGUF_Q4_0), 4, 32, 1);
        let mut g = Graph::new();
        let table = g.weight("t", vec![4, 16], dt); // width 16 is not a Q4_0 block multiple
        let ids = g.input("ids", vec![1], DType::U32);
        let y = g.add("emb", Op::Embedding, &[table, ids], vec![1, 16], DType::F32);
        g.output(y);
        let mut w = Weights::new();
        w.insert("t", t);
        let e = Session::new(rt(), &w)
            .unwrap()
            .run(&g, vec![Tensor::U32(vec![0])])
            .unwrap_err();
        assert!(e.contains("block"), "{e}");
    }

    // ---- MoeLinear ----

    /// `(CPU executor, Metal)` outputs of one `MoeLinear` over a random `[experts * rows, cols]` bank;
    /// `x_rows` is `ids.len() / top_k` (broadcast) or `ids.len()` (one row per slot).
    #[allow(clippy::too_many_arguments)]
    fn moe_both(
        fmt: Fmt,
        x_rows: usize,
        cols: usize,
        rows: usize,
        experts: usize,
        top_k: usize,
        ids: Vec<u32>,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let slots = ids.len();
        let (dt, bank) = weight(fmt, experts * rows, cols, seed);
        let mut g = Graph::new();
        let x = g.input("x", vec![x_rows, cols], DType::F32);
        let b = g.weight("bank", vec![experts * rows, cols], dt);
        let idv = g.input("ids", vec![slots / top_k, top_k], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear { experts, top_k },
            &[x, b, idv],
            vec![slots, rows],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("bank", bank);
        both(
            &g,
            &w,
            vec![Tensor::F32(rand(x_rows * cols, seed + 7)), Tensor::U32(ids)],
        )
    }

    #[test]
    fn moe_linear_matches_cpu_for_every_format() {
        // The CPU executor is the exact reference.
        let _exact = crate::model::qdot::exact();
        let mut fmts = vec![Fmt::F32, Fmt::F16, Fmt::BF16];
        fmts.extend(crate::graph::codec::formats().into_iter().map(Fmt::Q));
        let (experts, top_k, rows) = (6, 3, 35);
        let ids = vec![4, 0, 5, 2, 2, 1];
        super::TEST_ROWS_PER_SG.set(0);
        for f in crate::graph::codec::formats() {
            let c = crate::graph::codec::codec(f).expect("codec");
            // let want_r = match f {
            //     Q::GGUF_IQ4_NL | Q::GGUF_Q4_0 | Q::GGUF_IQ2_M => 4,
            //     _ => 2,
            // };
            let want_r = 2;
            assert_eq!(super::moe_rows_per_sg(c), want_r, "{f:?}");
        }
        for r in [2usize, 4usize] {
            super::TEST_ROWS_PER_SG.set(r);
            for (i, fmt) in fmts.iter().enumerate() {
                let cols = match fmt {
                    Fmt::Q(f) => {
                        let be =
                            crate::graph::codec::codec(*f).expect("codec").block_elems as usize;
                        (512 / be).max(1) * be
                    }
                    _ => 64,
                };
                // Broadcast x (2 rows x 3 slots) and one x row per slot.
                for x_rows in [2, 6] {
                    let (want, got) = moe_both(
                        *fmt,
                        x_rows,
                        cols,
                        rows,
                        experts,
                        top_k,
                        ids.clone(),
                        10 + i as u64,
                    );
                    let err = rel_err(&want, &got);
                    assert!(err < 1e-4, "r={r} {fmt:?} x_rows={x_rows}: {err}");
                }
            }
        }
        super::TEST_ROWS_PER_SG.set(0);
    }

    #[test]
    fn moe_linear_shapes_and_routing_match_cpu() {
        let mut s = 7u64;
        let mut route = |n: usize, experts: usize| -> Vec<u32> {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) % experts as u64) as u32
                })
                .collect()
        };
        for (i, (m, top_k, experts, rows, cols)) in [
            (1, 1, 1, 8, 8),
            (1, 8, 128, 17, 64), // a decode token: 8 of 128 experts
            (37, 2, 12, 33, 40), // prefill-sized, rows not a tile multiple
            (3, 4, 4, 129, 24),  // every expert routed several times
            (2, 1, 5, 1, 16),    // one output row per expert
        ]
        .into_iter()
        .enumerate()
        {
            let ids = route(m * top_k, experts);
            for x_rows in [m, m * top_k] {
                let (want, got) = moe_both(
                    Fmt::F32,
                    x_rows,
                    cols,
                    rows,
                    experts,
                    top_k,
                    ids.clone(),
                    40 + i as u64,
                );
                assert!(
                    rel_err(&want, &got) < 1e-4,
                    "m={m} top_k={top_k} E={experts} {rows}x{cols} x_rows={x_rows}"
                );
            }
        }
    }

    #[test]
    fn moe_linear_grouped_gemm_matches_cpu() {
        // The CPU executor is the exact reference.
        let _exact = crate::model::qdot::exact();
        let (experts, top_k, rows) = (16usize, 8usize, 65usize);
        let mut s = 123u64;
        let mut route = |n: usize| -> Vec<u32> {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) % experts as u64) as u32
                })
                .collect()
        };
        for (i, fmt) in [
            Fmt::F32,
            Fmt::F16,
            Fmt::BF16,
            Fmt::Q(Q::GGUF_IQ2_XXS),
            Fmt::Q(Q::GGUF_IQ4_NL),
            Fmt::Q(Q::GGUF_Q4_0),
            Fmt::Q(Q::GGUF_Q8_0),
            Fmt::Q(Q::GGUF_Q4_K_M),
        ]
        .into_iter()
        .enumerate()
        {
            let cols = match fmt {
                Fmt::Q(f) => {
                    let be = crate::graph::codec::codec(f).expect("codec").block_elems as usize;
                    (256 / be).max(1) * be
                }
                _ => 64,
            };
            for m in [4usize, 64usize] {
                let ids = route(m * top_k);
                for x_rows in [m, m * top_k] {
                    let (want, got) = moe_both(
                        fmt,
                        x_rows,
                        cols,
                        rows,
                        experts,
                        top_k,
                        ids.clone(),
                        100 + i as u64,
                    );
                    let err = rel_err(&want, &got);
                    assert!(
                        err < 2e-3,
                        "grouped gemm {fmt:?} m={m} x_rows={x_rows}: rel_err={err}"
                    );
                }
            }
        }
    }

    /// An expert no slot routes to is never read: NaN weights in it leave the output untouched.
    #[test]
    fn moe_linear_reads_only_routed_experts() {
        let (experts, rows, cols) = (4, 16, 32);
        let mut g = Graph::new();
        let x = g.input("x", vec![1, cols], DType::F32);
        let b = g.weight("bank", vec![experts * rows, cols], DType::F32);
        let ids = g.input("ids", vec![1, 2], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear { experts, top_k: 2 },
            &[x, b, ids],
            vec![2, rows],
            DType::F32,
        );
        g.output(y);
        let mut bank = rand(experts * rows * cols, 5);
        let run = |bank: &[f32]| {
            let mut w = Weights::new();
            w.insert("bank", Tensor::F32(bank.to_vec()));
            Session::new(rt(), &w)
                .unwrap()
                .run(
                    &g,
                    vec![Tensor::F32(rand(cols, 1)), Tensor::U32(vec![2, 0])],
                )
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap()
        };
        let clean = run(&bank);
        bank[3 * rows * cols..]
            .iter_mut()
            .for_each(|v| *v = f32::NAN);
        let poisoned = run(&bank);
        assert!(clean.iter().all(|v| v.is_finite()));
        assert_eq!(clean, poisoned);
    }

    #[test]
    fn moe_linear_rejects_malformed_nodes() {
        // (experts, top_k, bank rows, cols, x rows, ids, bank format) -> error must name the node.
        let run = |experts: usize,
                   top_k: usize,
                   bank_rows: usize,
                   cols: usize,
                   x_rows: usize,
                   fmt: Fmt| {
            let (dt, bank) = weight(fmt, bank_rows, cols, 3);
            let mut g = Graph::new();
            let x = g.input("x", vec![x_rows, cols], DType::F32);
            let b = g.weight("bank", vec![bank_rows, cols], dt);
            let ids = g.input("ids", vec![1, 2], DType::U32);
            let y = g.add(
                "moe",
                Op::MoeLinear { experts, top_k },
                &[x, b, ids],
                vec![2, 4],
                DType::F32,
            );
            g.output(y);
            let mut w = Weights::new();
            w.insert("bank", bank);
            Session::new(rt(), &w)
                .unwrap()
                .run(
                    &g,
                    vec![
                        Tensor::F32(vec![0.5; x_rows * cols]),
                        Tensor::U32(vec![0, 1]),
                    ],
                )
                .map(|_| ())
        };
        run(4, 2, 16, 8, 1, Fmt::F32).unwrap();
        let e = run(4, 0, 16, 8, 1, Fmt::F32).unwrap_err();
        assert!(e.contains("moe") && e.contains("top_k"), "{e}");
        let e = run(5, 2, 16, 8, 1, Fmt::F32).unwrap_err();
        assert!(e.contains("moe") && e.contains("stacked"), "{e}");
        let e = run(4, 2, 16, 8, 3, Fmt::F32).unwrap_err();
        assert!(e.contains("moe") && e.contains("rows"), "{e}");
        // 16 columns are not a whole Q4_K block; planning fails before any weight is read.
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 16], DType::F32);
        let b = g.weight("bank", vec![16, 16], DType::Quant(Q::GGUF_Q4_K_M));
        let ids = g.input("ids", vec![1, 2], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear {
                experts: 4,
                top_k: 2,
            },
            &[x, b, ids],
            vec![2, 4],
            DType::F32,
        );
        g.output(y);
        let e = Session::new(rt(), &Weights::new())
            .unwrap()
            .run(
                &g,
                vec![Tensor::F32(vec![0.5; 16]), Tensor::U32(vec![0, 1])],
            )
            .unwrap_err();
        assert!(e.contains("moe") && e.contains("block"), "{e}");
    }

    #[test]
    fn moe_linear_cpu_reference_rejects_out_of_range_experts() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 8], DType::F32);
        let b = g.weight("bank", vec![8, 8], DType::F32);
        let ids = g.input("ids", vec![1, 1], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear {
                experts: 2,
                top_k: 1,
            },
            &[x, b, ids],
            vec![1, 4],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("bank", Tensor::F32(rand(64, 1)));
        let e = cpu::run_stateless(&g, &w, vec![Tensor::F32(rand(8, 2)), Tensor::U32(vec![2])])
            .unwrap_err();
        assert!(e.contains("routes to expert 2"), "{e}");
    }

    /// A routed id past the bank writes zeros on Metal, never stale arena bytes.
    #[test]
    fn moe_linear_metal_zeroes_out_of_range_slots() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 8], DType::F32);
        let b = g.weight("bank", vec![8, 8], DType::F32);
        let ids = g.input("ids", vec![1, 2], DType::U32);
        let y = g.add(
            "moe",
            Op::MoeLinear {
                experts: 2,
                top_k: 2,
            },
            &[x, b, ids],
            vec![2, 4],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("bank", Tensor::F32(rand(64, 1)));
        let got = Session::new(rt(), &w)
            .unwrap()
            .run(&g, vec![Tensor::F32(rand(8, 2)), Tensor::U32(vec![1, 7])])
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        assert!(got[..4].iter().any(|v| *v != 0.0));
        assert_eq!(&got[4..], &[0.0; 4]);
    }

    // ---- guards on core kernels ----

    /// Two chained steps through persistent state, CPU vs Metal; returns the second step's outputs.
    fn two_steps(g: &Graph, w: &Weights, step: [Vec<Tensor>; 2]) -> (Vec<f32>, Vec<f32>) {
        let s = Session::new(rt(), w).unwrap();
        let (mut sc, mut sm) = (StateStore::new(), StateStore::new());
        let mut last = (vec![], vec![]);
        for inputs in step {
            let want = cpu::run(g, w, &mut sc, inputs.clone())
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            let got = s
                .run_with_state(g, Some(&mut sm), inputs)
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            last = (want, got);
        }
        last
    }

    #[test]
    fn causal_conv_dilated_state_matches_cpu() {
        for (k, d, ch, n) in [
            (2, 1, 1, 1),
            (4, 1, 5, 3),
            (3, 2, 5, 7),
            (4, 3, 37, 2),
            (2, 4, 3, 1),
        ] {
            let mut g = Graph::new();
            let x = g.input("x", vec![n, ch], DType::F32);
            let wv = g.weight("w", vec![ch, k], DType::F32);
            let st = g.state("st", vec![(k - 1) * d, ch], DType::F32, true);
            let y = g.add(
                "conv",
                Op::CausalConv1dCached {
                    kernel: k,
                    dilation: d,
                },
                &[x, wv, st],
                vec![n, ch],
                DType::F32,
            );
            g.output(y);
            let mut w = Weights::new();
            w.insert("w", Tensor::F32(rand(ch * k, 1)));
            let steps = [
                vec![Tensor::F32(rand(n * ch, 2))],
                vec![Tensor::F32(rand(n * ch, 3))],
            ];
            let (want, got) = two_steps(&g, &w, steps);
            assert!(
                rel_err(&want, &got) < 1e-5,
                "conv k={k} d={d} ch={ch} n={n}"
            );
        }
    }

    #[test]
    fn causal_conv_rejects_mis_sized_state() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let wv = g.weight("w", vec![4, 3], DType::F32);
        let st = g.state("st", vec![2, 4], DType::F32, true); // dilation 2 needs 4 rows
        let y = g.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 3,
                dilation: 2,
            },
            &[x, wv, st],
            vec![2, 4],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("w", Tensor::F32(rand(12, 1)));
        let e = Session::new(rt(), &w)
            .unwrap()
            .run(&g, vec![Tensor::F32(rand(8, 2))])
            .unwrap_err();
        assert!(e.contains("conv") && e.contains("state"), "{e}");
    }

    fn delta_graph(kh: usize, vh: usize, kd: usize, vd: usize, n: usize, beta_len: usize) -> Graph {
        let mut g = Graph::new();
        let q = g.input("q", vec![n, kh * kd], DType::F32);
        let k = g.input("k", vec![n, kh * kd], DType::F32);
        let v = g.input("v", vec![n, vh * vd], DType::F32);
        let beta = g.input("beta", vec![beta_len], DType::F32);
        let ld = g.input("ld", vec![n, vh], DType::F32);
        let st = g.state("ssm", vec![vh * kd * vd], DType::F32, true);
        let y = g.add(
            "delta",
            Op::DeltaRule {
                key_heads: kh,
                val_heads: vh,
                key_dim: kd,
                val_dim: vd,
            },
            &[q, k, v, beta, ld, st],
            vec![n, vh * vd],
            DType::F32,
        );
        g.output(y);
        g
    }

    #[test]
    fn delta_rule_state_matches_cpu() {
        for (kh, vh, kd, vd, n) in [
            (1, 1, 2, 2, 1),
            (2, 4, 16, 16, 3),
            (2, 2, 8, 37, 5),
            (1, 3, 5, 33, 2),
        ] {
            let g = delta_graph(kh, vh, kd, vd, n, n * vh);
            let step = |s: u64| {
                vec![
                    Tensor::F32(rand(n * kh * kd, s)),
                    Tensor::F32(rand(n * kh * kd, s + 1)),
                    Tensor::F32(rand(n * vh * vd, s + 2)),
                    Tensor::F32(rand(n * vh, s + 3).iter().map(|v| v * 0.5 + 0.5).collect()),
                    Tensor::F32(rand(n * vh, s + 4).iter().map(|v| v * 0.3 - 0.3).collect()),
                ]
            };
            let (want, got) = two_steps(&g, &Weights::new(), [step(10), step(20)]);
            assert!(
                rel_err(&want, &got) < 1e-4,
                "delta {kh}/{vh} {kd}x{vd} n={n}"
            );
        }
    }

    #[test]
    fn delta_rule_rejects_short_gates_and_ragged_heads() {
        let run = |g: &Graph, n_in: [usize; 5]| {
            let inputs = n_in
                .iter()
                .enumerate()
                .map(|(i, &n)| Tensor::F32(rand(n, i as u64)))
                .collect();
            Session::new(rt(), &Weights::new())
                .unwrap()
                .run(g, inputs)
                .unwrap_err()
        };
        let e = run(&delta_graph(1, 1, 2, 2, 3, 1), [6, 6, 6, 1, 3]);
        assert!(e.contains("delta"), "{e}");
        let e = run(&delta_graph(2, 3, 2, 2, 1, 3), [4, 4, 6, 3, 3]);
        assert!(e.contains("multiple"), "{e}");
    }

    #[test]
    fn moe_gemv_pipeline_max_threads_and_occupancy() {
        let s = Session::new(rt(), &Weights::new()).unwrap();
        for (fmt, r) in [
            (Q::GGUF_IQ2_XXS, 1),
            (Q::GGUF_IQ2_XXS, 2),
            (Q::GGUF_IQ2_XXS, 4),
            (Q::GGUF_IQ2_M, 2),
            (Q::GGUF_IQ2_M, 4),
            (Q::GGUF_IQ4_NL, 2),
            (Q::GGUF_IQ4_NL, 4),
            (Q::GGUF_Q4_0, 1),
            (Q::GGUF_Q4_0, 2),
            (Q::GGUF_Q4_0, 4),
            (Q::GGUF_Q4_K_M, 1),
            (Q::GGUF_Q4_K_M, 2),
            (Q::GGUF_Q6_K, 1),
            (Q::GGUF_Q6_K, 2),
        ] {
            let c = super::codec(fmt).unwrap();
            let pso = super::moe_gemv_pso(&s, c, r).unwrap();
            eprintln!(
                "{fmt:?} r={r}: max_total_threads_per_threadgroup={} thread_execution_width={}",
                pso.max_total_threads_per_threadgroup(),
                pso.thread_execution_width()
            );
            assert_eq!(
                pso.max_total_threads_per_threadgroup(),
                1024,
                "{fmt:?} r={r}"
            );
        }
    }

    /// 7 interleaved rounds (legs forward then reversed) of each format's expert GEMV variants:
    /// rows per simdgroup 2 vs 4, plus for IQ2_XXS the threadgroup-staged and bound-buffer grid
    /// lookups at r=2. Each of the 8 steps reads its own bank (>= 128 MB in all), so experts stream
    /// from DRAM rather than the system-level cache. Ratios are leg 0 (r=2, default grid) over the leg.
    #[test]
    #[ignore = "bandwidth measurement"]
    fn moe_gemv_bandwidth() {
        let (experts, top_k, steps) = (16usize, 8usize, 8usize);
        let ids = Tensor::U32(vec![0, 2, 5, 7, 9, 11, 13, 15]);
        for (f, rows, cols, x_rows) in [
            (Q::GGUF_IQ2_XXS, 1408usize, 2816usize, 1usize),
            (Q::GGUF_IQ2_M, 1408, 2816, 1),
            (Q::GGUF_IQ4_NL, 2816, 704, 8),
            (Q::GGUF_Q4_0, 2816, 704, 8),
        ] {
            let c = crate::graph::codec::codec(f).unwrap();
            let (dt, bank) = weight(Fmt::Q(f), experts * rows, cols, 1);
            let routed_bytes =
                top_k * rows * (cols / c.block_elems as usize) * c.block_bytes as usize;
            let mut g = Graph::new();
            let mut w = Weights::new();
            let x = g.input("x", vec![x_rows, cols], DType::F32);
            let id_in = g.input("ids", vec![1, top_k], DType::U32);
            for step in 0..steps {
                let name = format!("bank{step}");
                let b = g.weight(&name, vec![experts * rows, cols], dt);
                w.insert(name, bank.clone());
                let y = g.add(
                    format!("moe_{step}"),
                    Op::MoeLinear { experts, top_k },
                    &[x, b, id_in],
                    vec![top_k, rows],
                    DType::F32,
                );
                g.output(y);
            }
            let s = Session::new(rt(), &w).unwrap();
            let plan = s.prepare_plan(&g).unwrap();
            let xs = Tensor::F32(vec![0.25; x_rows * cols]);
            let legs = [("r2", 2usize), ("r4", 4)];
            let run_leg = |&(_, r): &(&str, usize), n: usize| -> f64 {
                super::TEST_ROWS_PER_SG.set(r);
                let mut leg_s = 0.0;
                for _ in 0..n {
                    s.run_prepared(&g, &plan, None, vec![xs.clone(), ids.clone()])
                        .unwrap();
                    leg_s += s.last_gpu_seconds() / (n * steps) as f64;
                }
                leg_s
            };
            for leg in &legs {
                run_leg(leg, 5);
            }
            let mut rounds = vec![Vec::with_capacity(7); legs.len()];
            for _round in 0..7 {
                let mut sums = vec![0.0f64; legs.len()];
                for i in (0..legs.len()).chain((0..legs.len()).rev()) {
                    sums[i] += 0.5 * run_leg(&legs[i], 10);
                }
                for (r, v) in rounds.iter_mut().zip(sums) {
                    r.push(v);
                }
            }
            super::TEST_ROWS_PER_SG.set(0);
            let median = |t: &[f64]| {
                let mut v = t.to_vec();
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            let (base, base_med) = (&rounds[0], median(&rounds[0]));
            let report: Vec<String> = legs
                .iter()
                .zip(&rounds)
                .map(|(&(label, _), t)| {
                    let wins = t.iter().zip(base).filter(|(a, b)| a < b).count();
                    let worst = t
                        .iter()
                        .zip(base)
                        .map(|(a, b)| b / a)
                        .fold(f64::INFINITY, f64::min);
                    let med = median(t);
                    format!(
                        "{label} med={:.1}us ({:.1} GB/s) [wins={wins}/7, med_ratio={:.4}, worst_ratio={worst:.4}]",
                        med * 1e6,
                        routed_bytes as f64 / med / 1e9,
                        base_med / med
                    )
                })
                .collect();
            println!("moe_gemv {f:?} {rows}x{cols}: {}", report.join(" vs "));
        }
    }

    #[test]
    #[ignore = "bandwidth measurement"]
    fn moe_gemv_iq2xxs_constant_decode_ablation() {
        let f = Q::GGUF_IQ2_XXS;
        let (experts, top_k, rows, cols, x_rows) = (16usize, 8usize, 1408usize, 2816usize, 1usize);
        let ids = Tensor::U32(vec![0, 2, 5, 7, 9, 11, 13, 15]);
        let c = crate::graph::codec::codec(f).unwrap();
        let (dt, bank) = weight(Fmt::Q(f), experts * rows, cols, 1);
        let routed_bytes = top_k * rows * (cols / c.block_elems as usize) * c.block_bytes as usize;
        let mut g = Graph::new();
        let x = g.input("x", vec![x_rows, cols], DType::F32);
        let b = g.weight("bank", vec![experts * rows, cols], dt);
        let id_in = g.input("ids", vec![1, top_k], DType::U32);
        for step in 0..8 {
            let y = g.add(
                format!("moe_{step}"),
                Op::MoeLinear { experts, top_k },
                &[x, b, id_in],
                vec![top_k, rows],
                DType::F32,
            );
            g.output(y);
        }
        let mut w = Weights::new();
        w.insert("bank", bank);
        let s = Session::new(rt(), &w).unwrap();
        let plan = s.prepare_plan(&g).unwrap();
        let xs = Tensor::F32(vec![0.25; x_rows * cols]);
        let params = super::GemvParams {
            rows_per_sg: 2,
            ..Default::default()
        };
        let real_pso = super::moe_gemv_pso(&s, c, 2).unwrap();
        let real_msl = super::gemv_indexed(c, params);
        let stub_msl = real_msl.replace(
            "IQ2XXS_GRID[(q4 >> (8 * k)) & 0xFFu]",
            "as_type<ulong>(uint2(0x08080808u ^ ((q4 >> (8 * k)) & 0xFFu), 0x08080808u))",
        );
        assert_ne!(
            stub_msl, real_msl,
            "expected IQ2XXS_GRID lookup in dot_group"
        );
        let u16_items: Vec<String> = crate::model::layers::IQ2XXS_GRID
            .iter()
            .map(|&v| {
                let mut code16 = 0u16;
                for j in 0..8 {
                    let b = ((v >> (8 * j)) & 0xFF) as u8;
                    let c: u16 = match b {
                        8 => 0,
                        25 => 1,
                        43 => 2,
                        other => panic!("unexpected IQ2XXS_GRID byte {other}"),
                    };
                    code16 |= c << (2 * j);
                }
                format!("{code16:#06x}u")
            })
            .collect();
        let u16_tbl = format!(
            "constant ushort IQ2XXS_GRID_U16[256] = {{\n{}\n}};\n",
            u16_items.join(",")
        );
        let u16_msl = format!(
            "{u16_tbl}{}",
            real_msl.replace(
                "uint2 gw = as_type<uint2>(IQ2XXS_GRID[(q4 >> (8 * k)) & 0xFFu]);\n        float4 g0 = float4(as_type<uchar4>(gw.x)), g1 = float4(as_type<uchar4>(gw.y));",
                "uint g16 = uint(IQ2XXS_GRID_U16[(q4 >> (8 * k)) & 0xFFu]);\n        uint4 c0 = (uint4(g16) >> uint4(0u, 2u, 4u, 6u)) & 3u;\n        uint4 c1 = (uint4(g16) >> uint4(8u, 10u, 12u, 14u)) & 3u;\n        float4 g0 = float4(8u + c0 * 17u + (c0 >> 1u)), g1 = float4(8u + c1 * 17u + (c1 >> 1u));",
            )
        );
        assert!(
            u16_msl.contains("IQ2XXS_GRID_U16[(q4 >> (8 * k)) & 0xFFu]"),
            "expected IQ2XXS_GRID_U16 lookup in u16_msl"
        );
        let opts = super::CompileOptions::new();
        let compile_pso = |msl: &str, label: &str| {
            let src = format!("#include <metal_stdlib>\nusing namespace metal;\n{msl}");
            let lib =
                s.rt.device
                    .new_library_with_source(&src, &opts)
                    .unwrap_or_else(|e| panic!("compile {label}: {e}"));
            let func = lib
                .get_function(&super::gemv_indexed_name(c, params), None)
                .expect("get function");
            s.rt.device
                .new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("create {label} PSO: {e}"))
        };
        let u16_pso = compile_pso(&u16_msl, "IQ2_XXS u16_grid");
        let stub_pso = compile_pso(&stub_msl, "IQ2_XXS const_stub");
        super::TEST_ROWS_PER_SG.set(2);
        for pso in [&real_pso, &u16_pso, &stub_pso] {
            super::TEST_PSO_OVERRIDE.with(|cell| *cell.borrow_mut() = Some((*pso).clone()));
            for _ in 0..5 {
                s.run_prepared(&g, &plan, None, vec![xs.clone(), ids.clone()])
                    .unwrap();
            }
        }
        let mut real_rounds = Vec::with_capacity(5);
        let mut u16_rounds = Vec::with_capacity(5);
        let mut stub_rounds = Vec::with_capacity(5);
        for _round in 0..5 {
            let mut sums = [0.0f64; 3];
            for idx in [0usize, 1, 2, 2, 1, 0] {
                let pso = match idx {
                    0 => &real_pso,
                    1 => &u16_pso,
                    _ => &stub_pso,
                };
                super::TEST_PSO_OVERRIDE.with(|cell| *cell.borrow_mut() = Some(pso.clone()));
                let mut leg_s = 0.0;
                for _ in 0..10 {
                    s.run_prepared(&g, &plan, None, vec![xs.clone(), ids.clone()])
                        .unwrap();
                    leg_s += s.last_gpu_seconds() / (10.0 * 8.0);
                }
                sums[idx] += 0.5 * leg_s;
            }
            real_rounds.push(sums[0]);
            u16_rounds.push(sums[1]);
            stub_rounds.push(sums[2]);
        }
        super::TEST_PSO_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
        super::TEST_ROWS_PER_SG.set(0);
        real_rounds.sort_by(|a, b| a.total_cmp(b));
        u16_rounds.sort_by(|a, b| a.total_cmp(b));
        stub_rounds.sort_by(|a, b| a.total_cmp(b));
        let (real_s, u16_s, stub_s) = (real_rounds[2], u16_rounds[2], stub_rounds[2]);
        let (real_med_us, u16_med_us, live_q4_no_lut_med_us) =
            (real_s * 1e6, u16_s * 1e6, stub_s * 1e6);
        let lut_cost_us = real_med_us - live_q4_no_lut_med_us;
        let real_gbps = routed_bytes as f64 / real_s / 1e9;
        let u16_gbps = routed_bytes as f64 / u16_s / 1e9;
        let stub_gbps = routed_bytes as f64 / stub_s / 1e9;
        println!(
            "iq2xxs_ablation 1408x2816: real_med_us={real_med_us:.1}us ({real_gbps:.1} GB/s) vs u16_med_us={u16_med_us:.1}us ({u16_gbps:.1} GB/s) vs live_q4_no_lut_med_us={live_q4_no_lut_med_us:.1}us ({stub_gbps:.1} GB/s) [lut_cost_us={lut_cost_us:.1}us]"
        );
    }

    /// `raw` uploaded as distinct buffers totalling >= 64 MB: a dispatch loop rotating through them
    /// streams weights from DRAM, where one 3-16 MB buffer re-read back to back stays resident in the
    /// system-level cache (bg_231's Q6_K leg read 414.5 GB/s, above the DRAM ceiling).
    fn dram_rotation(dev: &metal::DeviceRef, raw: &[u8]) -> Vec<metal::Buffer> {
        (0..(64usize << 20).div_ceil(raw.len()))
            .map(|_| {
                dev.new_buffer_with_data(
                    raw.as_ptr() as *const std::ffi::c_void,
                    raw.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )
            })
            .collect()
    }

    #[test]
    #[ignore = "bandwidth measurement"]
    fn tq2_0_tile_vector_vs_bit_hack_ablation() {
        use crate::graph::codec::{codec, planar};
        use crate::graph::codegen::{
            gemv_common, gemv_packed, gemv_packed_name, gemv_rows_per_tg, GemvParams,
        };
        use metal::MTLResourceOptions;
        use std::ffi::c_void;

        let f = Q::GGUF_TQ2_0;
        let c = codec(f).unwrap();
        let pl = planar(f).unwrap();
        let rt = rt();
        let opts = super::CompileOptions::new();
        opts.set_fast_math_enabled(false);

        let old_seg = "inline void tile_seg_GGUF_TQ2_0(uint4 q, uint k, thread float* out) {\n    const int lv = int(k >> 1) * 2;\n    for (uint w = 0; w < 2; w++) {\n        const uint x = q[(k & 1) * 2 + w];\n        for (int j = 0; j < 4; j++) {\n            const int sh = 21 - 8 * j - lv;\n            const uint y = sh >= 0 ? (x << sh) : (x >> -sh);\n            out[4 * w + j] = as_type<float>(0x3F800000u | (y & 0x00600000u)) - 1.25f;\n        }\n    }\n}";
        let new_seg = "inline void tile_seg_GGUF_TQ2_0(uint4 q, uint k, thread float* out) {\n    const uint lv = (k >> 1) * 2;\n    const uint i0 = (k & 1) * 2;\n    *(thread packed_float4*)(out + 0) = float4(as_type<uchar4>((q[i0] >> lv) & 0x03030303u)) - 1.0f;\n    *(thread packed_float4*)(out + 4) = float4(as_type<uchar4>((q[i0 + 1] >> lv) & 0x03030303u)) - 1.0f;\n}";

        for (rows, cols, r) in [
            (12288usize, 1536usize, 4usize),
            (1536, 12288, 1),
            (65536, 1536, 4),
        ] {
            let params = GemvParams {
                rows_per_sg: r,
                ..Default::default()
            };
            let fn_name = gemv_packed_name(c, params);
            let bithack_body = gemv_packed(c, params);
            assert!(
                bithack_body.contains(old_seg) && bithack_body.contains("acc0 += d * (4.0f * t);"),
                "expected TILE_TQ2_0 bithack in generated MSL"
            );
            let uchar4_body = bithack_body
                .replace(old_seg, new_seg)
                .replace("acc0 += d * (4.0f * t);", "acc0 += d * (1.0f * t);");

            let compile_pso = |body: &str, label: &str| {
                let src = format!(
                    "#include <metal_stdlib>\nusing namespace metal;\n{}{body}",
                    gemv_common(&super::super::super::KERNELS_MSL)
                );
                let lib = rt
                    .device
                    .new_library_with_source(&src, &opts)
                    .unwrap_or_else(|e| panic!("compile {label}: {e}"));
                let func = lib.get_function(&fn_name, None).expect("get function");
                rt.device
                    .new_compute_pipeline_state_with_function(&func)
                    .unwrap_or_else(|e| panic!("create {label} PSO: {e}"))
            };
            let bithack_pso = compile_pso(&bithack_body, "tq2_0 bithack");
            let uchar4_pso = compile_pso(&uchar4_body, "tq2_0 uchar4");

            let n_blocks = rows * (cols / c.block_elems as usize);
            let raw = random_blocks(f, n_blocks, true, 100 + rows as u64 + cols as u64);
            let repacked = pl.repack(&raw, c.block_bytes as usize);
            let w_bytes = n_blocks * c.block_bytes as usize;
            let xv = rand(cols, 200 + r as u64);

            let w_bufs = dram_rotation(&rt.device, &repacked);
            let reps = (2 * w_bufs.len()).max(20);
            let x_buf = rt.device.new_buffer_with_data(
                xv.as_ptr() as *const c_void,
                (xv.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let y_a = rt
                .device
                .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
            let y_b = rt
                .device
                .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);

            #[repr(C)]
            struct PackedGemvParams(u32, u32, u32, u32, u32);
            let p_bytes = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);
            let rows_per_tg = gemv_rows_per_tg(params) as u64;
            let grid = super::MTLSize::new((rows as u64).div_ceil(rows_per_tg), 1, 1);
            let tg = super::MTLSize::new(256, 1, 1);

            let run_n = |pso: &super::ComputePipelineState, out: &metal::Buffer, n: usize| -> f64 {
                metal::objc::rc::autoreleasepool(|| {
                    let cb = rt.queue.new_command_buffer();
                    let raw = cb.new_compute_command_encoder();
                    let enc = &super::super::super::Counted::new(raw);
                    enc.set_compute_pipeline_state(pso);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(out), 0);
                    super::set_bytes(enc, 3, &p_bytes);
                    enc.set_buffer(4, Some(out), 0);
                    enc.set_buffer(5, Some(out), 0);
                    for i in 0..n {
                        enc.set_buffer(0, Some(&w_bufs[i % w_bufs.len()]), 0);
                        enc.dispatch_thread_groups(grid, tg);
                    }
                    enc.end_encoding();
                    super::super::super::commit_and_wait(cb).unwrap();
                    super::super::super::gpu_seconds(cb) / n as f64
                })
            };

            run_n(&bithack_pso, &y_a, 1);
            run_n(&uchar4_pso, &y_b, 1);
            let got_a = unsafe { std::slice::from_raw_parts(y_a.contents() as *const f32, rows) };
            let got_b = unsafe { std::slice::from_raw_parts(y_b.contents() as *const f32, rows) };
            let err = rel_err(got_a, got_b);
            assert!(err < 1e-5, "TQ2_0 {rows}x{cols} r={r} parity rel_err={err}");

            for _ in 0..3 {
                run_n(&bithack_pso, &y_a, reps);
                run_n(&uchar4_pso, &y_b, reps);
            }

            let mut bithack_rounds = Vec::with_capacity(5);
            let mut uchar4_rounds = Vec::with_capacity(5);
            let mut uchar4_wins = 0usize;
            for _round in 0..5 {
                let mut a_sum = 0.0;
                let mut b_sum = 0.0;
                for leg in [0usize, 1, 1, 0] {
                    if leg == 0 {
                        a_sum += 0.5 * run_n(&bithack_pso, &y_a, reps);
                    } else {
                        b_sum += 0.5 * run_n(&uchar4_pso, &y_b, reps);
                    }
                }
                if b_sum < a_sum {
                    uchar4_wins += 1;
                }
                bithack_rounds.push(a_sum);
                uchar4_rounds.push(b_sum);
            }
            bithack_rounds.sort_by(|a, b| a.total_cmp(b));
            uchar4_rounds.sort_by(|a, b| a.total_cmp(b));
            let (a_s, b_s) = (bithack_rounds[2], uchar4_rounds[2]);
            let (bithack_med_us, uchar4_med_us) = (a_s * 1e6, b_s * 1e6);
            let bithack_gbps = w_bytes as f64 / a_s / 1e9;
            let uchar4_gbps = w_bytes as f64 / b_s / 1e9;
            let med_ratio = bithack_med_us / uchar4_med_us;
            println!(
                "tq2_0_ablation shape={rows}x{cols}_r{r}: bithack_med_us={bithack_med_us:.1}us ({bithack_gbps:.1} GB/s) vs uchar4_med_us={uchar4_med_us:.1}us ({uchar4_gbps:.1} GB/s) [uchar4_wins={uchar4_wins}/5, med_ratio={med_ratio:.4}]"
            );
        }
    }

    #[test]
    #[ignore = "bandwidth measurement"]
    fn kquant_fused_dot_ablation() {
        use crate::graph::codec::codec;
        use crate::graph::codegen::{
            gemv_common, gemv_packed, gemv_packed_name, gemv_rows_per_tg, GemvParams,
        };
        use metal::MTLResourceOptions;
        use std::ffi::c_void;

        let rt = rt();
        let opts = super::CompileOptions::new();
        opts.set_fast_math_enabled(false);

        let q4k_unfused_fn = "inline void decode_group_GGUF_Q4_K_M(device const uchar* blk, uint g, thread float* out) {\n    float d = cd_f16(blk), dmin = cd_f16(blk + 2);\n    uint p = g >> 2;\n    device const ushort* q = (device const ushort*)(blk + 16 + p * 32 + (g & 3) * 8);\n    for (uint w = 0; w < 2; w++) {\n        uint a = uint(q[2 * w]) | (uint(q[2 * w + 1]) << 16);\n        float4 lo = float4(as_type<uchar4>(a & 0x0F0F0F0Fu));\n        float4 hi = float4(as_type<uchar4>((a >> 4) & 0x0F0F0F0Fu));\n        for (uint i = 0; i < 4; i++) { out[4 * w + i] = lo[i]; out[8 + 4 * w + i] = hi[i]; }\n    }\n    for (uint k = 0; k < 2; k++) {\n        uint sv, mv;\n        cd_k4(2 * p + k, blk + 4, sv, mv);\n        out[16 + k] = d * float(sv);\n        out[18 + k] = -(dmin * float(mv));\n    }\n}";
        let q4k_fused_fn = "inline float dot_group_GGUF_Q4_K_M(device const uchar* blk, uint g, thread const float4* xs) {\n    float d = cd_f16(blk), dmin = cd_f16(blk + 2);\n    uint p = g >> 2;\n    ushort4 qw = *(device const packed_ushort4*)(blk + 16 + p * 32 + (g & 3) * 8);\n    uint a0 = uint(qw.x) | (uint(qw.y) << 16);\n    uint a1 = uint(qw.z) | (uint(qw.w) << 16);\n    float4 lo0 = float4(as_type<uchar4>(a0 & 0x0F0F0F0Fu));\n    float4 lo1 = float4(as_type<uchar4>(a1 & 0x0F0F0F0Fu));\n    float4 hi0 = float4(as_type<uchar4>((a0 >> 4) & 0x0F0F0F0Fu));\n    float4 hi1 = float4(as_type<uchar4>((a1 >> 4) & 0x0F0F0F0Fu));\n    float dot0 = dot(lo0, xs[0]) + dot(lo1, xs[1]);\n    float dot1 = dot(hi0, xs[2]) + dot(hi1, xs[3]);\n    uint sv0, mv0, sv1, mv1;\n    cd_k4(2 * p, blk + 4, sv0, mv0);\n    cd_k4(2 * p + 1, blk + 4, sv1, mv1);\n    float sx0 = dot(xs[0] + xs[1], float4(1.0f));\n    float sx1 = dot(xs[2] + xs[3], float4(1.0f));\n    return (d * float(sv0)) * dot0 - (dmin * float(mv0)) * sx0\n         + (d * float(sv1)) * dot1 - (dmin * float(mv1)) * sx1;\n}";

        let q6k_unfused_fn = "inline void decode_group_GGUF_Q6_K(device const uchar* blk, uint g, thread float* out) {\n    float d = cd_f16(blk + 208);\n    uint n = g >> 2, l0 = (g & 3) * 8;\n    device const ushort* lo = (device const ushort*)(blk + n * 64 + l0);\n    device const ushort* hi = (device const ushort*)(blk + n * 64 + 32 + l0);\n    device const ushort* qh = (device const ushort*)(blk + 128 + n * 32 + l0);\n    for (uint w = 0; w < 2; w++) {\n        uint a = uint(lo[2 * w]) | (uint(lo[2 * w + 1]) << 16);\n        uint b = uint(hi[2 * w]) | (uint(hi[2 * w + 1]) << 16);\n        uint h = uint(qh[2 * w]) | (uint(qh[2 * w + 1]) << 16);\n        float4 c0 = float4(as_type<uchar4>((a & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4)));\n        float4 c1 = float4(as_type<uchar4>((b & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2)));\n        float4 c2 = float4(as_type<uchar4>(((a >> 4) & 0x0F0F0F0Fu) | (h & 0x30303030u)));\n        float4 c3 = float4(as_type<uchar4>(((b >> 4) & 0x0F0F0F0Fu) | ((h & 0xC0C0C0C0u) >> 2)));\n        for (uint i = 0; i < 4; i++) {\n            out[4 * w + i] = c0[i];\n            out[8 + 4 * w + i] = c1[i];\n            out[16 + 4 * w + i] = c2[i];\n            out[24 + 4 * w + i] = c3[i];\n        }\n    }\n    device const char* sc = (device const char*)(blk + 192 + n * 8 + (l0 >> 4));\n    for (uint k = 0; k < 4; k++) {\n        float s = d * float(sc[2 * k]);\n        out[32 + k] = s;\n        out[36 + k] = -32.0f * s;\n    }\n}";
        let q6k_fused_fn = "inline float dot_group_GGUF_Q6_K(device const uchar* blk, uint g, thread const float4* xs) {\n    float d = cd_f16(blk + 208);\n    uint n = g >> 2, l0 = (g & 3) * 8;\n    device const ushort* lo = (device const ushort*)(blk + n * 64 + l0);\n    device const ushort* hi = (device const ushort*)(blk + n * 64 + 32 + l0);\n    device const ushort* qh = (device const ushort*)(blk + 128 + n * 32 + l0);\n    float4 dots = float4(0.0f);\n    for (uint w = 0; w < 2; w++) {\n        ushort2 lw = *(device const packed_ushort2*)(lo + 2 * w);\n        ushort2 hw = *(device const packed_ushort2*)(hi + 2 * w);\n        ushort2 qw = *(device const packed_ushort2*)(qh + 2 * w);\n        uint a = uint(lw.x) | (uint(lw.y) << 16);\n        uint b = uint(hw.x) | (uint(hw.y) << 16);\n        uint h = uint(qw.x) | (uint(qw.y) << 16);\n        float4 c0 = float4(as_type<uchar4>((a & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4)));\n        float4 c1 = float4(as_type<uchar4>((b & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2)));\n        float4 c2 = float4(as_type<uchar4>(((a >> 4) & 0x0F0F0F0Fu) | (h & 0x30303030u)));\n        float4 c3 = float4(as_type<uchar4>(((b >> 4) & 0x0F0F0F0Fu) | ((h & 0xC0C0C0C0u) >> 2)));\n        dots.x += dot(c0, xs[w]);\n        dots.y += dot(c1, xs[2 + w]);\n        dots.z += dot(c2, xs[4 + w]);\n        dots.w += dot(c3, xs[6 + w]);\n    }\n    device const char* sc = (device const char*)(blk + 192 + n * 8 + (l0 >> 4));\n    float acc = 0.0f;\n    for (uint k = 0; k < 4; k++) {\n        float s = d * float(sc[2 * k]);\n        float sx = dot(xs[2 * k] + xs[2 * k + 1], float4(1.0f));\n        acc += s * dots[k] - 32.0f * s * sx;\n    }\n    return acc;\n}";

        for (f, rows, cols, r) in [
            (Q::GGUF_Q4_K_M, 12288usize, 1536usize, 2usize),
            (Q::GGUF_Q4_K_M, 4096, 1536, 1),
            (Q::GGUF_Q6_K, 1536, 12288, 1),
        ] {
            let c = codec(f).unwrap();
            let params = GemvParams {
                rows_per_sg: r,
                ..Default::default()
            };
            let fn_name = gemv_packed_name(c, params);
            let unfused_body = gemv_packed(c, params);
            let mut fused_body = if f == Q::GGUF_Q4_K_M {
                assert!(unfused_body.contains(q4k_unfused_fn));
                unfused_body.replace(q4k_unfused_fn, q4k_fused_fn)
            } else {
                assert!(unfused_body.contains(q6k_unfused_fn));
                unfused_body.replace(q6k_unfused_fn, q6k_fused_fn)
            };
            let nseg = if f == Q::GGUF_Q4_K_M { 2usize } else { 4usize };
            for k in 0..nseg {
                let sx_line = format!(
                    "        const float sx{k} = dot(xs[{}], float4(1.0f)) + dot(xs[{}], float4(1.0f));\n",
                    2 * k,
                    2 * k + 1
                );
                assert!(fused_body.contains(&sx_line));
                fused_body = fused_body.replace(&sx_line, "");
            }
            for i in 0..r {
                let seg_dot = |k: usize| {
                    let o = 8 * k;
                    format!(
                        "dot(float4(v[{o}], v[{}], v[{}], v[{}]), xs[{}]) + dot(float4(v[{}], v[{}], v[{}], v[{}]), xs[{}])",
                        o + 1, o + 2, o + 3, 2 * k, o + 4, o + 5, o + 6, o + 7, 2 * k + 1
                    )
                };
                let mut dots = String::new();
                for k in 0..nseg {
                    dots += &format!(
                        "            acc{i} += v[{}] * ({}) + v[{}] * sx{k};\n",
                        8 * nseg + k,
                        seg_dot(k),
                        9 * nseg + k
                    );
                }
                let v_len = 10 * nseg;
                let old_row = format!(
                    "            float v[{v_len}];\n            decode_group_{f:?}(b, g, v);\n{dots}"
                );
                let new_row = format!("            acc{i} += dot_group_{f:?}(b, g, xs);\n");
                assert!(fused_body.contains(&old_row));
                fused_body = fused_body.replace(&old_row, &new_row);
            }

            let compile_pso = |body: &str, label: &str| {
                let src = format!(
                    "#include <metal_stdlib>\nusing namespace metal;\n{}{body}",
                    gemv_common(&super::super::super::KERNELS_MSL)
                );
                let lib = rt
                    .device
                    .new_library_with_source(&src, &opts)
                    .unwrap_or_else(|e| panic!("compile {label}: {e}"));
                let func = lib.get_function(&fn_name, None).expect("get function");
                rt.device
                    .new_compute_pipeline_state_with_function(&func)
                    .unwrap_or_else(|e| panic!("create {label} PSO: {e}"))
            };
            let unfused_pso = compile_pso(&unfused_body, "kquant unfused");
            let fused_pso = compile_pso(&fused_body, "kquant fused");

            let n_blocks = rows * (cols / c.block_elems as usize);
            let raw = random_blocks(f, n_blocks, true, 300 + rows as u64 + cols as u64);
            let w_bytes = raw.len();
            let xv = rand(cols, 400 + r as u64);

            let w_bufs = dram_rotation(&rt.device, &raw);
            let reps = (2 * w_bufs.len()).max(20);
            let x_buf = rt.device.new_buffer_with_data(
                xv.as_ptr() as *const c_void,
                (xv.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let y_a = rt
                .device
                .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
            let y_b = rt
                .device
                .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);

            #[repr(C)]
            struct PackedGemvParams(u32, u32, u32, u32, u32);
            let p_bytes = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);
            let rows_per_tg = gemv_rows_per_tg(params) as u64;
            let grid = super::MTLSize::new((rows as u64).div_ceil(rows_per_tg), 1, 1);
            let tg = super::MTLSize::new(256, 1, 1);

            let run_n = |pso: &super::ComputePipelineState, out: &metal::Buffer, n: usize| -> f64 {
                metal::objc::rc::autoreleasepool(|| {
                    let cb = rt.queue.new_command_buffer();
                    let raw = cb.new_compute_command_encoder();
                    let enc = &super::super::super::Counted::new(raw);
                    enc.set_compute_pipeline_state(pso);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(out), 0);
                    super::set_bytes(enc, 3, &p_bytes);
                    enc.set_buffer(4, Some(out), 0);
                    enc.set_buffer(5, Some(out), 0);
                    for i in 0..n {
                        enc.set_buffer(0, Some(&w_bufs[i % w_bufs.len()]), 0);
                        enc.dispatch_thread_groups(grid, tg);
                    }
                    enc.end_encoding();
                    super::super::super::commit_and_wait(cb).unwrap();
                    super::super::super::gpu_seconds(cb) / n as f64
                })
            };

            run_n(&unfused_pso, &y_a, 1);
            run_n(&fused_pso, &y_b, 1);
            let got_a = unsafe { std::slice::from_raw_parts(y_a.contents() as *const f32, rows) };
            let got_b = unsafe { std::slice::from_raw_parts(y_b.contents() as *const f32, rows) };
            let err = rel_err(got_a, got_b);
            assert!(err < 1e-5, "{f:?} {rows}x{cols} r={r} parity rel_err={err}");

            for _ in 0..3 {
                run_n(&unfused_pso, &y_a, reps);
                run_n(&fused_pso, &y_b, reps);
            }

            let mut unfused_rounds = Vec::with_capacity(5);
            let mut fused_rounds = Vec::with_capacity(5);
            let mut fused_wins = 0usize;
            for _round in 0..5 {
                let mut a_sum = 0.0;
                let mut b_sum = 0.0;
                for leg in [0usize, 1, 1, 0] {
                    if leg == 0 {
                        a_sum += 0.5 * run_n(&unfused_pso, &y_a, reps);
                    } else {
                        b_sum += 0.5 * run_n(&fused_pso, &y_b, reps);
                    }
                }
                if b_sum < a_sum {
                    fused_wins += 1;
                }
                unfused_rounds.push(a_sum);
                fused_rounds.push(b_sum);
            }
            unfused_rounds.sort_by(|a, b| a.total_cmp(b));
            fused_rounds.sort_by(|a, b| a.total_cmp(b));
            let (a_s, b_s) = (unfused_rounds[2], fused_rounds[2]);
            let (unfused_med_us, fused_med_us) = (a_s * 1e6, b_s * 1e6);
            let unfused_gbps = w_bytes as f64 / a_s / 1e9;
            let fused_gbps = w_bytes as f64 / b_s / 1e9;
            let med_ratio = unfused_med_us / fused_med_us;
            println!(
                "kquant_ablation format={f:?} shape={rows}x{cols}_r{r}: unfused_med_us={unfused_med_us:.1}us ({unfused_gbps:.1} GB/s) vs fused_med_us={fused_med_us:.1}us ({fused_gbps:.1} GB/s) [fused_wins={fused_wins}/5, med_ratio={med_ratio:.4}]"
            );
        }
    }
}
