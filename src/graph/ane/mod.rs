//! Apple Neural Engine backend through the private `AppleNeuralEngine.framework`: graph fragments
//! lower to MIL text ([`mil`]), compile in memory and evaluate over IOSurfaces ([`runtime`]).
//!
//! Facts this relies on, measured on macOS 27.2 / M5 Pro (`h17`, 16 cores):
//! - Programs take fp32 `[1, C, 1, S]` surfaces (channel rows 64-byte aligned, any `S`), compute
//!   in fp16 (fp32 arithmetic fails to compile), and bind multiple inputs and outputs when every
//!   input surface has one size (likewise outputs), in alphabetical MIL-name order.
//! - Non-scalar constants must be `BLOBFILE`s (inline fp16 tensors are rejected); `rsqrt` and `log`
//!   are rejected; `constexpr_blockwise_shift_scale` compiles only on 2-D data.
//! - About 119 compiles fit in one process ([`ANE_MAX_PROGRAMS`]); an eval costs ~0.1 ms.
//! - Shared events on the in-memory path fault, so evaluation is synchronous.

// objc 0.2's `msg_send!` expands a `cfg(feature = "cargo-clippy")` check.
#![allow(unexpected_cfgs)]

pub mod mil;
pub mod runtime;

pub use mil::{check, lower, rope_table, supports_op, IoBinding, IoLayout, IoSource};
pub use runtime::{programs_compiled, AneProgram, EvalEntry, Surface, ANE_MAX_PROGRAMS};

use crate::graph::{DType, Graph, Op, Tensor, Weights};
use crate::hardware::QuantFormat;
use objc::runtime::{BOOL, NO};
use objc::{msg_send, sel, sel_impl};
use std::sync::OnceLock;

/// The probed ANE: device facts, the eval entry point that worked, and per weight form either
/// `Ok(relative RMS error against the CPU)` or why it is unusable.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AneInfo {
    pub cores: u32,
    pub arch: String,
    pub eval: EvalEntry,
    /// fp16 `const` weights.
    pub fp16: Result<f32, String>,
    /// Per-channel int8 through `constexpr_affine_dequantize`.
    pub int8: Result<f32, String>,
    /// Blockwise int4 (and blockwise int8) through `constexpr_blockwise_shift_scale`.
    pub int4: Result<f32, String>,
    /// Compile and mean eval time of the fp16 probe program.
    pub compile_ms: f64,
    pub eval_us: f64,
}

/// Relative RMS error a probe form may show against the CPU.
const PROBE_TOL: f32 = 1e-2;

static PROBE: OnceLock<Result<AneInfo, String>> = OnceLock::new();

/// Probes the ANE once per process; the result is cached under `<cache>/host/` per OS build and
/// probe program, so later processes compile nothing.
pub fn probe() -> Result<&'static AneInfo, String> {
    PROBE
        .get_or_init(probe_cached)
        .as_ref()
        .map_err(Clone::clone)
}

/// The probe result if this process already has it (never probes).
pub fn probed() -> Option<&'static AneInfo> {
    PROBE.get()?.as_ref().ok()
}

/// Runs `graph` lowered into `prog` (by [`lower`] with layout `io`): packs `inputs` (bound to
/// `graph.inputs` in order) and the host-computed Rope tables into the surfaces the host fills,
/// evaluates, and unpacks the host-read outputs in `graph.outputs` order (the others, read on
/// the GPU, return empty).
pub fn run(
    prog: &mut AneProgram,
    io: &IoLayout,
    graph: &Graph,
    weights: &Weights,
    inputs: &[Tensor],
) -> Result<Vec<Tensor>, String> {
    let input = |v| {
        graph
            .inputs
            .iter()
            .position(|&x| x == v)
            .map(|i| &inputs[i])
            .ok_or_else(|| format!("'{}' is not a graph input", graph.value(v).name))
    };
    for (i, b) in io.inputs.iter().enumerate().filter(|(_, b)| b.host) {
        let data = match b.source {
            IoSource::Value(v) => input(v)?.to_f32(),
            IoSource::RopeTable(n) => {
                let node = &graph.nodes[n];
                let name = &graph.value(node.inputs[2]).name;
                let inv = weights
                    .get(name)
                    .ok_or_else(|| format!("weight '{name}' is not bound"))?
                    .to_f32();
                rope_table(&node.op, input(node.inputs[1])?.as_u32()?, &inv, b.spatial)?
            }
        };
        prog.input(i).write(|d| b.pack(&data, d))?;
    }
    prog.eval()?;
    io.outputs
        .iter()
        .enumerate()
        .map(|(i, b)| match b.host {
            true => Ok(Tensor::F32(prog.output(i).read(|s| b.unpack(s))?)),
            false => Ok(Tensor::F32(Vec::new())),
        })
        .collect()
}

/// Relative RMS error of `got` against `want`.
pub(crate) fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (w as f64).powi(2);
    }
    if got.len() != want.len() || num.is_nan() {
        return f32::INFINITY;
    }
    (num / den.max(1e-30)).sqrt() as f32
}

/// The `[64, 64] · [64, 64]ᵀ` probe Linear (a `[1, 64, 1, 64]` 1×1 conv) with weights stored as
/// `format`, its input, and the CPU result.
fn probe_case(format: QuantFormat) -> (Graph, Weights, Tensor, Vec<f32>) {
    const N: usize = 64;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let w: Vec<f32> = (0..N * N).map(|_| next() * 0.125).collect();
    let x: Vec<f32> = (0..N * N).map(|_| next()).collect();
    let mut g = Graph::new();
    let xv = g.input("x", vec![N, N], DType::F32);
    let (wt, dtype) = match format {
        QuantFormat::FP32 => (Tensor::F32(w), DType::F32),
        f => {
            let bytes = crate::model::layers::encode_rows(&w, N, f).expect("probe weight encodes");
            (
                Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                    f, bytes, N, N,
                )),
                DType::Quant(f),
            )
        }
    };
    let wv = g.weight("w", vec![N, N], dtype);
    let y = g.add("y", Op::Linear, &[xv, wv], vec![N, N], DType::F32);
    g.output(y);
    let wd = wt.to_f32();
    let mut want = vec![0.0f32; N * N];
    for r in 0..N {
        for o in 0..N {
            want[r * N + o] = (0..N).map(|k| x[r * N + k] * wd[o * N + k]).sum();
        }
    }
    let mut weights = Weights::new();
    weights.insert("w", wt);
    (g, weights, Tensor::F32(x), want)
}

/// The three probe forms: the weight format and the `AneInfo` that makes [`lower`] use the form.
fn probe_forms() -> [(QuantFormat, AneInfo); 3] {
    let only = |int8: bool, int4: bool| AneInfo {
        cores: 0,
        arch: String::new(),
        eval: EvalEntry::InMemory,
        fp16: Ok(0.0),
        int8: if int8 { Ok(0.0) } else { Err(String::new()) },
        int4: if int4 { Ok(0.0) } else { Err(String::new()) },
        compile_ms: 0.0,
        eval_us: 0.0,
    };
    [
        (QuantFormat::FP32, only(false, false)),
        (QuantFormat::GGUF_Q8_0, only(true, false)),
        (QuantFormat::GGUF_Q4_0, only(false, true)),
    ]
}

/// FNV-1a 64 over `parts` (each length-prefixed): a hash stable across builds and toolchains.
pub(crate) fn fnv64(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        for &b in (p.len() as u64).to_le_bytes().iter().chain(p.iter()) {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// `(hasANE, cores, architecture)` from `_ANEDeviceInfo` (no compile).
fn device_info() -> Result<(bool, u32, String), String> {
    runtime::framework()?;
    objc::rc::autoreleasepool(|| unsafe {
        let di = runtime::class("_ANEDeviceInfo")?;
        let has: BOOL = msg_send![di, hasANE];
        let cores: u32 = msg_send![di, numANECores];
        let arch: runtime::Id = msg_send![di, aneArchitectureType];
        Ok((has != NO, cores, runtime::string(arch)))
    })
}

/// The probe result from `<cache>/host/ane-<os build>-<key>.json`, keyed by the chip, the ANE
/// architecture and the probe programs, or measured and stored there.
fn probe_cached() -> Result<AneInfo, String> {
    let (_, _, arch) = device_info()?;
    let os = crate::hardware::sysctl::string(c"kern.osversion").unwrap_or_default();
    let chip = crate::hardware::sysctl::string(c"machdep.cpu.brand_string").unwrap_or_default();
    let mut parts: Vec<Vec<u8>> = vec![chip.into_bytes(), arch.into_bytes()];
    for (f, info) in probe_forms() {
        let (g, w, _, _) = probe_case(f);
        let (text, blob, _) = lower(&g, &w, &info)?;
        parts.extend([text.into_bytes(), blob]);
    }
    let key = fnv64(&parts.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let path = crate::hardware::default_cache_dir()?
        .join("host")
        .join(format!("ane-{os}-{key:016x}.json"));
    if let Some(info) = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice::<AneInfo>(&b).ok())
    {
        return Ok(info);
    }
    let info = measure()?;
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_vec_pretty(&info) {
        let _ = std::fs::write(&path, json);
    }
    Ok(info)
}

/// Reads the device facts, then compiles and evaluates the probe in its three weight forms; fp16
/// picks the first eval entry point that works.
fn measure() -> Result<AneInfo, String> {
    let (has, cores, arch) = device_info()?;
    if !has {
        return Err("no Apple Neural Engine on this host".into());
    }
    let [fp16, int8, int4] = probe_forms();
    type Probe = (AneProgram, IoLayout, Tensor, Vec<f32>);
    let compile = |(f, info): &(QuantFormat, AneInfo)| -> Result<Probe, String> {
        let (g, w, x, want) = probe_case(*f);
        let (text, blob, io) = lower(&g, &w, info)?;
        let prog = AneProgram::compile(
            &text,
            &blob,
            io.inputs.len(),
            io.in_bytes(),
            io.outputs.len(),
            io.out_bytes(),
        )?;
        Ok((prog, io, x, want))
    };
    let check = |prog: &mut AneProgram,
                 io: &IoLayout,
                 x: &Tensor,
                 want: &[f32],
                 entry|
     -> Result<f32, String> {
        prog.input(0).write(|d| io.inputs[0].pack(&x.to_f32(), d))?;
        prog.eval_with(entry)?;
        let got = prog.output(0).read(|s| io.outputs[0].unpack(s))?;
        let err = rel_rms(&got, want);
        if err <= PROBE_TOL {
            Ok(err)
        } else {
            Err(format!("relative RMS error {err:.2e} against the CPU"))
        }
    };
    let t0 = std::time::Instant::now();
    let (mut prog, io, x, want) = compile(&fp16)?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut tried = Vec::new();
    let mut picked = None;
    for entry in [EvalEntry::RealTime, EvalEntry::Direct, EvalEntry::InMemory] {
        match check(&mut prog, &io, &x, &want, entry) {
            Ok(err) => {
                picked = Some((entry, err));
                break;
            }
            Err(e) => tried.push(e),
        }
    }
    let (eval, err) =
        picked.ok_or_else(|| format!("ANE fp16 probe failed: {}", tried.join("; ")))?;
    const EVALS: u32 = 20;
    let t1 = std::time::Instant::now();
    for _ in 0..EVALS {
        prog.eval_with(eval)?;
    }
    let eval_us = t1.elapsed().as_secs_f64() * 1e6 / EVALS as f64;
    drop(prog);
    let form = |case: &(QuantFormat, AneInfo)| {
        let (mut p, io, x, want) = compile(case)?;
        check(&mut p, &io, &x, &want, eval)
    };
    Ok(AneInfo {
        cores,
        arch,
        eval,
        fp16: Ok(err),
        int8: form(&int8),
        int4: form(&int4),
        compile_ms,
        eval_us,
    })
}

#[cfg(test)]
mod tests;
