//! Portable GPU kernel layer: one kernel source (`kernels/*.gpk`, a C subset) compiled as OpenCL C
//! 1.2, CUDA C++ or HIP C++ ([`dialect`]), and one executor ([`exec::PortableSession`]) over any
//! [`api::GpuApi`] (the CUDA, ROCm and Intel XPU backends).

pub mod api;
pub mod dialect;
pub mod exec;

use crate::graph::{DType, Op};
use crate::hardware::QuantFormat;
use api::{Arg, GpuApi};
use dialect::Dialect;
use std::fmt::LowerHex;
use std::sync::LazyLock;

pub use exec::PortableSession;

const COMMON: &str = include_str!("kernels/common.gpk");
const CODECS: &str = include_str!("kernels/codecs.gpk");
const ELEMENTWISE: &str = include_str!("kernels/elementwise.gpk");
const NORM: &str = include_str!("kernels/norm.gpk");
const ATTENTION: &str = include_str!("kernels/attention.gpk");
const MOVEMENT: &str = include_str!("kernels/movement.gpk");
const SPATIAL: &str = include_str!("kernels/spatial.gpk");
const STATE: &str = include_str!("kernels/state.gpk");
const SELECT: &str = include_str!("kernels/select.gpk");
/// Per-encoding weight kernels: every `CODEC` / `BLOCK_ELEMS` / `BLOCK_BYTES` is substituted once
/// per [`encodings`] entry.
const WEIGHT_TEMPLATE: &str = include_str!("kernels/weights.gpk");

/// Tiny probe kernel: a backend is available only when this compiles and adds correctly.
const PROBE: &str = r#"
KERNEL gpk_probe_add(GLOBAL const float* a, GLOBAL const float* b, GLOBAL float* c, u32 n) {
    u32 i = GID(0);
    if (i < n) c[i] = a[i] + b[i];
}
"#;

/// A weight encoding the per-encoding kernels decode: its kernel-name suffix and block geometry
/// (dense encodings are 8-element blocks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    pub name: String,
    pub block_elems: usize,
    pub block_bytes: usize,
}

/// Every weight encoding: the dense ones (F32, F16, BF16), then every registered block codec.
static ENCODINGS: LazyLock<Vec<Encoding>> = LazyLock::new(|| {
    let dense = [("F32", 32), ("F16", 16), ("BF16", 16)].map(|(n, b)| Encoding {
        name: n.into(),
        block_elems: 8,
        block_bytes: b,
    });
    let blocks = crate::graph::codec::formats().into_iter().filter_map(|f| {
        let c = crate::graph::codec::codec(f)?;
        Some(Encoding {
            name: format!("{f:?}"),
            block_elems: c.block_elems as usize,
            block_bytes: c.block_bytes as usize,
        })
    });
    dense.into_iter().chain(blocks).collect()
});

pub fn encodings() -> &'static [Encoding] {
    &ENCODINGS
}

/// The encoding weights stored as `f` upload in: dense, or its block codec (IQ2_S decodes as
/// IQ2_M, which shares its layout).
pub fn encoding(f: QuantFormat) -> Option<&'static Encoding> {
    let name = match f {
        QuantFormat::FP32 => "F32".to_string(),
        QuantFormat::FP16 => "F16".to_string(),
        QuantFormat::BF16 => "BF16".to_string(),
        QuantFormat::GGUF_IQ2_S => format!("{:?}", QuantFormat::GGUF_IQ2_M),
        f => format!("{f:?}"),
    };
    ENCODINGS.iter().find(|e| e.name == name)
}

fn arr<T: LowerHex>(name: &str, ty: &str, items: &[T], lit: impl Fn(&T) -> String) -> String {
    let body: Vec<String> = items.iter().map(lit).collect();
    format!(
        "CONSTANT {ty} {name}[{}] = {{\n{}\n}};\n",
        items.len(),
        body.join(",")
    )
}

/// The IQ grids, sign and kvalue tables of `src/model/gguf_tables.rs` as program-scope constants.
fn tables() -> String {
    use crate::model::layers::{
        IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
        KVALUES_FP4, KVALUES_IQ4NL,
    };
    let u64h = |v: &u64| format!("{v:#018x}ull");
    let u32h = |v: &u32| format!("{v:#010x}u");
    let dec = |v: &dyn std::fmt::Display| v.to_string();
    [
        arr("KSIGNS_IQ2XS", "u8", &KSIGNS_IQ2XS, |v| dec(v)),
        arr("IQ2XXS_GRID", "u64", &IQ2XXS_GRID, u64h),
        arr("IQ2XS_GRID", "u64", &IQ2XS_GRID, u64h),
        arr("IQ2S_GRID", "u64", &IQ2S_GRID, u64h),
        arr("IQ3XXS_GRID", "u32", &IQ3XXS_GRID, u32h),
        arr("IQ3S_GRID", "u32", &IQ3S_GRID, u32h),
        arr("IQ1S_GRID", "u64", &IQ1S_GRID, u64h),
        arr("KVALUES_IQ4NL", "i8", &KVALUES_IQ4NL, |v| dec(v)),
        arr("KVALUES_FP4", "i8", &KVALUES_FP4, |v| dec(v)),
    ]
    .concat()
}

/// The complete translation unit for `d`: prelude, tables, helpers, codecs and every kernel.
pub fn source(d: Dialect) -> String {
    let mut s = String::from(dialect::prelude(d));
    s += &tables();
    s += COMMON;
    s += CODECS;
    for e in encodings() {
        s += &WEIGHT_TEMPLATE
            .replace("BLOCK_ELEMS", &e.block_elems.to_string())
            .replace("BLOCK_BYTES", &e.block_bytes.to_string())
            .replace("CODEC", &e.name);
    }
    for part in [
        ELEMENTWISE,
        NORM,
        ATTENTION,
        MOVEMENT,
        SPATIAL,
        STATE,
        SELECT,
        PROBE,
    ] {
        s += part;
    }
    s
}

/// Whether the portable executor runs `op` with element type `dtype`: every op, with weights
/// dense or in a block codec the kernels decode.
pub fn supports_op(_op: &Op, dtype: DType) -> bool {
    match dtype {
        DType::Quant(f) => encoding(f).is_some(),
        _ => true,
    }
}

/// Compiles the probe kernel on `api` and checks a vector add of 1000 elements.
pub fn probe(api: &dyn GpuApi) -> Result<(), String> {
    let n = 1000usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 3.0 - i as f32).collect();
    let module = api.compile(&format!("{}{PROBE}", dialect::prelude(api.dialect())))?;
    let k = api.kernel(&module, "gpk_probe_add")?;
    let bytes = n * 4;
    let bufs = [api.alloc(bytes)?, api.alloc(bytes)?, api.alloc(bytes)?];
    let run = || -> Result<Vec<f32>, String> {
        api.upload(bufs[0], 0, exec::as_bytes(&a))?;
        api.upload(bufs[1], 0, exec::as_bytes(&b))?;
        let args = [
            Arg::Buf(bufs[0]),
            Arg::Buf(bufs[1]),
            Arg::Buf(bufs[2]),
            Arg::U32(n as u32),
        ];
        api.launch(k, [n.div_ceil(64) as u32, 1, 1], [64, 1, 1], &args)?;
        let mut c = vec![0f32; n];
        api.download(bufs[2], 0, exec::as_bytes_mut(&mut c))?;
        Ok(c)
    };
    let got = run();
    for b in bufs {
        api.free(b);
    }
    let got = got?;
    match (0..n).find(|&i| got[i] != a[i] + b[i]) {
        Some(i) => Err(format!(
            "{}: probe kernel wrote {} at {i}, expected {}",
            api.device_name(),
            got[i],
            a[i] + b[i]
        )),
        None => Ok(()),
    }
}

/// Device memory bandwidth in GB/s from a timed device-to-device copy, cached in the host profile
/// (`~/.cache/gibbs/host/portable-<dialect>-<device>.json`) so it is measured once per device.
pub fn bandwidth_gbps(api: &dyn GpuApi) -> Option<f64> {
    let name = api.device_name().to_string();
    let file: String = format!("{:?}-{name}", api.dialect())
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let dir = crate::hardware::default_cache_dir().ok()?.join("host");
    let path = dir.join(format!("portable-{file}.json"));
    let cached = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|j| j.get("portable_bandwidth_gbps")?.as_f64());
    if let Some(bw) = cached {
        return Some(bw);
    }
    let bw = exec::measure_copy_bandwidth(api).ok()?;
    let _ = std::fs::create_dir_all(&dir);
    let json = serde_json::json!({"device": name, "portable_bandwidth_gbps": bw});
    let _ = std::fs::write(&path, json.to_string());
    Some(bw)
}

/// Whether `d` runs on the portable executor.
pub fn is_portable(d: crate::hardware::Device) -> bool {
    use crate::hardware::Device;
    matches!(d, Device::Cuda | Device::Rocm | Device::IntelXpu)
}

/// The probed device API of a portable device, `None` when it is not available on this host.
pub fn device_api(d: crate::hardware::Device) -> Option<std::sync::Arc<dyn GpuApi>> {
    use crate::hardware::Device;
    match d {
        Device::Cuda => crate::graph::cuda::api(),
        Device::Rocm => crate::graph::rocm::api(),
        Device::IntelXpu => crate::graph::intel_xpu::api(),
        _ => None,
    }
}
