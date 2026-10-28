use std::fmt;
use std::path::PathBuf;
pub mod cpu_compute;
pub mod profile;
pub use profile::HardwareProfile;

/// `$HOME/.cache/gibbs`: compiled kernels, host profiles, and per-model weight-format choices.
pub fn default_cache_dir() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set; cannot place the engine cache")?;
    Ok(PathBuf::from(home).join(".cache").join("gibbs"))
}

/// Runs `f` over the JSON cache `<cache>/host/<file>`, loaded into `slot` on first use (a missing
/// or unreadable file is the default); `store` writes it back.
pub(crate) fn with_host_json<T, R>(
    slot: &std::sync::Mutex<Option<T>>,
    file: &str,
    store: bool,
    f: impl FnOnce(&mut T) -> R,
) -> R
where
    T: serde::Serialize + serde::de::DeserializeOwned + Default,
{
    let path = default_cache_dir().ok().map(|d| d.join("host").join(file));
    let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
    let v = slot.get_or_insert_with(|| {
        path.as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    });
    let r = f(v);
    if let (true, Some(path)) = (store, path) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_vec_pretty(v) {
            let _ = std::fs::write(path, json);
        }
    }
    r
}

/// Exclusive advisory lock (`<cache>/gpu.lock`, flock) serializing benchmarks on the GPU.
/// Blocks (after saying so) while another benchmark holds it; released when the `File` drops.
pub fn gpu_lock() -> Result<std::fs::File, String> {
    let dir = default_cache_dir()?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("gpu.lock");
    let f = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    match f.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            if !crate::telemetry::TUI_OWNS_TERMINAL.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!("gpu lock: waiting for another benchmark");
            }
            f.lock().map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Err(std::fs::TryLockError::Error(e)) => return Err(format!("{}: {e}", path.display())),
    }
    Ok(f)
}

/// Compute device for a pipeline or graph node, selected with `--backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Device {
    Cpu,
    /// Apple GPU through Metal.
    Metal,
    /// Apple Neural Engine through the private `AppleNeuralEngine.framework` (in-memory MIL).
    Ane,
    /// NVIDIA GPU through CUDA Driver + NVRTC (`libcuda` / `libnvrtc`).
    Cuda,
    /// AMD GPU through ROCm HIP + hipRTC (`libamdhip64` / `libhiprtc`).
    Rocm,
    /// XLA PJRT device (Cloud TPU / GPU / CPU PJRT plugin via StableHLO MLIR).
    Xla,
    /// Intel Arc / Data Center Max XPU through OpenCL / Level Zero (`libOpenCL` / `libze_loader`).
    IntelXpu,
}

impl Device {
    /// The `--backend` spelling.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
            Device::Ane => "ane",
            Device::Cuda => "cuda",
            Device::Rocm => "rocm",
            Device::Xla => "xla",
            Device::IntelXpu => "intel-xpu",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "metal" => Ok(Device::Metal),
            "ane" | "neural-engine" => Ok(Device::Ane),
            "cuda" | "nvidia" => Ok(Device::Cuda),
            "rocm" | "hip" | "amd" => Ok(Device::Rocm),
            "xla" | "tpu" | "stablehlo" | "pjrt" => Ok(Device::Xla),
            "intel-xpu" | "xpu" | "intel_xpu" | "sycl" | "opencl" => Ok(Device::IntelXpu),
            other => Err(format!(
                "unknown --backend '{other}' (expected auto, cpu, metal, ane, cuda, rocm, xla, or intel-xpu)"
            )),
        }
    }
}

/// Model and graph placement mode, selected with `--backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PlacementMode {
    #[default]
    Auto,
    Pin(Device),
}

impl PlacementMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Pin(d) => d.name(),
        }
    }

    pub fn target_device(self) -> Option<Device> {
        match self {
            Self::Auto => None,
            Self::Pin(d) => Some(d),
        }
    }
}
impl From<Device> for PlacementMode {
    fn from(d: Device) -> Self {
        PlacementMode::Pin(d)
    }
}

impl fmt::Display for PlacementMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for PlacementMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("auto") {
            Ok(PlacementMode::Auto)
        } else {
            let d = Device::from_str(s)?;
            Ok(PlacementMode::Pin(d))
        }
    }
}

/// Returns hardware devices available at runtime.
pub fn available_devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    #[cfg(target_os = "macos")]
    {
        if metal::Device::system_default().is_some() {
            devices.push(Device::Metal);
        }
        // The ANE counts only once its fp16 probe program compiled and matched the CPU.
        match crate::graph::ane::probe() {
            Ok(info) if info.fp16.is_ok() => devices.push(Device::Ane),
            Ok(info) => log_ane_once(&format!("fp16 probe failed: {:?}", info.fp16)),
            Err(e) => log_ane_once(&e),
        }
    }
    if crate::graph::cuda::is_available() {
        devices.push(Device::Cuda);
    }
    if crate::graph::rocm::is_available() {
        devices.push(Device::Rocm);
    }
    if crate::graph::xla::is_available() {
        devices.push(Device::Xla);
    }
    if crate::graph::intel_xpu::is_available() {
        devices.push(Device::IntelXpu);
    }
    devices
}

/// Logs once per process why the ANE is not used (`auto` then runs Metal only).
#[cfg(target_os = "macos")]
fn log_ane_once(why: &str) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| eprintln!("ANE unavailable ({why}); placement uses Metal only"));
}

/// The default high-performance compute device on this machine.
pub fn default_device() -> Device {
    #[cfg(target_os = "macos")]
    {
        if metal::Device::system_default().is_some() {
            return Device::Metal;
        }
    }
    Device::Cpu
}

/// Current resident memory (RSS) of this process in bytes.
pub fn current_resident_memory_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_out_cnt: *mut u32,
            ) -> i32;
        }
        const MACH_TASK_BASIC_INFO: u32 = 20;
        let mut info = std::mem::MaybeUninit::<MachTaskBasicInfo>::uninit();
        let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32;
        unsafe {
            let kr = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            );
            if kr == 0 {
                return info.assume_init().resident_size as usize;
            }
        }
    }
    0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    NVFP4,
    NVFP4_E2M1,
    MXFP4,
    MXFP6,
    AWQ_W4A16,
    GPTQ_W4A16,
    GGUF_Q4_0,
    GGUF_Q4_1,
    GGUF_Q4_K_M,
    GGUF_Q5_0,
    GGUF_Q5_1,
    GGUF_Q5_K_M,
    GGUF_Q6_K,
    GGUF_Q8_0,
    GGUF_IQ4_XS,
    GGUF_IQ4_NL,
    GGUF_Q2_K,
    GGUF_Q3_K,
    GGUF_IQ3_XXS,
    GGUF_IQ3_S,
    GGUF_IQ2_XXS,
    GGUF_IQ2_XS,
    GGUF_IQ2_S,
    GGUF_IQ2_M,
    GGUF_IQ1_S,
    GGUF_IQ1_M,
    GGUF_TQ1_0,
    GGUF_TQ2_0,
    BitNet_B1_58,
    KIVI_2Bit,
    TurboQuant,
    /// bitsandbytes NF4/FP4 source tags; the GPU codecs are the blocksize-qualified `BNB_*_B*`.
    BNB_NF4,
    BNB_FP4,
    /// Repacked bitsandbytes 4-bit: per block of 64/128/256 elements, nibbles + f32 absmax.
    BNB_NF4_B64,
    BNB_NF4_B128,
    BNB_NF4_B256,
    BNB_FP4_B64,
    BNB_FP4_B128,
    BNB_FP4_B256,
    /// Repacked FP8: per block of 32 elements, 32 E4M3 bytes + f32 scale.
    FP8_E4M3_B32,
    /// Repacked MLX affine 4-bit: per group, nibbles + f16 scale + f16 bias.
    MLX_AFFINE4_G32,
    MLX_AFFINE4_G64,
    MLX_AFFINE4_G128,
    /// Repacked compressed-tensors int4: per group, nibbles + f16 scale + f16 zero point.
    INT4_G32,
    INT4_G64,
    INT4_G128,
    /// ggml type 40 `block_nvfp4`: per 64 elements, 4 UE4M3 scale bytes (one per 16) + 32 nibble bytes.
    /// Not the `NVFP4`/`NVFP4_E2M1` layout (`Nvfp4Tensor`: f32 scale per 16 elements).
    GGUF_NVFP4,
}
impl QuantFormat {
    pub fn bits_per_weight(&self) -> f32 {
        crate::graph::codec::bits_per_weight(*self)
    }

    pub fn is_smaller_than(&self, other: &QuantFormat) -> bool {
        self.bits_per_weight() < other.bits_per_weight()
    }
    pub fn tensor_bytes(&self, num_elements: usize) -> usize {
        match crate::graph::codec::block_geometry(*self) {
            Some((elems, bytes)) => (num_elements / elems) * bytes,
            None => (num_elements as f32 * self.bits_per_weight() / 8.0).ceil() as usize,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::FP32 => "FP32",
            Self::FP16 => "FP16",
            Self::BF16 => "BF16",
            Self::FP8_E4M3 => "FP8_E4M3",
            Self::FP8_E5M2 => "FP8_E5M2",
            Self::NVFP4 | Self::NVFP4_E2M1 => "NVFP4",
            Self::MXFP4 => "MXFP4",
            Self::MXFP6 => "MXFP6",
            Self::AWQ_W4A16 => "AWQ",
            Self::GPTQ_W4A16 => "GPTQ",
            Self::GGUF_Q4_0 => "Q4_0",
            Self::GGUF_Q4_1 => "Q4_1",
            Self::GGUF_Q4_K_M => "Q4_K_M",
            Self::GGUF_Q5_0 => "Q5_0",
            Self::GGUF_Q5_1 => "Q5_1",
            Self::GGUF_Q5_K_M => "Q5_K_M",
            Self::GGUF_Q6_K => "Q6_K",
            Self::GGUF_Q8_0 => "Q8_0",
            Self::GGUF_IQ4_XS => "IQ4_XS",
            Self::GGUF_IQ4_NL => "IQ4_NL",
            Self::GGUF_Q2_K => "Q2_K",
            Self::GGUF_Q3_K => "Q3_K",
            Self::GGUF_IQ3_XXS => "IQ3_XXS",
            Self::GGUF_IQ3_S => "IQ3_S",
            Self::GGUF_IQ2_XXS => "IQ2_XXS",
            Self::GGUF_IQ2_XS => "IQ2_XS",
            Self::GGUF_IQ2_S => "IQ2_S",
            Self::GGUF_IQ2_M => "IQ2_M",
            Self::GGUF_IQ1_S => "IQ1_S",
            Self::GGUF_IQ1_M => "IQ1_M",
            Self::GGUF_TQ1_0 => "TQ1_0",
            Self::GGUF_TQ2_0 => "TQ2_0",
            Self::BitNet_B1_58 => "BitNet_1.58",
            Self::KIVI_2Bit => "KIVI_2Bit",
            Self::TurboQuant => "TurboQuant",
            Self::BNB_NF4 => "BNB_NF4",
            Self::BNB_FP4 => "BNB_FP4",
            Self::BNB_NF4_B64 => "BNB_NF4_B64",
            Self::BNB_NF4_B128 => "BNB_NF4_B128",
            Self::BNB_NF4_B256 => "BNB_NF4_B256",
            Self::BNB_FP4_B64 => "BNB_FP4_B64",
            Self::BNB_FP4_B128 => "BNB_FP4_B128",
            Self::BNB_FP4_B256 => "BNB_FP4_B256",
            Self::FP8_E4M3_B32 => "FP8_E4M3_B32",
            Self::MLX_AFFINE4_G32 => "MLX_AFFINE4_G32",
            Self::MLX_AFFINE4_G64 => "MLX_AFFINE4_G64",
            Self::MLX_AFFINE4_G128 => "MLX_AFFINE4_G128",
            Self::INT4_G32 => "INT4_G32",
            Self::INT4_G64 => "INT4_G64",
            Self::INT4_G128 => "INT4_G128",
            Self::GGUF_NVFP4 => "GGUF_NVFP4",
        }
    }

    pub fn from_str_lenient(s: &str) -> Option<Self> {
        let clean = s.trim().to_lowercase().replace('-', "_");
        match clean.as_str() {
            "nvfp4" | "nvfp4_e2m1" | "fp4" => Some(Self::NVFP4_E2M1),
            "q4_k_m" | "q4_k" | "gguf_q4_k_m" => Some(Self::GGUF_Q4_K_M),
            "q4_0" | "gguf_q4_0" => Some(Self::GGUF_Q4_0),
            "q4_1" | "gguf_q4_1" => Some(Self::GGUF_Q4_1),
            "q5_0" | "gguf_q5_0" => Some(Self::GGUF_Q5_0),
            "q5_1" | "gguf_q5_1" => Some(Self::GGUF_Q5_1),
            "q5_k_m" | "q5_k" | "gguf_q5_k_m" => Some(Self::GGUF_Q5_K_M),
            "q6_k" | "gguf_q6_k" => Some(Self::GGUF_Q6_K),
            "q8_0" | "q8" | "gguf_q8_0" => Some(Self::GGUF_Q8_0),
            "bf16" | "bfloat16" => Some(Self::BF16),
            "fp16" | "f16" => Some(Self::FP16),
            "fp32" | "f32" | "uncompressed" => Some(Self::FP32),
            "fp8" | "fp8_e4m3" => Some(Self::FP8_E4M3),
            "fp8_e5m2" => Some(Self::FP8_E5M2),
            "iq4_xs" | "gguf_iq4_xs" => Some(Self::GGUF_IQ4_XS),
            "iq4_nl" | "gguf_iq4_nl" => Some(Self::GGUF_IQ4_NL),
            "q2_k" | "gguf_q2_k" => Some(Self::GGUF_Q2_K),
            "q3_k" | "gguf_q3_k" => Some(Self::GGUF_Q3_K),
            "iq3_xxs" | "gguf_iq3_xxs" => Some(Self::GGUF_IQ3_XXS),
            "iq3_s" | "gguf_iq3_s" => Some(Self::GGUF_IQ3_S),
            "iq2_xxs" | "gguf_iq2_xxs" => Some(Self::GGUF_IQ2_XXS),
            "iq2_xs" | "gguf_iq2_xs" => Some(Self::GGUF_IQ2_XS),
            "iq2_s" | "gguf_iq2_s" => Some(Self::GGUF_IQ2_S),
            "iq2_m" | "gguf_iq2_m" => Some(Self::GGUF_IQ2_M),
            "iq1_s" | "gguf_iq1_s" => Some(Self::GGUF_IQ1_S),
            "iq1_m" | "gguf_iq1_m" => Some(Self::GGUF_IQ1_M),
            "tq1_0" | "gguf_tq1_0" => Some(Self::GGUF_TQ1_0),
            "tq2_0" | "gguf_tq2_0" => Some(Self::GGUF_TQ2_0),
            "mxfp4" => Some(Self::MXFP4),
            "gguf_nvfp4" => Some(Self::GGUF_NVFP4),
            "kivi" | "kivi_2bit" => Some(Self::KIVI_2Bit),
            "turbo" | "turboquant" => Some(Self::TurboQuant),
            _ => None,
        }
    }
}

/// Hardware facts detected at runtime. Anything the platform cannot report is `None`/`false`, never guessed.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HardwareCapability {
    /// `machdep.cpu.brand_string` (macOS).
    pub cpu_name: Option<String>,
    /// `hw.memsize` (macOS).
    pub total_memory_bytes: Option<u64>,
    /// `hw.perflevel0.physicalcpu` (macOS, Apple Silicon).
    pub performance_cores: Option<u32>,
    /// `hw.perflevel1.physicalcpu` (macOS, Apple Silicon).
    pub efficiency_cores: Option<u32>,
    /// System GPU name from the Metal API (macOS).
    pub gpu_name: Option<String>,
    /// The Metal GPU shares memory with the CPU.
    pub has_unified_memory: bool,
    /// Advanced SIMD is architecturally mandatory on AArch64.
    pub cpu_has_arm_neon: bool,
    /// `hw.optional.arm.FEAT_*` runtime flags (macOS).
    pub cpu_has_arm_sme: bool,
    pub cpu_has_arm_sme2: bool,
    pub cpu_has_arm_i8mm: bool,
    pub cpu_has_arm_bf16: bool,
    pub cpu_has_arm_fp16: bool,
    /// `is_x86_feature_detected!` (x86_64).
    pub cpu_has_avx512_vnni: bool,
    pub cpu_has_amx: bool,
}

impl HardwareCapability {
    pub fn probe_system() -> Self {
        // Advanced SIMD is architecturally mandatory on AArch64.
        #[allow(unused_mut)] // nothing else is detected on non-macOS AArch64
        let mut hw = Self {
            cpu_has_arm_neon: cfg!(target_arch = "aarch64"),
            ..Self::default()
        };
        #[cfg(target_arch = "x86_64")]
        {
            hw.cpu_has_avx512_vnni = is_x86_feature_detected!("avx512vnni");
            hw.cpu_has_amx = is_x86_feature_detected!("amx-tile");
        }
        #[cfg(target_os = "macos")]
        {
            let feat = |name: &std::ffi::CStr| sysctl::int(name) == Some(1);
            hw.cpu_name = sysctl::string(c"machdep.cpu.brand_string");
            hw.total_memory_bytes = sysctl::int(c"hw.memsize");
            hw.performance_cores = sysctl::int(c"hw.perflevel0.physicalcpu").map(|n| n as u32);
            hw.efficiency_cores = sysctl::int(c"hw.perflevel1.physicalcpu").map(|n| n as u32);
            hw.cpu_has_arm_sme = feat(c"hw.optional.arm.FEAT_SME");
            hw.cpu_has_arm_sme2 = feat(c"hw.optional.arm.FEAT_SME2");
            hw.cpu_has_arm_i8mm = feat(c"hw.optional.arm.FEAT_I8MM");
            hw.cpu_has_arm_bf16 = feat(c"hw.optional.arm.FEAT_BF16");
            hw.cpu_has_arm_fp16 = feat(c"hw.optional.arm.FEAT_FP16");
            hw.gpu_name = crate::backend::metal::system_device_name().ok();
            hw.has_unified_memory =
                crate::backend::metal::system_has_unified_memory().unwrap_or(false);
        }
        hw
    }

    /// Detected values for the startup banner and TUI; undetected values say so.
    pub fn summary(&self) -> Vec<String> {
        const ND: &str = "not detected";
        let cores = match (self.performance_cores, self.efficiency_cores) {
            (Some(p), Some(e)) => format!(" ({p} performance + {e} efficiency cores)"),
            _ => String::new(),
        };
        let gpu = match &self.gpu_name {
            Some(n) if self.has_unified_memory => format!("{n} (unified memory)"),
            Some(n) => n.clone(),
            None => ND.into(),
        };
        let features: Vec<&str> = [
            (self.cpu_has_arm_neon, "NEON"),
            (self.cpu_has_arm_sme, "SME"),
            (self.cpu_has_arm_sme2, "SME2"),
            (self.cpu_has_arm_i8mm, "I8MM"),
            (self.cpu_has_arm_bf16, "BF16"),
            (self.cpu_has_arm_fp16, "FP16"),
            (self.cpu_has_avx512_vnni, "AVX512-VNNI"),
            (self.cpu_has_amx, "AMX"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect();
        vec![
            format!("CPU:    {}{cores}", self.cpu_name.as_deref().unwrap_or(ND)),
            format!(
                "Memory: {}",
                self.total_memory_bytes
                    .map_or(ND.into(), |b| format!("{:.1} GB", b as f64 / 1e9))
            ),
            format!("GPU:    {gpu}"),
            format!(
                "CPU features: {}",
                if features.is_empty() {
                    "none detected".into()
                } else {
                    features.join(" ")
                }
            ),
        ]
    }
}

/// `sysctlbyname(3)` reads (libSystem; no libc crate).
#[cfg(target_os = "macos")]
pub(crate) mod sysctl {
    use std::ffi::{c_char, c_int, c_void, CStr};

    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// Reads `name` into `buf`; `None` if the key does not exist (or does not fit).
    fn read(name: &CStr, buf: &mut [u8]) -> Option<usize> {
        let mut len = buf.len();
        // SAFETY: `buf` is writable for `len` bytes; sysctlbyname writes at most `len` and updates it.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                buf.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(len)
    }

    /// A 4- or 8-byte integer key.
    pub fn int(name: &CStr) -> Option<u64> {
        let mut buf = [0u8; 8];
        match read(name, &mut buf)? {
            4 => Some(u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]) as u64),
            8 => Some(u64::from_ne_bytes(buf)),
            _ => None,
        }
    }

    /// A NUL-terminated string key.
    pub fn string(name: &CStr) -> Option<String> {
        let mut buf = [0u8; 256];
        let len = read(name, &mut buf)?;
        let s = CStr::from_bytes_until_nul(&buf[..len])
            .ok()?
            .to_string_lossy();
        (!s.is_empty()).then(|| s.into_owned())
    }
}
