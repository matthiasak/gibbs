//! Dynamic hardware analyzer and hardware profile representation.
//!
//! Inspects the host machine dynamically at startup across Apple Metal and multi-core CPU. Guides megakernel JIT compilation, tile sizing,
//! and device placement.

use crate::hardware::{Device, HardwareCapability};

/// Complete architectural profile of a compute device discovered dynamically at runtime.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct HardwareProfile {
    pub device: Device,
    pub name: String,
    pub memory_bandwidth_gbps: f64,
    pub threadgroup_memory_bytes: usize,
    pub max_threads_per_threadgroup: usize,
    pub simd_width: usize,
    pub m_tile: usize,
    pub n_tile: usize,
    pub k_tile: usize,
    pub vector_width_bits: usize,
    pub compute_units: usize,
    pub has_unified_memory: bool,
    pub compute_capability: Option<(i32, i32)>,
    pub max_systolic_tile: Option<[usize; 4]>,
}

impl HardwareProfile {
    /// Detects the hardware profile for the system default compute device.
    pub fn detect() -> Self {
        Self::for_device(crate::hardware::default_device())
    }

    /// Dynamically inspects the specified compute device on the host.
    pub fn for_device(device: Device) -> Self {
        match device {
            Device::Metal => Self::detect_metal(),
            Device::Ane => Self::detect_ane(),
            Device::Cpu => Self::detect_cpu(),
            Device::Cuda => Self {
                device: Device::Cuda,
                name: crate::graph::cuda::device_name(),
                memory_bandwidth_gbps: 900.0,
                threadgroup_memory_bytes: 48 * 1024,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                m_tile: 1,
                n_tile: 32,
                k_tile: 32,
                vector_width_bits: 128,
                compute_units: 80,
                has_unified_memory: false,
                compute_capability: Some((8, 0)),
                max_systolic_tile: Some([16, 16, 16, 1]),
            },
            Device::Rocm => Self {
                device: Device::Rocm,
                name: crate::graph::rocm::device_name(),
                memory_bandwidth_gbps: 1600.0,
                threadgroup_memory_bytes: 64 * 1024,
                max_threads_per_threadgroup: 1024,
                simd_width: 64,
                m_tile: 1,
                n_tile: 64,
                k_tile: 32,
                vector_width_bits: 128,
                compute_units: 110,
                has_unified_memory: false,
                compute_capability: None,
                max_systolic_tile: Some([16, 16, 16, 1]),
            },
            Device::Xla => Self {
                device: Device::Xla,
                name: crate::graph::xla::device_name(),
                memory_bandwidth_gbps: 1200.0,
                threadgroup_memory_bytes: 64 * 1024,
                max_threads_per_threadgroup: 1024,
                simd_width: 128,
                m_tile: 8,
                n_tile: 128,
                k_tile: 128,
                vector_width_bits: 256,
                compute_units: 64,
                has_unified_memory: false,
                compute_capability: None,
                max_systolic_tile: Some([128, 128, 1, 1]),
            },
            Device::IntelXpu => Self {
                device: Device::IntelXpu,
                name: crate::graph::intel_xpu::device_name(),
                memory_bandwidth_gbps: 560.0,
                threadgroup_memory_bytes: 64 * 1024,
                max_threads_per_threadgroup: 1024,
                simd_width: 16,
                m_tile: 1,
                n_tile: 16,
                k_tile: 32,
                vector_width_bits: 512,
                compute_units: 32,
                has_unified_memory: false,
                compute_capability: None,
                max_systolic_tile: Some([8, 16, 16, 1]),
            },
            Device::Extropic => Self {
                device: Device::Extropic,
                name: crate::graph::extropic::device_name(),
                memory_bandwidth_gbps: 512.0,
                threadgroup_memory_bytes: 32 * 1024,
                max_threads_per_threadgroup: 256,
                simd_width: 32,
                m_tile: 1,
                n_tile: 32,
                k_tile: 32,
                vector_width_bits: 128,
                compute_units: 16,
                has_unified_memory: true,
                compute_capability: None,
                max_systolic_tile: None,
            },
        }
    }

    fn detect_metal() -> Self {
        #[cfg(target_os = "macos")]
        {
            if let Some(dev) = metal::Device::system_default() {
                let name = dev.name().to_string();
                let has_unified_memory = dev.has_unified_memory();
                let max_threads_dim = dev.max_threads_per_threadgroup();
                let max_threads_per_threadgroup =
                    (max_threads_dim.width * max_threads_dim.height * max_threads_dim.depth)
                        as usize;

                // Query cached host profile bandwidth if available
                let cached_bw = read_cached_metal_bandwidth(&name);
                let memory_bandwidth_gbps =
                    cached_bw.unwrap_or_else(|| estimate_metal_bandwidth(&name));

                // Apple Silicon GPUs feature 32 KB threadgroup memory and 32-wide SIMD
                let threadgroup_memory_bytes = 32 * 1024;
                let simd_width = 32;

                // Optimal GEMV tile sizes for single-batch decoding
                let m_tile = 1;
                let n_tile = 32;
                let k_tile = 32;

                return Self {
                    device: Device::Metal,
                    name,
                    memory_bandwidth_gbps,
                    threadgroup_memory_bytes,
                    max_threads_per_threadgroup: max_threads_per_threadgroup.min(1024),
                    simd_width,
                    m_tile,
                    n_tile,
                    k_tile,
                    vector_width_bits: 128,
                    compute_units: 32, // Apple Silicon GPU cores typical baseline
                    has_unified_memory,
                    compute_capability: None,
                    max_systolic_tile: None,
                };
            }
        }

        // Fallback if not macOS or Metal device unavailable
        Self {
            device: Device::Metal,
            name: "Metal Generic".to_string(),
            memory_bandwidth_gbps: 250.0,
            threadgroup_memory_bytes: 32 * 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            m_tile: 1,
            n_tile: 32,
            k_tile: 32,
            vector_width_bits: 128,
            compute_units: 16,
            has_unified_memory: true,
            compute_capability: None,
            max_systolic_tile: None,
        }
    }

    /// The Neural Engine shares the GPU's unified memory; cores come from the cached ANE probe.
    fn detect_ane() -> Self {
        #[cfg(target_os = "macos")]
        let probed = crate::graph::ane::probe().ok().map(|i| {
            (
                format!("Apple Neural Engine ({}-core {})", i.cores, i.arch),
                i.cores as usize,
            )
        });
        #[cfg(not(target_os = "macos"))]
        let probed: Option<(String, usize)> = None;
        let (name, compute_units) =
            probed.unwrap_or_else(|| ("Apple Neural Engine (unavailable)".to_string(), 0));
        Self {
            device: Device::Ane,
            name,
            compute_units,
            max_systolic_tile: None,
            ..Self::detect_metal()
        }
    }

    fn detect_cpu() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        let hw = HardwareCapability::probe_system();
        let vector_width_bits = if hw.cpu_has_avx512_vnni {
            512
        } else if hw.cpu_has_arm_neon {
            128
        } else {
            256
        };

        let simd_width = vector_width_bits / 32;
        let mem_bandwidth_gbps = if hw.has_unified_memory {
            hw.total_memory_bytes
                .map(|b| (b as f64 / 1e9) * 4.0)
                .unwrap_or(80.0)
                .clamp(40.0, 400.0)
        } else {
            50.0
        };

        Self {
            device: Device::Cpu,
            name: hw.cpu_name.unwrap_or_else(|| "Host CPU".into()),
            memory_bandwidth_gbps: mem_bandwidth_gbps,
            threadgroup_memory_bytes: 64 * 1024,
            max_threads_per_threadgroup: cores,
            simd_width,
            m_tile: 1,
            n_tile: simd_width * 4,
            k_tile: 32,
            vector_width_bits,
            compute_units: cores,
            has_unified_memory: hw.has_unified_memory,
            compute_capability: None,
            max_systolic_tile: None,
        }
    }
}

/// Estimates Apple Silicon memory bandwidth based on chip identifier.
fn estimate_metal_bandwidth(name: &str) -> f64 {
    let lower = name.to_lowercase();
    if lower.contains("ultra") {
        800.0
    } else if lower.contains("max") {
        400.0
    } else if lower.contains("pro") {
        200.0
    } else if lower.contains("m1")
        || lower.contains("m2")
        || lower.contains("m3")
        || lower.contains("m4")
    {
        100.0
    } else {
        250.0
    }
}

/// Reads cached memory bandwidth from ~/.cache/gibbs/host/*.json if available.
fn read_cached_metal_bandwidth(device_name: &str) -> Option<f64> {
    let cache_dir = crate::hardware::default_cache_dir().ok()?;
    let host_dir = cache_dir.join("host");
    if !host_dir.exists() {
        return None;
    }
    let entries = std::fs::read_dir(host_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(bytes) = std::fs::read(&path) {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    if json.get("device").and_then(|s| s.as_str()) == Some(device_name) {
                        if let Some(bw) = json.get("memory_bandwidth_gbps").and_then(|v| v.as_f64())
                        {
                            if bw > 0.0 {
                                return Some(bw);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile_detection() {
        let profile = HardwareProfile::detect();
        assert!(profile.memory_bandwidth_gbps > 0.0);
        assert!(profile.simd_width > 0);
        assert!(profile.m_tile > 0);
        assert!(profile.n_tile > 0);
        assert!(profile.k_tile > 0);

        let cpu_prof = HardwareProfile::for_device(Device::Cpu);
        assert_eq!(cpu_prof.device, Device::Cpu);
        assert!(cpu_prof.compute_units > 0);

        let metal_prof = HardwareProfile::for_device(Device::Metal);
        assert_eq!(metal_prof.device, Device::Metal);
        assert_eq!(metal_prof.simd_width, 32);
    }
}
