//! GPU-only enforcement: the avenue ladder every node must climb before it may leave the GPU.
//!
//! `explore_node` tries the avenues in order and stops at the first that really has a code path.
//! A graph with an unlowered node fails to place with a [`GpuReport`] naming every avenue's reason.

use crate::graph::{DType, Op};
use crate::hardware::Device;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avenue {
    DeviceKernel,
    DtypeAdapt,
}

impl Avenue {
    pub const ALL: [Avenue; 2] = [Avenue::DeviceKernel, Avenue::DtypeAdapt];

    pub fn name(self) -> &'static str {
        match self {
            Avenue::DeviceKernel => "device-kernel",
            Avenue::DtypeAdapt => "dtype-adapt",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Lowered(&'static str),
    Rejected(String),
}

#[derive(Clone, Debug)]
pub struct Attempt {
    pub avenue: Avenue,
    pub verdict: Verdict,
}

#[derive(Clone, Debug)]
pub struct NodeReport {
    pub node: String,
    pub op: String,
    pub dtype: DType,
    pub attempts: Vec<Attempt>,
}

impl NodeReport {
    /// The avenue that lowered the node, if any.
    pub fn lowered_by(&self) -> Option<(Avenue, &'static str)> {
        self.attempts.iter().find_map(|a| match a.verdict {
            Verdict::Lowered(how) => Some((a.avenue, how)),
            Verdict::Rejected(_) => None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct GpuReport {
    /// The device the nodes were placed on.
    pub device: Device,
    pub total_nodes: usize,
    pub unlowered: Vec<NodeReport>,
}

impl GpuReport {
    pub fn error(&self) -> String {
        let mut s = format!(
            "gpu-only: {} of {} graph nodes have no {:?} lowering\n",
            self.unlowered.len(),
            self.total_nodes,
            self.device
        );
        for n in &self.unlowered {
            s.push_str(&format!("  {} [{} {:?}]\n", n.node, n.op, n.dtype));
            for a in &n.attempts {
                if let Verdict::Rejected(reason) = &a.verdict {
                    s.push_str(&format!("    ✗ {}: {}\n", a.avenue.name(), reason));
                }
            }
        }
        s.push_str("run with --backend cpu to accept the CPU executor for this model");
        s
    }
}

fn device_kernel(op: &Op, dtype: DType, device: Device) -> Verdict {
    if crate::graph::placement::device_supports_op(device, op, dtype) {
        Verdict::Lowered("hand-written device kernel")
    } else {
        Verdict::Rejected(format!("no {device:?} kernel for {} at {:?}", op.name(), dtype))
    }
}

/// Whether a quantized weight with no Metal codec is dequantized at load (`metal.rs` weight upload).
fn load_time_dequant(op: &Op, dtype: DType) -> bool {
    matches!(op, Op::Linear | Op::Embedding | Op::Megakernel(_)) && matches!(dtype, DType::Quant(_))
}

fn dtype_adapt(op: &Op, dtype: DType, device: Device) -> Verdict {
    if load_time_dequant(op, dtype) {
        return Verdict::Lowered("load-time dequantize/re-encode per policy");
    }
    if matches!(dtype, DType::F16 | DType::BF16)
        && crate::graph::passes::dtype_adapt::can_cast(op, dtype, device)
    {
        return Verdict::Lowered("GPU Cast around the F32 kernel");
    }
    Verdict::Rejected(format!(
        "operand dtype {dtype:?} cannot be cast or re-encoded"
    ))
}

/// Tries the avenues for `device` in order and stops at the first `Lowered`.
pub fn explore_node(name: &str, op: &Op, dtype: DType, device: Device) -> NodeReport {
    let mut attempts = Vec::with_capacity(Avenue::ALL.len());
    for avenue in Avenue::ALL {
        let verdict = match avenue {
            Avenue::DeviceKernel => device_kernel(op, dtype, device),
            Avenue::DtypeAdapt => dtype_adapt(op, dtype, device),
        };
        let done = matches!(verdict, Verdict::Lowered(_));
        attempts.push(Attempt { avenue, verdict });
        if done {
            break;
        }
    }
    NodeReport {
        node: name.to_string(),
        op: format!("{op:?}"),
        dtype,
        attempts,
    }
}

/// The primary device a placement mode runs on: the pinned one (Metal for `ane`, which offloads
/// regions from it), or the default GPU when `Auto`.
pub fn resolved_backend(mode: crate::hardware::PlacementMode) -> Device {
    match mode {
        crate::hardware::PlacementMode::Pin(Device::Ane) => Device::Metal,
        crate::hardware::PlacementMode::Pin(d) => d,
        crate::hardware::PlacementMode::Auto => crate::hardware::default_device(),
    }
}
