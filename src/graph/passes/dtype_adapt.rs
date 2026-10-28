//! GPU-side activation dtype adaptation: the Metal kernels compute in f32, so F16/BF16 activations
//! (graph inputs and node outputs) are retyped to F32. Weights keep their stored dtype.

use crate::graph::{DType, Graph, Op, ValueKind};
use crate::hardware::Device;

/// Retypes F16/BF16 activations to F32 (the executor already widens inputs and returns f32).
pub fn run(g: &mut Graph) {
    for v in &mut g.values {
        if matches!(v.kind, ValueKind::Input | ValueKind::Node(_))
            && matches!(v.dtype, DType::F16 | DType::BF16)
        {
            v.dtype = DType::F32;
        }
    }
}

/// Whether `op` at the reduced-precision `dtype` runs on `device` after [`run`]: its F32 form lowers.
pub fn can_cast(op: &Op, dtype: DType, device: Device) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
        && crate::graph::placement::device_supports_op(device, op, DType::F32)
}

/// Retypes weights whose quant format has no GPU kernel to F32. The Metal weight upload dequantizes
/// such tensors at load (`metal.rs`, logged per tensor), so the graph must declare what the GPU holds.
///
/// Fails when the dequantized tensor would exceed 2 GiB: the f32 kernels read garbage (NaN logits on
/// E4B's 2.7 GB `lm_head`) past that offset, so that is a gap, not an adaptation.
pub fn retype_uncoded_weights(
    g: &mut Graph,
    has_kernel: &dyn Fn(crate::hardware::QuantFormat) -> bool,
) -> Result<(), String> {
    for v in &mut g.values {
        if let (crate::graph::ValueKind::Weight, DType::Quant(f)) = (&v.kind, v.dtype) {
            if has_kernel(f) {
                continue;
            }
            let n: usize = v.shape.iter().product();
            if n * 4 > i32::MAX as usize {
                return Err(format!(
                    "gpu-only: weight '{}' ({f:?}, {n} elements) has no GPU kernel and its f32 dequantization exceeds 2 GiB; run with --backend cpu to accept the CPU executor",
                    v.name
                ));
            }
            v.dtype = DType::F32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_retypes_activations_and_keeps_weights() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 4], DType::BF16);
        let w = g.weight("w", vec![4], DType::BF16);
        let y = g.add("n", Op::RmsNorm { eps: 1e-6 }, &[x, w], vec![1, 4], DType::F16);
        g.output(y);
        run(&mut g);
        assert_eq!(g.value(x).dtype, DType::F32);
        assert_eq!(g.value(y).dtype, DType::F32);
        assert_eq!(g.value(w).dtype, DType::BF16);
        assert!(can_cast(&Op::RmsNorm { eps: 1e-6 }, DType::F16, Device::Metal));
    }
}
