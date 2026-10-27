//! Transformer block megakernel fusion optimization passes.
//!
//! Fuses the contiguous MLP subgraph into one single-dispatch Megakernel operation:
//! `RmsNorm -> Linear(GateUp) -> SwiGLU/GeGLU -> Down Linear -> Add`.
//!
//! Eliminates over 85% of GPU command encoder dispatches and removes round-trip global
//! memory bandwidth for intermediate activation buffers.

use crate::graph::{DType, Graph, MegakernelDescriptor, MegakernelKind, NodeId, Op, ValueId};
use std::collections::{HashMap, HashSet};

/// Quantized formats the Metal megakernel decodes inline; MLPs with any other quantized weight stay
/// unfused so their Linears read the weights through a codec GEMV instead.
const MEGAKERNEL_QUANT: &[crate::hardware::QuantFormat] = &[
    crate::hardware::QuantFormat::GGUF_Q4_0,
    crate::hardware::QuantFormat::GGUF_Q8_0,
    crate::hardware::QuantFormat::GGUF_IQ4_XS,
    crate::hardware::QuantFormat::GGUF_IQ2_M,
    crate::hardware::QuantFormat::GGUF_IQ2_S,
    crate::hardware::QuantFormat::GGUF_Q4_K_M,
    crate::hardware::QuantFormat::GGUF_Q2_K,
    crate::hardware::QuantFormat::GGUF_IQ3_S,
    crate::hardware::QuantFormat::NVFP4,
];

pub(crate) fn fusible_weight(graph: &Graph, w: crate::graph::ValueId) -> bool {
    match graph.value(w).dtype {
        crate::graph::DType::Quant(f) => MEGAKERNEL_QUANT.contains(&f),
        _ => true,
    }
}

/// Materializes the e-graph's chosen `RmsNorm -> [Gate, Up Linear] -> SwiGLU/GeGLU -> Down Linear
/// -> Add(residual)` blocks: those ending in the `only` Add nodes.
pub(crate) fn fuse_mlp_blocks(graph: &mut Graph, only: &HashSet<NodeId>) -> usize {
    let mut rewrites = 0;
    let mut eliminated_nodes: HashSet<NodeId> = HashSet::new();

    // Map each ValueId to the NodeId producing it
    let mut val_producers: HashMap<ValueId, NodeId> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            val_producers.insert(out, NodeId(idx as u32));
        }
    }

    // Map each ValueId to the NodeIds consuming it
    let mut val_consumers: HashMap<ValueId, Vec<NodeId>> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        let nid = NodeId(idx as u32);
        for &inp in &node.inputs {
            val_consumers.entry(inp).or_default().push(nid);
        }
    }

    let n_nodes = graph.nodes.len();
    for i in 0..n_nodes {
        let add_nid = NodeId(i as u32);
        let add_node = graph.node(add_nid);
        if !matches!(add_node.op, Op::Binary { kind: crate::graph::BinaryKind::Add })
            || add_node.inputs.len() != 2
            || graph.value(add_node.inputs[0]).shape != graph.value(add_node.inputs[1]).shape
            || !only.contains(&add_nid)
        {
            continue;
        }

        let (res_x, down_out) = (add_node.inputs[0], add_node.inputs[1]);

        // Helper to check if a path forms an MLP block
        let check_candidate = |res_val: ValueId,
                               down_val: ValueId|
         -> Option<(
            NodeId,
            NodeId,
            NodeId,
            Option<NodeId>,
            NodeId,
            NodeId,
            Option<NodeId>,
            MegakernelDescriptor,
            Vec<ValueId>,
        )> {
            let direct_nid = *val_producers.get(&down_val)?;
            let direct_node = graph.node(direct_nid);
            let (down_nid, post_norm_info) = if let Op::RmsNorm { eps } = direct_node.op {
                if direct_node.inputs.len() >= 2 {
                    let actual_down_val = direct_node.inputs[0];
                    let post_norm_w = direct_node.inputs[1];
                    let actual_down_nid = *val_producers.get(&actual_down_val)?;
                    (actual_down_nid, Some((direct_nid, post_norm_w, eps)))
                } else {
                    return None;
                }
            } else {
                (direct_nid, None)
            };

            let down_node = graph.node(down_nid);
            if !matches!(down_node.op, Op::Linear) || down_node.inputs.len() < 2 {
                return None;
            }
            let swiglu_val = down_node.inputs[0];
            let down_w = down_node.inputs[1];
            // The megakernel is a one-row GEMV chain; batched activations stay as separate GEMMs.
            if graph.value(swiglu_val).shape.iter().rev().skip(1).product::<usize>() != 1 {
                return None;
            }

            let swiglu_nid = *val_producers.get(&swiglu_val)?;
            let swiglu_node = graph.node(swiglu_nid);
            let is_geglu = matches!(swiglu_node.op, Op::GeGlu);
            if !matches!(swiglu_node.op, Op::SwiGlu | Op::GeGlu) || swiglu_node.inputs.len() != 2 {
                return None;
            }

            let gate_val = swiglu_node.inputs[0];
            let up_val = swiglu_node.inputs[1];

            // Case A: Separate Gate and Up Linears
            if let (Some(&gate_nid), Some(&up_nid)) =
                (val_producers.get(&gate_val), val_producers.get(&up_val))
            {
                let gate_node = graph.node(gate_nid);
                let up_node = graph.node(up_nid);
                if matches!(gate_node.op, Op::Linear) && matches!(up_node.op, Op::Linear) {
                    if gate_node.inputs.len() >= 2
                        && up_node.inputs.len() >= 2
                        && gate_node.inputs[0] == up_node.inputs[0]
                    {
                        let norm_val = gate_node.inputs[0];
                        let gate_w = gate_node.inputs[1];
                        let up_w = up_node.inputs[1];
                        if ![gate_w, up_w, down_w].iter().all(|&w| fusible_weight(graph, w)) {
                            return None;
                        }

                        if let Some(&norm_nid) = val_producers.get(&norm_val) {
                            let norm_node = graph.node(norm_nid);
                            if let Op::RmsNorm { eps } = norm_node.op {
                                if norm_node.inputs.len() >= 2 && norm_node.inputs[0] == res_val {
                                    let norm_w = norm_node.inputs[1];
                                    // The fused kernel normalises and multiplies one row.
                                    if graph.value(res_val).shape.iter().rev().skip(1).product::<usize>() != 1 {
                                        return None;
                                    }
                                    let hidden_dim =
                                        graph.value(res_val).shape.last().copied().unwrap_or(0);
                                    let intermediate_dim =
                                        graph.value(gate_val).shape.last().copied().unwrap_or(0);

                                    let has_post_norm = post_norm_info.is_some();
                                    let desc = MegakernelDescriptor {
                                        kind: MegakernelKind::Mlp {
                                            hidden_dim,
                                            intermediate_dim,
                                            eps,
                                            is_geglu,
                                            has_post_norm,
                                        },
                                        name: format!("{}.fused_mlp", norm_node.name),
                                    };
                                    let mut inputs = vec![res_val, norm_w, gate_w, up_w, down_w];
                                    let post_norm_nid =
                                        if let Some((pn_id, pn_w, _)) = post_norm_info {
                                            inputs.push(pn_w);
                                            Some(pn_id)
                                        } else {
                                            None
                                        };
                                    return Some((
                                        norm_nid,
                                        gate_nid,
                                        up_nid,
                                        None,
                                        swiglu_nid,
                                        down_nid,
                                        post_norm_nid,
                                        desc,
                                        inputs,
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            None
        };

        let candidate =
            check_candidate(res_x, down_out).or_else(|| check_candidate(down_out, res_x));
        if let Some((
            norm_nid,
            gate_nid,
            up_nid,
            gu_opt,
            swiglu_nid,
            down_nid,
            pn_opt,
            desc,
            inputs,
        )) = candidate
        {
            if eliminated_nodes.contains(&norm_nid)
                || eliminated_nodes.contains(&gate_nid)
                || eliminated_nodes.contains(&up_nid)
                || eliminated_nodes.contains(&swiglu_nid)
                || eliminated_nodes.contains(&down_nid)
                || eliminated_nodes.contains(&add_nid)
                || gu_opt.is_some_and(|gu| eliminated_nodes.contains(&gu))
                || pn_opt.is_some_and(|pn| eliminated_nodes.contains(&pn))
            {
                continue;
            }
            let norm_out = graph.node(norm_nid).outputs[0];
            let gate_out = graph.node(gate_nid).outputs[0];
            let up_out = graph.node(up_nid).outputs[0];
            let swiglu_out = graph.node(swiglu_nid).outputs[0];
            let down_out_val = graph.node(down_nid).outputs[0];

            let safe_pn = match pn_opt {
                Some(pn) => {
                    let pn_out = graph.node(pn).outputs[0];
                    val_consumers.get(&down_out_val).map_or(0, |c| c.len()) == 1
                        && val_consumers.get(&pn_out).map_or(0, |c| c.len()) == 1
                }
                None => val_consumers.get(&down_out_val).map_or(0, |c| c.len()) == 1,
            };

            let safe = val_consumers.get(&norm_out).map_or(0, |c| c.len()) <= 2
                && val_consumers.get(&gate_out).map_or(0, |c| c.len()) == 1
                && val_consumers.get(&up_out).map_or(0, |c| c.len()) == 1
                && val_consumers.get(&swiglu_out).map_or(0, |c| c.len()) == 1
                && safe_pn;

            if !safe {
                continue;
            }

            // Create fused node
            let add_out_val = graph.node(add_nid).outputs[0];
            let out_shape = graph.value(add_out_val).shape.clone();
            let fused_out = graph.add(
                desc.name.clone(),
                Op::Megakernel(desc),
                &inputs,
                out_shape,
                DType::F32,
            );

            // Redirect consumers of add_node output to fused_out
            for node in &mut graph.nodes {
                for inp in &mut node.inputs {
                    if *inp == add_out_val {
                        *inp = fused_out;
                    }
                }
            }
            for out in &mut graph.outputs {
                if *out == add_out_val {
                    *out = fused_out;
                }
            }

            eliminated_nodes.insert(norm_nid);
            eliminated_nodes.insert(gate_nid);
            eliminated_nodes.insert(up_nid);
            if let Some(gu) = gu_opt {
                eliminated_nodes.insert(gu);
            }
            eliminated_nodes.insert(swiglu_nid);
            eliminated_nodes.insert(down_nid);
            eliminated_nodes.insert(add_nid);
            if let Some(pn) = pn_opt {
                eliminated_nodes.insert(pn);
            }
            rewrites += 1;
        }
    }
    if rewrites > 0 {
        crate::graph::pattern::compact_graph_public(graph, &eliminated_nodes);
    }
    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::egraph::Rule;
    use crate::graph::passes::fusion::FuseWeights;
    use crate::graph::{DType, Op, Tensor, Weights};

    #[test]
    fn test_megakernel_fusion() {
        let (hidden, inter) = (64, 128);
        let mut g = Graph::new();
        let mut w = Weights::new();

        let x = g.input("x", vec![1, hidden], DType::F32);

        let norm_w = g.weight("norm.weight", vec![hidden], DType::F32);
        w.insert("norm.weight", Tensor::F32(vec![1.0; hidden]));
        let normed = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, norm_w],
            vec![1, hidden],
            DType::F32,
        );

        let gate_w = g.weight("gate.weight", vec![inter, hidden], DType::F32);
        w.insert("gate.weight", Tensor::F32(vec![0.02; inter * hidden]));
        let gate = g.add(
            "gate",
            Op::Linear,
            &[normed, gate_w],
            vec![1, inter],
            DType::F32,
        );

        let up_w = g.weight("up.weight", vec![inter, hidden], DType::F32);
        w.insert("up.weight", Tensor::F32(vec![0.03; inter * hidden]));
        let up = g.add(
            "up",
            Op::Linear,
            &[normed, up_w],
            vec![1, inter],
            DType::F32,
        );

        let swiglu = g.add(
            "swiglu",
            Op::SwiGlu,
            &[gate, up],
            vec![1, inter],
            DType::F32,
        );

        let down_w = g.weight("down.weight", vec![hidden, inter], DType::F32);
        w.insert("down.weight", Tensor::F32(vec![0.01; hidden * inter]));
        let down = g.add(
            "down",
            Op::Linear,
            &[swiglu, down_w],
            vec![1, hidden],
            DType::F32,
        );

        let out = g.add("add", Op::Binary { kind: crate::graph::BinaryKind::Add }, &[x, down], vec![1, hidden], DType::F32);
        g.output(out);

        // 1. Run unfused on CPU to get reference output
        let mut ref_state = crate::graph::StateStore::new();
        let in_tensor = vec![Tensor::F32(vec![0.5; hidden])];
        let ref_res = crate::graph::cpu::run(&g, &w, &mut ref_state, in_tensor.clone())
            .expect("unfused CPU run");
        let ref_logits = ref_res[0].as_f32().expect("f32 output");

        // 2. Run fusion pass
        let rewrites = crate::graph::egraph::fuse_rule(&mut g, FuseWeights::None, Rule::MegakernelMlp);
        assert_eq!(rewrites, 1, "MLP block must be fused into megakernel");

        // 3. Validate SSA and check node structure
        crate::graph::pattern::validate_ssa(&g).expect("valid SSA after fusion");
        assert!(g.nodes.iter().any(|n| matches!(n.op, Op::Megakernel(_))));
        assert_eq!(
            g.nodes.len(),
            1,
            "Graph should only contain 1 fused Megakernel node"
        );

        // 4. Run fused on CPU and verify numerical parity
        let mut fused_state = crate::graph::StateStore::new();
        let fused_res = crate::graph::cpu::run(&g, &w, &mut fused_state, in_tensor.clone())
            .expect("fused CPU run");
        let fused_logits = fused_res[0].as_f32().expect("f32 output");

        assert_eq!(ref_logits.len(), fused_logits.len());
        for (r, f) in ref_logits.iter().zip(fused_logits) {
            assert!(
                (r - f).abs() < 1e-5,
                "CPU fused {f} vs unfused {r} mismatch"
            );
        }

        // 5. Run fused on Metal and verify numerical parity with CPU
        #[cfg(target_os = "macos")]
        if metal::Device::system_default().is_some() {
            let session = crate::graph::metal::Session::new(
                crate::graph::metal::Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir())
                    .unwrap(),
                &w,
            )
            .expect("Metal session creation");
            let metal_res = session.run(&g, in_tensor.clone()).expect("Metal fused run");
            let metal_logits = metal_res[0].as_f32().expect("Metal f32 output");
            assert_eq!(ref_logits.len(), metal_logits.len());
            for (r, m) in ref_logits.iter().zip(metal_logits) {
                assert!(
                    (r - m).abs() < 1e-4,
                    "Metal fused {m} vs CPU unfused {r} mismatch"
                );
            }
        }
    }

    #[test]
    fn test_megakernel_fusion_with_post_norm() {
        let (hidden, inter) = (64, 128);
        let mut g = Graph::new();
        let mut w = Weights::new();

        let x = g.input("x", vec![1, hidden], DType::F32);

        let norm_w = g.weight("norm.weight", vec![hidden], DType::F32);
        w.insert("norm.weight", Tensor::F32(vec![1.0; hidden]));
        let normed = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, norm_w],
            vec![1, hidden],
            DType::F32,
        );

        let gate_w = g.weight("gate.weight", vec![inter, hidden], DType::F32);
        w.insert("gate.weight", Tensor::F32(vec![0.02; inter * hidden]));
        let gate = g.add(
            "gate",
            Op::Linear,
            &[normed, gate_w],
            vec![1, inter],
            DType::F32,
        );

        let up_w = g.weight("up.weight", vec![inter, hidden], DType::F32);
        w.insert("up.weight", Tensor::F32(vec![0.03; inter * hidden]));
        let up = g.add(
            "up",
            Op::Linear,
            &[normed, up_w],
            vec![1, inter],
            DType::F32,
        );

        let swiglu = g.add(
            "swiglu",
            Op::SwiGlu,
            &[gate, up],
            vec![1, inter],
            DType::F32,
        );

        let down_w = g.weight("down.weight", vec![hidden, inter], DType::F32);
        w.insert("down.weight", Tensor::F32(vec![0.01; hidden * inter]));
        let down = g.add(
            "down",
            Op::Linear,
            &[swiglu, down_w],
            vec![1, hidden],
            DType::F32,
        );

        // Post-down RMSNorm (Gemma 4 style)
        let post_norm_w = g.weight("post_norm.weight", vec![hidden], DType::F32);
        w.insert("post_norm.weight", Tensor::F32(vec![0.9; hidden]));
        let down_norm = g.add(
            "post_norm",
            Op::RmsNorm { eps: 1e-5 },
            &[down, post_norm_w],
            vec![1, hidden],
            DType::F32,
        );

        let out = g.add("add", Op::Binary { kind: crate::graph::BinaryKind::Add }, &[x, down_norm], vec![1, hidden], DType::F32);
        g.output(out);

        // 1. Run unfused on CPU to get reference output
        let mut ref_state = crate::graph::StateStore::new();
        let in_tensor = vec![Tensor::F32(vec![0.5; hidden])];
        let ref_res = crate::graph::cpu::run(&g, &w, &mut ref_state, in_tensor.clone())
            .expect("unfused CPU run");
        let ref_logits = ref_res[0].as_f32().expect("f32 output");

        // 2. Run fusion pass
        let rewrites = crate::graph::egraph::fuse_rule(&mut g, FuseWeights::None, Rule::MegakernelMlp);
        assert_eq!(
            rewrites, 1,
            "MLP block with post-norm must be fused into megakernel"
        );

        // 3. Validate SSA and check node structure
        crate::graph::pattern::validate_ssa(&g).expect("valid SSA after fusion");
        assert!(g.nodes.iter().any(|n| matches!(n.op, Op::Megakernel(_))));
        assert_eq!(
            g.nodes.len(),
            1,
            "Graph should only contain 1 fused Megakernel node"
        );

        // 4. Run fused on CPU and verify numerical parity
        let mut fused_state = crate::graph::StateStore::new();
        let fused_res = crate::graph::cpu::run(&g, &w, &mut fused_state, in_tensor.clone())
            .expect("fused CPU run");
        let fused_logits = fused_res[0].as_f32().expect("f32 output");

        assert_eq!(ref_logits.len(), fused_logits.len());
        for (r, f) in ref_logits.iter().zip(fused_logits) {
            assert!(
                (r - f).abs() < 1e-5,
                "CPU fused {f} vs unfused {r} mismatch"
            );
        }

        // 5. Run fused on Metal and verify numerical parity with CPU
        #[cfg(target_os = "macos")]
        if metal::Device::system_default().is_some() {
            let session = crate::graph::metal::Session::new(
                crate::graph::metal::Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir())
                    .unwrap(),
                &w,
            )
            .expect("Metal session creation");
            let metal_res = session.run(&g, in_tensor.clone()).expect("Metal fused run");
            let metal_logits = metal_res[0].as_f32().expect("Metal f32 output");
            assert_eq!(ref_logits.len(), metal_logits.len());
            for (r, m) in ref_logits.iter().zip(metal_logits) {
                assert!(
                    (r - m).abs() < 1e-4,
                    "Metal fused {m} vs CPU unfused {r} mismatch"
                );
            }
        }
    }
}
