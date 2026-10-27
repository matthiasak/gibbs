use crate::graph::{BinaryKind, Graph, NodeId, Op, Tensor, UnaryKind, ValueId, ValueKind, Weights};
use std::collections::{HashMap, HashSet};

/// Graph-only canonical reductions every instance runs before fusion (no weight changes):
/// 1. Identities ([`reduce_identities`]): `Scale(1)`, `Mul` by a `[1]` weight holding 1, `Add` of
///    a `[1]` weight holding 0 and a `Reshape` to its input's shape are removed; a `Scale` or
///    `Reshape` read only by another of its kind is folded into it.
/// 2. Common subexpression elimination (commutative `Binary` inputs in either order).
/// 3. Dead code elimination.
///
/// Repeated until a round removes no node, at most 4 rounds.
pub fn run(g: &mut Graph, weights: &Weights) {
    for _ in 0..4 {
        let before = g.nodes.len();
        reduce_identities(g, weights);
        eliminate_common_subexpressions(g);
        eliminate_dead_nodes(g);
        if g.nodes.len() == before {
            break;
        }
    }
}

/// Remaps the readers of identity nodes to their input and folds `Scale -> Scale` /
/// `Reshape -> Reshape` pairs whose first node has no other reader into the second; the nodes
/// left unread go in [`eliminate_dead_nodes`]. Graph outputs keep their producer, and an identity
/// is removed only when its input is an activation of the same dtype and shape.
pub fn reduce_identities(g: &mut Graph, weights: &Weights) {
    // Readers per value; a graph output counts as one.
    let mut uses = vec![0usize; g.values.len()];
    for n in &g.nodes {
        for &v in &n.inputs {
            uses[v.0 as usize] += 1;
        }
    }
    for &v in &g.outputs {
        uses[v.0 as usize] += 1;
    }
    let outputs: HashSet<ValueId> = g.outputs.iter().copied().collect();
    let mut producer: HashMap<ValueId, usize> = HashMap::new();
    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
    for i in 0..g.nodes.len() {
        // Topological order: every remap target is already final.
        for v in &mut g.nodes[i].inputs {
            if let Some(&r) = remap.get(v) {
                *v = r;
            }
        }
        let n = &g.nodes[i];
        let Some(&out) = n.outputs.first() else {
            continue;
        };
        producer.insert(out, i);
        let scalar = |v: ValueId, want: f32| {
            let val = g.value(v);
            val.kind == ValueKind::Weight
                && val.shape == [1]
                && matches!(weights.get(&val.name), Some(Tensor::F32(d)) if d[..] == [want])
        };
        let identity_of = match (&n.op, &n.inputs[..]) {
            (Op::Unary { kind: UnaryKind::Scale(s) }, &[x]) if (s - 1.0).abs() < 1e-9 => Some(x),
            (Op::Binary { kind: BinaryKind::Mul }, &[a, b]) if scalar(b, 1.0) => Some(a),
            (Op::Binary { kind: BinaryKind::Mul }, &[a, b]) if scalar(a, 1.0) => Some(b),
            (Op::Binary { kind: BinaryKind::Add }, &[a, b]) if scalar(b, 0.0) => Some(a),
            (Op::Binary { kind: BinaryKind::Add }, &[a, b]) if scalar(a, 0.0) => Some(b),
            (Op::Reshape { .. }, &[x]) => Some(x),
            _ => None,
        };
        if let Some(x) = identity_of {
            let (xv, ov) = (g.value(x), g.value(out));
            if !outputs.contains(&out)
                && matches!(xv.kind, ValueKind::Input | ValueKind::Node(_))
                && !xv.shape.is_empty()
                && (xv.dtype, &xv.shape, xv.layout) == (ov.dtype, &ov.shape, ov.layout)
            {
                remap.insert(out, x);
                uses[x.0 as usize] += uses[out.0 as usize] - 1;
                continue;
            }
        }
        let (&[src], Some(&p)) = (&n.inputs[..], n.inputs.first().and_then(|v| producer.get(v)))
        else {
            continue;
        };
        if uses[src.0 as usize] != 1 || g.nodes[p].inputs.len() != 1 {
            continue;
        }
        let first = g.nodes[p].inputs[0];
        let op = match (&g.nodes[p].op, &n.op) {
            (Op::Unary { kind: UnaryKind::Scale(a) }, Op::Unary { kind: UnaryKind::Scale(b) }) => {
                Op::Unary {
                    kind: UnaryKind::Scale(a * b),
                }
            }
            (Op::Reshape { .. }, Op::Reshape { .. }) => n.op.clone(),
            _ => continue,
        };
        g.nodes[i].op = op;
        g.nodes[i].inputs[0] = first;
    }
}

/// Identifies identical operations on identical inputs and merges them (the inputs of a
/// commutative `Binary` in either order).
pub fn eliminate_common_subexpressions(g: &mut Graph) {
    let mut seen: HashMap<String, ValueId> = HashMap::new();
    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();

    for node in &mut g.nodes {
        // Remap inputs first
        for inp in &mut node.inputs {
            if let Some(&new_id) = remap.get(inp) {
                *inp = new_id;
            }
        }

        // Build a structural signature for deterministic CSE
        let mut inputs = node.inputs.clone();
        if let Op::Binary {
            kind: BinaryKind::Add | BinaryKind::Mul | BinaryKind::Max | BinaryKind::Min,
        } = node.op
        {
            inputs.sort_by_key(|v| v.0);
        }
        let key = format!("{:?}:{:?}", node.op, inputs);
        if let Some(&existing_out) = seen.get(&key) {
            if !node.outputs.is_empty() {
                remap.insert(node.outputs[0], existing_out);
            }
        } else if !node.outputs.is_empty() {
            seen.insert(key, node.outputs[0]);
        }
    }

    // Remap graph outputs if needed
    for out in &mut g.outputs {
        if let Some(&new_id) = remap.get(out) {
            *out = new_id;
        }
    }
}

/// Retains only nodes that transitively contribute to graph outputs, or write persistent state
/// (a `write: true` `CachedAttention` appends to its K/V state even when nothing reads its output:
/// a biased self-attention can take the state write from it and the scores from `Attention`; a
/// `StateWrite` exists for its write).
pub fn eliminate_dead_nodes(g: &mut Graph) {
    let mut live_values: HashSet<ValueId> = g.outputs.iter().copied().collect();
    for node in &g.nodes {
        let writes_state = match node.op {
            Op::CachedAttention { write: true, .. } => node
                .inputs
                .get(3)
                .is_some_and(|&s| matches!(g.value(s).kind, ValueKind::State { .. })),
            Op::StateWrite => true,
            _ => false,
        };
        if writes_state {
            live_values.extend(node.outputs.iter().copied());
        }
    }

    // Work backwards from outputs to inputs
    for node in g.nodes.iter().rev() {
        let produces_live = node.outputs.iter().any(|out| live_values.contains(out));
        if produces_live {
            for &inp in &node.inputs {
                live_values.insert(inp);
            }
        }
    }

    // Keep only live nodes
    g.nodes
        .retain(|node| node.outputs.iter().any(|out| live_values.contains(out)));
    // Outputs record their producer's index: renumber after nodes were dropped.
    for (i, node) in g.nodes.iter().enumerate() {
        for &out in &node.outputs {
            g.values[out.0 as usize].kind = ValueKind::Node(NodeId(i as u32));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Op};

    #[test]
    fn test_canonical_dce_and_cse() {
        let mut g = Graph::new();
        let a = g.input("a", vec![4], DType::F32);
        let b = g.input("b", vec![4], DType::F32);

        // Identical additions (CSE candidate)
        let add1 = g.add(
            "add1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[a, b],
            vec![4],
            DType::F32,
        );
        let add2 = g.add(
            "add2",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[a, b],
            vec![4],
            DType::F32,
        );

        // Dead node (never used by outputs)
        let _dead = g.add(
            "dead",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[a, b],
            vec![4],
            DType::F32,
        );

        // Output depends on add1 and add2
        let mul = g.add(
            "mul",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[add1, add2],
            vec![4],
            DType::F32,
        );
        g.output(mul);

        assert_eq!(g.nodes.len(), 4);
        run(&mut g, &Weights::new());

        // After CSE and DCE:
        // add2 is merged into add1, dead node is eliminated.
        // We should have only add1 and mul (2 nodes).
        assert_eq!(g.nodes.len(), 2);
        // `mul` now sits at index 1 and its value says so (the SSA check reads this).
        crate::graph::pattern::validate_ssa(&g)
            .expect("producer indices follow the retained nodes");
    }

    /// Identity scales, chained scales, reshape chains and commuted adds reduce to fewer nodes,
    /// the CPU outputs stay equal and a second `run` changes nothing.
    #[test]
    fn reductions_preserve_outputs_and_reach_fixed_point() {
        use crate::graph::{cpu, StateStore};

        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let y = g.input("y", vec![4, 2], DType::F32);
        let one = g.weight("one", vec![1], DType::F32);
        let zero = g.weight("zero", vec![1], DType::F32);
        let unary = |s| Op::Unary {
            kind: UnaryKind::Scale(s),
        };
        let bin = |kind| Op::Binary { kind };
        let reshape = |s: &[usize]| Op::Reshape {
            shape: Some(s.to_vec()),
        };
        let v = g.add("s1", unary(1.0), &[x], vec![2, 4], DType::F32);
        let v = g.add("m1", bin(BinaryKind::Mul), &[v, one], vec![2, 4], DType::F32);
        let v = g.add("a0", bin(BinaryKind::Add), &[zero, v], vec![2, 4], DType::F32);
        let v = g.add("r0", reshape(&[2, 4]), &[v], vec![2, 4], DType::F32);
        let v = g.add("c1", unary(2.0), &[v], vec![2, 4], DType::F32);
        let v = g.add("c2", unary(4.0), &[v], vec![2, 4], DType::F32);
        let v = g.add("r1", reshape(&[8]), &[v], vec![8], DType::F32);
        let v = g.add("r2", reshape(&[4, 2]), &[v], vec![4, 2], DType::F32);
        let a = g.add("add1", bin(BinaryKind::Add), &[v, y], vec![4, 2], DType::F32);
        let b = g.add("add2", bin(BinaryKind::Add), &[y, v], vec![4, 2], DType::F32);
        let out = g.add("out", bin(BinaryKind::Mul), &[a, b], vec![4, 2], DType::F32);
        g.output(out);

        let mut weights = Weights::new();
        weights.insert("one", Tensor::F32(vec![1.0]));
        weights.insert("zero", Tensor::F32(vec![0.0]));
        let inp = vec![
            Tensor::F32(vec![0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -0.75, 1.25]),
            Tensor::F32(vec![1.0, 2.0, -3.0, 0.5, 0.0, -1.5, 2.5, 4.0]),
        ];
        let run_cpu = |g: &Graph| {
            cpu::run(g, &weights, &mut StateStore::new(), inp.clone()).unwrap()[0].to_f32()
        };
        let want = run_cpu(&g);

        run(&mut g, &weights);
        let ops: Vec<&str> = g.nodes.iter().map(|n| n.name.as_str()).collect();
        // Scale(8) of x, one reshape to [4, 2], one add, the product.
        assert_eq!(ops, ["c2", "r2", "add1", "out"]);
        assert_eq!(g.nodes[0].op, unary(8.0));
        assert_eq!(g.nodes[0].inputs, [x]);
        crate::graph::pattern::validate_ssa(&g).expect("SSA after reductions");
        assert_eq!(run_cpu(&g), want);

        let reduced = format!("{:?}", g.nodes);
        run(&mut g, &weights);
        assert_eq!(format!("{:?}", g.nodes), reduced, "a second run is a fixed point");
    }
}
