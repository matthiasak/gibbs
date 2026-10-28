//! Extension point for Metal op lowerings. `metal.rs` owns the core executor; each group below
//! owns a disjoint set of ops (planning, steps, kernels, encode, tests) and is reached through
//! the dispatch functions here.

use super::Counted;
use super::{Planned, Session, TG};
use crate::graph::{DType, Graph, Node, Op, ValueId};
use metal::{Buffer, MTLSize};
use std::collections::HashMap;

pub(super) mod attn;
pub(super) mod conv;
pub(super) mod deltanet;
pub(super) mod elementwise;
pub(super) mod matmul;
pub(super) mod moe;
pub(super) mod movement;
pub(super) mod norm;

/// All group kernel sources, appended to `metal_kernels.metal`.
pub(super) fn msl() -> String {
    let mut s = String::new();
    s.push_str(movement::MSL);
    s.push_str(elementwise::MSL);
    s.push_str(norm::MSL);
    s.push_str(matmul::MSL);
    s.push_str(conv::MSL);
    s.push_str(attn::MSL);
    s.push_str(moe::MSL);
    s.push_str(deltanet::MSL);
    s.push_str(include_str!("ext/ane.metal"));
    s
}

/// Names of every group's compute functions.
pub(super) fn kernel_names() -> Vec<&'static str> {
    let mut v = Vec::new();
    v.extend(movement::KERNELS);
    v.extend(elementwise::KERNELS);
    v.extend(norm::KERNELS);
    v.extend(matmul::KERNELS);
    v.extend(conv::KERNELS);
    v.extend(attn::KERNELS);
    v.extend(moe::KERNELS);
    v.extend(deltanet::KERNELS);
    v.extend(["ane_pack", "ane_unpack"]);
    v
}

pub(super) enum PlanOut<S> {
    Step(S, usize),
    /// Output shares memory with an operand; no step is emitted.
    Alias,
}

#[derive(Debug, Clone)]
pub enum ExtStep {
    Movement(movement::Step),
    Elementwise(elementwise::Step),
    Norm(norm::Step),
    Matmul(matmul::Step),
    Conv(conv::Step),
    Attn(attn::Step),
    Experts(moe::Step),
    /// Built only by the [`deltanet::fuse`] post-pass.
    DeltaNet(deltanet::DeltaNetStep),
}

pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    false
        || movement::supports(op, dtype)
        || elementwise::supports(op, dtype)
        || norm::supports(op, dtype)
        || matmul::supports(op, dtype)
        || conv::supports(op, dtype)
        || attn::supports(op, dtype)
        || moe::supports(op, dtype)
}

/// `None`: no group owns the op; fall back to the core planner.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    aliases: &mut HashMap<ValueId, (ValueId, usize)>,
) -> Option<Result<PlanOut<ExtStep>, String>> {
    if let Some(r) = movement::plan(g, node, aliases) {
        return Some(r.map(|p| match p {
            PlanOut::Step(s, n) => PlanOut::Step(ExtStep::Movement(s), n),
            PlanOut::Alias => PlanOut::Alias,
        }));
    }
    if let Some(r) = elementwise::plan(g, node, aliases) {
        return Some(r.map(|p| match p {
            PlanOut::Step(s, n) => PlanOut::Step(ExtStep::Elementwise(s), n),
            PlanOut::Alias => PlanOut::Alias,
        }));
    }
    if let Some(r) = norm::plan(g, node, aliases) {
        return Some(r.map(|p| match p {
            PlanOut::Step(s, n) => PlanOut::Step(ExtStep::Norm(s), n),
            PlanOut::Alias => PlanOut::Alias,
        }));
    }
    if let Some(r) = matmul::plan(g, node, aliases) {
        return Some(r.map(|p| match p {
            PlanOut::Step(s, n) => PlanOut::Step(ExtStep::Matmul(s), n),
            PlanOut::Alias => PlanOut::Alias,
        }));
    }
    if let Some(r) = conv::plan(g, node, aliases) {
        return Some(r.map(|p| match p {
            PlanOut::Step(s, n) => PlanOut::Step(ExtStep::Conv(s), n),
            PlanOut::Alias => PlanOut::Alias,
        }));
    }
    if let Some(r) = attn::plan(g, node, aliases) {
        return Some(r.map(|p| match p {
            PlanOut::Step(s, n) => PlanOut::Step(ExtStep::Attn(s), n),
            PlanOut::Alias => PlanOut::Alias,
        }));
    }
    if let Some(r) = moe::plan(g, node, aliases) {
        return Some(r.map(|p| match p {
            PlanOut::Step(s, n) => PlanOut::Step(ExtStep::Experts(s), n),
            PlanOut::Alias => PlanOut::Alias,
        }));
    }
    None
}

pub(super) fn reads(st: &ExtStep) -> Vec<ValueId> {
    match st {
        ExtStep::Movement(e) => movement::reads(e),
        ExtStep::Elementwise(e) => elementwise::reads(e),
        ExtStep::Norm(e) => norm::reads(e),
        ExtStep::Matmul(e) => matmul::reads(e),
        ExtStep::Conv(e) => conv::reads(e),
        ExtStep::Attn(e) => attn::reads(e),
        ExtStep::Experts(e) => moe::reads(e),
        ExtStep::DeltaNet(e) => deltanet::reads(e),
    }
}

pub(super) fn allows_non_f32(st: &ExtStep, v: ValueId) -> bool {
    match st {
        ExtStep::Movement(e) => movement::allows_non_f32(e, v),
        ExtStep::Elementwise(e) => elementwise::allows_non_f32(e, v),
        ExtStep::Norm(e) => norm::allows_non_f32(e, v),
        ExtStep::Matmul(e) => matmul::allows_non_f32(e, v),
        ExtStep::Conv(e) => conv::allows_non_f32(e, v),
        ExtStep::Attn(e) => attn::allows_non_f32(e, v),
        ExtStep::Experts(e) => moe::allows_non_f32(e, v),
        ExtStep::DeltaNet(e) => deltanet::allows_non_f32(e, v),
    }
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &ExtStep) -> Result<(), String> {
    match st {
        ExtStep::Movement(e_) => movement::encode(sess, e, s, e_),
        ExtStep::Elementwise(e_) => elementwise::encode(sess, e, s, e_),
        ExtStep::Norm(e_) => norm::encode(sess, e, s, e_),
        ExtStep::Matmul(e_) => matmul::encode(sess, e, s, e_),
        ExtStep::Conv(e_) => conv::encode(sess, e, s, e_),
        ExtStep::Attn(e_) => attn::encode(sess, e, s, e_),
        ExtStep::Experts(e_) => moe::encode(sess, e, s, e_),
        ExtStep::DeltaNet(e_) => deltanet::encode(sess, e, s, e_),
    }
}

/// Encoding context handed to group `encode` functions.
pub(super) struct Enc<'a, 'b> {
    pub enc: &'b Counted<'b>,
    pub loc: &'b dyn Fn(ValueId) -> (&'a Buffer, usize),
}

impl Enc<'_, '_> {
    /// Binds the buffer holding `v` (at its arena/weight offset) to `slot`.
    pub fn bind(&self, slot: u64, v: ValueId) {
        let (b, off) = (self.loc)(v);
        self.enc.set_buffer(slot, Some(b), off as u64);
    }
    /// Binds `bytes` of parameters to `slot` inline (no shared buffer a queued run could race on).
    pub fn bind_param(&self, slot: u64, bytes: &[u8]) {
        self.enc
            .set_bytes(slot, bytes.len() as u64, bytes.as_ptr() as *const _);
    }
    /// One thread per element, `TG` threads per group.
    pub fn grid1(&self, n: usize) {
        self.enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(TG), 1, 1),
            MTLSize::new(TG, 1, 1),
        );
    }
}
