//! Universal compiler optimization passes for Graph IR.

pub mod canonical;
pub mod dtype_adapt;
pub mod fusion;
pub mod megakernel;

use crate::graph::pattern::validate_ssa;
use crate::graph::Graph;
use fusion::FuseWeights;

/// The passes before fusion: the graph [`crate::graph::egraph`] fusion starts from. The other
/// [`canonical::run`] reductions wait until after fusion: removing reshapes changes the dependency
/// depths the rewriters order steps by, which specs balance with alias reshapes.
pub fn prefuse(graph: &mut Graph) {
    canonical::eliminate_common_subexpressions(graph);
    canonical::eliminate_dead_nodes(graph);
    dtype_adapt::run(graph);
}

/// The graph fusions (megakernel MLP, gate/up, QKV), chosen by equality saturation and cost-model
/// extraction ([`crate::graph::egraph`]), and the SSA check.
pub fn fuse(graph: &mut Graph, weights: FuseWeights<'_>) -> Result<(), String> {
    let r = crate::graph::egraph::fuse(
        graph,
        weights,
        crate::graph::egraph::DEFAULT_RULES,
        &crate::graph::egraph::CostModel::from_tune_cache(),
        crate::graph::egraph::Budget::default(),
    )?;
    tracing::info!("egraph fusion: {r:?}, compile {:.1} ms", r.compile_ms());
    validate_ssa(graph)
}
