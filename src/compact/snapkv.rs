//! Tier 4: KV page eviction over a sequence's paged KV ([`crate::kv`]): StreamingLLM attention
//! sinks, the most recent pages, and H2O-style heavy hitters (pages with the largest keys).

use crate::kv::{KvView, PAGE_TOKENS};

/// Evicts full-attention pages of `view` down to `target_rows` stored rows: keeps the first
/// `sink_pages` (attention sinks) and the last `recent_pages` (at least the partial last one),
/// and fills the rest of the budget with the middle pages whose mean key L2 norm (rows of every
/// full-class `.k` state) is highest, in their order. Ring classes are untouched and positions
/// keep counting ([`KvView::evicted`]). Returns the positions evicted: 0 when `target_rows`
/// already holds the rows or the budget holds no more than the sinks and recent pages.
pub fn evict_pages(
    view: &mut KvView,
    target_rows: usize,
    sink_pages: usize,
    recent_pages: usize,
) -> usize {
    let pool = view.pool().clone();
    let Some(c) = pool.classes.iter().position(|c| c.window.is_none()) else {
        return 0;
    };
    let recent_pages = recent_pages.max(1);
    let n = view.lens[c].div_ceil(PAGE_TOKENS).min(view.tables[c].len());
    let budget = (target_rows / PAGE_TOKENS).max(sink_pages + recent_pages);
    if n <= budget {
        return 0;
    }
    // Middle pages are all full: one key-norm sum per page ranks them as the mean does.
    let middle = sink_pages..n - recent_pages;
    let mut score = vec![0.0f32; n];
    for name in pool.classes[c].names.iter().filter(|p| p.name.ends_with(".k")) {
        let Some(keys) = view.gather(&name.name) else {
            continue;
        };
        let w = name.row_width;
        for lp in middle.clone() {
            let page = &keys[lp * PAGE_TOKENS * w..(lp + 1) * PAGE_TOKENS * w];
            let norm = |r: &[f32]| r.iter().map(|x| x * x).sum::<f32>().sqrt();
            score[lp] += page.chunks_exact(w).map(norm).sum::<f32>();
        }
    }
    let mut heavy: Vec<usize> = middle.collect();
    heavy.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    heavy.truncate(budget - sink_pages - recent_pages);
    heavy.sort_unstable();
    let keep: Vec<usize> = (0..sink_pages).chain(heavy).chain(n - recent_pages..n).collect();
    let before = view.evicted;
    view.keep_pages(&keep);
    view.evicted - before
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, Weights};
    use crate::kv::KvPool;

    const KV: usize = 64;

    /// Causal attention over paged `kv.0.k` / `kv.0.v` (2 query heads, 1 kv head of 64).
    fn attn(m: usize) -> Graph {
        let mut g = Graph::new();
        let q = g.input("q", vec![m, 2 * KV], DType::F32);
        let k = g.input("k", vec![m, KV], DType::F32);
        let v = g.input("v", vec![m, KV], DType::F32);
        let ks = g.state("kv.0.k", vec![4096, KV], DType::F32, false);
        let vs = g.state("kv.0.v", vec![4096, KV], DType::F32, false);
        let pos = g.input("pos", vec![m], DType::U32);
        let op = Op::CachedAttention {
            heads: 2,
            kv_heads: 1,
            head_dim: KV,
            v_head_dim: KV,
            scale: 0.125,
            causal: true,
            window: None,
            soft_cap: None,
            write: true,
        };
        let out = g.add("attn", op, &[q, k, v, ks, vs, pos], vec![m, 2 * KV], DType::F32);
        g.output(out);
        g
    }

    fn run(g: &Graph, st: &mut StateStore, x: &(Vec<f32>, Vec<f32>, Vec<f32>), pos: usize) -> Vec<f32> {
        let m = x.1.len() / KV;
        let inputs = vec![
            Tensor::F32(x.0.clone()),
            Tensor::F32(x.1.clone()),
            Tensor::F32(x.2.clone()),
            Tensor::U32((pos as u32..(pos + m) as u32).collect()),
        ];
        cpu::run(g, &Weights::new(), st, inputs).unwrap().remove(0).into_f32().unwrap()
    }

    /// Five pages, budget three: page 0, the heaviest middle page (2) and the last page stay, in
    /// order; the dropped pages return to the pool; decode then continues exactly as attention
    /// over a contiguous cache holding only the kept rows.
    #[test]
    fn evict_pages_keeps_sinks_recent_and_heaviest() {
        let mut seed = 7u64;
        let mut rand = |n: usize, s: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * s
                })
                .collect()
        };
        let (prefill, decode) = (attn(PAGE_TOKENS), attn(1));
        let pool = KvPool::host(&crate::graph::paged_states(&prefill));
        let mut paged = StateStore::new();
        paged.kv = Some(KvView::new(pool.clone()));
        let chunks: Vec<_> = [1.0, 1.0, 4.0, 0.5, 1.0]
            .iter()
            .map(|&s| (rand(PAGE_TOKENS * 2 * KV, 1.0), rand(PAGE_TOKENS * KV, s), rand(PAGE_TOKENS * KV, 1.0)))
            .collect();
        for (i, x) in chunks.iter().enumerate() {
            run(&prefill, &mut paged, x, i * PAGE_TOKENS);
        }
        let kv = paged.kv.as_mut().unwrap();
        let (before, used) = (kv.tables[0].clone(), pool.used_pages());
        assert_eq!(evict_pages(kv, 3 * PAGE_TOKENS, 1, 1), 2 * PAGE_TOKENS);
        assert_eq!(kv.tables[0], [before[0], before[2], before[4]]);
        assert_eq!((kv.evicted, kv.lens[0]), (2 * PAGE_TOKENS, 3 * PAGE_TOKENS));
        assert_eq!(pool.used_pages(), used - 2);
        assert_eq!(evict_pages(kv, 3 * PAGE_TOKENS, 1, 1), 0, "already within budget");

        let mut plain = StateStore::new();
        for (i, x) in [&chunks[0], &chunks[2], &chunks[4]].into_iter().enumerate() {
            run(&prefill, &mut plain, x, i * PAGE_TOKENS);
        }
        let rows = 3 * PAGE_TOKENS * KV;
        let k = paged.kv.as_ref().unwrap().gather("kv.0.k").unwrap();
        assert_eq!(k, plain.get("kv.0.k").unwrap().as_f32().unwrap()[..rows]);
        for step in 0..70 {
            let x = (rand(2 * KV, 1.0), rand(KV, 1.0), rand(KV, 1.0));
            let got = run(&decode, &mut paged, &x, 5 * PAGE_TOKENS + step);
            let want = run(&decode, &mut plain, &x, 3 * PAGE_TOKENS + step);
            let err = got.iter().zip(&want).fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
            assert!(err < 1e-5, "decode step {step} after eviction: err {err}");
        }
    }
}
