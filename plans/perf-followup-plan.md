# Gibbs Follow-Up Performance Plan: Multi-Turn TUI TTFT, KV Slab Residency, and Llama 2 / MoE Roofline Gaps

## Context & Evidence

An end-to-end audit of multi-turn `--tui` latency on large models, combined with a `graphify-rs` structural pass (`7,057` nodes, `13,426` edges) and head-to-head benchmarking against `llama.cpp` on `Llama 2 7B` (`F16`, `Q8_0`, `Q4_0` in `bench/reference-7b.json`), surfaced five concrete performance bottlenecks:

1. **`KvPool` Slab Granularity (`SLAB_PAGES = 256`) & `MTLResidencySet` Asymmetry (`src/kv/mod.rs`, `src/graph/metal.rs`):**
   - `SLAB_PAGES = 256` (`256 × 64 = 16,384` tokens per slab) allocates 16,384 tokens of KV capacity on the **very first token**.
   - On an MHA model like `Llama 2 7B` (`32 layers × 2 × 4096 × 4 bytes`), a single slab is **16 GiB** (`17,179,869,184` bytes), which exceeds the default 25%-of-RAM `kv_mem_budget()` (12 GiB on a 48 GiB M5 Pro) and fails at 0 pages (`KV memory exhausted at 0 pages (raise --kv-mem)`).
   - Even worse, `KvPool::grow` (`src/kv/mod.rs:466`) adds every KV slab to an `MTLResidencySet` committed on the command queue, whereas `Session::new_for_graph` (`src/graph/metal.rs`) never adds weight buffers to an `MTLResidencySet`. Under GPU working-set pressure (`MTLDevice.recommendedMaxWorkingSetSize`), IOGPU keeps unused KV slab pages wired while paging out clean `mmap`-backed model weight pages between TUI turns.

2. **`PrefillTuner` Bandwidth-Bound Death Spiral & Unbucketed Shape Planning (`src/scheduler/mod.rs`, `src/graph/program.rs`):**
   - Unlike CLI `--prompt` (`whole_prompt = true`, full `prefill_rows = 512` chunks), `--tui` sizes prefill chunks to `IDLE_STEP = 100 ms` from a single linear EWMA `prefill_ns_per_token = elapsed / chunk_len`.
   - On a large model, a short prompt or tail chunk ($m \le 32$) still streams all model weights once, so `elapsed / chunk_len` records a high per-token cost (`5–50 ms/tok`), shrinking `prefill_budget` on subsequent turns and splitting a single turn's prompt into multiple small passes—each streaming the entire model weights again.
   - Because `chunk_len` varies continuously with the EWMA, `GraphProgram::run_reshaped` misses its 8-entry LRU shape cache (`INSTANCE_SHAPES = 8`) and re-runs `prefuse` + `fuse` + `canonical` + `Session::prepare_plan` (~6–40 ms of CPU work) during prefill, which is also folded into `prefill_time`.

3. **Sequence-End `RadixPrefixCache` Misses on Recurrent Models (`src/runtime/lm.rs`, `src/scheduler/mod.rs`):**
   - On hybrid/recurrent models (`Qwen 3.5` Gated DeltaNet, `LFM2` causal conv), `m.stream_token` queues 1 lookahead token in flight on the GPU (`FedStream`).
   - When generation hits EOS and `m.stream_end` drains that 1 in-flight token, `SpecLm::drain_stream` (`src/runtime/lm.rs:605-607`) sets `seq.recurrent_ahead = true`, causing the sequence-end `kv_export` (`src/runtime/lm.rs:1155`) to return `None`. Turn $N+1$ in `--tui` then falls back to Turn $N$'s prefill-end boundary and re-prefills the entire assistant reply from Turn $N$.

4. **Llama 2 7B Decode Gap vs `llama.cpp` (`0.91×–0.94×`):**
   - Measured on this M5 Pro (`bench/reference-7b.json`, `tg128`, `--reencode bit-exact`):
     - `Llama 2 7B F16`: **20.1 tok/s** vs `llama.cpp` **21.4 tok/s** (`0.94×`, `265.1 GB/s` streamed = `92.5%` of Gibbs ceiling, `293` dispatches/token, `50.10 ms` GPU vs `47.67 ms` weights_ms).
     - `Llama 2 7B Q8_0`: **35.6 tok/s** vs `llama.cpp` **38.9 tok/s** (`0.91×`, `249.9 GB/s` streamed = `87.3%` of ceiling, `293` dispatches/token, `28.55 ms` GPU vs `25.33 ms` weights_ms).
     - `Llama 2 7B Q4_0`: **62.4 tok/s** vs `llama.cpp` **67.6 tok/s** (`0.92×`, `231.9 GB/s` streamed = `81.0%` of ceiling, `293` dispatches/token, `16.21 ms` GPU vs `13.41 ms` weights_ms).
   - The `2.4–3.2 ms/token` non-weight gap across 32 layers (`~75–100 µs/layer`) is split across unfused packed `q`/`k`/`v` projections (3 dispatches/layer instead of 1), standalone `RmsNorm` dispatches (`65` dispatches/token), and `CachedAttention` decode overhead.

5. **Prefill Gap vs `llama.cpp` on K-Quants / IQ-Quants (`0.68×–0.75×`) and MoE (`0.35×` on Gemma 4 26B-A4B):**
   - In `Step::GemmBatchedPacked` (`src/graph/metal.rs:3652-3751`), when MPP is unavailable or `k % 8 != 0`, only `BF16`, `F16`, `Q4_0`, and `Q8_0` have direct packed batched GEMM kernels; other codecs fall back to full-matrix F32 expansion in `dequant_scratch`.
   - On `Gemma 4 26B-A4B · UD-IQ2_XXS`, prefill is `469 tok/s` vs `llama.cpp`'s `1346 tok/s` (`0.35×`): `Step::ExpertLinear` (`moe_map_rows` + `gemmgen_id_<fmt>`) launches tiles across all `num_experts` even for unrouted expert-tile pairs.

---

## Phased Implementation Plan

### Phase 1 — KV Slab Sizing & Weight Residency (`src/kv/mod.rs`, `src/graph/metal.rs`)

1. **Right-size `SLAB_PAGES` (`src/kv/mod.rs:52`):**
   - Reduce `pub const SLAB_PAGES: usize = 32` (`32 × 64 = 2,048` tokens per slab) and raise `pub const MAX_SLABS: usize = 8192` (keeping the 16.7M-token max capacity unchanged).
   - `KvView::metal_desc` (`src/kv/mod.rs:992-999`) already resolves each logical page to a 64-bit GPU address on the host, so no Metal kernel changes.
   - First-slab allocation on `Llama 2 7B` drops from **16 GiB → 2 GiB** (running under the default `--kv-mem` budget without `--kv-mem 18`), and on GQA models from **2–4 GiB → 256–512 MiB**.
2. **Add Weight Buffers to a `Session` `MTLResidencySet` + `MADV_WILLNEED` (`src/graph/metal.rs`):**
   - In `Session::new_for_graph`, call `libc::madvise(map_ptr as *mut c_void, map_len, libc::MADV_WILLNEED)` on each `GgufData::Map` window.
   - Create an `MTLResidencySet` on `Session` (using the same `new_residency_set` + `addAllocation:` + `commit` + `addResidencySet:` pattern as `KvPool` in `src/kv/mod.rs:230-285`) and register all weight `MTLBuffer`s (`map_buffers` windows and copied `bufs`) so IOGPU prioritizes keeping model weights wired in the GPU working set across idle periods between `--tui` turns.
3. **Verification:**
   - `target/release/gibbs --bench bench/reference-7b.json` succeeds on default `--kv-mem` (no `--kv-mem 18` needed).
   - Multi-turn `--tui` TTFT on a large model (> 10 GB) across 5 turns with 10 s idle gaps between turns shows no page-fault TTFT spike on turns 2–5.

### Phase 2 — `PrefillTuner` Fix & Power-of-Two Prefill Shape Bucketing (`src/scheduler/mod.rs`, `src/graph/program.rs`)

1. **Single-sequence interactive fast path (`src/scheduler/mod.rs:396-410, 1220-1222`):**
   - Pass `sole_request = !decoding && active.len() <= 1 && waiting.is_empty()` into `PrefillTuner::budget`. When `sole_request` is true (the standard `--tui` single-user chat turn), return `MAX_PREFILL_CHUNK` so `SpecLm::passes` runs full `prefill_rows` (`512`) chunks in one scheduler step instead of slicing a turn into small `IDLE_STEP` chunks.
2. **Affine prefill cost model & compile-time exclusion (`src/scheduler/mod.rs:413-428`):**
   - Exclude first-sight `run_reshaped` graph-planning time from `PrefillTuner::record`.
   - Update `prefill_ns_per_token` only when `prefill_tokens >= INITIAL_PREFILL_CHUNK` (`64`), preventing 1–15 token tail chunks from poisoning the per-token slope with full-model weight-read latency.
3. **Bucket prefill chunk lengths (`src/scheduler/mod.rs:1326-1328`):**
   - When `chunk_len < remaining` (a non-final prefill chunk), round `chunk_len` down to `{64, 128, 256, 512}` so non-final chunks only ever instantiate 4 shapes in `GraphProgram::run_reshaped` (`INSTANCE_SHAPES = 8`).
4. **Verification:**
   - Unit test in `scheduler::tests`: submitting a 10-token prompt followed by a 512-token prompt keeps `prefill_budget >= 64` (no collapse to 1-token chunks) and prefills the 512-token prompt in a single pass when no other sequence is decoding.

### Phase 3 — Preserve Recurrent Boundary Across `FedStream` Drain (`src/runtime/lm.rs`, `src/graph/metal.rs`)

1. **Checkpoint recurrent state before `FedStream` lookahead (`src/runtime/lm.rs:599-610, 950-1022`):**
   - In `FedStream` (or `SpecLm::stream_greedy_token` / `stream_token`), when `!self.recurrent.is_empty()`, keep a device-or-host snapshot of the contiguous recurrent states (`self.contiguous`) at the last committed position (`seq.len`), updated before the lookahead token overwrites it (or double-buffer recurrent states by `pos & 1` inside `submit_fed`).
   - In `SpecLm::drain_stream`, if `p.stream_end()?` discards an in-flight lookahead token on a recurrent model, restore the committed recurrent state instead of setting `seq.recurrent_ahead = true`.
2. **Export exact boundary after `spec::decode` EOS truncation (`src/scheduler/mod.rs:1608-1620`):**
   - When a sequence finishes at `eos < tokens.len()`, call `m.truncate(&mut seq.kv_cache, total_committed)` before `m.kv_export(&seq.kv_cache, total_committed)` so `seq.len == total_committed` holds for `needs_boundary()` models (`Gemma 4`, `Qwen 3.5`).
3. **Verification:**
   - Extend `scheduler::tests::second_turn_prefills_only_the_new_suffix` to cover both a sliding-window (`window: Some(128)`) model and a recurrent (`CausalConv1d` / `DeltaRule`) model over 2 turns: assert `cached_tokens` on Turn 2 includes Turn 1's generated assistant tokens, not just Turn 1's prompt.

### Phase 4 — Close the Llama 2 7B Decode Gap (`src/graph/passes/fusion.rs`, `src/graph/metal.rs`)

1. **Zero-duplicate packed QKV fusion for single-format layers (`src/graph/passes/fusion.rs`, `src/graph/program.rs`):**
   - Now that `GraphProgram::run_reshaped` runs `passes::fuse(.., FuseWeights::ReuseOnly(program))` (landed in Phase 1.5), every reshaped prefill instance reuses the decode graph's fused `q_fused_qkv_w` weight whenever `q`, `k`, `v` share a format and have no bias.
   - Verify whether any reshaped instance can ever reject a QKV fusion accepted on the decode graph; since `ReuseOnly` checks only `binds(program, &fused_w_name, dtype, [total_out, in_features])` (independent of `m`), it never rejects!
   - Therefore, `concat_mats` can fuse packed `Tensor::Mat` QKV weights and **drop the unfused `q`, `k`, `v` `Mat` entries** from `Weights` (zero memory overhead), reducing Llama 2 7B from `293` → `229` dispatches/token (−64 dispatches/token, −64 activation reads/token).
2. **ABBA gate on `bench/loop.json` + `bench/reference-7b.json`:**
   - Keep only if `bench/loop.json` geomean ≥ `1.00×` (min row ≥ `0.99×`) and `bench/reference-7b.json` (`F16`, `Q8_0`, `Q4_0`) improves toward `llama.cpp` (`67.6` / `38.9` / `21.4` tok/s).

### Phase 5 — MoE & Quantized Batched Prefill Acceleration (`src/graph/metal/ext/moe.rs`, `src/graph/metal.rs`)

1. **Compact active-expert tile dispatch in `Step::ExpertLinear` (`src/graph/metal/ext/moe.rs`):**
   - At prefill (`m = 512`), `moe_map_rows` already counts how many tokens route to each expert (`expert_counts`), but `gemmgen_id_<fmt>` dispatches threadgroups across `(max_m_tiles, n_tiles, num_experts)` or walks empty tiles.
   - Have `moe_map_rows` emit a compact list of `(expert_id, m_tile_start, m_tile_len)` work items and dispatch a 2D grid over active tiles only (or cap `max_m` from the observed max tokens per expert).
2. **Verification & Gate:**
   - ABBA `bench/loop.json` row 7 (`Gemma 4 26B-A4B · UD-IQ2_XXS`): target ≥ `1.5×` prefill speedup (`469 tok/s` → `700+ tok/s`) with 0 decode regression (`decode ratio >= 0.995`).
