//! The causal-LM interface the scheduler drives ([`NativeLm`]): text and multimodal prompts,
//! per-sequence state, speculative-decoding hooks. Implementations: `runtime::lm::SpecLm` (every
//! decoder family spec) and `runtime::encdec::SpecEncDec`.

use crate::hardware::Device;
use crate::model::tokenizer::Tokenizer;
use crate::scheduler::spec::Drafter;
use std::any::Any;
use std::sync::Arc;

/// A media item attached to a prompt, as received (encoded file bytes), in prompt order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaInput {
    /// PNG/JPEG/... bytes; [`crate::model::image::ImageBuffer::decode`] reads them.
    Image(Vec<u8>),
    /// WAV/... bytes.
    Audio(Vec<u8>),
}

/// One sequence's model state: KV caches, recurrent state, encoded media.
pub type SeqState = Box<dyn Any + Send + Sync>;

/// What the request allows [`NativeLm::prepare_ctx`]: models that decode inside `prepare` (seq2seq
/// beam search) stop at `max_new_tokens` and when `cancel` is set (client gone).
#[derive(Default, Clone)]
pub struct PrepareCtx {
    pub max_new_tokens: usize,
    pub cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
}

/// The logits of a forward result.
pub fn logits_of(t: crate::graph::Tensor) -> Result<Vec<f32>, String> {
    match t {
        crate::graph::Tensor::F32(v) => Ok(v),
        _ => Err("expected f32 logits, got a device-picked token".into()),
    }
}

pub trait NativeLm: Send + Sync {
    /// Compute label for logs and the TUI header, e.g. `CPU (NEON SIMD), 15 threads`.
    fn backend_label(&self) -> String;
    /// Hardware device this model runs on.
    fn device(&self) -> Device {
        Device::Cpu
    }
    /// Weights and shape, for logs and the TUI (e.g. `bf16 weights, 24 layers, hidden 1024`).
    fn summary(&self) -> String;
    /// The model's graph nodes by op kind, device and avenue; `None` when it runs no graph program.
    fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        None
    }
    fn tokenizer(&self) -> &Tokenizer;
    /// Rows one sequence may store (`--ctx`): prompt + output unless Tier-4 KV page eviction
    /// ([`Self::kv_evict`]) drops older rows.
    fn context_len(&self) -> usize;
    /// Positions one sequence may reach (the model's trained context; RoPE never extrapolates).
    fn max_positions(&self) -> usize {
        self.context_len()
    }
    /// Rows of a full prefill pass (the instance shape ANE offload gates), if the model chunks
    /// prefill.
    fn prefill_rows(&self) -> Option<usize> {
        None
    }
    /// Output cap when the caller sets none (seq2seq: generation_config `max_length`).
    fn default_max_new_tokens(&self) -> Option<usize> {
        None
    }
    /// Turns the template-rendered prompt `ids` (one placeholder per media item, in order) into
    /// the ids the model runs (placeholders expanded to each item's token count) and builds the
    /// sequence's state. Errs on media the model can't take.
    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String>;
    /// [`Self::prepare`] under the request's limits; the scheduler calls this one.
    fn prepare_ctx(
        &self,
        ids: Vec<u32>,
        media: &[MediaInput],
        _ctx: &PrepareCtx,
    ) -> Result<(Vec<u32>, SeqState), String> {
        self.prepare(ids, media)
    }
    /// Length of the ids [`Self::prepare`] would return, without building state. Override when
    /// the count is cheaper than running the media encoders (it is for every image model).
    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        Ok(self.prepare(ids, media)?.0.len())
    }
    /// Runs `tokens` at positions `start_pos..` against `state`; returns the logits after the last.
    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String>;

    /// The next token of a greedy device decode stream (see
    /// [`crate::graph::program::GraphProgram::stream_next`]): `last` at `start_pos`, at most
    /// `max_new` tokens queued. `None` when this model or state cannot stream.
    fn stream_greedy_token(
        &self,
        _last: u32,
        _start_pos: usize,
        _max_new: usize,
        _state: &mut SeqState,
    ) -> Result<Option<u32>, String> {
        Ok(None)
    }

    /// Device decode stream supporting both greedy (`temp <= 1e-4`) and stochastic (`temp > 1e-4`) sampling.
    fn stream_token(
        &self,
        last: u32,
        start_pos: usize,
        max_new: usize,
        temp: f32,
        _top_p: f32,
        state: &mut SeqState,
    ) -> Result<Option<u32>, String> {
        if temp <= 1e-4 {
            self.stream_greedy_token(last, start_pos, max_new, state)
        } else {
            Ok(None)
        }
    }

    /// Ends the state's device decode stream, if any.
    fn stream_end(&self, _state: &mut SeqState) -> Result<(), String> {
        Ok(())
    }

    /// Whether the state holds recurrent layers: a decode stream's in-flight token then cannot
    /// be taken back, so a sequence that may verify drafts later must not stream.
    fn recurrent(&self) -> bool {
        false
    }

    // Speculative decoding (see `crate::scheduler::spec`): a model supports it by verifying
    // several positions in one pass and rewinding the ones the target rejects.

    /// [`Self::forward`] returning the logits after every token, `[tokens.len(), vocab]`.
    fn forward_all(
        &self,
        _tokens: &[u32],
        _start_pos: usize,
        _state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        Err("this model cannot verify draft tokens".into())
    }

    /// Drops positions `len..` from `state`, exactly as if they had never run. Recurrent models
    /// may only rewind into the span of their last [`Self::forward_all`].
    fn truncate(&self, _state: &mut SeqState, _len: usize) -> Result<(), String> {
        Err("this model cannot rewind its sequence state".into())
    }

    /// From now on every forward of this sequence records, per position, the output of each
    /// decoder layer in `layers` (0-based, the residual stream before the final norm: HF
    /// `hidden_states[i + 1]`) as `[positions, layers.len() * hidden]`, drained by
    /// [`Self::take_taps`].
    fn set_taps(&self, _state: &mut SeqState, _layers: &[usize]) -> Result<(), String> {
        Err("this model cannot expose hidden states to a drafter".into())
    }

    fn take_taps(&self, _state: &mut SeqState) -> Vec<f32> {
        Vec::new()
    }

    /// The drafter the checkpoint ships (e.g. multi-token-prediction heads).
    fn drafter(&self) -> Option<Arc<dyn Drafter>> {
        None
    }

    /// Writes row `id` of the token-embedding table the model's first layer reads into `out` (the
    /// table's width); `false` when the model has none or `id` is out of range.
    fn embed_row(&self, _id: u32, _out: &mut [f32]) -> bool {
        false
    }

    /// An online drafter ([`crate::runtime::noise_draft::OnlineNoiseDrafter`]) over this model's
    /// token embeddings ([`Self::embed_row`]) and vocabulary.
    fn online_drafter(
        &self,
        _knobs: Arc<crate::runtime::noise_draft::LiveDraftKnobs>,
    ) -> Option<Arc<dyn Drafter>> {
        None
    }

    /// What the radix prefix cache keeps of `state`'s first `len` positions: references to its
    /// KV pages and, when the model needs one, its boundary state (`None`: not cacheable).
    fn kv_export(&self, _state: &SeqState, _len: usize) -> Option<crate::kv::KvExport> {
        None
    }

    /// Resumes a freshly prepared `state` from cached prefix `m`; the positions resumed (`None`:
    /// nothing reused).
    fn kv_restore(&self, _state: &mut SeqState, _m: &crate::kv::Match) -> Option<usize> {
        None
    }

    /// The model's KV page pool.
    fn kv_pool(&self) -> Option<Arc<crate::kv::KvPool>> {
        None
    }

    /// Tier-4 KV page eviction ([`crate::compact::snapkv::evict_pages`]): drops older pages of
    /// `state` down to about `target_rows` stored rows; the positions evicted (0: nothing to drop).
    fn kv_evict(&self, _state: &mut SeqState, _target_rows: usize) -> Result<usize, String> {
        Err("this model cannot evict KV pages".into())
    }

    /// The radix prefix-cache key of prepared prompt `ids`: equal to `ids` except at media
    /// placeholders, which get keys identifying their content.
    fn radix_key(&self, _state: &SeqState, ids: &[u32]) -> Vec<u32> {
        ids.to_vec()
    }

    /// The chat template's thinking delimiters as `(open, close)` pairs
    /// (`runtime.thinking_delimiters`).
    fn thinking_delimiters(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Whether prompts may attach images (an `image` media encoder that loaded).
    fn has_vision(&self) -> bool {
        false
    }
}
