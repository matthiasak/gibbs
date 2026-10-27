//! The causal-LM pipeline: [`SpecLm`] runs the `text` component of any `pipeline: causal_lm`
//! family spec as a [`NativeLm`].
//!
//! Instances (`arch::build`) are keyed by the vars `rows` (positions per pass), `head` (`last`:
//! logits of the last row; `all`: of every row), `taps` (`layer.{i}` extra outputs) and `media`
//! (media rows in the pass; absent when 0). The decode instance (`rows=1, head=last`) is compiled
//! once; every other instance runs on its weights and session through
//! `GraphProgram::run_reshaped`. Each sequence owns its `StateStore`, swapped into the program for
//! each pass.
//!
//! Inputs the runner fills: `token_ids`, `positions`, `visible_end` (the end of the bidirectional
//! block a row sits in, else its position + 1), every input with a `gen` (`arch::gen`; a gen
//! input's args may name `pos0`, the position of the pass's first row, and `ids`, the token ids of
//! positions `pos0 - back .. pos0 + rows` (`back`: the gen's own numeric arg, default 0, clipped at
//! the sequence start), which the runner substitutes) and the media inputs: `media_rows` (the
//! media tower's output 0 for the pass's media rows, in prompt order), `media_index` (their rows
//! within the pass), any other input named like a tower output (that output's rows for the same
//! media rows) and `rope_positions` (`[t, h, w]` per row from `positions: mrope_grid`; text rows
//! `p` repeated, rows past the prompt shifted by its delta).
//!
//! `runtime` keys: `prefill_rows` (512) rows per prefill pass; `bidir_media_blocks`: a pass that
//! would end inside a bidirectional block runs to the block's end (at most 512 rows), and the
//! block end's logits answer the positions it ran ahead; `media` ([`crate::runtime::media`]):
//! each attached item is preprocessed, run through its modality's tower component once at
//! `prepare`, and its placeholder expanded.

use crate::arch::build::{build, InputPlan, Instance, ShapeCache};
use crate::arch::expr::{Env, Val};
use crate::arch::gen::{generate, GenData};
use crate::arch::load::Model;
use crate::arch::Registry;
use crate::graph::program::GraphProgram;
use crate::graph::{StateStore, Tensor};
use crate::hardware::{Device, PlacementMode};
use crate::model::hub::ModelArtifacts;
use crate::model::native::{logits_of, MediaInput, NativeLm, SeqState};
use crate::model::tokenizer::Tokenizer;
use crate::runtime::media::{self, Expanded, Grid, Media, Towers};
use parking_lot::Mutex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

/// The component a causal-LM spec runs.
pub const TEXT: &str = "text";
/// The most rows one pass grows to when it finishes a bidirectional block.
pub const MAX_BLOCK_ROWS: usize = 512;

/// The `runtime` keys this pipeline reads (the rest belong to other glue).
#[derive(Deserialize)]
struct Runtime {
    #[serde(default = "prefill_rows")]
    prefill_rows: usize,
    #[serde(default)]
    bidir_media_blocks: bool,
    #[serde(default)]
    media: Option<serde_json::Value>,
    /// Thinking delimiters, consecutive elements forming `(open, close)` pairs.
    #[serde(default)]
    thinking_delimiters: Vec<String>,
}

fn prefill_rows() -> usize {
    512
}

/// One sequence's state.
pub struct Seq {
    state: StateStore,
    /// The prepared prompt: the tokens of positions a bidirectional block runs ahead to.
    prompt: Vec<u32>,
    /// Positions in `state`.
    len: usize,
    /// Token ids of positions `0..len` (gen inputs read them as `ids`).
    ids: Vec<u32>,
    /// Positions the caller has run (`< len` while a finished block runs ahead of it).
    asked: usize,
    /// Logits after position `len - 1` while `asked < len`.
    ahead: Vec<f32>,
    /// Bidirectional media blocks (positions), each attended whole in one pass; media
    /// preprocessing fills them when the spec sets `runtime.bidir_media_blocks`.
    pub blocks: Vec<Range<usize>>,
    /// The expanded prompt's media rows and rotary positions.
    media: Expanded,
    /// The tower outputs of every media row (`media.media_index` order): (text input name, row
    /// width, rows). Output 0 of a tower feeds `media_rows`.
    rows: Vec<(String, usize, Vec<f32>)>,
    taps: Vec<usize>,
    tapped: Vec<f32>,
    /// Recurrent-state snapshots after each position from `ckpt_from` (the last `forward_all`).
    ckpt: Vec<HashMap<String, Tensor>>,
    ckpt_from: usize,
    recurrent_ahead: bool,
}

impl Seq {
    fn new(prompt: Vec<u32>, media: Expanded, rows: Vec<(String, usize, Vec<f32>)>) -> Self {
        Seq {
            state: StateStore::new(),
            prompt,
            len: 0,
            ids: Vec::new(),
            asked: 0,
            ahead: Vec::new(),
            blocks: media.blocks.clone(),
            media,
            rows,
            taps: Vec::new(),
            tapped: Vec::new(),
            ckpt: Vec::new(),
            ckpt_from: 0,
            recurrent_ahead: false,
        }
    }

    /// Media rows (indices into `media.media_index`) at positions `at..end`.
    fn media_in(&self, at: usize, end: usize) -> Range<usize> {
        let idx = &self.media.media_index;
        idx.partition_point(|&q| (q as usize) < at)..idx.partition_point(|&q| (q as usize) < end)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Last,
    All,
}

/// A causal LM run from its family spec.
pub struct SpecLm {
    model: Arc<Model>,
    rt: Runtime,
    media: Media,
    towers: Towers,
    /// Modalities whose tower failed to build at load, with why (said when a prompt attaches one).
    off: HashMap<String, String>,
    tok: Arc<Tokenizer>,
    ctx: usize,
    shapes: ShapeCache,
    /// Compiled from the decode instance; every other instance runs reshaped on it.
    program: Mutex<GraphProgram>,
    recurrent: Vec<String>,
    /// States the KV pool does not page: a prefix-cache boundary snapshots them.
    contiguous: Vec<String>,
    kv_pool: Option<Arc<crate::kv::KvPool>>,
    /// The decode instance takes exactly `[token_ids, positions]`, outputs `logits` first and
    /// holds no recurrent state: greedy decode may stream on the device.
    streams: bool,
    weights: usize,
    mode: PlacementMode,
    /// The spec's own drafter (`runtime.drafter`), compiled at the first ask.
    drafter: std::sync::OnceLock<Option<Arc<dyn crate::scheduler::spec::Drafter>>>,
    /// The decode graph's token-embedding table, for [`NativeLm::embed_row`].
    embed: Option<Arc<EmbedTable>>,
}

/// The weight of the decode graph's first `Embedding` node reading `token_ids`, `[vocab, cols]`.
struct EmbedTable {
    t: Tensor,
    cols: usize,
}

impl EmbedTable {
    // ponytail: an unpacked (f32/f16/bf16 vector) table is copied once here; share tensor data
    // through `Arc` in `graph::Tensor` if that memory matters.
    fn find(g: &crate::graph::Graph, w: &crate::graph::Weights) -> Option<Self> {
        let ids = *g.inputs.iter().find(|&&v| g.value(v).name == "token_ids")?;
        g.nodes.iter().find_map(|n| {
            let (&table, &input) = (n.inputs.first()?, n.inputs.get(1)?);
            let v = g.value(table);
            (matches!(n.op, crate::graph::Op::Embedding)
                && input == ids
                && matches!(v.kind, crate::graph::ValueKind::Weight))
            .then(|| {
                Some(Self {
                    t: w.get(&v.name)?.clone(),
                    cols: *v.shape.last()?,
                })
            })
            .flatten()
        })
    }

    fn row(&self, id: u32, out: &mut [f32]) -> bool {
        let (r, c) = (id as usize, self.cols);
        let Some(out) = out.get_mut(..c) else {
            return false;
        };
        let span = r * c..(r + 1) * c;
        match &self.t {
            Tensor::F32(v) => v.get(span).map(|s| out.copy_from_slice(s)).is_some(),
            Tensor::F16(v) => v
                .get(span)
                .map(|s| {
                    for (o, &x) in out.iter_mut().zip(s) {
                        *o = crate::model::f16_to_f32(x);
                    }
                })
                .is_some(),
            Tensor::BF16(v) => v
                .get(span)
                .map(|s| {
                    for (o, &x) in out.iter_mut().zip(s) {
                        *o = crate::model::bf16_to_f32(x);
                    }
                })
                .is_some(),
            Tensor::Mat(m) if r < m.rows && m.cols == c => {
                m.decode_row_f32(r, out);
                true
            }
            _ => false,
        }
    }
}

fn vars(rows: usize, head: &str, taps: &[usize], media: usize) -> Vec<(&'static str, Val)> {
    let mut v = vec![
        ("rows", Val::Num(rows as f64)),
        ("head", Val::Str(head.to_string())),
    ];
    if !taps.is_empty() {
        let t = taps
            .iter()
            .map(|l| Val::Str(format!("layer.{l}")))
            .collect();
        v.push(("taps", Val::List(t)));
    }
    if media > 0 {
        v.push(("media", Val::Num(media as f64)));
    }
    v
}

/// One pass: rows `hist[pos0..]` of `seq`; `hist` holds the token ids of every position up to the
/// pass end.
struct Pass<'a> {
    hist: &'a [u32],
    pos0: usize,
    blocks: &'a [Range<usize>],
    seq: &'a Seq,
}

/// The runner-filled value of input `d` for pass `p`.
fn input(d: &InputPlan, p: &Pass) -> Result<Tensor, String> {
    let (hist, pos0, blocks) = (p.hist, p.pos0, p.blocks);
    let ids = &hist[pos0..];
    let n = ids.len();
    let ks = p.seq.media_in(pos0, pos0 + n);
    let m = &p.seq.media;
    let t = match (&d.gen, d.name.as_str()) {
        (Some((g, args)), _) => {
            let mut args = args.clone();
            if let Some(o) = args.as_object_mut() {
                let back = o.get("back").and_then(|b| b.as_f64()).unwrap_or(0.0) as usize;
                for v in o.values_mut() {
                    match v.as_str() {
                        Some("pos0") => *v = pos0.into(),
                        Some("ids") => *v = hist[pos0.saturating_sub(back)..].into(),
                        _ => {}
                    }
                }
            }
            match generate(g, &args)
                .map_err(|e| format!("input `{}`: {e}", d.name))?
                .data
            {
                GenData::F32(v) => Tensor::F32(v),
                GenData::U32(v) => Tensor::U32(v),
            }
        }
        (None, "token_ids") => Tensor::U32(ids.to_vec()),
        (None, "positions") => Tensor::U32((pos0..pos0 + n).map(|q| q as u32).collect()),
        (None, "visible_end") => Tensor::U32(
            (pos0..pos0 + n)
                .map(|q| blocks.iter().find(|b| b.contains(&q)).map_or(q + 1, |b| b.end) as u32)
                .collect(),
        ),
        (None, "media_index") => Tensor::U32(m.media_index[ks].iter().map(|&q| q - pos0 as u32).collect()),
        (None, "rope_positions") => Tensor::U32(
            (pos0..pos0 + n)
                .flat_map(|q| match m.pos3.get(q) {
                    Some(p) => *p,
                    None => [(q as i64 + m.delta) as u32; 3],
                })
                .collect(),
        ),
        (None, name) => match p.seq.rows.iter().find(|(n, _, _)| n == name) {
            Some((_, w, rows)) => Tensor::F32(rows[ks.start * w..ks.end * w].to_vec()),
            None => {
                return Err(format!(
                    "input `{name}` has no generator and is none the LM runtime supplies (token_ids, positions, visible_end, media_rows, media_index, rope_positions, tower outputs)"
                ))
            }
        },
    };
    let (want, got) = (
        d.shape.iter().product::<usize>(),
        match &t {
            Tensor::F32(v) => v.len(),
            Tensor::U32(v) => v.len(),
            _ => 0,
        },
    );
    if want != got {
        return Err(format!(
            "input `{}`: {got} values for shape {:?}",
            d.name, d.shape
        ));
    }
    Ok(t)
}

/// Loads `artifacts` as `family` (detected when `None`) behind the re-encode KL gate.
pub fn load(
    artifacts: &ModelArtifacts,
    reg: &Registry,
    family: Option<&str>,
    mode: PlacementMode,
) -> Result<Arc<dyn NativeLm>, String> {
    load_with_ctx(artifacts, reg, family, mode, None)
}

/// [`load`] with an explicit `--ctx` override (`runtime.ctx`).
pub fn load_with_ctx(
    artifacts: &ModelArtifacts,
    reg: &Registry,
    family: Option<&str>,
    mode: PlacementMode,
    ctx: Option<usize>,
) -> Result<Arc<dyn NativeLm>, String> {
    let on_metal = match mode {
        PlacementMode::Pin(d) => d == Device::Metal,
        PlacementMode::Auto => crate::hardware::default_device() == Device::Metal,
    };
    crate::arch::load::gated(
        on_metal,
        || {
            Some(
                crate::arch::load::load(artifacts, reg, family)
                    .and_then(|m| {
                        if let Some(c) = ctx {
                            m.env.set_ctx(c.into());
                        }
                        SpecLm::new(m, mode)
                    })
                    .map(|m| Arc::new(m) as Arc<dyn NativeLm>),
            )
        },
        |r, c| crate::model::weight_format::native_kl_gate(&**r, &**c),
    )
    .expect("the spec loader answers every attempt")
}

impl SpecLm {
    /// Compiles `model`'s decode instance on `mode`.
    pub fn new(model: Model, mode: PlacementMode) -> Result<Self, String> {
        crate::arch::load::check(&model)?;
        let fam = model.spec.family.clone();
        let rt: Runtime = model.spec.runtime_as()?;
        if rt.thinking_delimiters.len() % 2 != 0 {
            return Err(format!(
                "{fam}: runtime.thinking_delimiters needs open/close pairs"
            ));
        }
        let tok = match (&model.tokenizer, model.ckpt.synthetic) {
            (Some(t), _) => t.clone(),
            // Synthetic (random-weight) ids ship no tokenizer.
            (None, true) => Arc::new(Tokenizer::default_bpe()),
            (None, false) => return Err(format!("{fam}: the checkpoint has no tokenizer")),
        };
        let ctx = ["ctx", "max_position_embeddings"]
            .iter()
            .find_map(|v| model.env.var(v).ok().filter(|v| !v.is_null()))
            .ok_or_else(|| {
                format!("{fam}: config sets neither `ctx` nor `max_position_embeddings`")
            })?
            .usize()?;
        let shapes = ShapeCache::new(TEXT);
        let v = vars(1, "last", &[], 0);
        let decode = shapes.get(&v, || {
            build(&model.spec, TEXT, &model.env, &model.store, &v)
        })?;
        let media = match &rt.media {
            Some(raw) => media::resolve(raw, &model.env).map_err(|e| format!("{fam}: {e}"))?,
            None => Media::new(),
        };
        // Each tower is probed with a synthetic item: its weights count as read, and a tower that
        // does not build leaves its modality off (said when a prompt attaches one).
        let towers = Towers::new(mode);
        let mut used: HashSet<String> = decode.weights.iter().cloned().collect();
        used.extend(crate::runtime::draft::drafter_weights(&model)?);
        // `runtime.seqcls.head`: a checkpoint with the score head's weights (a
        // `*ForSequenceClassification` export) reads them through that instance, which runs on
        // this program (`run_head`), so they load with it; plain LM exports do not build it.
        let mut program_weights = decode.weights.clone();
        let mut seqcls_err = None;
        if let Some(h) = model.spec.runtime["seqcls"]["head"].as_str() {
            for head in [h, "hidden"] {
                let hv = vars(1, head, &[], 0);
                match shapes.get(&hv, || {
                    build(&model.spec, TEXT, &model.env, &model.store, &hv)
                }) {
                    Ok(i) => {
                        for w in &i.weights {
                            if model.env.has(w) && used.insert(w.clone()) {
                                program_weights.push(w.clone());
                            }
                        }
                    }
                    // Named with the unread-tensor error below when the checkpoint has a head.
                    Err(e) if head == h => seqcls_err = Some(e),
                    Err(_) => {}
                }
            }
        }
        let mut off = HashMap::new();
        for (name, s) in &media {
            let probe = media::measure(&s.preprocess, &media::probe_item(&s.preprocess))
                .and_then(|g| towers.instance(&model, &s.component, &g));
            match probe {
                Ok(i) => used.extend(i.weights.iter().cloned()),
                Err(e) => {
                    tracing::warn!("{fam}: {name} input is off: {e}");
                    off.insert(name.clone(), e);
                }
            }
        }
        let unused = model.store.unconsumed(&used, &model.ignore);
        if !unused.is_empty() {
            // A failed tower's weights stay unread: its probe error is the real cause.
            let mut why: Vec<String> = off.iter().map(|(m, e)| format!("{m} tower: {e}")).collect();
            why.extend(seqcls_err.map(|e| format!("seqcls head: {e}")));
            return Err(format!(
                "{fam}: checkpoint tensors no instance reads (list deliberate ones in `ignore`): {}{}",
                unused.join(", "),
                if why.is_empty() { String::new() } else { format!(" ({})", why.join("; ")) }
            ));
        }
        let weights = model.store.materialize(&program_weights)?;
        let embed = EmbedTable::find(&decode.graph, &weights).map(Arc::new);
        let program = GraphProgram::compile_reshapable(decode.graph.clone(), weights, mode)?;
        let recurrent = program.recurrent_states();
        let contiguous = program.contiguous_states();
        let kv_pool = program.kv_pool.clone();
        let streams = decode.output_index("logits") == Some(0)
            && decode
                .inputs
                .iter()
                .map(|i| i.name.as_str())
                .eq(["token_ids", "positions"]);
        Ok(Self {
            rt,
            media,
            towers,
            off,
            tok,
            ctx,
            shapes,
            program: Mutex::new(program),
            recurrent,
            contiguous,
            kv_pool,
            streams,
            weights: decode.weights.len(),
            mode,
            drafter: Default::default(),
            embed,
            model: Arc::new(model),
        })
    }

    fn seq(state: &mut SeqState) -> Result<&mut Seq, String> {
        state
            .downcast_mut::<Seq>()
            .ok_or_else(|| "sequence state was not built by this model".into())
    }

    /// Whether a cached prefix resumes only at an exported boundary: ring pages or contiguous
    /// states hold the sequence's state at one exact length.
    fn needs_boundary(&self) -> bool {
        !self.contiguous.is_empty()
            || self
                .kv_pool
                .as_ref()
                .is_some_and(|p| p.classes.iter().any(|c| c.window.is_some()))
    }

    fn instance(
        &self,
        rows: usize,
        head: &str,
        taps: &[usize],
        media: usize,
    ) -> Result<Arc<Instance>, String> {
        let v = vars(rows, head, taps, media);
        let m = &self.model;
        self.shapes
            .get(&v, || build(&m.spec, TEXT, &m.env, &m.store, &v))
    }

    /// `ids` with `media` laid out by the spec: each item measured (`encode`: also preprocessed
    /// and run through its tower), placeholders expanded.
    fn layout(
        &self,
        ids: &[u32],
        items: &[MediaInput],
        encode: bool,
    ) -> Result<(Expanded, Vec<(String, usize, Vec<f32>)>), String> {
        let mut grids: Vec<(&str, Grid, &[u8])> = Vec::with_capacity(items.len());
        for it in items {
            let (kind, bytes) = match it {
                MediaInput::Image(b) => ("image", b),
                MediaInput::Audio(b) => ("audio", b),
            };
            let Some((name, s)) = self.media.get_key_value(kind) else {
                return Err(format!("this model takes no {kind} input"));
            };
            if let Some(e) = self.off.get(kind) {
                return Err(format!("{kind} input is unavailable: {e}"));
            }
            let g = media::measure(&s.preprocess, bytes).map_err(|e| format!("{kind}: {e}"))?;
            grids.push((name.as_str(), g, bytes));
        }
        let refs: Vec<(&str, &Grid)> = grids.iter().map(|(k, g, _)| (*k, g)).collect();
        // Expansion text sits inside the prompt: no BOS of its own.
        let tok = |s: &str| {
            let v = self.tok.encode(s);
            Ok(match self.tok.bos {
                Some(b) if v.first() == Some(&b) => v[1..].to_vec(),
                _ => v,
            })
        };
        let e = media::expand(ids, &self.media, &refs, &tok)?;
        if e.ids.len() >= self.ctx {
            return Err(format!(
                "prompt with its media is {} tokens; the context holds {}",
                e.ids.len(),
                self.ctx
            ));
        }
        let mut rows: Vec<(String, usize, Vec<f32>)> = Vec::new();
        if encode {
            for (kind, g, bytes) in &grids {
                let s = &self.media[*kind];
                let item =
                    media::preprocess(&s.preprocess, bytes).map_err(|e| format!("{kind}: {e}"))?;
                let out = self.towers.run(&self.model, &s.component, &item)?;
                for (i, (name, v)) in out.into_iter().enumerate() {
                    let name = if i == 0 {
                        "media_rows".to_string()
                    } else {
                        name
                    };
                    if v.len() % g.tokens != 0 {
                        return Err(format!(
                            "{kind}: tower output `{name}` of {} values is not {} rows",
                            v.len(),
                            g.tokens
                        ));
                    }
                    let w = v.len() / g.tokens;
                    match rows.iter_mut().find(|(n, _, _)| *n == name) {
                        Some((_, rw, r)) if *rw == w => r.extend(v),
                        Some((_, rw, _)) => {
                            return Err(format!(
                                "tower output `{name}` is {w} wide, earlier items {rw}"
                            ))
                        }
                        None => rows.push((name, w, v)),
                    }
                }
            }
            if let Some((_, w, r)) = rows.first() {
                if r.len() / w != e.media_index.len() {
                    return Err(format!(
                        "towers returned {} media rows for {}",
                        r.len() / w,
                        e.media_index.len()
                    ));
                }
            }
        }
        Ok((e, rows))
    }

    fn drain_stream(&self, seq: &mut Seq) -> Result<(), String> {
        if seq.state.stream.0.is_some() {
            let mut p = self.program.lock();
            std::mem::swap(&mut p.state, &mut seq.state);
            let out = p.stream_end();
            std::mem::swap(&mut p.state, &mut seq.state);
            if out? && !self.recurrent.is_empty() {
                seq.recurrent_ahead = true;
            }
        }
        Ok(())
    }

    /// Runs `tokens` from `start`; see [`Mode`].
    fn run(
        &self,
        tokens: &[u32],
        start: usize,
        state: &mut SeqState,
        mode: Mode,
    ) -> Result<Tensor, String> {
        let seq = Self::seq(state)?;
        self.drain_stream(seq)?;
        if seq.recurrent_ahead {
            return Err(format!(
                "recurrent state ran past position {} in a decode stream; start a new sequence",
                seq.len
            ));
        }
        if tokens.is_empty() {
            return Err("forward needs at least one token".into());
        }
        if start != seq.asked {
            return Err(format!(
                "forward at position {start}, but the sequence continues at {}",
                seq.asked
            ));
        }
        let end = start + tokens.len();
        if end > self.ctx {
            return Err(format!(
                "position {} is past the context length {}",
                end - 1,
                self.ctx
            ));
        }
        if end <= seq.len {
            // A block ran ahead: its end's logits answer every position inside it.
            if mode == Mode::All {
                return Err("cannot verify tokens inside a bidirectional block".into());
            }
            seq.asked = end;
            return Ok(Tensor::F32(seq.ahead.clone()));
        }
        let blocks = if self.rt.bidir_media_blocks {
            seq.blocks.clone()
        } else {
            Vec::new()
        };
        let stop = blocks
            .iter()
            .find(|b| b.start < end && end < b.end)
            .map_or(end, |b| b.end);
        if stop > end && mode == Mode::All {
            return Err("cannot verify tokens ending inside a bidirectional block".into());
        }
        let ckpt = mode == Mode::All && !self.recurrent.is_empty();
        if !ckpt || seq.ckpt_from + seq.ckpt.len() != seq.len {
            seq.ckpt.clear();
            seq.ckpt_from = seq.len;
        }
        let ids = (seq.len..stop)
            .map(|q| match q < end {
                true => Ok(tokens[q - start]),
                false => seq.prompt.get(q).copied().ok_or_else(|| {
                    format!("position {q} of a bidirectional block is past the prepared prompt")
                }),
            })
            .collect::<Result<Vec<u32>, String>>()?;
        let mut p = self.program.lock();
        std::mem::swap(&mut p.state, &mut seq.state);
        let out = self.passes(&mut p, seq, &ids, end, &blocks, mode, ckpt);
        std::mem::swap(&mut p.state, &mut seq.state);
        out
    }

    /// Runs `ids` from `seq.len` (`p.state` is the sequence's) in passes of at most `prefill_rows`
    /// rows (1 when checkpointing), each grown to the end of a block it would split.
    #[allow(clippy::too_many_arguments)]
    fn passes(
        &self,
        p: &mut GraphProgram,
        seq: &mut Seq,
        ids: &[u32],
        end: usize,
        blocks: &[Range<usize>],
        mode: Mode,
        ckpt: bool,
    ) -> Result<Tensor, String> {
        let (from, cap) = (seq.len, if ckpt { 1 } else { self.rt.prefill_rows.max(1) });
        let stop = from + ids.len();
        seq.ids.truncate(from);
        seq.ids.extend_from_slice(ids);
        let refs: Vec<&str> = self.recurrent.iter().map(String::as_str).collect();
        let head = if mode == Mode::All { "all" } else { "last" };
        let mut logits = Vec::new();
        let mut at = from;
        while at < stop {
            let mut e = (at + cap).min(stop);
            if let Some(b) = blocks.iter().find(|b| b.start < e && e < b.end) {
                e = b.end;
            }
            let rows = e - at;
            if rows > cap.max(MAX_BLOCK_ROWS) {
                return Err(format!(
                    "a bidirectional block of {rows} rows exceeds the {MAX_BLOCK_ROWS}-row pass"
                ));
            }
            if ckpt && rows > 1 {
                return Err(
                    "cannot checkpoint recurrent state inside a bidirectional block".into(),
                );
            }
            let nm = seq.media_in(at, e).len();
            let inst = self.instance(rows, head, &seq.taps, nm)?;
            let li = inst
                .output_index("logits")
                .ok_or("the text component has no `logits` output")?;
            let pass = Pass {
                hist: &seq.ids[..e],
                pos0: at,
                blocks,
                seq,
            };
            let inputs = inst
                .inputs
                .iter()
                .map(|d| input(d, &pass))
                .collect::<Result<Vec<_>, _>>()?;
            // The compiled program is the decode instance; every other shape runs reshaped.
            let mut out = if rows == 1 && head == "last" && seq.taps.is_empty() && nm == 0 {
                p.run(inputs)?
            } else {
                p.run_reshaped(&inst.graph, inputs)?
            };
            // Moved out, not copied: a vocabulary row per decode token.
            match out
                .get_mut(li)
                .map(|t| std::mem::replace(t, Tensor::U32(Vec::new())))
            {
                Some(Tensor::F32(v)) if mode == Mode::All => logits.extend_from_slice(&v),
                Some(Tensor::F32(v)) => logits = v,
                _ => return Err("the text component returned no f32 logits".into()),
            }
            if !seq.taps.is_empty() {
                let taps = seq
                    .taps
                    .iter()
                    .map(|l| {
                        let i = inst
                            .output_index(&format!("layer.{l}"))
                            .ok_or_else(|| format!("tap `layer.{l}` missing"))?;
                        out[i].as_f32()
                    })
                    .collect::<Result<Vec<&[f32]>, String>>()?;
                for r in 0..rows {
                    for t in &taps {
                        let h = t.len() / rows;
                        seq.tapped.extend_from_slice(&t[r * h..(r + 1) * h]);
                    }
                }
            }
            if ckpt {
                seq.ckpt.push(p.state.snapshot(&refs));
            }
            seq.len = e;
            at = e;
        }
        seq.asked = end;
        if stop > end {
            seq.ahead.clone_from(&logits);
        }
        Ok(Tensor::F32(logits))
    }
}

impl SpecLm {
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// `ids` from position 0 on a fresh state, in passes of `prefill_rows` rows (instance head
    /// `hidden` before the last pass, `head` for it); output `out` of the last pass. `extra`
    /// fills inputs by name for the last pass (a row index there is relative to that pass: see
    /// [`Self::last_pass_start`]).
    pub fn run_head(
        &self,
        ids: &[u32],
        head: &str,
        out: &str,
        extra: &[(&str, Tensor)],
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty() || ids.len() > self.ctx {
            return Err(format!(
                "{} tokens: the context holds 1..={}",
                ids.len(),
                self.ctx
            ));
        }
        let cap = self.rt.prefill_rows.max(1);
        let mut p = self.program.lock();
        let saved = std::mem::replace(&mut p.state, StateStore::new());
        let seq = Seq::new(Vec::new(), Expanded::default(), Vec::new());
        let mut run = || -> Result<Vec<f32>, String> {
            let mut at = 0;
            loop {
                let e = (at + cap).min(ids.len());
                let last = e == ids.len();
                let inst = self.instance(e - at, if last { head } else { "hidden" }, &[], 0)?;
                let inputs = inst
                    .inputs
                    .iter()
                    .map(|d| match extra.iter().find(|x| x.0 == d.name) {
                        Some((_, t)) if last => Ok(t.clone()),
                        _ => input(
                            d,
                            &Pass {
                                hist: &ids[..e],
                                pos0: at,
                                blocks: &[],
                                seq: &seq,
                            },
                        ),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let res = p.run_reshaped(&inst.graph, inputs)?;
                if e == ids.len() {
                    let i = inst
                        .output_index(out)
                        .ok_or_else(|| format!("the `{head}` instance has no output `{out}`"))?;
                    return res[i].as_f32().map(<[f32]>::to_vec);
                }
                at = e;
            }
        };
        let r = run();
        p.state = saved;
        r
    }

    /// Row width of output `out` of the one-row `head` instance.
    pub fn output_width(&self, head: &str, out: &str) -> Result<usize, String> {
        let inst = self.instance(1, head, &[], 0)?;
        let i = inst
            .output_index(out)
            .ok_or_else(|| format!("the `{head}` instance has no output `{out}`"))?;
        let v = &inst.graph.values[inst.graph.outputs[i].0 as usize];
        v.shape
            .last()
            .copied()
            .ok_or_else(|| format!("output `{out}` has no shape"))
    }

    /// Start position of the last pass [`Self::run_head`] makes over `n` ids.
    pub fn last_pass_start(&self, n: usize) -> usize {
        let cap = self.rt.prefill_rows.max(1);
        (n.max(1) - 1) / cap * cap
    }
}

impl NativeLm for SpecLm {
    fn backend_label(&self) -> String {
        self.program.lock().backend_label()
    }

    fn device(&self) -> Device {
        self.program.lock().primary_device()
    }

    fn summary(&self) -> String {
        format!(
            "{} spec: {} weights, context {}",
            self.model.spec.family, self.weights, self.ctx
        )
    }

    fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        Some(self.program.lock().placement_summary())
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    fn context_len(&self) -> usize {
        self.ctx
    }

    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        if media.is_empty() && self.media.is_empty() {
            return Ok(ids.len());
        }
        Ok(self.layout(&ids, media, false)?.0.ids.len())
    }

    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String> {
        if media.is_empty() && self.media.is_empty() {
            return Ok((
                ids.clone(),
                Box::new(Seq::new(ids, Expanded::default(), Vec::new())),
            ));
        }
        if self.media.is_empty() {
            return Err("this model takes text only (no image or audio input)".into());
        }
        let (mut e, rows) = self.layout(&ids, media, true)?;
        let ids = std::mem::take(&mut e.ids);
        Ok((ids.clone(), Box::new(Seq::new(ids, e, rows))))
    }

    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        logits_of(self.run(tokens, start_pos, state, Mode::Last)?)
    }

    fn stream_greedy_token(
        &self,
        last: u32,
        start_pos: usize,
        max_new: usize,
        state: &mut SeqState,
    ) -> Result<Option<u32>, String> {
        let seq = Self::seq(state)?;
        if !self.streams
            || seq.recurrent_ahead
            || !seq.taps.is_empty()
            || start_pos != seq.asked
            || seq.len != seq.asked
        {
            return Ok(None);
        }
        let mut p = self.program.lock();
        std::mem::swap(&mut p.state, &mut seq.state);
        let out = p.stream_next(last, start_pos as u32, max_new, self.tok.vocab_size());
        std::mem::swap(&mut p.state, &mut seq.state);
        if let Ok(Some(_)) = out {
            (seq.len, seq.asked) = (start_pos + 1, start_pos + 1);
            seq.ids.truncate(start_pos);
            seq.ids.push(last);
            seq.ckpt.clear();
        }
        out
    }

    fn stream_token(
        &self,
        last: u32,
        start_pos: usize,
        max_new: usize,
        temp: f32,
        top_p: f32,
        recent: &[u32],
        state: &mut SeqState,
    ) -> Result<Option<u32>, String> {
        if temp <= 1e-4 {
            return self.stream_greedy_token(last, start_pos, max_new, state);
        }
        let seq = Self::seq(state)?;
        if !self.streams
            || seq.recurrent_ahead
            || !seq.taps.is_empty()
            || start_pos != seq.asked
            || seq.len != seq.asked
        {
            return Ok(None);
        }
        let mut p = self.program.lock();
        std::mem::swap(&mut p.state, &mut seq.state);
        let out = p.stream_next_sampled(
            last,
            start_pos as u32,
            max_new,
            self.tok.vocab_size(),
            temp,
            top_p,
            1.1,
            recent,
        );
        std::mem::swap(&mut p.state, &mut seq.state);
        if let Ok(Some(_)) = out {
            (seq.len, seq.asked) = (start_pos + 1, start_pos + 1);
            seq.ids.truncate(start_pos);
            seq.ids.push(last);
            seq.ckpt.clear();
        }
        out
    }

    fn stream_end(&self, state: &mut SeqState) -> Result<(), String> {
        let seq = Self::seq(state)?;
        self.drain_stream(seq)
    }

    fn forward_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        logits_of(self.run(tokens, start_pos, state, Mode::All)?)
    }

    fn truncate(&self, state: &mut SeqState, len: usize) -> Result<(), String> {
        let seq = Self::seq(state)?;
        self.drain_stream(seq)?;
        if seq.recurrent_ahead {
            return Err(format!(
                "recurrent state ran past position {} in a decode stream; start a new sequence",
                seq.len
            ));
        }
        if len > seq.asked {
            return Err(format!("cannot truncate {} positions to {len}", seq.asked));
        }
        if len < seq.len && !self.recurrent.is_empty() {
            // Attention reads only the rows below its position (later writes overwrite the rest);
            // recurrent states come back from the snapshot taken after position len - 1.
            let held = seq.ckpt.len();
            if len <= seq.ckpt_from || len > seq.ckpt_from + held {
                return Err(format!(
                    "recurrent state rewinds only into its last verified span {}..={}, not to {len}",
                    seq.ckpt_from + 1,
                    seq.ckpt_from + held
                ));
            }
            let snap = seq.ckpt.swap_remove(len - seq.ckpt_from - 1);
            seq.state.restore(snap);
        }
        seq.ckpt.clear();
        seq.ahead.clear();
        (seq.len, seq.asked) = (len, len);
        seq.ids.truncate(len);
        if let Some(kv) = seq.state.kv.as_mut() {
            kv.release_from(len);
        }
        Ok(())
    }

    fn set_taps(&self, state: &mut SeqState, layers: &[usize]) -> Result<(), String> {
        // Builds (and caches) a tapped instance: an unknown layer fails here, not mid-decode.
        self.instance(1, "last", layers, 0)?;
        Self::seq(state)?.taps = layers.to_vec();
        Ok(())
    }

    fn take_taps(&self, state: &mut SeqState) -> Vec<f32> {
        Self::seq(state).map_or_else(|_| Vec::new(), |s| std::mem::take(&mut s.tapped))
    }

    fn drafter(&self) -> Option<Arc<dyn crate::scheduler::spec::Drafter>> {
        self.drafter
            .get_or_init(|| {
                crate::runtime::draft::spec_drafter(&self.model, self.mode)
                    .map_err(|e| tracing::warn!("drafter: {e}"))
                    .ok()
                    .flatten()
            })
            .clone()
    }

    fn embed_row(&self, id: u32, out: &mut [f32]) -> bool {
        self.embed.as_ref().is_some_and(|t| t.row(id, out))
    }

    fn online_drafter(
        &self,
        knobs: Arc<crate::runtime::noise_draft::LiveDraftKnobs>,
    ) -> Option<Arc<dyn crate::scheduler::spec::Drafter>> {
        let vocab = self.model.env.var("vocab").and_then(|v| v.usize()).ok()?;
        let table = self.embed.clone();
        let hidden = table.as_ref().map_or(0, |t| t.cols);
        Some(crate::runtime::noise_draft::OnlineNoiseDrafter::new(
            Arc::new(move |id, out: &mut [f32]| table.as_ref().is_some_and(|t| t.row(id, out))),
            hidden,
            vocab,
            knobs,
        ))
    }

    fn thinking_delimiters(&self) -> Vec<(String, String)> {
        self.rt
            .thinking_delimiters
            .chunks_exact(2)
            .map(|p| (p[0].clone(), p[1].clone()))
            .collect()
    }

    fn has_vision(&self) -> bool {
        self.media.contains_key("image") && !self.off.contains_key("image")
    }

    fn kv_export(&self, state: &SeqState, len: usize) -> Option<crate::kv::KvExport> {
        let seq = state.downcast_ref::<Seq>()?;
        if len == 0
            || seq.len < len
            || seq.recurrent_ahead
            || !seq.blocks.is_empty()
            || !seq.media.media_index.is_empty()
            || !seq.taps.is_empty()
        {
            return None;
        }
        let boundary = self.needs_boundary();
        if boundary && seq.len != len {
            return None;
        }
        let recurrent = boundary.then(|| {
            let names: Vec<&str> = self.contiguous.iter().map(String::as_str).collect();
            seq.state.snapshot(&names)
        });
        match &seq.state.kv {
            Some(v) if v.evicted == 0 => v.export(len, recurrent).ok(),
            Some(_) => None,
            None if self.kv_pool.is_none() => Some(crate::kv::KvExport {
                pool: None,
                pages: Vec::new(),
                boundary: recurrent.map(|recurrent| crate::kv::Boundary {
                    len,
                    rings: Vec::new(),
                    recurrent,
                    pool: None,
                }),
            }),
            None => None,
        }
    }

    fn kv_restore(&self, state: &mut SeqState, m: &crate::kv::Match) -> Option<usize> {
        let seq = Self::seq(state).ok()?;
        if seq.len != 0
            || !seq.blocks.is_empty()
            || !seq.media.media_index.is_empty()
            || !seq.taps.is_empty()
        {
            return None;
        }
        let same_pool = match (&self.kv_pool, &m.pool) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (a, b) => a.is_none() && b.is_none(),
        };
        let b = m.boundary.as_deref();
        let len = if self.needs_boundary() { b?.len } else { m.len };
        if !same_pool || len == 0 || len > m.len || len > seq.prompt.len() {
            return None;
        }
        let rings = b.filter(|b| b.len == len).map_or(&[][..], |b| &b.rings[..]);
        seq.state = StateStore::new();
        if let Some(p) = &self.kv_pool {
            seq.state.kv = Some(crate::kv::KvView::resume(p.clone(), len, &m.pages, rings).ok()?);
        }
        if self.needs_boundary() {
            seq.state.restore(b?.recurrent.clone());
        }
        seq.len = len;
        seq.asked = len;
        seq.ids = seq.prompt[..len].to_vec();
        seq.ahead.clear();
        seq.ckpt.clear();
        seq.ckpt_from = 0;
        seq.recurrent_ahead = false;
        Some(len)
    }

    fn kv_pool(&self) -> Option<Arc<crate::kv::KvPool>> {
        self.kv_pool.clone()
    }
}

/// Index of the first largest element of `slice` (NaNs never win; 0 when nothing is above -inf).
/// Two passes with 16 independent running maxima so the max pass vectorises: the single-pass
/// `if v > best` scan is a loop-carried branch, ~0.12 ms for a 262k vocabulary on the decode path.
fn argmax_first(slice: &[f32]) -> usize {
    let mut lanes = [f32::NEG_INFINITY; 16];
    let chunks = slice.chunks_exact(16);
    let tail = chunks.remainder();
    for c in chunks {
        for (m, &v) in lanes.iter_mut().zip(c) {
            *m = m.max(v);
        }
    }
    let max = tail
        .iter()
        .chain(&lanes)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    slice.iter().position(|&v| v == max).unwrap_or(0)
}

/// Samples a token id from `slice` (logits over the tokenizer's vocabulary) with repetition
/// penalty, top-k, and nucleus top-p; `temperature <= 1e-4` is greedy argmax. Shared by every
/// causal LM the scheduler runs.
pub fn sample_logits(
    slice: &[f32],
    recent_tokens: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> u32 {
    if slice.is_empty() {
        return 0;
    }
    if temperature <= 1e-4 {
        if repetition_penalty <= 1.0 || recent_tokens.is_empty() {
            return argmax_first(slice) as u32;
        }

        // When repetition penalty > 1.0, find global unpenalized argmax first.
        let best_idx = argmax_first(slice);

        // Penalty strictly decreases/non-increases logits. If the unpenalized max
        // is not among recent tokens, it is guaranteed to remain the global max.
        if !recent_tokens.contains(&(best_idx as u32)) {
            return best_idx as u32;
        }

        let mut penalized_best_idx = 0;
        let mut penalized_best_val = f32::NEG_INFINITY;
        for (i, &raw_val) in slice.iter().enumerate() {
            let tok_id = i as u32;
            let count = recent_tokens.iter().filter(|&&t| t == tok_id).count();
            let val = if count > 0 {
                let mut v = raw_val;
                for _ in 0..count {
                    if v > 0.0 {
                        v /= repetition_penalty;
                    } else {
                        v *= repetition_penalty;
                    }
                }
                v - (count as f32 * 1.5)
            } else {
                raw_val
            };
            if val > penalized_best_val {
                penalized_best_val = val;
                penalized_best_idx = i;
            }
        }
        return penalized_best_idx as u32;
    }

    let (active_candidates, active_probs, n) = nucleus(
        slice,
        recent_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    );
    let active_candidates = &active_candidates[..n];
    let active_probs = &active_probs[..n];
    let active_sum: f32 = active_probs.iter().sum();

    // 3. Step-varying seeded PRNG mixing token history and active distribution
    let mut hash =
        0x811c9dc5u32.wrapping_add((recent_tokens.len() as u32).wrapping_mul(0x9e3779b9));
    for (pos, &tok) in recent_tokens.iter().enumerate() {
        hash ^= tok.wrapping_mul(0x5bd1e995).wrapping_add(pos as u32);
        hash = hash.rotate_left(13).wrapping_mul(0x01000193);
    }
    for &(idx, val) in active_candidates {
        hash ^= (idx as u32).wrapping_mul(0x27d4eb2d);
        hash ^= val.to_bits();
        hash = hash.rotate_left(7).wrapping_mul(0x01000193);
    }
    let r = (((hash & 0x007f_ffff) as f32) / 8_388_608.0) * active_sum;

    let mut cum_prob = 0.0f32;
    for (i, &p) in active_probs.iter().enumerate() {
        cum_prob += p;
        if cum_prob >= r || i == active_probs.len() - 1 {
            return active_candidates[i].0 as u32;
        }
    }

    active_candidates[0].0 as u32
}

/// The candidates [`sample_logits`] draws from at `temperature > 1e-4` (applying `repetition_penalty`
/// on the fly for `recent_tokens`): top-k (`<= 64`) by `(logit desc, id asc)`, softmax at
/// `temperature`, cut at cumulative `top_p`. Returns `(candidates, probs, count)`.
pub(crate) fn nucleus(
    logits: &[f32],
    recent_tokens: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> ([(usize, f32); 64], [f32; 64], usize) {
    assert!(top_k <= 64, "top_k ({top_k}) must be <= 64");
    let k = top_k.max(1).min(logits.len());
    let counts = if repetition_penalty > 1.0 && !recent_tokens.is_empty() {
        let mut m = HashMap::with_capacity(recent_tokens.len());
        for &tok in recent_tokens {
            *m.entry(tok).or_insert(0u32) += 1;
        }
        Some(m)
    } else {
        None
    };

    let mut cands = [(0usize, f32::NEG_INFINITY); 64];
    let mut len = 0usize;
    for (i, &raw) in logits.iter().enumerate() {
        let mut val = raw;
        if let Some(m) = &counts {
            if let Some(&c) = m.get(&(i as u32)) {
                for _ in 0..c {
                    if val > 0.0 {
                        val /= repetition_penalty;
                    } else {
                        val *= repetition_penalty;
                    }
                    val -= 1.5;
                }
            }
        }
        if len == k && val <= cands[k - 1].1 {
            continue;
        }
        let mut pos = len;
        if len < k {
            len += 1;
        }
        while pos > 0 && val > cands[pos - 1].1 {
            if pos < k {
                cands[pos] = cands[pos - 1];
            }
            pos -= 1;
        }
        if pos < k {
            cands[pos] = (i, val);
        }
    }

    let max_val = cands[..len]
        .iter()
        .map(|&(_, val)| val / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = [0.0f32; 64];
    let mut sum_exp = 0.0f32;
    for (p, &(_, val)) in probs[..len].iter_mut().zip(&cands[..len]) {
        let e = ((val / temperature) - max_val).exp();
        *p = e;
        sum_exp += e;
    }
    let denom = sum_exp.max(1e-12);
    for p in &mut probs[..len] {
        *p /= denom;
    }

    // Nucleus (top-p) cutoff.
    let p_thresh = top_p.clamp(0.01, 1.0);
    let mut cum_p = 0.0f32;
    let mut cutoff_idx = len;
    for (i, &p) in probs[..len].iter().enumerate() {
        cum_p += p;
        if cum_p >= p_thresh {
            cutoff_idx = i + 1;
            break;
        }
    }
    (cands, probs, cutoff_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};

    /// The vectorised argmax equals the scalar first-wins scan: ties, NaNs, every length mod 16.
    #[test]
    fn argmax_first_matches_the_scalar_scan() {
        let mut seed = 0x9E37_79B9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for len in [1usize, 2, 15, 16, 17, 33, 100, 4099] {
            for _ in 0..20 {
                let v: Vec<f32> = (0..len)
                    .map(|_| match next() % 11 {
                        0 => f32::NAN,
                        1 => f32::NEG_INFINITY,
                        r => (r % 4) as f32 - 1.0, // few distinct values: many ties
                    })
                    .collect();
                let (mut best, mut idx) = (f32::NEG_INFINITY, 0);
                for (i, &x) in v.iter().enumerate() {
                    if x > best {
                        best = x;
                        idx = i;
                    }
                }
                assert_eq!(argmax_first(&v), idx, "{v:?}");
            }
        }
    }

    /// Allocation-free `sample_logits` / `nucleus` matches the brute-force full-sort reference
    /// across 200 seeded logit vectors (vocab 1,000 and 262,144; temperatures 0.3/0.7/1.2;
    /// top-p 0.5/0.9/1.0; with and without repetition penalty, including exact tie-breaking).
    #[test]
    fn sample_logits_matches_brute_force_reference() {
        fn brute_force(
            slice: &[f32],
            recent_tokens: &[u32],
            temperature: f32,
            top_k: usize,
            top_p: f32,
            repetition_penalty: f32,
        ) -> u32 {
            let mut working_logits = slice.to_vec();
            if repetition_penalty > 1.0 && !recent_tokens.is_empty() {
                for &tok in recent_tokens {
                    let idx = tok as usize;
                    if idx < working_logits.len() {
                        if working_logits[idx] > 0.0 {
                            working_logits[idx] /= repetition_penalty;
                        } else {
                            working_logits[idx] *= repetition_penalty;
                        }
                        working_logits[idx] -= 1.5;
                    }
                }
            }
            let k = top_k.max(1).min(working_logits.len());
            let mut indexed: Vec<(usize, f32)> =
                working_logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            indexed.truncate(k);
            let max_val = indexed
                .iter()
                .map(|&(_, val)| val / temperature)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = indexed
                .iter()
                .map(|&(_, val)| ((val / temperature) - max_val).exp())
                .collect();
            let sum_exp: f32 = exp_vals.iter().sum();
            let mut probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp.max(1e-12)).collect();
            let p_thresh = top_p.clamp(0.01, 1.0);
            let mut cum_p = 0.0f32;
            let mut cutoff_idx = indexed.len();
            for (i, &p) in probs.iter().enumerate() {
                cum_p += p;
                if cum_p >= p_thresh {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff_idx);
            probs.truncate(cutoff_idx);
            let active_sum: f32 = probs.iter().sum();
            let mut hash =
                0x811c9dc5u32.wrapping_add((recent_tokens.len() as u32).wrapping_mul(0x9e3779b9));
            for (pos, &tok) in recent_tokens.iter().enumerate() {
                hash ^= tok.wrapping_mul(0x5bd1e995).wrapping_add(pos as u32);
                hash = hash.rotate_left(13).wrapping_mul(0x01000193);
            }
            for &(idx, val) in &indexed {
                hash ^= (idx as u32).wrapping_mul(0x27d4eb2d);
                hash ^= val.to_bits();
                hash = hash.rotate_left(7).wrapping_mul(0x01000193);
            }
            let r = (((hash & 0x007f_ffff) as f32) / 8_388_608.0) * active_sum;
            let mut cum_prob = 0.0f32;
            for (i, &p) in probs.iter().enumerate() {
                cum_prob += p;
                if cum_prob >= r || i == probs.len() - 1 {
                    return indexed[i].0 as u32;
                }
            }
            indexed[0].0 as u32
        }

        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        let temps = [0.3f32, 0.7, 1.2];
        let top_ps = [0.5f32, 0.9, 1.0];
        let penalties = [1.0f32, 1.15];
        for i in 0..200 {
            let vocab = if i >= 192 { 262_144 } else { 1_000 };
            let ties = i % 5 == 0;
            let logits: Vec<f32> = (0..vocab)
                .map(|_| {
                    let u = next();
                    if ties {
                        ((u >> 28) as f32 - 8.0) * 0.5
                    } else {
                        ((u >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 8.0
                    }
                })
                .collect();
            let top_id = argmax_first(&logits) as u32;
            let recent: Vec<u32> = if i % 7 == 0 {
                Vec::new()
            } else {
                (0..32)
                    .map(|j| {
                        if j < 4 {
                            top_id
                        } else if j < 16 {
                            next() % 16
                        } else {
                            next() % (vocab as u32)
                        }
                    })
                    .collect()
            };
            let temp = temps[i % temps.len()];
            let top_p = top_ps[(i / 3) % top_ps.len()];
            let top_k = [1usize, 10, 50, 64][i % 4];
            let pen = penalties[(i / 2) % penalties.len()];
            let got = sample_logits(&logits, &recent, temp, top_k, top_p, pen);
            let want = brute_force(&logits, &recent, temp, top_k, top_p, pen);
            assert_eq!(
                got, want,
                "case {i} (vocab={vocab}, temp={temp}, top_p={top_p}, top_k={top_k}, pen={pen})"
            );
        }
    }

    /// A toy causal LM over synthetic weights: GQA attention with RoPE and a `visible_end` input,
    /// an optional recurrent causal conv per layer (`$CONV`), a `gen` input reading `pos0`, taps.
    const SPEC: &str = r#"{
      family: "toy", pipeline: "causal_lm",
      config: {vocab: 32, hidden: 16, heads: 2, kv_heads: 1, hd: 8, ctx: 64, layers: [0, 1], conv: $CONV},
      runtime: {prefill_rows: $ROWS, bidir_media_blocks: true},
      constants: {
        inv_freq: {gen: "rope_inv_freq", args: {dim: "hd", theta: 10000}},
        ones: {gen: "ones", args: {shape: ["hidden"]}},
      },
      synthetic: {
        emb: {shape: ["vocab", "hidden"]},
        "l{i}.q": {shape: ["heads*hd", "hidden"], over: {i: "layers"}},
        "l{i}.k": {shape: ["kv_heads*hd", "hidden"], over: {i: "layers"}},
        "l{i}.v": {shape: ["kv_heads*hd", "hidden"], over: {i: "layers"}},
        "l{i}.o": {shape: ["hidden", "heads*hd"], over: {i: "layers"}},
        "l{i}.conv": {shape: ["hidden*3"], over: {i: "layers"}},
      },
      components: {text: {
        vars: ["rows", "head"],
        inputs: [
          {name: "token_ids", dtype: "u32", shape: ["rows"]},
          {name: "positions", dtype: "u32", shape: ["rows"]},
          {name: "visible_end", dtype: "u32", shape: ["rows"], when: "rows > 1"},
          {name: "pos_f", dtype: "f32", shape: ["rows", 1], gen: "iota", args: {n: "rows", dtype: "'f32'", start: "pos0"}, when: "!conv"},
        ],
        body: [
          {op: "Embedding", in: ["w:emb", "i:token_ids"], out: "x"},
          {block: [
            {op: "Unary", args: {kind: {Scale: 0.01}}, in: ["i:pos_f"], out: "pf"},
            {op: "Binary", args: {kind: "Add"}, in: ["x", "pf"], out: "x"},
          ], when: "!conv"},
          {repeat: "i", over: "layers", body: [
            {state: "kv.{i}.k", shape: ["ctx", "kv_heads*hd"]},
            {state: "kv.{i}.v", shape: ["ctx", "kv_heads*hd"]},
            {op: "RmsNorm", args: {eps: 1e-5}, in: ["x", "c:ones"], out: "n"},
            {op: "Linear", in: ["n", "w:l{i}.q"], out: "q"},
            {op: "Linear", in: ["n", "w:l{i}.k"], out: "k"},
            {op: "Linear", in: ["n", "w:l{i}.v"], out: "v"},
            {op: "Rope", args: {head_dim: "hd", rotary_dim: "hd", variant: "HalfSplit", mscale: 1.0}, in: ["q", "i:positions", "c:inv_freq"], out: "q"},
            {op: "Rope", args: {head_dim: "hd", rotary_dim: "hd", variant: "HalfSplit", mscale: 1.0}, in: ["k", "i:positions", "c:inv_freq"], out: "k"},
            {op: "CachedAttention", args: {heads: "heads", kv_heads: "kv_heads", head_dim: "hd", v_head_dim: "hd", scale: "1/sqrt(hd)", causal: true, write: true},
             in: ["q", "k", "v", "s:kv.{i}.k", "s:kv.{i}.v", "i:positions", "i:visible_end"], out: "a", when: "rows > 1"},
            {op: "CachedAttention", args: {heads: "heads", kv_heads: "kv_heads", head_dim: "hd", v_head_dim: "hd", scale: "1/sqrt(hd)", causal: true, write: true},
             in: ["q", "k", "v", "s:kv.{i}.k", "s:kv.{i}.v", "i:positions"], out: "a", when: "rows == 1"},
            {op: "Linear", in: ["a", "w:l{i}.o"], out: "a"},
            {op: "Unary", args: {kind: {Scale: 50.0}}, in: ["a"], out: "a"},
            {op: "Binary", args: {kind: "Add"}, in: ["x", "a"], out: "x"},
            {block: [
              {state: "conv.{i}", shape: [2, "hidden"], recurrent: true},
              {op: "CausalConv1dCached", args: {kernel: 3, dilation: 1}, in: ["x", "w:l{i}.conv", "s:conv.{i}"], out: "c"},
              {op: "Binary", args: {kind: "Add"}, in: ["x", "c"], out: "x"},
            ], when: "conv"},
            {tap: "layer.{i}", in: "x"},
          ]},
          {op: "RmsNorm", args: {eps: 1e-5}, in: ["x", "c:ones"], out: "x"},
          {op: "Slice", args: {axis: 0, start: "rows - 1", end: "rows"}, in: ["x"], out: "x", when: "head == 'last' && rows > 1"},
          {op: "Linear", in: ["x", "w:emb"], out: "logits"},
        ],
        outputs: ["logits"],
      }},
    }"#;

    const V: usize = 32;
    const PROMPT: [u32; 12] = [3, 17, 5, 29, 11, 2, 8, 30, 14, 21, 6, 9];

    fn devices() -> Vec<Device> {
        let mut d = vec![Device::Cpu];
        if crate::hardware::available_devices().contains(&Device::Metal) {
            d.push(Device::Metal);
        }
        d
    }

    fn toy(dev: Device, conv: bool, rows: usize) -> SpecLm {
        let spec = SPEC
            .replace("$CONV", if conv { "true" } else { "false" })
            .replace("$ROWS", &rows.to_string());
        let reg = Registry::from_files(&[("families/toy.json5".into(), spec)], &[]).unwrap();
        let a = ModelArtifacts {
            model_id: "gibbs/toy".into(),
            ..Default::default()
        };
        let exact = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            ..Default::default()
        };
        let m = crate::arch::load::load_with(&a, &reg, Some("toy"), exact).unwrap();
        SpecLm::new(m, PlacementMode::Pin(dev)).unwrap()
    }

    fn fresh(lm: &SpecLm, ids: &[u32]) -> SeqState {
        lm.prepare(ids.to_vec(), &[]).unwrap().1
    }

    fn tol(dev: Device) -> f32 {
        if dev == Device::Metal {
            3e-3
        } else {
            1e-5
        }
    }

    #[track_caller]
    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        let scale = b.iter().fold(1e-3f32, |m, x| m.max(x.abs()));
        let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= tol * scale, "max diff {d} (scale {scale})");
    }

    /// Logits of `PROMPT` prefilled in 4-row passes equal one 64-row pass; decoding the tail one
    /// token at a time equals `forward_all`'s rows; the greedy pick is the logits' argmax.
    #[test]
    fn chunked_prefill_decode_and_forward_all_agree() {
        for dev in devices() {
            let (chunked, oneshot) = (toy(dev, false, 4), toy(dev, false, 64));
            let mut s = fresh(&chunked, &PROMPT);
            let a = chunked.forward(&PROMPT, 0, &mut s).unwrap();
            let mut s = fresh(&oneshot, &PROMPT);
            let b = oneshot.forward(&PROMPT, 0, &mut s).unwrap();
            assert_eq!(a.len(), V);
            close(&a, &b, tol(dev));

            let mut s = fresh(&oneshot, &PROMPT);
            let all = oneshot.forward_all(&PROMPT, 0, &mut s).unwrap();
            assert_eq!(all.len(), PROMPT.len() * V);
            close(&all[(PROMPT.len() - 1) * V..], &b, tol(dev));
            let mut s = fresh(&chunked, &PROMPT);
            chunked.forward(&PROMPT[..5], 0, &mut s).unwrap();
            for q in 5..PROMPT.len() {
                let l = chunked.forward(&PROMPT[q..q + 1], q, &mut s).unwrap();
                close(&l, &all[q * V..(q + 1) * V], tol(dev));
            }
            let mut s = fresh(&chunked, &PROMPT);
            chunked.forward(&PROMPT[..11], 0, &mut s).unwrap();
            let l = chunked.forward(&PROMPT[11..], 11, &mut s).unwrap();
            let t = sample_logits(&l, &[], 0.0, 50, 1.0, 1.0);
            assert_eq!(t, sample_logits(&b, &[], 0.0, 50, 1.0, 1.0), "{dev:?}");
            assert!(chunked.forward(&PROMPT[..1], 3, &mut s).is_err());
        }
    }

    /// Verify 4 tokens, rewind to 2 of them, run 2 others: equals a fresh run of that sequence,
    /// with attention only (no snapshot needed) and with a recurrent conv state (snapshots).
    #[test]
    fn truncate_then_forward_equals_a_fresh_run() {
        for dev in devices() {
            for conv in [false, true] {
                let lm = toy(dev, conv, 4);
                let mut s = fresh(&lm, &PROMPT);
                lm.forward(&PROMPT[..8], 0, &mut s).unwrap();
                lm.forward_all(&PROMPT[8..12], 8, &mut s).unwrap();
                lm.truncate(&mut s, 10).unwrap();
                let other = [1u32, 31];
                let got = lm.forward(&other, 10, &mut s).unwrap();
                let seq: Vec<u32> = [&PROMPT[..10], &other[..]].concat();
                let mut f = fresh(&lm, &seq);
                let want = lm.forward(&seq, 0, &mut f).unwrap();
                close(&got, &want, tol(dev));
                if conv {
                    // A plain forward drops the snapshots: no rewind into it.
                    assert!(lm.truncate(&mut s, 11).is_err());
                    let mut g = fresh(&lm, &PROMPT);
                    lm.forward_all(&PROMPT[..4], 0, &mut g).unwrap();
                    assert!(lm.truncate(&mut g, 0).is_err(), "before the verified span");
                }
            }
        }
    }

    /// A toy spec with a `CausalConv1dCached` recurrent state streams on Metal: 12 streamed
    /// greedy tokens equal 12 per-token `forward` + argmax tokens; ending a stream with a token
    /// in flight marks `recurrent_ahead` so `forward` and `truncate` return an error.
    #[cfg(target_os = "macos")]
    #[test]
    fn stream_decode_recurrent_matches_per_token_greedy() {
        if !crate::hardware::available_devices().contains(&Device::Metal) {
            return;
        }
        let lm = toy(Device::Metal, true, 64);
        assert!(lm.streams);
        let mut s_ref = fresh(&lm, &[PROMPT[0]]);
        let mut tok = PROMPT[0];
        let mut want = Vec::new();
        for pos in 0..12 {
            let l = lm.forward(&[tok], pos, &mut s_ref).unwrap();
            tok = argmax_first(&l[..V]) as u32;
            want.push(tok);
        }

        let mut s = fresh(&lm, &[PROMPT[0]]);
        let mut got = Vec::new();
        for pos in 0..12 {
            let last = if pos == 0 { PROMPT[0] } else { got[pos - 1] };
            let t = lm
                .stream_greedy_token(last, pos, 12 - pos, &mut s)
                .unwrap()
                .expect("recurrent model streams on Metal");
            got.push(t);
        }
        lm.stream_end(&mut s).unwrap();
        assert_eq!(got, want);
        assert!(lm.forward(&[got[11]], 12, &mut s).is_ok());

        let mut s2 = fresh(&lm, &[PROMPT[0]]);
        let t0 = lm
            .stream_greedy_token(PROMPT[0], 0, 5, &mut s2)
            .unwrap()
            .unwrap();
        lm.stream_end(&mut s2).unwrap();
        let err = lm.forward(&[t0], 1, &mut s2).unwrap_err();
        assert!(
            err.contains(
                "recurrent state ran past position 1 in a decode stream; start a new sequence"
            ),
            "{err}"
        );
        let err = lm.truncate(&mut s2, 1).unwrap_err();
        assert!(
            err.contains(
                "recurrent state ran past position 1 in a decode stream; start a new sequence"
            ),
            "{err}"
        );
    }

    /// Taps record every layer's residual per position, independent of the pass shape.
    #[test]
    fn taps_record_each_layer_per_position() {
        for dev in devices() {
            let (chunked, oneshot) = (toy(dev, true, 4), toy(dev, true, 64));
            let mut a = fresh(&chunked, &PROMPT);
            chunked.set_taps(&mut a, &[0, 1]).unwrap();
            chunked.forward(&PROMPT, 0, &mut a).unwrap();
            let ta = chunked.take_taps(&mut a);
            assert_eq!(ta.len(), PROMPT.len() * 2 * 16);
            assert!(chunked.take_taps(&mut a).is_empty());
            let mut b = fresh(&oneshot, &PROMPT);
            oneshot.set_taps(&mut b, &[0, 1]).unwrap();
            oneshot.forward(&PROMPT[..7], 0, &mut b).unwrap();
            oneshot.forward(&PROMPT[7..], 7, &mut b).unwrap();
            close(&ta, &oneshot.take_taps(&mut b), tol(dev));
            assert!(chunked.set_taps(&mut a, &[2]).is_err(), "no layer 2");
        }
    }

    /// A pass that would end inside a bidirectional block runs to its end and answers the
    /// positions it ran ahead from cache; the result equals one pass over the whole prompt, and
    /// the block changes the logits (rows see the whole block).
    #[test]
    fn bidirectional_block_extends_its_pass() {
        for dev in devices() {
            let (chunked, oneshot) = (toy(dev, false, 4), toy(dev, false, 64));
            let with_block = |lm: &SpecLm| {
                let mut s = fresh(lm, &PROMPT);
                SpecLm::seq(&mut s).unwrap().blocks = vec![3..9];
                s
            };
            let mut s = with_block(&chunked);
            // 0..6 ends inside 3..9: the pass runs 0..9.
            let ahead = chunked.forward(&PROMPT[..6], 0, &mut s).unwrap();
            assert_eq!(SpecLm::seq(&mut s).unwrap().len, 9);
            assert_eq!(chunked.forward(&PROMPT[6..8], 6, &mut s).unwrap(), ahead);
            let a = chunked.forward(&PROMPT[8..], 8, &mut s).unwrap();
            let mut s = with_block(&oneshot);
            let b = oneshot.forward(&PROMPT, 0, &mut s).unwrap();
            close(&a, &b, tol(dev));
            let mut s = with_block(&oneshot);
            close(
                &oneshot.forward(&PROMPT[..9], 0, &mut s).unwrap(),
                &ahead,
                tol(dev),
            );
            if dev == Device::Cpu {
                let mut s = fresh(&oneshot, &PROMPT);
                let plain = oneshot.forward(&PROMPT, 0, &mut s).unwrap();
                let d = plain
                    .iter()
                    .zip(&b)
                    .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                let scale = b.iter().fold(0f32, |m, x| m.max(x.abs()));
                assert!(
                    d > 1e-2 * scale,
                    "the block must change the logits ({d} of {scale})"
                );
            }
        }
    }

    /// A toy LM with an audio modality: raw 4-sample frames through a linear tower (output 0 the
    /// soft rows, output `extra` added to them before attention), rows scattered after the embedding, the
    /// item one bidirectional block; an `image` modality switched off by `when`.
    const MEDIA_SPEC: &str = r#"{
      family: "toym", pipeline: "causal_lm",
      config: {vocab: 32, hidden: 16, heads: 2, kv_heads: 1, hd: 8, ctx: 64},
      runtime: {prefill_rows: $ROWS, bidir_media_blocks: true, media: {
        audio: {component: "tower", preprocess: {kind: "raw_frames", sampling_rate: 16000, frame: 4, max_tokens: 16},
                expand: {kind: "repeat_between", placeholder: 31, open: 29, close: 30}, bidirectional: true},
        image: {when: "false", component: "nope", preprocess: {kind: "raw_frames", sampling_rate: 1, frame: 1, max_tokens: 1},
                expand: {kind: "repeat_between", placeholder: 28, open: 27, close: 26}},
      }},
      constants: {
        inv_freq: {gen: "rope_inv_freq", args: {dim: "hd", theta: 10000}},
        ones: {gen: "ones", args: {shape: ["hidden"]}},
      },
      synthetic: {
        emb: {shape: ["vocab", "hidden"]},
        tw: {shape: ["hidden", 4]}, tx: {shape: ["hidden", 4]},
        q: {shape: ["heads*hd", "hidden"]}, k: {shape: ["kv_heads*hd", "hidden"]},
        v: {shape: ["kv_heads*hd", "hidden"]}, o: {shape: ["hidden", "heads*hd"]},
      },
      components: {
        tower: {vars: ["tokens"], inputs: [{name: "frames", dtype: "f32", shape: ["tokens", 4]}],
          body: [{op: "Linear", in: ["i:frames", "w:tw"], out: "rows"}, {op: "Linear", in: ["i:frames", "w:tx"], out: "extra"}],
          outputs: ["rows", "extra"]},
        text: {
          vars: ["rows", "head"],
          inputs: [
            {name: "token_ids", dtype: "u32", shape: ["rows"]},
            {name: "positions", dtype: "u32", shape: ["rows"]},
            {name: "visible_end", dtype: "u32", shape: ["rows"], when: "rows > 1"},
            {name: "media_rows", dtype: "f32", shape: ["media", "hidden"], when: "first(media, 0) > 0"},
            {name: "media_index", dtype: "u32", shape: ["media"], when: "first(media, 0) > 0"},
            {name: "extra", dtype: "f32", shape: ["media", "hidden"], when: "first(media, 0) > 0"},
          ],
          body: [
            {op: "Embedding", in: ["w:emb", "i:token_ids"], out: "x"},
            {op: "Scatter", args: {axis: 0}, in: ["x", "i:media_index", "i:media_rows"], out: "x", when: "first(media, 0) > 0"},
            {block: [
              {op: "Unary", args: {kind: {Scale: 0.0}}, in: ["x"], out: "z"},
              {op: "Scatter", args: {axis: 0}, in: ["z", "i:media_index", "i:extra"], out: "z"},
              {op: "Binary", args: {kind: "Add"}, in: ["x", "z"], out: "x"},
            ], when: "first(media, 0) > 0"},
            {state: "kv.k", shape: ["ctx", "kv_heads*hd"]},
            {state: "kv.v", shape: ["ctx", "kv_heads*hd"]},
            {op: "RmsNorm", args: {eps: 1e-5}, in: ["x", "c:ones"], out: "n"},
            {op: "Linear", in: ["n", "w:q"], out: "q"},
            {op: "Linear", in: ["n", "w:k"], out: "k"},
            {op: "Linear", in: ["n", "w:v"], out: "v"},
            {op: "Rope", args: {head_dim: "hd", rotary_dim: "hd", variant: "HalfSplit", mscale: 1.0}, in: ["q", "i:positions", "c:inv_freq"], out: "q"},
            {op: "Rope", args: {head_dim: "hd", rotary_dim: "hd", variant: "HalfSplit", mscale: 1.0}, in: ["k", "i:positions", "c:inv_freq"], out: "k"},
            {op: "CachedAttention", args: {heads: "heads", kv_heads: "kv_heads", head_dim: "hd", v_head_dim: "hd", scale: "1/sqrt(hd)", causal: true, write: true},
             in: ["q", "k", "v", "s:kv.k", "s:kv.v", "i:positions", "i:visible_end"], out: "a", when: "rows > 1"},
            {op: "CachedAttention", args: {heads: "heads", kv_heads: "kv_heads", head_dim: "hd", v_head_dim: "hd", scale: "1/sqrt(hd)", causal: true, write: true},
             in: ["q", "k", "v", "s:kv.k", "s:kv.v", "i:positions"], out: "a", when: "rows == 1"},
            {op: "Linear", in: ["a", "w:o"], out: "a"},
            {op: "Unary", args: {kind: {Scale: 50.0}}, in: ["a"], out: "a"},
            {op: "Binary", args: {kind: "Add"}, in: ["x", "a"], out: "x"},
            {op: "Slice", args: {axis: 0, start: "rows - 1", end: "rows"}, in: ["x"], out: "x", when: "head == 'last' && rows > 1"},
            {op: "Linear", in: ["x", "w:emb"], out: "logits"},
          ],
          outputs: ["logits"],
        },
      },
    }"#;

    fn toy_media(dev: Device, rows: usize) -> SpecLm {
        let spec = MEDIA_SPEC.replace("$ROWS", &rows.to_string());
        let reg = Registry::from_files(&[("families/toym.json5".into(), spec)], &[]).unwrap();
        let a = ModelArtifacts {
            model_id: "gibbs/toym".into(),
            ..Default::default()
        };
        let exact = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            ..Default::default()
        };
        let m = crate::arch::load::load_with(&a, &reg, Some("toym"), exact).unwrap();
        SpecLm::new(m, PlacementMode::Pin(dev)).unwrap()
    }

    /// 16 kHz mono 16-bit WAV of `n` loud (near full-scale) samples.
    fn wav(n: usize) -> Vec<u8> {
        let pcm: Vec<u8> = (0..n)
            .flat_map(|i| (((i * 7919) % 60000) as i32 - 30000).to_le_bytes()[..2].to_vec())
            .collect();
        let mut b = b"RIFF".to_vec();
        b.extend((36 + pcm.len() as u32).to_le_bytes());
        b.extend(b"WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend([1, 0, 1, 0]);
        b.extend(16_000u32.to_le_bytes());
        b.extend(32_000u32.to_le_bytes());
        b.extend([2, 0, 16, 0]);
        b.extend(b"data");
        b.extend((pcm.len() as u32).to_le_bytes());
        b.extend(pcm);
        b
    }

    /// Media rows and a tower's extra output land on their rows whatever the pass split (the
    /// block runs whole), change the logits, and decode continues after them.
    #[test]
    fn media_rows_scatter_into_any_pass_split() {
        for dev in devices() {
            let (chunked, oneshot) = (toy_media(dev, 3), toy_media(dev, 64));
            let prompt = [3u32, 17, 31, 5, 9];
            let clip = [MediaInput::Audio(wav(22))]; // 6 frames of 4 (the last zero-padded)
            assert!(
                chunked
                    .prepare(prompt.to_vec(), &[MediaInput::Image(vec![1])])
                    .is_err(),
                "image is off"
            );
            let (ids, mut a) = chunked.prepare(prompt.to_vec(), &clip).unwrap();
            assert_eq!(ids, vec![3, 17, 29, 31, 31, 31, 31, 31, 31, 30, 5, 9]);
            assert_eq!(
                chunked.prompt_len(prompt.to_vec(), &clip).unwrap(),
                ids.len()
            );
            assert_eq!(SpecLm::seq(&mut a).unwrap().blocks, vec![3..9]);
            // 0..4 ends inside the block 3..9: the pass runs to 9.
            let ahead = chunked.forward(&ids[..4], 0, &mut a).unwrap();
            assert_eq!(chunked.forward(&ids[4..8], 4, &mut a).unwrap(), ahead);
            let la = chunked.forward(&ids[8..], 8, &mut a).unwrap();
            let (_, mut b) = oneshot.prepare(prompt.to_vec(), &clip).unwrap();
            let lb = oneshot.forward(&ids, 0, &mut b).unwrap();
            close(&la, &lb, tol(dev));
            let da = chunked.forward(&[7], ids.len(), &mut a).unwrap();
            close(
                &da,
                &oneshot.forward(&[7], ids.len(), &mut b).unwrap(),
                tol(dev),
            );
            if dev == Device::Cpu {
                let mut t: SeqState =
                    Box::new(Seq::new(ids.clone(), Expanded::default(), Vec::new()));
                SpecLm::seq(&mut t).unwrap().blocks = vec![3..9];
                let plain = oneshot.forward(&ids, 0, &mut t).unwrap();
                assert!(
                    plain.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-3),
                    "media rows must matter"
                );
                // The tower's second output reaches the logits through its rows.
                let (_, mut z) = oneshot.prepare(prompt.to_vec(), &clip).unwrap();
                let s = SpecLm::seq(&mut z).unwrap();
                let extra = s
                    .rows
                    .iter_mut()
                    .find(|(n, _, _)| n == "extra")
                    .expect("extra rows");
                extra.2.fill(0.0);
                let lz = oneshot.forward(&ids, 0, &mut z).unwrap();
                assert!(
                    lz.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-3),
                    "the extra output must matter"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn forward_all_rows_scaling() {
        let _ = crate::graph::metal::Runtime::shared();
        let hub = crate::model::hub::HuggingFaceHub::new();
        let reg = crate::arch::registry().unwrap();
        let models = [
            "unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf",
            "unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf",
        ];
        let mut failures = Vec::new();
        for id in models {
            let artifacts = hub.auto_download_and_cache(id, None).unwrap();
            let lm = super::load(&artifacts, reg, None, PlacementMode::Pin(Device::Metal)).unwrap();
            let prompt: Vec<u32> = (0..32).map(|i| 10 + (i as u32 % 50)).collect();
            let (ids, mut state) = lm.prepare(prompt.clone(), &[]).unwrap();
            let prefill_len = 16;
            lm.forward(&ids[..prefill_len], 0, &mut state).unwrap();
            let mut times = Vec::new();
            for m in [1usize, 2, 4, 8] {
                let step_ids = &ids[prefill_len..prefill_len + m];
                for _ in 0..2 {
                    lm.forward_all(step_ids, prefill_len, &mut state).unwrap();
                    lm.truncate(&mut state, prefill_len).unwrap();
                }
                let mut samples = Vec::new();
                for _ in 0..8 {
                    let t0 = std::time::Instant::now();
                    let out = lm.forward_all(step_ids, prefill_len, &mut state).unwrap();
                    std::hint::black_box(&out);
                    samples.push(t0.elapsed().as_secs_f64() * 1e3);
                    lm.truncate(&mut state, prefill_len).unwrap();
                }
                samples.sort_by(|a, b| a.total_cmp(b));
                times.push((m, samples[samples.len() / 2]));
            }
            let t1 = times[0].1;
            let t8 = times[3].1;
            println!(
                "{id}: m=1 {:.2}ms, m=2 {:.2}ms ({:.2}x), m=4 {:.2}ms ({:.2}x), m=8 {:.2}ms ({:.2}x)",
                times[0].1,
                times[1].1,
                times[1].1 / t1,
                times[2].1,
                times[2].1 / t1,
                t8,
                t8 / t1
            );
            if t8 > 3.0 * t1 {
                failures.push(format!("{id}: m=8 ({t8:.2}ms) > 3.0 * m=1 ({t1:.2}ms)"));
            }
        }
        if failures.is_empty() {
            println!("P10.0 verdict: GO (all models <= 3.0x at m=8)");
        } else {
            println!("P10.0 verdict: NO-GO ({})", failures.join("; "));
        }
    }
}
