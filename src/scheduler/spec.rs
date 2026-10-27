//! Speculative decoding. A [`Drafter`] proposes `k` tokens; the target verifies all `k + 1`
//! positions in one forward ([`NativeLm::forward_all`]) and the sequence keeps the longest
//! accepted prefix plus one token of the target's own. Greedy accepts exact argmax matches, so
//! outputs are identical to plain decoding; sampling uses standard rejection sampling (accept a
//! draft `x` with probability `min(1, p(x)/q(x))`, else draw from `max(0, p - q)`), so the output
//! distribution is the target's. Rejected positions are rewound with [`NativeLm::truncate`].
//! A drafter that [`Drafter::overlaps`] proposes the next step's drafts on a fork of its state
//! while the target verifies; they replace the next propose when every draft was accepted and
//! the target's bonus token is the one they assumed.
//!
//! [`NativeLm::forward_all`]: crate::model::native::NativeLm::forward_all
//! [`NativeLm::truncate`]: crate::model::native::NativeLm::truncate

use super::{recent, ExecutableModel, NO_STATE};
use crate::model::native::SeqState;
use crate::runtime::lm::{nucleus, sample_logits};
use crate::runtime::noise_draft::{DraftKnobs, DraftMode};
use crate::runtime::sched::Rng;
use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

/// A drafter's per-sequence state.
pub type DraftState = Box<dyn Any + Send + Sync>;

/// Draft logits a drafter hands `pick`: over the whole vocabulary, or over candidate ids only.
pub enum DraftLogits<'a> {
    Dense(&'a [f32]),
    Sparse { ids: &'a [u32], logits: &'a [f32] },
}

/// Proposes draft tokens from the target's hidden states.
pub trait Drafter: Send + Sync {
    /// Short name for logs and the TUI, e.g. `MTP`.
    fn label(&self) -> String;
    /// Target decoder layers whose outputs the drafter reads (tap `i` = the residual stream after
    /// layer `i`, HF `hidden_states[i + 1]`).
    fn tap_layers(&self) -> Vec<usize>;
    fn new_state(&self) -> DraftState;
    /// Proposes up to `k` tokens after `tokens` (prompt + output; the target has run all but
    /// the last). `hidden` holds the tap rows `[rows, taps * hidden]` of the target positions
    /// committed since the previous call (the prompt's on the first), in order. `pick` chooses
    /// each draft token from the draft logits: call it once per token, in order.
    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(DraftLogits<'_>) -> u32,
    ) -> Result<Vec<u32>, String>;
    /// Most tokens one [`Self::propose`] call may draft; the scheduler tunes `k` in `0..=max_k`.
    fn max_k(&self) -> usize {
        4
    }
    /// After a verify: the target's logits rows (`width` wide) for the sequence's last token and
    /// the accepted drafts; `accepted[j]` is the token committed from row `j`.
    fn observe(&self, _st: &mut DraftState, _accepted: &[u32], _logits: &[f32], _width: usize) {}
    /// A copy of `st` that a [`Self::propose`] may run ahead on during the target's verify;
    /// `None` when the drafter cannot.
    fn fork_state(&self, _st: &DraftState) -> Option<DraftState> {
        None
    }
    /// Proposals need no verified hidden state, so the next step's drafts may be proposed while
    /// the target verifies this step's.
    fn overlaps(&self) -> bool {
        false
    }
}

/// Top-k the scheduler samples with (see `ExecutableModel::sample`).
const TOP_K: usize = 50;

/// One sequence's speculative state, tied to the drafter it started with.
pub struct SpecSeq {
    drafter: Arc<dyn Drafter>,
    draft: DraftState,
    /// Tap rows of committed target positions the drafter has not seen yet.
    pending: Vec<f32>,
    /// The last step drafted, so the drafter only has that step's rows to catch up on (the
    /// prompt or a run of plain steps make the next step's time unrepresentative).
    synced: bool,
    rng: Rng,
    prefetched: Option<Prefetch>,
}

/// Drafts an overlapping drafter proposed during the previous verify, assuming it accepted every
/// draft and emitted the bonus token they start after.
struct Prefetch {
    /// The sequence length they follow.
    len: usize,
    drafts: Vec<u32>,
    /// Their draft distributions (sampled picks only).
    qs: Vec<Vec<(u32, f32)>>,
    /// The drafter state that proposed them.
    state: DraftState,
}

impl SpecSeq {
    /// Turns on the target's taps for a sequence that has not run yet.
    pub(super) fn start(
        m: &ExecutableModel,
        drafter: Arc<dyn Drafter>,
        kv: &mut Option<SeqState>,
        seed: u64,
    ) -> Result<Self, String> {
        m.set_taps(kv, &drafter.tap_layers())?;
        Ok(Self {
            draft: drafter.new_state(),
            drafter,
            pending: Vec::new(),
            synced: false,
            rng: Rng::new(seed),
            prefetched: None,
        })
    }

    pub(super) fn max_k(&self) -> usize {
        self.drafter.max_k()
    }
}

/// What one decode step produced.
pub(super) struct Step {
    /// Output tokens, in order (accepted drafts, then the target's token).
    pub tokens: Vec<u32>,
    pub drafted: usize,
    pub accepted: usize,
    /// The step's time is representative of its `k` (see [`KTuner::record`]): the drafter was in
    /// sync, or the step did not draft.
    pub synced: bool,
    pub logprobs: Vec<super::TokenLogprobInfo>,
}

/// The target-side hooks, per backend (see the matching [`NativeLm`] methods).
///
/// [`NativeLm`]: crate::model::native::NativeLm
impl ExecutableModel {
    /// Logits after every token of `tokens` run at `start..`, `[n, vocab]`.
    fn forward_all(
        &self,
        tokens: &[u32],
        start: usize,
        kv: &mut Option<SeqState>,
    ) -> Result<Vec<f32>, String> {
        self.0
            .forward_all(tokens, start, kv.as_mut().ok_or(NO_STATE)?)
    }

    /// Rewinds the sequence to `len` positions.
    fn truncate(&self, kv: &mut Option<SeqState>, len: usize) -> Result<(), String> {
        self.0.truncate(kv.as_mut().ok_or(NO_STATE)?, len)
    }

    fn set_taps(&self, kv: &mut Option<SeqState>, layers: &[usize]) -> Result<(), String> {
        self.0.set_taps(kv.as_mut().ok_or(NO_STATE)?, layers)
    }

    fn take_taps(&self, kv: &mut Option<SeqState>) -> Vec<f32> {
        kv.as_mut().map_or_else(Vec::new, |st| self.0.take_taps(st))
    }
}

/// One decode step of a sequence whose model has run all of `all` but the last token: drafts up
/// to `k` tokens (none when `k == 0`), verifies them with one target forward, and returns the
/// tokens to emit. `vocab` is the tokenizer's vocabulary size (0 = all logits).
#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    m: &ExecutableModel,
    spec: &mut SpecSeq,
    all: &[u32],
    kv: &mut Option<SeqState>,
    temp: f32,
    top_p: f32,
    vocab: usize,
    k: usize,
    top_logprobs: Option<usize>,
) -> Result<Step, String> {
    let last = *all.last().ok_or("empty sequence")?;
    let start = all.len() - 1;
    spec.pending.extend(m.take_taps(kv));
    let valid = |n: usize| if vocab > 0 { n.min(vocab) } else { n };
    let greedy = temp <= 1e-4;
    let mut qs = Vec::new();
    let prefetched = spec.prefetched.take().filter(|p| p.len == all.len());
    let drafts = if let Some(p) = prefetched {
        spec.draft = p.state;
        let n = p.drafts.len().min(k);
        qs = p.qs;
        qs.truncate(n);
        p.drafts[..n].to_vec()
    } else if k == 0 {
        let _ = spec
            .drafter
            .propose(&mut spec.draft, all, &spec.pending, 0, &mut |_| 0);
        spec.pending.clear();
        Vec::new()
    } else {
        let (rng, picked) = (&mut spec.rng, &mut qs);
        let drafts = spec.drafter.propose(
            &mut spec.draft,
            all,
            &spec.pending,
            k,
            &mut |l| pick(l, temp, top_p, vocab, rng, picked),
        )?;
        spec.pending.clear();
        if drafts.len() > k || (!greedy && qs.len() != drafts.len()) {
            return Err(format!(
                "drafter returned {} tokens for k = {k} after {} picks",
                drafts.len(),
                qs.len()
            ));
        }
        drafts
    };
    if drafts.is_empty() {
        // Plain decode (no checkpoints); the drafter catches up on these taps next time.
        let (t, lp) = m.step_token_with_kv(all, last, all.len(), temp, top_p, top_logprobs, kv)?;
        spec.pending.extend(m.take_taps(kv));
        spec.synced = false;
        return Ok(Step {
            tokens: vec![t],
            drafted: 0,
            accepted: 0,
            synced: k == 0,
            logprobs: lp.into_iter().collect(),
        });
    }
    let synced = std::mem::replace(&mut spec.synced, true);
    let mut input = Vec::with_capacity(drafts.len() + 1);
    input.push(last);
    input.extend_from_slice(&drafts);
    let (logits, ahead) = if spec.drafter.overlaps() {
        verify_overlapped(m, spec, all, &drafts, &input, kv, (temp, top_p, vocab))?
    } else {
        (m.forward_all(&input, start, kv)?, None)
    };
    if logits.is_empty() || logits.len() % input.len() != 0 {
        return Err(format!(
            "verify returned {} logits for {} positions",
            logits.len(),
            input.len()
        ));
    }
    let width = logits.len() / input.len();
    let mut out = Vec::with_capacity(input.len());
    let mut logprobs = Vec::new();
    for (j, row) in logits.chunks_exact(width).enumerate() {
        let l = &row[..valid(width)];
        if greedy {
            let t = m.sample(l, recent(all), temp, top_p);
            if let Some(k_lp) = top_logprobs {
                logprobs.push(super::logprob_info(l, t, k_lp));
            }
            out.push(t);
            if drafts.get(j) != Some(&t) {
                break;
            }
            continue;
        }
        let p = dist(l, temp, top_p);
        let Some(&x) = drafts.get(j) else {
            let t = draw(&p, spec.rng.uniform());
            if let Some(k_lp) = top_logprobs {
                logprobs.push(super::logprob_info(l, t, k_lp));
            }
            out.push(t);
            break;
        };
        let (t, ok) = verify_draft(&p, &qs[j], x, &mut spec.rng);
        if let Some(k_lp) = top_logprobs {
            logprobs.push(super::logprob_info(l, t, k_lp));
        }
        out.push(t);
        if !ok {
            break;
        }
    }
    // The drafts proposed during the verify stand when it accepted every draft and its bonus
    // token is the one they assumed.
    let kept = ahead.filter(|(d, ..)| {
        out.len() == drafts.len() + 1 && d.len() > 1 && d.first() == out.last()
    });
    spec.prefetched = kept.map(|(d, state, qs)| Prefetch {
        len: all.len() + out.len(),
        drafts: d[1..].to_vec(),
        qs: qs.into_iter().skip(1).collect(),
        state,
    });
    let st = match &mut spec.prefetched {
        Some(p) => &mut p.state,
        None => &mut spec.draft,
    };
    spec.drafter
        .observe(st, &out, &logits[..out.len() * width], width);
    // The target keeps `last` and the accepted drafts; the final token has not run yet.
    m.truncate(kv, start + out.len())?;
    let taps = m.take_taps(kv);
    let w = taps.len() / input.len();
    spec.pending.extend_from_slice(&taps[..out.len() * w]);
    Ok(Step {
        accepted: out.len() - 1,
        drafted: drafts.len(),
        tokens: out,
        synced,
        logprobs,
    })
}

/// Next-step drafts proposed during a verify, the state that proposed them and their picks.
type Ahead = (Vec<u32>, DraftState, Vec<Vec<(u32, f32)>>);

/// The verify forward of `input`, while a scoped thread brings the drafter up to `all` (a
/// prefetched step proposed nothing yet) and proposes on a fork of its state the bonus token
/// and the next step's `drafts.len()` drafts, assuming every draft is accepted.
fn verify_overlapped(
    m: &ExecutableModel,
    spec: &mut SpecSeq,
    all: &[u32],
    drafts: &[u32],
    input: &[u32],
    kv: &mut Option<SeqState>,
    (temp, top_p, vocab): (f32, f32, usize),
) -> Result<(Vec<f32>, Option<Ahead>), String> {
    let seed = spec.rng.next_u64();
    let mut st = std::mem::replace(&mut spec.draft, Box::new(()));
    let drafter = &*spec.drafter;
    let toks = [all, drafts].concat();
    let (logits, joined) = std::thread::scope(|sc| {
        let h = sc.spawn(move || {
            let _ = drafter.propose(&mut st, all, &[], 0, &mut |_| 0);
            let ahead = drafter.fork_state(&st).and_then(|mut fork| {
                let (mut rng, mut qs) = (Rng::new(seed), Vec::new());
                let d = drafter
                    .propose(&mut fork, &toks, &[], drafts.len() + 1, &mut |l| {
                        pick(l, temp, top_p, vocab, &mut rng, &mut qs)
                    })
                    .ok()?;
                Some((d, fork, qs))
            });
            (st, ahead)
        });
        (m.forward_all(input, all.len() - 1, kv), h.join())
    });
    let (st, ahead) = joined.map_err(|_| "the overlapped drafter panicked".to_string())?;
    spec.draft = st;
    Ok((logits?, ahead))
}

/// Chooses a draft token from `l` the way the scheduler samples (`vocab`: the tokenizer's size, 0 =
/// all ids); a sampled pick records its distribution `q` in `qs` for rejection sampling.
fn pick(
    l: DraftLogits<'_>,
    temp: f32,
    top_p: f32,
    vocab: usize,
    rng: &mut Rng,
    qs: &mut Vec<Vec<(u32, f32)>>,
) -> u32 {
    let greedy = temp <= 1e-4;
    let q = match l {
        DraftLogits::Dense(l) => {
            let l = &l[..if vocab > 0 { l.len().min(vocab) } else { l.len() }];
            if greedy {
                return sample_logits(l, &[], temp, TOP_K, top_p, 1.0);
            }
            dist(l, temp, top_p)
        }
        DraftLogits::Sparse { ids, logits } => {
            let (ids, logits): (Vec<u32>, Vec<f32>) = ids
                .iter()
                .zip(logits)
                .filter(|e| vocab == 0 || (*e.0 as usize) < vocab)
                .map(|(&t, &l)| (t, l))
                .unzip();
            match (ids.is_empty(), greedy) {
                (true, true) => return 0,
                (true, false) => vec![(0, 1.0)],
                (false, true) => {
                    return ids[sample_logits(&logits, &[], temp, TOP_K, top_p, 1.0) as usize]
                }
                (false, false) => dist(&logits, temp, top_p)
                    .into_iter()
                    .map(|(i, p)| (ids[i as usize], p))
                    .collect(),
            }
        }
    };
    let t = draw(&q, rng.uniform());
    qs.push(q);
    t
}

/// Rejection sampling of draft `x`, drawn from `q`, against the target's `p`: `x` with
/// probability `min(1, p(x) / q(x))`, else a draw from `max(0, p - q)`; whether `x` was kept.
fn verify_draft(p: &[(u32, f32)], q: &[(u32, f32)], x: u32, rng: &mut Rng) -> (u32, bool) {
    if rng.uniform() * prob(q, x) as f64 <= prob(p, x) as f64 {
        return (x, true);
    }
    (draw(&residual(p, q), rng.uniform()), false)
}

/// The scheduler's sampling distribution over `logits` (sparse, normalized).
fn dist(logits: &[f32], temp: f32, top_p: f32) -> Vec<(u32, f32)> {
    let (cands, probs, n) = nucleus(logits, &[], temp, TOP_K, top_p, 1.0);
    let sum: f32 = probs[..n].iter().sum();
    cands[..n]
        .iter()
        .zip(&probs[..n])
        .map(|(&(t, _), &p)| (t as u32, p / sum))
        .collect()
}

fn prob(p: &[(u32, f32)], x: u32) -> f32 {
    p.iter().find(|e| e.0 == x).map_or(0.0, |e| e.1)
}

/// `max(0, p - q)` (unnormalized; [`draw`] scales by its total).
fn residual(p: &[(u32, f32)], q: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let r: Vec<(u32, f32)> = p
        .iter()
        .map(|&(t, pt)| (t, (pt - prob(q, t)).max(0.0)))
        .collect();
    // p == q up to rounding: a rejection is then (almost) impossible; p itself is the residual.
    if r.iter().map(|e| e.1).sum::<f32>() <= 0.0 {
        return p.to_vec();
    }
    r
}

/// Inverse-CDF draw from `p` (weights need not sum to 1) with `u` uniform in (0, 1].
fn draw(p: &[(u32, f32)], u: f64) -> u32 {
    let total: f64 = p.iter().map(|e| e.1 as f64).sum();
    let mut acc = 0.0;
    for &(t, w) in p {
        acc += w as f64;
        if acc >= u * total {
            return t;
        }
    }
    p.last().map_or(0, |e| e.0)
}

/// Picks `k` from measured throughput (output tokens per second of decode time; `k = 0` is plain
/// decoding, which skips the drafter and the checkpointing verify). It measures every `k` until
/// the sample is [`Acc::decisive`], runs the fastest, and then sticks with it: every so often it
/// probes another `k` and switches only if the probe, measured decisively too, beat the current
/// `k`'s recent rate (a probe that is losing after [`PLAIN_STEPS`] steps just ends). Rates are
/// total tokens over total time, never a mean of per-step rates: a drafting step's tokens swing
/// with how many drafts were accepted, and a few lucky steps must not flip the choice. Comparing
/// two rates measured moments apart keeps host-load drift from flipping it either. Probes rotate
/// over the neighbours of the current `k` and plain decoding (or, from plain decoding, the `k`
/// that rated best so far), so a stale rate never keeps a losing `k` running. The probe interval
/// doubles, up to [`MAX_EXPLORE`] steps, while probes lose and drops to [`MIN_EXPLORE`] when one
/// wins, so a losing drafter settles at a small fraction of plain decoding's cost.
#[derive(Default)]
pub struct KTuner {
    /// Per `k`, EWMA (tokens, seconds) per step: current for the `k` being run, as of its last
    /// probe for the others.
    avg: Vec<(f64, f64)>,
    /// Per `k`, EWMA (accepted, drafted) per step for acceptance-rate gating.
    avg_accept: Vec<(f64, f64)>,
    /// Per `k`, the first measurement, until every `k` has a decisive one.
    sweep: Vec<Acc>,
    /// The `k` being run, once every `k` was measured.
    cur: Option<usize>,
    /// Steps since the last probe, the steps between probes, and the probe rotation.
    since: usize,
    every: usize,
    turn: usize,
    probe: Option<Probe>,
}

/// Steps, tokens and seconds summed over a measurement.
#[derive(Clone, Copy, Default)]
struct Acc {
    steps: usize,
    tokens: usize,
    accepted: usize,
    drafted: usize,
    secs: f64,
}

impl Acc {
    /// Enough to judge `k`: plain steps barely vary, drafting steps need a run of them.
    fn decisive(&self, k: usize) -> bool {
        if k == 0 {
            self.steps >= PLAIN_STEPS
        } else {
            self.steps >= DRAFT_STEPS && self.tokens >= DRAFT_TOKENS
        }
    }

    fn per_step(&self) -> (f64, f64) {
        let n = self.steps.max(1) as f64;
        (self.tokens as f64 / n, self.secs / n)
    }

    fn accept_per_step(&self) -> (f64, f64) {
        let n = self.steps.max(1) as f64;
        (self.accepted as f64 / n, self.drafted as f64 / n)
    }
}

struct Probe {
    k: usize,
    /// Calls left to hand out `k`, and what was recorded so far.
    calls: usize,
    acc: Acc,
}
const MIN_EXPLORE: usize = 16;
const MAX_EXPLORE: usize = 512;
/// Recorded steps that make a measurement of plain decoding, and of a drafting `k`
/// (which also needs [`DRAFT_TOKENS`] tokens).
const PLAIN_STEPS: usize = 3;
const DRAFT_STEPS: usize = 8;
const DRAFT_TOKENS: usize = 16;
/// Calls a probe stays on: its samples plus a catch-up step (unrecorded) for each of a few
/// concurrent sequences.
const PROBE_CALLS: usize = 48;
/// How much faster a probe of a larger `k` must run to replace the current one.
const DRAFT_MORE_MARGIN: f64 = 0.05;

impl KTuner {
    /// The `k` for the next step of a drafter that drafts at most `max_k`.
    pub fn choose(&mut self, max_k: usize) -> usize {
        self.choose_with_knobs(max_k, &DraftKnobs::default())
    }

    /// Multi-objective `k` selection respecting live [`DraftKnobs`].
    pub fn choose_with_knobs(&mut self, max_k: usize, knobs: &DraftKnobs) -> usize {
        if knobs.mode == DraftMode::Off {
            return 0;
        }
        let max_k = max_k.min(knobs.max_k);
        if max_k == 0 {
            return 0;
        }
        if self.avg.len() != max_k + 1 {
            *self = Self {
                avg: vec![(0.0, 0.0); max_k + 1],
                avg_accept: vec![(0.0, 0.0); max_k + 1],
                sweep: vec![Acc::default(); max_k + 1],
                every: MIN_EXPLORE,
                ..Self::default()
            };
        }
        let Some(cur) = self.cur else {
            // Measure every k, plain decoding last.
            return (1..=max_k)
                .chain([0])
                .find(|&k| !self.sweep[k].decisive(k))
                .unwrap_or(0);
        };
        let cur = cur.min(max_k);
        if let Some(p) = &mut self.probe {
            if p.calls > 0 && p.k <= max_k {
                p.calls -= 1;
                return p.k;
            }
            self.probe = None;
        }
        self.since += 1;
        if self.since < self.every {
            return cur;
        }
        self.since = 0;
        self.turn += 1;
        let best_drafting =
            (1..=max_k).max_by(|&a, &b| self.score(a, knobs).total_cmp(&self.score(b, knobs)));
        let options = if cur == 0 {
            [best_drafting, None, None]
        } else {
            [
                (cur < max_k).then_some(cur + 1),
                Some(cur - 1),
                (cur > 1).then_some(0),
            ]
        };
        let Some(k) = (0..3).find_map(|i| options[(self.turn + i) % 3]) else {
            return cur;
        };
        self.probe = Some(Probe {
            k,
            calls: PROBE_CALLS - 1,
            acc: Acc::default(),
        });
        k
    }

    /// Records a step of `k` that emitted `tokens` in `took` with default knobs.
    pub fn record(&mut self, k: usize, tokens: usize, took: Duration) -> Option<String> {
        let accepted = tokens.saturating_sub(1);
        self.record_with_knobs(
            k,
            tokens,
            k,
            accepted,
            took,
            &DraftKnobs {
                cost_weight: 0.0,
                ..DraftKnobs::default()
            },
        )
    }

    /// Records a step of `k` that drafted `drafted`, accepted `accepted`, and emitted `tokens` in
    /// `took`, scoring candidates under `knobs`.
    pub fn record_with_knobs(
        &mut self,
        k: usize,
        tokens: usize,
        drafted: usize,
        accepted: usize,
        took: Duration,
        knobs: &DraftKnobs,
    ) -> Option<String> {
        let secs = took.as_secs_f64().max(1e-9);
        let before = self.cur;
        match (self.cur, self.probe.as_mut()) {
            (None, _) => {
                let a = self.sweep.get_mut(k)?;
                a.steps += 1;
                a.tokens += tokens;
                a.accepted += accepted;
                a.drafted += drafted;
                a.secs += secs;
                self.avg[k] = a.per_step();
                self.avg_accept[k] = a.accept_per_step();
                if self.sweep.iter().enumerate().all(|(k, a)| a.decisive(k)) {
                    self.cur = self.fastest_for(knobs);
                }
            }
            (Some(cur), Some(p)) if p.k == k && k != cur => {
                p.acc.steps += 1;
                p.acc.tokens += tokens;
                p.acc.accepted += accepted;
                p.acc.drafted += drafted;
                p.acc.secs += secs;
                let acc = p.acc;
                // Drafting more also spends more compute: it has to win clearly, not by noise.
                let margin = if k > cur {
                    1.0 + DRAFT_MORE_MARGIN
                } else {
                    1.0
                };
                let (t, s) = acc.per_step();
                let (ac, dr) = acc.accept_per_step();
                let probe_score = Self::compute_score(k, t, s, ac, dr, knobs);
                let won = probe_score > self.score(cur, knobs) * margin;
                // A loss ends the probe early (that only backs off); a win needs a decisive run.
                if acc.steps >= PLAIN_STEPS && (!won || acc.decisive(k)) {
                    self.avg[k] = (t, s);
                    self.avg_accept[k] = (ac, dr);
                    self.probe = None;
                    self.every = if won {
                        self.cur = Some(k);
                        MIN_EXPLORE
                    } else {
                        (self.every * 2).min(MAX_EXPLORE)
                    };
                }
            }
            _ => {
                let (t, s) = self.avg.get_mut(k)?;
                *t = 0.8 * *t + 0.2 * tokens as f64;
                *s = 0.8 * *s + 0.2 * secs;
                if let Some((ac, dr)) = self.avg_accept.get_mut(k) {
                    *ac = 0.8 * *ac + 0.2 * accepted as f64;
                    *dr = 0.8 * *dr + 0.2 * drafted as f64;
                }
            }
        }
        let rates = (0..self.avg.len())
            .map(|k| format!("k {k}: {:.2}", self.rate(k)))
            .collect::<Vec<_>>()
            .join(", ");
        match (before, self.cur) {
            (b, Some(0)) if b != Some(0) => Some(format!(
                "speculative decoding off: plain decoding is faster (tok/s {rates})"
            )),
            (Some(0), Some(a)) if a > 0 => Some(format!(
                "speculative decoding back on at k {a} (tok/s {rates})"
            )),
            _ => None,
        }
    }

    /// Tokens per second of `k`: EWMA tokens per step over EWMA seconds per step.
    pub fn rate(&self, k: usize) -> f64 {
        let Some(&(t, s)) = self.avg.get(k) else {
            return 0.0;
        };
        if s > 0.0 {
            t / s
        } else {
            0.0
        }
    }

    /// Measured acceptance rate `accepted / drafted` at `k` (`1.0` for `k == 0`).
    pub fn acceptance_rate(&self, k: usize) -> f64 {
        if k == 0 {
            return 1.0;
        }
        let Some(&(ac, dr)) = self.avg_accept.get(k) else {
            return 0.0;
        };
        if dr > 0.0 {
            (ac / dr).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn compute_score(k: usize, t: f64, s: f64, ac: f64, dr: f64, knobs: &DraftKnobs) -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        let raw = t / s;
        if k == 0 || knobs.cost_weight <= 1e-5 {
            return raw;
        }
        let cw = (knobs.cost_weight as f64).clamp(0.0, 1.0);
        let acc = if dr > 0.0 {
            (ac / dr).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let target = (knobs.target_acceptance as f64).clamp(0.10, 0.98);
        let acc_factor = if acc >= target {
            1.0
        } else {
            (acc / target).clamp(0.15, 1.0).powf(cw * 1.5)
        };
        // Penalize verify waste when extra drafted tokens are rejected.
        let verify_eff = (t / (1.0 + k as f64 * 0.15 * cw)).max(0.1);
        raw * acc_factor * (1.0 - 0.15 * cw + 0.15 * cw * verify_eff.min(2.0))
    }

    /// Multi-objective score of `k` under `knobs`.
    pub fn score(&self, k: usize, knobs: &DraftKnobs) -> f64 {
        let Some(&(t, s)) = self.avg.get(k) else {
            return 0.0;
        };
        let (ac, dr) = self.avg_accept.get(k).copied().unwrap_or((0.0, 0.0));
        Self::compute_score(k, t, s, ac, dr, knobs)
    }

    fn fastest_for(&self, knobs: &DraftKnobs) -> Option<usize> {
        (0..self.avg.len()).max_by(|&a, &b| self.score(a, knobs).total_cmp(&self.score(b, knobs)))
    }

    /// The `k` being run; `None` until every `k` was measured.
    pub fn best(&self) -> Option<usize> {
        self.cur
    }

    /// Summary `(k, tok_per_sec, accept_rate)` for each measured `k`.
    pub fn rates_summary(&self) -> Vec<(usize, f64, f64)> {
        (0..self.avg.len())
            .map(|k| (k, self.rate(k), self.acceptance_rate(k)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An overlapping drafter that drafts `script` (wrong at the indices in `bad`) and counts the
    /// proposals that draft.
    struct Oracle {
        script: Vec<u32>,
        bad: Vec<usize>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Drafter for Oracle {
        fn label(&self) -> String {
            "oracle".into()
        }
        fn tap_layers(&self) -> Vec<usize> {
            Vec::new()
        }
        fn new_state(&self) -> DraftState {
            Box::new(())
        }
        fn overlaps(&self) -> bool {
            true
        }
        fn fork_state(&self, _: &DraftState) -> Option<DraftState> {
            Some(Box::new(()))
        }
        fn propose(
            &self,
            _: &mut DraftState,
            tokens: &[u32],
            _: &[f32],
            k: usize,
            pick: &mut dyn FnMut(DraftLogits<'_>) -> u32,
        ) -> Result<Vec<u32>, String> {
            if k > 0 {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let end = (tokens.len() + k).min(self.script.len());
            Ok((tokens.len()..end)
                .map(|i| {
                    let t = self.script[i] ^ u32::from(self.bad.contains(&i));
                    pick(DraftLogits::Sparse {
                        ids: &[t],
                        logits: &[0.0],
                    })
                })
                .collect())
        }
    }

    /// Drafts proposed during a verify replace the next propose only after that verify accepted
    /// every draft and emitted the bonus token they assumed; greedy output is unchanged.
    #[test]
    fn prefetched_drafts_used_only_after_full_accept() {
        use crate::model::native::PrepareCtx;
        let dir = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({"max_position_embeddings": 512});
        let m = crate::arch::testing::synthetic_lm(dir.path(), cfg, crate::hardware::Device::Cpu);
        let prompt = vec![5u32, 9, 13, 2];
        let fresh = || {
            let (ids, mut st) = m.prepare(prompt.clone(), &[], &PrepareCtx::default()).unwrap();
            m.prefill_logits(&ids[..ids.len() - 1], 0, &mut st).unwrap();
            st
        };
        let (mut script, mut st) = (prompt.clone(), fresh());
        while script.len() < 20 {
            let last = *script.last().unwrap();
            let (t, _) = m
                .step_token_with_kv(&script, last, script.len(), 0.0, 1.0, None, &mut st)
                .unwrap();
            script.push(t);
        }
        let oracle = Arc::new(Oracle {
            script: script.clone(),
            bad: vec![8, 11],
            calls: Default::default(),
        });
        let mut kv = fresh();
        let mut spec = SpecSeq::start(&m, oracle.clone(), &mut kv, 7).unwrap();
        let (mut all, mut calls) = (prompt.clone(), Vec::new());
        for _ in 0..4 {
            let s = decode(&m, &mut spec, &all, &mut kv, 0.0, 1.0, 256, 2, None).unwrap();
            all.extend(s.tokens);
            calls.push(oracle.calls.load(std::sync::atomic::Ordering::Relaxed));
        }
        assert_eq!(all, script[..all.len()], "greedy output changed");
        // 1: propose + run ahead (accepted, bonus as assumed). 2: the prefetch, no propose; its
        // wrong draft at 8 drops the next one. 3: propose + run ahead; the bonus at 11 is not the
        // one assumed. 4: propose again.
        assert_eq!(calls, [2, 3, 5, 7]);
    }

    /// A sparse draft (candidate ids only, over a support that misses target tokens and adds one
    /// the target excludes) verified by rejection sampling emits exactly the target's
    /// distribution: χ² over 20 000 seeded draws.
    #[test]
    fn sparse_draft_rejection_sampling_matches_target() {
        let target = [2.0f32, 0.5, 1.5, -1.0, 0.0, 1.0, -30.0, 0.3];
        let (temp, top_p) = (0.8, 0.95);
        let p = dist(&target, temp, top_p);
        let (ids, logits) = ([1u32, 3, 5, 6], [1.0f32, 0.2, 2.0, 0.5]);
        let mut rng = Rng::new(11);
        let n = 20_000;
        let mut counts = [0usize; 8];
        for _ in 0..n {
            let mut qs = Vec::new();
            let sparse = DraftLogits::Sparse {
                ids: &ids,
                logits: &logits,
            };
            let x = pick(sparse, temp, top_p, target.len(), &mut rng, &mut qs);
            counts[verify_draft(&p, &qs[0], x, &mut rng).0 as usize] += 1;
        }
        let mut chi2 = 0.0;
        for (t, &c) in counts.iter().enumerate() {
            let e = prob(&p, t as u32) as f64 * n as f64;
            if e == 0.0 {
                assert_eq!(c, 0, "token {t} is outside the target's support");
                continue;
            }
            chi2 += (c as f64 - e).powi(2) / e;
        }
        // 99.9th percentile of χ² with at most 7 degrees of freedom.
        assert!(chi2 < 24.32, "χ² = {chi2}: {counts:?} vs {p:?}");
    }

    /// Rejection sampling over a mismatched draft keeps the target distribution.
    #[test]
    fn rejection_sampling_preserves_target_distribution() {
        let p = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let q = vec![(1, 0.6), (2, 0.1), (3, 0.3)];
        let mut rng = Rng::new(7);
        let n = 200_000;
        let mut counts = [0usize; 4];
        for _ in 0..n {
            let x = draw(&q, rng.uniform());
            let t = if rng.uniform() * prob(&q, x) as f64 <= prob(&p, x) as f64 {
                x
            } else {
                draw(&residual(&p, &q), rng.uniform())
            };
            counts[t as usize] += 1;
        }
        for &(t, pt) in &p {
            let f = counts[t as usize] as f32 / n as f32;
            assert!((f - pt).abs() < 0.005, "token {t}: {f} vs {pt}");
        }
        assert_eq!(
            counts[3], 0,
            "a token outside the target's support was emitted"
        );
    }

    /// A drafter that loses settles on plain decoding (logged once) and is then probed rarely
    /// enough to stay within 1% of plain, even when plain decoding was first measured during a
    /// load spike or the drafter's accepted counts swing step to step; one that wins keeps
    /// drafting.
    #[test]
    fn k_tuner_turns_off_a_losing_drafter_and_keeps_a_winning_one() {
        // (tokens, ms) of step `i` at each k; one plain step = 1 token in 100 ms.
        let run = |step: fn(usize, usize) -> (usize, f64)| {
            let mut t = KTuner::default();
            let (mut time, mut tokens, mut logs) = (0.0, 0, Vec::new());
            for i in 0..3000 {
                let k = t.choose(4);
                let (n, ms) = step(k, i);
                if i >= 1000 {
                    time += ms;
                    tokens += n;
                }
                logs.extend(t.record(k, n, Duration::from_secs_f64(ms / 1000.0)));
            }
            (tokens as f64 / time * 1000.0, t.best(), logs)
        };
        // Losing: each draft token costs 60 ms and few are accepted.
        let (rate, best, logs) = run(|k, _| (1 + k / 3, 100.0 + 60.0 * k as f64));
        assert_eq!(best, Some(0));
        assert!(rate > 10.0 * 0.99, "settled at {rate} tok/s vs plain 10");
        assert_eq!(logs.len(), 1, "{logs:?}");
        assert!(logs[0].starts_with("speculative decoding off"), "{logs:?}");
        // Losing by a little (best drafting k is 2.4% slower than plain), while plain decoding
        // ran 3x slower over its first measurement (a load spike): drafting is picked first and
        // only a probe of plain decoding brings it back.
        let (rate, best, _) = run(|k, i| {
            let ms = 100.0 + 35.0 * k as f64;
            (1 + k / 3, if k == 0 && i < 100 { 3.0 * ms } else { ms })
        });
        assert_eq!(best, Some(0));
        assert!(rate > 10.0 * 0.99, "settled at {rate} tok/s vs plain 10");
        // Losing by 4-7% on average, with 0..=k drafts accepted per step at random: lucky runs
        // of a few steps beat plain decoding, a decisive measurement doesn't.
        let (rate, best, logs) = run(|k, i| {
            let u = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
            (1 + (u % (k as u64 + 1)) as usize, 100.0 + 56.0 * k as f64)
        });
        assert_eq!(best, Some(0));
        assert!(rate > 10.0 * 0.98, "settled at {rate} tok/s vs plain 10");
        assert!(logs.len() <= 3, "{logs:?}");
        // Winning: each draft token costs 10 ms and most are accepted (best k = 3).
        let (rate, best, _) = run(|k, _| (1 + k * 7 / 10, 100.0 + 10.0 * k as f64));
        assert_eq!(best, Some(3));
        assert!(rate > 20.0, "{rate}");
    }
}
