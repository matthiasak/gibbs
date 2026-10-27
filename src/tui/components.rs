//! Instrument components: STATUS row, sidebar panels and the compact dock. Every value shown is a
//! telemetry reading or a derivation of one; colours come from the theme only.

use super::layout::Comp;
use super::mascot::MascotState;
use super::theme::{Theme, Token};
use super::widgets::{gauge, instrument, orbit, spark, stacked_bar, text_width, trunc};
use super::{Load, TuiApp};
use crate::scheduler::SequenceState;
use crate::telemetry::{gpu_share, KernelSort, TelemetrySnapshot};
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        canvas::{Canvas, Line as CLine},
        Paragraph,
    },
    Frame,
};

const LN2: f32 = std::f32::consts::LN_2;

/// One frame's readings, computed once and shared by every component.
pub(super) struct Vitals {
    pub s: TelemetrySnapshot,
    /// Achieved / roof tok/s in percent (unclamped).
    pub pct: f32,
    pub roof_tok_s: f64,
    pub bytes_per_tok: u64,
    pub ceiling: f32,
    pub kv_pct: f32,
    /// Inter-token latencies, newest last.
    pub itl: Vec<f32>,
}

impl Vitals {
    /// (p50, p95, p99, σ) of the ITL window.
    pub fn itl_stats(&self) -> Option<(f32, f32, f32, f32)> {
        if self.itl.is_empty() {
            return None;
        }
        let mut v = self.itl.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        let q = |p: f32| v[((n - 1) as f32 * p).round() as usize];
        let mean = v.iter().sum::<f32>() / n as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        Some((q(0.5), q(0.95), q(0.99), var.sqrt()))
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum State {
    Equilibrium,
    Decode,
    Prefill,
    Loading(f32),
    Error,
}

/// Download/map/compile progress of a load: (stage index 0..4, done, total, percent).
pub(super) fn load_stage(l: &Load) -> (usize, u64, u64, f32) {
    let stage = *l.stage.lock();
    let done = l.progress.done.load(std::sync::atomic::Ordering::Relaxed);
    let total = l.progress.total.load(std::sync::atomic::Ordering::Relaxed);
    let idx = if stage == "downloading" {
        1
    } else if stage.contains("reading") || stage.contains("format") {
        2
    } else if stage.contains("placing") || stage.contains("compil") {
        3
    } else {
        0
    };
    let pct = if total > 0 {
        (done as f64 / total as f64 * 100.0).clamp(0.0, 100.0) as f32
    } else if idx >= 2 {
        100.0
    } else {
        0.0
    };
    (idx, done, total, pct)
}

/// The codec micro-benchmark as the Device panel shows it: `(state, done, total)` with state 1
/// running, 2 done, anything else (idle, failed) not available.
pub(super) fn codec_bench_text(codec: (u8, usize, usize)) -> String {
    match codec {
        (1, done, total) => format!("codec-bench {done}/{total}"),
        (2, _, _) => "codec-bench ✓".to_string(),
        _ => "codec-bench —".to_string(),
    }
}

/// One row of the Kernels table: the kind, then right-aligned numeric columns.
fn kernel_line(
    kind: &str,
    steps: String,
    us: String,
    share: String,
    gbps: String,
    pct: String,
) -> String {
    format!("{kind:<11}{steps:>5}{us:>6}{share:>6}{gbps:>6}{pct:>4}")
}

/// Parts joined by `sep`, left to right while they fit; the rest are dropped.
fn fit(parts: Vec<Vec<Span<'static>>>, width: usize, sep: &str, sep_style: Style) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut used = 0;
    for p in parts {
        let w: usize = p.iter().map(|s| s.width()).sum();
        let sw = if spans.is_empty() { 0 } else { text_width(sep) };
        if used + sw + w > width {
            break;
        }
        if sw > 0 {
            spans.push(Span::styled(sep.to_string(), sep_style));
        }
        used += sw + w;
        spans.extend(p);
    }
    Line::from(spans)
}

fn pill(th: &Theme, tok: Token, text: &str) -> Span<'static> {
    let style = if th.mono {
        Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD)
    } else {
        th.fg(Token::Void)
            .patch(th.bg(tok))
            .add_modifier(Modifier::BOLD)
    };
    Span::styled(text.to_string(), style)
}

fn seg(text: impl Into<String>, style: Style) -> Vec<Span<'static>> {
    vec![Span::styled(text.into(), style)]
}

impl TuiApp {
    pub(super) fn vitals(&self) -> Vitals {
        let s = self.telemetry.get_summary();
        let bytes_per_tok = self.telemetry.bytes_per_token();
        let ceiling = s.memory_bandwidth_gbps;
        let roof_tok_s = if bytes_per_tok > 0 && ceiling > 0.0 {
            ceiling as f64 / (bytes_per_tok as f64 / 1e9)
        } else {
            0.0
        };
        let pct = if roof_tok_s > 0.0 {
            (s.current_tps as f64 / roof_tok_s * 100.0) as f32
        } else {
            s.roofline_pct
        };
        Vitals {
            pct,
            roof_tok_s,
            bytes_per_tok,
            ceiling,
            kv_pct: {
                let (used, cap) = self.kv_pages();
                used as f32 * 100.0 / cap.max(1) as f32
            },
            itl: self.telemetry.recent_itl_ms(512),
            s,
        }
    }

    pub(super) fn state(&self) -> State {
        if let Some(l) = &self.load {
            return State::Loading(load_stage(l).3);
        }
        if self.mascot.state == MascotState::Error {
            return State::Error;
        }
        if self.running > 0 {
            let prefilling = self.scheduler.as_ref().is_some_and(|s| {
                s.view()
                    .seqs
                    .iter()
                    .any(|q| q.state == SequenceState::Prefilling)
            });
            return if prefilling {
                State::Prefill
            } else {
                State::Decode
            };
        }
        State::Equilibrium
    }

    fn focused(&self, c: Comp) -> bool {
        self.detail == Some(c)
    }

    fn token_str(&self, id: u32) -> String {
        let s = self
            .scheduler
            .as_ref()
            .map(|s| s.tokenizer().decode_token(id))
            .unwrap_or_else(|| format!("#{id}"));
        s.replace('\n', "⏎").replace('\t', "⇥")
    }

    // ---- STATUS ---------------------------------------------------------------------------

    pub(super) fn render_status(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let dim = th.fg(Token::Dim);
        let (state_text, state_tok) = match self.state() {
            State::Equilibrium => ("● EQUILIBRIUM".to_string(), Token::Jade),
            State::Decode => ("⚡ DECODE".to_string(), Token::Ember),
            State::Prefill => ("◇ PREFILL".to_string(), Token::Gold),
            State::Loading(p) => (
                format!("{} LOADING {p:.0}%", orbit(self.mascot.frame_tick / 2)),
                Token::Cyanotype,
            ),
            State::Error => ("▲ SPINODAL".to_string(), Token::Crimson),
        };
        let state_span = Span::styled(
            format!(" {state_text}"),
            th.fg(state_tok).add_modifier(Modifier::BOLD),
        );
        let w = area.width as usize;
        let avail = w.saturating_sub(state_span.width());

        let model = self.model_id();
        let model_max = (w * 40 / 100).max(8);
        let model_seg = if model.is_empty() {
            seg("no model", dim)
        } else {
            seg(
                trunc(&model, model_max),
                th.fg(Token::Parchment).add_modifier(Modifier::BOLD),
            )
        };
        let classifier = self.classifier.read().as_ref().map(|c| c.summary());
        let backend = self.scheduler.as_ref().map(|s| s.backend_label());
        let device = match (&classifier, backend) {
            (Some(c), _) => Some(format!("text classifier: {c}")),
            (None, Some(b)) => Some(
                match b.strip_prefix("Metal (").and_then(|r| r.strip_suffix(')')) {
                    Some(name) => format!("{name} · Metal"),
                    None => b,
                },
            ),
            (None, None) => None,
        };
        let loading = self.load.is_some();
        let sched = self.scheduler.as_ref().map(|s| s.view());
        // (name, spans, joined to the previous segment by a space instead of ` · `)
        let mut segs: Vec<(&str, Vec<Span<'static>>, bool)> =
            vec![("pill", vec![pill(th, Token::Brass, " GIBBS ")], false)];
        if w < 100 {
            segs.push(("badge", self.mascot.render_mini_badge(), true));
        }
        segs.push(("model", model_seg, true));
        if !loading {
            if let Some(d) = device {
                segs.push(("device", seg(d, th.fg(Token::Parchment)), false));
            }
            if classifier.is_none() {
                segs.push((
                    "tps",
                    seg(
                        format!("{:.1} tok/s", v.s.current_tps),
                        th.fg(Token::Brass).add_modifier(Modifier::BOLD),
                    ),
                    false,
                ));
                let tier = Theme::tier(v.pct);
                segs.push((
                    "roof",
                    seg(
                        format!("ROOF {:.0}%", v.pct),
                        th.fg(tier).add_modifier(Modifier::BOLD),
                    ),
                    false,
                ));
                segs.push(("gauge", seg(gauge(v.pct / 100.0, 8), th.fg(tier)), true));
                segs.push((
                    "kv",
                    seg(format!("KV {:.0}%", v.kv_pct), th.fg(Token::Parchment)),
                    false,
                ));
                if let Some(sv) = &sched {
                    let active = sv
                        .seqs
                        .iter()
                        .filter(|q| {
                            matches!(q.state, SequenceState::Prefilling | SequenceState::Decoding)
                        })
                        .count();
                    segs.push((
                        "seq",
                        seg(
                            format!("SEQ {active}/{}", active + sv.waiting),
                            th.fg(Token::Parchment),
                        ),
                        false,
                    ));
                }
            }
            if let Some(a) = self.http_addr {
                segs.push(("port", seg(format!(":{}", a.port()), dim), false));
            }
        }
        let build = |segs: &[(&str, Vec<Span<'static>>, bool)]| -> Vec<Span<'static>> {
            let mut out = Vec::new();
            for (i, (_, spans, tight)) in segs.iter().enumerate() {
                if i > 0 {
                    out.push(Span::styled(if *tight { " " } else { " · " }, dim));
                }
                out.extend(spans.iter().cloned());
            }
            out
        };
        let width = |s: &[Span<'static>]| s.iter().map(|x| x.width()).sum::<usize>();
        for drop in [
            "badge", "port", "seq", "kv", "gauge", "format", "device", "roof", "tps",
        ] {
            if width(&build(&segs)) <= avail {
                break;
            }
            segs.retain(|(n, _, _)| *n != drop);
        }
        let mut left = build(&segs);
        while width(&left) > avail && left.len() > 1 {
            left.pop();
        }
        let pad = w.saturating_sub(width(&left) + state_span.width());
        left.push(Span::raw(" ".repeat(pad)));
        left.push(state_span);
        f.render_widget(Paragraph::new(Line::from(left)), area);
    }

    // ---- ROOFLINE -------------------------------------------------------------------------

    pub(super) fn render_roofline(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        let tier = Theme::tier(v.pct);
        let metric = Line::from(Span::styled(
            format!("{:.1}%", v.pct),
            th.fg(tier).add_modifier(Modifier::BOLD),
        ));
        let block = instrument("Roofline", Some(metric), self.focused(Comp::Roofline), th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let w = inner.width as usize;
        let dim = th.fg(Token::Dim);
        let bw = th.fg(Token::Cyanotype);
        let mut lines: Vec<Line<'static>> = Vec::new();

        let speed = format!("{:.1}/{:.1} tok/s", v.s.current_tps, v.roof_tok_s);
        let mut row1 = vec![Span::styled(speed.clone(), th.fg(Token::Parchment))];
        if w >= 28 {
            let cells = (w - 24)
                .max(4)
                .min(w.saturating_sub(text_width(&speed) + 1));
            if cells >= 4 {
                row1.push(Span::raw(" "));
                row1.push(Span::styled(gauge(v.pct / 100.0, cells), th.fg(tier)));
            }
        }
        lines.push(Line::from(row1));

        lines.push(fit(
            vec![
                seg(
                    format!(
                        "BW {:.1}/{:.1} GB/s",
                        v.s.effective_bandwidth_gbps, v.ceiling
                    ),
                    bw,
                ),
                seg(format!("{:.3} GB/tok", v.bytes_per_tok as f64 / 1e9), bw),
            ],
            w,
            "  ",
            dim,
        ));

        // Decode streams every weight once and spends 2 flops on it: AI = 2 / (bpw / 8) F/B.
        // ponytail: estimated from the `--quant` format only; exact value needs
        // PreparedPlan::flops_per_run plumbed into telemetry (graph/scheduler are outside the TUI).
        let ai = self
            .quant_override
            .map(|f| f.bits_per_weight())
            .filter(|b| *b > 0.0)
            .map(|b| 16.0 / b as f64);
        let peak = self.gemm_peak_gflops;
        let ridge = (peak > 0.0 && v.ceiling > 0.0).then(|| peak / v.ceiling as f64);
        let mut row3 = vec![
            seg(
                ai.map_or("AI —".to_string(), |a| format!("AI {a:.1} F/B")),
                th.fg(Token::Parchment),
            ),
            seg(
                ridge.map_or("ridge —".to_string(), |r| format!("ridge {r:.1}")),
                dim,
            ),
        ];
        if let (Some(a), Some(r)) = (ai, ridge) {
            row3.push(seg(
                format!("▸ {}-BOUND", if a < r { "BANDWIDTH" } else { "COMPUTE" }),
                if a < r { bw } else { th.fg(Token::Brass) },
            ));
        }
        lines.push(fit(row3, w, "  ", dim));

        let detail = self.focused(Comp::Roofline);
        if detail {
            if let Some(p) = &self.roofline_probe_result {
                if let Some(l) = p.lines().nth(1) {
                    lines.push(Line::from(Span::styled(trunc(l, w), dim)));
                }
            }
        }
        // The last row is the profiler line when there is room for canvas plus it.
        let rest = (inner.height as usize).saturating_sub(lines.len());
        let profile_row = rest >= 3;
        let canvas_h = rest.saturating_sub(profile_row as usize);
        let top = lines.len();
        f.render_widget(
            Paragraph::new(lines),
            Rect::new(
                inner.x,
                inner.y,
                inner.width,
                top.min(inner.height as usize) as u16,
            ),
        );
        if canvas_h > 0 && peak > 0.0 && v.ceiling > 0.0 {
            let c = Rect::new(inner.x, inner.y + top as u16, inner.width, canvas_h as u16);
            self.roofline_canvas(f, c, v, th, ai, peak, tier);
        }
        if profile_row {
            let text = if !self.profile_sampling {
                "profile off · /profile on".to_string()
            } else if self.gemm_top.is_empty() {
                "profile on · no host GEMM timings".to_string()
            } else {
                format!("GEMM {}", self.gemm_top.join(" · "))
            };
            f.render_widget(
                Paragraph::new(Span::styled(trunc(&text, w), dim)),
                Rect::new(inner.x, inner.y + inner.height - 1, inner.width, 1),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn roofline_canvas(
        &self,
        f: &mut Frame,
        area: Rect,
        v: &Vitals,
        th: &Theme,
        ai: Option<f64>,
        peak: f64,
        tier: Token,
    ) {
        let bw = v.ceiling as f64;
        let (x0, x1) = (-1.0f64, 3.0f64);
        let (y0, y1) = (1.0f64, (peak * 1.5).log10());
        let ridge = peak / bw;
        // GFLOP/s = AI [F/B] × BW [GB/s]; the slope starts where it enters the plot.
        let a0 = (10f64.powf(y0) / bw).max(10f64.powf(x0));
        let bg = if th.truecolor {
            th.color(Token::Plinth)
        } else {
            ratatui::style::Color::Reset
        };
        let (c_roof, c_bw, c_pt) = (
            th.color(Token::Brass),
            th.color(Token::Cyanotype),
            th.color(tier),
        );
        let point = ai.zip(
            (v.s.effective_bandwidth_gbps > 0.0).then_some(v.s.effective_bandwidth_gbps as f64),
        );
        f.render_widget(
            Canvas::default()
                .marker(Marker::Braille)
                .background_color(bg)
                .x_bounds([x0, x1])
                .y_bounds([y0, y1])
                .paint(move |ctx| {
                    if ridge > 10f64.powf(x0) {
                        ctx.draw(&CLine {
                            x1: a0.log10(),
                            y1: (a0 * bw).log10(),
                            x2: ridge.log10().min(x1),
                            y2: peak.log10(),
                            color: c_bw,
                        });
                    }
                    ctx.draw(&CLine {
                        x1: ridge.log10().clamp(x0, x1),
                        y1: peak.log10(),
                        x2: x1,
                        y2: peak.log10(),
                        color: c_roof,
                    });
                    if let Some((a, eff)) = point {
                        let (px, py) = (a.log10().clamp(x0, x1), (a * eff).log10().clamp(y0, y1));
                        ctx.print(px, py, Span::styled("●", Style::default().fg(c_pt)));
                    }
                }),
            area,
        );
    }

    // ---- TOKENS ---------------------------------------------------------------------------

    pub(super) fn render_tokens(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        let s = &v.s;
        let block = instrument(
            "Tokens",
            Some(Line::from(format!("{:.1}/s", s.current_tps))),
            self.focused(Comp::Tokens),
            th,
        );
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let w = inner.width as usize;
        let (dim, ink) = (th.fg(Token::Dim), th.fg(Token::Parchment));
        let active: Vec<u64> = self.spark.iter().copied().filter(|&x| x > 0).collect();
        let avg = if active.is_empty() {
            0.0
        } else {
            active.iter().sum::<u64>() as f64 / active.len() as f64
        };
        let peak = self.spark.iter().copied().max().unwrap_or(0);
        let mut lines = vec![
            fit(
                vec![
                    seg(format!("NOW {:.1}", s.current_tps), th.fg(Token::Brass)),
                    seg(format!("AVG {avg:.1}"), ink),
                    seg(format!("PEAK {peak:.1} tok/s"), ink),
                ],
                w,
                "  ",
                dim,
            ),
            Line::from(Span::styled(
                spark(&self.spark.iter().map(|&x| x as f32).collect::<Vec<_>>(), w),
                th.fg(Token::Brass),
            )),
            fit(
                vec![
                    seg(
                        format!("Σ {} tok", super::render::thousands(s.total_tokens)),
                        ink,
                    ),
                    seg(format!("{} req", s.completed_requests), ink),
                    seg(
                        format!("TTFT {:.0}/{:.0} ms", s.ttft_p50_ms, s.ttft_p90_ms),
                        ink,
                    ),
                ],
                w,
                " · ",
                dim,
            ),
        ];
        if let Some(sv) = self.scheduler.as_ref().and_then(|sc| sc.view().spec) {
            let spec_text = if s.spec_forwards > 0 {
                let k_tag = sv.k.map_or_else(
                    || format!("0..{}", self.draft_knobs.max_k),
                    |k| k.to_string(),
                );
                format!(
                    "SPEC {:.0}% · {:.2} tok/fwd · {} (k={})",
                    s.spec_accept_rate * 100.0,
                    s.spec_tokens_per_forward,
                    sv.label,
                    k_tag
                )
            } else {
                format!(
                    "SPEC {} · ready (k=0..{})",
                    sv.label, self.draft_knobs.max_k
                )
            };
            lines.push(Line::from(Span::styled(
                trunc(&spec_text, w),
                th.fg(Token::Gold),
            )));
        }
        let (kv_hits, kv_misses) = self.scheduler.as_ref().map_or((0, 0), |s| s.radix.metrics());
        let turns = self.chat_history.len() / 2;
        if turns > 0 || kv_hits > 0 {
            lines.push(Line::from(Span::styled(
                trunc(
                    &format!("TURN {turns} · KV HIT {kv_hits} tok ({kv_misses} miss)"),
                    w,
                ),
                th.fg(Token::Jade),
            )));
        }
        if self.focused(Comp::Tokens) {
            let last: Vec<String> = self
                .spark
                .iter()
                .rev()
                .take(10)
                .rev()
                .map(|x| x.to_string())
                .collect();
            lines.push(Line::from(Span::styled(
                trunc(&format!("last {} s: {}", last.len(), last.join(" ")), w),
                dim,
            )));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- LATENCY --------------------------------------------------------------------------

    pub(super) fn render_latency(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        let stats = v.itl_stats();
        let metric = Line::from(stats.map_or("p99 —".to_string(), |(_, _, p99, _)| {
            format!("p99 {p99:.1} ms")
        }));
        let block = instrument("Latency", Some(metric), self.focused(Comp::Latency), th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let w = inner.width as usize;
        let (dim, ink) = (th.fg(Token::Dim), th.fg(Token::Parchment));
        let Some((p50, p95, p99, sd)) = stats else {
            f.render_widget(
                Paragraph::new(Span::styled("no latency samples (idle)", dim)),
                inner,
            );
            return;
        };
        let (min, max) = v
            .itl
            .iter()
            .fold((f32::MAX, 0.0f32), |(a, b), &x| (a.min(x), b.max(x)));
        let hist = |buckets: usize| -> Vec<f32> {
            let width = ((max - min) / buckets as f32).max(1e-3);
            let mut c = vec![0f32; buckets];
            for &x in &v.itl {
                c[(((x - min) / width) as usize).min(buckets - 1)] += 1.0;
            }
            c
        };
        let mut lines = vec![
            fit(
                vec![
                    seg(format!("ITL p50 {p50:.1}"), ink),
                    seg(format!("p95 {p95:.1}"), ink),
                    seg(format!("p99 {p99:.1} ms"), ink),
                ],
                w,
                "  ",
                dim,
            ),
            Line::from(Span::styled(
                spark(&hist(w.min(16)), w.min(16)),
                th.fg(Token::Cyanotype),
            )),
            Line::from(Span::styled(
                trunc(&format!("σ {sd:.2} ms · n={}", v.itl.len()), w),
                dim,
            )),
        ];
        if self.focused(Comp::Latency) {
            let buckets = 8;
            let c = hist(buckets);
            let top = c.iter().copied().fold(1.0f32, f32::max);
            let bw = ((max - min) / buckets as f32).max(1e-3);
            for (i, n) in c.iter().enumerate() {
                let lo = min + i as f32 * bw;
                let head = format!("{lo:>6.1} ms │ ");
                let cells = w.saturating_sub(text_width(&head) + 6);
                let bar = "█".repeat((n / top * cells as f32).round() as usize);
                let tok = if i < 4 {
                    Token::Jade
                } else if i < 6 {
                    Token::Gold
                } else {
                    Token::Crimson
                };
                lines.push(Line::from(vec![
                    Span::styled(head, dim),
                    Span::styled(bar, th.fg(tok)),
                    Span::styled(format!(" {}", *n as usize), dim),
                ]));
            }
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- LOGPROBS -------------------------------------------------------------------------

    pub(super) fn render_logprobs(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let detail = self.focused(Comp::Logprobs);
        let scrub = self
            .logprob_scrub_idx
            .min(self.logprob_history.len().saturating_sub(1));
        let target = if !self.logprob_history.is_empty() {
            Some(&self.logprob_history[scrub])
        } else {
            self.latest_logprobs.as_ref()
        };
        let metric = Line::from(target.map_or("H —".to_string(), |t| {
            format!("H {:.2} bits", t.entropy / LN2)
        }));
        let block = instrument("Logprobs", Some(metric), detail, th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let w = inner.width as usize;
        let dim = th.fg(Token::Dim);
        let Some(info) = target else {
            f.render_widget(
                Paragraph::new(Span::styled("no logprob data (idle)", dim)),
                inner,
            );
            return;
        };
        let ent_tok = |nats: f32| {
            let bits = nats / LN2;
            if bits < 1.0 {
                Token::Jade
            } else if bits < 3.0 {
                Token::Gold
            } else if bits < 5.0 {
                Token::Ember
            } else {
                Token::Crimson
            }
        };
        let mut lines = vec![Line::from(vec![
            Span::styled(
                trunc(
                    &format!("'{}'", self.token_str(info.chosen_id)),
                    w.saturating_sub(8),
                ),
                th.fg(Token::Parchment).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  p {:.2}", info.chosen_prob), th.fg(Token::Brass)),
        ])];
        // Entropy strip, newest on the right; the scrubbed token is reversed when focused.
        let n = self.logprob_history.len();
        let start = n.saturating_sub(w.min(64));
        let strip: Vec<Span<'static>> = self
            .logprob_history
            .iter()
            .enumerate()
            .skip(start)
            .map(|(i, h)| {
                let level = ((h.entropy / LN2 / 6.0) * 7.0).clamp(0.0, 7.0).round() as usize;
                let mut st = th.fg(ent_tok(h.entropy));
                if detail && i == scrub {
                    st = st.add_modifier(Modifier::REVERSED);
                }
                Span::styled(
                    ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"][level].to_string(),
                    st,
                )
            })
            .collect();
        lines.push(Line::from(strip));
        let want = if detail { 6 } else { 3 };
        for (tok, p) in info.top_candidates.iter().take(want) {
            let chosen = *tok == info.chosen_id;
            let st = if chosen {
                th.fg(Token::Brass)
            } else {
                th.fg(Token::Parchment)
            };
            lines.push(Line::from(vec![
                Span::styled(gauge(*p, 10), th.fg(Token::Cyanotype)),
                Span::styled(format!("  {p:.2}  "), st),
                Span::styled(
                    trunc(&format!("'{}'", self.token_str(*tok)), w.saturating_sub(18)),
                    st,
                ),
            ]));
        }
        if detail && n > 0 {
            lines.push(Line::from(Span::styled(
                trunc(
                    &format!("[token {} of {n}]  ← → scrub · Esc close", scrub + 1),
                    w,
                ),
                dim,
            )));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- WEIGHTS --------------------------------------------------------------------------

    pub(super) fn render_weights(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let dim = th.fg(Token::Dim);
        let block = instrument("Weights", None, false, th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let iw = inner.width as usize;
        let serving = self.classifier.read().is_some()
            || self.scheduler.as_ref().is_some_and(|s| s.has_model());
        // Spec loads store weights as the checkpoint has them, or re-encoded per the policy (its
        // KL gate may fall back to bit-exact), or as `--quant`.
        let mut lines = vec![Line::from(Span::styled(
            trunc(
                if serving {
                    "weights as stored or re-encoded"
                } else {
                    "no LLM loaded"
                },
                iw,
            ),
            dim,
        ))];
        let policy = match &self.quant_override {
            Some(f) => format!("--quant {}", f.name()),
            None => {
                let fastest = if self.reencode_policy
                    == crate::model::weight_format::ReencodePolicy::MaxSpeed
                    && self.variant_policy == crate::model::weight_format::VariantPolicy::Fastest
                {
                    " [fastest: max GB/s codec, KL≤0.05]"
                } else {
                    ""
                };
                format!(
                    "reencode {:?} · variant {:?}{fastest}",
                    self.reencode_policy, self.variant_policy
                )
            }
        };
        lines.push(Line::from(Span::styled(trunc(&policy, iw), dim)));
        let active_drafter = self
            .scheduler
            .as_ref()
            .and_then(|sc| sc.view().spec)
            .map(|sv| {
                sv.label
                    .strip_prefix("auto → ")
                    .unwrap_or(&sv.label)
                    .to_string()
            })
            .unwrap_or_else(|| {
                if self.draft_knobs.mode == crate::runtime::noise_draft::DraftMode::Off {
                    "off".to_string()
                } else {
                    "online-noise+suffix".to_string()
                }
            });
        let draft_line = format!(
            "draft {} → {} (k=0..{}, KTuner plain k=0 fallback)",
            self.draft_knobs.mode.name(),
            active_drafter,
            self.draft_knobs.max_k
        );
        lines.push(Line::from(Span::styled(trunc(&draft_line, iw), dim)));
        let compact_line = format!(
            "compact {} (85%→60% ctx) · Paged-Radix KV prefix reuse",
            self.compact_config.mode.name()
        );
        lines.push(Line::from(Span::styled(trunc(&compact_line, iw), dim)));
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- MEMORY ---------------------------------------------------------------------------

    pub(super) fn render_memory(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        let block = instrument(
            "Memory",
            Some(Line::from(format!("KV {:.0}%", v.kv_pct))),
            false,
            th,
        );
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let iw = inner.width as usize;
        let (used, total) = self.kv_pages();
        let rss = crate::hardware::current_resident_memory_bytes() as f64 / 1e9;
        let lines = vec![
            Line::from(vec![
                Span::styled("KV ", th.fg(Token::Dim)),
                Span::styled(
                    gauge(v.kv_pct / 100.0, iw.saturating_sub(10)),
                    th.fg(Token::Cyanotype),
                ),
                Span::styled(format!(" {:>3.0}%", v.kv_pct), th.fg(Token::Parchment)),
            ]),
            Line::from(Span::styled(
                trunc(&format!("RSS {rss:.1} GB · pages {used}/{total}"), iw),
                th.fg(Token::Dim),
            )),
        ];
        f.render_widget(Paragraph::new(lines), inner);
    }

    /// `(used, capacity)` KV pages of the active model's pool.
    fn kv_pages(&self) -> (usize, usize) {
        self.scheduler
            .as_ref()
            .and_then(|s| s.model())
            .and_then(|m| m.kv_pool())
            .map_or((0, 0), |p| (p.used_pages(), p.capacity_pages()))
    }

    // ---- DEVICE ---------------------------------------------------------------------------

    pub(super) fn render_device(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        let block = instrument(
            "Device",
            Some(Line::from(Span::styled(
                format!("{:.0} GB/s", v.ceiling),
                th.fg(Token::Cyanotype),
            ))),
            false,
            th,
        );
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let iw = inner.width as usize;
        let name = if v.s.device_name.is_empty() {
            self.host_lines
                .first()
                .cloned()
                .unwrap_or_else(|| "CPU".into())
        } else {
            v.s.device_name.clone()
        };
        let ds = self.device_status;
        let mut lines = vec![
            Line::from(Span::styled(trunc(&name, iw), th.fg(Token::Parchment))),
            Line::from(vec![
                Span::styled(
                    codec_bench_text(ds.codec),
                    th.fg(if ds.codec.0 == 2 {
                        Token::Jade
                    } else {
                        Token::Dim
                    }),
                ),
                Span::styled(
                    format!(" · tune {}/{}", ds.tune.0, ds.tune.1),
                    th.fg(Token::Dim),
                ),
            ]),
            Line::from(Span::styled(
                format!("ceiling {:.1} GB/s", v.ceiling),
                th.fg(Token::Cyanotype),
            )),
        ];
        #[cfg(target_os = "macos")]
        if let Some(a) = crate::graph::ane::probed() {
            lines.push(Line::from(Span::styled(
                trunc(
                    &format!(
                        "ANE {}-core {} · programs {}/{}",
                        a.cores,
                        a.arch,
                        crate::graph::ane::programs_compiled(),
                        crate::graph::ane::ANE_MAX_PROGRAMS
                    ),
                    iw,
                ),
                th.fg(Token::Dim),
            )));
        }
        if let Some(a) = self.http_addr {
            lines.push(Line::from(Span::styled(
                trunc(&format!("http {a}"), iw),
                th.fg(Token::Dim),
            )));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- GAP ------------------------------------------------------------------------------

    /// Where a decode token's wall time goes against the floor of streaming its weights at the
    /// measured ceiling: weights, GPU time beyond that, host time inside the run, time outside it.
    pub(super) fn render_gap(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        let g = v.s.gap;
        let metric = Line::from(match g.roofline_pct() {
            Some(p) => Span::styled(
                format!("{p:.0}%"),
                th.fg(Theme::tier(p)).add_modifier(Modifier::BOLD),
            ),
            None => Span::styled("—", th.fg(Token::Dim)),
        });
        let block = instrument("Gap", Some(metric), self.focused(Comp::Gap), th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let (w, h, dim) = (
            inner.width as usize,
            inner.height as usize,
            th.fg(Token::Dim),
        );
        if g.samples == 0 {
            f.render_widget(
                Paragraph::new(Span::styled(trunc("no Metal decode samples yet", w), dim)),
                inner,
            );
            return;
        }
        let ms = g.segments();
        let parts = [
            ("weights", Token::Jade),
            ("gpu excess", Token::Ember),
            ("run host", Token::Gold),
            ("outside", Token::Crimson),
        ];
        let bar: Vec<(Token, f32)> = parts.iter().zip(ms).map(|(p, m)| (p.1, m)).collect();
        let ink = th.fg(Token::Parchment);
        let item = |i: usize| -> Vec<Span<'static>> {
            vec![
                Span::styled("■ ", th.fg(parts[i].1)),
                Span::styled(format!("{} {:.3}", parts[i].0, ms[i]), ink),
            ]
        };
        let mut lines = vec![Line::from(stacked_bar(&bar, w, th))];
        if h >= 4 {
            lines.push(fit(vec![item(0), item(1)], w, "  ", dim));
            lines.push(fit(vec![item(2), item(3)], w, "  ", dim));
        } else if h == 3 {
            let mut spans = Vec::new();
            for (i, m) in ms.iter().enumerate() {
                spans.push(Span::styled("■", th.fg(parts[i].1)));
                spans.push(Span::styled(format!("{m:.3} "), ink));
            }
            spans.push(Span::styled("ms", dim));
            lines.push(Line::from(spans));
        }
        lines.push(Line::from(match g.deficit_labels(0.9) {
            None => Span::styled("to 90%: ceiling unmeasured", dim),
            Some((long, short)) => {
                let reached = g.ms_to(0.9) <= 0.0;
                let text = if text_width(&long) <= w { long } else { short };
                Span::styled(
                    trunc(&text, w),
                    th.fg(if reached { Token::Jade } else { Token::Ember }),
                )
            }
        }));
        if h >= 5 {
            lines.push(Line::from(Span::styled(
                trunc(
                    &format!(
                        "wall {:.3} · run {:.3} · gpu {:.3} ms · {:.0} disp",
                        g.wall_ms, g.run_ms, g.gpu_ms, g.dispatches
                    ),
                    w,
                ),
                dim,
            )));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- KERNELS --------------------------------------------------------------------------

    /// The last per-kind kernel profile (`/profile on`, or Ctrl-K, samples one decode token).
    pub(super) fn render_kernels(&self, f: &mut Frame, area: Rect, th: &Theme) {
        self.kernels_visible.set(true);
        let total_us = self
            .kernel_profile
            .as_ref()
            .map(|(_, rows)| rows.iter().map(|r| r.us).sum::<f64>());
        let metric = Line::from(Span::styled(
            total_us.map_or("—".to_string(), |u| format!("{u:.0} µs")),
            th.fg(Token::Cyanotype),
        ));
        let block = instrument("Kernels", Some(metric), self.focused(Comp::Kernels), th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let (w, dim) = (inner.width as usize, th.fg(Token::Dim));
        let Some((at, rows)) = &self.kernel_profile else {
            let text = if self.profile_sampling || self.focused(Comp::Kernels) {
                "sampling: one decode token every 5 s"
            } else {
                "profile off · /profile on"
            };
            f.render_widget(Paragraph::new(Span::styled(trunc(text, w), dim)), inner);
            return;
        };
        let mark = |col: KernelSort, s: &str| {
            if self.kernel_sort == col {
                format!("{s}▼")
            } else {
                s.to_string()
            }
        };
        let mut lines = vec![Line::from(Span::styled(
            trunc(
                &kernel_line(
                    "kind",
                    "steps".into(),
                    mark(KernelSort::Time, "µs"),
                    mark(KernelSort::GpuShare, "%gpu"),
                    mark(KernelSort::Bandwidth, "GB/s"),
                    "%".into(),
                ),
                w,
            ),
            dim,
        ))];
        let total_us: f64 = rows.iter().map(|r| r.us).sum();
        for r in self
            .kernel_sort
            .sorted(rows)
            .iter()
            .take((inner.height as usize).saturating_sub(2))
        {
            lines.push(Line::from(Span::styled(
                trunc(
                    &kernel_line(
                        &trunc(&r.kind, 10),
                        r.steps.to_string(),
                        format!("{:.0}", r.us),
                        format!("{:.1}", gpu_share(r.us, total_us)),
                        format!("{:.1}", r.gbps),
                        format!("{:.0}%", r.pct),
                    ),
                    w,
                ),
                th.fg(Theme::tier(r.pct as f32)),
            )));
        }
        lines.push(Line::from(Span::styled(
            trunc(
                &format!(
                    "{} steps · sort: {} (s) · {} ago",
                    rows.iter().map(|r| r.steps).sum::<usize>(),
                    self.kernel_sort.name(),
                    super::render::secs(at.elapsed())
                ),
                w,
            ),
            dim,
        )));
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- LOAD -----------------------------------------------------------------------------

    pub(super) fn render_load(&self, f: &mut Frame, area: Rect, l: &Load, th: &Theme) {
        let (idx, done, total, pct) = load_stage(l);
        let block = instrument(
            "Load",
            Some(Line::from(Span::styled(
                format!("{pct:.0}%"),
                th.fg(Token::Cyanotype).add_modifier(Modifier::BOLD),
            ))),
            false,
            th,
        );
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let iw = inner.width as usize;
        let dim = th.fg(Token::Dim);
        let rate = if l.rate > 0.0 {
            format!("{:.1} MB/s", l.rate / 1e6)
        } else {
            "measuring".to_string()
        };
        let eta = if l.rate > 0.0 && total > done {
            format!("{:.0}s", (total - done) as f64 / l.rate)
        } else {
            "—".to_string()
        };
        let mut lines = vec![Line::from(vec![
            Span::styled(
                trunc(&l.id, iw.saturating_sub(rate.len() + eta.len() + 9)),
                th.fg(Token::Parchment).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {rate} · ETA {eta}"), dim),
        ])];
        let mut track = Vec::new();
        for (i, name) in ["RESOLVE", "DOWNLOAD", "MAP", "COMPILE"].iter().enumerate() {
            if i > 0 {
                track.push(Span::styled(" ▸ ", dim));
            }
            track.push(if i < idx {
                Span::styled(format!("✓{name}"), th.fg(Token::Jade))
            } else if i == idx {
                Span::styled(
                    format!("[{name}]"),
                    th.fg(Token::Brass).add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled(name.to_string(), dim)
            });
        }
        lines.push(Line::from(track));
        lines.push(Line::from(vec![
            Span::styled(
                gauge(pct / 100.0, iw.saturating_sub(14).max(4)),
                th.fg(Token::Cyanotype),
            ),
            Span::styled(
                format!(" {:.0}/{:.0} MB", done as f64 / 1e6, total as f64 / 1e6),
                th.fg(Token::Parchment),
            ),
        ]));
        lines.push(Line::from(Span::styled(
            trunc(&l.reencode_text(self.reencode_policy), iw),
            dim,
        )));
        lines.push(Line::from(Span::styled(trunc(&l.progress_text(), iw), dim)));
        f.render_widget(Paragraph::new(lines), inner);
    }

    // ---- AVATAR ---------------------------------------------------------------------------

    pub(super) fn render_avatar(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let name = self.mascot.state_name();
        let block = instrument(name, None, false, th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let ticker = self.mascot.ticker();
        let [art, tick] = ratatui::layout::Layout::vertical([
            ratatui::layout::Constraint::Min(0),
            ratatui::layout::Constraint::Length(1),
        ])
        .areas(inner);
        match self.mascot.render_sprite(art) {
            Some(lines) => f.render_widget(Paragraph::new(lines), art),
            None => f.render_widget(
                Paragraph::new(Line::from(self.mascot.render_mini_badge())),
                art,
            ),
        }
        f.render_widget(
            Paragraph::new(Span::styled(
                trunc(&ticker, tick.width as usize),
                th.fg(Token::Dim),
            )),
            tick,
        );
    }

    // ---- DOCK (Compact tier) --------------------------------------------------------------

    /// One-line form of each component; each stays on a single dock row.
    pub(super) fn dock_atoms(&self, v: &Vitals, th: &Theme) -> Vec<Vec<Span<'static>>> {
        let tier = Theme::tier(v.pct);
        let mut a = vec![
            vec![
                Span::styled(format!("ROOF {:.0}% ", v.pct), th.fg(tier)),
                Span::styled("▕", th.fg(Token::Hairline)),
                Span::styled(gauge(v.pct / 100.0, 8), th.fg(tier)),
                Span::styled("▏", th.fg(Token::Hairline)),
            ],
            seg(format!("TOK {:.0}", v.s.current_tps), th.fg(Token::Brass)),
            seg(
                v.itl_stats()
                    .map_or("p99 —".to_string(), |s| format!("p99 {:.1} ms", s.2)),
                th.fg(Token::Cyanotype),
            ),
            seg(format!("KV {:.0}%", v.kv_pct), th.fg(Token::Parchment)),
        ];
        if let Some(p) = v.s.gap.roofline_pct() {
            a.push(seg(format!("gap {p:.0}%"), th.fg(Theme::tier(p))));
        }
        if let Some(t) = self
            .logprob_history
            .back()
            .or(self.latest_logprobs.as_ref())
        {
            a.push(seg(
                format!("H {:.1} bits", t.entropy / LN2),
                th.fg(Token::Parchment),
            ));
        }
        if v.ceiling > 0.0 {
            a.push(seg(
                format!("{:.0} GB/s", v.ceiling),
                th.fg(Token::Cyanotype),
            ));
        }
        a
    }

    /// Atoms wrapped greedily into rows of at most `width` cells, never splitting an atom.
    pub(super) fn dock_lines(&self, v: &Vitals, th: &Theme, width: usize) -> Vec<Line<'static>> {
        let sep = " · ";
        let mut lines: Vec<Vec<Span<'static>>> = vec![vec![]];
        let mut used = 0;
        for atom in self.dock_atoms(v, th) {
            let w: usize = atom.iter().map(|s| s.width()).sum();
            let row = lines.last_mut().expect("one row exists");
            if !row.is_empty() && used + text_width(sep) + w > width {
                lines.push(atom);
                used = w;
                continue;
            }
            if !row.is_empty() {
                row.push(Span::styled(sep, th.fg(Token::Hairline)));
                used += text_width(sep);
            }
            row.extend(atom);
            used += w;
        }
        lines.into_iter().map(Line::from).collect()
    }

    pub(super) fn render_dock(&self, f: &mut Frame, area: Rect, v: &Vitals, th: &Theme) {
        let metric = Line::from(Span::styled(
            format!("{:.1}%", v.pct),
            th.fg(Theme::tier(v.pct)).add_modifier(Modifier::BOLD),
        ));
        let block = instrument("Telemetry", Some(metric), false, th);
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(self.dock_lines(v, th, inner.width as usize)),
            inner,
        );
    }

    /// Renders one sidebar component into `rect`.
    pub(super) fn render_comp(&self, f: &mut Frame, c: Comp, rect: Rect, v: &Vitals, th: &Theme) {
        match c {
            Comp::Avatar => self.render_avatar(f, rect, th),
            Comp::Roofline => self.render_roofline(f, rect, v, th),
            Comp::Gap => self.render_gap(f, rect, v, th),
            Comp::Tokens => self.render_tokens(f, rect, v, th),
            Comp::Latency => self.render_latency(f, rect, v, th),
            Comp::Logprobs => self.render_logprobs(f, rect, th),
            Comp::Weights => self.render_weights(f, rect, th),
            Comp::Memory => self.render_memory(f, rect, v, th),
            Comp::Device => self.render_device(f, rect, v, th),
            Comp::Kernels => self.render_kernels(f, rect, th),
            Comp::Load => {
                if let Some(l) = &self.load {
                    self.render_load(f, rect, l, th);
                }
            }
        }
    }
}
