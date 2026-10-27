//! Drawing. Every panel shows measured or configured values only.

use super::layout::{self, Tier};
use super::theme::{theme, Theme, Token};
use super::widgets::{chip, instrument};
use super::{Entry, Focus, Kind, TuiApp};
use crate::graph::placement::PlacementRow;
use crate::hardware::Device;
use crate::scheduler::SequenceState;
use crate::telemetry::LogLevel;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Row, Table, Wrap},
    Frame,
};

/// Console text indent under the `HH:MM:SS ` timestamp.
const INDENT: &str = "         ";

/// `12345678` → `12,345,678`.
pub(super) fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub(super) fn secs(d: std::time::Duration) -> String {
    let s = d.as_secs_f64();
    if s < 1.0 {
        format!("{:.0} ms", s * 1e3)
    } else {
        format!("{s:.1} s")
    }
}

/// Splits `text` into lines of at most `width` chars, breaking after the last space that fits
/// (mid-word only when a word is longer than the line); newlines are kept.
// ponytail: counts chars, not display cells; wide CJK/emoji output overflows its line.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for line in text.split('\n') {
        let chars: Vec<char> = line.chars().collect();
        let mut start = 0;
        while chars.len() - start > width {
            let cut = match chars[start..start + width].iter().rposition(|&c| c == ' ') {
                Some(i) if i > 0 => i + 1,
                _ => width,
            };
            let piece: String = chars[start..start + cut].iter().collect();
            out.push(piece.trim_end().to_string());
            start += cut;
        }
        out.push(chars[start..].iter().collect());
    }
    out
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let (w, h) = (w.min(area.width), h.min(area.height));
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

impl TuiApp {
    pub(super) fn render(&self, f: &mut Frame) {
        let area = f.area();
        let th = theme();
        if th.truecolor {
            f.render_widget(Block::default().style(th.bg(Token::Void)), area);
        }
        let r = layout::regions(area);
        let v = self.vitals();
        self.kernels_visible.set(false);
        self.render_status(f, r.status, &v, &th);

        let compact = r.tier == Tier::Compact;
        let loading = self.load.is_some();
        let mut main = r.main;
        let mut hidden = 0;

        if compact {
            if let Some(dock) = r.dock {
                // The dock shrinks to its content height; unused flex rows return to `main`.
                let need = self
                    .dock_lines(&v, &th, dock.width.saturating_sub(2) as usize)
                    .len() as u16
                    + 2;
                let used = need.min(dock.height);
                main.height += dock.height - used;
                let d = Rect::new(dock.x, dock.y + dock.height - used, dock.width, used);
                self.render_dock(f, d, &v, &th);
            }
        } else if let Some(side) = r.sidebar_area() {
            let cw = r.sidebar.first().map_or(0, |c| c.width);
            let specs = layout::default_specs(cw, area.height, loading, self.detail);
            let plan = layout::flow(&specs, side, r.sidebar.len() as u16);
            hidden = plan.hidden;
            for (c, rect) in plan.placed {
                self.render_comp(f, c, rect, &v, &th);
            }
        }
        self.render_keys(f, r.keys, &th, hidden);

        let stack = layout::main_stack(
            main,
            true,
            loading && compact,
            compact && self.detail.is_some(),
        );
        if let (Some(c), Some(top)) = (self.detail, stack.detail) {
            self.render_comp(f, c, top, &v, &th);
        }
        if let Some(req) = stack.requests {
            match &self.load {
                Some(l) if compact => self.render_load(f, req, l, &th),
                _ => self.render_requests(f, req, &th),
            }
        }
        self.render_console(f, stack.console, &th);
        self.render_log(f, stack.log, &th);

        if self.focus == Focus::Picker {
            let loading = self.load.as_ref().map(|l| l.id.as_str());
            self.picker.render(f, area, &self.model_id(), loading);
        }
        if self.show_template_modal {
            self.render_template_modal(f, area, &th);
        }
        if self.show_settings_modal {
            self.render_settings_modal(f, area, &th);
        }
        if let Some((rows, scroll)) = &self.placement_modal {
            self.render_placement_modal(f, area, rows, *scroll, &th);
        }
    }

    fn render_requests(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let dim = th.fg(Token::Dim);
        let Some(sched) = &self.scheduler else {
            f.render_widget(
                Paragraph::new(Span::styled("no scheduler", dim))
                    .block(instrument("Requests", None, false, th)),
                area,
            );
            return;
        };
        let v = sched.view();
        let active = v
            .seqs
            .iter()
            .filter(|s| matches!(s.state, SequenceState::Prefilling | SequenceState::Decoding))
            .count();
        let footer = format!(
            " waiting {} · prefill {} tok/step · step {} ",
            v.waiting,
            v.prefill_budget,
            if v.busy_step.is_zero() {
                "-".to_string()
            } else {
                secs(v.busy_step)
            }
        );
        let block = instrument(
            "Requests",
            Some(Line::from(format!("{active} active"))),
            false,
            th,
        )
        .title_bottom(Span::styled(footer, dim));
        if v.seqs.is_empty() {
            f.render_widget(Paragraph::new(Span::styled("idle", dim)).block(block), area);
            return;
        }
        let rows = v.seqs.iter().map(|s| {
            let (state, tok, prompt) = match s.state {
                SequenceState::Prefilling => (
                    "prefill",
                    Token::Gold,
                    format!("{}/{}", s.prefilled, s.prompt),
                ),
                SequenceState::Decoding => ("decode", Token::Ember, s.prompt.to_string()),
                SequenceState::Waiting => ("waiting", Token::Dim, s.prompt.to_string()),
                SequenceState::Finished => ("done", Token::Dim, s.prompt.to_string()),
            };
            let ink = th.fg(Token::Parchment);
            Row::new(vec![
                Span::styled(format!("#{}", s.id), ink),
                Span::styled(state, th.fg(tok)),
                Span::styled(prompt, ink),
                Span::styled(format!("{}/{}", s.generated, s.max_new), ink),
                Span::styled(secs(s.submitted.elapsed()), dim),
                Span::styled(
                    crate::server::core::request_label(s.id).unwrap_or_else(|| "tui".into()),
                    dim,
                ),
            ])
        });
        let header = Row::new(["id", "state", "prompt", "output", "age", "client"]).style(dim);
        f.render_widget(
            Table::new(
                rows,
                [
                    Constraint::Length(7),
                    Constraint::Length(8),
                    Constraint::Length(11),
                    Constraint::Length(11),
                    Constraint::Length(8),
                    Constraint::Min(10),
                ],
            )
            .header(header)
            .block(block),
            area,
        );
    }

    fn render_console(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let dim = th.fg(Token::Dim);
        let hint: String = if let Some(i) = self.prompt_history_idx {
            format!(
                " history {}/{} · enter retry · edit · ↓ newer · esc cancel/back ",
                i + 1,
                self.prompt_history.len()
            )
        } else {
            match self.focus {
                Focus::Prompt => {
                    " enter run · ↑↓ history · F4 options · esc cancel/back · trackpad/pgup/pgdn scroll "
                        .into()
                }
                _ => {
                    " /chat /image /decide <q> | a, b /embed /classify <t> [|| pair] /connect /agent "
                        .into()
                }
            }
        };
        let block = instrument("Console", None, self.focus == Focus::Prompt, th)
            .title_bottom(Span::styled(hint, dim));
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 2 {
            return;
        }
        let [hist, input] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
        let width = hist.width as usize;
        let height = hist.height as usize;

        // Newest entries first, until the window (plus scroll) is covered.
        let mut blocks: Vec<Vec<Line>> = Vec::new();
        let mut total = 0;
        for e in self.history.iter().rev() {
            let lines = entry_lines(e, width, th);
            total += lines.len();
            blocks.push(lines);
            if total >= height + self.scroll {
                break;
            }
        }
        let lines: Vec<Line> = blocks.into_iter().rev().flatten().collect();
        let scroll = self.scroll.min(lines.len().saturating_sub(height));
        let end = lines.len() - scroll;
        let start = end.saturating_sub(height);
        let mut shown: Vec<Line> = lines[start..end].to_vec();
        if self.history.is_empty() {
            shown.push(Line::from(Span::styled(
                "Prompts and model loads appear here.",
                dim,
            )));
        }
        f.render_widget(Paragraph::new(shown), hist);

        let line = if self.focus == Focus::Prompt {
            let room = width.saturating_sub(3);
            let skip = self.input.chars().count().saturating_sub(room);
            let visible: String = self.input.chars().skip(skip).collect();
            Line::from(vec![
                Span::styled("› ", th.fg(Token::Brass)),
                Span::styled(visible, th.fg(Token::Parchment)),
                Span::styled("▏", th.fg(Token::Brass)),
            ])
        } else {
            Line::from(Span::styled("› press i to type a prompt", dim))
        };
        f.render_widget(Paragraph::new(line), input);
    }

    fn render_log(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let block = instrument("Log", None, false, th);
        let rows = block.inner(area).height as usize;
        let lines: Vec<Line> = self
            .telemetry
            .get_recent_logs(rows)
            .into_iter()
            .map(|e| {
                let level = match e.level {
                    LogLevel::Info => th.fg(Token::Jade),
                    LogLevel::Warn => th.fg(Token::Gold).add_modifier(Modifier::BOLD),
                    LogLevel::Debug => th.fg(Token::Dim),
                };
                Line::from(vec![
                    Span::styled(format!("{} ", e.timestamp), th.fg(Token::Dim)),
                    Span::styled(format!("{:<5} ", e.level.as_str()), level),
                    Span::styled(e.message, th.fg(Token::Parchment)),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_keys(&self, f: &mut Frame, area: Rect, th: &Theme, hidden: usize) {
        let keys: &[(&str, &str)] = match self.focus {
            Focus::Dashboard => &[
                ("i", "prompt"),
                ("m", "models"),
                ("F2", "templates"),
                ("F4", "options"),
                ("↑↓", "history"),
                ("q", "quit"),
            ],
            Focus::Prompt => &[
                ("enter", "run"),
                ("↑↓", "history"),
                ("F2", "templates"),
                ("F4", "options"),
                ("esc/ctrl-x", "cancel/back"),
                ("ctrl-c", "quit"),
            ],
            Focus::Picker => &[
                ("enter", "load"),
                ("tab", "sort"),
                ("ctrl-d", "delete cached"),
                ("esc", "close"),
                ("ctrl-c", "quit"),
            ],
        };
        let mut spans = Vec::new();
        for (k, what) in keys {
            if !spans.is_empty() {
                spans.push(Span::raw("  "));
            }
            spans.extend(chip(k, what, th));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
        if hidden > 0 {
            let note = format!("+{hidden} hidden (resize)");
            let w = (note.chars().count() as u16).min(area.width);
            let right = Rect::new(area.x + area.width - w, area.y, w, area.height);
            f.render_widget(Paragraph::new(Span::styled(note, th.fg(Token::Dim))), right);
        }
    }

    fn render_template_modal(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let popup = centered(
            area,
            (area.width * 60 / 100).max(45),
            (area.height * 50 / 100).max(12),
        );
        f.render_widget(Clear, popup);
        let block = instrument("Chat Template & System Prompt (F2)", None, true, th);
        let names = crate::arch::templates::names();
        let bold = th.fg(Token::Parchment).add_modifier(Modifier::BOLD);
        let mut lines = vec![
            Line::from(Span::styled(
                format!(
                    " Select Template (0-{} / ↑↓, Enter/Esc to confirm):",
                    names.len()
                ),
                bold,
            )),
            Line::default(),
        ];
        let entries = std::iter::once(("Model Default", "tokenizer_config.json / GGUF metadata"))
            .chain(names.iter().map(|n| (*n, "arch/templates/*.jinja")));
        for (num, (name, preview)) in entries.enumerate() {
            let active = num == self.active_template;
            let marker = if active { " [*] " } else { " [ ] " };
            let style = if active {
                th.fg(Token::Brass).add_modifier(Modifier::BOLD)
            } else {
                th.fg(Token::Dim)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{marker}{num}. {name}: "), style),
                Span::styled(preview, th.fg(Token::Dim)),
            ]));
        }
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled(" System Prompt: ", bold),
            Span::styled(&self.system_prompt, th.fg(Token::Gold)),
        ]));
        f.render_widget(Paragraph::new(lines).block(block), popup);
    }

    fn render_settings_modal(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let popup = centered(
            area,
            (area.width * 74 / 100).max(58),
            (area.height * 60 / 100).max(16),
        );
        f.render_widget(Clear, popup);
        let block = instrument(
            "Policy & Speed Options (F4 — defaults = fastest for model)",
            None,
            true,
            th,
        );
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let (label, value, dim) = (
            th.fg(Token::Brass),
            th.fg(Token::Gold).add_modifier(Modifier::BOLD),
            th.fg(Token::Dim),
        );
        let hud = if self.show_roofline_hud {
            Span::styled("ENABLED", th.fg(Token::Jade))
        } else {
            Span::styled("DISABLED", th.fg(Token::Crimson))
        };
        let reencode_val = format!(
            "{:?}{}",
            self.reencode_policy,
            if self.reencode_policy == crate::model::weight_format::ReencodePolicy::MaxSpeed {
                " [FASTEST]"
            } else {
                ""
            }
        );
        let variant_val = format!(
            "{:?}{}",
            self.variant_policy,
            if self.variant_policy == crate::model::weight_format::VariantPolicy::Fastest {
                " [FASTEST]"
            } else {
                ""
            }
        );
        let resolved_drafter = self
            .scheduler
            .as_ref()
            .and_then(|sc| sc.view().spec)
            .map(|sv| {
                sv.label
                    .strip_prefix("auto → ")
                    .unwrap_or(&sv.label)
                    .to_string()
            })
            .unwrap_or_else(|| "online-noise+suffix".to_string());
        let draft_val = if self.draft_knobs.mode == crate::runtime::noise_draft::DraftMode::Auto {
            format!("AUTO → {resolved_drafter} [FASTEST]")
        } else {
            self.draft_knobs.mode.name().to_uppercase()
        };
        let compact_val = format!(
            "{}{}",
            self.compact_config.mode.name().to_uppercase(),
            if self.compact_config.mode == crate::compact::CompactionMode::Auto {
                " [FASTEST]"
            } else {
                ""
            }
        );
        let lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled(" [1] Re-encode Policy:   ", label),
                Span::styled(reencode_val, value),
                Span::styled(
                    " — highest-GB/s GPU codec for host ceiling (KL ≤ 0.05)  (1: max-speed / low-bit / bit-exact)",
                    dim,
                ),
            ]),
            Line::from(vec![
                Span::styled(" [2] Variant Policy:     ", label),
                Span::styled(variant_val, value),
                Span::styled(
                    " — ranks checkpoints by predicted decode tok/s  (2: fastest / fastest-4bit / quality)",
                    dim,
                ),
            ]),
            Line::from(vec![
                Span::styled(" [3] Roofline HUD:       ", label),
                hud,
                Span::styled(
                    " — live bandwidth/compute roofline telemetry  (3: toggle)",
                    dim,
                ),
            ]),
            Line::from(vec![
                Span::styled(" [4] Context Limit:      ", label),
                Span::styled(
                    format!("{} tokens", self.ctx_limit),
                    th.fg(Token::Parchment),
                ),
                Span::styled("  (4: cycle 4096 / 8192 / 16384 / 32768)", dim),
            ]),
            Line::from(vec![
                Span::styled(" [5] Spec Draft Mode:    ", label),
                Span::styled(draft_val, value),
                Span::styled(
                    " — MTP if in checkpoint, else zero-tap online-noise+suffix with KTuner k=0 plain fallback  (5: cycle auto / noise / suffix / model / off)",
                    dim,
                ),
            ]),
            Line::from(vec![
                Span::styled(" [6] Spec Max K:         ", label),
                Span::styled(format!("{}", self.draft_knobs.max_k), value),
                Span::styled("  (6: cycle 0 / 2 / 4 / 6 / 8 / 12)", dim),
            ]),
            Line::from(vec![
                Span::styled(" [7] Spec Target Accept: ", label),
                Span::styled(
                    format!("{:.0}%", self.draft_knobs.target_acceptance * 100.0),
                    value,
                ),
                Span::styled("  (7: cycle 50% / 65% / 75% / 85%)", dim),
            ]),
            Line::from(vec![
                Span::styled(" [8] Auto Compaction:    ", label),
                Span::styled(compact_val, value),
                Span::styled(
                    " — 4-tier trim at 85%→60% ctx preserving Paged-Radix KV prefix  (8: cycle auto / light / aggressive / off)",
                    dim,
                ),
            ]),
            Line::from(vec![
                Span::styled(" [9] Spec Objective:     ", label),
                Span::styled(self.draft_knobs.objective.name().to_uppercase(), value),
                Span::styled(
                    " — what KTuner maximizes: tok/s or tokens per verify forward  (9: speed / tokens)",
                    dim,
                ),
            ]),
            Line::from(vec![
                Span::styled(" [0] Draft Denoise Steps:", label),
                Span::styled(format!(" {}", self.draft_knobs.denoise_steps), value),
                Span::styled("  (0: cycle 1 / 2 / 3 / 4)", dim),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                " Esc / F4: Close modal | Enter: Save settings",
                dim,
            )),
        ];
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    /// `/placement`: every graph node group (op, device, avenue) of the loaded program, with the
    /// verdict; CPU nodes and the CPU verdict are drawn in the alert colour.
    fn render_placement_modal(
        &self,
        f: &mut Frame,
        area: Rect,
        rows: &[PlacementRow],
        scroll: usize,
        th: &Theme,
    ) {
        let popup = centered(
            area,
            (area.width * 70 / 100).max(50),
            (area.height * 70 / 100).max(14),
        );
        f.render_widget(Clear, popup);
        let block = instrument("Placement", None, true, th);
        let inner = block.inner(popup);
        f.render_widget(block, popup);
        let body = inner.height.saturating_sub(1) as usize;
        let lines = placement_modal_lines(rows);
        let start = scroll.min(lines.len().saturating_sub(body));
        let mut out: Vec<Line> = lines
            .into_iter()
            .skip(start)
            .take(body)
            .map(|(text, alert)| {
                Line::from(Span::styled(
                    text,
                    if alert {
                        th.fg(Token::Crimson).add_modifier(Modifier::BOLD)
                    } else {
                        th.fg(Token::Parchment)
                    },
                ))
            })
            .collect();
        out.push(Line::from(Span::styled(
            " Esc / Enter / q: close | ↑↓ PgUp PgDn: scroll",
            th.fg(Token::Dim),
        )));
        f.render_widget(Paragraph::new(out), inner);
    }
}

/// The `/placement` modal text: the placement table, a blank line, then the verdict. The flag
/// marks lines drawn as alerts: rows placed on the CPU and a verdict naming CPU ops.
pub(super) fn placement_modal_lines(rows: &[PlacementRow]) -> Vec<(String, bool)> {
    let table = crate::graph::placement::placement_table(rows);
    let mut out: Vec<(String, bool)> = table
        .lines()
        .enumerate()
        .map(|(i, l)| {
            let cpu = i > 0 && rows.get(i - 1).is_some_and(|r| r.device == Device::Cpu);
            (l.to_string(), cpu)
        })
        .collect();
    out.push((String::new(), false));
    out.push(match crate::graph::placement::placement_verdict(rows) {
        Ok(all) => (all, false),
        Err(cpu) => (cpu, true),
    });
    out
}

/// One console entry as wrapped lines, followed by a blank separator line.
fn entry_lines(e: &Entry, width: usize, th: &Theme) -> Vec<Line<'static>> {
    let body_w = width.saturating_sub(INDENT.len()).max(1);
    let mut out = Vec::new();
    for (i, chunk) in wrap(&e.prompt, body_w).into_iter().enumerate() {
        out.push(Line::from(vec![
            if i == 0 {
                Span::styled(format!("{} ", e.time), th.fg(Token::Dim))
            } else {
                Span::raw(INDENT)
            },
            Span::styled(chunk, th.fg(Token::Brass).add_modifier(Modifier::BOLD)),
        ]));
    }
    let tok = match e.kind {
        Kind::Running | Kind::Text => Token::Parchment,
        Kind::Json => Token::Gold,
        Kind::Image => Token::Cyanotype,
        Kind::Error => Token::Crimson,
    };
    if !e.output.is_empty() {
        for chunk in wrap(&e.output, body_w) {
            out.push(Line::from(vec![
                Span::raw(INDENT),
                Span::styled(chunk, th.fg(tok)),
            ]));
        }
    }
    let meta = match e.took {
        None => Span::styled(
            format!("running {}…", secs(e.started.elapsed())),
            th.fg(Token::Gold),
        ),
        Some(t) => {
            let mut parts: Vec<String> = Vec::new();
            if !e.stats.is_empty() {
                parts.push(e.stats.clone());
            }
            if !t.is_zero() {
                parts.push(secs(t));
            }
            Span::styled(parts.join(" · "), th.fg(Token::Dim))
        }
    };
    if !meta.content.is_empty() {
        out.push(Line::from(vec![Span::raw(INDENT), meta]));
    }
    out.push(Line::default());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_and_thousands() {
        assert_eq!(wrap("abcdef\n\nxy", 4), ["abcd", "ef", "", "xy"]);
        assert_eq!(wrap("the quick brown", 10), ["the quick", "brown"]);
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(999), "999");
    }

    #[test]
    fn placement_modal_flags_cpu_rows_and_the_verdict() {
        let row = |op: &str, device, count| PlacementRow {
            op: op.into(),
            device,
            avenue: "kernel".into(),
            count,
        };
        let gpu = placement_modal_lines(&[row("Linear", Device::Metal, 4)]);
        assert!(gpu.iter().all(|l| !l.1), "{gpu:?}");
        assert_eq!(gpu.last().unwrap().0, "100% Metal");
        let mixed =
            placement_modal_lines(&[row("Linear", Device::Metal, 3), row("Relu", Device::Cpu, 2)]);
        let flagged: Vec<&str> = mixed.iter().filter(|l| l.1).map(|l| l.0.as_str()).collect();
        assert_eq!(flagged.len(), 2, "{mixed:?}");
        assert!(flagged[0].starts_with("Relu") && flagged[1] == "2 CPU nodes (of 5): Relu ×2");
    }
}
