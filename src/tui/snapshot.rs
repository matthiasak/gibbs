//! Test-only: a frame buffer as HTML, and the fixture app the README screenshot is taken from.
//! `GIBBS_TUI_SNAPSHOT=1 cargo test --lib -q tui::snapshot -- --ignored` writes
//! `target/tui-snapshot.html`.

use super::{mascot::MascotState, theme, DeviceStatus, Kind, TuiApp};
use crate::telemetry::{EngineTelemetry, GapSample, KernelRow};
use ratatui::{
    backend::TestBackend,
    buffer::Buffer,
    style::{Color, Modifier},
    Terminal,
};
use std::{fmt::Write as _, time::{Duration, Instant}};

const DEFAULT_FG: (u8, u8, u8) = (0xF5, 0xEE, 0xDB);
const DEFAULT_BG: (u8, u8, u8) = (0x14, 0x10, 0x0D);

/// xterm-256 palette entry.
pub(super) fn xterm256(i: u8) -> (u8, u8, u8) {
    const BASE: [(u8, u8, u8); 16] = [
        (0, 0, 0),
        (205, 0, 0),
        (0, 205, 0),
        (205, 205, 0),
        (0, 0, 238),
        (205, 0, 205),
        (0, 205, 205),
        (229, 229, 229),
        (127, 127, 127),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (92, 92, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ];
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    match i {
        0..=15 => BASE[i as usize],
        16..=231 => {
            let n = (i - 16) as usize;
            (LEVELS[n / 36], LEVELS[(n / 6) % 6], LEVELS[n % 6])
        }
        _ => {
            let g = 8 + 10 * (i - 232);
            (g, g, g)
        }
    }
}

fn rgb(c: Color, default: (u8, u8, u8)) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::Indexed(i) => xterm256(i),
        Color::Reset => default,
        Color::Black => xterm256(0),
        Color::Red => xterm256(1),
        Color::Green => xterm256(2),
        Color::Yellow => xterm256(3),
        Color::Blue => xterm256(4),
        Color::Magenta => xterm256(5),
        Color::Cyan => xterm256(6),
        Color::Gray => xterm256(7),
        Color::DarkGray => xterm256(8),
        Color::LightRed => xterm256(9),
        Color::LightGreen => xterm256(10),
        Color::LightYellow => xterm256(11),
        Color::LightBlue => xterm256(12),
        Color::LightMagenta => xterm256(13),
        Color::LightCyan => xterm256(14),
        Color::White => xterm256(15),
    }
}

/// Escapes `s`; symbols outside the font's box/block coverage get a one-cell box so a fallback
/// font's wider advance cannot shift the row.
fn escape(s: &str, out: &mut String) {
    for c in s.chars() {
        let u = c as u32;
        let fallback = (0x2190..=0x24FF).contains(&u) || (0x25A0..=0x2BFF).contains(&u);
        if fallback {
            out.push_str("<span style=\"display:inline-block;width:1ch;height:16px;vertical-align:top;overflow:hidden\">");
        }
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            c => out.push(c),
        }
        if fallback {
            out.push_str("</span>");
        }
    }
}

/// Monospace `<pre id="t">` with one coloured span per run of identically styled cells.
pub(super) fn buffer_to_html(buf: &Buffer) -> String {
    let area = buf.area;
    let mut out = String::from(
        "<!doctype html><meta charset=\"utf-8\"><body style=\"margin:0;background:#14100D\">\
         <pre id=\"t\" style=\"display:inline-block;margin:0;padding:8px;background:#14100D;\
         color:#F5EEDB;font:14px/16px 'SF Mono',Menlo,'DejaVu Sans Mono',monospace\">",
    );
    for y in 0..area.height {
        let mut run: Option<(String, String)> = None; // (style, text)
        let mut skip = 0usize;
        let flush = |run: &mut Option<(String, String)>, out: &mut String| {
            if let Some((style, text)) = run.take() {
                let _ = write!(out, "<span style=\"{style}\">");
                escape(&text, out);
                out.push_str("</span>");
            }
        };
        for x in 0..area.width {
            let cell = &buf[(x, y)];
            if skip > 0 {
                skip -= 1;
                continue;
            }
            skip = ratatui::text::Span::raw(cell.symbol())
                .width()
                .saturating_sub(1);
            let (mut fg, mut bg) = (rgb(cell.fg, DEFAULT_FG), rgb(cell.bg, DEFAULT_BG));
            if cell.modifier.contains(Modifier::REVERSED) {
                std::mem::swap(&mut fg, &mut bg);
            }
            let mut style = format!(
                "color:#{:02X}{:02X}{:02X};background:#{:02X}{:02X}{:02X}",
                fg.0, fg.1, fg.2, bg.0, bg.1, bg.2
            );
            if cell.modifier.contains(Modifier::BOLD) {
                style.push_str(";font-weight:bold");
            }
            if cell.modifier.contains(Modifier::UNDERLINED) {
                style.push_str(";text-decoration:underline");
            }
            if cell.modifier.contains(Modifier::DIM) {
                style.push_str(";opacity:.65");
            }
            match &mut run {
                Some((s, text)) if *s == style => text.push_str(cell.symbol()),
                _ => {
                    flush(&mut run, &mut out);
                    run = Some((style, cell.symbol().to_string()));
                }
            }
        }
        flush(&mut run, &mut out);
        out.push('\n');
    }
    out.push_str("</pre>");
    out
}

/// E2B-like decode: 187 tok/s against a 329 tok/s roof.
pub(super) fn fixture_app() -> TuiApp {
    let telemetry = EngineTelemetry::new();
    telemetry
        .is_tui_active
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let mut app = TuiApp::new(telemetry.clone());
    // `new` probes the real GPU; the fixture pins its own readings.
    telemetry.set_hardware_profile("Apple M5 Pro", 273.0);
    telemetry.set_model_size_bytes((273.0e9 / 329.0) as u64);
    telemetry.record_tokens(561);
    let mut s = 1873u64;
    for _ in 0..300 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let jitter = ((s >> 40) % 1200) as u64;
        let spike = if (s >> 20) % 41 == 0 { 3000 } else { 0 };
        telemetry.record_itl(Duration::from_micros(5000 + jitter + spike));
    }
    for ms in [31, 36, 44, 29, 52] {
        telemetry.record_ttft(Duration::from_millis(ms));
    }
    for _ in 0..42 {
        telemetry.record_request_completed();
    }
    telemetry
        .total_tokens_generated
        .fetch_add(9_412, std::sync::atomic::Ordering::Relaxed);
    telemetry.log_info("model gemma-4-E2B-it-qat ready");
    telemetry.log_info("compiled graph program: 1 segment, Metal");
    telemetry.log_info("codec-bench: measured 14 formats");
    app.gemm_peak_gflops = 14_000.0;
    // The runtime-fed instruments: where a 5.4 ms token goes, the last kernel profile, and a
    // finished codec bench with its tuned shapes.
    telemetry.record_gap(GapSample {
        wall_ms: 5.4,
        run_ms: 5.2,
        gpu_ms: 4.5,
        weights_ms: 3.9,
        dispatches: 612,
    });
    app.profile_sampling = true;
    app.device_status = DeviceStatus {
        codec: (2, 14, 14),
        tune: (31, 31),
    };
    let row = |kind: &str, steps, us, gbps, pct| KernelRow {
        kind: kind.into(),
        steps,
        us,
        gbps,
        pct,
    };
    app.kernel_profile = Some((
        Instant::now(),
        vec![
            row("GemvPacked", 232, 3_650.0, 231.0, 84.0),
            row("RmsNorm", 212, 410.0, 38.0, 14.0),
            row("CachedAttention", 35, 260.0, 52.0, 19.0),
            row("Elementwise", 128, 150.0, 21.0, 8.0),
            row("Rope", 50, 40.0, 9.0, 3.0),
        ],
    ));
    app.running = 1;
    for (i, v) in app.spark.iter_mut().enumerate().skip(80) {
        *v = 150 + ((i * 37) % 60) as u64;
    }
    app.push_entry(
        "/chat Explain Gibbs free energy".into(),
        Kind::Text,
        "G = H − TS: the energy available to do work at constant T and P.".into(),
    );
    app.push_entry(
        "/chat And the phase rule?".into(),
        Kind::Running,
        String::new(),
    );
    for (i, ent) in [0.3f32, 0.8, 2.4, 1.1, 4.2, 0.4, 0.2, 1.9, 3.3, 0.6]
        .iter()
        .enumerate()
    {
        app.logprob_history
            .push_back(crate::scheduler::TokenLogprobInfo {
                chosen_id: 100 + i as u32,
                chosen_prob: (1.0 - ent / 6.0).clamp(0.05, 0.99),
                top_candidates: vec![(100 + i as u32, 0.62), (7, 0.21), (9, 0.08)],
                entropy: *ent,
            });
    }
    app.logprob_scrub_idx = app.logprob_history.len() - 1;
    app.mascot.set_state(MascotState::Generating);
    app.mascot.frame_tick = 5;
    app
}

pub(super) fn render_buffer(app: &TuiApp, w: u16, h: u16) -> Buffer {
    let mut t = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    t.draw(|f| app.render(f)).expect("draw");
    t.backend().buffer().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_dump_groups_runs_and_resolves_indexed_colours() {
        let mut buf = Buffer::empty(ratatui::layout::Rect::new(0, 0, 4, 1));
        buf.set_string(
            0,
            0,
            "a<",
            ratatui::style::Style::default().fg(Color::Indexed(178)),
        );
        buf.set_string(
            2,
            0,
            "bc",
            ratatui::style::Style::default().fg(Color::Rgb(1, 2, 3)),
        );
        let html = buffer_to_html(&buf);
        assert!(html.contains("color:#D7AF00"), "{html}");
        assert!(html.contains("a&lt;") && html.contains("color:#010203"));
        assert_eq!(html.matches("<span").count(), 2);
        assert_eq!(xterm256(232), (8, 8, 8));
    }

    /// `GIBBS_TUI_SNAPSHOT=1 cargo test --lib -q tui::snapshot -- --ignored`
    #[test]
    #[ignore]
    fn snapshot_tui() {
        if std::env::var("GIBBS_TUI_SNAPSHOT").is_err() {
            return;
        }
        let _g = theme::test_guard(theme::Theme {
            truecolor: true,
            mono: false,
        });
        let app = fixture_app();
        let html = buffer_to_html(&render_buffer(&app, 160, 50));
        std::fs::create_dir_all("target").expect("target dir");
        std::fs::write("target/tui-snapshot.html", html).expect("write snapshot");
    }
}
