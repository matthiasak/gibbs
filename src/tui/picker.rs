//! Model picker overlay: fuzzy type-to-search over locally cached models, pinned ids, and Hugging Face Hub results.

use super::theme::{theme, Token};
use super::widgets::instrument;
use crate::model::hub::{
    branch_models, detect_quant_format, is_metadata_file, list_repo_files, select_weight_files,
    split_revision, split_weight_file, HuggingFaceHub, RepoFile,
};
use crate::runtime::draft::{draft_target, is_draft_for};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, Paragraph},
    Frame,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DEBOUNCE: Duration = Duration::from_millis(250);
/// Hub results per request; also the pool the fuzzy filter narrows.
const SEARCH_LIMIT: usize = 100;
/// Fewer hits than this widens the search with shortened terms (the server matches substrings).
const FEW_HITS: usize = 20;
/// Second ctrl-d within this window deletes.
const DELETE_CONFIRM: Duration = Duration::from_secs(3);

/// What the caller should do after a key press.
pub enum PickerAction {
    None,
    Close,
    Load(String),
    /// Confirmed removal of this cached repo.
    Delete(String),
    /// Attach this drafter to the active model (speculative decoding).
    AttachDraft(String),
    /// The re-encode (`"reencode"`) or variant (`"variant"`) policy was cycled and saved: the
    /// caller reloads the loaded model under it.
    Reload(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Sort {
    Likes,
    Downloads,
    Recent,
}

impl Sort {
    fn next(self) -> Self {
        match self {
            Sort::Likes => Sort::Downloads,
            Sort::Downloads => Sort::Recent,
            Sort::Recent => Sort::Likes,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Sort::Likes => "favorites",
            Sort::Downloads => "downloads",
            Sort::Recent => "recent",
        }
    }
    fn api(self) -> &'static str {
        match self {
            Sort::Likes => "likes",
            Sort::Downloads => "downloads",
            Sort::Recent => "lastModified",
        }
    }
    /// Index of the column this key orders in the picker table.
    fn column(self) -> usize {
        match self {
            Sort::Likes => 2,
            Sort::Downloads => 3,
            Sort::Recent => 4,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Entry {
    id: String,
    params: Option<u64>,
    downloads: Option<u64>,
    likes: Option<u64>,
    /// `lastModified`, RFC 3339 (sorts lexicographically).
    updated: Option<String>,
    /// Bytes on disk (cached rows) or estimated download size (hub rows).
    size: Option<u64>,
    gated: bool,
    cached: bool,
    /// Why the loader cannot load this model; `None` = loadable.
    blocker: Option<String>,
    /// Speculative drafter for this target model (repo id or name).
    draft_for: Option<String>,
    /// A quantized variant that replaces this too-large model: enter loads it instead.
    via: Option<Box<Entry>>,
    /// Available quantized variants sorted by size ascending.
    quants: Vec<Entry>,
    /// Active variant index in `quants`.
    quant_idx: usize,
}
struct Row {
    entry: Entry,
    hits: Vec<usize>,
}

type Key = (String, Sort);
type SearchResult = (Key, u64, Result<Vec<Entry>, String>);

/// Model picker state: query, merged candidate rows, and background hub searches.
pub struct Picker {
    hub: HuggingFaceHub,
    hf_token: Option<String>,
    pinned: Vec<String>,
    total_memory: Option<u64>,
    pub(super) query: String,
    query_changed: Instant,
    sort: Sort,
    selected: usize,
    selection_changed: Instant,
    /// One-shot status override (e.g. the blocker of a row the user tried to load).
    note: Option<(String, Token)>,
    /// Host memory-bandwidth ceiling (GB/s) behind the predicted tok/s; 0 = unmeasured.
    ceiling_gbps: f32,
    /// Id and time of the first ctrl-d press.
    pending_delete: Option<(String, Instant)>,
    local: Vec<Entry>,
    /// Hub list currently merged into the rows; kept while a newer search runs.
    hub_results: Vec<Entry>,
    /// Finished searches by (trimmed query, sort), so backspacing and re-sorting reuse them.
    searches: HashMap<Key, Result<Vec<Entry>, String>>,
    /// Generation of the request in flight per key; a response carrying any other
    /// generation is stale and dropped.
    in_flight: HashMap<Key, u64>,
    generation: u64,
    tx: Sender<SearchResult>,
    rx: Receiver<SearchResult>,
    /// Exact download size (selected weights + metadata) per hub id.
    sizes: HashMap<String, Result<u64, String>>,
    sizing: HashSet<String>,
    size_tx: Sender<(String, Result<u64, String>)>,
    size_rx: Receiver<(String, Result<u64, String>)>,
    /// `<id>@<branch>` rows of highlighted hub repos whose main holds no weights, by repo id.
    branches: HashMap<String, Vec<Entry>>,
    branch_tx: Sender<(String, Vec<Entry>)>,
    branch_rx: Receiver<(String, Vec<Entry>)>,
    /// Loadable quantized derivatives per too-large hub id, sorted by size ascending.
    quants: HashMap<String, Vec<Entry>>,
    quanting: HashSet<String>,
    quant_tx: Sender<(String, Vec<Entry>)>,
    quant_rx: Receiver<(String, Vec<Entry>)>,
    scan_tx: Sender<(Vec<Entry>, Option<u64>)>,
    scan_rx: Receiver<(Vec<Entry>, Option<u64>)>,
    pub scanning_local: bool,
    free_disk: Option<u64>,
    rows: Vec<Row>,
    total: usize,
    /// The model the scheduler runs (drafters for it are loadable).
    active: String,
    /// Weight headers by variant id: `None` while being read, `Some(Err)` when unreadable.
    headers: Headers,
}

type Headers = Arc<
    parking_lot::Mutex<
        HashMap<String, Option<Result<Arc<crate::model::variant::WeightHeader>, String>>>,
    >,
>;

impl Picker {
    /// `hub`: where models are cached/downloaded (`hub.cache_root`). `pinned`: ids always offered.
    /// `total_memory`: host RAM bytes, for the f32-size check.
    pub fn new(
        hub: HuggingFaceHub,
        hf_token: Option<String>,
        pinned: &[&str],
        total_memory: Option<u64>,
    ) -> Self {
        let (tx, rx) = channel();
        let (size_tx, size_rx) = channel();
        let (branch_tx, branch_rx) = channel();
        let (quant_tx, quant_rx) = channel();
        let (scan_tx, scan_rx) = channel();
        Self {
            hub,
            hf_token,
            pinned: pinned.iter().map(|s| s.to_string()).collect(),
            total_memory,
            query: String::new(),
            query_changed: Instant::now(),
            sort: Sort::Likes,
            selected: 0,
            selection_changed: Instant::now(),
            note: None,
            ceiling_gbps: 0.0,
            pending_delete: None,
            local: Vec::new(),
            hub_results: Vec::new(),
            searches: HashMap::new(),
            in_flight: HashMap::new(),
            generation: 0,
            tx,
            rx,
            sizes: HashMap::new(),
            sizing: HashSet::new(),
            size_tx,
            size_rx,
            branches: HashMap::new(),
            branch_tx,
            branch_rx,
            quants: HashMap::new(),
            quanting: HashSet::new(),
            quant_tx,
            quant_rx,
            scan_tx,
            scan_rx,
            scanning_local: false,
            free_disk: None,
            rows: Vec::new(),
            total: 0,
            active: String::new(),
            headers: Headers::default(),
        }
    }

    /// Clears query and selection, rescans the local cache; selects `active_model` if listed.
    pub fn open(&mut self, active_model: &str) {
        self.active = active_model.to_string();
        self.query.clear();
        self.query_changed = Instant::now();
        self.note = None;
        self.pending_delete = None;
        self.hub_results.clear();
        // Failed searches and sizes get retried on the next open.
        self.searches.retain(|_, r| r.is_ok());
        self.sizes.retain(|_, r| r.is_ok());
        self.scanning_local = true;
        self.selected = 0;
        self.refilter();

        let scan_tx = self.scan_tx.clone();
        let hub = self.hub.clone();
        let total_mem = self.total_memory;
        std::thread::spawn(move || {
            crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
            let entries = scan_local_static(&hub, total_mem);
            let free = free_disk(&hub.cache_root);
            let _ = scan_tx.send((entries, free));
        });
    }
    /// Handles one key press while the picker is open.
    pub fn handle_key(&mut self, key: KeyEvent) -> PickerAction {
        if key.kind == KeyEventKind::Release {
            return PickerAction::None;
        }
        self.note = None;
        let pending = self.pending_delete.take();
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => return PickerAction::Close,
            KeyCode::Enter => return self.enter(),
            KeyCode::Delete => return self.delete(pending),
            KeyCode::Char('d') if ctrl => return self.delete(pending),
            KeyCode::Tab => {
                self.sort = self.sort.next();
                if let Some(Ok(entries)) = self.searches.get(&self.key()) {
                    self.hub_results = entries.clone();
                }
                self.refilter();
            }
            KeyCode::Up => self.move_by(-1),
            KeyCode::Down => self.move_by(1),
            KeyCode::Left => {
                if self.select_quant_relative(-1) {
                    return PickerAction::None;
                }
            }
            KeyCode::Right => {
                if self.select_quant_relative(1) {
                    return PickerAction::None;
                }
            }
            KeyCode::PageUp => self.move_by(-10),
            KeyCode::PageDown => self.move_by(10),
            KeyCode::Char('p') if ctrl => self.move_by(-1),
            KeyCode::Char('n') if ctrl => self.move_by(1),
            KeyCode::Char('u') if ctrl => self.set_query(String::new()),
            KeyCode::Char('w') if ctrl => {
                let kept = self.query.trim_end();
                let cut = kept.rfind(char::is_whitespace).map_or(0, |i| i + 1);
                let q = kept[..cut].to_string();
                self.set_query(q);
            }
            KeyCode::Char('<') | KeyCode::Char('>') => {
                if self.cycle_variant() {
                    return PickerAction::None;
                }
            }
            KeyCode::Char('v')
                if !ctrl
                    && self.query.is_empty()
                    && self
                        .rows
                        .get(self.selected)
                        .is_some_and(|r| r.entry.quants.len() > 1) =>
            {
                if self.cycle_variant() {
                    return PickerAction::None;
                }
            }
            KeyCode::Char('r') if ctrl => {
                let mut s = crate::tui::settings::load_settings();
                s.reencode = s.reencode.next();
                let _ = crate::tui::settings::save_settings(&s);
                return PickerAction::Reload("reencode");
            }
            KeyCode::Char('f') if ctrl => {
                let mut s = crate::tui::settings::load_settings();
                s.variant = s.variant.next();
                let _ = crate::tui::settings::save_settings(&s);
                self.refilter();
                return PickerAction::Reload("variant");
            }
            KeyCode::Char('o') if ctrl => {
                let mut s = crate::tui::settings::load_settings();
                s.hope = s.hope.next();
                let _ = crate::tui::settings::save_settings(&s);
                return PickerAction::Reload("hope");
            }
            KeyCode::Char(c) if !ctrl => {
                let mut q = self.query.clone();
                q.push(c);
                self.set_query(q);
            }
            KeyCode::Backspace => {
                let mut q = self.query.clone();
                q.pop();
                self.set_query(q);
            }
            _ => {}
        }
        PickerAction::None
    }

    /// Drains finished hub searches and size lookups; starts a search when the query has been
    /// unchanged for 250 ms, and an exact-size lookup for a row highlighted that long.
    /// Returns true when what `render` would draw changed.
    pub fn tick(&mut self) -> bool {
        let mut changed = false;
        while let Ok((key, generation, result)) = self.rx.try_recv() {
            if self.in_flight.get(&key) != Some(&generation) {
                continue;
            }
            self.in_flight.remove(&key);
            if key == self.key() {
                if let Ok(entries) = &result {
                    self.hub_results = entries.clone();
                }
                changed = true;
            }
            self.searches.insert(key, result);
        }
        if changed {
            self.refilter();
        }
        while let Ok((id, size)) = self.size_rx.try_recv() {
            self.sizing.remove(&id);
            self.sizes.insert(id, size);
            changed = true;
        }
        while let Ok((id, entries)) = self.branch_rx.try_recv() {
            self.branches.insert(id, entries);
            self.refilter();
            changed = true;
        }
        while let Ok((id, best)) = self.quant_rx.try_recv() {
            self.quanting.remove(&id);
            self.quants.insert(id, best);
            self.refilter();
            changed = true;
        }
        while let Ok((entries, free)) = self.scan_rx.try_recv() {
            self.local = entries;
            self.free_disk = free;
            self.scanning_local = false;
            self.refilter();
            if let Some(i) = self.rows.iter().position(|r| r.entry.id == self.active) {
                self.selected = i;
            }
            changed = true;
        }
        let need_quants = self
            .rows
            .get(self.selected)
            .map(|r| &r.entry)
            .filter(|e| {
                !e.cached
                    && e.id.contains('/')
                    && !e.id.contains('@')
                    && (e.quants.is_empty()
                        || e.blocker
                            .as_deref()
                            .is_some_and(|b| b.starts_with("needs ~")))
            })
            .map(|e| (e.id.clone(), e.params));
        if let Some((id, base_params)) = need_quants {
            let urgent = self.rows.get(self.selected).is_some_and(|r| {
                r.entry
                    .blocker
                    .as_deref()
                    .is_some_and(|b| b.starts_with("needs ~"))
            });
            if !self.quants.contains_key(&id)
                && !self.quanting.contains(&id)
                && (urgent || self.selection_changed.elapsed() >= DEBOUNCE)
            {
                self.quanting.insert(id.clone());
                let (tx, token, total, free) = (
                    self.quant_tx.clone(),
                    self.hf_token.clone(),
                    self.total_memory,
                    self.free_disk,
                );
                std::thread::spawn(move || {
                    let variants =
                        quantized_variants(&id, token.as_deref(), total, free, base_params);
                    let _ = tx.send((id, variants));
                });
            }
        }

        let key = self.key();
        if !key.0.is_empty()
            && !self.searches.contains_key(&key)
            && !self.in_flight.contains_key(&key)
            && self.query_changed.elapsed() >= DEBOUNCE
        {
            self.generation += 1;
            self.in_flight.insert(key.clone(), self.generation);
            let (tx, token, total_memory, generation) = (
                self.tx.clone(),
                self.hf_token.clone(),
                self.total_memory,
                self.generation,
            );
            std::thread::spawn(move || {
                let result = hub_search(&key.0, key.1, token.as_deref(), total_memory);
                let _ = tx.send((key, generation, result));
            });
        }

        let wanted = self
            .rows
            .get(self.selected)
            .map(|r| &r.entry)
            .filter(|e| !e.cached && e.id.contains('/'))
            .map(|e| e.id.clone());
        if let Some(id) = wanted {
            if !self.sizes.contains_key(&id)
                && !self.sizing.contains(&id)
                && self.selection_changed.elapsed() >= DEBOUNCE
            {
                self.sizing.insert(id.clone());
                let (tx, btx, token) = (
                    self.size_tx.clone(),
                    self.branch_tx.clone(),
                    self.hf_token.clone(),
                );
                let total_memory = self.total_memory;
                std::thread::spawn(move || {
                    let size = exact_download_size(&id, token.as_deref());
                    // Main without weights: offer each branch (e.g. one EXL3 bitrate per branch).
                    if size.is_err() && !id.contains('@') {
                        let rows = branch_models(&id, token.as_deref())
                            .and_then(|j| parse_hub_results(&j, total_memory));
                        if let Some(rows) = rows.ok().filter(|r| !r.is_empty()) {
                            let _ = btx.send((id.clone(), rows));
                        }
                    }
                    let _ = tx.send((id, size));
                });
            }
        }
        changed
    }

    /// Record that `id` is now in the local cache (after a successful load).
    pub fn mark_cached(&mut self, id: &str) {
        let size = Some(dir_size(&self.repo_dir(split_weight_file(id).0)));
        match self.local.iter_mut().find(|e| e.id == id) {
            Some(e) => (e.cached, e.size) = (true, size),
            None => self.local.push(Entry {
                id: id.to_string(),
                cached: true,
                size,
                ..Default::default()
            }),
        }
        self.refilter();
    }

    /// `<cache>/models--<org>--<repo>`: everything the hub stored for `id`; for `<id>@<revision>`
    /// only that revision's snapshot.
    pub fn repo_dir(&self, id: &str) -> std::path::PathBuf {
        repo_dir_path(&self.hub.cache_root, id)
    }

    /// Predicted decode tok/s of variant `q`, whether it comes from its weight header, and the
    /// share of the roofline that is (tok/s over the ceiling's tokens per second at the bytes
    /// streamed per token): the header is read in the background (local file, else Hub range
    /// request) the first time it is asked for; until then, and when it cannot be read, the
    /// estimate is size × bpw (the failure shows as `None`, like an unmeasured ceiling).
    fn predict(&self, q: &Entry) -> Option<(f64, bool, f64)> {
        self.predict_variant(q, true)
    }

    fn predict_variant(&self, q: &Entry, spawn_fetch: bool) -> Option<(f64, bool, f64)> {
        use crate::model::variant::{host_eff, predict_with, reencode_policy, size_header};
        let ceiling = f64::from(self.ceiling_gbps);
        if ceiling <= 0.0 {
            return None;
        }
        let state = {
            let mut m = self.headers.lock();
            let known = m.get(&q.id).cloned();
            if known.is_none() && spawn_fetch {
                m.insert(q.id.clone(), None);
                let (headers, hub, token, id, cached) = (
                    self.headers.clone(),
                    self.hub.clone(),
                    self.hf_token.clone(),
                    q.id.clone(),
                    q.cached,
                );
                std::thread::spawn(move || {
                    crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
                    let got = (|| {
                        let (repo, file) = crate::model::hub::split_weight_file(&id);
                        let file = file.ok_or("not a weights file")?;
                        let local = cached.then(|| hub.locate_model(repo).local_dir).flatten();
                        match local {
                            Some(dir) => crate::model::variant::read_local_header(&dir.join(file)),
                            None => crate::model::variant::fetch_remote_header(
                                repo,
                                file,
                                token.as_deref(),
                            ),
                        }
                    })();
                    headers.lock().insert(id, Some(got.map(Arc::new)));
                });
            }
            known.flatten()
        };
        let policy = reencode_policy();
        let (h, exact) = match state {
            Some(Ok(h)) => (h, true),
            Some(Err(_)) => return None,
            None => {
                let fmt = detect_quant_format(Path::new(&q.id))?;
                (Arc::new(size_header(fmt, q.size)), false)
            }
        };
        let tok_s = predict_with(&h, policy, ceiling, &host_eff);
        let pct = tok_s * h.streamed_bytes_per_token(policy) as f64 / (ceiling * 1e9) * 100.0;
        Some((tok_s, exact, pct))
    }
}

pub fn repo_dir_path(cache_root: &Path, id: &str) -> std::path::PathBuf {
    let (repo, rev) = split_revision(id);
    let dir = cache_root.join(format!("models--{}", repo.replace('/', "--")));
    match repo.len() < id.len() {
        true => dir.join("snapshots").join(rev.replace('/', "--")),
        false => dir,
    }
}

/// Orders variant entries best first under the persisted variant policy (S3.6). Only file size and
/// format are known for a listing, so this predicts from size × bpw; headers are read lazily for
/// the variant on screen ([`Picker::predict`]).
fn variant_blocker(q: &Entry, total_memory: Option<u64>, free_disk: Option<u64>) -> Option<String> {
    if let Some(b) = packed_memory_blocker(q.size, total_memory) {
        return Some(b);
    }
    if !q.cached {
        if let Some((size, free)) = q.size.zip(free_disk) {
            if size > free {
                return Some(format!(
                    "needs ~{} of disk; {} free",
                    fmt_bytes(size),
                    fmt_bytes(free)
                ));
            }
        }
    }
    None
}

fn variant_fits(q: &Entry, total_memory: Option<u64>, free_disk: Option<u64>) -> bool {
    variant_blocker(q, total_memory, free_disk).is_none()
}

fn sort_variants(v: &mut [Entry], total_memory: Option<u64>, free_disk: Option<u64>) {
    use crate::model::variant::{rank_variants, reencode_policy, variant_policy, Variant};
    for e in v.iter_mut() {
        e.blocker = variant_blocker(e, total_memory, free_disk);
    }
    let policy = reencode_policy();
    let mut ranked: Vec<Variant> = v
        .iter()
        .map(|e| {
            let resident = e.size.map_or(u64::MAX, |s| (s as f64 * 1.1) as u64);
            match detect_quant_format(Path::new(&e.id)) {
                Some(f) => {
                    let mut var = Variant::from_size(&e.id, f, e.size, policy, 1.0);
                    if e.size.is_some() {
                        var.resident_bytes = resident;
                    }
                    var
                }
                None => Variant {
                    name: e.id.clone(),
                    download_bytes: e.size.unwrap_or(u64::MAX),
                    resident_bytes: resident,
                    tok_s: 0.0,
                    bpw: 0.0,
                },
            }
        })
        .collect();
    let ram = total_memory.map_or(u64::MAX, |m| (m as f64 / 0.7).ceil() as u64);
    rank_variants(variant_policy(), ram, &mut ranked);
    v.sort_by_key(|e| {
        (
            !variant_fits(e, total_memory, free_disk),
            ranked.iter().position(|r| r.name == e.id),
        )
    });
}
impl Picker {
    /// `id` left the local cache: local-only rows vanish, hub rows lose `cached`.
    pub fn mark_deleted(&mut self, id: &str, freed: u64) {
        self.local.retain(|e| e.id != id);
        self.note = Some((
            format!("deleted {id} (freed {})", fmt_bytes(freed)),
            Token::Jade,
        ));
        self.free_disk = free_disk(&self.hub.cache_root);
        self.refilter();
    }

    /// Shows `text` in the picker status until the next key press.
    pub fn set_note(&mut self, text: String, tok: Token) {
        self.note = Some((text, tok));
    }

    /// Sets the memory-bandwidth ceiling (GB/s) the predicted tok/s derives from.
    pub fn set_ceiling_gbps(&mut self, gbps: f32) {
        self.ceiling_gbps = gbps;
    }

    /// Draws the picker as a centered overlay over `area`.
    pub fn render(&self, f: &mut Frame, area: Rect, active_model: &str, loading: Option<&str>) {
        if area.width < 20 || area.height < 7 {
            return;
        }
        let w = (area.width - area.width / 8).max(20).min(area.width);
        let h = (area.height - area.height / 5).max(7).min(area.height);
        let popup = Rect::new(
            area.x + (area.width - w) / 2,
            area.y + (area.height - h) / 2,
            w,
            h,
        );
        f.render_widget(Clear, popup);
        let th = theme();
        let block = instrument(
            &format!("Load model · sort: {} (tab)", self.sort.label()),
            None,
            true,
            &th,
        );
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let width = inner.width as usize;
        let list_h = inner.height.saturating_sub(3) as usize;
        let dim = th.fg(Token::Dim);
        let accent = th.fg(Token::Brass);
        let mut lines: Vec<Line> = Vec::with_capacity(inner.height as usize);

        let (status, tok) = match (&self.note, loading) {
            (None, Some(id)) => (format!("loading {id}…"), Token::Gold),
            _ => self.status(),
        };
        lines.push(spread(
            vec![
                Span::styled("› ", accent),
                Span::raw(self.query.clone()),
                Span::styled("▏", accent),
            ],
            vec![Span::styled(status, th.fg(tok))],
            width,
        ));

        // Stateless scroll: keep the selection and its inline quant sub-rows on screen.
        const MAX_INLINE_QUANTS: usize = 4;
        let selected_entry = self.rows.get(self.selected).map(|r| &r.entry);
        let sub_cap = list_h.saturating_sub(2).min(MAX_INLINE_QUANTS);
        let sub_count = selected_entry.map_or(0, |e| {
            if e.quants.is_empty() {
                0
            } else {
                e.quants.len().min(sub_cap)
            }
        });
        let model_rows_h = if sub_count > 0 {
            list_h.saturating_sub(1 + sub_count).max(1)
        } else {
            list_h.max(1)
        };
        let offset = (self.selected + 1).saturating_sub(model_rows_h);
        let visible =
            &self.rows[offset.min(self.rows.len())..(offset + model_rows_h).min(self.rows.len())];
        let now = chrono::Utc::now();
        let sub_rows: Vec<(usize, &Entry, [String; 6], bool)> =
            match selected_entry.filter(|_| sub_count > 0) {
                Some(e) => {
                    let q_start = e
                        .quant_idx
                        .saturating_sub(sub_count / 2)
                        .min(e.quants.len().saturating_sub(sub_count));
                    (q_start..q_start + sub_count)
                        .map(|qi| {
                            let q = &e.quants[qi];
                            let (q_cells, fits) =
                                self.quant_subrow_cells(e, q, qi, e.quants.len(), now);
                            (qi, q, q_cells, fits)
                        })
                        .collect()
                }
                None => Vec::new(),
            };
        let cells: Vec<[String; 6]> = visible
            .iter()
            .map(|r| {
                let e = &r.entry;
                [
                    e.params.map(human).unwrap_or_default(),
                    self.size_cell(e),
                    e.likes.map(human).unwrap_or_default(),
                    e.downloads.map(human).unwrap_or_default(),
                    e.updated
                        .as_deref()
                        .map(|u| relative(u, now))
                        .unwrap_or_default(),
                    tail(e).0,
                ]
            })
            .collect();
        // (header, right-aligned); dropped right to left by priority when the id gets too narrow.
        const HEADS: [(&str, bool); 6] = [
            ("params", true),
            ("size", true),
            ("♥", true),
            ("↓", true),
            ("updated", true),
            ("status", false),
        ];
        const DROP_ORDER: [usize; 5] = [2, 4, 3, 0, 1];
        let sorted = self.sort.column();
        let col_w = |c: usize| {
            let data = cells
                .iter()
                .chain(sub_rows.iter().map(|(_, _, r, _)| r))
                .map(|r| r[c].chars().count())
                .max()
                .unwrap_or(0);
            let w = data.max(HEADS[c].0.chars().count());
            if c == 5 {
                w.min(width / 3)
            } else {
                w
            }
        };
        let mut shown = [true; 6];
        let used = |shown: &[bool; 6]| -> usize {
            (0..6).filter(|&c| shown[c]).map(|c| col_w(c) + 2).sum()
        };
        for c in DROP_ORDER.into_iter().filter(|&c| c != sorted) {
            if width.saturating_sub(2 + used(&shown)) >= 24 {
                break;
            }
            shown[c] = false;
        }
        let id_w = width.saturating_sub(2 + used(&shown)).max(4);
        let columns = |row: &[String], style: &dyn Fn(usize) -> Style| -> Vec<Span<'static>> {
            let mut spans = Vec::new();
            for c in (0..6).filter(|&c| shown[c]) {
                let w = col_w(c);
                let text = truncate(&row[c], w);
                let text = if HEADS[c].1 {
                    format!("  {text:>w$}")
                } else {
                    format!("  {text:<w$}")
                };
                spans.push(Span::styled(text, style(c)));
            }
            spans
        };

        let head: Vec<String> = HEADS.iter().map(|h| h.0.to_string()).collect();
        let mut header = vec![Span::styled(format!("  {:<id_w$}", "model"), dim)];
        let hot = accent.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
        header.extend(columns(&head, &|c| if c == sorted { hot } else { dim }));
        lines.push(Line::from(header));

        for (i, (row, cells)) in visible.iter().zip(&cells).enumerate() {
            let e = &row.entry;
            let mut spans = vec![if e.id == active_model {
                Span::styled("● ", th.fg(Token::Jade))
            } else {
                Span::raw("  ")
            }];
            let id_len = e.id.chars().count();
            let keep = if id_len > id_w { id_w - 1 } else { id_len };
            for (ci, c) in e.id.chars().take(keep).enumerate() {
                let style = if row.hits.binary_search(&ci).is_ok() {
                    accent.add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                spans.push(Span::styled(c.to_string(), style));
            }
            if keep < id_len {
                spans.push(Span::raw("…"));
            } else {
                spans.push(Span::raw(" ".repeat(id_w - keep)));
            }
            let status_color = tail(e).1;
            spans.extend(columns(cells, &|c| match c {
                5 => th.fg(status_color),
                1 if !e.cached && !self.sizes.get(&e.id).is_some_and(|s| s.is_ok()) => dim,
                0 | 1 => Style::default(),
                c if c == sorted => accent,
                _ => dim,
            }));

            let mut line = spread(spans, Vec::new(), width);
            if offset + i == self.selected {
                line = line.style(th.selected());
            }
            lines.push(line);
            if offset + i == self.selected && !sub_rows.is_empty() {
                let total_model_w = 2 + id_w;
                let repo_prefix = format!("{}/", e.id);
                for (sub_i, &(qi, q, ref q_cells, fits)) in sub_rows.iter().enumerate() {
                    let branch = if sub_i + 1 == sub_rows.len() {
                        "  └─ "
                    } else {
                        "  ├─ "
                    };
                    let is_active = qi == e.quant_idx;
                    let is_best = qi == 0;
                    let marker = match (is_active, is_best) {
                        (true, true) => "▸★",
                        (true, false) => "▸ ",
                        (false, true) => " ★",
                        (false, false) => "  ",
                    };
                    let q_tag = detect_quant_format(Path::new(&q.id))
                        .map_or_else(|| "quant".to_string(), |f| f.name().to_string());
                    let badge = format!("[{q_tag}] ");
                    let raw_label = q.id.strip_prefix(&repo_prefix).unwrap_or(&q.id);
                    let prefix_w =
                        branch.chars().count() + marker.chars().count() + 1 + badge.chars().count();
                    let active_style = accent.add_modifier(Modifier::BOLD);
                    let mut sub_spans = if prefix_w <= total_model_w {
                        let rem_w = total_model_w - prefix_w;
                        let trunc = middle_truncate(raw_label, rem_w);
                        let pad = rem_w.saturating_sub(trunc.chars().count());
                        let label_padded = format!("{trunc}{}", " ".repeat(pad));
                        let marker_style = if is_active {
                            active_style
                        } else if is_best {
                            th.fg(Token::Gold)
                        } else {
                            dim
                        };
                        let label_style = if is_active { active_style } else { dim };
                        vec![
                            Span::styled(branch, dim),
                            Span::styled(format!("{marker} {badge}"), marker_style),
                            Span::styled(label_padded, label_style),
                        ]
                    } else {
                        vec![Span::styled(
                            truncate(&format!("{branch}{marker} {badge}"), total_model_w),
                            if is_active { active_style } else { dim },
                        )]
                    };
                    let status_style = if is_active {
                        th.fg(if fits { Token::Gold } else { Token::Crimson })
                            .add_modifier(Modifier::BOLD)
                    } else {
                        th.fg(if fits { Token::Jade } else { Token::Crimson })
                    };
                    sub_spans.extend(columns(q_cells, &|c| match c {
                        5 => status_style,
                        _ if is_active => accent,
                        _ => dim,
                    }));
                    lines.push(spread(sub_spans, Vec::new(), width));
                }
            }
        }
        if self.rows.is_empty() && list_h > 0 {
            let q = self.query.trim();
            let hint = if q.contains('/') {
                format!("  enter loads {q} from the hub")
            } else {
                "  no matches".to_string()
            };
            lines.push(Line::from(Span::styled(hint, dim)));
        }
        while lines.len() < 1 + list_h {
            lines.push(Line::default());
        }
        let settings = crate::tui::settings::load_settings();
        if let Some(e) = selected_entry.filter(|e| !e.quants.is_empty()) {
            let q_active = e.quants.get(e.quant_idx);
            let sz_gb = q_active
                .and_then(|q| q.size)
                .map(|s| s as f64 / 1e9)
                .unwrap_or(e.params.map(|p| p as f64 * 0.6 / 1e9).unwrap_or(2.0));
            let fits = match q_active.filter(|q| q.size.is_some()) {
                Some(q) => variant_fits(q, self.total_memory, self.free_disk),
                None => self
                    .total_memory
                    .map(|m| (sz_gb * 1e9) as u64 <= m)
                    .unwrap_or(true),
            };
            let fit_tag = if fits { "fit" } else { "no-fit" };
            // Predicted decode speed (S3.6) and its share of the roofline: from the weight header
            // once read, `~` while it is being read (size × bpw), `?` when it cannot be read or
            // the ceiling is unmeasured.
            let pred_tok_s = match q_active.and_then(|q| self.predict(q)) {
                Some((t, true, pct)) => format!("{:.0} tok/s {pct:.0}%", t.min(999.0)),
                Some((t, false, pct)) => format!("~{:.0} tok/s ~{pct:.0}%", t.min(999.0)),
                None => "? tok/s".to_string(),
            };
            let pred_text = format!(
                " · {pred_tok_s} {sz_gb:.1}GB [{fit_tag}] ({}/{})",
                e.quant_idx + 1,
                e.quants.len()
            );
            let right_hint_str = format!(
                "(★ best for {} · ←/→ choose · enter load)",
                settings.variant
            );
            let formatted_tags: Vec<String> = e
                .quants
                .iter()
                .enumerate()
                .map(|(i, q_entry)| {
                    let q_fmt = detect_quant_format(Path::new(&q_entry.id));
                    let tag = q_fmt.map_or_else(
                        || {
                            q_entry
                                .id
                                .rsplit('/')
                                .next()
                                .unwrap_or(&q_entry.id)
                                .to_string()
                        },
                        |q| q.name().to_string(),
                    );
                    let tag = if i == 0 { format!("★{tag}") } else { tag };
                    if i == e.quant_idx {
                        format!("[{tag}]")
                    } else {
                        tag
                    }
                })
                .collect();
            let active_idx = e.quant_idx.min(formatted_tags.len().saturating_sub(1));
            let base_w = "Quant: ".chars().count() + pred_text.chars().count();
            let hint_w = right_hint_str.chars().count();
            let min_tags_w = formatted_tags[active_idx].chars().count()
                + if active_idx > 0 { 3 } else { 0 }
                + if active_idx + 1 < formatted_tags.len() {
                    3
                } else {
                    0
                };
            let show_hint = base_w + min_tags_w + 1 + hint_w <= width;
            let budget = width.saturating_sub(base_w + if show_hint { 1 + hint_w } else { 0 });
            let window_width = |lo: usize, hi: usize| -> usize {
                let left_ell = if lo > 0 { 3 } else { 0 };
                let right_ell = if hi + 1 < formatted_tags.len() { 3 } else { 0 };
                let tags_w: usize = formatted_tags[lo..=hi]
                    .iter()
                    .map(|t| t.chars().count())
                    .sum::<usize>()
                    + (hi - lo) * 2;
                left_ell + tags_w + right_ell
            };
            let (mut lo, mut hi) = (active_idx, active_idx);
            loop {
                let mut grew = false;
                if hi + 1 < formatted_tags.len() && window_width(lo, hi + 1) <= budget {
                    hi += 1;
                    grew = true;
                }
                if lo > 0 && window_width(lo - 1, hi) <= budget {
                    lo -= 1;
                    grew = true;
                }
                if !grew {
                    break;
                }
            }
            let mut quant_spans: Vec<Span> = Vec::new();
            quant_spans.push(Span::styled(
                "Quant: ",
                th.fg(Token::Gold).add_modifier(Modifier::BOLD),
            ));
            if lo > 0 {
                quant_spans.push(Span::styled("‹… ", dim));
            }
            for i in lo..=hi {
                if i > lo {
                    quant_spans.push(Span::raw("  "));
                }
                if i == active_idx {
                    quant_spans.push(Span::styled(
                        formatted_tags[i].clone(),
                        th.fg(Token::Brass).add_modifier(Modifier::BOLD),
                    ));
                } else {
                    quant_spans.push(Span::styled(formatted_tags[i].clone(), dim));
                }
            }
            if hi + 1 < formatted_tags.len() {
                quant_spans.push(Span::styled(" …›", dim));
            }
            quant_spans.push(Span::styled(
                pred_text,
                th.fg(if fits { Token::Gold } else { Token::Crimson }),
            ));
            let right_hint = if show_hint {
                vec![Span::styled(right_hint_str, dim)]
            } else {
                Vec::new()
            };
            lines.push(spread(quant_spans, right_hint, width));
        } else {
            lines.push(Line::default());
        }
        let free = self
            .free_disk
            .map(|b| format!(" · {} free", fmt_bytes(b)))
            .unwrap_or_default();
        let s = &settings;
        let policy_str = format!(
            " · reencode: {} (^r) · variant: {} (^f) · hope: {} (^o)",
            s.reencode, s.variant, s.hope
        );
        lines.push(spread(
            vec![
                Span::styled(
                    "↑↓ move · enter load · tab sort · ctrl-d delete · esc close",
                    dim,
                ),
                Span::styled(policy_str, th.fg(Token::Gold)),
            ],
            vec![Span::styled(
                format!("{} of {}{free}", self.rows.len(), self.total),
                dim,
            )],
            width,
        ));
        f.render_widget(Paragraph::new(lines).style(th.fg(Token::Parchment)), inner);
    }

    /// Size column: on-disk bytes for cached rows, the exact download once looked up, else `~estimate`.
    fn size_cell(&self, e: &Entry) -> String {
        if e.cached {
            return e.size.map(fmt_bytes).unwrap_or_default();
        }
        match self.sizes.get(&e.id) {
            Some(Ok(b)) => fmt_bytes(*b),
            _ => e
                .size
                .map(|b| format!("~{}", fmt_bytes(b)))
                .unwrap_or_default(),
        }
    }

    fn quant_subrow_cells(
        &self,
        parent: &Entry,
        q: &Entry,
        qi: usize,
        total_quants: usize,
        now: chrono::DateTime<chrono::Utc>,
    ) -> ([String; 6], bool) {
        let is_in_repo = q.id.starts_with(&format!("{}/", parent.id));
        let est_bytes = q
            .params
            .or(parent.params)
            .map_or(2_000_000_000, |p| (p as f64 * 0.6) as u64);
        let fits = match q.size {
            Some(_) => variant_fits(q, self.total_memory, self.free_disk),
            None => self.total_memory.map_or(true, |m| est_bytes <= m),
        };
        let pred_tok_s = match self.predict_variant(q, qi == parent.quant_idx) {
            Some((t, true, pct)) => format!("{:.0} tok/s {pct:.0}%", t.min(999.0)),
            Some((t, false, pct)) => format!("~{:.0} tok/s ~{pct:.0}%", t.min(999.0)),
            None => "? tok/s".to_string(),
        };
        let cells = [
            q.params.or(parent.params).map(human).unwrap_or_default(),
            self.size_cell(q),
            q.likes
                .or_else(|| is_in_repo.then_some(parent.likes).flatten())
                .map(human)
                .unwrap_or_default(),
            q.downloads
                .or_else(|| is_in_repo.then_some(parent.downloads).flatten())
                .map(human)
                .unwrap_or_default(),
            q.updated
                .as_deref()
                .or_else(|| is_in_repo.then_some(parent.updated.as_deref()).flatten())
                .map(|u| relative(u, now))
                .unwrap_or_default(),
            format!(
                "{pred_tok_s} [{}] ({}/{})",
                if fits { "fit" } else { "no-fit" },
                qi + 1,
                total_quants
            ),
        ];
        (cells, fits)
    }

    fn key(&self) -> Key {
        (self.query.trim().to_string(), self.sort)
    }

    fn status(&self) -> (String, Token) {
        if let Some((note, tok)) = &self.note {
            return (note.clone(), *tok);
        }
        if self.scanning_local {
            return ("scanning cache…".into(), Token::Gold);
        }
        let key = self.key();
        if key.0.is_empty() {
            return ("type to search huggingface.co".into(), Token::Dim);
        }
        match self.searches.get(&key) {
            Some(Ok(v)) => (format!("hub: {} results", v.len()), Token::Dim),
            Some(Err(e)) => (format!("hub error: {e}"), Token::Crimson),
            None => ("searching hub…".into(), Token::Gold),
        }
    }

    fn enter(&mut self) -> PickerAction {
        match self.rows.get(self.selected) {
            Some(row) => {
                let chosen = row
                    .entry
                    .via
                    .as_deref()
                    .or_else(|| row.entry.quants.get(row.entry.quant_idx));
                if let Some(v) = chosen {
                    if row
                        .entry
                        .blocker
                        .as_deref()
                        .map_or(true, |b| b.starts_with("needs ~"))
                    {
                        return match variant_blocker(v, self.total_memory, self.free_disk) {
                            Some(b) => {
                                self.note = Some((b, Token::Crimson));
                                PickerAction::None
                            }
                            None => PickerAction::Load(v.id.clone()),
                        };
                    }
                }
                match &row.entry.blocker {
                    Some(b) => {
                        self.note = Some((b.clone(), Token::Crimson));
                        PickerAction::None
                    }
                    None if row.entry.draft_for.is_some() => {
                        PickerAction::AttachDraft(row.entry.id.clone())
                    }
                    None => PickerAction::Load(
                        row.entry
                            .via
                            .as_ref()
                            .map_or(&row.entry.id, |v| &v.id)
                            .clone(),
                    ),
                }
            }
            None => {
                let q = self.query.trim();
                if q.contains('/') {
                    PickerAction::Load(q.to_string())
                } else {
                    PickerAction::None
                }
            }
        }
    }

    /// First press arms (status asks to confirm); a second press on the same row within 3 s deletes.
    fn delete(&mut self, pending: Option<(String, Instant)>) -> PickerAction {
        let Some(e) = self.rows.get(self.selected).map(|r| &r.entry) else {
            return PickerAction::None;
        };
        if !e.cached {
            self.note = Some((
                format!("{} is not in the local cache", e.id),
                Token::Crimson,
            ));
            return PickerAction::None;
        }
        if pending.is_some_and(|(id, at)| id == e.id && at.elapsed() < DELETE_CONFIRM) {
            return PickerAction::Delete(e.id.clone());
        }
        let size = e.size.map(fmt_bytes).unwrap_or_else(|| "?".into());
        self.note = Some((
            format!("press ctrl-d again to delete {} ({size})", e.id),
            Token::Gold,
        ));
        self.pending_delete = Some((e.id.clone(), Instant::now()));
        PickerAction::None
    }

    fn move_by(&mut self, delta: isize) {
        let last = self.rows.len().saturating_sub(1) as isize;
        self.select((self.selected as isize + delta).clamp(0, last) as usize);
    }

    fn select(&mut self, i: usize) {
        if i != self.selected {
            self.selected = i;
            self.selection_changed = Instant::now();
        }
    }

    fn set_query(&mut self, query: String) {
        if query == self.query {
            return;
        }
        self.query = query;
        self.query_changed = Instant::now();
        self.select(0);
        let key = self.key();
        if key.0.is_empty() {
            self.hub_results.clear();
        } else if let Some(Ok(entries)) = self.searches.get(&key) {
            self.hub_results = entries.clone();
        }
        self.refilter();
    }
}

fn scan_local_static(hub: &HuggingFaceHub, total_memory: Option<u64>) -> Vec<Entry> {
    let mut local = Vec::new();
    let Ok(dir) = std::fs::read_dir(&hub.cache_root) else {
        return local;
    };
    for entry in dir.flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let Some(repo) = name.strip_prefix("models--") else {
            continue;
        };
        // Branch snapshots (`snapshots/<rev>`, not `main` or a commit hash) are `<id>@<rev>` rows.
        let snaps = entry.path().join("snapshots");
        let revs: Vec<String> = std::fs::read_dir(&snaps)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|r| r != "main" && !(r.len() == 40 && r.bytes().all(|b| b.is_ascii_hexdigit())))
            .collect();
        let base = repo.replacen("--", "/", 1);
        let plain = (revs.is_empty() || snaps.join("main").is_dir()).then(|| base.clone());
        for id in plain
            .into_iter()
            .chain(revs.iter().map(|r| format!("{base}@{r}")))
        {
            let art = hub.locate_model(&id);
            let classifier = art
                .config_path
                .as_deref()
                .and_then(crate::model::classifier::architecture_in)
                .is_some_and(|a| crate::model::classifier::runs_architecture(&a));
            let cfg: Option<Value> = art
                .config_path
                .as_deref()
                .and_then(|p| serde_json::from_slice(&std::fs::read(p).ok()?).ok());
            let gguf_arch = art
                .weights_path
                .as_deref()
                .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
                .and_then(crate::model::loaders::gguf::architecture);
            let draft_for = draft_target(
                &id,
                cfg.as_ref().unwrap_or(&Value::Null),
                &[],
                gguf_arch.as_deref(),
            );
            let system1 = cfg
                .as_ref()
                .is_some_and(crate::runtime::encoder::system1::is_sentence_encoder_config);
            let runnable =
                super::is_chat_model(&id) || classifier || system1 || draft_for.is_some();
            if !runnable || art.weights_path.is_none() {
                continue;
            }
            let params = Some(art.estimated_params as u64).filter(|&p| p > 0);
            let mut quants: Vec<Entry> = Vec::new();
            if let Some(target_dir) = &art.local_dir {
                let local_files = crate::model::hub::list_local_files(target_dir);
                for f in &local_files {
                    let p = &f.path;
                    let name = p.rsplit('/').next().unwrap_or(p).to_lowercase();
                    if p.ends_with(".gguf")
                        && !name.contains("mmproj")
                        && !name.starts_with("mtp-")
                        && !name.starts_with("imatrix")
                        && !name.ends_with(".imatrix.gguf")
                        && (!p.contains("-of-") || p.contains("-00001-of-"))
                    {
                        if detect_quant_format(Path::new(p)).is_some() {
                            quants.push(Entry {
                                id: format!("{id}/{p}"),
                                params,
                                size: f.size,
                                cached: true,
                                ..Default::default()
                            });
                        }
                    }
                }
                sort_variants(&mut quants, total_memory, None);
            }
            let via = quants.first().cloned().map(Box::new);
            let packed_gguf = gguf_arch
                .as_deref()
                .is_some_and(crate::arch::dispatch::runs_spec_lm_gguf);
            let variants_stay_packed = !quants.is_empty()
                && gguf_arch
                    .as_deref()
                    .map_or(true, crate::arch::dispatch::runs_spec_lm_gguf);
            let blocker = if variants_stay_packed {
                if quants.iter().any(|q| variant_fits(q, total_memory, None)) {
                    None
                } else {
                    quants
                        .first()
                        .and_then(|q| q.blocker.clone())
                        .or_else(|| memory_blocker(params, total_memory))
                }
            } else if packed_gguf {
                packed_memory_blocker(
                    art.weights_path
                        .as_deref()
                        .and_then(|p| std::fs::metadata(p).ok())
                        .map(|m| m.len()),
                    total_memory,
                )
            } else {
                memory_blocker(params, total_memory)
            };
            local.push(Entry {
                size: Some(dir_size(&repo_dir_path(&hub.cache_root, &id))),
                id,
                params,
                cached: true,
                blocker,
                via,
                quants,
                quant_idx: 0,
                ..Default::default()
            });
        }
    }
    local
}

impl Picker {
    /// Merges local, pinned, and hub entries (deduped by id), filters by the query, and orders by
    /// the active sort key, then cached first, then id.
    fn refilter(&mut self) {
        let mut all = self.local.clone();
        for id in &self.pinned {
            if !all.iter().any(|e| &e.id == id) {
                all.push(Entry {
                    id: id.clone(),
                    ..Default::default()
                });
            }
        }
        for h in self
            .hub_results
            .iter()
            .chain(self.branches.values().flatten())
        {
            match all.iter_mut().find(|e| e.id == h.id) {
                Some(e) => {
                    // The hub's count is exact; the local one is estimated from config.json.
                    e.params = h.params.or(e.params);
                    (e.downloads, e.likes, e.updated, e.gated) =
                        (h.downloads, h.likes, h.updated.clone(), h.gated);
                    if !e.cached {
                        (e.blocker, e.size) = (h.blocker.clone(), h.size);
                    }
                    for q in &mut e.quants {
                        if q.downloads.is_none() {
                            q.downloads = h.downloads;
                            q.likes = h.likes;
                            q.updated = h.updated.clone();
                        }
                    }
                    let prev_len = e.quants.len();
                    for hq in &h.quants {
                        if !e.quants.iter().any(|q| q.id == hq.id) {
                            e.quants.push(hq.clone());
                        }
                    }
                    if e.quants.len() > prev_len {
                        sort_variants(&mut e.quants, self.total_memory, self.free_disk);
                        e.quant_idx = e.quant_idx.min(e.quants.len().saturating_sub(1));
                        e.via = e.quants.get(e.quant_idx).cloned().map(Box::new);
                    }
                }
                None => all.push(h.clone()),
            }
        }
        for e in all.iter_mut().filter(|e| e.blocker.is_some()) {
            if let Some(b) = self.branches.get(&e.id) {
                let revs: Vec<&str> = b.iter().map(|b| &b.id[e.id.len()..]).collect();
                e.blocker = Some(format!("main has no weights; branches: {}", revs.join(" ")));
            }
        }
        for e in all.iter_mut().filter(|e| !e.cached) {
            for q in &mut e.quants {
                q.blocker = variant_blocker(q, self.total_memory, self.free_disk);
            }
            let any_in_repo_fits = e
                .quants
                .iter()
                .any(|q| variant_fits(q, self.total_memory, self.free_disk));
            if e.quants.is_empty() {
                if let (None, Some((size, free))) = (&e.blocker, e.size.zip(self.free_disk)) {
                    if size > free {
                        e.blocker = Some(format!(
                            "needs ~{} of disk; {} free",
                            fmt_bytes(size),
                            fmt_bytes(free)
                        ));
                    }
                }
            }
            if e.quants.is_empty() || !any_in_repo_fits {
                if let Some(variants) = self.quants.get(&e.id) {
                    let mut refreshed = variants.clone();
                    for q in &mut refreshed {
                        q.blocker = variant_blocker(q, self.total_memory, self.free_disk);
                    }
                    if refreshed
                        .iter()
                        .any(|q| variant_fits(q, self.total_memory, self.free_disk))
                    {
                        e.quants = refreshed;
                        e.quant_idx = 0;
                    } else if e.quants.is_empty() && refreshed.is_empty() {
                        if let Some(b) = &mut e.blocker {
                            if b.starts_with("needs ~")
                                && !b.ends_with("no loadable quantized variant")
                            {
                                b.push_str("; the hub lists no loadable quantized variant");
                            }
                        }
                    }
                }
            }
            if !e.quants.is_empty() {
                if let Some(first_fit) = e
                    .quants
                    .iter()
                    .position(|q| variant_fits(q, self.total_memory, self.free_disk))
                {
                    let idx = e.quant_idx.min(e.quants.len() - 1);
                    let active = if variant_fits(&e.quants[idx], self.total_memory, self.free_disk)
                    {
                        idx
                    } else {
                        first_fit
                    };
                    e.quant_idx = active;
                    e.via = Some(Box::new(e.quants[active].clone()));
                    if e.blocker
                        .as_deref()
                        .is_some_and(|b| b.starts_with("needs ~"))
                    {
                        e.blocker = None;
                    }
                } else {
                    let idx = e.quant_idx.min(e.quants.len() - 1);
                    e.quant_idx = idx;
                    e.via = Some(Box::new(e.quants[idx].clone()));
                    if e.blocker.is_none() {
                        e.blocker = e.quants[idx].blocker.clone();
                    }
                }
            }
        }
        self.total = all.len();

        let q = self.query.trim();
        let mut rows: Vec<Row> = all
            .into_iter()
            .filter_map(|mut entry| {
                let hits = fuzzy_hits(q, &entry.id)?;
                if let Some(t) = entry.draft_for.as_deref() {
                    if entry.blocker.is_none() && !is_draft_for(t, &self.active) {
                        entry.blocker = Some(format!("draft model for {t}: load {t} first"));
                    }
                }
                Some(Row { entry, hits })
            })
            .collect();
        let sort = self.sort;
        rows.sort_by(|a, b| {
            let (a, b) = (&a.entry, &b.entry);
            match sort {
                Sort::Recent => b.updated.cmp(&a.updated),
                Sort::Downloads => b.downloads.cmp(&a.downloads),
                Sort::Likes => b.likes.cmp(&a.likes),
            }
            .then(b.cached.cmp(&a.cached))
            .then(a.id.cmp(&b.id))
        });
        self.rows = rows;
        self.selected = self.selected.min(self.rows.len().saturating_sub(1));
    }

    /// Moves selection relative to active quantization level (wrapping around available variants).
    pub fn select_quant_relative(&mut self, delta: isize) -> bool {
        let Some(row) = self.rows.get_mut(self.selected) else {
            return false;
        };
        let e = &mut row.entry;
        if e.quants.len() <= 1 {
            return false;
        }
        let len = e.quants.len() as isize;
        let new_idx = ((e.quant_idx as isize + delta).rem_euclid(len)) as usize;
        e.quant_idx = new_idx;
        e.via = Some(Box::new(e.quants[new_idx].clone()));
        true
    }

    /// Cycles the active quantized derivative for the currently selected model.
    pub fn cycle_variant(&mut self) -> bool {
        self.select_quant_relative(1)
    }
}

/// Status column: the blocker (red), a drafter's target (green), `cached` (green), `gated`
/// (yellow), or nothing.
fn tail(e: &Entry) -> (String, Token) {
    match (&e.blocker, &e.draft_for) {
        (Some(b), _) => (b.clone(), Token::Crimson),
        (None, _) if e.via.is_some() || !e.quants.is_empty() => {
            let v = e.via.as_deref().unwrap_or(e);
            let size = v
                .size
                .map(|s| format!(": {}", fmt_bytes(s)))
                .unwrap_or_default();
            let is_in_repo = v.id.starts_with(&format!("{}/", e.id));
            let cycle_note = if e.quants.len() > 1 {
                format!(
                    " (variant {} of {}{size}, 'v' to cycle)",
                    e.quant_idx + 1,
                    e.quants.len()
                )
            } else {
                v.size
                    .map(|s| format!(" ({})", fmt_bytes(s)))
                    .unwrap_or_default()
            };
            let q_tag = detect_quant_format(Path::new(&v.id))
                .map(|q| format!(" [{}]", q.name()))
                .unwrap_or_default();
            let msg = if is_in_repo {
                let file_part = v.id.rsplit('/').next().unwrap_or(&v.id);
                format!("loads {file_part}{q_tag}{cycle_note}")
            } else {
                format!("too large; loads quantized {}{q_tag}{cycle_note}", v.id)
            };
            (msg, Token::Jade)
        }
        (None, Some(t)) => (format!("draft model for {t}"), Token::Jade),
        _ if e.cached => ("cached".into(), Token::Jade),
        _ if e.gated => ("gated".into(), Token::Gold),
        _ => (String::new(), Token::Dim),
    }
}

/// `left`, padding, then `right` flush to `width` (right side clips when it does not fit).
fn spread<'a>(mut left: Vec<Span<'a>>, right: Vec<Span<'a>>, width: usize) -> Line<'a> {
    let used: usize = left.iter().chain(&right).map(Span::width).sum();
    left.push(Span::raw(" ".repeat(width.saturating_sub(used))));
    left.extend(right);
    Line::from(left)
}

fn truncate(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_string()
    } else if w == 0 {
        String::new()
    } else {
        s.chars().take(w - 1).chain(std::iter::once('…')).collect()
    }
}

fn middle_truncate(s: &str, max_w: usize) -> String {
    let len = s.chars().count();
    if len <= max_w {
        s.to_string()
    } else if max_w <= 1 {
        truncate(s, max_w)
    } else {
        let head = (max_w - 1) / 2;
        let tail = max_w - 1 - head;
        let mut out: String = s.chars().take(head).collect();
        out.push('…');
        out.extend(s.chars().skip(len - tail));
        out
    }
}

/// `135M`, `1.5B`, `70B`, `1.8M`.
fn human(n: u64) -> String {
    let (v, unit) = match n {
        n if n >= 1_000_000_000 => (n as f64 / 1e9, "B"),
        n if n >= 1_000_000 => (n as f64 / 1e6, "M"),
        n if n >= 1_000 => (n as f64 / 1e3, "K"),
        n => return n.to_string(),
    };
    if v < 10.0 {
        format!("{v:.1}{unit}")
    } else {
        format!("{v:.0}{unit}")
    }
}

/// `796 MB`, `1.2 GB`, `12 KB` (decimal units, like the Hub).
pub(super) fn fmt_bytes(b: u64) -> String {
    let (v, unit) = match b {
        b if b >= 1_000_000_000 => (b as f64 / 1e9, "GB"),
        b if b >= 1_000_000 => (b as f64 / 1e6, "MB"),
        b if b >= 1_000 => (b as f64 / 1e3, "KB"),
        b => return format!("{b} B"),
    };
    if v < 10.0 {
        format!("{v:.1} {unit}")
    } else {
        format!("{v:.0} {unit}")
    }
}

/// `45m`, `3h`, `5d`, else the date (`2025-11-03`).
fn relative(rfc3339: &str, now: chrono::DateTime<chrono::Utc>) -> String {
    let Ok(t) = chrono::DateTime::parse_from_rfc3339(rfc3339) else {
        return String::new();
    };
    let age = now.signed_duration_since(t);
    match age.num_minutes() {
        m if m < 60 => format!("{}m", m.max(0)),
        m if m < 24 * 60 => format!("{}h", m / 60),
        m if m < 30 * 24 * 60 => format!("{}d", m / (24 * 60)),
        _ => t.format("%Y-%m-%d").to_string(),
    }
}

/// Bytes under `path` (a symlinked `path` is resolved; symlinks inside are not followed).
pub(super) fn dir_size(path: &Path) -> u64 {
    fn tree(path: &Path) -> u64 {
        let Ok(meta) = std::fs::symlink_metadata(path) else {
            return 0;
        };
        if !meta.is_dir() {
            return meta.len();
        }
        std::fs::read_dir(path)
            .map(|rd| rd.flatten().map(|e| tree(&e.path())).sum())
            .unwrap_or(0)
    }
    tree(&std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// Available bytes on the filesystem holding `dir` (`df -k`).
fn free_disk(dir: &Path) -> Option<u64> {
    let out = Command::new("df").arg("-k").arg(dir).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = text
        .lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

/// Bytes a load of `id` downloads: the selected weight files plus the listed metadata files.
fn exact_download_size(id: &str, token: Option<&str>) -> Result<u64, String> {
    let files = list_repo_files(id, token)?;
    let weights = select_weight_files(&files, None)?;
    Ok(files
        .iter()
        .filter(|f| is_metadata_file(&f.path) || weights.contains(&f.path))
        .filter_map(|f| f.size)
        .sum())
}

/// The query, then (for few hits) variants with one term shortened, merged by id; every
/// response is parsed into entries with loadability.
fn hub_search(
    query: &str,
    sort: Sort,
    token: Option<&str>,
    total_memory: Option<u64>,
) -> Result<Vec<Entry>, String> {
    let run = |q: &str, sort: Sort| {
        search_hub(q, sort, token).and_then(|body| parse_hub_results(&body, total_memory))
    };
    let mut entries = run(query, sort)?;
    if entries.len() < FEW_HITS {
        let variants = widen(query);
        // Shortened terms match many fresh fine-tunes; by downloads the well-known repos stay in
        // the capped pool. Rows are re-sorted client-side by the active key.
        let more: Vec<Result<Vec<Entry>, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = variants
                .iter()
                .map(|v| s.spawn(|| run(v, Sort::Downloads)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("search panicked".into())))
                .collect()
        });
        for e in more.into_iter().flatten().flatten() {
            if !entries.iter().any(|x| x.id == e.id) {
                entries.push(e);
            }
        }
    }
    Ok(entries)
}

/// Server search is a substring AND over terms, so a typo (`smolm2`) finds nothing; these
/// variants shorten one term at a time (down to 3 chars, at most 3 shorter) for the fuzzy filter.
fn widen(query: &str) -> Vec<String> {
    let terms: Vec<&str> = query.split_whitespace().collect();
    let mut out = Vec::new();
    for (i, t) in terms.iter().enumerate() {
        let chars: Vec<(usize, char)> = t.char_indices().collect();
        for k in (3.max(chars.len().saturating_sub(3))..chars.len()).rev() {
            let mut v = terms.clone();
            v[i] = &t[..chars[k].0];
            out.push(v.join(" "));
        }
    }
    // ponytail: at most 8 extra requests per query; a smarter (n-gram) index needs a server.
    out.truncate(8);
    out
}

/// Runs the hub search through curl; returns the response body or curl's first stderr line.
/// No pipeline/tag filter: untagged GGUF repos and not-for-all-audiences repos must show.
fn search_hub(query: &str, sort: Sort, token: Option<&str>) -> Result<String, String> {
    hub_list(
        &format!("search={}&sort={}", pct_encode(query), sort.api()),
        token,
    )
}

/// Best loadable quantized derivative of hub model `id` (most downloaded repo tagged
/// `base_model:quantized:<id>` whose own blocker is none and whose download fits the free disk),
/// or `None` when the hub lists no such repo.
fn quantized_variants(
    id: &str,
    token: Option<&str>,
    total: Option<u64>,
    free_disk: Option<u64>,
    base_params: Option<u64>,
) -> Vec<Entry> {
    let body = match hub_list(
        &format!(
            "filter={}&sort=downloads",
            pct_encode(&format!("base_model:quantized:{id}"))
        ),
        token,
    ) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    let whole = |e: &Entry| match (e.params, base_params) {
        (Some(p), Some(b)) => p.saturating_mul(10) >= b.saturating_mul(9),
        _ => true,
    };
    let mut candidates: Vec<Entry> = Vec::new();
    for mut e in parse_hub_rows(&body, total, base_params)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| whole(e) && e.draft_for.is_none() && !e.gated)
    {
        if !e.quants.is_empty()
            && e.blocker
                .as_deref()
                .map_or(true, |b| b.starts_with("needs ~"))
        {
            candidates.extend(
                e.quants
                    .into_iter()
                    .filter(|q| variant_fits(q, total, free_disk)),
            );
        } else if e.blocker.is_none() && variant_fits(&e, total, free_disk) {
            candidates.push(e);
        } else if e
            .blocker
            .as_deref()
            .is_some_and(|b| b.starts_with("needs ~"))
            && e.quants.is_empty()
            && e.size.is_some()
            && variant_fits(&e, total, free_disk)
        {
            e.blocker = None;
            candidates.push(e);
        }
    }
    sort_variants(&mut candidates, total, free_disk);
    candidates
}

/// `/api/models?<query>` with the fields a row needs, through curl: the response body or curl's
/// first stderr line.
fn hub_list(query: &str, token: Option<&str>) -> Result<String, String> {
    let expand: String = [
        "downloads",
        "likes",
        "lastModified",
        "safetensors",
        "gguf",
        "config",
        "siblings",
        "tags",
        "gated",
        "pipeline_tag",
        "library_name",
    ]
    .iter()
    .map(|e| format!("&expand%5B%5D={e}"))
    .collect();
    let url = format!(
        "https://huggingface.co/api/models?{query}&direction=-1&limit={SEARCH_LIMIT}{expand}"
    );
    let mut cmd = Command::new("curl");
    cmd.args(["-fsS", "-m", "15"]);
    if let Some(t) = token.filter(|t| !t.is_empty()) {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    let out = cmd
        .arg(&url)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(stderr
            .lines()
            .next()
            .map(str::to_string)
            .unwrap_or_else(|| format!("curl exited with {}", out.status)));
    }
    let body = String::from_utf8(out.stdout).map_err(|e| format!("non-UTF-8 response: {e}"))?;
    // A NInfer artifact's format and architecture sit in its file header, not in any search field.
    let Ok(mut models) = serde_json::from_str::<Vec<Value>>(&body) else {
        return Ok(body);
    };
    crate::model::ninfer::attach_catalogs(&mut models, token);
    serde_json::to_string(&models).map_err(|e| e.to_string())
}

fn pct_encode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

/// Bytes per element of a safetensors dtype.
fn dtype_bytes(dtype: &str) -> f64 {
    match dtype {
        "F64" | "I64" | "U64" => 8.0,
        "F32" | "I32" | "U32" => 4.0,
        "F16" | "BF16" | "I16" | "U16" => 2.0,
        "F4" | "F4_E2M1" => 0.5,
        _ => 1.0,
    }
}

/// Parses `/api/models` JSON into entries with loadability blockers and a size estimate.
fn parse_hub_results(json: &str, total_memory: Option<u64>) -> Result<Vec<Entry>, String> {
    parse_hub_rows(json, total_memory, None)
}

/// [`parse_hub_results`] for derivatives of a model with `base_params` weights.
fn parse_hub_rows(
    json: &str,
    total_memory: Option<u64>,
    base_params: Option<u64>,
) -> Result<Vec<Entry>, String> {
    let models: Vec<Value> =
        serde_json::from_str(json).map_err(|e| format!("unexpected response: {e}"))?;
    Ok(models
        .iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?.to_string();
            let files: Vec<RepoFile> = m["siblings"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s["rfilename"].as_str())
                        .map(|p| RepoFile {
                            path: p.to_string(),
                            size: None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            let selection = select_weight_files(&files, None);
            // ponytail: hub `total` counts packed words inconsistently (some 4-bit repos
            // reports logical U32 counts, -mxfp4 raw words), so packed repos may be under-counted.
            let ninfer = selection.as_ref().is_ok_and(|s| {
                s.first()
                    .is_some_and(|f| f.ends_with(crate::model::ninfer::EXT))
            });
            let params = m["safetensors"]["total"]
                .as_u64()
                .or_else(|| m["gguf"]["total"].as_u64())
                .or_else(|| {
                    ninfer
                        .then(|| crate::model::ninfer::hub_params(m))
                        .flatten()
                });
            // Packed quantized repos (EXL3, NVFP4, ...) count stored words, not weights. A derivative
            // has its base model's weight count; `base_params` is that count when the base is known.
            let packed = ["quantization", "quantization_config"]
                .iter()
                .any(|k| m["config"][*k].is_object());
            let params = match base_params {
                Some(b) if packed => params.map(|p| p.max(b)),
                _ => params,
            };
            let size = selection.as_ref().ok().and_then(|sel| {
                let primary = sel.first()?;
                if primary.ends_with(".gguf") {
                    let bpw = detect_quant_format(Path::new(primary))?.bits_per_weight();
                    Some((m["gguf"]["total"].as_u64()? as f64 * bpw as f64 / 8.0) as u64)
                } else if primary.ends_with(".safetensors") || primary.ends_with(".index.json") {
                    let by_dtype = m["safetensors"]["parameters"].as_object()?;
                    Some(
                        by_dtype
                            .iter()
                            .map(|(d, n)| n.as_u64().unwrap_or(0) as f64 * dtype_bytes(d))
                            .sum::<f64>() as u64,
                    )
                } else if primary.ends_with(crate::model::ninfer::EXT) {
                    crate::model::ninfer::hub_size(m)
                } else {
                    None
                }
            });
            // PEFT adapters and fastText models are files the classifier/merge reads itself.
            let own_files = m["config"]["peft"].is_object() || m["library_name"] == "fasttext";
            let tags: Vec<&str> = m["tags"]
                .as_array()
                .map_or(Vec::new(), |t| t.iter().filter_map(Value::as_str).collect());
            let gguf_arch = m["gguf"]["architecture"].as_str();
            let draft_for = draft_target(&id, &m["config"], &tags, gguf_arch);
            // GGUF draft sidecars attach like safetensors drafters; fork
            // architectures (another tensor layout) are refused by name.
            let gguf_draft = gguf_arch.and_then(crate::runtime::draft::refused_draft_gguf);
            // GGUF `mmproj` repos (arch `clip`): projectors of a language-model repo.
            let gguf_projector = (m["gguf"]["architecture"] == "clip")
                .then(|| crate::model::hub::MMPROJ_ONLY.to_string());
            let non_mem_blocker = gguf_draft
                .or(gguf_projector)
                .or_else(|| pipeline_blocker(m).filter(|_| draft_for.is_none()))
                .or_else(|| quant_blocker(m))
                .or_else(|| split_gguf_blocker(m, &selection))
                .or_else(|| selection.err().filter(|_| !own_files));
            let packed_gguf = gguf_arch.is_some_and(crate::arch::dispatch::runs_spec_lm_gguf);
            let gguf_total = m["gguf"]["total"].as_u64().or(params);
            let downloads = m["downloads"].as_u64();
            let likes = m["likes"].as_u64();
            let updated = m["lastModified"].as_str().map(String::from);
            let mut variants: Vec<Entry> = files
                .iter()
                .map(|f| f.path.as_str())
                .filter(|p| {
                    let name = p.rsplit('/').next().unwrap_or(p).to_lowercase();
                    p.ends_with(".gguf")
                        && !name.contains("mmproj")
                        && !name.starts_with("mtp-")
                        && !name.starts_with("imatrix")
                        && !name.ends_with(".imatrix.gguf")
                        && (!p.contains("-of-") || p.contains("-00001-of-"))
                })
                .filter_map(|p| {
                    let bpw = detect_quant_format(Path::new(p))?.bits_per_weight();
                    Some(Entry {
                        id: format!("{id}/{p}"),
                        params,
                        downloads,
                        likes,
                        updated: updated.clone(),
                        size: gguf_total.map(|t| (t as f64 * bpw as f64 / 8.0) as u64),
                        gated: false,
                        cached: false,
                        blocker: None,
                        draft_for: None,
                        via: None,
                        quants: Vec::new(),
                        quant_idx: 0,
                    })
                })
                .collect();
            sort_variants(&mut variants, total_memory, None);
            let via = variants.first().cloned().map(Box::new);
            let variants_stay_packed = !variants.is_empty()
                && gguf_arch.map_or(true, crate::arch::dispatch::runs_spec_lm_gguf);
            let mut size = size;
            let blocker = non_mem_blocker.or_else(|| {
                if variants_stay_packed {
                    if variants.iter().any(|q| variant_fits(q, total_memory, None)) {
                        if size.is_none()
                            || packed_memory_blocker(size, total_memory).is_some()
                            || (!packed_gguf && memory_blocker(params, total_memory).is_some())
                        {
                            size = via.as_ref().and_then(|v| v.size).or(size);
                        }
                        None
                    } else {
                        variants
                            .first()
                            .and_then(|q| q.blocker.clone())
                            .or_else(|| packed_memory_blocker(size, total_memory))
                            .or_else(|| memory_blocker(params, total_memory))
                    }
                } else if packed_gguf {
                    packed_memory_blocker(size, total_memory)
                } else {
                    memory_blocker(params, total_memory)
                }
            });
            Some(Entry {
                id,
                params,
                downloads,
                likes,
                updated,
                size,
                gated: !matches!(m["gated"], Value::Null | Value::Bool(false)),
                cached: false,
                blocker,
                draft_for,
                via,
                quants: variants,
                quant_idx: 0,
            })
        })
        .collect())
}

/// A GGUF architecture a diffusion spec's denoiser claims (`match.gguf_arch`).
fn diffusion_gguf_arch(a: &str) -> bool {
    crate::arch::registry().is_ok_and(|r| {
        r.pipeline_gguf_archs(crate::arch::Pipeline::Diffusion)
            .iter()
            .any(|g| g == a)
    })
}

/// Text-to-image repos an image loader runs: a diffusion family's diffusers pipeline, or a
/// derivative / GGUF of a listed base (`match.bases`).
fn image_supported(m: &Value) -> bool {
    if let Some(c) = m["config"]["diffusers"]["_class_name"].as_str() {
        return c.ends_with("Pipeline")
            && crate::arch::registry().is_ok_and(|r| {
                r.matchers().iter().any(|(_, m, p)| {
                    *p == crate::arch::Pipeline::Diffusion
                        && m.diffusers_class.iter().any(|d| d == c)
                })
            });
    }
    let tags: Vec<&str> = m["tags"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    // Derivatives of a diffusion spec's `match.bases` (card `base_model`, [`crate::runtime::diffusion::load_image`]).
    let spec_bases = crate::arch::registry().map_or_else(
        |_| Vec::new(),
        |r| r.derived_bases(crate::arch::Pipeline::Diffusion),
    );
    let derived = |kind: &str| {
        tags.iter()
            .filter_map(|t| t.strip_prefix(kind))
            .any(|b| spec_bases.iter().any(|s| s == b))
    };
    crate::arch::compat()["picker"]["diffusion_gguf_arch"]
        .as_str()
        .is_some_and(|a| m["gguf"]["architecture"] == a)
        || (m["gguf"]["architecture"]
            .as_str()
            .is_some_and(diffusion_gguf_arch)
            && !tags.iter().any(|t| {
                [
                    "base_model:adapter:",
                    "base_model:finetune:",
                    "base_model:quantized:",
                ]
                .iter()
                .filter_map(|k| t.strip_prefix(k))
                .any(|b| !spec_bases.iter().any(|s| s == b))
            }))
        || [
            "base_model:adapter:",
            "base_model:finetune:",
            "base_model:quantized:",
        ]
        .iter()
        .any(|k| derived(k))
}

/// Video repos: the loader's own file rule ([`crate::model::video_repo::pick_gguf`]) for video GGUF
/// repos, or the pipeline class for diffusers ones; `None` when `m` is not a video repo.
fn video_verdict(m: &Value) -> Option<Result<(), String>> {
    let reg = crate::arch::registry().ok()?;
    let vg = &crate::arch::compat()["video_gguf"];
    if vg["architecture"]
        .as_str()
        .is_some_and(|a| m["gguf"]["architecture"] == a)
    {
        let files: Vec<RepoFile> = m["siblings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s["rfilename"].as_str())
            .map(|p| RepoFile {
                path: p.to_string(),
                size: None,
            })
            .collect();
        return Some(crate::model::video_repo::pick_gguf(&files, None).map(|_| ()));
    }
    let c = m["config"]["diffusers"]["_class_name"].as_str()?;
    if let Some(why) = vg["refused_diffusers_classes"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|r| {
            (r[0].as_str() == Some(c))
                .then(|| r[1].as_str().map(String::from))
                .flatten()
        })
    {
        return Some(Err(why));
    }
    (c.ends_with("Pipeline")
        && reg.matchers().iter().any(|(_, m, p)| {
            *p == crate::arch::Pipeline::Video && m.diffusers_class.iter().any(|d| d == c)
        }))
    .then_some(Ok(()))
}

/// Repos tagged for a task the engine does not run. Untagged repos, the classifier verdict
/// ([`crate::model::classifier::hub_verdict`]: sequence classifiers, PEFT/fastText/SetFit repos,
/// generators tagged for classification, or the specific reason a classification repo is
/// refused), repos whose `model_type` a native implementation covers, and supported image repos
/// decide first.
fn pipeline_blocker(m: &Value) -> Option<String> {
    if image_supported(m) {
        return None;
    }
    if let Some(verdict) = video_verdict(m) {
        return verdict.err();
    }
    if let Some(c) = m["config"]["diffusers"]["_class_name"].as_str() {
        return Some(if c.contains("Inpaint") {
            format!("{c}: inpainting needs a mask, which no endpoint takes")
        } else {
            format!("diffusers {c} (not supported)")
        });
    }
    if let Some(verdict) = crate::model::ninfer::hub_verdict(m) {
        return verdict.err();
    }
    let model_type = m["config"]["model_type"].as_str();
    if let Some((t, why)) = model_type.and_then(|t| Some((t, crate::arch::unsupported(t)?))) {
        return Some(format!("{t}: {why}"));
    }
    let is_split_gguf = m["siblings"].as_array().map_or(false, |a| {
        a.iter()
            .filter_map(|s| s["rfilename"].as_str())
            .any(|f| f.contains("-00001-of-"))
    });
    if !is_split_gguf {
        if let Some(a) = m["gguf"]["architecture"]
            .as_str()
            .filter(|_| model_type.is_none())
        {
            if let Some(why) = crate::arch::unsupported(a) {
                return Some(format!("{a}: {why}"));
            }
        }
    }
    let tag = m["pipeline_tag"].as_str()?;
    if matches!(tag, "feature-extraction" | "sentence-similarity") {
        let gguf = m["gguf"]["architecture"].as_str();
        if crate::arch::registry().is_ok_and(|r| {
            model_type.is_some_and(|t| r.runs_system1(t))
                || gguf.is_some_and(|a| {
                    r.pipeline_gguf_archs(crate::arch::Pipeline::Encoder)
                        .iter()
                        .any(|g| g == a)
                })
        }) {
            return None;
        }
    }
    if let Some(verdict) = crate::model::classifier::hub_verdict(m) {
        return verdict.err();
    }
    if let Some(a) = m["gguf"]["architecture"]
        .as_str()
        .filter(|a| diffusion_gguf_arch(a))
    {
        return Some(format!(
            "{tag} model, GGUF architecture '{a}' derived from a base no image spec runs (e.g. image editing, which conditions on reference images and is not implemented); only text-to-image GGUFs of this architecture load"
        ));
    }
    if tag == "text-to-image" {
        let base = m["tags"].as_array().into_iter().flatten().find_map(|t| {
            let t = t.as_str()?;
            [
                "base_model:adapter:",
                "base_model:finetune:",
                "base_model:quantized:",
            ]
            .iter()
            .find_map(|k| t.strip_prefix(k))
        });
        return Some(match base {
            Some(b) => format!(
                "derivative of {b}: {}",
                crate::arch::compat()["picker"]["image_derivative"]
                    .as_str()
                    .expect("compat.picker.image_derivative")
            ),
            None => "text-to-image repo without a diffusers pipeline (model_index.json) or base model tag".into(),
        });
    }
    if spec_lm_type(m) {
        return None;
    }
    let decoder_task = matches!(
        tag,
        "translation" | "summarization" | "text2text-generation"
    ) && model_type
        .is_some_and(|t| crate::arch::registry().is_ok_and(|r| r.runs_model_type(t)));
    // GGUF-only repos carry no config; their GGUF architecture names the model.
    let gguf = m["gguf"]["architecture"]
        .as_str()
        .filter(|_| model_type.is_none());
    let text_gen = decoder_task
        || gguf.is_some_and(|a| gguf_lm_archs().iter().any(|x| x == a))
        || matches!(tag, "text-generation" | "conversational");
    (!text_gen).then(|| match (model_type, gguf) {
        (Some(t), _) => format!("{tag} model, architecture '{t}' (not supported)"),
        (None, Some(a)) => format!(
            "{tag} model, {} GGUF architecture '{a}' (not supported; {} GGUFs load)",
            crate::arch::compat()["picker"]["gguf_source"]
                .as_str()
                .expect("compat.picker.gguf_source"),
            gguf_lm_archs().join(", ")
        ),
        (None, None) => match m["library_name"].as_str().filter(|l| *l != "transformers") {
            Some(l) => format!("{tag} model for '{l}' (not a transformers checkpoint)"),
            None => {
                let files: Vec<&str> = m["siblings"]
                    .as_array()
                    .map_or(Vec::new(), |a| a.iter().filter_map(|s| s["rfilename"].as_str()).collect());
                let has = |e: &str| files.iter().any(|f| f.ends_with(e));
                if has(".task") || has(".litertlm") || has(".tflite") {
                    format!("{tag} model shipped as a LiteRT / MediaPipe bundle (no transformers or GGUF weights)")
                } else if has(".nexa") {
                    format!("{tag} model shipped as a Nexa SDK bundle (no transformers or GGUF weights)")
                } else if files.contains(&"params.json") && files.iter().any(|f| f.starts_with("consolidated") && f.ends_with(".pth")) {
                    format!("{tag} model as a Meta-native checkpoint (consolidated.*.pth + params.json, no config.json); convert it to Hugging Face format first")
                } else if files.contains(&"config.json") {
                    format!("{tag} model whose config.json the Hub could not read (git-LFS stored) or that has no model_type or architectures (project metadata or custom code): the architecture cannot be inferred")
                } else {
                    format!("{tag} model without a config.json (weights only): the architecture cannot be inferred")
                }
            }
        },
    })
}

/// GGUF architectures a generation path reads, whatever task the repo is tagged with.
fn gguf_lm_archs() -> Vec<String> {
    crate::arch::registry().map_or_else(|_| Vec::new(), |r| r.causal_lm_gguf_archs())
}

/// Split GGUF sets (`-00001-of-00003.gguf`, read as one weight set by every GGUF loader) of an
/// architecture no loader runs: the reason names it (typically huge MoE / DSA models).
fn split_gguf_blocker(m: &Value, selection: &Result<Vec<String>, String>) -> Option<String> {
    let sel = selection.as_ref().ok()?;
    let a = m["gguf"]["architecture"].as_str()?;
    let allowed = crate::arch::compat()["picker"]["split_gguf_allowed"]
        .as_array()
        .is_some_and(|arr| arr.iter().any(|x| x.as_str() == Some(a)));
    (sel.len() > 1
        && sel[0].ends_with(".gguf")
        && !gguf_lm_archs().iter().any(|x| x == a)
        && !allowed)
        .then(|| {
            format!(
                "split GGUF of {} architecture '{a}' (not supported; {} GGUFs load)",
                crate::arch::compat()["picker"]["gguf_source"]
                    .as_str()
                    .expect("compat.picker.gguf_source"),
                gguf_lm_archs().join(", ")
            )
        })
}

/// Pre-quantized schemes the loader rejects ([`crate::model::prequant::scheme_blocker`], the same
/// rule the loader applies; MLX `quantization` included).
fn quant_blocker(m: &Value) -> Option<String> {
    crate::model::prequant::scheme_blocker(&m["config"], None)
}

/// `model_type` is a causal-LM family spec's own match ([`crate::arch::dispatch::runs_spec_lm`]).
fn spec_lm_type(m: &Value) -> bool {
    m["config"]["model_type"]
        .as_str()
        .is_some_and(crate::arch::dispatch::runs_spec_lm)
}

/// GGUF architectures whose expert banks stay packed in the file data (never expanded).
/// A causal-LM GGUF row (its family spec keeps the blocks packed) is sized by its weight file
/// (`gguf.total` counts every expert; a 14B-A3B MXFP4 file ran at 13 GB RSS for its 8.8 GB): file
/// bytes + ~10% for the caches and activations.
// ponytail: a flat 1.1 over the file; decode scratch and KV grow with context.
fn packed_memory_blocker(file_bytes: Option<u64>, total_memory: Option<u64>) -> Option<String> {
    let (bytes, total) = (file_bytes?, total_memory?);
    let gb = |b: f64| b / (1u64 << 30) as f64;
    let need = bytes as f64 * 1.1;
    (need > total as f64).then(|| {
        format!(
            "needs ~{:.0} GB (weights stay packed in the {:.0} GB GGUF); host has {:.0} GB",
            gb(need),
            gb(bytes as f64),
            gb(total as f64)
        )
    })
}

/// Rows without a packed-GGUF size (safetensors LMs, image pipelines) are bounded by 16-bit
/// weights: 2 bytes per parameter.
fn memory_blocker(params: Option<u64>, total_memory: Option<u64>) -> Option<String> {
    let (params, total) = (params?, total_memory?);
    let gb = |b: f64| b / (1u64 << 30) as f64;
    let need = params.saturating_mul(2);
    (need > total).then(|| {
        format!(
            "needs ~{:.0} GB as bf16; host has {:.0} GB",
            gb(need as f64),
            gb(total as f64)
        )
    })
}

/// Case-insensitive multi-term subsequence match. Every whitespace-separated term must match in order
/// within the candidate (terms independent of each other). Returns the sorted matched char indices,
/// per term the tightest match (for highlighting).
fn fuzzy_hits(query: &str, candidate: &str) -> Option<Vec<usize>> {
    let orig: Vec<char> = candidate.chars().collect();
    let lower: Vec<char> = orig.iter().map(|&c| lower_char(c)).collect();
    let mut hits = Vec::new();
    for term in query.split_whitespace() {
        let term: Vec<char> = term.chars().map(lower_char).collect();
        let mut best: Option<(i32, Vec<usize>)> = None;
        // ponytail: greedy match from every start of the first char, O(n^2) per term; fine for repo ids.
        for start in (0..lower.len()).filter(|&i| lower[i] == term[0]) {
            let Some(pos) = match_from(&lower, &term, start) else {
                break; // no match from here means none from any later start
            };
            let score = score_positions(&orig, &pos);
            if best.as_ref().map_or(true, |(b, _)| score > *b) {
                best = Some((score, pos));
            }
        }
        hits.extend(best?.1);
    }
    hits.sort_unstable();
    hits.dedup();
    Some(hits)
}

fn lower_char(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

fn match_from(lower: &[char], term: &[char], start: usize) -> Option<Vec<usize>> {
    let mut pos = Vec::with_capacity(term.len());
    let mut i = start;
    for &t in term {
        while *lower.get(i)? != t {
            i += 1;
        }
        pos.push(i);
        i += 1;
    }
    Some(pos)
}

fn score_positions(c: &[char], pos: &[usize]) -> i32 {
    let mut score = 0;
    for (k, &i) in pos.iter().enumerate() {
        score += 1;
        let prev = i.checked_sub(1).map(|p| c[p]);
        let word_start = match prev {
            None => true,
            Some(p) => {
                matches!(p, '/' | '-' | '_' | '.') || (p.is_lowercase() && c[i].is_uppercase())
            }
        };
        if word_start {
            score += 10;
        }
        if prev == Some('/') {
            score += 6;
        }
        if k > 0 {
            let gap = i - pos[k - 1] - 1;
            if gap == 0 {
                score += 8;
            } else {
                score -= gap as i32;
            }
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_terms_rank_and_highlight() {
        let smol = "HuggingFaceTB/SmolLM2-135M-Instruct";
        let hits = fuzzy_hits("smol inst", smol).expect("smol inst matches SmolLM2 Instruct");
        let matched: String = hits.iter().map(|&i| smol.chars().nth(i).unwrap()).collect();
        assert_eq!(matched, "SmolInst");
        assert!(fuzzy_hits("smol inst", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").is_none());
        // Typo-tolerant subsequence match, and the widened server queries reach it.
        assert!(fuzzy_hits("smolm2 inst", smol).is_some());
        assert!(widen("smolm2 inst").contains(&"smol inst".to_string()));

        // The tightest occurrence is highlighted, not the first scattered one.
        assert_eq!(
            fuzzy_hits("glm", "THUDM/glm-edge-1.5b-chat").unwrap(),
            vec![6, 7, 8]
        );
        assert!(fuzzy_hits("xyz", smol).is_none());
        assert_eq!(fuzzy_hits("  ", smol), Some(Vec::new()));
    }

    #[test]
    fn hub_results_carry_loadability() {
        let json = r#"[
            {"id":"a/single","downloads":10,"likes":3,"lastModified":"2025-11-03T10:00:00.000Z",
             "pipeline_tag":"text-generation","safetensors":{"parameters":{"BF16":135000000},"total":135000000},
             "config":{"model_type":"llama"},"siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"}]},
            {"id":"b/sharded","gated":"auto","siblings":[{"rfilename":"model.safetensors.index.json"},{"rfilename":"model-00001-of-00002.safetensors"},{"rfilename":"model-00002-of-00002.safetensors"}]},
            {"id":"c/gguf","gguf":{"total":1000000000,"architecture":"qwen2"},"siblings":[{"rfilename":"x-Q4_K_M.gguf"},{"rfilename":"x-Q8_0.gguf"}]},
            {"id":"c/gguf-mt","pipeline_tag":"translation","gguf":{"total":1000000000,"architecture":"qwen2"},"siblings":[{"rfilename":"mt-Q4_K_M.gguf"}]},
            {"id":"c/hy-gguf","pipeline_tag":"translation","gguf":{"total":1800000000,"architecture":"hunyuan-dense"},"siblings":[{"rfilename":"hy-Q4_K_M.gguf"}]},
            {"id":"c/q35-gguf","pipeline_tag":"translation","gguf":{"total":2000000000,"architecture":"qwen35"},"siblings":[{"rfilename":"q35-Q4_K_M.gguf"}]},
            {"id":"c/q35-cfg-gguf","pipeline_tag":"image-text-to-text","config":{"model_type":"qwen3_5"},"gguf":{"total":800000000,"architecture":"qwen35"},"siblings":[{"rfilename":"q35-Q8_0.gguf"}]},
            {"id":"c/q35moe-gguf","pipeline_tag":"translation","gguf":{"total":35000000000,"architecture":"qwen35moe"},"siblings":[{"rfilename":"moe-Q4_K_M.gguf"}]},
            {"id":"c/mamba-gguf","pipeline_tag":"translation","gguf":{"total":3000000000,"architecture":"mamba"},"siblings":[{"rfilename":"mamba-Q4_K_M.gguf"}]},
            {"id":"g/madlad-gguf","pipeline_tag":"translation","config":{"model_type":"t5","architectures":["T5ForConditionalGeneration"]},"siblings":[{"rfilename":"config.json"},{"rfilename":"model-q2k.gguf"},{"rfilename":"model-q4k.gguf"}]},
            {"id":"c/gemma4-gguf","pipeline_tag":"image-text-to-text","config":{"model_type":"gemma4"},"gguf":{"total":4000000000,"architecture":"gemma4"},"siblings":[{"rfilename":"g-Q4_K_M.gguf"}]},
            {"id":"d/onnx","siblings":[{"rfilename":"onnx/model_q4.onnx"}]},
            {"id":"e/big","safetensors":{"total":70000000000},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"e/q35-27b-gguf","pipeline_tag":"image-text-to-text","gguf":{"total":27000000000,"architecture":"qwen35"},"siblings":[{"rfilename":"m-Q4_K_M.gguf"}]},
            {"id":"f/awq","config":{"quantization_config":{"quant_method":"awq","bits":4}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"g/gptq","config":{"quantization_config":{"quant_method":"gptq"}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"h/mlx","library_name":"mlx","safetensors":{"parameters":{"U32":100,"F16":10}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"h/mlx7","library_name":"mlx","config":{"quantization_config":{"bits":7}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"i/ct","config":{"quantization_config":{"quant_method":"compressed-tensors"}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"j/modelopt","config":{"quantization_config":{"quant_method":"modelopt"}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"k/asr","pipeline_tag":"automatic-speech-recognition","siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"n/cls","pipeline_tag":"text-classification","config":{"model_type":"distilbert","architectures":["DistilBertForSequenceClassification"]},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"o/ner","pipeline_tag":"token-classification","config":{"model_type":"bert","architectures":["BertForTokenClassification"]},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"p/gemma4u","pipeline_tag":"any-to-any","config":{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"]},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"p/lora-cls","pipeline_tag":"text-classification","library_name":"peft","config":{"peft":{"base_model_name_or_path":"o/bert","task_type":"SEQ_CLS"}},"siblings":[{"rfilename":"adapter_config.json"},{"rfilename":"adapter_model.safetensors"}]},
            {"id":"p/lora-lm","pipeline_tag":"text-classification","library_name":"peft","config":{"peft":{"base_model_name_or_path":"Qwen/Qwen3-0.6B","task_type":"CAUSAL_LM"}},"siblings":[{"rfilename":"adapter_config.json"},{"rfilename":"adapter_model.safetensors"}]},
            {"id":"p/lora-head","pipeline_tag":"text-classification","library_name":"peft","config":{"peft":{"base_model_name_or_path":"Qwen/Qwen3.5-4B","task_type":"CAUSAL_LM"}},"siblings":[{"rfilename":"adapter_config.json"},{"rfilename":"adapter_model.safetensors"},{"rfilename":"head.pt"}]},
            {"id":"s/setfit","pipeline_tag":"text-classification","library_name":"setfit","config":{"model_type":"bert","architectures":["BertModel"]},"siblings":[{"rfilename":"config_setfit.json"},{"rfilename":"model_head.pkl"},{"rfilename":"model.safetensors"}]},
            {"id":"t/fasttext","pipeline_tag":"text-classification","library_name":"fasttext","siblings":[{"rfilename":"model.bin"}]},
            {"id":"t/gguf-cls","pipeline_tag":"text-classification","gguf":{"architecture":"bert","total":568000000},"siblings":[{"rfilename":"r-Q8_0.gguf"}]},
            {"id":"t/gguf-lora","pipeline_tag":"text-classification","siblings":[{"rfilename":"loras/aml.gguf"}]},
            {"id":"u/modernbert","pipeline_tag":"text-classification","config":{"model_type":"modernbert","architectures":["ModernBertForSequenceClassification"]},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"u/gen","pipeline_tag":"text-classification","config":{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"u/custom","pipeline_tag":"text-classification","config":{"model_type":"fev","architectures":["FevForDecision"],"auto_map":{"AutoModel":"modeling_fev.FevForDecision"}},"siblings":[{"rfilename":"model.safetensors"},{"rfilename":"modeling_fev.py"}]},
            {"id":"u/base","pipeline_tag":"text-classification","config":{"model_type":"bert","architectures":["BertModel"]},"siblings":[{"rfilename":"pytorch_model.bin"}]},
            {"id":"u/lib","pipeline_tag":"text-classification","library_name":"catboost","siblings":[{"rfilename":"model.cbm"}]},
            {"id":"u/nocfg","pipeline_tag":"text-classification","siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"l/bin","siblings":[{"rfilename":"pytorch_model.bin"}]},
            {"id":"q/hunyuan-mt","pipeline_tag":"translation","config":{"model_type":"hunyuan_v1_dense"},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"r/qwen3-big","safetensors":{"total":14000000000},"config":{"model_type":"qwen3"},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"p/quark","config":{"model_type":"llama","quantization_config":{"quant_method":"quark"}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"q/fp8","config":{"model_type":"llama","quantization_config":{"quant_method":"fp8"}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"m/marlin","config":{"quantization_config":{"quant_method":"compressed-tensors","format":"marlin-24"}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"p/exl3","config":{"model_type":"llama","quantization_config":{"quant_method":"exl3","bits":4}},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"p/glq","config":{"model_type":"llama","quantization_config":{"quant_method":"glq"}},"siblings":[{"rfilename":"model.safetensors"},{"rfilename":"e8_codebook.pt"}]},
            {"id":"p/marian","pipeline_tag":"translation","config":{"model_type":"marian","architectures":["MarianMTModel"]},"siblings":[{"rfilename":"pytorch_model.bin"}]},
            {"id":"q/bart-nli","pipeline_tag":"zero-shot-classification","config":{"model_type":"bart","architectures":["BartForSequenceClassification"]},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"p/t5","pipeline_tag":"translation","config":{"model_type":"t5","architectures":["T5ForConditionalGeneration"]},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"q/mt5","pipeline_tag":"summarization","config":{"model_type":"mt5"},"siblings":[{"rfilename":"pytorch_model.bin"}]},
            {"id":"r/llama-mt","pipeline_tag":"translation","config":{"model_type":"llama"},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"s/hibiki-mt","pipeline_tag":"translation","config":{"model_type":"hibiki"},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"p/sdxl","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"StableDiffusionXLPipeline"}},"safetensors":{"total":2567463684},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"unet/config.json"},{"rfilename":"unet/diffusion_pytorch_model.fp16.safetensors"},{"rfilename":"unet/diffusion_pytorch_model.safetensors"},{"rfilename":"text_encoder/model.onnx"},{"rfilename":"text_encoder/model.safetensors"},{"rfilename":"sd_xl_base_1.0.safetensors"}]},
            {"id":"q/inpaint","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"StableDiffusionInpaintPipeline"}},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"unet/diffusion_pytorch_model.safetensors"}]},
            {"id":"r/lora","pipeline_tag":"text-to-image","tags":["lora","base_model:adapter:stabilityai/stable-diffusion-xl-base-1.0"],"siblings":[{"rfilename":"pixel.safetensors"}]},
            {"id":"s/flux-lora","pipeline_tag":"text-to-image","tags":["lora","base_model:adapter:black-forest-labs/FLUX.1-dev"],"siblings":[{"rfilename":"x.safetensors"}]},
            {"id":"s/flux-gguf","pipeline_tag":"text-to-image","library_name":"gguf","gguf":{"total":11891178560,"architecture":"flux"},"tags":["base_model:quantized:black-forest-labs/FLUX.1-schnell"],"siblings":[{"rfilename":"flux1-schnell-Q4_0.gguf"},{"rfilename":"flux1-schnell-Q8_0.gguf"}]},
            {"id":"s/chroma","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"ChromaPipeline"}},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors"}]},
            {"id":"s/sana-lora","pipeline_tag":"text-to-image","tags":["lora","base_model:adapter:Efficient-Large-Model/Sana_1600M_1024px_diffusers"],"siblings":[{"rfilename":"x.safetensors"}]},
            {"id":"t/finetune","pipeline_tag":"text-to-image","tags":["base_model:finetune:stable-diffusion-v1-5/stable-diffusion-v1-5"],"siblings":[{"rfilename":"juggernaut.safetensors"}]},
            {"id":"u/ckpt-only","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"StableDiffusionPipeline"}},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"unet/diffusion_pytorch_model.bin"}]},
            {"id":"v/zimage","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"ZImagePipeline"}},"safetensors":{"parameters":{"F32":6154908736},"total":6154908736},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors.index.json"},{"rfilename":"transformer/diffusion_pytorch_model-00001-of-00003.safetensors"}]},
            {"id":"w/sd3","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"StableDiffusion3Pipeline"}},"safetensors":{"total":2469663936},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors"}]},
            {"id":"x/qwen-image","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"QwenImagePipeline"}},"safetensors":{"total":20430401088},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors"}]},
            {"id":"q/qwen21","pipeline_tag":"text-to-image","config":{"diffusers":{"_class_name":"QwenImage21Pipeline"}},"safetensors":{"total":4200000000},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors"}]},
            {"id":"q/qwen21-gguf","pipeline_tag":"text-to-image","library_name":"gguf","gguf":{"total":7115124736,"architecture":"qwen_image21"},"tags":["base_model:quantized:Qwen/Qwen-Image-2.1"],"siblings":[{"rfilename":"qwen-image-2.1-Q4_0.gguf"},{"rfilename":"qwen-image-2.1-Q8_0.gguf"}]},
            {"id":"q/qwen21-turbo","pipeline_tag":"text-to-image","gguf":{"total":7115124736,"architecture":"qwen_image"},"tags":["lora","base_model:adapter:Qwen/Qwen-Image-2.1"],"siblings":[{"rfilename":"turbo-Q4_K_M.gguf"},{"rfilename":"turbo-lora.safetensors"},{"rfilename":"comfyui/t2i.json"}]},
            {"id":"q/qwen-edit-gguf","pipeline_tag":"image-to-image","gguf":{"total":20430401088,"architecture":"qwen_image"},"tags":["base_model:quantized:Qwen/Qwen-Image-Edit"],"siblings":[{"rfilename":"edit-Q4_0.gguf"}]},
            {"id":"q/qwen-lora","pipeline_tag":"text-to-image","tags":["lora","base_model:adapter:Qwen/Qwen-Image"],"siblings":[{"rfilename":"x.safetensors"}]},
            {"id":"z-lab/Qwen3-4B-DFlash-b16","pipeline_tag":"feature-extraction","config":{"architectures":["DFlashDraftModel"],"model_type":"qwen3"},"siblings":[{"rfilename":"model.safetensors"}]},
            {"id":"r/Qwen3.5-2B-DSpark","tags":["base_model:finetune:Qwen/Qwen3.5-2B","base_model:Qwen/Qwen3.5-2B"],"config":{"architectures":["DSparkDraftModel"]},"siblings":[{"rfilename":"model.safetensors"},{"rfilename":"optimizer_state_dict.pt"}]},
            {"id":"r/Qwen3.5-2B-DSpark-GGUF","gguf":{"total":451477841,"architecture":"dflash"},"siblings":[{"rfilename":"Qwen3.5-2B-DSpark-Q8_0.gguf"}]},
            {"id":"r/fork-DFlash-GGUF","gguf":{"total":537427200,"architecture":"dflash-draft"},"siblings":[{"rfilename":"f-Q4_K_M.gguf"}]},
            {"id":"r/smol-split","pipeline_tag":"text-generation","gguf":{"total":135000000,"architecture":"llama"},"siblings":[{"rfilename":"Q4_0/s-00001-of-00003.gguf"},{"rfilename":"Q4_0/s-00002-of-00003.gguf"},{"rfilename":"Q4_0/s-00003-of-00003.gguf"}]},
            {"id":"r/glm-split","pipeline_tag":"text-generation","gguf":{"total":744000000000,"architecture":"glm_moe_dsa"},"siblings":[{"rfilename":"IQ2_S/G-00001-of-00003.gguf"},{"rfilename":"IQ2_S/G-00002-of-00003.gguf"},{"rfilename":"IQ2_S/G-00003-of-00003.gguf"}]},
            {"id":"g/gemma-4-E2B-any-GGUF","pipeline_tag":"any-to-any","gguf":{"total":4600000000,"architecture":"gemma4"},"siblings":[{"rfilename":"g-Q4_K_M.gguf"},{"rfilename":"mmproj-g-F16.gguf"}]},
            {"id":"g/gemma-4-26B-A4B-it-GGUF","pipeline_tag":"image-text-to-text","config":{"model_type":"gemma4"},"gguf":{"total":25000000000,"architecture":"gemma4"},"siblings":[{"rfilename":"g-Q4_K_M.gguf"}]},
            {"id":"g/gemma-4-E2B-mmproj","pipeline_tag":"image-text-to-text","gguf":{"architecture":"clip"},"siblings":[{"rfilename":"mmproj-g-F16.gguf"}]},
            {"id":"g/t5-llamacpp-gguf","pipeline_tag":"translation","gguf":{"total":60000000,"architecture":"t5"},"siblings":[{"rfilename":"t5-Q8_0.gguf"}]},
            {"id":"n/qwen-ninfer","pipeline_tag":"image-text-to-text","library_name":"ninfer","siblings":[{"rfilename":"qwen3_5_0_8b.ninfer"}]},
            {"id":"n/Qwen3.6-35B-A3B-NInfer","pipeline_tag":"image-text-to-text","library_name":"ninfer","tags":["base_model:Qwen/Qwen3.6-35B-A3B"],"siblings":[{"rfilename":"qwen3_6_35b_a3b.ninfer"}]},
            {"id":"n/q4exp","pipeline_tag":"image-text-to-text","library_name":"ninfer","siblings":[{"rfilename":"q.ninfer"}],"ninfer_catalog":{"version":3,"catalog":{"components":{"text":{"config":{"architectures":["Qwen4ExpForCausalLM"]}}},"objects":[]}}},
            {"id":"n/v1","pipeline_tag":"image-text-to-text","library_name":"ninfer","siblings":[{"rfilename":"q.ninfer"}],"ninfer_catalog":{"version":1,"catalog":{"objects":[]}}},
            {"id":"n/dsv41","pipeline_tag":"image-text-to-text","config":{"model_type":"deepseek_v41"},"siblings":[{"rfilename":"model.safetensors"}]}
        ]"#;
        let entries = parse_hub_results(json, Some(16 << 30)).unwrap();
        let by_id = |id: &str| entries.iter().find(|e| e.id == id).unwrap();
        let blocker = |id: &str| by_id(id).blocker.clone().unwrap_or_default();

        assert_eq!(by_id("n/qwen-ninfer").blocker, None);
        assert!(blocker("n/Qwen3.6-35B-A3B-NInfer").contains("mixture-of-experts"));
        assert!(blocker("n/q4exp").starts_with("NInfer architecture Qwen4ExpForCausalLM"));
        assert!(
            blocker("n/v1").contains("container v1 is not supported"),
            "{}",
            blocker("n/v1")
        );
        let ds = blocker("n/dsv41");
        assert!(
            ds.starts_with("deepseek_v41: DeepSeek-V4 family") && ds.contains("hyper-connections"),
            "{ds}"
        );
        let single = by_id("a/single");
        assert_eq!(single.blocker, None);
        assert_eq!(
            (single.params, single.downloads, single.likes, single.size),
            (Some(135_000_000), Some(10), Some(3), Some(270_000_000))
        );
        assert_eq!(single.updated.as_deref(), Some("2025-11-03T10:00:00.000Z"));
        for loadable in [
            "b/sharded",
            "c/gguf",
            "c/gguf-mt",
            "c/hy-gguf",
            "c/q35-gguf",
            "c/q35-cfg-gguf",
            "g/madlad-gguf",
            "d/onnx",
            "i/ct",
            "j/modelopt",
            "l/bin",
            "p/glq",
            "p/gemma4u",
            "h/mlx",
            "q/hunyuan-mt",
            "p/quark",
            "q/fp8",
            "p/exl3",
            "p/marian",
            "p/t5",
            "q/mt5",
            "r/llama-mt",
            "p/sdxl",
            "r/lora",
            "t/finetune",
            "v/zimage",
            "w/sd3",
            "s/flux-lora",
            "s/chroma",
        ] {
            assert_eq!(by_id(loadable).blocker, None, "{loadable}");
        }
        assert!(by_id("b/sharded").gated && !single.gated);
        // GGUF estimate follows the file the selection picks (Q4_K_M, 4.5 bpw).
        assert_eq!(by_id("c/gguf").size, Some(562_500_000));
        assert!(
            blocker("e/big").starts_with("needs ~130 GB as bf16; host has 16 GB"),
            "{}",
            blocker("e/big")
        );
        // 25B parameters are 47 GB as bf16: they fit 48 GB.
        assert!(memory_blocker(Some(25_000_000_000), Some(48 << 30)).is_none());
        assert!(blocker("r/qwen3-big").starts_with("needs ~26 GB as bf16; host has 16 GB"));
        // Spec LMs keep GGUF blocks packed: a 27B Q4_K_M is sized by its ~15 GB file (fits 16 GB).
        assert_eq!(blocker("e/q35-27b-gguf"), "");
        // Qwen-Image rows load (2.1 and 1, diffusers and transformer-only GGUFs, a 2.1 finetune
        // whose GGUF says `qwen_image`, LoRAs of a base), or say why not: Edit conditions on images.
        assert!(blocker("x/qwen-image").starts_with("needs ~38 GB as bf16; host has 16 GB"));
        for loadable in ["q/qwen21", "q/qwen21-gguf", "q/qwen21-turbo", "q/qwen-lora"] {
            assert_eq!(by_id(loadable).blocker, None, "{loadable}");
        }
        assert!(blocker("q/qwen-edit-gguf").contains("image editing"));
        // GPTQ / AWQ int4 load (repacked to int4 group blocks or expanded to f32).
        for loadable in ["f/awq", "g/gptq"] {
            assert_eq!(by_id(loadable).blocker, None, "{loadable}");
        }
        // Same rule as the loader's `prequant::check`: not every compressed-tensors format loads.
        assert!(blocker("m/marlin").starts_with("compressed-tensors format 'marlin-24'"));
        assert!(
            blocker("h/mlx7").starts_with("MLX affine 7-bit with group size 64 is not supported")
        );
        assert_eq!(
            blocker("k/asr"),
            "automatic-speech-recognition model without a config.json (weights only): the architecture cannot be inferred"
        );
        // A 35B-A3B Q4_K_M (experts packed) is sized by its file, not params x 2.
        let b = blocker("c/q35moe-gguf");
        assert!(
            b.contains("weights stay packed") && b.ends_with("host has 16 GB"),
            "{b}"
        );
        assert_eq!(
            blocker("c/mamba-gguf"),
            format!(
                "translation model, llama.cpp GGUF architecture 'mamba' (not supported; {} GGUFs load)",
                gguf_lm_archs().join(", ")
            )
        );
        // Gemma 4 GGUFs load (text + mmproj), tagged image-text-to-text or any-to-any; projector-only
        // repos say why not.
        assert_eq!(blocker("c/gemma4-gguf"), "");
        assert_eq!(blocker("g/gemma-4-E2B-any-GGUF"), "");
        assert_eq!(
            blocker("g/gemma-4-E2B-mmproj"),
            crate::model::hub::MMPROJ_ONLY
        );
        // A config-less llama.cpp T5 GGUF has no loader: the row keeps its reason.
        assert!(blocker("g/t5-llamacpp-gguf").contains("GGUF architecture 't5'"));
        assert_eq!(by_id("n/cls").blocker, None);
        assert_eq!(
            blocker("o/ner"),
            "token-classification model, architecture 'bert' (not supported)"
        );
        assert_eq!(
            blocker("q/bart-nli"),
            "architecture 'BartForSequenceClassification' is not supported"
        );
        assert_eq!(
            blocker("s/hibiki-mt"),
            "translation model, architecture 'hibiki' (not supported)"
        );
        // Classification repos: loadable families and adapters, or a specific reason.
        for loadable in [
            "p/lora-cls",
            "p/lora-lm",
            "s/setfit",
            "t/fasttext",
            "t/gguf-cls",
            "u/modernbert",
            "u/gen",
        ] {
            assert_eq!(by_id(loadable).blocker, None, "{loadable}");
        }
        for (id, why) in [
            (
                "p/lora-head",
                "PEFT adapter on Qwen/Qwen3.5-4B with a custom head (head.pt)",
            ),
            ("t/gguf-lora", "GGUF LoRA adapter (needs its base model)"),
            (
                "u/custom",
                "custom architecture 'FevForDecision' (remote code)",
            ),
            ("u/base", "'BertModel' has no sequence-classification head"),
            ("u/lib", "'catboost' model (not a transformers checkpoint)"),
            ("u/nocfg", "no config.json (custom checkpoint)"),
        ] {
            assert_eq!(blocker(id), why, "{id}");
        }
        assert_eq!(
            blocker("q/inpaint"),
            "StableDiffusionInpaintPipeline: inpainting needs a mask, which no endpoint takes"
        );
        // Past the pipeline gate; only memory (2 bytes/param) stops a 12B FLUX GGUF on 16 GB.
        assert!(blocker("s/flux-gguf").starts_with("needs ~22 GB as bf16; host has 16 GB"));
        assert_eq!(
            blocker("s/sana-lora"),
            "derivative of Efficient-Large-Model/Sana_1600M_1024px_diffusers: only SD 1.x/2.x, SDXL and FLUX.1 LoRAs, finetunes and GGUFs load"
        );
        assert_eq!(
            blocker("u/ckpt-only"),
            "diffusers component unet/ has no safetensors weights"
        );
        // Drafters name their target and load only onto it.
        let dflash = by_id("z-lab/Qwen3-4B-DFlash-b16");
        assert_eq!(
            (dflash.draft_for.as_deref(), &dflash.blocker),
            (Some("Qwen3-4B"), &None)
        );
        assert_eq!(
            by_id("r/Qwen3.5-2B-DSpark").draft_for.as_deref(),
            Some("Qwen/Qwen3.5-2B")
        );
        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            None,
        );
        p.open("Qwen/Qwen3.5-2B");
        p.hub_results = entries.clone();
        p.query = "Qwen3".into();
        p.refilter();
        let row = |id: &str| &p.rows.iter().find(|r| r.entry.id == id).unwrap().entry;
        assert_eq!(
            tail(row("r/Qwen3.5-2B-DSpark")),
            ("draft model for Qwen/Qwen3.5-2B".to_string(), Token::Jade)
        );
        assert_eq!(
            row("z-lab/Qwen3-4B-DFlash-b16").blocker.as_deref(),
            Some("draft model for Qwen3-4B: load Qwen3-4B first")
        );
        // llama.cpp `dflash` GGUFs attach as drafters; fork layouts and split sets of archs no
        // loader runs get a reason naming the architecture.
        let side = by_id("r/Qwen3.5-2B-DSpark-GGUF");
        assert_eq!(
            (side.draft_for.as_deref(), &side.blocker),
            (Some("Qwen3.5-2B"), &None)
        );
        assert!(blocker("r/fork-DFlash-GGUF").contains("fork architecture 'dflash-draft'"));
        let split = by_id("r/smol-split");
        assert_eq!((&split.blocker, split.size), (&None, Some(75_937_500)));
        assert!(
            blocker("r/glm-split").contains("split GGUF of llama.cpp architecture 'glm_moe_dsa'")
        );

        assert!(parse_hub_results("{}", None).is_err());

        let now = chrono::DateTime::parse_from_rfc3339("2025-11-10T10:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert_eq!(relative("2025-11-10T07:00:00.000Z", now), "3h");
        assert_eq!(relative("2025-11-05T10:00:00.000Z", now), "5d");
        assert_eq!(relative("2025-08-01T10:00:00.000Z", now), "2025-08-01");
    }

    /// The Hub listings of the Qwen-Image 2.1 GGUF repos (FP8 / INT8 / MLX / NVFP4 / LoRA siblings
    /// beside the GGUFs, ComfyUI companion files, a diffusers-less `transformer/`) select a
    /// transformer GGUF, which [`crate::runtime::diffusion::load_image`] reads.
    #[test]
    fn qwen_image_gguf_repos_select_a_gguf() {
        let pick = |names: &[&str]| {
            let files: Vec<RepoFile> = names
                .iter()
                .map(|p| RepoFile {
                    path: p.to_string(),
                    size: None,
                })
                .collect();
            crate::model::variant::with_unmeasured_host(|| {
                select_weight_files(&files, None).unwrap()
            })
        };
        let uc = pick(&[
            "README.md",
            "assets/Qwen-Image-2.1-Benchmark.png",
            "qwen-image-2.1-UC-BF16.gguf",
            "qwen-image-2.1-UC-MLX-4bit.safetensors",
            "qwen-image-2.1-UC-MLX-8bit.safetensors",
            "qwen-image-2.1-UC-NVFP4.safetensors",
            "qwen-image-2.1-UC-Q4_0.gguf",
            "qwen-image-2.1-UC-Q4_K_M.gguf",
            "qwen-image-2.1-UC-Q8_0.gguf",
            "qwen-image-2.1-UC-fp8.safetensors",
            "qwen-image-2.1-UC-int8_convrot.safetensors",
            "qwen-image-2.1-uncensored-lora.safetensors",
            "text_encoders/qwen3vl_8b_bf16.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ]);
        assert_eq!(uc, ["qwen-image-2.1-UC-Q4_0.gguf"]);
        let turbo = pick(&[
            "README.md",
            "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors",
            "Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r128.safetensors",
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-Q4_K_M.gguf",
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-Q8_0.gguf",
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-fp8_e4m3fn.safetensors",
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
            "Qwen-Image-2.1-viggle-turbo-v0.3-6step-Q4_K_M.gguf",
            "Qwen-Image-2.1-viggle-turbo-v0.3-6step-Q8_0.gguf",
            "Qwen-Image-2.1-viggle-turbo-v0.3-6step-fp8_e4m3fn.safetensors",
            "Qwen-Image-2.1-viggle-turbo-v0.3-6step-int8_convrot.safetensors",
            "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
            "comfyui/Qwen-Image-2.1-viggle-turbo-t2i.json",
            "peft/adapter_model.safetensors",
            "peft_v0.3/adapter_model.safetensors",
            "scheduler/scheduler_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
        ]);
        // Two releases tie on bits per weight; either is a transformer GGUF (v0.3 is the one run).
        assert!(
            turbo.len() == 1 && turbo[0].ends_with("-6step-Q4_K_M.gguf"),
            "{turbo:?}"
        );
    }

    /// Rows follow the active sort key (Tab cycles), cached first on ties; a response for an
    /// older request never replaces a newer one.
    #[test]
    fn sort_cycle_orders_rows_and_ignores_stale_results() {
        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            None,
        );
        let e = |id: &str, updated: &str, downloads, likes| Entry {
            id: id.into(),
            updated: Some(updated.into()),
            downloads: Some(downloads),
            likes: Some(likes),
            ..Default::default()
        };
        p.query = "m".into();
        let key = p.key();
        p.in_flight.insert(key.clone(), 6);
        let fresh = vec![
            e("a/m-old", "2024-01-01T00:00:00Z", 900, 1),
            e("b/m-new", "2025-01-01T00:00:00Z", 5, 2),
            e("c/m-mid", "2024-06-01T00:00:00Z", 50, 300),
        ];
        p.tx.send((key.clone(), 6, Ok(fresh))).unwrap();
        p.tx.send((
            key.clone(),
            4,
            Ok(vec![e("z/m-stale", "2026-01-01T00:00:00Z", 1, 1)]),
        ))
        .unwrap();
        p.tx.send((key.clone(), 6, Ok(Vec::new()))).unwrap(); // duplicate after the answer: dropped
        p.tick();
        assert!(p.in_flight.is_empty() && p.searches[&key].as_ref().unwrap().len() == 3);
        let ids = |p: &Picker| {
            p.rows
                .iter()
                .map(|r| r.entry.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(p.sort, Sort::Likes);
        assert_eq!(ids(&p), ["c/m-mid", "b/m-new", "a/m-old"]);
        let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        p.handle_key(tab);
        assert_eq!(ids(&p), ["a/m-old", "c/m-mid", "b/m-new"]);
        p.handle_key(tab);
        assert_eq!(ids(&p), ["b/m-new", "c/m-mid", "a/m-old"]);
        p.handle_key(tab);
        assert_eq!(p.sort, Sort::Likes);
    }

    /// A model too large for the host (memory) or the free disk is replaced by its loadable
    /// quantized derivative: the row turns loadable and enter loads the derivative.
    #[test]
    fn too_large_model_loads_its_quantized_variant() {
        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            None,
        );
        let big = |id: &str, size, blocker: Option<&str>| Entry {
            id: id.into(),
            size: Some(size),
            blocker: blocker.map(String::from),
            ..Default::default()
        };
        p.free_disk = Some(10 << 30);
        p.hub_results = vec![
            big("o/huge", 40 << 30, None), // fits memory, not the disk
            big(
                "o/huger",
                1 << 30,
                Some("needs ~102 GB as f32; host has 48 GB"),
            ),
            big("o/small", 1 << 30, None),
        ];
        p.refilter();
        let row = |p: &Picker, id: &str| p.rows.iter().position(|r| r.entry.id == id).unwrap();
        let blocker = |p: &Picker, id: &str| p.rows[row(p, id)].entry.blocker.clone();
        assert!(blocker(&p, "o/huge").unwrap().contains(" of disk; "));
        // No quantized variant listed: the refusal says so.
        p.quants.insert("o/huger".into(), Vec::new());
        p.refilter();
        assert!(blocker(&p, "o/huger")
            .unwrap()
            .ends_with("no loadable quantized variant"));
        // With multiple variants, the smallest is default and 'v' cycles through them.
        p.quants.insert(
            "o/huge".into(),
            vec![
                big("o/huge-Q4", 5 << 30, None),
                big("o/huge-Q8", 8 << 30, None),
            ],
        );
        p.refilter();
        assert_eq!(blocker(&p, "o/huge"), None);
        p.selected = row(&p, "o/huge");
        // Default to smallest:
        assert!(matches!(p.enter(), PickerAction::Load(id) if id == "o/huge-Q4"));
        assert_eq!(p.rows[p.selected].entry.quant_idx, 0);
        let (status, _) = tail(&p.rows[p.selected].entry);
        assert!(status.contains("variant 1 of 2"), "status was: {status}");
        assert!(status.contains("'v' to cycle"), "status was: {status}");

        // Press 'v' to cycle:
        p.handle_key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::NONE));
        assert!(matches!(p.enter(), PickerAction::Load(id) if id == "o/huge-Q8"));
        assert_eq!(p.rows[p.selected].entry.quant_idx, 1);
        let (status2, _) = tail(&p.rows[p.selected].entry);
        assert!(status2.contains("variant 2 of 2"), "status was: {status2}");

        // Press 'v' again to cycle back to 0:
        p.handle_key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::NONE));
        assert!(matches!(p.enter(), PickerAction::Load(id) if id == "o/huge-Q4"));
        assert_eq!(p.rows[p.selected].entry.quant_idx, 0);

        p.selected = row(&p, "o/small");
        assert!(matches!(p.enter(), PickerAction::Load(id) if id == "o/small"));
    }

    #[test]
    fn test_picker_quant_arrow_selection() {
        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            None,
        );
        p.hub_results = vec![Entry {
            id: "org/model".into(),
            quants: vec![
                Entry {
                    id: "org/model/m-Q4_0.gguf".into(),
                    ..Default::default()
                },
                Entry {
                    id: "org/model/m-Q8_0.gguf".into(),
                    ..Default::default()
                },
                Entry {
                    id: "org/model/m-BF16.gguf".into(),
                    ..Default::default()
                },
            ],
            quant_idx: 0,
            via: Some(Box::new(Entry {
                id: "org/model/m-Q4_0.gguf".into(),
                ..Default::default()
            })),
            ..Default::default()
        }];
        p.refilter();
        p.selected = 0;

        // Initial state: quant_idx = 0, via is m-Q4_0.gguf
        assert_eq!(p.rows[0].entry.quant_idx, 0);
        assert_eq!(
            p.rows[0].entry.via.as_ref().unwrap().id,
            "org/model/m-Q4_0.gguf"
        );

        // Right arrow advances to index 1
        p.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(p.rows[0].entry.quant_idx, 1);
        assert_eq!(
            p.rows[0].entry.via.as_ref().unwrap().id,
            "org/model/m-Q8_0.gguf"
        );

        // Right arrow advances to index 2
        p.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(p.rows[0].entry.quant_idx, 2);
        assert_eq!(
            p.rows[0].entry.via.as_ref().unwrap().id,
            "org/model/m-BF16.gguf"
        );

        // Right arrow wraps around to index 0
        p.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(p.rows[0].entry.quant_idx, 0);
        assert_eq!(
            p.rows[0].entry.via.as_ref().unwrap().id,
            "org/model/m-Q4_0.gguf"
        );

        // Left arrow decrements with wrap-around to index 2
        p.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(p.rows[0].entry.quant_idx, 2);
        assert_eq!(
            p.rows[0].entry.via.as_ref().unwrap().id,
            "org/model/m-BF16.gguf"
        );

        // Left arrow decrements to index 1
        p.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(p.rows[0].entry.quant_idx, 1);
        assert_eq!(
            p.rows[0].entry.via.as_ref().unwrap().id,
            "org/model/m-Q8_0.gguf"
        );

        // Enter loads the currently selected quant
        assert!(matches!(p.enter(), PickerAction::Load(id) if id == "org/model/m-Q8_0.gguf"));
    }

    /// The hub counts an EXL3 repo's stored 16-bit words (11B for a 27B model); a derivative of a
    /// known base is sized by the base, so a model the host cannot hold is not offered.
    #[test]
    fn packed_derivative_takes_its_bases_weight_count() {
        let json = r#"[{"id":"x/exl3","config":{"model_type":"qwen3_5","quantization_config":{"quant_method":"exl3"}},"safetensors":{"total":11364805360,"parameters":{"I16":11364805360}},"siblings":[{"rfilename":"model.safetensors"}]}]"#;
        let blocker = |base| {
            parse_hub_rows(json, Some(48 << 30), base).unwrap()[0]
                .blocker
                .clone()
        };
        assert_eq!(blocker(None), None);
        assert!(blocker(Some(27_000_000_000))
            .unwrap()
            .starts_with("needs ~50 GB as bf16"));
    }

    /// The header of the column the rows are sorted by is bold, underlined and cyan; the others are dim.
    #[test]
    fn sorted_column_header_is_highlighted() {
        use ratatui::{backend::TestBackend, style::Modifier, Terminal};
        let _theme = crate::tui::theme::test_guard(crate::tui::theme::Theme {
            truecolor: true,
            mono: false,
        });
        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            None,
        );
        p.hub_results = vec![Entry {
            id: "a/m".into(),
            likes: Some(5),
            downloads: Some(9),
            updated: Some("2025-01-01T00:00:00Z".into()),
            ..Default::default()
        }];
        p.refilter();
        // The first cell holding `sym` is the header's (the hint line below also has ↓).
        let header = |p: &Picker, sym: &str| {
            let mut t = Terminal::new(TestBackend::new(120, 20)).unwrap();
            t.draw(|f| p.render(f, f.area(), "", None)).unwrap();
            let buf = t.backend().buffer().clone();
            buf.content()
                .iter()
                .find(|c| c.symbol() == sym)
                .unwrap()
                .style()
        };
        let hot = |s: ratatui::style::Style| {
            s.fg == Some(theme().color(Token::Brass))
                && s.add_modifier.contains(Modifier::UNDERLINED)
        };
        assert!(hot(header(&p, "♥")) && !hot(header(&p, "↓")));
        p.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert!(hot(header(&p, "↓")) && !hot(header(&p, "♥")));
    }

    /// The active quant shows its predicted share of the roofline, and the row ranked first
    /// carries `★` (the pick under the variant policy).
    #[test]
    fn quant_row_shows_roofline_percent_and_star() {
        use crate::model::variant::{size_header, with_unmeasured_host};
        use ratatui::{backend::TestBackend, Terminal};
        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            None,
        );
        let quant = |id: &str, size| Entry {
            id: id.into(),
            size: Some(size),
            ..Default::default()
        };
        p.hub_results = vec![Entry {
            id: "org/model".into(),
            quants: vec![
                quant("org/model/m-Q4_0.gguf", 1_000_000_000),
                quant("org/model/m-Q8_0.gguf", 2_000_000_000),
            ],
            ..Default::default()
        }];
        p.refilter();
        p.selected = 0;
        p.set_ceiling_gbps(300.0);
        // The saved variant policy orders the quants; the pick is whichever comes first.
        for q in &p.rows[0].entry.quants {
            let fmt = detect_quant_format(Path::new(&q.id)).unwrap();
            p.headers
                .lock()
                .insert(q.id.clone(), Some(Ok(Arc::new(size_header(fmt, q.size)))));
        }
        let tag = |i: usize| {
            detect_quant_format(Path::new(&p.rows[0].entry.quants[i].id))
                .unwrap()
                .name()
                .to_string()
        };
        // Every codec at the default efficiency (0.9): the prediction is 90% of the roofline.
        let text = with_unmeasured_host(|| {
            let mut t = Terminal::new(TestBackend::new(140, 20)).unwrap();
            t.draw(|f| p.render(f, f.area(), "", None)).unwrap();
            let buf = t.backend().buffer().clone();
            (0..buf.area.height)
                .map(|y| {
                    (0..buf.area.width)
                        .map(|x| buf[(x, y)].symbol())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        });
        assert!(text.contains(&format!("[★{}]", tag(0))), "{text}");
        assert!(
            text.contains(&tag(1)) && !text.contains(&format!("★{}", tag(1))),
            "{text}"
        );
        assert!(text.contains("tok/s 90%"), "{text}");
    }

    #[test]
    fn quant_subrows_and_windowed_bar_keep_fit_and_metadata_visible() {
        use crate::model::variant::with_unmeasured_host;
        use ratatui::{backend::TestBackend, Terminal};

        let bartowski_files: String = (0..32)
            .map(|i| {
                let tag = match i % 6 {
                    0 => "Q4_K_M",
                    1 => "Q4_0",
                    2 => "Q5_K_M",
                    3 => "Q6_K",
                    4 => "Q8_0",
                    _ => "BF16",
                };
                format!(r#"{{"rfilename":"Nanbeige4.2-3B-v{i}-{tag}.gguf"}}"#)
            })
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"[
                {{"id":"iamimmanuelraj/Nanbeige4.2-3B-GGUF","pipeline_tag":"text-generation",
                  "likes":14,"downloads":4200,"lastModified":"2025-06-15T00:00:00Z",
                  "gguf":{{"total":3000000000,"architecture":"llama"}},
                  "siblings":[{{"rfilename":"Nanbeige4.2-3B-TQ2_0.gguf"}}]}},
                {{"id":"bartowski/Nanbeige4.2-3B-GGUF","pipeline_tag":"text-generation",
                  "likes":39,"downloads":12000,"lastModified":"2025-07-28T00:00:00Z",
                  "gguf":{{"total":3000000000,"architecture":"llama"}},
                  "siblings":[{bartowski_files}]}}
            ]"#
        );
        let pkgs = parse_hub_rows(&json, Some(48 << 30), Some(3_000_000_000)).unwrap();
        let mut quants: Vec<Entry> = pkgs.into_iter().flat_map(|e| e.quants).collect();
        assert_eq!(quants.len(), 33);
        assert_eq!(quants[0].likes, Some(14));
        assert_eq!(quants[1].likes, Some(39));
        assert_eq!(quants[1].downloads, Some(12000));
        sort_variants(&mut quants, Some(48 << 30), None);

        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            Some(48 << 30),
        );
        p.hub_results = vec![Entry {
            id: "Nanbeige/Nanbeige4.2-3B-Thinking".into(),
            params: Some(3_000_000_000),
            size: Some(6_000_000_000),
            likes: Some(500),
            downloads: Some(85_000),
            updated: Some("2025-05-01T00:00:00Z".into()),
            via: quants.first().cloned().map(Box::new),
            quants,
            quant_idx: 0,
            ..Default::default()
        }];
        p.refilter();
        p.selected = 0;
        p.set_ceiling_gbps(300.0);
        p.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(p.rows[0].entry.quant_idx, 1);

        let text = with_unmeasured_host(|| {
            let mut t = Terminal::new(TestBackend::new(140, 24)).unwrap();
            t.draw(|f| p.render(f, f.area(), "", None)).unwrap();
            let buf = t.backend().buffer().clone();
            (0..buf.area.height)
                .map(|y| {
                    (0..buf.area.width)
                        .map(|x| buf[(x, y)].symbol())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        });
        assert!(text.contains("├─"), "{text}");
        assert!(text.contains("└─"), "{text}");
        assert!(text.contains("bartowski"), "{text}");
        assert!(text.contains("39"), "{text}");
        assert!(text.contains("12K"), "{text}");
        assert!(text.contains("2025-07-28"), "{text}");
        let quant_line = text
            .lines()
            .find(|l| l.contains("Quant: "))
            .expect("Quant: line present");
        assert!(quant_line.contains("tok/s"), "{quant_line}");
        assert!(quant_line.contains("[fit]"), "{quant_line}");
        assert!(quant_line.contains("(2/33)"), "{quant_line}");
        assert!(quant_line.contains("…›"), "{quant_line}");
    }

    /// LTX-Video rows follow the loader's file rule: v0.9.0 GGUFs and `LTXPipeline` repos load; later
    /// versions carry the reason, never a bare "not supported".
    #[test]
    fn ltx_rows_follow_the_loader_rule() {
        let json = r#"[
            {"id":"a/ltx-gguf","pipeline_tag":"text-to-video","gguf":{"total":1900000000,"architecture":"ltxv"},
             "siblings":[{"rfilename":"ltx-video-2b-v0.9-Q4_0.gguf"},{"rfilename":"ltx-video-2b-v0.9.1-q4_0.gguf"},{"rfilename":"LTX-Video-VAE-BF16.safetensors"}]},
            {"id":"b/ltx-13b","pipeline_tag":"text-to-video","gguf":{"total":13000000000,"architecture":"ltxv"},
             "siblings":[{"rfilename":"ltxv-13b-0.9.7-dev-Q4_0.gguf"}]},
            {"id":"c/ltx-diffusers","pipeline_tag":"image-to-video","config":{"diffusers":{"_class_name":"LTXPipeline"}},
             "siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors"}]},
            {"id":"d/ltx-cond","pipeline_tag":"image-to-video","config":{"diffusers":{"_class_name":"LTXConditionPipeline"}},
             "siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors"}]}
        ]"#;
        let entries = parse_hub_results(json, Some(48 << 30)).unwrap();
        let blocker = |id: &str| entries.iter().find(|e| e.id == id).unwrap().blocker.clone();
        assert_eq!(blocker("a/ltx-gguf"), None);
        assert_eq!(blocker("c/ltx-diffusers"), None);
        let b = blocker("b/ltx-13b").unwrap();
        assert!(
            b.contains("ltxv-13b-0.9.7-dev-Q4_0.gguf") && b.contains("v0.9.0"),
            "{b}"
        );
        assert!(blocker("d/ltx-cond")
            .unwrap()
            .contains("LTXConditionPipeline"));
    }

    /// A 35B-A3B Q4_K_M GGUF (experts stay packed in the mapping) loads on a 48 GB host.
    #[test]
    fn packed_moe_gguf_rows_fit_by_file_size() {
        let json = r#"[{"id":"c/q35moe-gguf","pipeline_tag":"text-generation","gguf":{"total":35000000000,"architecture":"qwen35moe"},"siblings":[{"rfilename":"moe-Q4_K_M.gguf"}]}]"#;
        let entries = parse_hub_results(json, Some(48 << 30)).unwrap();
        assert_eq!(entries[0].blocker, None);
    }

    /// Krea 2 pipelines, ComfyUI's split files, and LoRAs / finetunes / GGUFs of a Krea 2 base are
    /// loadable; the ComfyUI repo with only quantized transformers says why it is not.
    #[test]
    fn krea2_rows_are_loadable() {
        let json = r#"[
            {"id":"krea/Krea-2-Turbo","pipeline_tag":"text-to-image","gated":"auto","config":{"diffusers":{"_class_name":"Krea2Pipeline"}},"siblings":[{"rfilename":"model_index.json"},{"rfilename":"transformer/diffusion_pytorch_model.safetensors"}]},
            {"id":"Comfy-Org/Krea-2","tags":["comfyui","base_model:finetune:krea/Krea-2-Raw"],"siblings":[{"rfilename":"diffusion_models/krea2_turbo_bf16.safetensors"},{"rfilename":"text_encoders/qwen3vl_4b_bf16.safetensors"},{"rfilename":"vae/qwen_image_vae.safetensors"}]},
            {"id":"q/krea-quant","pipeline_tag":"text-to-image","tags":["comfyui","base_model:finetune:krea/Krea-2-Turbo"],"siblings":[{"rfilename":"diffusion_models/krea2_turbo_nvfp4.safetensors"},{"rfilename":"text_encoders/qwen3vl_4b_bf16.safetensors"},{"rfilename":"vae/qwen_image_vae.safetensors"}]},
            {"id":"v/krea-gguf","pipeline_tag":"image-to-image","gguf":{"architecture":"krea2"},"siblings":[{"rfilename":"krea2_turbo-Q4_0.gguf"}]},
            {"id":"l/krea-lora","pipeline_tag":"text-to-image","tags":["lora","base_model:adapter:krea/Krea-2-Turbo"],"siblings":[{"rfilename":"lora.safetensors"}]}
        ]"#;
        let entries = parse_hub_results(json, Some(64 << 30)).unwrap();
        let blocker = |id: &str| entries.iter().find(|e| e.id == id).unwrap().blocker.clone();
        for ok in [
            "krea/Krea-2-Turbo",
            "Comfy-Org/Krea-2",
            "v/krea-gguf",
            "l/krea-lora",
        ] {
            assert_eq!(blocker(ok), None, "{ok}");
        }
        assert!(blocker("q/krea-quant").unwrap().contains("only the bf16"));
    }

    #[test]
    fn test_feature_extraction_and_architecture_refusals() {
        // 1. BERT feature-extraction is unblocked
        let bert_json = serde_json::json!({
            "id": "BAAI/bge-small-en-v1.5",
            "pipeline_tag": "feature-extraction",
            "config": { "model_type": "bert" },
            "siblings": [{ "rfilename": "model.safetensors" }]
        });
        assert_eq!(pipeline_blocker(&bert_json), None);

        // 2. Non-BERT feature-extraction is refused
        let llama_embed = serde_json::json!({
            "id": "meta-llama/Llama-Embed-1B",
            "pipeline_tag": "feature-extraction",
            "config": { "model_type": "llama" },
            "siblings": [{ "rfilename": "model.safetensors" }]
        });
        assert!(pipeline_blocker(&llama_embed).is_some());

        // 3. GGUF with deepseek2 architecture is refused by pipeline_blocker
        let deepseek_gguf = serde_json::json!({
            "id": "deepseek-ai/DeepSeek-V2.5-GGUF",
            "pipeline_tag": "text-generation",
            "gguf": { "architecture": "deepseek2" },
            "siblings": [{ "rfilename": "model-Q4_K_M.gguf" }]
        });
        assert_eq!(
            pipeline_blocker(&deepseek_gguf),
            Some("deepseek2: architecture 'deepseek' uses Multi-Head Latent Attention (MLA), which is not implemented".into())
        );
        // 4. Architecture refusals
        assert_eq!(
            crate::arch::unsupported("deepseek_v3").as_deref(),
            Some("architecture 'deepseek' uses Multi-Head Latent Attention (MLA), which is not implemented")
        );
        assert_eq!(
            crate::arch::unsupported("deepseek2").as_deref(),
            Some("architecture 'deepseek' uses Multi-Head Latent Attention (MLA), which is not implemented")
        );
        assert_eq!(
            crate::arch::unsupported("glm_moe_dsa").as_deref(),
            Some("architecture 'glm-dsa' uses Dynamic Sparse Attention (DSA), which is not implemented")
        );
        assert_eq!(
            crate::arch::unsupported("wan").as_deref(),
            Some("text-to-video model, architecture 'wan' (not supported)")
        );

        // 5. bitsandbytes: 4-bit decodes, LLM.int8 is refused
        let bnb = |q: serde_json::Value| {
            crate::model::prequant::scheme_blocker(
                &serde_json::json!({ "quantization_config": q }),
                None,
            )
        };
        assert_eq!(
            bnb(serde_json::json!({ "quant_method": "bitsandbytes" })),
            None
        );
        assert!(
            bnb(serde_json::json!({ "quant_method": "bitsandbytes", "load_in_8bit": true }))
                .is_some()
        );
    }

    #[test]
    fn fitting_quant_clears_parent_blocker_and_plain_r_f_search() {
        let json = r#"[
            {"id":"org/hybrid-20b","pipeline_tag":"text-generation",
             "safetensors":{"total":20000000000,"parameters":{"BF16":20000000000}},
             "config":{"model_type":"qwen3"},
             "siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"},{"rfilename":"m-BF16.gguf"},{"rfilename":"m-Q4_K_M.gguf"}]},
            {"id":"org/too-big-gguf","pipeline_tag":"text-generation",
             "gguf":{"total":70000000000,"architecture":"qwen3"},
             "siblings":[{"rfilename":"m-Q8_0.gguf"},{"rfilename":"m-BF16.gguf"}]}
        ]"#;
        let parsed = parse_hub_results(json, Some(16 << 30)).unwrap();
        let hybrid = parsed.iter().find(|e| e.id == "org/hybrid-20b").unwrap();
        assert_eq!(hybrid.blocker, None);
        assert_eq!(
            hybrid.via.as_ref().unwrap().id,
            "org/hybrid-20b/m-Q4_K_M.gguf"
        );
        let huge = parsed.iter().find(|e| e.id == "org/too-big-gguf").unwrap();
        assert!(huge
            .blocker
            .as_deref()
            .is_some_and(|b| b.starts_with("needs ~")));

        let cache = tempfile::tempdir().unwrap();
        let mut p = Picker::new(
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            &[],
            Some(16 << 30),
        );
        p.free_disk = Some(14 << 30);
        p.hub_results = parsed;
        p.refilter();

        let h_idx = p
            .rows
            .iter()
            .position(|r| r.entry.id == "org/hybrid-20b")
            .unwrap();
        p.selected = h_idx;
        assert_eq!(p.rows[h_idx].entry.blocker, None);
        assert!(matches!(
            p.enter(),
            PickerAction::Load(id) if id == "org/hybrid-20b/m-Q4_K_M.gguf"
        ));

        let b_idx = p
            .rows
            .iter()
            .position(|r| r.entry.id == "org/too-big-gguf")
            .unwrap();
        p.selected = b_idx;
        assert!(p.rows[b_idx]
            .entry
            .blocker
            .as_deref()
            .is_some_and(|b| b.starts_with("needs ~")));
        assert!(matches!(p.enter(), PickerAction::None));

        p.set_query(String::new());
        assert!(matches!(
            p.handle_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE)),
            PickerAction::None
        ));
        assert!(matches!(
            p.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE)),
            PickerAction::None
        ));
        assert_eq!(p.query, "rf");
        assert!(matches!(
            p.handle_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL)),
            PickerAction::Reload("reencode")
        ));
        assert!(matches!(
            p.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL)),
            PickerAction::Reload("variant")
        ));
    }
}
