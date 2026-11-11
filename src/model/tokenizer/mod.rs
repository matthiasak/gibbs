pub mod seq2seq;
use std::collections::HashMap;

/// Byte-level Pair / Byte-level Tokenizer & Detokenizer.
/// Thread-safe (Send + Sync) and zero-allocation friendly.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub inv_vocab: HashMap<u32, String>,
    pub special_tokens: HashMap<String, u32>,
    /// SentencePiece vocab: '▁' (U+2581) stands for a space instead of GPT-2 byte-level 'Ġ'.
    pub spm: bool,
    /// SentencePiece dummy prefix: `encode` prepends '▁' and full-sequence `decode` strips it.
    pub add_space_prefix: bool,
    /// BOS id `encode` prepends; set only when the model declares it adds one.
    pub bos: Option<u32>,
    /// HF `tokenizers` pipeline from tokenizer.json (added tokens, pre-tokenizer, BPE merges).
    /// `encode` uses it when present; GGUF / synthetic vocabs use the greedy matcher.
    pub hf: Option<std::sync::Arc<tokenizers::Tokenizer>>,
    /// Model-declared chat template (tokenizer_config.json / GGUF `tokenizer.chat_template`).
    pub chat_template: Option<ChatTemplateSpec>,
    /// `generation_config.json` `eos_token_id`(s). They end generation in addition to special
    /// tokens (other than tool-call markers): a turn-end special the config leaves out still stops.
    pub stop_ids: Option<Vec<u32>>,
    /// `generation_config.json` (temperature, top_p) for requests that leave them unset;
    /// temperature 0 is greedy (`do_sample` false). `None` keeps the engine defaults.
    pub sampling_defaults: Option<(f32, f32)>,
}

/// A model's own Jinja chat template plus the `bos_token`/`eos_token` strings it may reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateSpec {
    pub template: String,
    pub bos_token: String,
    pub eos_token: String,
}

impl ChatTemplateSpec {
    /// Renders like HF `apply_chat_template(messages, tools=tools, add_generation_prompt=..,
    /// **kwargs)` (`kwargs`: a JSON object such as `{"enable_thinking": false}`, else ignored):
    /// trim/lstrip blocks, Python string methods, `raise_exception`. Messages keep their
    /// structured content parts and tool fields; string content becomes one text part for
    /// templates that only iterate content as parts (e.g. SmolVLM's).
    pub fn render(
        &self,
        messages: &[crate::model::hf_loader::ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        kwargs: &serde_json::Value,
    ) -> Result<String, minijinja::Error> {
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        use minijinja::{Error, ErrorKind};
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        // ponytail: textual check for a content loop without an `is string` branch; parse the
        // template AST (vLLM's content-format detection) if a template slips past it.
        let parts_only = [
            "in message['content']",
            "in message[\"content\"]",
            "in message.content",
        ]
        .iter()
        .any(|p| self.template.contains(p))
            && !self.template.contains("is string");
        let messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let mut v = serde_json::to_value(m).unwrap_or_default();
                if let (true, Some(text)) = (parts_only, m.content.as_str()) {
                    v["content"] = serde_json::json!([{"type": "text", "text": text}]);
                }
                v
            })
            .collect();
        let extra =
            minijinja::Value::from_serialize(kwargs.as_object().cloned().unwrap_or_default());
        env.template_from_str(&self.template)?
            .render(minijinja::context! {
                messages => minijinja::Value::from_serialize(&messages),
                tools => tools.map(minijinja::Value::from_serialize),
                add_generation_prompt => add_generation_prompt,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
                ..extra
            })
    }
}

/// Canonical GPT-2 byte-to-unicode character mapping.
pub const fn byte_to_gpt2_unicode(b: u8) -> char {
    let cp = if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) {
        b as u32
    } else if b <= 32 {
        256 + b as u32
    } else if b >= 127 && b <= 160 {
        256 + 33 + (b - 127) as u32
    } else if b == 173 {
        256 + 67
    } else {
        b as u32
    };
    match char::from_u32(cp) {
        Some(c) => c,
        None => b as char,
    }
}

/// Inverts canonical GPT-2 byte-to-unicode mapping back to original byte.
pub const fn gpt2_unicode_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    if (cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255) {
        Some(cp as u8)
    } else if cp >= 256 && cp <= 288 {
        Some((cp - 256) as u8)
    } else if cp >= 289 && cp <= 322 {
        Some((cp - 162) as u8)
    } else if cp == 323 {
        Some(173)
    } else {
        None
    }
}

/// Whether a HF `normalizer` / `pre_tokenizer` declares the SentencePiece dummy '▁' prefix.
fn declares_space_prefix(v: &serde_json::Value) -> bool {
    match v.get("type").and_then(|t| t.as_str()) {
        Some("Sequence") => ["normalizers", "pretokenizers"]
            .iter()
            .filter_map(|k| v.get(*k)?.as_array())
            .flatten()
            .any(declares_space_prefix),
        Some("Prepend") => v.get("prepend").and_then(|p| p.as_str()) == Some("\u{2581}"),
        Some("Metaspace") => match v.get("prepend_scheme").and_then(|s| s.as_str()) {
            Some(scheme) => scheme != "never",
            None => v
                .get("add_prefix_space")
                .and_then(|b| b.as_bool())
                .unwrap_or(true),
        },
        _ => false,
    }
}

/// BOS declared by a HF `post_processor`: a `TemplateProcessing` (possibly nested in a
/// `Sequence`) whose single-sequence template starts with a special token.
fn template_bos(pp: Option<&serde_json::Value>, vocab: &HashMap<String, u32>) -> Option<u32> {
    let pp = pp?;
    match pp.get("type")?.as_str()? {
        "Sequence" => pp
            .get("processors")?
            .as_array()?
            .iter()
            .find_map(|p| template_bos(Some(p), vocab)),
        "TemplateProcessing" => {
            let name = pp
                .get("single")?
                .get(0)?
                .get("SpecialToken")?
                .get("id")?
                .as_str()?;
            pp.get("special_tokens")
                .and_then(|s| s.get(name)?.get("ids")?.get(0)?.as_u64())
                .map(|id| id as u32)
                .or_else(|| vocab.get(name).copied())
        }
        _ => None,
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::default_bpe()
    }
}

impl Tokenizer {
    /// Creates a production-grade default Byte-Pair / Byte-level Tokenizer
    /// with built-in pre-populated vocabulary covering all 256 individual byte characters,
    /// common English words/subwords, numbers, punctuation, and chat template special tokens.
    pub fn default_bpe() -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        let mut special_tokens = HashMap::new();

        let mut next_id = 0u32;
        let mut add_token =
            |token: &str,
             is_special: bool,
             vocab: &mut HashMap<String, u32>,
             inv_vocab: &mut HashMap<u32, String>,
             special_tokens: &mut HashMap<String, u32>| {
                if let std::collections::hash_map::Entry::Vacant(e) = vocab.entry(token.to_string())
                {
                    let id = next_id;
                    next_id += 1;
                    e.insert(id);
                    inv_vocab.insert(id, token.to_string());
                    if is_special {
                        special_tokens.insert(token.to_string(), id);
                    }
                }
            };

        // 1. Special tokens
        let specials = [
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "<|im_start|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        ];
        for s in &specials {
            add_token(s, true, &mut vocab, &mut inv_vocab, &mut special_tokens);
        }

        // 2. All 256 individual byte characters (ISO-8859-1 / Latin-1 range)
        // Ensures any UTF-8 or raw byte sequence can be losslessly encoded without <unk>.
        for b in 0u8..=255u8 {
            let byte_char = (b as char).to_string();
            add_token(
                &byte_char,
                false,
                &mut vocab,
                &mut inv_vocab,
                &mut special_tokens,
            );
        }

        // 3. Numbers: double digits "00".."99"
        for i in 0..100 {
            let num_str = format!("{:02}", i);
            add_token(
                &num_str,
                false,
                &mut vocab,
                &mut inv_vocab,
                &mut special_tokens,
            );
        }

        // 4. Common English subwords/tokens from specification (with and without leading spaces)
        let common_words = [
            " the",
            " of",
            " and",
            " to",
            " a",
            " in",
            " is",
            " it",
            " you",
            " that",
            " he",
            " was",
            " for",
            " on",
            " are",
            " as",
            " with",
            " his",
            " they",
            " I",
            " at",
            " be",
            " this",
            " have",
            " from",
            " or",
            " one",
            " had",
            " by",
            " word",
            " but",
            " not",
            " what",
            " all",
            " were",
            " we",
            " when",
            " your",
            " can",
            " said",
            " there",
            " use",
            " an",
            " each",
            " which",
            " she",
            " do",
            " how",
            " their",
            " if",
            " will",
            " up",
            " other",
            " about",
            " out",
            " many",
            " then",
            " them",
            " these",
            " so",
            " some",
            " her",
            " would",
            " make",
            " like",
            " him",
            " into",
            " time",
            " has",
            " look",
            " two",
            " more",
            " write",
            " go",
            " see",
            " number",
            " no",
            " way",
            " could",
            " people",
            " my",
            " than",
            " first",
            " water",
            " been",
            " call",
            " who",
            " oil",
            " its",
            " now",
            " find",
            " long",
            " down",
            " day",
            " did",
            " get",
            " come",
            " made",
            " may",
            " part",
            " strawberry",
            " r",
            " letter",
            " count",
        ];

        for w in &common_words {
            add_token(w, false, &mut vocab, &mut inv_vocab, &mut special_tokens);
            let trimmed = w.trim_start();
            if !trimmed.is_empty() {
                add_token(
                    trimmed,
                    false,
                    &mut vocab,
                    &mut inv_vocab,
                    &mut special_tokens,
                );
            }
        }

        // Additional common capitalized words & subwords
        let extra_tokens = [
            "The",
            "And",
            "To",
            "A",
            "In",
            "Is",
            "It",
            "You",
            "That",
            "He",
            "We",
            "What",
            "When",
            "There",
            "Strawberry",
            "Count",
            "Letter",
            "ing",
            " ing",
            "ed",
            " ed",
            "ly",
            " ly",
            "er",
            " er",
            "es",
            " es",
            "tion",
            " tion",
            "ment",
            " ment",
            "ness",
            " ness",
            "able",
            " able",
            "al",
            " al",
            "ic",
            " ic",
            "ous",
            " ous",
            "ive",
            " ive",
            "en",
            " en",
            "re",
            " re",
            "un",
            " un",
            "th",
            "ch",
            "sh",
            "ph",
            "wh",
            "qu",
            "st",
            "ll",
            "ss",
            "ee",
            "oo",
            "tt",
            "ff",
            "rr",
            "==",
            "!=",
            "<=",
            ">=",
            "->",
            "=>",
            "::",
            "...",
            "//",
            "/*",
            "*/",
            "\n\n",
            "  ",
            "    ",
            "        ",
            "\t",
            "http",
            "https",
            "://",
            "www",
            ".com",
            ".org",
            ".io",
            ".rs",
            ".json",
            "user",
            "assistant",
            "system",
            "role",
            "content",
            "message",
            "messages",
            "true",
            "false",
            "null",
            "fn",
            "let",
            "mut",
            "pub",
            "struct",
            "impl",
            "é",
            "è",
            "à",
            "ç",
            "ñ",
            "ü",
            "ö",
            "ä",
            "ô",
            "ê",
            "î",
            "ï",
            "á",
            "í",
            "ó",
            "ú",
        ];

        for t in &extra_tokens {
            add_token(t, false, &mut vocab, &mut inv_vocab, &mut special_tokens);
        }

        Tokenizer {
            vocab,
            inv_vocab,
            special_tokens,
            spm: false,
            add_space_prefix: false,
            bos: None,
            hf: None,
            chat_template: None,
            stop_ids: None,
            sampling_defaults: None,
        }
    }

    /// Parses HuggingFace `tokenizer.json` format.
    pub fn from_json_str(json_str: &str) -> Result<Self, String> {
        let val: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("Invalid JSON: {}", e))?;

        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        let mut special_tokens = HashMap::new();

        // 1. Extract vocabulary from model.vocab or top-level vocab
        let vocab_value = val
            .get("model")
            .and_then(|m| m.get("vocab"))
            .or_else(|| val.get("vocab"));

        if let Some(v) = vocab_value {
            if let Some(map) = v.as_object() {
                for (token, id_val) in map {
                    if let Some(id) = id_val.as_u64() {
                        let id = id as u32;
                        vocab.insert(token.clone(), id);
                        inv_vocab.insert(id, token.clone());
                    }
                }
            } else if let Some(arr) = v.as_array() {
                for (idx, item) in arr.iter().enumerate() {
                    if let Some(token) = item.as_str() {
                        let id = idx as u32;
                        vocab.insert(token.to_string(), id);
                        inv_vocab.insert(id, token.to_string());
                    } else if let Some(pair) = item.as_array() {
                        if let Some(token) = pair.first().and_then(|t| t.as_str()) {
                            let id = idx as u32;
                            vocab.insert(token.to_string(), id);
                            inv_vocab.insert(id, token.to_string());
                        }
                    }
                }
            }
        }

        // 2. Added / special tokens
        if let Some(added) = val.get("added_tokens").and_then(|a| a.as_array()) {
            for item in added {
                if let (Some(id), Some(content)) = (
                    item.get("id").and_then(|i| i.as_u64()),
                    item.get("content").and_then(|c| c.as_str()),
                ) {
                    let id = id as u32;
                    vocab.insert(content.to_string(), id);
                    inv_vocab.insert(id, content.to_string());
                    let is_special = item
                        .get("special")
                        .and_then(|s| s.as_bool())
                        .unwrap_or(true);
                    if is_special {
                        special_tokens.insert(content.to_string(), id);
                    }
                }
            }
        }

        // 3. Check explicit special_tokens array if present
        if let Some(specials) = val.get("special_tokens").and_then(|s| s.as_array()) {
            for item in specials {
                if let Some(content) = item.as_str() {
                    if let Some(&id) = vocab.get(content) {
                        special_tokens.insert(content.to_string(), id);
                    }
                }
            }
        }

        // 4. Identify standard special tokens in vocab if not yet marked
        let common_specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<s>",
            "</s>",
            "<unk>",
            "<pad>",
            "<|endoftext|>",
        ];
        for &s in &common_specials {
            if !special_tokens.contains_key(s) {
                if let Some(&id) = vocab.get(s) {
                    special_tokens.insert(s.to_string(), id);
                }
            }
        }

        if vocab.is_empty() {
            return Err("No vocabulary found in JSON".to_string());
        }

        // 5. SentencePiece-style BPE: '▁' vocab; dummy prefix declared by the normalizer
        //    (`Prepend "▁"`) or a Metaspace pre-tokenizer. A ByteLevel decoder is GPT-2 style even
        //    when added tokens carry '▁' (e.g. "▁<EOT>").
        let byte_level = std::iter::once(&val["decoder"])
            .chain(val["decoder"]["decoders"].as_array().into_iter().flatten())
            .any(|d| d["type"] == "ByteLevel");
        let spm = !byte_level && vocab.keys().any(|k| k.starts_with('\u{2581}'));
        let add_space_prefix = spm
            && ["normalizer", "pre_tokenizer"]
                .iter()
                .any(|k| val.get(*k).is_some_and(declares_space_prefix));

        // 6. BOS only if the post-processor template starts with a special token (e.g. `<s>`).
        //    BPE / Unigram (causal LM) vocabs only: WordPiece `[CLS] … [SEP]` is an encoder
        //    template, not a BOS.
        let bos = matches!(val["model"]["type"].as_str(), Some("BPE" | "Unigram"))
            .then(|| template_bos(val.get("post_processor"), &vocab))
            .flatten();

        Ok(Tokenizer {
            vocab,
            inv_vocab,
            special_tokens,
            spm,
            add_space_prefix,
            bos,
            // Partial / hand-written JSON (no `model.type`) keeps the greedy matcher.
            hf: match json_str.parse::<tokenizers::Tokenizer>() {
                Ok(t) => Some(std::sync::Arc::new(t)),
                Err(e) => {
                    tracing::warn!("tokenizer.json not HF-loadable ({e}); greedy vocab matching");
                    None
                }
            },
            chat_template: None,
            stop_ids: None,
            sampling_defaults: None,
        })
    }

    /// Builds a tokenizer directly from an ordered sequence of vocabulary tokens (e.g. from GGUF metadata).
    pub fn from_tokens(tokens: Vec<String>) -> Self {
        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut inv_vocab = HashMap::with_capacity(tokens.len());
        let mut special_tokens = HashMap::new();

        for (idx, token) in tokens.into_iter().enumerate() {
            let id = idx as u32;
            let is_special = (token.starts_with("<|") && token.ends_with("|>"))
                || token == "<s>"
                || token == "</s>"
                || token == "<unk>"
                || token == "<pad>";
            if is_special {
                special_tokens.insert(token.clone(), id);
            }
            vocab.insert(token.clone(), id);
            inv_vocab.insert(id, token);
        }

        let spm = vocab.keys().any(|k: &String| k.starts_with('\u{2581}'));
        Self {
            vocab,
            inv_vocab,
            special_tokens,
            spm,
            // Default for SentencePiece vocabs; GGUF metadata may override.
            add_space_prefix: spm,
            bos: None,
            hf: None,
            chat_template: None,
            stop_ids: None,
            sampling_defaults: None,
        }
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    #[inline]
    pub fn is_special(&self, token: u32) -> bool {
        if let Some(tok_str) = self.inv_vocab.get(&token) {
            self.special_tokens.contains_key(tok_str)
                || (tok_str.starts_with("<|") && tok_str.ends_with("|>"))
        } else {
            false
        }
    }

    /// The text of a tool-call marker token (`[TOOL_CALLS]`, `<|python_tag|>`, ...): special
    /// tokens that open or delimit a tool call rather than end the reply.
    pub fn tool_marker(&self, token: u32) -> Option<&'static str> {
        const TOOL_MARKERS: [&str; 4] = ["[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "<|python_tag|>"];
        let s = self.inv_vocab.get(&token)?;
        TOOL_MARKERS.into_iter().find(|m| m == s)
    }

    /// Whether sampling `token` ends a sequence: a declared stop id, or a special token other than
    /// a tool-call marker.
    pub fn ends_generation(&self, token: u32) -> bool {
        self.stop_ids
            .as_ref()
            .is_some_and(|ids| ids.contains(&token))
            || (self.is_special(token) && self.tool_marker(token).is_none())
    }

    /// Encodes `text` like HF `tokenizer(text)`: tokenizer.json pipelines run through HF
    /// `tokenizers` (special tokens split out first, pre-tokenizer, BPE merges). Prepends the
    /// model-declared BOS unless the text already starts with it (chat templates that render
    /// `bos_token`), so templated prompts get exactly one.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let hf_ids = match self.hf.as_ref().map(|hf| hf.encode_fast(text, false)) {
            Some(Ok(enc)) => Some(enc.get_ids().to_vec()),
            Some(Err(e)) => {
                tracing::warn!("HF tokenizer encode failed ({e}); greedy vocab matching");
                None
            }
            None => None,
        };
        let mut ids = hf_ids.unwrap_or_else(|| self.encode_greedy(text));
        if let Some(bos) = self.bos {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        ids
    }
    /// Encodes a mid-stream continuation fragment without prepending BOS or SentencePiece leading `▁`.
    pub fn encode_fragment(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        if !self.add_space_prefix {
            if let Some(Ok(enc)) = self.hf.as_ref().map(|hf| hf.encode_fast(text, false)) {
                return enc.get_ids().to_vec();
            }
        }
        self.encode_greedy_opts(text, false)
    }

    /// Greedy longest-match encoding against vocabulary, falling back to raw byte tokens
    /// (GGUF / synthetic vocabs). SentencePiece vocabs see spaces as '▁' (plus the declared
    /// dummy prefix), like HF `Metaspace`.
    fn encode_greedy(&self, text: &str) -> Vec<u32> {
        self.encode_greedy_opts(text, self.add_space_prefix)
    }

    fn encode_greedy_opts(&self, text: &str, add_prefix: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();
        let spm_text;
        let text = if self.spm && !text.is_empty() {
            let prefix = if add_prefix { "\u{2581}" } else { "" };
            spm_text = format!("{prefix}{}", text.replace(' ', "\u{2581}"));
            spm_text.as_str()
        } else {
            text
        };
        let bytes = text.as_bytes();
        let mut i = 0;

        // Determine max token byte length in vocab to bound candidate window
        let max_len = self.vocab.keys().map(|k| k.len()).max().unwrap_or(1);

        // GPT-2 byte-level vocab ('Ġ' = space). SentencePiece vocabs may contain a literal 'Ġ'.
        let has_gpt2_space = !self.spm
            && (self.vocab.contains_key("Ġ") || self.vocab.keys().any(|k| k.starts_with('Ġ')));

        while i < bytes.len() {
            let mut matched = false;
            let limit = (i + max_len).min(bytes.len());

            // Greedy longest match: test substrings from longest to shortest
            for end in (i + 1..=limit).rev() {
                if text.is_char_boundary(i) && text.is_char_boundary(end) {
                    let slice = &text[i..end];

                    // 1. Direct vocabulary match
                    if let Some(&id) = self.vocab.get(slice) {
                        tokens.push(id);
                        i = end;
                        matched = true;
                        break;
                    }

                    // 2. GPT-2 space prefix handling (' ' -> 'Ġ')
                    if has_gpt2_space && slice.starts_with(' ') {
                        let gpt2_token = format!("Ġ{}", &slice[1..]);
                        if let Some(&id) = self.vocab.get(&gpt2_token) {
                            tokens.push(id);
                            i = end;
                            matched = true;
                            break;
                        }
                    }

                    // 3. If GPT-2 vocabulary and slice contains multi-byte or non-ASCII chars
                    if has_gpt2_space && !slice.is_ascii() {
                        let gpt2_token: String = slice.bytes().map(byte_to_gpt2_unicode).collect();
                        if let Some(&id) = self.vocab.get(&gpt2_token) {
                            tokens.push(id);
                            i = end;
                            matched = true;
                            break;
                        }
                    }
                }
            }

            if !matched {
                // Byte fallback: SentencePiece `<0xXX>` pieces; otherwise the GPT-2 byte char,
                // the raw byte char, then `<0xXX>`.
                let b = bytes[i];
                let hex = || self.vocab.get(&format!("<0x{:02X}>", b));
                let id = if self.spm {
                    hex()
                } else {
                    self.vocab
                        .get(&byte_to_gpt2_unicode(b).to_string())
                        .or_else(|| self.vocab.get(&(b as char).to_string()))
                        .or_else(hex)
                };
                if let Some(&id) = id.or_else(|| self.special_tokens.get("<unk>")) {
                    tokens.push(id);
                }
                i += 1;
            }
        }

        tokens
    }

    /// Decodes a single token ID into its string chunk (ideal for SSE delta streaming).
    pub fn decode_token(&self, token: u32) -> String {
        if self.is_special(token) {
            return String::new();
        }

        if let Some(s) = self.inv_vocab.get(&token) {
            if s.starts_with("<|") && s.ends_with("|>") {
                return String::new();
            }
            if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&s[3..5], 16) {
                    if let Ok(ch_str) = String::from_utf8(vec![byte_val]) {
                        return ch_str;
                    } else {
                        return (byte_val as char).to_string();
                    }
                }
            }
            // Invert canonical GPT-2 byte-to-unicode mapping
            let mut bytes_vec = Vec::with_capacity(s.len());
            let mut can_convert = true;
            for c in s.chars() {
                if let Some(b) = gpt2_unicode_to_byte(c) {
                    bytes_vec.push(b);
                } else if c == ' ' || c == '\u{2581}' || c == '\u{00A0}' {
                    bytes_vec.push(b' ');
                } else if c == '\n' {
                    bytes_vec.push(b'\n');
                } else if c == '\t' {
                    bytes_vec.push(b'\t');
                } else {
                    can_convert = false;
                    break;
                }
            }

            if can_convert && !bytes_vec.is_empty() {
                if let Ok(utf8_str) = String::from_utf8(bytes_vec) {
                    return utf8_str;
                }
            }

            // Fallback direct replacements
            s.replace('Ġ', " ")
                .replace('Ċ', "\n")
                .replace('ĉ', "\t")
                .replace('\u{2581}', " ")
        } else if token < 256 {
            let b = token as u8;
            if b.is_ascii_graphic() || b == b' ' || b == b'\n' || b == b'\t' {
                (b as char).to_string()
            } else {
                format!("<0x{:02X}>", b)
            }
        } else {
            // Fallback non-empty representation for out-of-vocab IDs rather than silent emptiness
            format!(" <tok_{}>", token)
        }
    }

    /// Appends a token's raw bytes for full-sequence decoding; `false` if it has no byte form.
    pub fn append_token_bytes(&self, token: u32, buf: &mut Vec<u8>) -> bool {
        if self.is_special(token) {
            return true;
        }
        let Some(s) = self.inv_vocab.get(&token) else {
            if token < 256 {
                buf.push(token as u8);
                return true;
            }
            return false;
        };
        if s.starts_with("<|") && s.ends_with("|>") {
            return true;
        }
        if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
            if let Ok(byte_val) = u8::from_str_radix(&s[3..5], 16) {
                buf.push(byte_val);
                return true;
            }
        }
        if self.spm {
            // SentencePiece pieces are plain UTF-8 with '▁' for spaces.
            for c in s.chars() {
                let c = if c == '\u{2581}' { ' ' } else { c };
                buf.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
            }
            return true;
        }
        for c in s.chars() {
            if let Some(b) = gpt2_unicode_to_byte(c) {
                buf.push(b);
            } else if c == ' ' || c == '\u{2581}' || c == '\u{00A0}' {
                buf.push(b' ');
            } else if c == '\n' {
                buf.push(b'\n');
            } else if c == '\t' {
                buf.push(b'\t');
            } else if (c as u32) < 256 {
                buf.push(c as u8);
            } else {
                return false;
            }
        }
        true
    }

    /// Maps tokens to strings and concatenates them into clean human-readable text.
    /// A full SentencePiece sequence (starting at BOS) drops its dummy-prefix space, like HF /
    /// Metaspace detokenizers; continuation slices keep leading spaces, matching `decode_token`.
    pub fn decode(&self, tokens: &[u32]) -> String {
        // First try collective byte-level decoding across all tokens
        let mut byte_stream = Vec::with_capacity(tokens.len() * 4);
        if tokens
            .iter()
            .all(|&t| self.append_token_bytes(t, &mut byte_stream))
        {
            let full_sequence = self.bos.is_some() && tokens.first() == self.bos.as_ref();
            let start = (self.spm
                && self.add_space_prefix
                && full_sequence
                && byte_stream.first() == Some(&b' ')) as usize;
            if let Ok(s) = std::str::from_utf8(&byte_stream[start..]) {
                return s.to_string();
            }
        }

        // Fallback: token-by-token decoding
        let mut out = String::with_capacity(tokens.len() * 4);
        for &token in tokens {
            out.push_str(&self.decode_token(token));
        }
        out
    }

    /// Formats role/content pairs with fallback chat template `name` ([`crate::arch::templates`]).
    pub fn apply_chat_template_with_style(&self, messages: &[(&str, &str)], name: &str) -> String {
        let messages: Vec<_> = messages
            .iter()
            .map(|(role, content)| crate::model::hf_loader::ChatMessage::new(*role, *content))
            .collect();
        render_fallback(&messages, name)
    }

    /// The fallback template for a tokenizer without its own, detected by vocabulary: `chatml`
    /// if `<|im_start|>`, `header` if `<|start_header_id|>`, `turn` if `<start_of_turn>`, else
    /// `plain`.
    pub fn fallback_template(&self) -> &'static str {
        if self.vocab.contains_key("<|im_start|>") {
            "chatml"
        } else if self.vocab.contains_key("<|start_header_id|>") {
            "header"
        } else if self.vocab.contains_key("<start_of_turn>") {
            "turn"
        } else {
            "plain"
        }
    }

    /// Formats role/content pairs with the model's own chat template (HF
    /// `apply_chat_template(add_generation_prompt=True)`). Models without one fall back to
    /// [`Self::fallback_template`].
    pub fn apply_chat_template(&self, messages: &[(&str, &str)]) -> String {
        let messages: Vec<_> = messages
            .iter()
            .map(|(role, content)| crate::model::hf_loader::ChatMessage::new(*role, *content))
            .collect();
        self.apply_chat_template_messages(&messages, None)
    }

    /// Attaches the chat template and `bos_token`/`eos_token` declared by HF `tokenizer_config.json`.
    /// `jinja` is a standalone `chat_template.jinja`; like transformers, it wins over the config's.
    pub fn set_tokenizer_config(
        &mut self,
        json: &str,
        jinja: Option<String>,
    ) -> Result<(), String> {
        let cfg: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        // Special tokens are plain strings or AddedToken objects `{"content": ...}`.
        let token = |key: &str| {
            let v = &cfg[key];
            v.as_str()
                .or_else(|| v["content"].as_str())
                .unwrap_or_default()
                .to_string()
        };
        // `chat_template` is a string, or a list of named templates of which "default" is used.
        let template = jinja.or_else(|| match &cfg["chat_template"] {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(named) => named
                .iter()
                .find(|t| t["name"] == "default")
                .and_then(|t| t["template"].as_str())
                .map(String::from),
            _ => None,
        });
        self.chat_template = template.map(|template| ChatTemplateSpec {
            template,
            bos_token: token("bos_token"),
            eos_token: token("eos_token"),
        });
        Ok(())
    }

    /// [`Self::apply_chat_template`] for structured messages (content parts, tool calls) and the
    /// request's `tools`. Built-in fallback styles see only each message's text.
    pub fn apply_chat_template_messages(
        &self,
        messages: &[crate::model::hf_loader::ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> String {
        if let Some(spec) = &self.chat_template {
            match spec.render(messages, tools, true, &serde_json::Value::Null) {
                Ok(prompt) => return prompt,
                Err(e) => tracing::warn!("model chat_template failed ({e}); built-in template"),
            }
        }
        let texts: Vec<_> = messages
            .iter()
            .map(|m| crate::model::hf_loader::ChatMessage::new(m.role.as_str(), m.text()))
            .collect();
        render_fallback(&texts, self.fallback_template())
    }
}

/// Renders `messages` with embedded fallback template `name` (`arch/templates/<name>.jinja`).
fn render_fallback(messages: &[crate::model::hf_loader::ChatMessage], name: &str) -> String {
    let template = crate::arch::templates::template(name)
        .unwrap_or_else(|| panic!("no fallback chat template '{name}'"));
    ChatTemplateSpec {
        template: template.into(),
        bos_token: String::new(),
        eos_token: String::new(),
    }
    .render(messages, None, true, &serde_json::Value::Null)
    .expect("embedded fallback chat templates render")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_default_vocab_coverage() {
        let tokenizer = Tokenizer::default_bpe();

        // Check special tokens
        assert!(tokenizer.special_tokens.contains_key("<|im_start|>"));
        assert!(tokenizer.special_tokens.contains_key("<|im_end|>"));
        assert!(tokenizer.special_tokens.contains_key("<unk>"));
        assert!(tokenizer.special_tokens.contains_key("<s>"));
        assert!(tokenizer.special_tokens.contains_key("</s>"));

        // Check 256 byte tokens present
        for b in 0u8..=255u8 {
            let s = (b as char).to_string();
            assert!(tokenizer.vocab.contains_key(&s), "Missing byte char: {}", b);
        }

        // Check specific words
        assert!(tokenizer.vocab.contains_key(" strawberry"));
        assert!(tokenizer.vocab.contains_key("strawberry"));
        assert!(tokenizer.vocab.contains_key(" the"));
        assert!(tokenizer.vocab.contains_key(" count"));
        assert!(tokenizer.vocab.contains_key(" letter"));
        assert!(tokenizer.vocab.contains_key(" r"));
    }

    #[test]
    fn test_tokenizer_roundtrip_english_and_punctuation() {
        let tokenizer = Tokenizer::default_bpe();

        let sample = "How many 'r's are in the word strawberry? Count: 3.";
        let encoded = tokenizer.encode(sample);
        assert!(!encoded.is_empty());
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, sample);
    }

    #[test]
    fn test_tokenizer_roundtrip_utf8_accents() {
        let tokenizer = Tokenizer::default_bpe();

        let sample = "Café with résumé test 1234!";
        let encoded = tokenizer.encode(sample);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, sample);
    }

    #[test]
    fn test_tokenizer_byte_fallback_handling() {
        let tokenizer = Tokenizer::default_bpe();

        let sample = "🦀";
        let encoded = tokenizer.encode(sample);
        assert_eq!(encoded.len(), 4); // 4 UTF-8 bytes fallback
        for &t in &encoded {
            assert!(tokenizer.inv_vocab.contains_key(&t));
        }
    }
    #[test]
    fn test_decode_token_streaming() {
        let tokenizer = Tokenizer::default_bpe();

        let text = "strawberry";
        let tokens = tokenizer.encode(text);
        assert_eq!(tokens.len(), 1);
        let chunk = tokenizer.decode_token(tokens[0]);
        assert_eq!(chunk, "strawberry");
    }

    #[test]
    fn test_apply_chat_template() {
        let tokenizer = Tokenizer::default_bpe();

        let messages = [("system", "You are an assistant."), ("user", "Hello!")];
        let prompt = tokenizer.apply_chat_template(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are an assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_from_json_str() {
        let json_data = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {
                    "hello": 0,
                    " world": 1,
                    "<unk>": 2
                }
            },
            "added_tokens": [
                {
                    "id": 3,
                    "content": "<|im_start|>",
                    "special": true
                }
            ]
        }"#;

        let tokenizer = Tokenizer::from_json_str(json_data).unwrap();
        assert_eq!(tokenizer.vocab.get("hello"), Some(&0));
        assert_eq!(tokenizer.vocab.get(" world"), Some(&1));
        assert_eq!(tokenizer.special_tokens.get("<|im_start|>"), Some(&3));

        let encoded = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_pythia_bpe_space_prefixes_and_decode() {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        let special_tokens = HashMap::new();

        // Populate byte tokens 0..=255 with GPT-2 canonical unicode
        for b in 0u8..=255u8 {
            let s = byte_to_gpt2_unicode(b).to_string();
            let id = b as u32;
            vocab.insert(s.clone(), id);
            inv_vocab.insert(id, s);
        }

        let words = ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"];
        let mut next_id = 256u32;
        for &w in &words {
            vocab.insert(w.to_string(), next_id);
            inv_vocab.insert(next_id, w.to_string());
            next_id += 1;
        }

        let tokenizer = Tokenizer {
            vocab,
            inv_vocab,
            special_tokens,
            spm: false,
            add_space_prefix: false,
            bos: None,
            hf: None,
            chat_template: None,
            stop_ids: None,
            sampling_defaults: None,
        };

        let sample = "The capital of France is";
        let tokens = tokenizer.encode(sample);

        // Verify tokens matched the BPE prefixed words
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokenizer.inv_vocab.get(&tokens[0]).unwrap(), "The");
        assert_eq!(tokenizer.inv_vocab.get(&tokens[1]).unwrap(), "Ġcapital");
        assert_eq!(tokenizer.inv_vocab.get(&tokens[2]).unwrap(), "Ġof");
        assert_eq!(tokenizer.inv_vocab.get(&tokens[3]).unwrap(), "ĠFrance");
        assert_eq!(tokenizer.inv_vocab.get(&tokens[4]).unwrap(), "Ġis");

        // Verify decode reconstructs clean text
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, sample);
    }

    #[test]
    fn test_gpt2_byte_fallback_and_decode() {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        let special_tokens = HashMap::new();

        for b in 0u8..=255u8 {
            let s = byte_to_gpt2_unicode(b).to_string();
            let id = b as u32;
            vocab.insert(s.clone(), id);
            inv_vocab.insert(id, s);
        }

        let tokenizer = Tokenizer {
            vocab,
            inv_vocab,
            special_tokens,
            spm: false,
            add_space_prefix: false,
            bos: None,
            hf: None,
            chat_template: None,
            stop_ids: None,
            sampling_defaults: None,
        };

        let sample = "Hello\tWorld\n123!";
        let tokens = tokenizer.encode(sample);
        assert!(!tokens.is_empty());
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, sample);
    }

    #[test]
    fn test_chat_template_variants() {
        let tokenizer = Tokenizer::default_bpe();
        let messages = [("system", "You are an assistant."), ("user", "Hello!")];

        // ChatML
        let chatml = tokenizer.apply_chat_template_with_style(&messages, "chatml");
        assert_eq!(
            chatml,
            "<|im_start|>system\nYou are an assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
        // Default matches ChatML
        assert_eq!(tokenizer.apply_chat_template(&messages), chatml);

        // Llama 3
        let llama3 = tokenizer.apply_chat_template_with_style(&messages, "header");
        assert_eq!(
            llama3,
            "<|start_header_id|>system<|end_header_id|>\n\nYou are an assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        // Standard
        let standard = tokenizer.apply_chat_template_with_style(&messages, "inst");
        assert_eq!(
            standard,
            "System: You are an assistant.\nUser: Hello!\nAssistant: "
        );

        // Gemma
        let gemma = tokenizer.apply_chat_template_with_style(&messages, "turn");
        assert_eq!(
            gemma,
            "<start_of_turn>system\nYou are an assistant.<end_of_turn>\n<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn test_control_tokens_stripped_in_decode() {
        let tokenizer = Tokenizer::default_bpe();
        let im_end_id = *tokenizer.special_tokens.get("<|im_end|>").unwrap();
        assert_eq!(tokenizer.decode_token(im_end_id), "");
        assert_eq!(tokenizer.decode(&[im_end_id]), "");
    }

    /// A template that only iterates content parts (SmolVLM's) still renders string content;
    /// templates with an `is string` branch keep the string.
    #[test]
    fn test_string_content_renders_through_parts_only_template() {
        let spec = |template: &str| ChatTemplateSpec {
            template: template.into(),
            bos_token: String::new(),
            eos_token: String::new(),
        };
        let msgs = [crate::model::hf_loader::ChatMessage::new(
            "user", "Hi there",
        )];
        let smol = spec("{% for message in messages %}{{ message['role'] }}: {% for line in message['content'] %}{% if line['type'] == 'text' %}{{ line['text'] }}{% elif line['type'] == 'image' %}<image>{% endif %}{% endfor %}\n{% endfor %}");
        let null = serde_json::Value::Null;
        assert_eq!(
            smol.render(&msgs, None, true, &null).unwrap(),
            "user: Hi there"
        );
        let both = spec("{% for message in messages %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for p in message['content'] %}[{{ p['text'] }}]{% endfor %}{% endif %}{% endfor %}");
        assert_eq!(both.render(&msgs, None, true, &null).unwrap(), "Hi there");
    }

    /// Edge cases of the fallback templates (case-insensitive roles, empty content, template
    /// syntax in content); outputs are those of the hand-written formatter the files replaced.
    #[test]
    fn fallback_template_edge_cases() {
        let tok = Tokenizer::default_bpe();
        let m = [("tool", "x {{ y }} %}"), ("ASSISTANT", ""), ("System", "s")];
        let cases = [
            ("chatml", "<|im_start|>tool\nx {{ y }} %}<|im_end|>\n<|im_start|>ASSISTANT\n<|im_end|>\n<|im_start|>System\ns<|im_end|>\n<|im_start|>assistant\n"),
            ("header", "<|start_header_id|>tool<|end_header_id|>\n\nx {{ y }} %}<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>\n\n<|eot_id|><|start_header_id|>System<|end_header_id|>\n\ns<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
            ("turn", "<start_of_turn>tool\nx {{ y }} %}<end_of_turn>\n<start_of_turn>model\n<end_of_turn>\n<start_of_turn>System\ns<end_of_turn>\n<start_of_turn>model\n"),
            ("inst", "tool: x {{ y }} %}\nAssistant: \nSystem: s\nAssistant: "),
            ("plain", "x {{ y }} %}\n\ns"),
        ];
        for (name, want) in cases {
            assert_eq!(tok.apply_chat_template_with_style(&m, name), want, "{name}");
        }
        let empty_first = [("user", ""), ("user", "a"), ("user", "b")];
        assert_eq!(
            tok.apply_chat_template_with_style(&empty_first, "plain"),
            "a\nb"
        );
        assert_eq!(tok.apply_chat_template_with_style(&[], "plain"), "");
    }
}

/// GPT-4-style pre-tokenizer split (Qwen2 tokenizers).
const BPE_SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A Qwen2-style byte-level BPE tokenizer built from `<dir>/{vocab.json, merges.txt,
/// added_tokens.json}` (for tokenizer folders that ship no tokenizer.json): NFC, GPT-4-style split,
/// byte-level BPE, added tokens split out of the text at their stored ids.
pub fn byte_bpe_from_vocab_merges(dir: &std::path::Path) -> Result<tokenizers::Tokenizer, String> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::NFC;
    use tokenizers::pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    };
    use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};
    let err = |e: tokenizers::Error| format!("{}: {e}", dir.display());
    let path = |n: &str| dir.join(n).to_string_lossy().into_owned();
    let bpe = BPE::from_file(&path("vocab.json"), &path("merges.txt"))
        .build()
        .map_err(err)?;
    let mut tok = Tokenizer::new(bpe);
    let split = Split::new(
        SplitPattern::Regex(BPE_SPLIT_PATTERN.into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(err)?;
    tok.with_normalizer(Some(NFC));
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::from(split),
        PreTokenizerWrapper::from(ByteLevel::new(false, false, false)),
    ])));
    let added_path = dir.join("added_tokens.json");
    let raw = std::fs::read_to_string(&added_path)
        .map_err(|e| format!("{}: {e}", added_path.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", added_path.display()))?;
    let mut added: Vec<(u32, &str)> = Vec::with_capacity(map.len());
    for (content, id) in &map {
        let id = id
            .as_u64()
            .ok_or_else(|| format!("{}: id of {content} is not a number", added_path.display()))?;
        added.push((id as u32, content.as_str()));
    }
    // Added tokens take consecutive ids after the vocabulary, in the order they are added.
    added.sort();
    // ponytail: every added token is registered as special (tokenizer_config.json marks a few, e.g.
    // `<tool_call>`, non-special); matching in the text is identical, only decoding differs.
    let tokens: Vec<AddedToken> = added
        .iter()
        .map(|(_, c)| AddedToken::from(*c, true))
        .collect();
    tok.add_special_tokens(&tokens);
    for (id, c) in &added {
        if tok.token_to_id(c) != Some(*id) {
            return Err(format!(
                "{}: added token {c} does not land on id {id} (vocab.json has gaps?)",
                dir.display()
            ));
        }
    }
    Ok(tok)
}

#[cfg(test)]
mod byte_bpe_tests {
    use super::*;

    #[test]
    fn tokenizer_is_built_from_vocab_and_merges_and_splits_out_special_tokens() {
        let dir = std::env::temp_dir().join(format!("gibbs-qwen1-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.json"), r#"{"h":0,"i":1,"Ġ":2,"hi":3}"#).unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\nh i\n").unwrap();
        std::fs::write(
            dir.join("added_tokens.json"),
            r#"{"<|im_end|>":5,"<|im_start|>":4}"#,
        )
        .unwrap();
        let tok = byte_bpe_from_vocab_merges(&dir).unwrap();
        let ids = |s: &str| tok.encode(s, false).unwrap().get_ids().to_vec();
        assert_eq!(ids("hi"), [3]);
        assert_eq!(ids("hi hi"), [3, 2, 3]);
        assert_eq!(ids("<|im_start|>hi<|im_end|>"), [4, 3, 5]);
        // As the diffusion runtime loads it: serialized, then read as a tokenizer.json.
        let json = tok.to_string(false).unwrap();
        let t = crate::model::hf_loader::tokenizer_from_files(&json, &|n| {
            std::fs::read(dir.join(n)).ok()
        })
        .unwrap();
        assert_eq!(t.encode("<|im_start|>hi hi<|im_end|>"), [4, 3, 2, 3, 5]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `GIBBS_QWEN1_TOKENIZER`: Qwen-Image's `tokenizer/` files (vocab.json, merges.txt,
    /// added_tokens.json) next to `ref_tokenizer.json` (Qwen2.5-VL's tokenizer.json): the tokenizer
    /// built from the former must give the latter's ids, and the template head must be 34 tokens.
    #[test]
    fn real_tokenizer_matches_the_reference_tokenizer_json() {
        let Some(dir) = std::env::var_os("GIBBS_QWEN1_TOKENIZER").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP: GIBBS_QWEN1_TOKENIZER not set");
            return;
        };
        let built = byte_bpe_from_vocab_merges(&dir).unwrap();
        let reference = tokenizers::Tokenizer::from_file(dir.join("ref_tokenizer.json")).unwrap();
        let rt =
            crate::model::hf_loader::tokenizer_from_files(&built.to_string(false).unwrap(), &|n| {
                std::fs::read(dir.join(n)).ok()
            })
            .unwrap();
        let ids =
            |t: &tokenizers::Tokenizer, s: &str| t.encode(s, false).unwrap().get_ids().to_vec();
        for s in [
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n",
            "A cat 🐱 on 123456 books, 日本語のテキスト \n\n  indented\ttabs  ",
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
            "don't won't I'LL we've; \"quoted\" (x) [y] ...",
            "<|endoftext|>x<|vision_start|><|image_pad|>",
            " ",
        ] {
            assert_eq!(ids(&built, s), ids(&reference, s), "{s:?}");
            // The form the diffusion runtime loads: serialized, read back as the crate tokenizer.
            assert_eq!(rt.encode(s), ids(&reference, s), "runtime form: {s:?}");
        }
        assert_eq!(ids(&built, "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n").len(),  34);
    }
}
