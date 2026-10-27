//! Encoder-decoder translation on real weights from the shared `disk/` cache, driven the way the
//! scheduler drives a native model: chat template (last user message), byte encode, `prepare`,
//! then `forward` + sampling until the tokenizer says the generation ended.

use gibbs_engine::hardware::{Device, PlacementMode};
use gibbs_engine::model::hub::HuggingFaceHub;
use gibbs_engine::model::native::{NativeLm, PrepareCtx};
use gibbs_engine::runtime::lm::sample_logits;
use std::sync::Arc;

fn model(repo: &str) -> Option<Arc<dyn NativeLm>> {
    let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(repo);
    if art.weights_path.is_none() {
        eprintln!("SKIP: {repo} not in disk/ (gibbs --model {repo} --storage disk)");
        return None;
    }
    let reg = gibbs_engine::arch::registry().expect("registry");
    Some(
        gibbs_engine::runtime::encdec::load(&art, reg, None, PlacementMode::Pin(Device::Cpu))
            .expect("seq2seq model loads"),
    )
}

/// The reply to one chat turn at temperature 0 (beam models replay their search whatever the
/// temperature; greedy-config models decode step by step).
fn chat(m: &dyn NativeLm, text: &str) -> Result<String, String> {
    let tok = m.tokenizer();
    let prompt = tok.apply_chat_template(&[("system", "Be brief."), ("user", text)]);
    let (ids, mut st) = m.prepare(tok.encode(&prompt), &[])?;
    let mut logits = m.forward(&ids, 0, &mut st)?;
    let mut out = Vec::new();
    loop {
        let next = sample_logits(&logits, &[], 0.0, 50, 0.9);
        if tok.ends_generation(next) {
            break;
        }
        out.push(next);
        assert!(out.len() < 400, "no EOS");
        logits = m.forward(&[next], ids.len() + out.len() - 1, &mut st)?;
    }
    // HTTP streams per token (`append_token_bytes`); the TUI decodes the whole sequence.
    let mut streamed = Vec::new();
    for &t in &out {
        assert!(tok.append_token_bytes(t, &mut streamed));
    }
    let text = tok.decode(&out);
    assert_eq!(String::from_utf8(streamed).unwrap(), text);
    Ok(text)
}

/// Model card / hosted-inference example of Helsinki-NLP/opus-mt-en-de (beam 4).
#[test]
fn marian_en_de_translates() {
    let Some(m) = model("Helsinki-NLP/opus-mt-en-de") else {
        return;
    };
    assert_eq!(
        chat(m.as_ref(), "The weather is nice today.").unwrap(),
        "Das Wetter ist heute schön."
    );
    // A single-target model refuses a target tag instead of ignoring it.
    let e = chat(m.as_ref(), ">>fra<< Hello").unwrap_err();
    assert!(e.contains("one language only"), "{e}");
    // The request's max_tokens caps the beam search; a cancelled request stops it.
    let tok = m.tokenizer();
    let ids = tok.encode("The weather is nice today.");
    let ctx = PrepareCtx {
        max_new_tokens: 3,
        cancel: None,
    };
    let (p, mut st) = m.prepare_ctx(ids.clone(), &[], &ctx).unwrap();
    let mut logits = m.forward(&p, 0, &mut st).unwrap();
    let mut n = 0;
    loop {
        let t = sample_logits(&logits, &[], 0.0, 50, 0.9);
        if tok.ends_generation(t) {
            break;
        }
        n += 1;
        logits = m.forward(&[t], p.len() + n - 1, &mut st).unwrap();
    }
    assert_eq!(n, 3);
    let ctx = PrepareCtx {
        max_new_tokens: usize::MAX,
        cancel: Some(Arc::new(true.into())),
    };
    let e = m
        .prepare_ctx(ids, &[], &ctx)
        .err()
        .expect("cancelled search errs");
    assert!(e.contains("cancelled"), "{e}");
}

/// facebook/nllb-200-distilled-600M (greedy): target from `>>code<<`, required.
#[test]
fn nllb_translates_by_target_tag() {
    let Some(m) = model("facebook/nllb-200-distilled-600M") else {
        return;
    };
    let fr = chat(m.as_ref(), ">>fra_Latn<< The weather is nice today.").unwrap();
    let de = chat(m.as_ref(), ">>deu_Latn<< The weather is nice today.").unwrap();
    assert_eq!(fr, "Le temps est beau aujourd'hui.");
    assert_eq!(de, "Das Wetter ist heute schön.");
    // Source language tag: German in, English out.
    assert_eq!(
        chat(
            m.as_ref(),
            "<<deu_Latn>> >>eng_Latn<< Das Wetter ist heute schön."
        )
        .unwrap(),
        "The weather is nice today."
    );
    let e = chat(m.as_ref(), "The weather is nice today.").unwrap_err();
    assert!(e.contains(">>") && e.contains("target"), "{e}");
    let e = chat(m.as_ref(), ">>xx_Nope<< Hi").unwrap_err();
    assert!(e.contains("unknown target language 'xx_Nope'"), "{e}");
}

/// The SentencePiece BPE reader segments like the model's own tokenizer.json (NLLB ships both),
/// Hindi and Chinese included.
#[test]
fn spm_bpe_matches_tokenizer_json() {
    let dir = std::path::Path::new("disk/models--facebook--nllb-200-distilled-600M/snapshots/main");
    let (Ok(spm), Ok(fast)) = (
        gibbs_engine::model::spm::SpmModel::open(&dir.join("sentencepiece.bpe.model")),
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")),
    ) else {
        eprintln!("SKIP: NLLB tokenizer files not in disk/");
        return;
    };
    // fairseq layout: <s> <pad> </s> <unk>, then spm ids shifted by one.
    let mut vocab: std::collections::HashMap<String, u32> =
        [("<s>", 0), ("<pad>", 1), ("</s>", 2), ("<unk>", 3)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
    for (i, p) in spm.pieces.iter().enumerate().skip(3) {
        vocab.entry(p.text.clone()).or_insert(i as u32 + 1);
    }
    let seg = spm.segmenter(&vocab).unwrap();
    for text in [
        "The weather is nice today.",
        "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है",
        "生活就像一盒巧克力。",
        // Edge spaces are trimmed before encoding (spm strips them; tokenizer.json keeps a `▁`).
        "Multiple   spaces, ünïcödé and isn't it?",
    ] {
        let ours = seg.encode_fast(text, false).unwrap().get_ids().to_vec();
        let theirs = fast.encode_fast(text, false).unwrap().get_ids().to_vec();
        assert_eq!(ours, theirs, "{text}");
    }
}

/// The SentencePiece Unigram reader (T5's `spiece.model`, spm ids) segments like the repo's
/// tokenizer.json.
#[test]
fn spm_unigram_matches_tokenizer_json() {
    let Some(dir) = std::fs::read_dir("disk/models--google-t5--t5-small/snapshots")
        .ok()
        .and_then(|mut d| Some(d.next()?.ok()?.path()))
    else {
        eprintln!("SKIP: google-t5/t5-small not in disk/");
        return;
    };
    let ours = gibbs_engine::model::spm::load(&dir.join("spiece.model")).unwrap();
    let fast = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    for text in [
        "translate English to German: The house is wonderful.",
        "Multiple   spaces, ünïcödé, 1,000.50 and isn't it?",
    ] {
        let theirs = fast.encode_fast(text, false).unwrap().get_ids().to_vec();
        assert_eq!(ours.encode(text), theirs, "{text}");
    }
}
