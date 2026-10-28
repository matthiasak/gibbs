//! Encoder goldens: the sequence classifiers (`model::classifier`: BERT, ModernBERT, ELECTRA,
//! ALBERT, DeBERTa v1/v2, SetFit, GPT-2/OPT decoder seq-cls, fastText), System 1
//! (`runtime::encoder::system1`) and Idefics3/SmolVLM (the `idefics3` spec through `runtime::lm`). Every case writes a
//! Hugging Face cache snapshot (`models--gibbs-golden--<case>/snapshots/main`) or a checkpoint
//! directory and loads it through the public loader (`SequenceClassifier::load`,
//! `JevSystem1Model::load`, `runtime::lm::load`). Shapes follow `classifier/synth.rs`,
//! the deleted `ln_decoder.rs` `tiny` and `idefics3.rs` `tiny_engine`.

use crate::util::*;
use gibbs_engine::model::classifier::SequenceClassifier;
use gibbs_engine::runtime::encoder::LabelScore;
use gibbs_engine::model::weight_format::{set_reencode_override, ReencodeConfig, ReencodePolicy};
use gibbs_engine::model::hub::HuggingFaceHub;
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::runtime::encoder::system1::{ChoiceOption, JevDecision, JevQuestion, JevSystem1Model};
use gibbs_engine::model::native::MediaInput;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const ORG: &str = "gibbs-golden";
const VOCAB: usize = 40;
const TEXT: &str = "t5 t9 t4 t11 t6 t7 t13 t8 t10";
const PAIR: &str = "t12 t15 t6";

/// `<root>/models--gibbs-golden--<name>/snapshots/main` (the HF cache layout the hub reads).
fn snapshot(root: &Path, name: &str) -> PathBuf {
    let d = root.join(format!("models--{ORG}--{name}/snapshots/main"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// [`write_tokenizer`] plus a BERT-style template: `<s> A </s>` and `<s> A </s> B </s>` (type 1
/// on the second segment).
fn write_tokenizer_cls(dir: &Path, n: usize) {
    write_tokenizer(dir, n);
    let p = dir.join("tokenizer.json");
    let mut t: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    let sp = |id: &str, ty: u32| json!({"SpecialToken": {"id": id, "type_id": ty}});
    let sq = |id: &str, ty: u32| json!({"Sequence": {"id": id, "type_id": ty}});
    t["post_processor"] = json!({
        "type": "TemplateProcessing",
        "single": [sp("<s>", 0), sq("A", 0), sp("</s>", 0)],
        "pair": [sp("<s>", 0), sq("A", 0), sp("</s>", 0), sq("B", 1), sp("</s>", 1)],
        "special_tokens": {
            "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
            "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
        }
    });
    write_json(&p, &t);
}

fn lin(w: &mut Weights, name: &str, out: usize, inp: usize) {
    w.w(format!("{name}.weight"), &[out, inp]);
    w.s(format!("{name}.bias"), &[out], 0.1);
}

fn ln(w: &mut Weights, name: &str, d: usize) {
    w.norm(format!("{name}.weight"), d, 1.0);
    w.s(format!("{name}.bias"), &[d], 0.1);
}

/// A 3-label head with logits of order 1 (so the probabilities are far from uniform).
fn head(w: &mut Weights, name: &str, inp: usize) {
    w.s(format!("{name}.weight"), &[3, inp], 1.0);
    w.s(format!("{name}.bias"), &[3], 0.1);
}

// ---------------------------------------------------------------------------
// Encoder sequence classifiers (`classifier/synth.rs` shapes)
// ---------------------------------------------------------------------------

/// Writes the classifier repo `case` (config.json with 3 labels, model.safetensors, tokenizer)
/// under `fixture_dir(case)` and returns its hub.
fn cls_repo(case: &str, mut cfg: Value, w: &Weights, template: bool) -> HuggingFaceHub {
    let root = fixture_dir(case);
    let snap = snapshot(&root, case);
    cfg["id2label"] = json!({"0": "A", "1": "B", "2": "C"});
    cfg["vocab_size"] = json!(VOCAB);
    write_json(&snap.join("config.json"), &cfg);
    write_json(&snap.join("tokenizer_config.json"), &json!({}));
    w.write(&snap.join("model.safetensors"));
    if template {
        write_tokenizer_cls(&snap, VOCAB);
    } else {
        write_tokenizer(&snap, VOCAB);
    }
    HuggingFaceHub::with_cache_dir(root)
}

/// `SequenceClassifier::load` + `classify` on each input: per-label scores (label order) and the
/// token count.
fn classify_golden(case: &str, hub: &HuggingFaceHub, inputs: &[(&str, Option<&str>)], tol: Tol) {
    let id = format!("{ORG}/{case}");
    golden(case, tol, |dev| {
        let c = SequenceClassifier::load(hub, &id, dev).unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(c.device(), dev, "{case}");
        let mut out = Out::default();
        for (i, (text, pair)) in inputs.iter().enumerate() {
            let (mut s, n) = c
                .classify(text, *pair)
                .unwrap_or_else(|e| panic!("{case}: {e}"));
            s.sort_by(|a, b| a.label.cmp(&b.label));
            out.f(format!("scores.{i}"), &s.iter().map(|x| x.score).collect::<Vec<_>>());
            out.ids(format!("labels.{i}"), &label_bytes(&s));
            out.ids(format!("tokens.{i}"), &[n as u32]);
        }
        out
    });
}

/// The label names in score order, `\n`-joined, as bytes (an exact check on `id2label` /
/// `classes_` / `__label__` handling).
fn label_bytes(s: &[LabelScore]) -> Vec<u32> {
    let names: Vec<&str> = s.iter().map(|x| x.label.as_str()).collect();
    names.join("\n").bytes().map(u32::from).collect()
}

const TEXT_AND_PAIR: &[(&str, Option<&str>)] = &[(TEXT, None), (TEXT, Some(PAIR))];

fn bert_cfg(arch: &str, problem: Option<&str>) -> Value {
    json!({"architectures": [arch], "model_type": "bert", "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "intermediate_size": 24, "max_position_embeddings": 32,
        "type_vocab_size": 2, "layer_norm_eps": 1e-12, "hidden_act": "gelu", "problem_type": problem})
}

fn bert_layer(w: &mut Weights, p: &str, h: usize, inter: usize) {
    for s in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    ] {
        lin(w, &format!("{p}.{s}"), h, h);
    }
    ln(w, &format!("{p}.attention.output.LayerNorm"), h);
    lin(w, &format!("{p}.intermediate.dense"), inter, h);
    lin(w, &format!("{p}.output.dense"), h, inter);
    ln(w, &format!("{p}.output.LayerNorm"), h);
}

/// BERT encoder tensors under `prefix` (`"bert."` or `""`).
fn bert_body(w: &mut Weights, prefix: &str, vocab: usize, h: usize, inter: usize, layers: usize) {
    w.s(format!("{prefix}embeddings.word_embeddings.weight"), &[vocab, h], 0.5);
    w.s(format!("{prefix}embeddings.position_embeddings.weight"), &[32, h], 0.3);
    w.s(format!("{prefix}embeddings.token_type_embeddings.weight"), &[2, h], 0.3);
    ln(w, &format!("{prefix}embeddings.LayerNorm"), h);
    for l in 0..layers {
        bert_layer(w, &format!("{prefix}encoder.layer.{l}"), h, inter);
    }
}

fn bert(case: &str, problem: Option<&str>, seed: u64) {
    let mut w = Weights::new(seed);
    bert_body(&mut w, "bert.", VOCAB, 16, 24, 2);
    lin(&mut w, "bert.pooler.dense", 16, 16);
    head(&mut w, "classifier", 16);
    let hub = cls_repo(case, bert_cfg("BertForSequenceClassification", problem), &w, true);
    classify_golden(case, &hub, TEXT_AND_PAIR, TOL);
}

#[test]
fn bert_softmax() {
    bert("enc_bert_softmax", None, 0xE1);
}

#[test]
fn bert_multi_label() {
    bert("enc_bert_multi_label", Some("multi_label_classification"), 0xE2);
}

fn modernbert(case: &str, pooling: &str, yarn: bool, seed: u64) {
    let (h, inter, layers) = (16, 24, 3);
    let mut cfg = json!({"architectures": ["ModernBertForSequenceClassification"],
        "model_type": "modernbert", "hidden_size": h, "num_attention_heads": 2,
        "intermediate_size": inter, "num_hidden_layers": layers, "local_attention": 4,
        "global_attn_every_n_layers": 3, "norm_eps": 1e-5, "norm_bias": false,
        "classifier_pooling": pooling, "max_position_embeddings": 64});
    if yarn {
        cfg["rope_scaling"] =
            json!({"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32});
    }
    let mut w = Weights::new(seed);
    w.s("model.embeddings.tok_embeddings.weight", &[VOCAB, h], 0.5);
    w.norm("model.embeddings.norm.weight", h, 1.0);
    for l in 0..layers {
        let p = format!("model.layers.{l}");
        if l > 0 {
            w.norm(format!("{p}.attn_norm.weight"), h, 1.0);
        }
        w.w(format!("{p}.attn.Wqkv.weight"), &[3 * h, h]);
        w.w(format!("{p}.attn.Wo.weight"), &[h, h]);
        w.norm(format!("{p}.mlp_norm.weight"), h, 1.0);
        w.w(format!("{p}.mlp.Wi.weight"), &[2 * inter, h]);
        w.w(format!("{p}.mlp.Wo.weight"), &[h, inter]);
    }
    w.norm("model.final_norm.weight", h, 1.0);
    w.w("head.dense.weight", &[h, h]);
    w.norm("head.norm.weight", h, 1.0);
    head(&mut w, "classifier", h);
    let hub = cls_repo(case, cfg, &w, true);
    classify_golden(case, &hub, TEXT_AND_PAIR, TOL);
}

/// Windowed (local 4) + global attention layers, CLS pooling.
#[test]
fn modernbert_cls() {
    modernbert("enc_modernbert_cls", "cls", false, 0xE3);
}

/// Mean pooling and a YaRN attention factor (cos/sin scale != 1).
#[test]
fn modernbert_mean() {
    modernbert("enc_modernbert_mean", "mean", true, 0xE4);
}

/// Projected (8 → 16) embeddings, GELU classification head.
#[test]
fn electra() {
    let (h, e, inter) = (16, 8, 24);
    let mut cfg = bert_cfg("ElectraForSequenceClassification", None);
    cfg["model_type"] = json!("electra");
    cfg["embedding_size"] = json!(e);
    let mut w = Weights::new(0xE5);
    w.s("electra.embeddings.word_embeddings.weight", &[VOCAB, e], 0.5);
    w.s("electra.embeddings.position_embeddings.weight", &[32, e], 0.3);
    w.s("electra.embeddings.token_type_embeddings.weight", &[2, e], 0.3);
    ln(&mut w, "electra.embeddings.LayerNorm", e);
    lin(&mut w, "electra.embeddings_project", h, e);
    for l in 0..2 {
        bert_layer(&mut w, &format!("electra.encoder.layer.{l}"), h, inter);
    }
    lin(&mut w, "classifier.dense", h, h);
    head(&mut w, "classifier.out_proj", h);
    let hub = cls_repo("enc_electra", cfg, &w, true);
    classify_golden("enc_electra", &hub, TEXT_AND_PAIR, TOL);
}

/// Three steps over one shared layer group, tanh GELU.
#[test]
fn albert() {
    let (h, e, inter) = (16, 8, 24);
    let mut cfg = bert_cfg("AlbertForSequenceClassification", None);
    cfg["model_type"] = json!("albert");
    cfg["embedding_size"] = json!(e);
    cfg["num_hidden_layers"] = json!(3);
    cfg["num_hidden_groups"] = json!(1);
    cfg["inner_group_num"] = json!(1);
    cfg["hidden_act"] = json!("gelu_new");
    let mut w = Weights::new(0xE6);
    w.s("albert.embeddings.word_embeddings.weight", &[VOCAB, e], 0.5);
    w.s("albert.embeddings.position_embeddings.weight", &[32, e], 0.3);
    w.s("albert.embeddings.token_type_embeddings.weight", &[2, e], 0.3);
    ln(&mut w, "albert.embeddings.LayerNorm", e);
    lin(&mut w, "albert.encoder.embedding_hidden_mapping_in", h, e);
    let p = "albert.encoder.albert_layer_groups.0.albert_layers.0";
    for s in [
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        lin(&mut w, &format!("{p}.{s}"), h, h);
    }
    ln(&mut w, &format!("{p}.attention.LayerNorm"), h);
    lin(&mut w, &format!("{p}.ffn"), inter, h);
    lin(&mut w, &format!("{p}.ffn_output"), h, inter);
    ln(&mut w, &format!("{p}.full_layer_layer_norm"), h);
    lin(&mut w, "albert.pooler", h, h);
    head(&mut w, "classifier", h);
    let hub = cls_repo("enc_albert", cfg, &w, true);
    classify_golden("enc_albert", &hub, TEXT_AND_PAIR, TOL);
}

fn deberta_layers(w: &mut Weights, h: usize, inter: usize, layers: usize, v1: bool) {
    for l in 0..layers {
        let p = format!("deberta.encoder.layer.{l}");
        if v1 {
            w.w(format!("{p}.attention.self.in_proj.weight"), &[3 * h, h]);
            w.s(format!("{p}.attention.self.q_bias"), &[h], 0.1);
            w.s(format!("{p}.attention.self.v_bias"), &[h], 0.1);
            w.w(format!("{p}.attention.self.pos_proj.weight"), &[h, h]);
            lin(w, &format!("{p}.attention.self.pos_q_proj"), h, h);
        } else {
            for s in ["query_proj", "key_proj", "value_proj"] {
                lin(w, &format!("{p}.attention.self.{s}"), h, h);
            }
        }
        lin(w, &format!("{p}.attention.output.dense"), h, h);
        ln(w, &format!("{p}.attention.output.LayerNorm"), h);
        lin(w, &format!("{p}.intermediate.dense"), inter, h);
        lin(w, &format!("{p}.output.dense"), h, inter);
        ln(w, &format!("{p}.output.LayerNorm"), h);
    }
    lin(w, "pooler.dense", h, h);
    head(w, "classifier", h);
}

/// v2/v3: shared keys, no absolute positions, log-bucketed relative positions (4 buckets, so the
/// 11-15 token inputs reach the log branch), `norm_rel_ebd = layer_norm`.
#[test]
fn deberta_v2() {
    let h = 16;
    let cfg = json!({"architectures": ["DebertaV2ForSequenceClassification"],
        "model_type": "deberta-v2", "hidden_size": h, "num_hidden_layers": 2,
        "num_attention_heads": 2, "intermediate_size": 24, "max_position_embeddings": 32,
        "relative_attention": true, "pos_att_type": ["p2c", "c2p"], "share_att_key": true,
        "position_biased_input": false, "norm_rel_ebd": "layer_norm", "position_buckets": 4,
        "max_relative_positions": 16, "type_vocab_size": 0, "layer_norm_eps": 1e-7});
    let mut w = Weights::new(0xE7);
    w.s("deberta.embeddings.word_embeddings.weight", &[VOCAB, h], 0.5);
    ln(&mut w, "deberta.embeddings.LayerNorm", h);
    w.s("deberta.encoder.rel_embeddings.weight", &[8, h], 0.5);
    ln(&mut w, "deberta.encoder.LayerNorm", h);
    deberta_layers(&mut w, h, 24, 2, false);
    let hub = cls_repo("enc_deberta_v2", cfg, &w, true);
    classify_golden("enc_deberta_v2", &hub, TEXT_AND_PAIR, TOL);
}

/// v1: `in_proj` with per-head interleaved q/k/v, separate pos projections, absolute positions.
#[test]
fn deberta_v1() {
    let h = 16;
    let cfg = json!({"architectures": ["DebertaForSequenceClassification"],
        "model_type": "deberta", "hidden_size": h, "num_hidden_layers": 2,
        "num_attention_heads": 2, "intermediate_size": 24, "max_position_embeddings": 32,
        "relative_attention": true, "pos_att_type": "c2p|p2c", "position_biased_input": true,
        "max_relative_positions": 6, "type_vocab_size": 0, "layer_norm_eps": 1e-7});
    let mut w = Weights::new(0xE8);
    w.s("deberta.embeddings.word_embeddings.weight", &[VOCAB, h], 0.5);
    w.s("deberta.embeddings.position_embeddings.weight", &[32, h], 0.3);
    ln(&mut w, "deberta.embeddings.LayerNorm", h);
    w.s("deberta.encoder.rel_embeddings.weight", &[12, h], 0.5);
    deberta_layers(&mut w, h, 24, 2, true);
    let hub = cls_repo("enc_deberta_v1", cfg, &w, true);
    classify_golden("enc_deberta_v1", &hub, TEXT_AND_PAIR, TOL);
}

// ---------------------------------------------------------------------------
// SetFit: sentence-transformers BERT body + joblib-pickled sklearn LogisticRegression
// ---------------------------------------------------------------------------

/// Minimal protocol-4 pickle writer for what joblib emits for a LogisticRegression.
struct Pkl(Vec<u8>);

impl Pkl {
    fn str(&mut self, s: &str) -> &mut Self {
        self.0.push(0x8c); // SHORT_BINUNICODE
        self.0.push(s.len() as u8);
        self.0.extend_from_slice(s.as_bytes());
        self
    }
    fn global(&mut self, module: &str, name: &str) -> &mut Self {
        self.str(module).str(name);
        self.0.push(0x93); // STACK_GLOBAL
        self
    }
    fn op(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self
    }
    /// A `joblib.numpy_pickle.NumpyArrayWrapper` (C order, little endian) and its raw bytes.
    fn array(&mut self, code: &str, shape: &[usize], raw: &[u8]) -> &mut Self {
        self.global("joblib.numpy_pickle", "NumpyArrayWrapper").op(b")\x81}(");
        self.str("subclass").global("numpy", "ndarray");
        self.str("shape").op(b"(");
        for &d in shape {
            self.op(&[b'K', d as u8]);
        }
        self.op(b"t");
        self.str("order").str("C");
        // numpy.dtype(code, False, True) with state (3, '<', None, None, None, -1, -1, 0).
        self.str("dtype").global("numpy", "dtype").str(code).op(b"\x89\x88\x87R(K\x03");
        self.str("<").op(b"NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb");
        self.str("allow_mmap").op(b"\x88");
        self.str("numpy_array_alignment_bytes").op(b"K\x10");
        self.op(b"ub");
        // joblib: a length byte, then padding so the raw array starts 16-byte aligned.
        let pad = (16 - (self.0.len() + 1) % 16) % 16;
        self.op(&[pad as u8]).op(&vec![0xff; pad]).op(raw)
    }
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `model_head.pkl`: a 3-class multinomial LogisticRegression over `features` inputs.
fn logistic_pickle(rng: &mut Rng, features: usize) -> Vec<u8> {
    let coef: Vec<f64> = rng.vec(3 * features, 2.0).into_iter().map(f64::from).collect();
    let intercept: Vec<f64> = rng.vec(3, 0.5).into_iter().map(f64::from).collect();
    let classes: Vec<u8> = [0i64, 1, 2].iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut p = Pkl(vec![0x80, 4]);
    p.global("sklearn.linear_model._logistic", "LogisticRegression")
        .op(b")\x81}(");
    p.str("_sklearn_version").str("1.5.2");
    p.str("solver").str("lbfgs");
    p.str("classes_").array("i8", &[3], &classes);
    p.str("coef_").array("f8", &[3, features], &f64_bytes(&coef));
    p.str("intercept_").array("f8", &[3], &f64_bytes(&intercept));
    p.op(b"ub.");
    p.0
}

/// Mean pooling + Normalize module, 3-class multinomial head (sklearn `predict_proba`).
#[test]
fn setfit() {
    let case = "enc_setfit";
    let (h, inter) = (16, 24);
    let root = fixture_dir(case);
    let snap = snapshot(&root, case);
    let mut cfg = bert_cfg("BertModel", None);
    cfg["vocab_size"] = json!(VOCAB);
    write_json(&snap.join("config.json"), &cfg);
    let mut w = Weights::new(0xE9);
    bert_body(&mut w, "", VOCAB, h, inter, 2);
    lin(&mut w, "pooler.dense", h, h);
    w.write(&snap.join("model.safetensors"));
    write_tokenizer_cls(&snap, VOCAB);
    write_json(&snap.join("tokenizer_config.json"), &json!({}));
    write_json(
        &snap.join("config_setfit.json"),
        &json!({"labels": ["neg", "neu", "pos"], "normalize_embeddings": false}),
    );
    write_json(
        &snap.join("modules.json"),
        &json!([
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
        ]),
    );
    write_json(
        &snap.join("1_Pooling/config.json"),
        &json!({"word_embedding_dimension": h, "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": true, "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false, "pooling_mode_weightedmean_tokens": false,
            "pooling_mode_lasttoken": false}),
    );
    write_json(
        &snap.join("sentence_bert_config.json"),
        &json!({"max_seq_length": 32, "do_lower_case": false}),
    );
    std::fs::write(
        snap.join("model_head.pkl"),
        logistic_pickle(&mut Rng::new(0xEA), h),
    )
    .unwrap();
    let hub = HuggingFaceHub::with_cache_dir(root);
    // SetFit classifies single texts (a pair is refused).
    classify_golden(case, &hub, &[(TEXT, None), (PAIR, None)], TOL);
}

// ---------------------------------------------------------------------------
// GPT-2 / OPT decoder seq-cls (old `ln_decoder.rs` `tiny`: 2 blocks, 2 heads, hidden 32, ffn 64)
// ---------------------------------------------------------------------------

const DEC_VOCAB: usize = 64;

/// GPT-2 Conv1D `[in, out]` weight + bias.
fn conv1d(w: &mut Weights, name: &str, inp: usize, out: usize) {
    w.s(format!("{name}.weight"), &[inp, out], 1.0 / (inp as f32).sqrt());
    w.s(format!("{name}.bias"), &[out], 0.1);
}

fn decoder_repo(case: &str, mut cfg: Value, w: &Weights) -> HuggingFaceHub {
    let root = fixture_dir(case);
    let snap = snapshot(&root, case);
    cfg["id2label"] = json!({"0": "A", "1": "B", "2": "C"});
    cfg["vocab_size"] = json!(DEC_VOCAB);
    cfg["pad_token_id"] = json!(0);
    write_json(&snap.join("config.json"), &cfg);
    write_json(&snap.join("tokenizer_config.json"), &json!({}));
    w.write(&snap.join("model.safetensors"));
    write_tokenizer(&snap, DEC_VOCAB);
    HuggingFaceHub::with_cache_dir(root)
}

/// Pre-LN, tanh GELU, learned positions, `transformer.` prefix.
#[test]
fn gpt2() {
    let (h, ffn, ctx) = (32, 64, 32);
    let mut w = Weights::new(0xEB);
    w.s("transformer.wte.weight", &[DEC_VOCAB, h], 0.5);
    w.s("transformer.wpe.weight", &[ctx, h], 0.3);
    for l in 0..2 {
        let p = format!("transformer.h.{l}");
        ln(&mut w, &format!("{p}.ln_1"), h);
        conv1d(&mut w, &format!("{p}.attn.c_attn"), h, 3 * h);
        conv1d(&mut w, &format!("{p}.attn.c_proj"), h, h);
        ln(&mut w, &format!("{p}.ln_2"), h);
        conv1d(&mut w, &format!("{p}.mlp.c_fc"), h, ffn);
        conv1d(&mut w, &format!("{p}.mlp.c_proj"), ffn, h);
    }
    ln(&mut w, "transformer.ln_f", h);
    w.s("score.weight", &[3, h], 0.3);
    let cfg = json!({"architectures": ["GPT2ForSequenceClassification"], "model_type": "gpt2",
        "n_layer": 2, "n_head": 2, "n_embd": h, "n_positions": ctx, "n_ctx": ctx,
        "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5});
    let hub = decoder_repo("enc_gpt2", cfg, &w);
    // The reference (`ln_decoder.rs` host forward) is deleted: the golden is frozen.
    if writing() {
        return;
    }
    classify_golden("enc_gpt2", &hub, TEXT_AND_PAIR, TOL);
}

/// OPT with `project_in`/`project_out` (embeddings 16, hidden 32) and positions offset by 2.
fn opt(case: &str, pre_ln: bool, act: &str, seed: u64) {
    let (h, e, ffn, ctx) = (32, 16, 64, 32);
    let p = "model.decoder";
    let mut w = Weights::new(seed);
    w.s(format!("{p}.embed_tokens.weight"), &[DEC_VOCAB, e], 0.5);
    w.s(format!("{p}.embed_positions.weight"), &[ctx + 2, h], 0.3);
    w.w(format!("{p}.project_in.weight"), &[h, e]);
    w.w(format!("{p}.project_out.weight"), &[e, h]);
    for l in 0..2 {
        let b = format!("{p}.layers.{l}");
        ln(&mut w, &format!("{b}.self_attn_layer_norm"), h);
        for s in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(&mut w, &format!("{b}.self_attn.{s}"), h, h);
        }
        ln(&mut w, &format!("{b}.final_layer_norm"), h);
        lin(&mut w, &format!("{b}.fc1"), ffn, h);
        lin(&mut w, &format!("{b}.fc2"), h, ffn);
    }
    if pre_ln {
        ln(&mut w, &format!("{p}.final_layer_norm"), h);
    }
    w.s("score.weight", &[3, e], 1.0);
    let cfg = json!({"architectures": ["OPTForSequenceClassification"], "model_type": "opt",
        "hidden_size": h, "ffn_dim": ffn, "num_hidden_layers": 2, "num_attention_heads": 2,
        "word_embed_proj_dim": e, "max_position_embeddings": ctx,
        "do_layer_norm_before": pre_ln, "activation_function": act, "enable_bias": true,
        "layer_norm_elementwise_affine": true});
    let hub = decoder_repo(case, cfg, &w);
    // The reference (`ln_decoder.rs` host forward) is deleted: the golden is frozen.
    if writing() {
        return;
    }
    classify_golden(case, &hub, TEXT_AND_PAIR, TOL);
}

#[test]
fn opt_post_ln() {
    opt("enc_opt_post_ln", false, "relu", 0xEC);
}

#[test]
fn opt_pre_ln() {
    opt("enc_opt_pre_ln", true, "gelu", 0xED);
}

// ---------------------------------------------------------------------------
// fastText `.bin` (softmax loss, word bigrams, char 2-3-grams, 64 buckets)
// ---------------------------------------------------------------------------

/// A version-12 supervised fastText `.bin` in the layout `FastText::load` reads (`Args`,
/// `Dictionary`, dense input `[nwords + bucket, dim]` and output `[nlabels, dim]` matrices).
fn write_fasttext(path: &Path) {
    let (dim, bucket) = (8i32, 64i32);
    let words = ["</s>", "the", "cat", "sat", "dog"];
    let labels = ["__label__a", "__label__b", "__label__c"];
    let mut b = Vec::new();
    let i32s = |b: &mut Vec<u8>, v: &[i32]| v.iter().for_each(|x| b.extend(x.to_le_bytes()));
    i32s(&mut b, &[793712314, 12]);
    // dim ws epoch minCount neg wordNgrams loss(softmax) model(sup) bucket minn maxn lrUpdateRate
    i32s(&mut b, &[dim, 5, 5, 1, 5, 2, 3, 3, bucket, 2, 3, 100]);
    b.extend(1e-4f64.to_le_bytes());
    let size = (words.len() + labels.len()) as i32;
    i32s(&mut b, &[size, words.len() as i32, labels.len() as i32]);
    b.extend(100i64.to_le_bytes()); // ntokens
    b.extend((-1i64).to_le_bytes()); // pruneidx_size: not pruned
    for (i, w) in words.iter().chain(&labels).enumerate() {
        b.extend_from_slice(w.as_bytes());
        b.push(0);
        b.extend((20 - i as i64).to_le_bytes());
        b.push((i >= words.len()) as u8);
    }
    let mut rng = Rng::new(0xEE);
    let mut matrix = |b: &mut Vec<u8>, rows: usize, scale: f32| {
        b.push(0); // not quantized
        b.extend((rows as i64).to_le_bytes());
        b.extend((dim as i64).to_le_bytes());
        rng.vec(rows * dim as usize, scale)
            .iter()
            .for_each(|x| b.extend(x.to_le_bytes()));
    };
    matrix(&mut b, words.len() + bucket as usize, 1.0);
    matrix(&mut b, labels.len(), 2.0);
    std::fs::write(path, b).unwrap();
}

/// Known words with subwords and bigrams, then a line with out-of-vocabulary words (subwords only).
#[test]
fn fasttext() {
    let case = "enc_fasttext";
    let path = fixture_dir(case).join("model.bin");
    write_fasttext(&path);
    golden(case, TOL, |_| {
        // runner: host-only (no device argument); the same model is checked once per device.
        let dev = gibbs_engine::hardware::Device::Cpu;
        let c = SequenceClassifier::from_fasttext("gibbs-golden/fasttext", &path, dev)
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        let mut out = Out::default();
        for (i, text) in ["the cat sat", "a dog sat on the mat"].iter().enumerate() {
            let (mut s, n) = c.classify(text, None).unwrap();
            s.sort_by(|a, b| a.label.cmp(&b.label));
            out.f(format!("scores.{i}"), &s.iter().map(|x| x.score).collect::<Vec<_>>());
            out.ids(format!("labels.{i}"), &label_bytes(&s));
            out.ids(format!("tokens.{i}"), &[n as u32]);
        }
        out
    });
}

// ---------------------------------------------------------------------------
// System 1 (`jev.rs` BertEncoder: MiniLM-like post-LN BERT, mean pool + L2)
// ---------------------------------------------------------------------------

/// Embeddings of two texts, a zero-shot choice and a zero-shot bool decision.
#[test]
fn system1() {
    let case = "enc_system1";
    let (vocab, h, inter) = (64, 32, 64);
    let root = fixture_dir(case);
    let snap = snapshot(&root, case);
    write_json(
        &snap.join("config.json"),
        &json!({"architectures": ["BertModel"], "model_type": "bert", "hidden_size": h,
            "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": inter,
            "max_position_embeddings": 32, "vocab_size": vocab, "type_vocab_size": 2,
            "layer_norm_eps": 1e-12, "hidden_act": "gelu"}),
    );
    let mut w = Weights::new(0xEF);
    bert_body(&mut w, "", vocab, h, inter, 2);
    w.write(&snap.join("model.safetensors"));
    write_tokenizer_cls(&snap, vocab);
    let hub = HuggingFaceHub::with_cache_dir(root);
    let id = format!("{ORG}/{case}");
    golden(case, TOL, |dev| {
        let m = JevSystem1Model::load(&hub, &id, dev).unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(m.encoder().map(|b| b.device()), Some(dev));
        let mut out = Out::default();
        let options = ["t20 t22", "t24 t26", "t28 t30"];
        let choice = JevQuestion::Choice {
            id: "c".into(),
            instructions: "pick".into(),
            options: options.iter().map(|o| ChoiceOption::new(*o, None::<&str>)).collect(),
        };
        let bool_q = JevQuestion::Bool {
            id: "b".into(),
            instructions: "is it".into(),
            criteria_true: Some("t32 t33".into()),
            criteria_false: Some("t34 t35".into()),
        };
        for (i, state) in ["t5 t9 t4 t11 t6 t7", "t40 t41 t42"].iter().enumerate() {
            out.f(format!("embed.{i}"), &m.embed(state).unwrap());
            match m.evaluate(state, &choice).unwrap() {
                JevDecision::Choice {
                    selected,
                    probabilities,
                    confidence,
                } => {
                    let p: Vec<f32> = options.iter().map(|o| probabilities[*o]).collect();
                    out.f(format!("choice.{i}.probs"), &p);
                    out.f(format!("choice.{i}.confidence"), &[confidence]);
                    let sel = options.iter().position(|o| *o == selected).unwrap();
                    out.ids(format!("choice.{i}.selected"), &[sel as u32]);
                }
                d => panic!("{case}: {d:?} is not a choice"),
            }
            match m.evaluate(state, &bool_q).unwrap() {
                JevDecision::Bool {
                    bool_prob,
                    decision,
                } => {
                    out.f(format!("bool.{i}.prob"), &[bool_prob]);
                    out.ids(format!("bool.{i}.decision"), &[decision as u32]);
                }
                d => panic!("{case}: {d:?} is not a bool"),
            }
        }
        out
    });
}

// ---------------------------------------------------------------------------
// Idefics3 / SmolVLM (`idefics3.rs` `tiny_engine` + `siglip.rs` `tiny`)
// ---------------------------------------------------------------------------

const VLM_VOCAB: usize = 128;
/// `<image>`; the other image tokens follow it.
const IMG: u32 = 120;
const VLM_SPECIALS: [&str; 7] = [
    "<image>",
    "<fake_token_around_image>",
    "<global-img>",
    "<row_1_col_1>",
    "<row_1_col_2>",
    "<row_2_col_1>",
    "<row_2_col_2>",
];

/// SigLIP (hidden 16, 2 layers, 16 px tiles, 4 px patches) → pixel shuffle (scale 2: 4 rows per
/// tile) → Llama (hidden 64, 2 layers, 4 heads, 2 KV heads, vocab 128); images resized to a
/// longest edge of 32 and split into 16 px tiles plus a global view.
fn smolvlm_checkpoint(dir: &Path) {
    write_tokenizer(dir, IMG as usize);
    let p = dir.join("tokenizer.json");
    let mut t: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    for (i, s) in VLM_SPECIALS.iter().enumerate() {
        let id = IMG as usize + i;
        t["model"]["vocab"][s] = json!(id);
        t["added_tokens"].as_array_mut().unwrap().push(json!({"id": id, "content": s,
            "single_word": false, "lstrip": false, "rstrip": false, "normalized": false,
            "special": true}));
    }
    write_json(&p, &t);
    write_json(
        &dir.join("preprocessor_config.json"),
        &json!({"do_resize": true, "size": {"longest_edge": 32},
            "max_image_size": {"longest_edge": 16}, "do_image_splitting": true, "resample": 1,
            "do_rescale": true, "rescale_factor": 1.0 / 255.0, "do_normalize": true,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}),
    );
    let (vd, vi, patch, th, ti, kv) = (16, 32, 4, 64, 128, 32);
    write_json(
        &dir.join("config.json"),
        &json!({"architectures": ["Idefics3ForConditionalGeneration"], "model_type": "idefics3",
            "image_token_id": IMG, "scale_factor": 2,
            "vision_config": {"hidden_size": vd, "intermediate_size": vi, "num_hidden_layers": 2,
                "num_attention_heads": 2, "image_size": 16, "patch_size": patch,
                "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6},
            "text_config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "hidden_size": th, "intermediate_size": ti, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": VLM_VOCAB,
                "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128,
                "hidden_act": "silu", "tie_word_embeddings": false}}),
    );
    let mut w = Weights::new(0xF0);
    let v = "model.vision_model";
    w.s(
        format!("{v}.embeddings.patch_embedding.weight"),
        &[vd, 3, patch, patch],
        1.0 / ((3 * patch * patch) as f32).sqrt(),
    );
    w.s(format!("{v}.embeddings.patch_embedding.bias"), &[vd], 0.05);
    w.s(format!("{v}.embeddings.position_embedding.weight"), &[16, vd], 0.2);
    for l in 0..2 {
        let p = format!("{v}.encoder.layers.{l}");
        ln(&mut w, &format!("{p}.layer_norm1"), vd);
        for s in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(&mut w, &format!("{p}.self_attn.{s}"), vd, vd);
        }
        ln(&mut w, &format!("{p}.layer_norm2"), vd);
        lin(&mut w, &format!("{p}.mlp.fc1"), vi, vd);
        lin(&mut w, &format!("{p}.mlp.fc2"), vd, vi);
    }
    ln(&mut w, &format!("{v}.post_layernorm"), vd);
    w.w("model.connector.modality_projection.proj.weight", &[th, vd * 4]);
    let t = "model.text_model";
    w.s(format!("{t}.embed_tokens.weight"), &[VLM_VOCAB, th], 1.0);
    for l in 0..2 {
        let p = format!("{t}.layers.{l}");
        w.norm(format!("{p}.input_layernorm.weight"), th, 1.0);
        w.w(format!("{p}.self_attn.q_proj.weight"), &[th, th]);
        w.w(format!("{p}.self_attn.k_proj.weight"), &[kv, th]);
        w.w(format!("{p}.self_attn.v_proj.weight"), &[kv, th]);
        w.w(format!("{p}.self_attn.o_proj.weight"), &[th, th]);
        w.norm(format!("{p}.post_attention_layernorm.weight"), th, 1.0);
        w.w(format!("{p}.mlp.gate_proj.weight"), &[ti, th]);
        w.w(format!("{p}.mlp.up_proj.weight"), &[ti, th]);
        w.w(format!("{p}.mlp.down_proj.weight"), &[th, ti]);
    }
    w.norm(format!("{t}.norm.weight"), th, 1.0);
    w.w("lm_head.weight", &[VLM_VOCAB, th]);
    w.write(&dir.join("model.safetensors"));
}

/// One `<image>` prompt through the `idefics3` spec (`runtime::lm::load`) → `prepare` (tiles,
/// SigLIP tower, connector, placeholder expansion) → prefill logits, then three teacher-forced
/// decode steps. runner: the golden was written by the deleted-path `model::idefics3` engine.
fn smolvlm(case: &str, width: usize, height: usize) {
    let dir = fixture_dir(case);
    smolvlm_checkpoint(&dir);
    let data = (0..width * height * 3).map(|i| (i * 37 % 251) as u8).collect();
    let png = ImageBuffer::from_rgb(width, height, data).unwrap().to_png();
    let art = artifacts(&dir, "gibbs-golden/smolvlm", "model.safetensors");
    // The reference (the old Idefics3 engine) is deleted from this path: the golden is frozen.
    if writing() {
        return;
    }
    // Pin the bit-exact load: the settings-file re-encode policy (and its cache) must not change
    // the Metal weights. ponytail: process-global, set (never cleared) by every spec-load case.
    set_reencode_override(Some(ReencodeConfig {
        policy: ReencodePolicy::BitExact,
        ..Default::default()
    }));
    golden(case, TOL, |dev| {
        let reg = gibbs_engine::arch::registry().unwrap();
        let m = gibbs_engine::runtime::lm::load(&art, reg, Some("idefics3"), dev.into())
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(m.device(), dev, "{case}");
        let (ids, mut st) = m
            .prepare(vec![1, 10, 11, IMG, 12, 13], &[MediaInput::Image(png.clone())])
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        let mut out = Out::default();
        out.ids("prompt", &ids);
        out.f("prefill", &m.forward(&ids, 0, &mut st).unwrap());
        for (i, t) in [20u32, 21, 22].into_iter().enumerate() {
            let logits = m.forward(&[t], ids.len() + i, &mut st).unwrap();
            out.f(format!("decode.{i}"), &logits);
        }
        out
    });
}

/// 24×24 → 32×32: a 2×2 tile grid plus the global view (5 SigLIP passes).
#[test]
fn smolvlm_grid() {
    smolvlm("enc_smolvlm_grid", 24, 24);
}

/// 30×10 → 32×10 → split 32×16: a 1×2 tile grid plus the global view.
#[test]
fn smolvlm_wide() {
    smolvlm("enc_smolvlm_wide", 30, 10);
}
