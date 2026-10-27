//! NInfer (`.ninfer`, github.com/Neroued/ninfer) artifacts of dense gated-delta-net hybrid causal
//! LMs: one file (or a file plus `.part-NNNN` volumes) of a JSON catalog and 256-byte-aligned
//! encoded objects, read back as the transformers tensors the matching family spec loads. The
//! container is v2 (`NINFER\0\x02`: named objects, geometry implied by the tensors) or v3
//! (`NINFER\0\x03`: `components.text.config`, logical `bindings` over object element ranges). v1,
//! MoE / speech architectures and the codecs outside [`Codec`] are refused with their name.
//!
//! [`Catalog`] is everything but the payload: the loader ([`Artifact`]) and the hub picker
//! ([`hub_verdict`], over the catalog it fetches with two Range requests) run the same
//! [`Catalog::plan`], so a row the picker shows as loadable parses at load.
//!
//! Spec: ninfer `docs/maintainer/{artifact-container,storage-layouts,tensor-formats}.md`.

use crate::hardware::QuantFormat;
use crate::model::par::par_rows;
use crate::model::weights::Weights;
use crate::model::{bf16_to_f32, f16_to_f32, f8_e4m3_to_f32};
use crate::model::layers::NVFP4_E2M1_LUT;
use memmap2::Mmap;
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

/// File extension of an artifact entry; volumes append `.part-NNNN`.
pub const EXT: &str = ".ninfer";

/// Numeric format + layout of a tensor object (names normalized across container v2 and v3).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Codec {
    Bf16,
    Fp32,
    Int32,
    /// `row_split_k128_v1`: signed `bits`-wide codes, one f16 multiplier per `group`.
    Group {
        bits: usize,
        group: usize,
    },
    /// `block_scale_k16_m128x4_v1`: E2M1 codes, swizzled E4M3FN G16 scales, f32 divisor.
    Nvfp4,
    /// `row_scale_v1`: E4M3FN codes, one bf16 multiplier per row.
    Fp8Row,
    /// `gguf_blocks_v1`: ggml quantized blocks, rows back to back.
    Gguf(QuantFormat),
}

struct Object {
    shape: Vec<usize>,
    codec: Result<Codec, String>,
    offset: usize,
    bytes: usize,
}

/// One element range `[begin, end)` of an object, in logical C order.
struct Part {
    object: String,
    begin: usize,
    end: usize,
}

/// How one transformers tensor is assembled from catalog parts.
enum Recipe {
    Cat(Vec<Part>),
    /// Per head: `head_dim` query rows, then `head_dim` gate rows (transformers' `q_proj` of a
    /// gated attention).
    Interleave(Vec<Part>, Vec<Part>),
    /// `[taps, channels]` stored, `[channels, taps]` wanted.
    Transpose(Vec<Part>, usize),
}

impl Recipe {
    fn parts(&self) -> impl Iterator<Item = &Part> {
        let (a, b): (&[Part], &[Part]) = match self {
            Recipe::Cat(p) | Recipe::Transpose(p, _) => (p, &[]),
            Recipe::Interleave(q, g) => (q, g),
        };
        a.iter().chain(b)
    }
}

/// Row counts of the text model's projections.
struct Geometry {
    hidden: usize,
    vocab: usize,
    inter: usize,
    head_dim: usize,
    /// Query width (`heads * head_dim`), K/V width.
    q: usize,
    kv: usize,
    /// GDN key width, value width, value heads, value head dim, conv taps.
    kg: usize,
    vg: usize,
    nv: usize,
    dv: usize,
    taps: usize,
}

/// `config.json` for the hybrid causal-LM family spec, its dimensions, and every transformers
/// tensor with its assembly recipe.
struct Plan {
    cfg: Value,
    geo: Geometry,
    specs: Vec<(String, Vec<usize>, Recipe)>,
}

const fn align(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// `(container version, catalog length, catalog offset)` of the header at the start of a file.
fn header(head: &[u8]) -> Result<(u8, usize, usize), String> {
    if head.len() < 16 || &head[..6] != b"NINFER" || head[6] != 0 {
        return Err("not a NInfer artifact".into());
    }
    let json_len = u64::from_le_bytes(head[8..16].try_into().unwrap()) as usize;
    Ok((head[7], json_len, if head[7] == 3 { 32 } else { 16 }))
}

/// The parsed JSON catalog of an artifact, without its payload.
pub struct Catalog {
    root: Value,
    v3: bool,
    objects: HashMap<String, Object>,
}

impl Catalog {
    fn new(version: u8, root: Value) -> Result<Self, String> {
        if !matches!(version, 2 | 3) {
            return Err(format!(
                "NInfer container v{version} is not supported (v2 and v3 are)"
            ));
        }
        let mut objects = HashMap::new();
        for o in root["objects"]
            .as_array()
            .ok_or("catalog has no `objects`")?
        {
            let id = o["id"]
                .as_str()
                .or(o["name"].as_str())
                .ok_or("object without id")?;
            let shape: Vec<usize> = o["shape"]
                .as_array()
                .map(|s| {
                    s.iter()
                        .filter_map(|d| d.as_u64().map(|d| d as usize))
                        .collect()
                })
                .unwrap_or_default();
            let (offset, bytes) = (
                o["offset"].as_u64().unwrap_or(0) as usize,
                o["bytes"].as_u64().unwrap_or(0) as usize,
            );
            let codec = if o["kind"] == "tensor" {
                codec(
                    o["format"].as_str().unwrap_or(""),
                    o["layout"].as_str().unwrap_or(""),
                    &shape,
                    bytes,
                )
            } else {
                Err("resource".into())
            };
            objects.insert(
                id.to_string(),
                Object {
                    shape,
                    codec,
                    offset,
                    bytes,
                },
            );
        }
        Ok(Self {
            root,
            v3: version == 3,
            objects,
        })
    }

    fn has_object(&self, name: &str) -> bool {
        if self.v3 {
            !self.root["bindings"][name].is_null()
        } else {
            self.objects.contains_key(name)
        }
    }

    /// The parts of logical text parameter `name` (`text/layers/3/attention/query`): its binding
    /// (v3), or the same-named object / its row slice of a fused one (v2: `query_key`,
    /// `gate_value`, `value_z`, `gate_up`).
    fn parts(&self, name: &str, g: &Geometry) -> Result<Vec<Part>, String> {
        if self.v3 {
            let b = &self.root["bindings"][name];
            if let Some(o) = b["object"].as_str() {
                let n = self.objects.get(o).map_or(0, |o| o.shape.iter().product());
                return Ok(vec![Part {
                    object: o.into(),
                    begin: 0,
                    end: n,
                }]);
            }
            return b["parts"]
                .as_array()
                .ok_or_else(|| format!("no binding for '{name}'"))?
                .iter()
                .map(|p| {
                    let r = p["range"]
                        .as_array()
                        .and_then(|r| Some((r[0].as_u64()? as usize, r[1].as_u64()? as usize)));
                    let object = p["object"].as_str().ok_or("part without object")?;
                    let (begin, end) = r.ok_or("part without range")?;
                    Ok(Part {
                        object: object.into(),
                        begin,
                        end,
                    })
                })
                .collect();
        }
        let (stem, role) = name.rsplit_once('/').ok_or("bad parameter name")?;
        let attn = stem.ends_with("attention");
        // (fused object, rows above the slice, rows in it)
        let fused = match role {
            "query" if attn => Some(("query_key", 0, g.q)),
            "key" if attn => Some(("query_key", g.q, g.kv)),
            "gate" if attn => Some(("gate_value", 0, g.q)),
            "value" if attn => Some(("gate_value", g.q, g.kv)),
            "query" => Some(("query_key", 0, g.kg)),
            "key" => Some(("query_key", g.kg, g.kg)),
            "value" => Some(("value_z", 0, g.vg)),
            "z" => Some(("value_z", g.vg, g.vg)),
            "gate" => Some(("gate_up", 0, g.inter)),
            "up" => Some(("gate_up", g.inter, g.inter)),
            _ => None,
        };
        let (object, lo, rows) = match fused {
            Some((o, lo, rows)) => (format!("{stem}/{o}"), lo, rows),
            None => (name.to_string(), 0, 0),
        };
        let o = self
            .objects
            .get(&object)
            .ok_or_else(|| format!("object '{object}' missing"))?;
        let k = o.shape.get(1).copied().unwrap_or(1);
        let all: usize = o.shape.iter().product();
        let (begin, end) = if fused.is_some() {
            (lo * k, (lo + rows) * k)
        } else {
            (0, all)
        };
        if end > all {
            return Err(format!(
                "object '{object}' has {all} elements, '{name}' needs {end}"
            ));
        }
        Ok(vec![Part { object, begin, end }])
    }

    /// `config.json` and dimensions of the text model. v3 carries the text config; for v2 it is
    /// read off the tensor shapes with the fixed hybrid constants (GDN heads are 128 wide, RoPE
    /// 25 % interleaved M-RoPE `[11, 11, 10]` at theta 1e7, eps 1e-6).
    fn config(&self) -> Result<(Value, Geometry), String> {
        let ns = crate::arch::registry()?.ninfer_runtime().unwrap_or_default();
        let want_arch = ns["architecture"].as_str().unwrap_or_default();
        let mut cfg = if self.v3 {
            let c = &self.root["components"]["text"]["config"];
            match c["architectures"][0].as_str() {
                Some(a) if a == want_arch => c.clone(),
                Some(a)
                    if ns["refused_architectures"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|r| r.as_str() == Some(a)) =>
                {
                    return Err(format!(
                        "NInfer architecture {a}: only the dense {want_arch} text model is implemented (no expert-routed MLP)"
                    ))
                }
                a => {
                    return Err(format!(
                        "NInfer architecture {a:?} is not a supported text model"
                    ))
                }
            }
        } else {
            self.v2_config()?
        };
        let n = |k: &str| {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("NInfer text config has no {k}"))
        };
        let get = |k: &str| n(k).unwrap_or(0);
        let geo = Geometry {
            hidden: n("hidden_size")?,
            vocab: n("vocab_size")?,
            inter: n("intermediate_size")?,
            head_dim: get("head_dim"),
            q: get("num_attention_heads") * get("head_dim"),
            kv: get("num_key_value_heads") * get("head_dim"),
            kg: get("linear_num_key_heads") * get("linear_key_head_dim"),
            vg: get("linear_num_value_heads") * get("linear_value_head_dim"),
            nv: get("linear_num_value_heads"),
            dv: get("linear_value_head_dim"),
            taps: get("linear_conv_kernel_dim"),
        };
        // The artifact's head is stored apart; transformers' "tied" flag would make the loader skip it.
        cfg["tie_word_embeddings"] = json!(!self.has_object("text/output_head"));
        cfg["attn_output_gate"] = json!(true);
        cfg["rope_parameters"]["mrope_interleaved"] = json!(true);
        Ok((cfg, geo))
    }

    fn v2_config(&self) -> Result<Value, String> {
        let ns = crate::arch::registry()?.ninfer_runtime().unwrap_or_default();
        let want_arch = ns["architecture"].as_str().unwrap_or_default();
        let model_type = ns["model_type"].as_str().unwrap_or_default();
        let id = self.root["identity"]["model_id"].as_str().unwrap_or("?");
        let shape = |o: &str| {
            self.objects
                .get(o)
                .map(|o| o.shape.clone())
                .ok_or_else(|| format!("NInfer '{id}': object '{o}' missing"))
        };
        let emb = shape("text/token_embedding")?;
        let layers = (0..)
            .take_while(|i| {
                self.objects
                    .contains_key(&format!("text/layers/{i}/input_norm"))
            })
            .count();
        let (mut kinds, mut attn, mut gdn) = (Vec::new(), None, None);
        for i in 0..layers {
            let t = format!("text/layers/{i}");
            if !self.objects.contains_key(&format!("{t}/mlp/gate_up")) {
                return Err(format!(
                    "NInfer '{id}' has no dense MLP objects (a mixture-of-experts model: only dense {want_arch} text models are implemented)"
                ));
            }
            if self
                .objects
                .contains_key(&format!("{t}/attention/query_key"))
            {
                kinds.push("full_attention");
                attn = Some(i);
            } else {
                kinds.push("linear_attention");
                gdn = Some(i);
            }
        }
        let mut cfg = json!({
            "architectures": [want_arch], "model_type": model_type,
            "hidden_size": emb[1], "vocab_size": emb[0], "num_hidden_layers": layers,
            "intermediate_size": shape("text/layers/0/mlp/gate_up")?[0] / 2,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-6, "layer_types": kinds,
            "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
        });
        if let Some(i) = attn {
            let t = format!("text/layers/{i}/attention");
            let d = shape(&format!("{t}/query_norm"))?[0];
            let q = shape(&format!("{t}/output"))?[1];
            cfg["head_dim"] = json!(d);
            cfg["num_attention_heads"] = json!(q / d);
            cfg["num_key_value_heads"] = json!((shape(&format!("{t}/query_key"))?[0] - q) / d);
        }
        if let Some(i) = gdn {
            let t = format!("text/layers/{i}/gdn");
            cfg["linear_key_head_dim"] = json!(128);
            cfg["linear_num_key_heads"] = json!(shape(&format!("{t}/query_key"))?[0] / 2 / 128);
            cfg["linear_value_head_dim"] = json!(shape(&format!("{t}/norm"))?[0]);
            cfg["linear_num_value_heads"] = json!(shape(&format!("{t}/a_projection"))?[0]);
            cfg["linear_conv_kernel_dim"] = json!(shape(&format!("{t}/convolution"))?[0]);
        }
        Ok(cfg)
    }

    /// Every transformers tensor of the text model with its recipe, after checking that the
    /// artifact is a dense text model whose parts all have a decodable codec and add up to
    /// the tensor's shape. Errs with the reason otherwise.
    fn plan(&self) -> Result<Plan, String> {
        let (mut cfg, geo) = self.config()?;
        let h = geo.hidden;
        let kinds = cfg["layer_types"]
            .as_array()
            .ok_or("NInfer text config has no layer_types")?;
        let mut specs: Vec<(String, Vec<usize>, Recipe)> = Vec::new();
        // Parts of the logical parameters `names`, concatenated.
        let cat = |names: &[&str]| -> Result<Recipe, String> {
            let mut parts = Vec::new();
            for n in names {
                parts.extend(self.parts(n, &geo)?);
            }
            Ok(Recipe::Cat(parts))
        };
        for (i, kind) in kinds.iter().enumerate() {
            let (t, l) = (format!("text/layers/{i}"), format!("model.layers.{i}"));
            let mut add = |tail: &str, shape: &[usize], names: &[&str]| -> Result<(), String> {
                let names: Vec<String> = names.iter().map(|n| format!("{t}/{n}")).collect();
                let names: Vec<&str> = names.iter().map(String::as_str).collect();
                specs.push((format!("{l}.{tail}"), shape.to_vec(), cat(&names)?));
                Ok(())
            };
            add("input_layernorm.weight", &[h], &["input_norm"])?;
            add(
                "post_attention_layernorm.weight",
                &[h],
                &["post_attention_norm"],
            )?;
            add("mlp.gate_proj.weight", &[geo.inter, h], &["mlp/gate"])?;
            add("mlp.up_proj.weight", &[geo.inter, h], &["mlp/up"])?;
            add("mlp.down_proj.weight", &[h, geo.inter], &["mlp/down"])?;
            if kind == "full_attention" {
                add("self_attn.k_proj.weight", &[geo.kv, h], &["attention/key"])?;
                add(
                    "self_attn.v_proj.weight",
                    &[geo.kv, h],
                    &["attention/value"],
                )?;
                add(
                    "self_attn.o_proj.weight",
                    &[h, geo.q],
                    &["attention/output"],
                )?;
                add(
                    "self_attn.q_norm.weight",
                    &[geo.head_dim],
                    &["attention/query_norm"],
                )?;
                add(
                    "self_attn.k_norm.weight",
                    &[geo.head_dim],
                    &["attention/key_norm"],
                )?;
                let q = self.parts(&format!("{t}/attention/query"), &geo)?;
                let gate = self.parts(&format!("{t}/attention/gate"), &geo)?;
                specs.push((
                    format!("{l}.self_attn.q_proj.weight"),
                    vec![2 * geo.q, h],
                    Recipe::Interleave(q, gate),
                ));
            } else {
                let channels = 2 * geo.kg + geo.vg;
                add(
                    "linear_attn.in_proj_qkv.weight",
                    &[channels, h],
                    &["gdn/query", "gdn/key", "gdn/value"],
                )?;
                add("linear_attn.in_proj_z.weight", &[geo.vg, h], &["gdn/z"])?;
                add(
                    "linear_attn.in_proj_b.weight",
                    &[geo.nv, h],
                    &["gdn/b_projection"],
                )?;
                add(
                    "linear_attn.in_proj_a.weight",
                    &[geo.nv, h],
                    &["gdn/a_projection"],
                )?;
                add("linear_attn.A_log", &[geo.nv], &["gdn/a_log"])?;
                add("linear_attn.dt_bias", &[geo.nv], &["gdn/dt_bias"])?;
                add("linear_attn.norm.weight", &[geo.dv], &["gdn/norm"])?;
                add("linear_attn.out_proj.weight", &[h, geo.vg], &["gdn/output"])?;
                // [taps, channels] in the artifact, [channels, 1, taps] in transformers.
                let conv = self.parts(&format!("{t}/gdn/convolution"), &geo)?;
                specs.push((
                    format!("{l}.linear_attn.conv1d.weight"),
                    vec![channels, 1, geo.taps],
                    Recipe::Transpose(conv, geo.taps),
                ));
            }
        }
        specs.push((
            "model.embed_tokens.weight".into(),
            vec![geo.vocab, h],
            cat(&["text/token_embedding"])?,
        ));
        specs.push((
            "model.norm.weight".into(),
            vec![h],
            cat(&["text/final_norm"])?,
        ));
        // The head is its own (often higher-precision) object even when the config ties it.
        if self.has_object("text/output_head") {
            specs.push((
                "lm_head.weight".into(),
                vec![geo.vocab, h],
                cat(&["text/output_head"])?,
            ));
        }
        if self.v3 && self.has_object("vision/patch_embedding") {
            self.vision_specs(&mut cfg, &mut specs, &geo)?;
        }
        for (name, shape, recipe) in &specs {
            let mut elems = 0;
            for p in recipe.parts() {
                let o = self
                    .objects
                    .get(&p.object)
                    .ok_or_else(|| format!("{name}: object '{}' missing", p.object))?;
                let codec = o
                    .codec
                    .as_ref()
                    .map_err(|e| format!("{name}: object '{}': {e}", p.object))?;
                let k = o.shape.last().copied().unwrap_or(1);
                let total: usize = o.shape.iter().product();
                let rows_only = !matches!(codec, Codec::Bf16 | Codec::Fp32 | Codec::Int32);
                if p.begin >= p.end
                    || p.end > total
                    || (rows_only && (o.shape.len() != 2 || p.begin % k != 0 || p.end % k != 0))
                {
                    return Err(format!(
                        "{name}: range {}..{} does not fit object '{}' {:?}",
                        p.begin, p.end, p.object, o.shape
                    ));
                }
                elems += p.end - p.begin;
            }
            if elems != shape.iter().product::<usize>() {
                return Err(format!(
                    "{name}: its parts hold {elems} elements, shape {shape:?} needs {}",
                    shape.iter().product::<usize>()
                ));
            }
        }
        Ok(Plan { cfg, geo, specs })
    }

    /// The vision tower of a v3 artifact as transformers `model.visual.*` tensors (all bf16
    /// objects, fused QKV rebuilt from its three parts) and `vision_config`.
    fn vision_specs(
        &self,
        cfg: &mut Value,
        specs: &mut Vec<(String, Vec<usize>, Recipe)>,
        g: &Geometry,
    ) -> Result<(), String> {
        let mut vc = self.root["components"]["vision"]["config"].clone();
        let get = |k: &str| {
            vc[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("NInfer vision config has no {k}"))
        };
        let (d, depth, inter, patch, temporal, merge, pos) = (
            get("hidden_size")?,
            get("depth")?,
            get("intermediate_size")?,
            get("patch_size")?,
            get("temporal_patch_size")?,
            get("spatial_merge_size")?,
            get("num_position_embeddings")?,
        );
        vc["hidden_act"] = json!("gelu_pytorch_tanh");
        cfg["vision_config"] = vc;
        let m = d * merge * merge;
        let mut add = |hf: String, shape: &[usize], names: &[&str]| -> Result<(), String> {
            let mut parts = Vec::new();
            for n in names {
                parts.extend(self.parts(&format!("vision/{n}"), g)?);
            }
            specs.push((
                format!("model.visual.{hf}"),
                shape.to_vec(),
                Recipe::Cat(parts),
            ));
            Ok(())
        };
        add(
            "patch_embed.proj.weight".into(),
            &[d, 3, temporal, patch, patch],
            &["patch_embedding"],
        )?;
        add(
            "patch_embed.proj.bias".into(),
            &[d],
            &["patch_embedding_bias"],
        )?;
        add(
            "pos_embed.weight".into(),
            &[pos, d],
            &["position_embedding"],
        )?;
        for i in 0..depth {
            let (b, l) = (format!("blocks.{i}"), format!("layers/{i}"));
            for n in ["norm1", "norm2"] {
                add(
                    format!("{b}.{n}.weight"),
                    &[d],
                    &[&format!("{l}/{n}_weight")],
                )?;
                add(format!("{b}.{n}.bias"), &[d], &[&format!("{l}/{n}_bias")])?;
            }
            let qkv = ["query", "key", "value"].map(|r| format!("{l}/attention/{r}"));
            let qkv_bias = qkv.clone().map(|n| n + "_bias");
            add(
                format!("{b}.attn.qkv.weight"),
                &[3 * d, d],
                &qkv.each_ref().map(String::as_str),
            )?;
            add(
                format!("{b}.attn.qkv.bias"),
                &[3 * d],
                &qkv_bias.each_ref().map(String::as_str),
            )?;
            for (hf, n, shape) in [
                ("attn.proj", "attention/output", [d, d]),
                ("mlp.linear_fc1", "mlp/fc1", [inter, d]),
                ("mlp.linear_fc2", "mlp/fc2", [d, inter]),
            ] {
                add(format!("{b}.{hf}.weight"), &shape, &[&format!("{l}/{n}")])?;
                add(
                    format!("{b}.{hf}.bias"),
                    &shape[..1],
                    &[&format!("{l}/{n}_bias")],
                )?;
            }
        }
        add("merger.norm.weight".into(), &[d], &["merger/norm_weight"])?;
        add("merger.norm.bias".into(), &[d], &["merger/norm_bias"])?;
        add("merger.linear_fc1.weight".into(), &[m, m], &["merger/fc1"])?;
        add("merger.linear_fc1.bias".into(), &[m], &["merger/fc1_bias"])?;
        add(
            "merger.linear_fc2.weight".into(),
            &[g.hidden, m],
            &["merger/fc2"],
        )?;
        add(
            "merger.linear_fc2.bias".into(),
            &[g.hidden],
            &["merger/fc2_bias"],
        )?;
        Ok(())
    }

    /// Parameters of the text model (what the loader holds as bf16), or why it can't be loaded.
    pub fn text_params(&self) -> Result<u64, String> {
        Ok(self
            .plan()?
            .specs
            .iter()
            .map(|(_, s, _)| s.iter().product::<usize>() as u64)
            .sum())
    }

    /// Bytes of every stored object.
    fn stored_bytes(&self) -> u64 {
        self.objects.values().map(|o| o.bytes as u64).sum()
    }
}

/// A mapped artifact: catalog plus every payload segment.
pub struct Artifact {
    cat: Catalog,
    /// `(logical payload start, mapped file, offset of the payload in the file, payload length)`.
    segments: Vec<(usize, Mmap, usize, usize)>,
}

impl Artifact {
    /// Opens the entry file `path` (and its volumes). Errs on anything that is not a v2 / v3
    /// container.
    pub fn open(path: &Path) -> Result<Self, String> {
        let err = |e: String| format!("{}: {e}", path.display());
        let map = |p: &Path| -> Result<Mmap, String> {
            let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            // SAFETY: read-only mapping of a cache file nothing rewrites while the model loads.
            unsafe { Mmap::map(&f) }.map_err(|e| format!("{}: {e}", p.display()))
        };
        let entry = map(path)?;
        let (version, json_len, json_at) = header(&entry).map_err(err)?;
        let text = entry
            .get(json_at..json_at + json_len)
            .ok_or_else(|| err("truncated catalog".into()))?;
        let root: Value = serde_json::from_slice(text).map_err(|e| err(format!("catalog: {e}")))?;
        let start = align(json_at + json_len, 4096);
        let mut segments = Vec::new();
        if version == 3 {
            let files = root["files"]
                .as_array()
                .ok_or_else(|| err("no `files`".into()))?;
            let mut logical = 0;
            for (i, f) in files.iter().enumerate() {
                let len = f["payload_bytes"]
                    .as_u64()
                    .ok_or_else(|| err("bad `files`".into()))? as usize;
                let (m, at) = match f["path"].as_str() {
                    Some(name) if i > 0 => (map(&path.with_file_name(name))?, 4096),
                    _ => (map(path)?, start),
                };
                if m.len() < at + len {
                    return Err(err(format!(
                        "volume {i} is {} bytes, the catalog needs {}",
                        m.len(),
                        at + len
                    )));
                }
                segments.push((logical, m, at, len));
                logical += len;
            }
        } else {
            let len = entry
                .len()
                .checked_sub(start)
                .ok_or_else(|| err("no payload".into()))?;
            segments.push((0, entry, start, len));
        }
        let cat = Catalog::new(version, root).map_err(err)?;
        Ok(Self { cat, segments })
    }

    /// The payload bytes of object `id` (copied only when it straddles two volumes).
    fn payload(&self, id: &str) -> Result<Cow<'_, [u8]>, String> {
        let o = self
            .cat
            .objects
            .get(id)
            .ok_or_else(|| format!("object '{id}' missing"))?;
        let (lo, hi) = (o.offset, o.offset + o.bytes);
        let mut out: Vec<u8> = Vec::new();
        for (start, m, at, len) in &self.segments {
            let (a, b) = (lo.max(*start), hi.min(start + len));
            if a < b {
                let s = &m[at + (a - start)..at + (b - start)];
                if (a, b) == (lo, hi) {
                    return Ok(Cow::Borrowed(s));
                }
                out.extend_from_slice(s);
            }
        }
        if out.len() != o.bytes {
            return Err(format!("object '{id}' lies outside the payload"));
        }
        Ok(Cow::Owned(out))
    }

    /// A catalog resource (`tokenizer.json`, `chat_template.jinja`, `preprocessor_config.json`,
    /// ...) of the text or vision component.
    pub fn resource(&self, name: &str) -> Option<Vec<u8>> {
        let id = if self.cat.v3 {
            let c = &self.cat.root["components"];
            c["text"]["resources"][name]
                .as_str()
                .or(c["vision"]["resources"][name].as_str())?
                .to_string()
        } else {
            format!("frontend/{name}")
        };
        self.payload(&id).ok().map(Cow::into_owned)
    }

    /// Elements `[begin, end)` of `part.object` decoded to f32 (2-D codecs: whole rows, which
    /// [`Catalog::plan`] checked).
    fn decode(&self, part: &Part) -> Result<Vec<f32>, String> {
        let o = &self.cat.objects[&part.object];
        let codec = o
            .codec
            .clone()
            .map_err(|e| format!("object '{}': {e}", part.object))?;
        let raw = self.payload(&part.object)?;
        let (b, e) = (part.begin, part.end);
        let k = o.shape.last().copied().unwrap_or(1);
        let n = o.shape.iter().product::<usize>() / k.max(1);
        let words = |w: usize, f: &dyn Fn(&[u8]) -> f32| -> Result<Vec<f32>, String> {
            raw.get(b * w..e * w)
                .map(|s| s.chunks_exact(w).map(f).collect())
                .ok_or_else(|| format!("object '{}': range outside the object", part.object))
        };
        match codec {
            Codec::Bf16 => return words(2, &|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            Codec::Fp32 => return words(4, &|c| f32::from_le_bytes(c.try_into().unwrap())),
            Codec::Int32 => return words(4, &|c| i32::from_le_bytes(c.try_into().unwrap()) as f32),
            _ => {}
        }
        let (r0, rows) = (b / k, (e - b) / k);
        let mut out = vec![0.0f32; rows * k];
        match codec {
            Codec::Group { bits, group } => {
                let gpr = align(k, 128) / group;
                let (base_len, high_per) = (n * gpr * 32, (bits.saturating_sub(4)) * group / 8);
                let high_at = align(base_len, 256);
                let scale_at = high_at + align(n * gpr * high_per, 256);
                par_rows(&mut out, k, &|i, row| {
                    let g0 = (r0 + i) * gpr;
                    for (c, v) in row.iter_mut().enumerate() {
                        let (g, lane) = (g0 + c / group, c % group);
                        let q = if bits == 8 {
                            raw[g * 32 + lane] as i8 as i32
                        } else {
                            let byte = raw[g * 32 + lane / 2];
                            let mut u = (if lane % 2 == 0 { byte & 15 } else { byte >> 4 }) as u32;
                            for j in 0..bits - 4 {
                                let t = lane * (bits - 4) + j;
                                u |= (((raw[high_at + g * high_per + t / 8] >> (t % 8)) & 1)
                                    as u32)
                                    << (4 + j);
                            }
                            ((u << (32 - bits)) as i32) >> (32 - bits)
                        };
                        let s = &raw[scale_at + g * 2..][..2];
                        *v = q as f32 * f16_to_f32(u16::from_le_bytes([s[0], s[1]]));
                    }
                });
            }
            Codec::Nvfp4 => {
                let scale_at = align(n * k / 2, 256);
                let div_at = scale_at + n * k / 16;
                let divisor = f32::from_le_bytes(raw[div_at..div_at + 4].try_into().unwrap());
                let k_tiles = k / 64;
                par_rows(&mut out, k, &|i, row| {
                    let r = r0 + i;
                    for (c, v) in row.iter_mut().enumerate() {
                        let g = c / 16;
                        let s_at = scale_at
                            + ((r / 128) * k_tiles + g / 4) * 512
                            + (r % 128 % 32) * 16
                            + (r % 128 / 32) * 4
                            + g % 4;
                        let byte = raw[r * (k / 2) + c / 2];
                        let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                        *v = NVFP4_E2M1_LUT[code as usize] * f8_e4m3_to_f32(raw[s_at]) / divisor;
                    }
                });
            }
            Codec::Fp8Row => {
                let scale_at = align(n * k, 256);
                par_rows(&mut out, k, &|i, row| {
                    let r = r0 + i;
                    let s = &raw[scale_at + 2 * r..][..2];
                    let s = bf16_to_f32(u16::from_le_bytes([s[0], s[1]]));
                    for (c, v) in row.iter_mut().enumerate() {
                        *v = f8_e4m3_to_f32(raw[r * k + c]) * s;
                    }
                });
            }
            Codec::Gguf(f) => {
                let (per, bytes, _) =
                    crate::model::layers::gguf_block_codec(f).ok_or("no decoder")?;
                let row_bytes = k / per * bytes;
                let d = crate::model::layers::dequantize_gguf_blocks(
                    f,
                    &raw[r0 * row_bytes..(r0 + rows) * row_bytes],
                )
                .ok_or("no decoder")?;
                out.copy_from_slice(&d);
            }
            Codec::Bf16 | Codec::Fp32 | Codec::Int32 => unreachable!("decoded above"),
        }
        Ok(out)
    }

    fn read<'a>(&self, parts: impl IntoIterator<Item = &'a Part>) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        for p in parts {
            out.extend(self.decode(p)?);
        }
        Ok(out)
    }

    /// The model as transformers tensors (text, plus the vision tower when the artifact has one)
    /// and its `config.json` equivalent. Errs with the reason for any artifact outside the dense
    /// supported family or a codec we can't decode.
    pub fn checkpoint(&self) -> Result<(Weights, Value), String> {
        let Plan {
            mut cfg,
            geo,
            specs,
        } = self.cat.plan()?;
        if cfg["vision_config"].is_object() {
            let tok: Value = serde_json::from_slice(
                &self
                    .resource("tokenizer.json")
                    .ok_or("no tokenizer.json resource")?,
            )
            .map_err(|e| format!("tokenizer.json: {e}"))?;
            let id = tok["added_tokens"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|t| t["content"] == "<|image_pad|>");
            cfg["image_token_id"] = id
                .map(|t| t["id"].clone())
                .ok_or("the tokenizer has no <|image_pad|> token")?;
        }
        let need: usize = specs
            .iter()
            .map(|(_, s, _)| s.iter().product::<usize>() * 2)
            .sum();
        if let Some(total) = crate::hardware::HardwareCapability::probe_system()
            .total_memory_bytes
            .filter(|t| need as u64 > t / 10 * 9)
        {
            let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
            return Err(format!(
                "needs ~{:.0} GB as bf16; host has {:.0} GB",
                gb(need as u64),
                gb(total)
            ));
        }
        let by_name: HashMap<&str, &Recipe> =
            specs.iter().map(|(n, _, r)| (n.as_str(), r)).collect();
        let shapes = specs
            .iter()
            .map(|(n, s, _)| (n.clone(), s.clone()))
            .collect();
        // ponytail: `Weights::from_f32` rounds A_log / dt_bias (f32 in the artifact) to bf16 like the
        // GGUF path does; keep them f32 if a long-context drift shows up.
        let ck = Weights::from_f32(shapes, |n| match by_name[n] {
            Recipe::Cat(parts) => self.read(parts),
            Recipe::Interleave(q, gate) => {
                let (q, gate) = (self.read(q)?, self.read(gate)?);
                let run = geo.head_dim * (q.len() / geo.q);
                Ok(q.chunks_exact(run)
                    .zip(gate.chunks_exact(run))
                    .flat_map(|(a, b)| a.iter().chain(b))
                    .copied()
                    .collect())
            }
            Recipe::Transpose(parts, taps) => {
                let v = self.read(parts)?;
                let channels = v.len() / taps;
                Ok((0..channels)
                    .flat_map(|c| (0..*taps).map(move |t| (t, c)))
                    .map(|(t, c)| v[t * channels + c])
                    .collect())
            }
        })?;
        Ok((ck, cfg))
    }
}

/// The codec of a tensor object from its (v2 or v3 spelled) format and layout names, with the
/// object's `bytes` checked against the layout's encoded size.
fn codec(format: &str, layout: &str, shape: &[usize], bytes: usize) -> Result<Codec, String> {
    let norm = |s: &str| {
        s.to_lowercase()
            .replace('-', "_")
            .replace("blockscale", "block_scale")
    };
    let (f, l) = (norm(format), norm(layout));
    let n = shape.iter().rev().skip(1).product::<usize>().max(1);
    let k = shape.last().copied().unwrap_or(1);
    let elems: usize = shape.iter().product();
    let (codec, want, want_layout) = match f.as_str() {
        "bf16" => (Codec::Bf16, elems * 2, "contiguous_le_v1"),
        "fp32" => (Codec::Fp32, elems * 4, "contiguous_le_v1"),
        "int32" | "i32" => (Codec::Int32, elems * 4, "contiguous_le_v1"),
        "q4_g64_fp16" | "q4g64_f16s" | "q5_g64_fp16" | "q5g64_f16s" | "q6_g64_fp16" | "q6g64_f16s" | "q8_g32_fp16" | "w8g32_f16s" => {
            let bits = if f.starts_with('w') { 8 } else { (f.as_bytes()[1] - b'0') as usize };
            let group = if bits == 8 { 32 } else { 64 };
            let gpr = align(k, 128) / group;
            let high = align(n * gpr * bits.saturating_sub(4) * group / 8, 256);
            (Codec::Group { bits, group }, align(n * gpr * 32, 256) + high + n * gpr * 2, "row_split_k128_v1")
        }
        "nvfp4" => (Codec::Nvfp4, align(n * k / 2, 256) + n * k / 16 + 4, "block_scale_k16_m128x4_v1"),
        "fp8_e4m3fn_row_bf16" | "fp8_e4m3fn_row_bf16s" => (Codec::Fp8Row, align(n * k, 256) + 2 * n, "row_scale_v1"),
        g if g.starts_with("gguf_") => {
            let q = match &g[5..] {
                "q4_0" => QuantFormat::GGUF_Q4_0,
                "q5_0" => QuantFormat::GGUF_Q5_0,
                "q8_0" => QuantFormat::GGUF_Q8_0,
                "q4_k" => QuantFormat::GGUF_Q4_K_M,
                "q5_k" => QuantFormat::GGUF_Q5_K_M,
                "q6_k" => QuantFormat::GGUF_Q6_K,
                "q2_k" => QuantFormat::GGUF_Q2_K,
                "q3_k" => QuantFormat::GGUF_Q3_K,
                "iq4_nl" => QuantFormat::GGUF_IQ4_NL,
                "iq4_xs" => QuantFormat::GGUF_IQ4_XS,
                "iq3_xxs" => QuantFormat::GGUF_IQ3_XXS,
                "iq3_s" => QuantFormat::GGUF_IQ3_S,
                "iq2_xxs" => QuantFormat::GGUF_IQ2_XXS,
                "iq2_xs" => QuantFormat::GGUF_IQ2_XS,
                "iq2_s" => QuantFormat::GGUF_IQ2_S,
                "iq1_s" => QuantFormat::GGUF_IQ1_S,
                "iq1_m" => QuantFormat::GGUF_IQ1_M,
                _ => return Err(format!("format '{format}' has no decoder here")),
            };
            let (per, size, _) = crate::model::layers::gguf_block_codec(q).ok_or("no decoder")?;
            if k % per != 0 {
                return Err(format!("{k} columns are not whole {format} blocks"));
            }
            (Codec::Gguf(q), n * k / per * size, "gguf_blocks_v1")
        }
        _ => {
            return Err(format!(
                "format '{format}' ({layout}) is not supported (bf16, fp32, int32, q4/q5/q6/q8 groupwise, nvfp4, fp8 row-scaled and ggml block formats are)"
            ))
        }
    };
    if l != want_layout || bytes != want {
        return Err(format!(
            "format '{format}' / layout '{layout}' with {bytes} bytes does not match {shape:?} (expected '{want_layout}', {want} bytes)"
        ));
    }
    Ok(codec)
}

/// The root `.ninfer` file the loader would take from Hub search result `m`, if it ships one.
fn hub_file(m: &Value) -> Option<String> {
    let files: Vec<crate::model::hub::RepoFile> = m["siblings"]
        .as_array()?
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .map(|p| crate::model::hub::RepoFile {
            path: p.to_string(),
            size: None,
        })
        .collect();
    crate::model::hub::select_weight_files(&files, None)
        .ok()?
        .into_iter()
        .next()
        .filter(|f| f.ends_with(EXT))
}

/// Catalogs fetched this session, by `repo/file`.
static CATALOGS: std::sync::LazyLock<parking_lot::Mutex<HashMap<String, Value>>> =
    std::sync::LazyLock::new(Default::default);

fn cached_catalog(
    key: &str,
    fetch: impl FnOnce() -> Result<Value, String>,
) -> Result<Value, String> {
    if let Some(v) = CATALOGS.lock().get(key) {
        return Ok(v.clone());
    }
    let v = fetch()?;
    CATALOGS.lock().insert(key.to_string(), v.clone());
    Ok(v)
}

/// `{"version", "catalog"}` of Hub file `repo/file`: the 32-byte header and the JSON catalog behind
/// it, two Range requests.
fn fetch_catalog(repo: &str, file: &str, token: Option<&str>) -> Result<Value, String> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let get = |range: String| -> Result<Vec<u8>, String> {
        let mut cmd = std::process::Command::new("curl");
        cmd.args(["-fsSL", "-m", "20", "-r", &range]);
        if let Some(t) = token.filter(|t| !t.is_empty()) {
            cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
        }
        let out = cmd
            .arg(&url)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("curl: {e}"))?;
        if out.status.success() {
            Ok(out.stdout)
        } else {
            Err(String::from_utf8_lossy(&out.stderr)
                .lines()
                .next()
                .unwrap_or("curl failed")
                .to_string())
        }
    };
    let head = get("0-31".into())?;
    let (version, json_len, json_at) = header(&head)?;
    let body = get(format!("{json_at}-{}", json_at + json_len - 1))?;
    let catalog: Value = serde_json::from_slice(&body).map_err(|e| format!("catalog: {e}"))?;
    Ok(json!({"version": version, "catalog": catalog}))
}

/// Gives every Hub search result that ships a `.ninfer` artifact its catalog as `ninfer_catalog`
/// (or `ninfer_error`), fetched in parallel; [`hub_verdict`] reads it.
pub fn attach_catalogs(models: &mut [Value], token: Option<&str>) {
    std::thread::scope(|s| {
        for m in models.iter_mut() {
            let Some(file) = hub_file(m) else { continue };
            let repo = m["id"].as_str().unwrap_or("").to_string();
            s.spawn(move || {
                match cached_catalog(&format!("{repo}/{file}"), || {
                    fetch_catalog(&repo, &file, token)
                }) {
                    Ok(c) => m["ninfer_catalog"] = c,
                    Err(e) => m["ninfer_error"] = e.into(),
                }
            });
        }
    });
}

/// The catalog [`attach_catalogs`] fetched for `m`.
fn hub_catalog(m: &Value) -> Option<Result<Catalog, String>> {
    let c = m.get("ninfer_catalog")?;
    Some(Catalog::new(
        c["version"].as_u64()? as u8,
        c["catalog"].clone(),
    ))
}

/// Whether the Hub repo `m` (search JSON with siblings and tags, catalog attached) ships a
/// `.ninfer` artifact for a text generator: `None` for any other repo, `Some(Ok)` when the loader
/// reads it, `Some(Err(reason))` with the loader's own reason when it does not. Without a catalog
/// (offline, fetch failed) a name test for mixture-of-experts stands in for the architecture.
// ponytail: the name test is the offline fallback only; the catalog is the rule.
pub fn hub_verdict(m: &Value) -> Option<Result<(), String>> {
    hub_file(m)?;
    let text = matches!(
        m["pipeline_tag"].as_str(),
        None | Some("image-text-to-text" | "text-generation" | "conversational" | "any-to-any")
    );
    if !text {
        return None;
    }
    if let Some(cat) = hub_catalog(m) {
        return Some(cat.and_then(|c| c.plan().map(|_| ())));
    }
    let tags = m["tags"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    let hay = std::iter::once(m["id"].as_str().unwrap_or(""))
        .chain(tags)
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let ns = crate::arch::registry().ok()?.ninfer_runtime()?;
    let want_arch = ns["architecture"].as_str().unwrap_or_default();
    let hit = ns["hub_moe_keywords"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|k| hay.contains(*k));
    Some(match hit {
        Some(k) => Err(format!("NInfer artifact of a mixture-of-experts model ('{k}'): only dense {want_arch} text models are implemented (no expert-routed MLP)")),
        None => Ok(()),
    })
}

/// Parameters the loader holds for the `.ninfer` repo `m` (from its attached catalog).
pub fn hub_params(m: &Value) -> Option<u64> {
    hub_catalog(m)?.ok()?.text_params().ok()
}

/// Bytes of the artifact's objects (the download size, give or take the container padding).
pub fn hub_size(m: &Value) -> Option<u64> {
    hub_catalog(m)?.ok().map(|c| c.stored_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::hub::ModelArtifacts;
    use crate::model::loaders::{SafetensorsWeightReader, WeightReader};
    use std::path::PathBuf;

    /// One dense gated-delta-net layer (hidden 64) as a v3 catalog: every parameter its own bf16
    /// object, `mlp/down` stored as `down_format`.
    fn tiny_catalog(down_format: &str) -> Value {
        let params: [(&str, &[usize]); 18] = [
            ("text/token_embedding", &[32, 64]),
            ("text/final_norm", &[64]),
            ("text/layers/0/input_norm", &[64]),
            ("text/layers/0/post_attention_norm", &[64]),
            ("text/layers/0/mlp/gate", &[128, 64]),
            ("text/layers/0/mlp/up", &[128, 64]),
            ("text/layers/0/mlp/down", &[64, 128]),
            ("text/layers/0/gdn/query", &[8, 64]),
            ("text/layers/0/gdn/key", &[8, 64]),
            ("text/layers/0/gdn/value", &[16, 64]),
            ("text/layers/0/gdn/z", &[16, 64]),
            ("text/layers/0/gdn/b_projection", &[2, 64]),
            ("text/layers/0/gdn/a_projection", &[2, 64]),
            ("text/layers/0/gdn/a_log", &[2]),
            ("text/layers/0/gdn/dt_bias", &[2]),
            ("text/layers/0/gdn/norm", &[8]),
            ("text/layers/0/gdn/output", &[64, 16]),
            ("text/layers/0/gdn/convolution", &[4, 32]),
        ];
        let (mut objects, mut bindings, mut offset) = (Vec::new(), serde_json::Map::new(), 0);
        for (name, shape) in params {
            let bytes = shape.iter().product::<usize>() * 2;
            let format = if name.ends_with("mlp/down") {
                down_format
            } else {
                "bf16"
            };
            objects.push(json!({"id": format!("w/{name}"), "kind": "tensor", "shape": shape, "format": format, "layout": "contiguous_le_v1", "offset": offset, "bytes": bytes}));
            bindings.insert(name.into(), json!({"object": format!("w/{name}")}));
            offset += align(bytes, 256);
        }
        json!({
            "components": {"text": {"config": {
                "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "hidden_size": 64,
                "vocab_size": 32, "num_hidden_layers": 1, "intermediate_size": 128, "rms_norm_eps": 1e-6,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 256,
                "layer_types": ["linear_attention"], "linear_num_key_heads": 1, "linear_key_head_dim": 8,
                "linear_num_value_heads": 2, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
                "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
            }}},
            "objects": objects, "bindings": bindings,
        })
    }

/// Writes the [`tiny_catalog`] artifact (bf16 objects of known values) to `path`; returns each
/// object's payload offset. Fixture of `tests/cpu_free_matrix.rs` and this module's tests.
pub fn write_tiny_artifact(path: &Path) -> Result<HashMap<String, usize>, String> {
    let mut root = tiny_catalog("bf16");
    let mut payload = Vec::new();
    let mut off = HashMap::new();
    for o in root["objects"].as_array().ok_or("no objects")? {
        let at = o["offset"].as_u64().ok_or("no offset")? as usize;
        let n = o["shape"]
            .as_array()
            .ok_or("no shape")?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0) as usize)
            .product::<usize>();
        payload.resize(at, 0);
        off.insert(o["id"].as_str().unwrap_or_default().to_string(), at);
        payload.extend(
            (0..n).flat_map(|i| crate::model::f32_to_bf16(((i * 5 + at) % 11) as f32 / 4.0).to_le_bytes()),
        );
    }
    root["files"] = json!([{"path": null, "payload_bytes": payload.len()}]);
    let catalog = serde_json::to_vec(&root).map_err(|e| e.to_string())?;
    let mut file = b"NINFER\0\x03".to_vec();
    file.extend((catalog.len() as u64).to_le_bytes());
    file.extend([7u8; 16]);
    file.extend(&catalog);
    file.resize(align(file.len(), 4096), 0);
    file.extend(&payload);
    std::fs::write(path, file).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(off)
}

    fn snapshot(repo: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "disk/models--{}/snapshots/main",
            repo.replace('/', "--")
        ))
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// `ninfer` repo's artifact decoded tensor by tensor against the source checkpoint's tensors
    /// (`reference`: a safetensors holding at least `names`, transformers names with the
    /// `language_model.` infix), then a greedy completion through the normal native loader.
    fn matches_source(
        ninfer: &str,
        file: &str,
        reference: &str,
        layers: &[usize],
        embed_rows: usize,
    ) {
        let path = snapshot(ninfer).join(file);
        let refpath = snapshot(reference).join("model.safetensors");
        if !path.exists() || !refpath.exists() {
            println!("SKIP: {ninfer}/{file} or {reference} not in disk/");
            return;
        }
        let art = Artifact::open(&path).unwrap();
        let (ck, cfg) = art.checkpoint().unwrap();
        let src = SafetensorsWeightReader::new(refpath).unwrap();
        let mut names: Vec<String> = vec!["model.norm.weight".into()];
        for l in layers {
            let prefix = format!("model.layers.{l}.");
            names.extend(
                (cfg["layer_types"][*l] == "full_attention")
                    .then(|| {
                        [
                            "self_attn.q_proj",
                            "self_attn.k_proj",
                            "self_attn.v_proj",
                            "self_attn.o_proj",
                            "self_attn.q_norm",
                            "self_attn.k_norm",
                        ]
                        .as_slice()
                    })
                    .unwrap_or(&[
                        "linear_attn.in_proj_qkv",
                        "linear_attn.in_proj_z",
                        "linear_attn.in_proj_a",
                        "linear_attn.in_proj_b",
                        "linear_attn.out_proj",
                        "linear_attn.conv1d",
                        "linear_attn.norm",
                    ])
                    .iter()
                    .chain(&[
                        "input_layernorm",
                        "post_attention_layernorm",
                        "mlp.gate_proj",
                        "mlp.up_proj",
                        "mlp.down_proj",
                    ])
                    .map(|n| format!("{prefix}{n}.weight")),
            );
            if cfg["layer_types"][*l] != "full_attention" {
                names.extend(["A_log", "dt_bias"].map(|n| format!("{prefix}linear_attn.{n}")));
            }
        }
        names.push("model.embed_tokens.weight".into());
        let mut worst = 1.0f64;
        for n in &names {
            let mut want = src
                .load_tensor(&n.replace("model.", "model.language_model."))
                .unwrap();
            let mut got = if ck.shape(n).unwrap().len() == 1 || n.contains("conv1d") {
                ck.vec(n).unwrap()
            } else {
                ck.mat(n).unwrap().to_f32()
            };
            if n.contains("embed") {
                got.truncate(embed_rows * ck.shape(n).unwrap()[1]);
                want.truncate(got.len());
            }
            assert_eq!(got.len(), want.len(), "{n}");
            let cos = cosine(&got, &want);
            println!("{ninfer} {n}: cos {cos:.5}");
            worst = worst.min(cos);
        }
        println!("{ninfer}: worst cos {worst:.5}");
        assert!(worst >= 0.98, "{ninfer}: cosine {worst} below 0.98");

        let artifacts = ModelArtifacts {
            model_id: ninfer.into(),
            weights_path: Some(path),
            ..Default::default()
        };
        let lm = crate::runtime::lm::load(
            &artifacts,
            crate::arch::registry().unwrap(),
            None,
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Cpu),
        )
        .expect("a .ninfer file loads as a spec LM");
        println!("{ninfer}: {}", lm.summary());
        let msgs = [crate::model::hf_loader::ChatMessage::new(
            "user",
            "What is the capital of France? Answer briefly.",
        )];
        let prompt = lm
            .tokenizer()
            .chat_template
            .as_ref()
            .expect("the artifact's chat template")
            .render(
                &msgs,
                None,
                true,
                &serde_json::json!({"enable_thinking": false}),
            )
            .unwrap();
        let (ids, mut st) = lm.prepare(lm.tokenizer().encode(&prompt), &[]).unwrap();
        let mut logits = lm.forward(&ids, 0, &mut st).unwrap();
        let mut out = Vec::new();
        for _ in 0..8 {
            let t = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap() as u32;
            out.push(t);
            logits = lm
                .forward(&[t], ids.len() + out.len() - 1, &mut st)
                .unwrap();
        }
        let text = lm.tokenizer().decode(&out);
        println!("{ninfer}: {text:?}");
        assert!(text.contains("Paris"), "{ninfer}: {text:?}");
    }

    /// v2 `groupwise-int` Qwen3.5-0.8B (Q4/Q5/Q6/W8 groupwise objects, fused projections).
    #[test]
    fn v2_groupwise_qwen35_0_8b() {
        matches_source(
            "v381654729/qwen3.5-0.8b-ninfer",
            "qwen3_5_0_8b.ninfer",
            "Qwen/Qwen3.5-0.8B",
            &[0, 2, 3],
            4096,
        );
    }

    /// The same Qwen3.5-0.8B as a 4-6 bit NInfer artifact and as llama.cpp's near-lossless Q8_0:
    /// last-position logits of a chat prompt agree (cosine) and both pick the same first tokens.
    #[test]
    fn v2_logits_track_q8_gguf_twin() {
        let (nin, gg) = (
            snapshot("v381654729/qwen3.5-0.8b-ninfer").join("qwen3_5_0_8b.ninfer"),
            snapshot("unsloth/Qwen3.5-0.8B-GGUF").join("Qwen3.5-0.8B-Q8_0.gguf"),
        );
        if !nin.exists() || !gg.exists() {
            println!("SKIP: the 0.8B NInfer artifact or its Q8_0 GGUF twin is not in disk/");
            return;
        }
        let load = |p: PathBuf| {
            let a = ModelArtifacts {
                weights_path: Some(p),
                ..Default::default()
            };
            crate::runtime::lm::load(
                &a,
                crate::arch::registry().unwrap(),
                None,
                crate::hardware::PlacementMode::Pin(crate::hardware::Device::Cpu),
            )
            .unwrap()
        };
        let (a, b) = (load(nin), load(gg));
        let msgs = [crate::model::hf_loader::ChatMessage::new(
            "user",
            "What is the capital of France? Answer briefly.",
        )];
        let prompt = a
            .tokenizer()
            .chat_template
            .as_ref()
            .unwrap()
            .render(
                &msgs,
                None,
                true,
                &serde_json::json!({"enable_thinking": false}),
            )
            .unwrap();
        let ids = a.tokenizer().encode(&prompt);
        assert_eq!(b.tokenizer().encode(&prompt), ids, "tokenizers disagree");
        let last = |lm: &dyn crate::model::native::NativeLm| {
            let (ids, mut st) = lm.prepare(ids.clone(), &[]).unwrap();
            lm.forward(&ids, 0, &mut st).unwrap()
        };
        let (la, lb) = (last(a.as_ref()), last(b.as_ref()));
        let top = |l: &[f32]| (0..l.len()).max_by(|&x, &y| l[x].total_cmp(&l[y])).unwrap();
        let cos = cosine(&la, &lb);
        println!(
            "last-position logits: cosine {cos:.5}, top-1 {} vs {}",
            top(&la),
            top(&lb)
        );
        for (n, l) in [("ninfer", &la), ("gguf", &lb)] {
            let mut o: Vec<usize> = (0..l.len()).collect();
            o.sort_by(|&x, &y| l[y].total_cmp(&l[x]));
            println!(
                "{n}: {:?}",
                o[..6]
                    .iter()
                    .map(|&t| (t, l[t], a.tokenizer().decode(&[t as u32])))
                    .collect::<Vec<_>>()
            );
        }
        // 4-6 bit weights against Q8_0: noisy logits, but the same candidate answers.
        assert!(cos > 0.9, "logits cosine {cos}");
        let mut o: Vec<usize> = (0..lb.len()).collect();
        o.sort_by(|&x, &y| lb[y].total_cmp(&lb[x]));
        assert!(
            o[..3].contains(&top(&la)),
            "ninfer top-1 {} not in the twin's top 3",
            top(&la)
        );
    }


    /// The picker's verdict and the loader's plan read one catalog: a codec outside the supported set
    /// refuses the artifact with the codec named, in both.
    #[test]
    fn catalog_verdict_names_the_unsupported_codec() {
        let params = Catalog::new(3, tiny_catalog("bf16"))
            .unwrap()
            .text_params()
            .unwrap();
        let m = |down: &str| json!({"id": "x/y", "siblings": [{"rfilename": "y.ninfer"}], "ninfer_catalog": {"version": 3, "catalog": tiny_catalog(down)}});
        assert_eq!(hub_verdict(&m("bf16")), Some(Ok(())));
        assert_eq!(hub_params(&m("bf16")), Some(params));
        let why = hub_verdict(&m("t2_g128_fp16")).unwrap().unwrap_err();
        assert!(
            why.contains("mlp.down_proj") && why.contains("t2_g128_fp16"),
            "{why}"
        );
        // Not a root `.ninfer` file: some other repo.
        assert_eq!(
            hub_verdict(&json!({"id": "x/y", "siblings": [{"rfilename": "m.safetensors"}]})),
            None
        );
    }

    /// A v3 file of [`tiny_catalog`] (bf16 objects of known values): the temp dir holding it, its
    /// path, and each object's payload offset.
    fn write_tiny() -> (tempfile::TempDir, std::path::PathBuf, HashMap<String, usize>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.ninfer");
        let off = write_tiny_artifact(&path).unwrap();
        (dir, path, off)
    }

    /// The artifact loads through the `qwen3_5` spec (detected from its catalog) and its Metal
    /// program matches the CPU one: prefill 5 tokens and decode 3 (the tiny artifact ships no
    /// tokenizer resource).
    #[test]
    fn v3_artifact_metal_program_matches_host() {
        use crate::hardware::{cpu_compute, Device, PlacementMode};
        use crate::model::native::NativeLm;
        let _g = cpu_compute::allow();
        // The CPU program is the exact reference.
        let _exact = crate::model::qdot::exact();
        let (_dir, path, _) = write_tiny();
        let a = ModelArtifacts {
            model_id: "x/tiny".into(),
            weights_path: Some(path),
            ..Default::default()
        };
        let m = |d| {
            let mut s = crate::arch::load::load(&a, crate::arch::registry().unwrap(), None).unwrap();
            assert_eq!(s.spec.family, "qwen3_5");
            s.tokenizer = Some(std::sync::Arc::new(crate::model::tokenizer::Tokenizer::default_bpe()));
            crate::runtime::lm::SpecLm::new(s, PlacementMode::Pin(d)).unwrap()
        };
        let (cpu, gpu) = (m(Device::Cpu), m(Device::Metal));
        assert_eq!(gpu.device(), Device::Metal);
        let prompt = vec![1u32, 5, 9, 13, 2];
        let (_, mut sc) = cpu.prepare(prompt.clone(), &[]).unwrap();
        let (_, mut sg) = gpu.prepare(prompt.clone(), &[]).unwrap();
        let (mut a, mut b) = (
            cpu.forward(&prompt, 0, &mut sc).unwrap(),
            gpu.forward(&prompt, 0, &mut sg).unwrap(),
        );
        for step in 0..4 {
            let scale = a.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
            let err = a.iter().zip(&b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
            println!("ninfer step {step}: rel err {err:e}");
            assert!(err < 1e-3, "step {step}: {err}");
            let tok = a.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
            a = cpu.forward(&[tok], prompt.len() + step, &mut sc).unwrap();
            b = gpu.forward(&[tok], prompt.len() + step, &mut sg).unwrap();
        }
    }

    /// A v3 file written here (bf16 objects of known values) reads back as transformers tensors:
    /// fused projection rows in order, the convolution transposed.
    #[test]
    fn v3_file_reads_back_as_transformers_tensors() {
        let (_dir, path, off) = write_tiny();

        let (ck, cfg) = Artifact::open(&path).unwrap().checkpoint().unwrap();
        assert_eq!(cfg["tie_word_embeddings"], true, "no output_head object");
        let at = |name: &str, i: usize| {
            ((i * 5 + off[&format!("w/text/layers/0/gdn/{name}")]) % 11) as f32 / 4.0
        };
        let qkv = ck
            .mat("model.layers.0.linear_attn.in_proj_qkv.weight")
            .unwrap()
            .to_f32();
        assert_eq!(
            (qkv[3], qkv[8 * 64 + 3], qkv[16 * 64 + 5]),
            (at("query", 3), at("key", 3), at("value", 5))
        );
        let conv = ck.vec("model.layers.0.linear_attn.conv1d.weight").unwrap();
        assert_eq!(
            ck.shape("model.layers.0.linear_attn.conv1d.weight")
                .unwrap(),
            [32, 1, 4]
        );
        // stored [tap, channel]; transformers [channel, 1, tap]
        assert_eq!(
            (conv[3 * 4 + 2], conv[31 * 4 + 1]),
            (at("convolution", 2 * 32 + 3), at("convolution", 32 + 31))
        );
    }

    /// Layer-3 K and V projections of real v3 Qwen3.8-27B artifacts, one per codec, against the
    /// source checkpoint's tensors. The artifacts are the header, catalog and just the objects
    /// those two bindings use (a 20 GB file otherwise sparse): the codec, the bindings and the
    /// fused-parent row ranges are all exercised on real bytes.
    #[test]
    fn v3_codecs_match_qwen38_source() {
        let refpath = snapshot("Qwen/Qwen3.8-27B").join("model.safetensors");
        if !refpath.exists() {
            println!("SKIP: Qwen/Qwen3.8-27B layer-3 K/V subset not in disk/");
            return;
        }
        let src = SafetensorsWeightReader::new(refpath).unwrap();
        // (repo, file, codec of the K/V parent, minimum cosine to the source)
        let cases = [
            (
                "neroued/Qwen3.8-27B-NInfer",
                "qwen3_8_27b.ninfer",
                "q4_g64_fp16 K, q5_g64_fp16 V",
                0.98,
            ),
            (
                "neroued/Qwen3.8-27B-nvfp4-NInfer",
                "qwen3_8_27b_nvfp4.ninfer",
                "fp8_e4m3fn_row_bf16",
                0.995,
            ),
            (
                "Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer",
                "qwen3_8_27b_quasar_nvfp4.ninfer",
                "nvfp4 (QAT-tuned)",
                0.9,
            ),
            (
                "WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3",
                "Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer",
                "gguf_iq3_s",
                0.9,
            ),
        ];
        for (repo, file, what, min) in cases {
            let path = snapshot(repo).join(file);
            if !path.exists() {
                println!("SKIP: {repo} partial artifact not in disk/");
                continue;
            }
            let art = Artifact::open(&path).unwrap();
            let (_, geo) = art.cat.config().unwrap();
            for (role, hf) in [("key", "k_proj"), ("value", "v_proj")] {
                let parts = art
                    .cat
                    .parts(&format!("text/layers/3/attention/{role}"), &geo)
                    .unwrap();
                let got = art.read(&parts).unwrap();
                let want = src
                    .load_tensor(&format!(
                        "model.language_model.layers.3.self_attn.{hf}.weight"
                    ))
                    .unwrap();
                assert_eq!(got.len(), want.len(), "{repo} {role}");
                let cos = cosine(&got, &want);
                println!("{repo} [{what}] layer 3 {hf}: cos {cos:.5}");
                assert!(cos >= min, "{repo} {hf}: cosine {cos} below {min}");
            }
        }
    }

    /// A real v3 bf16 artifact (Xiaomi-OCR-0, Qwen3.5-0.8B text + 12-block vision tower) loads
    /// through the engine's loader with its vision tower, and answers a text prompt.
    #[test]
    fn v3_xiaomi_ocr_loads_with_vision() {
        let path = snapshot("ByronLeeee/Xiaomi-OCR-0-Ninfer").join("xiaomi-ocr-0-bf16.ninfer");
        if !path.exists() {
            println!("SKIP: Xiaomi-OCR-0 NInfer artifact not in disk/");
            return;
        }
        let (ck, cfg) = Artifact::open(&path).unwrap().checkpoint().unwrap();
        assert!(
            ck.has("model.visual.patch_embed.proj.weight")
                && cfg["image_token_id"].as_u64().is_some()
        );
        assert_eq!(
            ck.shape("model.visual.patch_embed.proj.weight").unwrap(),
            [768, 3, 2, 16, 16]
        );
        let artifacts = ModelArtifacts {
            model_id: "x".into(),
            weights_path: Some(path),
            ..Default::default()
        };
        let m = crate::runtime::lm::load(
            &artifacts,
            crate::arch::registry().unwrap(),
            Some("qwen3_5"),
            crate::hardware::PlacementMode::Pin(crate::hardware::Device::Cpu),
        )
        .unwrap();
        println!("loaded {} with its vision tower", artifacts.model_id);
        drop(m);
    }
}
