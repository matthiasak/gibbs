//! Bag-of-n-grams text classifier (fastText-style): a line becomes hashed input-row ids (words,
//! word n-grams, char n-grams), and the spec component `classify` (vars `rows`) averages those
//! rows through a linear output layer to per-row `scores` (softmax or sigmoid). Hashing, product-
//! quantized row decoding and the hierarchical-softmax tree walk stay host glue here; file
//! formats are read elsewhere into a [`BagModel`].
//!
//! Spec contract: inputs `ids` [rows] u32 (dense input table, weight `input` [n, dim]) or
//! `vectors` [rows, dim] f32 (quantized tables, decoded here); weight `output` [out, dim]; `cfg`
//! scope `{dim, quantized, softmax}`; output `scores` [1, out].

use crate::arch::build::{build, Instance, ShapeCache};
use crate::arch::expr::{walk, Env, Val};
use crate::arch::{Signals, Source, Spec};
use crate::graph::program::GraphProgram;
use crate::graph::{DType, Tensor, Weights};
use crate::hardware::{Device, PlacementMode};
use parking_lot::Mutex;
use serde_json::{json, Value as J};
use std::collections::HashMap;

const COMPONENT: &str = "classify";
/// The end-of-line token every line ends with.
const EOS: &[u8] = b"</s>";
/// Bytes tokens split on (`Dictionary::readWord`).
const SPACES: &[u8] = b" \n\r\t\x0b\x0c\0";

/// FNV-1a with the int8 sign extension of `Dictionary::hash`.
fn hash(s: &[u8]) -> u32 {
    s.iter().fold(2166136261u32, |h, &c| {
        (h ^ (c as i8 as i32 as u32)).wrapping_mul(16777619)
    })
}

/// Token-to-row hashing (`Dictionary`): known words, word n-grams and char n-grams.
pub struct Vocab {
    pub nwords: usize,
    pub bucket: u32,
    pub minn: usize,
    pub maxn: usize,
    pub word_ngrams: usize,
    /// Label token prefix (`__label__`): such tokens are skipped.
    pub label: Vec<u8>,
    pub word2int: HashMap<Vec<u8>, i32>,
    pub words: Vec<Vec<u8>>,
    pub is_label: Vec<bool>,
    /// Bucket remap of a pruned dictionary (`None` = not pruned).
    pub prune: Option<HashMap<i32, i32>>,
}

impl Vocab {
    /// `Dictionary::pushHash`.
    fn push_hash(&self, line: &mut Vec<u32>, id: i32) {
        let id = match &self.prune {
            None => id,
            Some(p) => match p.get(&id) {
                Some(&j) => j,
                None => return,
            },
        };
        if id >= 0 {
            line.push((self.nwords + id as usize) as u32);
        }
    }

    /// `Dictionary::computeSubwords` over `<word>` (UTF-8 aware char n-grams).
    fn subwords(&self, token: &[u8], line: &mut Vec<u32>) {
        let w = [b"<", token, b">"].concat();
        for i in 0..w.len() {
            if w[i] & 0xC0 == 0x80 {
                continue;
            }
            let (mut j, mut n) = (i, 1);
            while j < w.len() && n <= self.maxn {
                j += 1;
                while j < w.len() && w[j] & 0xC0 == 0x80 {
                    j += 1;
                }
                if n >= self.minn && !(n == 1 && (i == 0 || j == w.len())) {
                    self.push_hash(line, (hash(&w[i..j]) % self.bucket) as i32);
                }
                n += 1;
            }
        }
    }

    /// Input-row ids of `text` and the number of tokens read (`getLine` tokens + `</s>`).
    pub fn line(&self, text: &str) -> Result<(Vec<u32>, usize), String> {
        if text.contains('\n') {
            return Err("fastText predicts one line at a time (remove '\\n')".into());
        }
        let tokens = text
            .as_bytes()
            .split(|c| SPACES.contains(c))
            .filter(|t| !t.is_empty())
            .chain([EOS]);
        let (mut line, mut hashes, mut ntokens) = (Vec::new(), Vec::new(), 0);
        for t in tokens {
            ntokens += 1;
            let wid = self.word2int.get(t).copied();
            if wid.map_or(t.starts_with(&self.label), |w| self.is_label[w as usize]) {
                continue;
            }
            match wid {
                Some(w) => {
                    line.push(w as u32);
                    if self.maxn > 0 && t != EOS {
                        self.subwords(&self.words[w as usize], &mut line);
                    }
                }
                None if t != EOS => self.subwords(t, &mut line),
                None => {}
            }
            hashes.push(hash(t) as i32);
        }
        // addWordNgrams: int32 hashes widen (sign-extended) to uint64.
        for i in 0..hashes.len() {
            let mut h = hashes[i] as i64 as u64;
            for &hj in hashes.iter().take(i + self.word_ngrams).skip(i + 1) {
                h = h.wrapping_mul(116049371).wrapping_add(hj as i64 as u64);
                self.push_hash(&mut line, (h % self.bucket as u64) as i32);
            }
        }
        if line.is_empty() {
            return Err("fastText: the input has no known word or n-gram".into());
        }
        Ok((line, ntokens))
    }
}

/// Product quantizer: 256 centroids per sub-quantizer; the last one spans `lastdsub` dims.
pub struct Pq {
    pub nsubq: usize,
    pub dsub: usize,
    pub lastdsub: usize,
    pub centroids: Vec<f32>,
}

impl Pq {
    fn centroid(&self, m: usize, i: u8) -> &[f32] {
        let i = i as usize;
        if m == self.nsubq - 1 {
            &self.centroids[m * 256 * self.dsub + i * self.lastdsub..][..self.lastdsub]
        } else {
            &self.centroids[(m * 256 + i) * self.dsub..][..self.dsub]
        }
    }
}

/// Product-quantized rows: `nsubq` codes per row, optionally one norm code per row.
pub struct QuantRows {
    pub codes: Vec<u8>,
    pub pq: Pq,
    pub norm: Option<(Vec<u8>, Pq)>,
}

impl QuantRows {
    pub fn rows(&self) -> usize {
        self.codes.len() / self.pq.nsubq
    }

    /// Row `i` decoded into `out` (the row width).
    pub fn decode(&self, i: usize, out: &mut [f32]) {
        let scale = self
            .norm
            .as_ref()
            .map_or(1.0, |(c, npq)| npq.centroid(0, c[i])[0]);
        let code = &self.codes[i * self.pq.nsubq..][..self.pq.nsubq];
        for (m, &c) in code.iter().enumerate() {
            let o = &mut out[m * self.pq.dsub..];
            for (o, v) in o.iter_mut().zip(self.pq.centroid(m, c)) {
                *o = scale * v;
            }
        }
    }
}

/// An input-row table.
pub enum Rows {
    /// Row-major `[rows, dim]`.
    Dense(Vec<f32>),
    Quant(QuantRows),
}

/// How output-row scores become label probabilities.
pub enum Head {
    /// One output row per label, softmax over them.
    Softmax,
    /// One output row per label, independent sigmoids (one-vs-all, negative sampling).
    Sigmoid,
    /// Hierarchical softmax: per label, root-first (output row, code) pairs; code 1 takes
    /// `sigmoid(row · h)`, code 0 its complement.
    Tree(Vec<Vec<(usize, bool)>>),
}

/// A parsed bag classifier.
pub struct BagModel {
    pub vocab: Vocab,
    pub dim: usize,
    pub input: Rows,
    /// Row-major `[out, dim]` output layer.
    pub output: Vec<f32>,
    pub head: Head,
    /// Label names in output order.
    pub labels: Vec<String>,
}

/// The `cfg` scope of the spec.
struct Cfg(J);

impl Env for Cfg {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        (root == "cfg").then(|| walk(Val::from_json(&self.0), path))?
    }
}

/// F32 weight shapes by name.
struct Shapes(Vec<(&'static str, Vec<usize>)>);

impl Source for Shapes {
    fn tensor(&self, name: &str) -> Option<(Vec<usize>, DType)> {
        self.0
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| (s.clone(), DType::F32))
    }
}

/// A [`BagModel`] compiled from a `bag_classifier` family spec.
pub struct BagClassifier {
    vocab: Vocab,
    dim: usize,
    /// Quantized input table (rows decoded per line); `None` = the graph gathers `input` rows.
    quant: Option<QuantRows>,
    head: Head,
    labels: Vec<String>,
    spec: Spec,
    cfg: Cfg,
    src: Shapes,
    shapes: ShapeCache,
    program: Mutex<GraphProgram>,
    dev: Device,
}

fn vars(rows: usize) -> [(&'static str, Val); 1] {
    [("rows", Val::Num(rows as f64))]
}

impl BagClassifier {
    /// Compiles family `family`'s `classify` component over `m`'s tables under `mode`.
    pub fn new(family: &str, m: BagModel, mode: impl Into<PlacementMode>) -> Result<Self, String> {
        let mode = mode.into();
        let dev = crate::graph::avenues::resolved_backend(mode);
        let spec = crate::arch::registry()?.spec(family, &Signals::default())?;
        let BagModel {
            vocab,
            dim,
            input,
            output,
            head,
            labels,
        } = m;
        let mut src = vec![("output", vec![output.len() / dim, dim])];
        let mut w = Weights::new();
        w.insert("output", Tensor::F32(output));
        let quant = match input {
            Rows::Dense(v) => {
                src.push(("input", vec![v.len() / dim, dim]));
                w.insert("input", Tensor::F32(v));
                None
            }
            Rows::Quant(q) => Some(q),
        };
        let cfg = Cfg(json!({
            "dim": dim,
            "quantized": quant.is_some(),
            "softmax": matches!(head, Head::Softmax),
        }));
        let src = Shapes(src);
        let shapes = ShapeCache::new(COMPONENT);
        let v = vars(1);
        let first = shapes.get(&v, || build(&spec, COMPONENT, &cfg, &src, &v))?;
        let program = GraphProgram::compile_reshapable(first.graph.clone(), w, mode)?;
        if dev == Device::Metal
            && (program.primary_device() != dev || program.cpu_node_count() > 0)
        {
            return Err(format!(
                "{family}: {} graph nodes cannot run on Metal",
                program.cpu_node_count()
            ));
        }
        Ok(Self {
            vocab,
            dim,
            quant,
            head,
            labels,
            spec,
            cfg,
            src,
            shapes,
            program: Mutex::new(program),
            dev,
        })
    }

    /// Label names in output order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn device(&self) -> Device {
        self.dev
    }

    fn instance(&self, rows: usize) -> Result<std::sync::Arc<Instance>, String> {
        let v = vars(rows);
        self.shapes
            .get(&v, || build(&self.spec, COMPONENT, &self.cfg, &self.src, &v))
    }

    /// Probability of every label (in [`Self::labels`] order) and the number of tokens read, as
    /// `fasttext.predict(text, k=-1)` reports them: `p + 1e-5` (its `exp(log(p + 1e-5))`); labels
    /// below its implicit threshold (hierarchical-softmax paths under 1e-5) are 0.
    pub fn predict(&self, text: &str) -> Result<(Vec<f32>, usize), String> {
        let (ids, ntokens) = self.vocab.line(text)?;
        let inst = self.instance(ids.len())?;
        let input = match &self.quant {
            None => Tensor::U32(ids),
            Some(q) => {
                let mut v = vec![0f32; ids.len() * self.dim];
                for (row, &i) in v.chunks_exact_mut(self.dim).zip(&ids) {
                    q.decode(i as usize, row);
                }
                Tensor::F32(v)
            }
        };
        let scores = self
            .program
            .lock().run_reshaped(&inst.graph, vec![input], true)?
            .into_iter()
            .nth(inst.output_index("scores").ok_or("no `scores` output")?)
            .ok_or("missing `scores` output")?
            .into_f32()?;
        let probs = match &self.head {
            // HierarchicalSoftmaxLoss::dfs from the root with threshold 0: a branch whose log
            // score falls under log(1e-5) is pruned (label absent → 0).
            Head::Tree(paths) => {
                let floor = 1e-5f32.ln();
                paths
                    .iter()
                    .map(|path| {
                        let mut score = 0f32;
                        for &(row, code) in path {
                            if score < floor {
                                return 0.0;
                            }
                            let f = scores[row];
                            score += (if code { f } else { 1.0 - f } + 1e-5).ln();
                        }
                        if score < floor {
                            0.0
                        } else {
                            score.exp()
                        }
                    })
                    .collect()
            }
            Head::Softmax | Head::Sigmoid => {
                scores.into_iter().map(|p| (p + 1e-5).ln().exp()).collect()
            }
        };
        Ok((probs, ntokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_sign_extension() {
        // ASCII is plain FNV-1a; bytes >= 0x80 xor in a sign-extended int8.
        assert_eq!(hash(b"a"), 0xe40c292c);
        assert_eq!(
            hash(&[0xc3]),
            (2166136261u32 ^ 0xffff_ffc3).wrapping_mul(16777619)
        );
    }
}
