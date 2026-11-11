//! SGLang-style compressed finite-state / pushdown JSON grammar constraint + deterministic
//! `jump_forward` for `response_format` (`json_object` and `json_schema`).
//!
//! Enforces 100% RFC 8259 + JSON Schema validity at token-sampling time (`mask_logits`) while
//! skipping deterministic structural runs (`{"key":`, `,"next":`, `"rue"`, `"null"`, enum
//! suffixes, closing `}` / `]`) without running separate autoregressive decode steps.

use crate::model::tokenizer::Tokenizer;
use serde_json::Value;
use std::sync::Arc;

/// Compiled JSON schema node.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaNode {
    Object {
        /// Ordered `(property_name, schema, required)` fields.
        props: Vec<(String, SchemaNode, bool)>,
        /// Allow arbitrary extra string keys beyond `props`.
        additional: bool,
    },
    Array {
        items: Box<SchemaNode>,
        min_items: usize,
        max_items: Option<usize>,
    },
    String {
        enum_vals: Option<Vec<String>>,
    },
    Integer,
    Number,
    Boolean,
    Null,
    AnyJson,
    AnyOf(Vec<SchemaNode>),
}

impl SchemaNode {
    pub fn from_schema(schema: &Value, root: &Value) -> Result<Self, String> {
        if let Some(r) = schema.get("$ref").and_then(Value::as_str) {
            let target =
                resolve_ref(r, root).ok_or_else(|| format!("unresolved JSON schema $ref '{r}'"))?;
            return Self::from_schema(target, root);
        }
        if let Some(any) = schema
            .get("anyOf")
            .or_else(|| schema.get("oneOf"))
            .and_then(Value::as_array)
        {
            let mut branches = Vec::with_capacity(any.len());
            for b in any {
                branches.push(Self::from_schema(b, root)?);
            }
            if branches.is_empty() {
                return Err("anyOf/oneOf must contain at least one branch".into());
            }
            return Ok(Self::AnyOf(branches));
        }
        if let Some(ev) = schema.get("enum").and_then(Value::as_array) {
            let mut lits = Vec::with_capacity(ev.len());
            for v in ev {
                match v {
                    Value::String(s) => lits.push(Self::String {
                        enum_vals: Some(vec![s.clone()]),
                    }),
                    Value::Bool(true) => lits.push(Self::Boolean),
                    Value::Bool(false) => lits.push(Self::Boolean),
                    Value::Null => lits.push(Self::Null),
                    Value::Number(n) if n.is_i64() || n.is_u64() => lits.push(Self::Integer),
                    Value::Number(_) => lits.push(Self::Number),
                    _ => lits.push(Self::AnyJson),
                }
            }
            if ev.iter().all(Value::is_string) {
                let strs: Vec<String> = ev
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                if strs.is_empty() {
                    return Err("enum array cannot be empty".into());
                }
                return Ok(Self::String {
                    enum_vals: Some(strs),
                });
            }
            return Ok(Self::AnyOf(lits));
        }
        if let Some(arr) = schema.get("type").and_then(Value::as_array) {
            let mut branches = Vec::with_capacity(arr.len());
            for t in arr {
                if let Some(ts) = t.as_str() {
                    let mut sub = schema.clone();
                    sub["type"] = Value::String(ts.to_string());
                    branches.push(Self::from_schema(&sub, root)?);
                }
            }
            if !branches.is_empty() {
                return Ok(Self::AnyOf(branches));
            }
        }
        match schema.get("type").and_then(Value::as_str) {
            Some("object") => {
                let req: Vec<&str> = schema
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let mut props = Vec::new();
                if let Some(pobj) = schema.get("properties").and_then(Value::as_object) {
                    for (k, v) in pobj {
                        let is_req = req.contains(&k.as_str());
                        props.push((k.clone(), Self::from_schema(v, root)?, is_req));
                    }
                }
                let additional = match schema.get("additionalProperties") {
                    Some(Value::Bool(b)) => *b,
                    Some(Value::Object(_)) => true,
                    None => props.is_empty(),
                    _ => false,
                };
                Ok(Self::Object { props, additional })
            }
            Some("array") => {
                let items = match schema.get("items") {
                    Some(it) => Self::from_schema(it, root)?,
                    None => Self::AnyJson,
                };
                let min_items =
                    schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
                let max_items = schema
                    .get("maxItems")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize);
                Ok(Self::Array {
                    items: Box::new(items),
                    min_items,
                    max_items,
                })
            }
            Some("string") => Ok(Self::String { enum_vals: None }),
            Some("integer") => Ok(Self::Integer),
            Some("number") => Ok(Self::Number),
            Some("boolean") => Ok(Self::Boolean),
            Some("null") => Ok(Self::Null),
            None if schema.get("properties").is_some() => {
                let mut sub = schema.clone();
                sub["type"] = Value::String("object".into());
                Self::from_schema(&sub, root)
            }
            None => Ok(Self::AnyJson),
            Some(other) => Err(format!("unsupported JSON schema type '{other}'")),
        }
    }
}

fn resolve_ref<'a>(r: &str, root: &'a Value) -> Option<&'a Value> {
    let path = r.strip_prefix("#/")?;
    let mut cur = root;
    for seg in path.split('/') {
        cur = cur.get(seg)?;
    }
    Some(cur)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StrStage {
    Body {
        escaped: bool,
        hex_rem: u8,
        utf8_rem: u8,
        /// JSON-escaped raw content bytes emitted inside quotes so far (for enum matching).
        raw: Vec<u8>,
        /// Escaped JSON byte representation of allowed enum strings, if constrained.
        enums: Option<Vec<Vec<u8>>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NumStage {
    Start { allow_float: bool },
    AfterMinus { allow_float: bool },
    Zero { allow_float: bool },
    IntDigits { allow_float: bool },
    Dot,
    FracDigits,
    ExpSign,
    ExpDigitsFirst,
    ExpDigits,
}

#[derive(Debug, Clone, PartialEq)]
enum Frame {
    Literal {
        bytes: &'static [u8],
        idx: usize,
    },
    ExactBytes {
        bytes: Vec<u8>,
        idx: usize,
    },
    Str(StrStage),
    Num(NumStage),
    BoolStart,
    AnyValue,
    /// Object continuation after `{` has been consumed.
    ObjNextProp {
        props: Vec<(String, SchemaNode, bool)>,
        idx: usize,
        emitted: usize,
        additional: bool,
    },
    /// Free-form object after `{`: expects `}` (if `emitted > 0`? only when `first`!) or `"key": val`.
    FreeObj {
        first: bool,
    },
    FreeObjAfterComma,
    FreeObjColon,
    FreeObjAfterVal,
    /// Array after `[` has been consumed.
    ArrNext {
        items: SchemaNode,
        count: usize,
        min_items: usize,
        max_items: Option<usize>,
    },
    ArrAfterElem {
        items: SchemaNode,
        count: usize,
        min_items: usize,
        max_items: Option<usize>,
    },
}

/// One deterministic execution thread of the pushdown automaton.
#[derive(Debug, Clone, PartialEq)]
struct Thread {
    stack: Vec<Frame>,
}

fn json_escape_bytes(s: &str) -> Vec<u8> {
    let encoded = serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""));
    encoded.as_bytes()[1..encoded.len() - 1].to_vec()
}

fn push_node(stack: &mut Vec<Frame>, node: &SchemaNode) -> Vec<Thread> {
    match node {
        SchemaNode::Null => {
            stack.push(Frame::Literal {
                bytes: b"null",
                idx: 0,
            });
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::Boolean => {
            stack.push(Frame::BoolStart);
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::Integer => {
            stack.push(Frame::Num(NumStage::Start { allow_float: false }));
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::Number => {
            stack.push(Frame::Num(NumStage::Start { allow_float: true }));
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::String { enum_vals } => {
            let enums = enum_vals
                .as_ref()
                .map(|vs| vs.iter().map(|s| json_escape_bytes(s)).collect());
            stack.push(Frame::Str(StrStage::Body {
                escaped: false,
                hex_rem: 0,
                utf8_rem: 0,
                raw: Vec::new(),
                enums,
            }));
            stack.push(Frame::Literal {
                bytes: b"\"",
                idx: 0,
            });
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::Array {
            items,
            min_items,
            max_items,
        } => {
            stack.push(Frame::ArrNext {
                items: (**items).clone(),
                count: 0,
                min_items: *min_items,
                max_items: *max_items,
            });
            stack.push(Frame::Literal {
                bytes: b"[",
                idx: 0,
            });
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::Object { props, additional } => {
            if props.is_empty() && *additional {
                stack.push(Frame::FreeObj { first: true });
            } else {
                stack.push(Frame::ObjNextProp {
                    props: props.clone(),
                    idx: 0,
                    emitted: 0,
                    additional: *additional,
                });
            }
            stack.push(Frame::Literal {
                bytes: b"{",
                idx: 0,
            });
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::AnyJson => {
            stack.push(Frame::AnyValue);
            vec![Thread {
                stack: stack.clone(),
            }]
        }
        SchemaNode::AnyOf(branches) => {
            let mut out = Vec::new();
            for b in branches {
                let mut st = stack.clone();
                out.extend(push_node(&mut st, b));
            }
            out
        }
    }
}

impl Thread {
    fn is_complete(&self) -> bool {
        let mut st = self.stack.clone();
        while let Some(top) = st.last() {
            match top {
                Frame::Num(
                    NumStage::Zero { .. }
                    | NumStage::IntDigits { .. }
                    | NumStage::FracDigits
                    | NumStage::ExpDigits,
                ) => {
                    st.pop();
                }
                _ => return false,
            }
        }
        true
    }

    fn step_byte(&self, b: u8) -> Vec<Thread> {
        let mut stack = self.stack.clone();
        let Some(top) = stack.pop() else {
            // Allow trailing whitespace? No: strict JSON ends cleanly at root closer so EOS fires at once.
            return Vec::new();
        };
        match top {
            Frame::Literal { bytes, idx } => {
                if bytes[idx] != b {
                    return Vec::new();
                }
                if idx + 1 < bytes.len() {
                    stack.push(Frame::Literal {
                        bytes,
                        idx: idx + 1,
                    });
                }
                vec![Thread { stack }]
            }
            Frame::ExactBytes { bytes, idx } => {
                if bytes[idx] != b {
                    return Vec::new();
                }
                if idx + 1 < bytes.len() {
                    stack.push(Frame::ExactBytes {
                        bytes,
                        idx: idx + 1,
                    });
                }
                vec![Thread { stack }]
            }
            Frame::BoolStart => match b {
                b't' => {
                    stack.push(Frame::Literal {
                        bytes: b"rue",
                        idx: 0,
                    });
                    vec![Thread { stack }]
                }
                b'f' => {
                    stack.push(Frame::Literal {
                        bytes: b"alse",
                        idx: 0,
                    });
                    vec![Thread { stack }]
                }
                _ => Vec::new(),
            },
            Frame::Str(StrStage::Body {
                escaped,
                hex_rem,
                utf8_rem,
                mut raw,
                enums,
            }) => {
                if let Some(ref allowed) = enums {
                    if !escaped && hex_rem == 0 && utf8_rem == 0 && b == b'"' {
                        if allowed.iter().any(|a| a.as_slice() == raw.as_slice()) {
                            return vec![Thread { stack }];
                        }
                        return Vec::new();
                    }
                    raw.push(b);
                    if !allowed.iter().any(|a| a.starts_with(&raw)) {
                        return Vec::new();
                    }
                    stack.push(Frame::Str(StrStage::Body {
                        escaped: false,
                        hex_rem: 0,
                        utf8_rem: 0,
                        raw,
                        enums,
                    }));
                    return vec![Thread { stack }];
                }
                if utf8_rem > 0 {
                    if (b & 0xC0) != 0x80 {
                        return Vec::new();
                    }
                    stack.push(Frame::Str(StrStage::Body {
                        escaped: false,
                        hex_rem: 0,
                        utf8_rem: utf8_rem - 1,
                        raw,
                        enums,
                    }));
                    return vec![Thread { stack }];
                }
                if hex_rem > 0 {
                    if !b.is_ascii_hexdigit() {
                        return Vec::new();
                    }
                    stack.push(Frame::Str(StrStage::Body {
                        escaped: false,
                        hex_rem: hex_rem - 1,
                        utf8_rem: 0,
                        raw,
                        enums,
                    }));
                    return vec![Thread { stack }];
                }
                if escaped {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            stack.push(Frame::Str(StrStage::Body {
                                escaped: false,
                                hex_rem: 0,
                                utf8_rem: 0,
                                raw,
                                enums,
                            }));
                            vec![Thread { stack }]
                        }
                        b'u' => {
                            stack.push(Frame::Str(StrStage::Body {
                                escaped: false,
                                hex_rem: 4,
                                utf8_rem: 0,
                                raw,
                                enums,
                            }));
                            vec![Thread { stack }]
                        }
                        _ => Vec::new(),
                    }
                } else {
                    match b {
                        b'"' => vec![Thread { stack }],
                        b'\\' => {
                            stack.push(Frame::Str(StrStage::Body {
                                escaped: true,
                                hex_rem: 0,
                                utf8_rem: 0,
                                raw,
                                enums,
                            }));
                            vec![Thread { stack }]
                        }
                        0x00..=0x1F => Vec::new(),
                        0x20..=0x7F => {
                            stack.push(Frame::Str(StrStage::Body {
                                escaped: false,
                                hex_rem: 0,
                                utf8_rem: 0,
                                raw,
                                enums,
                            }));
                            vec![Thread { stack }]
                        }
                        0xC2..=0xDF => {
                            stack.push(Frame::Str(StrStage::Body {
                                escaped: false,
                                hex_rem: 0,
                                utf8_rem: 1,
                                raw,
                                enums,
                            }));
                            vec![Thread { stack }]
                        }
                        0xE0..=0xEF => {
                            stack.push(Frame::Str(StrStage::Body {
                                escaped: false,
                                hex_rem: 0,
                                utf8_rem: 2,
                                raw,
                                enums,
                            }));
                            vec![Thread { stack }]
                        }
                        0xF0..=0xF4 => {
                            stack.push(Frame::Str(StrStage::Body {
                                escaped: false,
                                hex_rem: 0,
                                utf8_rem: 3,
                                raw,
                                enums,
                            }));
                            vec![Thread { stack }]
                        }
                        _ => Vec::new(),
                    }
                }
            }
            Frame::Num(ns) => match ns {
                NumStage::Start { allow_float } => match b {
                    b'-' => {
                        stack.push(Frame::Num(NumStage::AfterMinus { allow_float }));
                        vec![Thread { stack }]
                    }
                    b'0' => {
                        stack.push(Frame::Num(NumStage::Zero { allow_float }));
                        vec![Thread { stack }]
                    }
                    b'1'..=b'9' => {
                        stack.push(Frame::Num(NumStage::IntDigits { allow_float }));
                        vec![Thread { stack }]
                    }
                    _ => Vec::new(),
                },
                NumStage::AfterMinus { allow_float } => match b {
                    b'0' => {
                        stack.push(Frame::Num(NumStage::Zero { allow_float }));
                        vec![Thread { stack }]
                    }
                    b'1'..=b'9' => {
                        stack.push(Frame::Num(NumStage::IntDigits { allow_float }));
                        vec![Thread { stack }]
                    }
                    _ => Vec::new(),
                },
                NumStage::Zero { allow_float } => match b {
                    b'.' if allow_float => {
                        stack.push(Frame::Num(NumStage::Dot));
                        vec![Thread { stack }]
                    }
                    b'e' | b'E' if allow_float => {
                        stack.push(Frame::Num(NumStage::ExpSign));
                        vec![Thread { stack }]
                    }
                    _ => Thread { stack }.step_byte(b),
                },
                NumStage::IntDigits { allow_float } => match b {
                    b'0'..=b'9' => {
                        stack.push(Frame::Num(NumStage::IntDigits { allow_float }));
                        vec![Thread { stack }]
                    }
                    b'.' if allow_float => {
                        stack.push(Frame::Num(NumStage::Dot));
                        vec![Thread { stack }]
                    }
                    b'e' | b'E' if allow_float => {
                        stack.push(Frame::Num(NumStage::ExpSign));
                        vec![Thread { stack }]
                    }
                    _ => Thread { stack }.step_byte(b),
                },
                NumStage::Dot => match b {
                    b'0'..=b'9' => {
                        stack.push(Frame::Num(NumStage::FracDigits));
                        vec![Thread { stack }]
                    }
                    _ => Vec::new(),
                },
                NumStage::FracDigits => match b {
                    b'0'..=b'9' => {
                        stack.push(Frame::Num(NumStage::FracDigits));
                        vec![Thread { stack }]
                    }
                    b'e' | b'E' => {
                        stack.push(Frame::Num(NumStage::ExpSign));
                        vec![Thread { stack }]
                    }
                    _ => Thread { stack }.step_byte(b),
                },
                NumStage::ExpSign => match b {
                    b'+' | b'-' => {
                        stack.push(Frame::Num(NumStage::ExpDigitsFirst));
                        vec![Thread { stack }]
                    }
                    b'0'..=b'9' => {
                        stack.push(Frame::Num(NumStage::ExpDigits));
                        vec![Thread { stack }]
                    }
                    _ => Vec::new(),
                },
                NumStage::ExpDigitsFirst => match b {
                    b'0'..=b'9' => {
                        stack.push(Frame::Num(NumStage::ExpDigits));
                        vec![Thread { stack }]
                    }
                    _ => Vec::new(),
                },
                NumStage::ExpDigits => match b {
                    b'0'..=b'9' => {
                        stack.push(Frame::Num(NumStage::ExpDigits));
                        vec![Thread { stack }]
                    }
                    _ => Thread { stack }.step_byte(b),
                },
            },
            Frame::AnyValue => match b {
                b'{' => {
                    stack.push(Frame::FreeObj { first: true });
                    vec![Thread { stack }]
                }
                b'[' => {
                    stack.push(Frame::ArrNext {
                        items: SchemaNode::AnyJson,
                        count: 0,
                        min_items: 0,
                        max_items: None,
                    });
                    vec![Thread { stack }]
                }
                b'"' => {
                    stack.push(Frame::Str(StrStage::Body {
                        escaped: false,
                        hex_rem: 0,
                        utf8_rem: 0,
                        raw: Vec::new(),
                        enums: None,
                    }));
                    vec![Thread { stack }]
                }
                b't' => {
                    stack.push(Frame::Literal {
                        bytes: b"rue",
                        idx: 0,
                    });
                    vec![Thread { stack }]
                }
                b'f' => {
                    stack.push(Frame::Literal {
                        bytes: b"alse",
                        idx: 0,
                    });
                    vec![Thread { stack }]
                }
                b'n' => {
                    stack.push(Frame::Literal {
                        bytes: b"ull",
                        idx: 0,
                    });
                    vec![Thread { stack }]
                }
                b'-' | b'0'..=b'9' => {
                    stack.push(Frame::Num(NumStage::Start { allow_float: true }));
                    Thread { stack }.step_byte(b)
                }
                _ => Vec::new(),
            },
            Frame::FreeObj { first } => match b {
                b'}' if first => vec![Thread { stack }],
                b'"' => {
                    stack.push(Frame::FreeObjAfterVal);
                    stack.push(Frame::AnyValue);
                    stack.push(Frame::FreeObjColon);
                    stack.push(Frame::Str(StrStage::Body {
                        escaped: false,
                        hex_rem: 0,
                        utf8_rem: 0,
                        raw: Vec::new(),
                        enums: None,
                    }));
                    vec![Thread { stack }]
                }
                _ => Vec::new(),
            },
            Frame::FreeObjColon => {
                if b == b':' {
                    vec![Thread { stack }]
                } else {
                    Vec::new()
                }
            }
            Frame::FreeObjAfterVal => match b {
                b'}' => vec![Thread { stack }],
                b',' => {
                    stack.push(Frame::FreeObjAfterComma);
                    vec![Thread { stack }]
                }
                _ => Vec::new(),
            },
            Frame::FreeObjAfterComma => {
                if b == b'"' {
                    stack.push(Frame::FreeObjAfterVal);
                    stack.push(Frame::AnyValue);
                    stack.push(Frame::FreeObjColon);
                    stack.push(Frame::Str(StrStage::Body {
                        escaped: false,
                        hex_rem: 0,
                        utf8_rem: 0,
                        raw: Vec::new(),
                        enums: None,
                    }));
                    vec![Thread { stack }]
                } else {
                    Vec::new()
                }
            }
            Frame::ObjNextProp {
                props,
                idx,
                emitted,
                additional,
            } => {
                let mut out = Vec::new();
                for k in idx..props.len() {
                    let (ref pname, ref pschema, preq) = props[k];
                    let mut st = stack.clone();
                    st.push(Frame::ObjNextProp {
                        props: props.clone(),
                        idx: k + 1,
                        emitted: emitted + 1,
                        additional,
                    });
                    let mut val_threads = push_node(&mut st, pschema);
                    let mut prefix = Vec::with_capacity(pname.len() + 4);
                    if emitted > 0 {
                        prefix.push(b',');
                    }
                    prefix.push(b'"');
                    prefix.extend_from_slice(&json_escape_bytes(pname));
                    prefix.extend_from_slice(b"\":");
                    for vt in &mut val_threads {
                        vt.stack.push(Frame::ExactBytes {
                            bytes: prefix.clone(),
                            idx: 0,
                        });
                    }
                    for vt in val_threads {
                        out.extend(vt.step_byte(b));
                    }
                    if preq {
                        return out;
                    }
                }
                // All remaining `props[idx..]` are optional: may close `}` (or emit extra free props if `additional`).
                if additional {
                    let mut st = stack.clone();
                    if emitted == 0 {
                        st.push(Frame::FreeObj { first: true });
                    } else {
                        st.push(Frame::FreeObjAfterVal);
                    }
                    out.extend(Thread { stack: st }.step_byte(b));
                } else if b == b'}' {
                    out.push(Thread { stack });
                }
                out
            }
            Frame::ArrNext {
                items,
                count,
                min_items,
                max_items,
            } => {
                let mut out = Vec::new();
                if count >= min_items && b == b']' {
                    out.push(Thread {
                        stack: stack.clone(),
                    });
                }
                if max_items.is_none_or(|mx| count < mx) {
                    let mut st = stack;
                    st.push(Frame::ArrAfterElem {
                        items: items.clone(),
                        count: count + 1,
                        min_items,
                        max_items,
                    });
                    for vt in push_node(&mut st, &items) {
                        out.extend(vt.step_byte(b));
                    }
                }
                out
            }
            Frame::ArrAfterElem {
                items,
                count,
                min_items,
                max_items,
            } => {
                let mut out = Vec::new();
                if count >= min_items && b == b']' {
                    out.push(Thread {
                        stack: stack.clone(),
                    });
                }
                if max_items.is_none_or(|mx| count < mx) && b == b',' {
                    let mut st = stack;
                    st.push(Frame::ArrAfterElem {
                        items: items.clone(),
                        count: count + 1,
                        min_items,
                        max_items,
                    });
                    out.extend(push_node(&mut st, &items));
                }
                out
            }
        }
    }
}

/// Active per-sequence grammar automaton with lazy decoded-vocabulary cache.
#[derive(Clone)]
pub struct GrammarConstraint {
    threads: Vec<Thread>,
    vocab_bytes: Option<Arc<Vec<(Vec<u8>, bool)>>>,
}

impl std::fmt::Debug for GrammarConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarConstraint")
            .field("threads", &self.threads.len())
            .finish()
    }
}

impl GrammarConstraint {
    pub fn from_schema_node(root: &SchemaNode) -> Self {
        let mut st = Vec::new();
        let threads = push_node(&mut st, root);
        Self {
            threads,
            vocab_bytes: None,
        }
    }

    pub fn json_object() -> Self {
        Self::from_schema_node(&SchemaNode::Object {
            props: Vec::new(),
            additional: true,
        })
    }

    pub fn from_json_schema(schema: &Value) -> Result<Self, String> {
        let node = SchemaNode::from_schema(schema, schema)?;
        Ok(Self::from_schema_node(&node))
    }

    /// Parses OpenAI `response_format` (`{"type": "text" | "json_object" | "json_schema", ...}`)
    /// or Responses API `text.format` (`{"type": "json_schema", "schema": ...}`).
    pub fn from_response_format(fmt: Option<&Value>) -> Result<Option<Self>, String> {
        let Some(f) = fmt else {
            return Ok(None);
        };
        if f.is_null() {
            return Ok(None);
        }
        match f.get("type").and_then(Value::as_str) {
            None | Some("text") => Ok(None),
            Some("json_object") => Ok(Some(Self::json_object())),
            Some("json_schema") => {
                let schema = f
                    .get("json_schema")
                    .and_then(|js| js.get("schema"))
                    .or_else(|| f.get("schema"))
                    .ok_or("response_format json_schema requires a 'schema' object")?;
                Self::from_json_schema(schema).map(Some)
            }
            Some(other) => Err(format!(
                "unsupported response_format type '{other}' (expected text, json_object, or json_schema)"
            )),
        }
    }

    /// The NDJSON API's `format`: `"json"`, or a JSON schema object; `""` / `null` are unconstrained.
    pub fn from_format_field(fmt: Option<&Value>) -> Result<Option<Self>, String> {
        match fmt {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) if s.is_empty() => Ok(None),
            Some(Value::String(s)) if s == "json" => Ok(Some(Self::json_object())),
            Some(Value::String(s)) => Err(format!(
                "unknown format '{s}' (expected \"json\" or a JSON schema)"
            )),
            Some(schema) => Self::from_json_schema(schema).map(Some),
        }
    }

    /// True when at least one active thread has closed its root JSON value cleanly.
    pub fn is_complete(&self) -> bool {
        self.threads.iter().any(Thread::is_complete)
    }

    fn step_threads(threads: &[Thread], b: u8) -> Vec<Thread> {
        let mut next: Vec<Thread> = Vec::new();
        for t in threads {
            for nt in t.step_byte(b) {
                if !next.contains(&nt) && next.len() < 16 {
                    next.push(nt);
                }
            }
        }
        next
    }

    pub fn can_accept_bytes(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }
        let mut cur = self.threads.clone();
        for &b in bytes {
            cur = Self::step_threads(&cur, b);
            if cur.is_empty() {
                return false;
            }
        }
        true
    }

    pub fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        let mut cur = self.threads.clone();
        for &b in bytes {
            cur = Self::step_threads(&cur, b);
            if cur.is_empty() {
                return false;
            }
        }
        self.threads = cur;
        true
    }

    pub fn advance_token(&mut self, tok_id: u32, tok: &Tokenizer) -> bool {
        if tok.ends_generation(tok_id) || tok.is_special(tok_id) {
            return self.is_complete();
        }
        let mut bytes = Vec::new();
        if !tok.append_token_bytes(tok_id, &mut bytes) || bytes.is_empty() {
            bytes = tok.decode_token(tok_id).into_bytes();
        }
        self.advance_bytes(&bytes)
    }

    /// If the current grammar state admits **exactly one** valid next byte (and cannot stop here),
    /// returns that unique byte and the successor threads.
    fn unique_next(&self) -> Option<(u8, Vec<Thread>)> {
        if self.is_complete() {
            return None;
        }
        // Fast check when a single thread sits on `ExactBytes` or `Literal`: O(1) without scanning 128 ASCII bytes!
        if self.threads.len() == 1 {
            match self.threads[0].stack.last() {
                Some(Frame::Literal { bytes, idx }) => {
                    let b = bytes[*idx];
                    return Some((b, Self::step_threads(&self.threads, b)));
                }
                Some(Frame::ExactBytes { bytes, idx }) => {
                    let b = bytes[*idx];
                    return Some((b, Self::step_threads(&self.threads, b)));
                }
                _ => {}
            }
        }
        let mut found: Option<(u8, Vec<Thread>)> = None;
        for b in 0x20u8..=0x7Eu8 {
            let next = Self::step_threads(&self.threads, b);
            if !next.is_empty() {
                if found.is_some() {
                    return None;
                }
                found = Some((b, next));
            }
        }
        found
    }

    /// SGLang-style compressed finite-state jump-forward: advances across maximal deterministic
    /// byte runs (`{"key":`, `,"k2":`, `"rue"`, `"null"`, enum completions, closing `}` / `]`)
    /// and returns their token IDs when they round-trip cleanly through `tok`.
    pub fn jump_forward(&mut self, tok: &Tokenizer) -> Vec<u32> {
        let mut probe = self.clone();
        let mut forced = Vec::new();
        while forced.len() < 64 {
            let Some((b, next)) = probe.unique_next() else {
                break;
            };
            // Stop right before an unconstrained string's opening `"` if a single BPE token often
            // merges `"value`? No: only stop AFTER `:` so `{"key":` jumps cleanly and leaves `"Paris"`
            // intact for natural BPE `"Paris` tokenization!
            let was_colon = b == b':';
            forced.push(b);
            probe.threads = next;
            if was_colon {
                // Peek if the value itself is a fixed literal (`null` or single-value enum) — if not,
                // stop right after `:` so BPE tokens like `"hello` stay unsplit!
                let next_is_open_str = probe.threads.len() == 1
                    && matches!(
                        probe.threads[0].stack.as_slice(),
                        [
                            ..,
                            Frame::Str(StrStage::Body { enums: None, .. }),
                            Frame::Literal {
                                bytes: b"\"",
                                idx: 0
                            }
                        ]
                    );
                if next_is_open_str {
                    break;
                }
            }
        }
        if forced.is_empty() {
            return Vec::new();
        }
        let Ok(s) = std::str::from_utf8(&forced) else {
            return Vec::new();
        };
        // Find the longest prefix of `s` whose `encode_fragment` decodes back to that exact prefix.
        for end in (1..=s.len()).rev() {
            if !s.is_char_boundary(end) {
                continue;
            }
            let sub = &s[..end];
            let ids = tok.encode_fragment(sub);
            if !ids.is_empty()
                && ids
                    .iter()
                    .all(|&id| !tok.ends_generation(id) && !tok.is_special(id))
                && tok.decode(&ids) == sub
            {
                let ok = self.advance_bytes(sub.as_bytes());
                debug_assert!(ok);
                return ids;
            }
        }
        Vec::new()
    }

    fn ensure_vocab(&mut self, tok: &Tokenizer) -> Arc<Vec<(Vec<u8>, bool)>> {
        let n = tok.vocab_size();
        if let Some(ref v) = self.vocab_bytes {
            if v.len() == n {
                return v.clone();
            }
        }
        let mut table = Vec::with_capacity(n);
        for id in 0..n as u32 {
            let is_eos = tok.ends_generation(id);
            if is_eos || tok.is_special(id) {
                table.push((Vec::new(), is_eos));
            } else {
                let mut b = Vec::new();
                if !tok.append_token_bytes(id, &mut b) || b.is_empty() {
                    b = tok.decode_token(id).into_bytes();
                }
                table.push((b, false));
            }
        }
        let arc = Arc::new(table);
        self.vocab_bytes = Some(arc.clone());
        arc
    }

    /// Masks `logits` in place so ONLY tokens valid at the current grammar state (or EOS once
    /// `is_complete()` is true) retain finite logits.
    pub fn mask_logits(&mut self, logits: &mut [f32], tok: &Tokenizer) {
        let vocab = self.ensure_vocab(tok);
        let done = self.is_complete();
        let limit = logits.len().min(vocab.len());
        let mut any_allowed = false;
        for (i, slot) in logits[..limit].iter_mut().enumerate() {
            if !slot.is_finite() {
                continue;
            }
            let (ref bytes, is_eos) = vocab[i];
            let allow = if is_eos {
                done
            } else if bytes.is_empty() {
                false
            } else {
                self.can_accept_bytes(bytes)
            };
            if allow {
                any_allowed = true;
            } else {
                *slot = f32::NEG_INFINITY;
            }
        }
        for slot in &mut logits[limit..] {
            *slot = f32::NEG_INFINITY;
        }
        // Safety fallback if a tiny synthetic vocab lacks an explicit EOS token once `done`:
        if !any_allowed && done && !logits.is_empty() {
            if let Some((eos_idx, _)) = vocab.iter().enumerate().find(|(_, (_, eos))| *eos) {
                if eos_idx < logits.len() {
                    logits[eos_idx] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_jump_forward_and_masked_sampling_produce_100pct_valid_json() {
        let tok = Tokenizer::default_bpe();
        let fmt = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "role": { "type": "string", "enum": ["engineer", "researcher"] },
                        "active": { "type": "boolean" },
                        "years": { "type": "integer" }
                    },
                    "required": ["name", "role", "active", "years"],
                    "additionalProperties": false
                }
            }
        });
        let mut g = GrammarConstraint::from_response_format(Some(&fmt))
            .unwrap()
            .unwrap();

        // 1. Opening jump_forward must deterministically emit `{"name":`!
        let jumped0 = g.jump_forward(&tok);
        let s0 = tok.decode(&jumped0);
        assert_eq!(s0, "{\"name\":", "jump_forward opening prefix");

        // 2. Drive masked greedy/pseudo-random sampling to completion and parse JSON!
        let mut out_tokens = jumped0;
        let mut step = 0u64;
        // A pseudo-random "model" that grows keener to close open strings / numbers / objects.
        let closes: Vec<bool> = (0..tok.vocab_size() as u32)
            .map(|i| {
                tok.decode(&[i])
                    .bytes()
                    .any(|b| b == b'"' || b == b'}' || b == b',')
            })
            .collect();
        while !g.is_complete() && step < 256 {
            step += 1;
            let mut logits: Vec<f32> = (0..tok.vocab_size())
                .map(|i| {
                    let base = ((i as u64 * 17 + step * 31) % 97) as f32 * 0.1;
                    base + if closes[i] { step as f32 } else { 0.0 }
                })
                .collect();
            g.mask_logits(&mut logits, &tok);
            let best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            assert!(
                best.1.is_finite(),
                "at least one valid token at step {step}"
            );
            let chosen = best.0 as u32;
            if tok.ends_generation(chosen) {
                break;
            }
            assert!(g.advance_token(chosen, &tok));
            out_tokens.push(chosen);
            let jf = g.jump_forward(&tok);
            out_tokens.extend(jf);
        }
        assert!(g.is_complete(), "grammar reached complete state");
        // Jump-forward emitted the keys and punctuation: fewer sampling steps than tokens.
        assert!(
            (step as usize) < out_tokens.len(),
            "{step} steps for {} tokens",
            out_tokens.len()
        );
        let text = tok.decode(&out_tokens);
        let parsed: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("invalid JSON emitted: '{text}' ({e})"));
        assert!(parsed["name"].is_string());
        assert!(matches!(
            parsed["role"].as_str(),
            Some("engineer" | "researcher")
        ));
        assert!(parsed["active"].is_boolean());
        assert!(parsed["years"].is_i64());
    }

    #[test]
    fn json_object_mode_accepts_nested_json_and_rejects_prose() {
        let tok = Tokenizer::default_bpe();
        let fmt = json!({"type": "json_object"});
        let mut g = GrammarConstraint::from_response_format(Some(&fmt))
            .unwrap()
            .unwrap();
        assert!(!g.can_accept_bytes(b"Sure, here is JSON:"));
        assert!(g.advance_bytes(b"{\"items\":[1,true,null,\"ok\"]}"));
        assert!(g.is_complete());
        let mut logits = vec![1.0f32; tok.vocab_size()];
        g.mask_logits(&mut logits, &tok);
        for (i, &l) in logits.iter().enumerate() {
            if l.is_finite() {
                assert!(tok.ends_generation(i as u32));
            }
        }
    }
}
