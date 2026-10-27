//! Real-weight checks for the Idefics3/SmolVLM vision-language path: image in, greedy answer out.
//! Model dir: `GIBBS_SMOLVLM_DIR` or the shared cache's SmolVLM-256M-Instruct snapshot.

use gibbs_engine::hardware::Device;
use gibbs_engine::model::hf_loader::ChatMessage;
use gibbs_engine::model::hub::ModelArtifacts;
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::native::{MediaInput, NativeLm};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn model() -> Option<Arc<dyn NativeLm>> {
    let dir = PathBuf::from(std::env::var("GIBBS_SMOLVLM_DIR").unwrap_or_else(|_| {
        "disk/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/main".into()
    }));
    if !dir.join("model.safetensors").exists() {
        println!("SKIP: no SmolVLM weights at {}", dir.display());
        return None;
    }
    let artifacts = ModelArtifacts {
        model_id: "HuggingFaceTB/SmolVLM".into(),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(dir.join("model.safetensors")),
        local_dir: Some(dir),
        ..Default::default()
    };
    Some(
        gibbs_engine::runtime::lm::load(
            &artifacts,
            gibbs_engine::arch::registry().expect("registry"),
            None,
            gibbs_engine::hardware::PlacementMode::Pin(Device::Cpu),
        )
        .expect("loads"),
    )
}

/// Greedy answer to `question` about `images` (one `{"type":"image"}` part each, then the text).
fn ask(m: &dyn NativeLm, images: &[Vec<u8>], question: &str, max_tokens: usize) -> String {
    let tok = m.tokenizer();
    let mut parts: Vec<_> = images
        .iter()
        .map(|_| serde_json::json!({"type": "image"}))
        .collect();
    parts.push(serde_json::json!({"type": "text", "text": question}));
    let mut msg = ChatMessage::new("user", "");
    msg.content = serde_json::Value::Array(parts);
    let ids = tok.encode(&tok.apply_chat_template_messages(&[msg], None));
    let media: Vec<_> = images.iter().cloned().map(MediaInput::Image).collect();
    let t = Instant::now();
    let (ids, mut state) = m.prepare(ids, &media).expect("prepare");
    let prep = t.elapsed().as_secs_f64();
    let mut logits = m.forward(&ids, 0, &mut state).expect("prefill");
    println!(
        "  prepare (decode + vision) {prep:.1}s, prefill {} tokens {:.1}s",
        ids.len(),
        t.elapsed().as_secs_f64() - prep
    );
    let mut out = Vec::new();
    for pos in ids.len()..ids.len() + max_tokens {
        let next = gibbs_engine::runtime::lm::sample_logits(&logits, &out, 0.0, 1, 1.0);
        if tok.is_special(next) {
            break;
        }
        out.push(next);
        logits = m.forward(&[next], pos, &mut state).expect("decode");
    }
    tok.decode(&out)
}

/// A drawn image (red disc on white) is seen: the answer names the color. Catches broken
/// preprocessing, vision tower, connector, or image-row placement (the text model alone can't
/// know the color).
#[test]
fn smolvlm_sees_a_drawn_image() {
    let Some(m) = model() else { return };
    let mut img = ImageBuffer::new(320, 240);
    for y in 0..240 {
        for x in 0..320 {
            let (dx, dy) = (x as f32 - 160.0, y as f32 - 120.0);
            let c = if dx * dx + dy * dy < 80.0 * 80.0 {
                (220, 20, 30)
            } else {
                (255, 255, 255)
            };
            img.set_pixel(x, y, c.0, c.1, c.2);
        }
    }
    let t = Instant::now();
    let answer = ask(
        &*m,
        &[img.to_png()],
        "What color is the circle? Answer in one word.",
        8,
    );
    println!("drawn image ({:.1}s): {answer}", t.elapsed().as_secs_f64());
    assert!(answer.to_lowercase().contains("red"), "answer: {answer}");
    // Placeholders must match the media count; audio is refused.
    let tok = m.tokenizer();
    let ids = tok.encode(&tok.apply_chat_template(&[("user", "hi")]));
    assert!(m
        .prepare(ids.clone(), &[MediaInput::Image(img.to_png())])
        .is_err());
    let err = m.prepare(ids, &[MediaInput::Audio(vec![0; 16])]).err();
    assert!(err.is_some_and(|e| e.contains("audio")));
    let text = ask(&*m, &[], "What is the capital of France?", 12);
    println!("text only: {text}");
    assert!(text.contains("Paris"), "answer: {text}");
}

/// Photos in `GIBBS_VLM_IMAGES` (a directory of png/jpg files named after their subject, e.g.
/// `cat.jpg`): prints each description; files named cat/dog/eiffel/banana must mention it.
#[test]
fn smolvlm_describes_photos() {
    let Ok(dir) = std::env::var("GIBBS_VLM_IMAGES") else {
        println!("SKIP: GIBBS_VLM_IMAGES not set");
        return;
    };
    let Some(m) = model() else { return };
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("GIBBS_VLM_IMAGES dir")
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("png" | "jpg" | "jpeg")
            )
        })
        .collect();
    files.sort();
    for path in files {
        let bytes = std::fs::read(&path).unwrap();
        let t = Instant::now();
        let answer = ask(&*m, &[bytes], "Can you describe this image?", 60);
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        println!("{name} ({:.1}s): {answer}", t.elapsed().as_secs_f64());
        let subject = [
            ("cat", "cat"),
            ("dog", "dog"),
            ("eiffel", "tower"),
            ("banana", "banana"),
        ]
        .iter()
        .find(|(stem, _)| name.starts_with(stem));
        if let Some((_, word)) = subject {
            assert!(answer.to_lowercase().contains(word), "{name}: {answer}");
        }
    }
}

#[test]
fn test_vlm_metal_lowering_zero_cpu_segments() {
    use gibbs_engine::graph::program::GraphProgram;
    use gibbs_engine::graph::{DType, Graph, Op, Weights};
    use gibbs_engine::hardware::{available_devices, Device, PlacementMode};

    if !available_devices().contains(&Device::Metal) {
        println!("SKIP: Metal not available");
        return;
    }

    let mut g = Graph::new();
    let mut w = Weights::new();

    let pixels = g.input("pixels", vec![3, 14, 14], DType::F32);
    let patch_w = g.weight("patch.weight", vec![16, 3, 14, 14], DType::F32);
    let patch_b = g.weight("patch.bias", vec![16], DType::F32);
    w.insert(
        "patch.weight",
        gibbs_engine::graph::Tensor::F32(vec![0.1; 16 * 3 * 14 * 14]),
    );
    w.insert(
        "patch.bias",
        gibbs_engine::graph::Tensor::F32(vec![0.0; 16]),
    );

    let conv = g.add(
        "conv",
        Op::Conv2d {
            stride: [14, 14],
            padding: [0, 0, 0, 0],
            dilation: [1, 1],
            groups: 1,
        },
        &[pixels, patch_w, patch_b],
        vec![16, 1, 1],
        DType::F32,
    );

    let gelu = g.add(
        "act",
        Op::Unary {
            kind: gibbs_engine::graph::UnaryKind::GeluTanh,
        },
        &[conv],
        vec![16, 1, 1],
        DType::F32,
    );
    g.output(gelu);

    let prog = GraphProgram::compile(g, w, PlacementMode::Pin(Device::Metal)).expect("compile");
    assert_eq!(prog.primary_device(), Device::Metal);
    assert!(prog.is_single_metal(), "expected one Metal segment");
}
