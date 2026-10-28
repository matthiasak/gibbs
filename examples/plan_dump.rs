//! CPU-only Metal plan dump of one bench-suite model: the model loads exactly as `gibbs --bench`
//! loads it (`bench::load_for_bench`, the suite's re-encode policy, this host's cached codec rates),
//! and its Metal compile prints `graph::metal::plan_dump` instead of creating a device
//! (`GIBBS_PLAN_DUMP_CPU`). No Metal device, weight upload or tuning.
//!
//! `cargo run --release --example plan_dump -- <suite.json> <row> [--ctx N] [--rates <host profile>]`
//! `<row>`: 0-based index into the suite's `models`, or a row's `name` / `id`. `--ctx`: keys in the
//! KV cache when the token decodes (attention dispatch count; default 1). `--rates`: use this host
//! profile's codec rates even when they were measured with other codec sources, so dumps of two
//! builds re-encode alike.
use gibbs_engine::bench::{default_reencode, load_for_bench, BenchSuite};
use gibbs_engine::graph::metal::HostProfile;
use gibbs_engine::graph::program::PLAN_DUMPED;
use gibbs_engine::model::hub::HuggingFaceHub;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [suite_path, row, rest @ ..] = &args[..] else {
        return Err(
            "usage: plan_dump <suite.json> <row> [--ctx N] [--rates <host profile>]".into(),
        );
    };
    let (mut ctx, mut rates) = (1usize, None);
    for pair in rest.chunks(2) {
        match pair {
            [flag, n] if flag == "--ctx" => {
                ctx = n.parse().map_err(|e| format!("--ctx {n}: {e}"))?
            }
            [flag, p] if flag == "--rates" => rates = Some(std::path::PathBuf::from(p)),
            _ => return Err(format!("unexpected arguments {rest:?}")),
        }
    }
    let text = std::fs::read_to_string(suite_path).map_err(|e| format!("{suite_path}: {e}"))?;
    let suite: BenchSuite =
        serde_json::from_str(&text).map_err(|e| format!("{suite_path}: {e}"))?;
    let m = match row.parse::<usize>() {
        Ok(i) => suite.models.get(i),
        Err(_) => suite
            .models
            .iter()
            .find(|m| m.name.as_deref() == Some(row) || m.id == *row),
    }
    .ok_or_else(|| format!("no suite row '{row}'"))?;
    let reencode = m
        .reencode
        .clone()
        .or_else(|| suite.defaults.as_ref().map(|d| d.reencode.clone()))
        .unwrap_or_else(default_reencode);

    // `Runtime::shared` publishes the cached codec rates the max-speed rule reads; here they come
    // from the newest host profile measured with the current codec sources, without a device.
    let codec_hash = gibbs_engine::graph::codec::source_hash();
    let host = gibbs_engine::hardware::default_cache_dir()?.join("host");
    let read = |p: &std::path::Path| -> Option<HostProfile> {
        serde_json::from_slice(&std::fs::read(p).ok()?).ok()
    };
    let profile = match &rates {
        Some(p) => {
            let hp =
                read(p).ok_or_else(|| format!("--rates {}: not a host profile", p.display()))?;
            let c = hp
                .codec
                .clone()
                .ok_or_else(|| format!("--rates {}: no codec rates", p.display()))?;
            Some((
                std::time::SystemTime::UNIX_EPOCH,
                p.clone(),
                hp.ceiling_gbps.max(hp.stream_gbps),
                c,
            ))
        }
        None => std::fs::read_dir(&host)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = read(&e.path())?;
                let c = p.codec.clone().filter(|c| c.codec_hash == codec_hash)?;
                let mtime = e.metadata().ok()?.modified().ok()?;
                Some((mtime, e.path(), p.ceiling_gbps.max(p.stream_gbps), c))
            })
            .max_by_key(|t| t.0),
    };
    match profile {
        Some((_, path, ceiling, codec)) => {
            println!("host-rates: {}", path.display());
            gibbs_engine::model::variant::set_host_rates(ceiling, Some(codec));
        }
        None => println!(
            "host-rates: none cached for codec hash {codec_hash:016x} (static re-encode rule)"
        ),
    }

    let artifacts = HuggingFaceHub::new().auto_download_and_cache(&m.id, None)?;
    println!(
        "model: {} ({}) reencode={reencode} ctx={ctx}",
        m.id,
        artifacts
            .weights_path
            .as_ref()
            .map_or("?".into(), |p| p.display().to_string())
    );
    std::env::set_var("GIBBS_PLAN_DUMP_CPU", ctx.to_string());
    let metal = gibbs_engine::hardware::PlacementMode::Pin(gibbs_engine::hardware::Device::Metal);
    let err = match load_for_bench(&artifacts, &reencode, metal) {
        Ok(_) => return Err("the model loaded without a Metal compile: nothing to dump".into()),
        Err(e) => e,
    };
    if !err.contains(PLAN_DUMPED) {
        return Err(err);
    }
    let s = gibbs_engine::model::reencode::native_stats();
    println!(
        "reencode: {} of {} projection tensors re-encoded under {reencode} (the bench keeps them only if its KL gate passes)",
        s.reencoded, s.tensors
    );
    Ok(())
}
