#!/usr/bin/env bash
# Short paired A/B, one model per process, each well under a minute.
#   scripts/quick_abba.sh ab LABEL "BASE_CMD" "CAND_CMD" ROW...
#       Per bench/loop.json row, one scheduled job runs base, cand, cand, base `--bench` legs at 64 decode
#       tokens (warm-up + 2 timed runs). CMDs are prefixes: "target/ab-base/gibbs",
#       "env GIBBS_DISABLE=pipeline target/release/gibbs". Output: target/quick/LABEL/summary.txt.
#   scripts/quick_abba.sh parity ROW...
#       `target/release/gibbs --bench-parity` on each row alone (marked reference).
set -uo pipefail
cd "$(dirname "$0")/.."
mode=$1; shift
suite() { # file row out reference(0|1)
python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys
src, row, out, ref = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
s = json.load(open(src))
m = dict(s["models"][row]); m.pop("min_tok_s", None)
if "prompt_file" in m:
    m["prompt_file"] = os.path.abspath(os.path.join("bench", m["prompt_file"]))
d = dict(s.get("defaults") or {})
if ref: m["reference"] = True
else:
    if src == "bench/loop.json":
        d.update(decode_tokens=64, runs=2)
    else:
        d.setdefault("runs", 2)
json.dump({"defaults": d, "models": [m]}, open(out, "w"))
# ABBA_PLACEMENT: candidate legs load under this placement; the base binary predates the field.
if os.environ.get("ABBA_PLACEMENT"):
    d["placement"] = os.environ["ABBA_PLACEMENT"]
json.dump({"defaults": d, "models": [m]}, open(out.replace(".json", "-cand.json"), "w"))
PY
}
mkdir -p target/quick
for r in 0 1 2 3 4 5 6 7 8; do suite "bench/loop.json" "$r" "target/quick/suite-$r.json" 0; done
if [ -f bench/nonlm.json ]; then
  for r in 0 1 2; do suite "bench/nonlm.json" "$r" "target/quick/nonlm-$r.json" 0; done
fi
if [ "$mode" = parity ]; then
  for row in "$@"; do
    suite "bench/loop.json" "$row" "target/quick/parity-$row.json" 1
    python3 ~/.cache/gibbs/central_scheduler.py run --label "qp-$row" -- \
      target/release/gibbs --bench-parity "target/quick/parity-$row.json"
    echo "parity row $row rc=$?"
  done
  exit 0
fi
suite_prefix="suite"
if [ "$mode" = "ab-nonlm" ]; then
  suite_prefix="nonlm"
fi
label=$1 base=$2 cand=$3; shift 3
d=target/quick/$label
rm -rf "$d" && mkdir -p "$d"
for row in "$@"; do
  for leg in base1 cand1 cand2 base2; do
    case $leg in base*) c=$base s=${suite_prefix}-$row.json ;; *) c=$cand s=${suite_prefix}-$row-cand.json ;; esac
    echo "$c --bench target/quick/$s > $d/r$row-$leg.out 2> $d/r$row-$leg.err"
  done > "$d/row-$row.sh"
  python3 ~/.cache/gibbs/central_scheduler.py run --label "qa-$label-$row" -- bash "$d/row-$row.sh"
done
python3 - "$mode" "$d" "$@" <<'PY' | tee "$d/summary.txt"
import json, math, sys
mode, d, rows = sys.argv[1], sys.argv[2], sys.argv[3:]
ratios, spreads, pp_ratios = [], [], []
for row in rows:
    leg = {}
    for t in ("base1", "cand1", "cand2", "base2"):
        js = [json.loads(l) for l in open(f"{d}/r{row}-{t}.out") if l.startswith("{")]
        leg[t] = js[0] if js else {"error": "no result row"}
    bad = [t for t in leg if "error" in leg[t]]
    if bad:
        print(f"row {row}: ERROR {[(t, leg[t]['error']) for t in bad]}"); ratios.append(0.0); spreads.append(1.0); continue
    b1 = leg["base1"]
    if "s_per_step" in b1:
        val = {t: leg[t]["s_per_step"] for t in leg}
        b, c = (val["base1"] + val["base2"]) / 2, (val["cand1"] + val["cand2"]) / 2
        ratio = b / max(c, 1e-9)
        sp = abs(val["base1"] - val["base2"]) / max(b, 1e-9)
        ratios.append(ratio); spreads.append(sp)
        print(f"row {row} {b1['model'][-46:]:46s} base {b:7.3f}s/step cand {c:7.3f}s/step x{ratio:.3f} spread {sp:.3f} rss {b1.get('rss_gb')}->{leg['cand1'].get('rss_gb')}")
    elif "items_per_s" in b1:
        val = {t: leg[t]["items_per_s"] for t in leg}
        b, c = (val["base1"] + val["base2"]) / 2, (val["cand1"] + val["cand2"]) / 2
        ratio = c / max(b, 1e-9)
        sp = abs(val["base1"] - val["base2"]) / max(b, 1e-9)
        ratios.append(ratio); spreads.append(sp)
        print(f"row {row} {b1['model'][-46:]:46s} base {b:7.1f}it/s cand {c:7.1f}it/s x{ratio:.3f} spread {sp:.3f} rss {b1.get('rss_gb')}->{leg['cand1'].get('rss_gb')}")
    else:
        tok = {t: leg[t]["decode_tok_s"] for t in leg}
        b, c = (tok["base1"] + tok["base2"]) / 2, (tok["cand1"] + tok["cand2"]) / 2
        sp = abs(tok["base1"] - tok["base2"]) / b
        ratios.append(c / b); spreads.append(sp)
        g = lambda t, k: leg[t].get(k)
        pp_str = ""
        if all(g(t, "prefill_tok_s") is not None for t in leg):
            pb = (g("base1", "prefill_tok_s") + g("base2", "prefill_tok_s")) / 2
            pc = (g("cand1", "prefill_tok_s") + g("cand2", "prefill_tok_s")) / 2
            pp_ratios.append(pc / pb)
            pp_str = f" pp {pb:.0f}->{pc:.0f} x{pc/pb:.3f}"
        print(f"row {row} {b1['model'][-46:]:46s} base {b:7.1f} cand {c:7.1f} x{c/b:.3f} spread {sp:.3f}{pp_str} "
              f"gpu {g('base1','gpu_ms')}->{g('cand1','gpu_ms')} host {g('base1','host_ms')}->{g('cand1','host_ms')} "
              f"disp {g('base1','dispatches')}->{g('cand1','dispatches')} rss {g('base1','rss_gb')}->{g('cand1','rss_gb')}")
gm = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios))
pp_summary = ""
if pp_ratios:
    pp_gm = math.exp(sum(math.log(max(r, 1e-9)) for r in pp_ratios) / len(pp_ratios))
    pp_summary = f" pp_geomean x{pp_gm:.3f} pp_min x{min(pp_ratios):.3f}"
print(f"SUMMARY geomean x{gm:.3f} min x{min(ratios):.3f}{pp_summary} max_spread {max(spreads):.3f} rows {len(rows)}")
PY
