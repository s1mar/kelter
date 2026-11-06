"""Reported input tokens, cost and latency of a JevBench run.

python -m kelter.eval.jevbench_cost results/jevbench/<run> --out results/jevbench/<name>_cost.json

Reads each tier's results.jsonl from the harness. Cost per 1,000 decisions is
mean usage.input_tokens times PRICE_PER_M / 1e6 times 1000, against CAP_PER_1K.
Pooled means weight every item equally.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PRICE_PER_M = 0.04   # USD per million input tokens, the board's reference for a 4B Qwen row
CAP_PER_1K = 0.0646  # USD per 1,000 decisions, the board's cost cap
TIERS = ("original", "hard", "easy")


def tier_stats(rows):
    tok = [r["usage"]["input_tokens"] for r in rows if r.get("usage")]
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    conf = [max(r["probs"].values()) for r in rows if r.get("probs")]
    return {"n": len(rows), "correct": sum(bool(r.get("correct")) for r in rows),
            "accuracy": sum(bool(r.get("correct")) for r in rows) / len(rows),
            "mean_top_prob": statistics.mean(conf) if conf else None,
            "input_tokens_mean": statistics.mean(tok), "input_tokens_median": statistics.median(tok),
            "input_tokens_max": max(tok), "latency_p50_s": statistics.median(lat), "latency_max_s": max(lat),
            "usd_per_1k": statistics.mean(tok) * PRICE_PER_M / 1e6 * 1000}


def summarize(run: Path):
    out, allrows = {"run": run.name, "price_per_m": PRICE_PER_M, "cap_per_1k": CAP_PER_1K, "tiers": {}}, []
    for t in TIERS:
        rows = [json.loads(l) for l in (run / t / "results.jsonl").open(encoding="utf-8")]
        out["tiers"][t] = tier_stats(rows)
        allrows += rows
    out["pooled"] = tier_stats(allrows)
    out["under_cap"] = out["pooled"]["usd_per_1k"] < CAP_PER_1K
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = summarize(Path(a.run))
    res["command"] = f"python -m kelter.eval.jevbench_cost {a.run} --out {a.out}"
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
