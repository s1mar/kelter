"""Baselines table, rendered only from results files.

python -m kelter.eval.table

Reads results/eval/<tag>.json for every system in results/baselines.json
(a list of {"tag", "label"} plus {"failed": reason}
rows for systems that could not be run) and writes results/baselines_table.md
and results/baselines_table.json. Quoted third-party numbers (Jev) appear in a
separate, labelled section and never in the measured table.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COLS = [
    ("heldout_accuracy", "Held-out acc"),
    ("ece_15", "ECE"),
    ("brier", "Brier"),
    ("flip_rate_canonical", "Flip (canon)"),
    ("flip_rate_no_canonical", "Flip (no canon)"),
    ("nudge_asr", "Nudge-ASR"),
    ("forge_asr", "Forge-ASR"),
    ("null_recall_at_5fpr", "Null recall @5% FPR (old set)"),
    ("null_recall_at_5fpr_corrected", "Null recall @5% FPR (corrected set)"),
    ("witness_f1_esnli", "Witness F1 e-SNLI"),
    ("witness_necessity_esnli", "Necessity e-SNLI"),
    ("random_mask_flip_esnli", "Random-mask flip"),
    ("anli_accuracy", "ANLI acc"),
]

QUOTED = [
    ("Jev 1.13 (hosted, not run here)", "ECE 0.045, paired flip rate 0.021, injection ASR 0.063",
     "Respan run as cited in docs/own-decision-model-design.md (read 2026-10-25); different items from ours"),
    ("Jev 1.13 (hosted, not run here)", "JevBench v1.4 composite 63.3",
     "JevBench v1.4 results as cited in docs/own-decision-model-design.md (read 2026-10-25)"),
]


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        if "asr" in v:
            return "n/a" if v["asr"] != v["asr"] else f"{v['asr']:.3f} (n={v['n']})"
        if "recall" in v:
            return f"{v['recall']:.3f}"
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.3f}"
    return str(v)


def main() -> int:
    spec = json.loads((ROOT / "results" / "baselines.json").read_text(encoding="utf-8"))
    rows, md = [], []
    md.append("# Baselines (measured by kelter/eval/run.py on the same held-out sets)")
    md.append("")
    md.append("| System | " + " | ".join(c[1] for c in COLS) + " | Source file |")
    md.append("|---|" + "---|" * (len(COLS) + 1))
    for s in spec["systems"]:
        if "failed" in s:
            md.append(f"| {s['label']} | " + " | ".join("not run" for _ in COLS) + f" | {s['failed']} |")
            rows.append({"label": s["label"], "failed": s["failed"]})
            continue
        path = ROOT / "results" / "eval" / f"{s['tag']}.json"
        d = json.loads(path.read_text(encoding="utf-8"))
        md.append(f"| {s['label']} | " + " | ".join(fmt(d.get(k)) for k, _ in COLS) + f" | results/eval/{s['tag']}.json |")
        rows.append({"label": s["label"], "tag": s["tag"], **{k: d.get(k) for k, _ in COLS}, "command": d.get("command")})
    md.append("")
    md.append("Systems without a null output use 1 - max probability as the null score.")
    md.append("")
    md.append("## Quoted, not measured here")
    md.append("")
    md.append("| System | Numbers | Source and date |")
    md.append("|---|---|---|")
    for sysname, nums, src in QUOTED:
        md.append(f"| {sysname} | {nums} | {src} |")
    (ROOT / "results" / "baselines_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "results" / "baselines_table.json").write_text(json.dumps({"rows": rows, "quoted": QUOTED}, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("evaluation (python -m kelter.eval.table)")
    raise SystemExit(main())
