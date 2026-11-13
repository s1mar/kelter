"""Null recall diagnostics per source and per kind of unknowable item.

python -m kelter.eval.null_by_source --ckpt checkpoints/r2
Uses the global 5% FPR threshold from all answerable test items, then reports
the share of unknowable items above it per source and kind (answer removed
vs natural null). Writes results/eval/null_<run>.json. Diagnostic only.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from kelter.eval import metrics as M
from kelter.eval.run import load, null_score
from kelter.model import Predictor

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args(argv)
    pred = Predictor(a.ckpt)
    test = load("test")
    ans = [r for r in test if not r["null"]]
    nat = [r for r in test if r["null"]]
    ids = {r["id"] for r in ans}
    rm = [r for r in load("test_null") if r["id"].rsplit("~", 1)[0] in ids]
    sa = [null_score(o) for o in pred.predict(ans)]
    t = M.null_recall_at_fpr(sa, [0.0], 0.05)["threshold"]
    res = {"threshold": t, "groups": {}}
    by = collections.defaultdict(list)
    for kind, recs in (("answer_removed", rm), ("natural_null", nat)):
        for r, o in zip(recs, pred.predict(recs)):
            by[f"{r['source']}/{kind}"].append(null_score(o) > t)
    fpr = collections.defaultdict(list)
    for r, s in zip(ans, sa):
        fpr[r["source"]].append(s > t)
    for k, v in sorted(by.items()):
        res["groups"][k] = {"recall": sum(v) / len(v), "n": len(v)}
    res["false_null_by_source"] = {k: {"rate": sum(v) / len(v), "n": len(v)} for k, v in sorted(fpr.items())}
    out = ROOT / "results" / "eval" / f"null_{Path(a.ckpt).name}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("evaluation (python -m kelter.eval.null_by_source)")
    raise SystemExit(main())
