"""Dev (in-distribution validation) accuracy per source, clean records only.

python -m kelter.eval.dev_by_source --ckpt checkpoints/r1 [--per-source 300]
Writes results/eval/dev_<run>.json. Used for training-round decisions; held-out
numbers come from kelter.eval.run.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from kelter.eval.run import gold_label, load, verdict
from kelter.model import Predictor

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-source", type=int, default=300)
    a = ap.parse_args(argv)
    dev = load("dev")
    by = collections.defaultdict(list)
    for r in dev:
        by[r["source"]].append(r)
    rng = random.Random(0)
    pick = []
    for src, rs in sorted(by.items()):
        rng.shuffle(rs)
        pick.extend(rs[: a.per_source])
    pred = Predictor(a.ckpt)
    outs = pred.predict(pick)
    acc = collections.defaultdict(list)
    for r, o in zip(pick, outs):
        acc[(r["source"], r["question"]["type"])].append(verdict(o).replace("__null__", "__null__") == gold_label(r))
    res = {f"{s}/{t}": {"acc": sum(v) / len(v), "n": len(v)} for (s, t), v in sorted(acc.items())}
    out = ROOT / "results" / "eval" / f"dev_{Path(a.ckpt).name}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
