"""GPU canary before a long run: Q1's probabilities on 200 fixed
held-out items, the same batches every time, written to results/eval/canary/<tag>.json.

python -m kelter.eval.canary --tag a        (first process)
python -m kelter.eval.canary --tag b --ref a  (second process, compared with the first)

The comparison passes when every item has the same argmax and the largest probability
difference is at most TOL. The two runs use identical inputs and batches, so on a sound machine
they agree to within bf16 kernel noise. Accuracy on the 200 items is reported as a sanity check.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N, SEED, TOL = 200, 20261111, 0.005


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/Q1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--power-mode", choices=["quiet", "balance"], default=None,
                    help="laptop power mode as read back before the run; recorded in the results file")
    a = ap.parse_args(argv)
    from kelter.eval.run import gold_label, load
    from kelter.llm import LLMPredictor
    test = load("test")
    recs = random.Random(SEED).sample(test, N)
    pred = LLMPredictor(str(ROOT / a.ckpt), witness=False)
    out = pred.predict(recs)
    rows = []
    for r, o in zip(recs, out):
        probs = dict(o["probs"], __null__=o["null"])
        top = max(probs, key=probs.get)
        rows.append({"id": r["id"], "probs": probs, "argmax": top, "correct": top == gold_label(r)})
    res = {"ckpt": a.ckpt, "n": N, "seed": SEED, "prefill_chunk": pred.prefill_chunk, "power_mode": a.power_mode or "unrecorded",
           "accuracy": sum(x["correct"] for x in rows) / N, "items": rows}
    if a.ref:
        ref = json.loads((ROOT / "results" / "eval" / "canary" / f"{a.ref}.json").read_text(encoding="utf-8"))
        assert [x["id"] for x in ref["items"]] == [x["id"] for x in rows]
        diff = max(abs(x["probs"][k] - y["probs"].get(k, 0.0)) for x, y in zip(rows, ref["items"]) for k in x["probs"])
        same = sum(x["argmax"] == y["argmax"] for x, y in zip(rows, ref["items"]))
        res["compare"] = {"ref": a.ref, "argmax_agree": same, "max_abs_prob_diff": diff, "tol": TOL,
                          "pass": same == N and diff <= TOL}
    res["command"] = f"python -m kelter.eval.canary --tag {a.tag}" + (f" --ref {a.ref}" if a.ref else "") + (f" --power-mode {a.power_mode}" if a.power_mode else "")
    d = ROOT / "results" / "eval" / "canary"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=2))
    return 0 if res.get("compare", {}).get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
