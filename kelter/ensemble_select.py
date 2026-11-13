"""Member selection for a cheaper stacked ensemble.

python -m kelter.ensemble_select --from E2 --name E3

Calibration split only. Each member's null-score quantile column is computed
once on calib.jsonl's answerable records and the calib-side unknowables (as in
kelter.ensemble.fit_stacked) and cached in checkpoints/<name>/columns.npz.
A subset is scored by recall at 5% FPR over ten stratified half splits of
those rows (stacker trained on one half, scored on the other), with E2's
stacker parameters. Backward elimination drops, one at a time, the member
whose removal leaves the highest score (ties: the costlier member first); the
option checkpoint's own null column costs nothing and is never dropped. The
selected subset is the cheapest one on that path scoring at least the full
set's score minus TOL. Cost is the summed model.pt size of the members other
than the option checkpoint. Writes results/ensemble/<name>_select.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from kelter.ensemble import OUT, ROOT, null_score

TOL = 0.005
SPLITS = 10


def columns(name: str, null: List[str]) -> Dict[str, np.ndarray]:
    cache = ROOT / "checkpoints" / name / "columns.npz"
    if cache.exists():
        # allow_pickle=False: the cache is plain arrays and nothing is unpickled here. (kelter.ensemble is the only
        # place a pickle is loaded, after its sha256 is checked.)
        z = np.load(cache, allow_pickle=False)
        if list(z["members"]) == null:
            return dict(z)
    from sentence_transformers import SentenceTransformer

    from kelter.device import default_device
    from kelter.eval import unknowable as U
    from kelter.model import Predictor
    calib = [json.loads(l) for l in (ROOT / "data" / "built" / "calib.jsonl").open(encoding="utf-8")]
    m = SentenceTransformer(U.MODEL, device=default_device())
    unk, _ = U.build(calib, lambda t: m.encode(t, normalize_embeddings=True, convert_to_numpy=True, batch_size=128))
    del m
    ans = [r for r in calib if not r["null"]]
    rows = ans + unk
    cols = []
    for c in null:
        p = Predictor(str(ROOT / "checkpoints" / c))
        s = np.array([null_score(o) for o in p.predict(rows)])
        ref = np.sort(s[: len(ans)])  # this member's null scores on the answerable records only (the first len(ans) rows)
        # The quantile is an empirical CDF against that reference: the fraction of answerable calibration scores at
        # or below s (side="right" counts ties), the same quantile kelter.ensemble uses at inference.
        cols.append(np.searchsorted(ref, s, side="right") / len(ref))
        del p
    z = {"X": np.stack(cols, 1), "y": np.r_[np.zeros(len(ans)), np.ones(len(unk))], "members": np.array(null)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **z)
    return z


def recall_at_5fpr(score: np.ndarray, y: np.ndarray) -> float:
    thr = np.quantile(score[y == 0], 0.95)  # the score that 5% of the answerable rows (y == 0) exceed: 5% false positives
    return float((score[y == 1] > thr).mean())


def split_score(X: np.ndarray, y: np.ndarray, params: Dict) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    out = []
    for seed in range(SPLITS):
        a, b = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
        clf = HistGradientBoostingClassifier(**params).fit(X[a], y[a])
        out.append(recall_at_5fpr(clf.predict_proba(X[b])[:, 1], y[b]))
    return float(np.mean(out))


def select(src: str, name: str) -> Dict:
    spec = json.loads((OUT / f"{src}.json").read_text(encoding="utf-8"))
    opt, null, params = spec["options"], list(spec["null"]), spec["stacker_params"]
    z = columns(name, null)
    X, y = z["X"], z["y"]
    cost = {c: 0 if c == opt else (ROOT / "checkpoints" / c / "model.pt").stat().st_size for c in null}
    keep = list(null)
    path = [{"members": list(keep), "score": split_score(X, y, params), "cost": sum(cost.values())}]
    while len([c for c in keep if c != opt]) > 0:
        trials = []
        for c in keep:
            if c == opt:
                continue
            rest = [k for k in keep if k != c]
            trials.append((split_score(X[:, [null.index(k) for k in rest]], y, params), cost[c], c))
        best = max(trials)
        keep.remove(best[2])
        path.append({"members": list(keep), "dropped": best[2], "score": best[0],
                     "cost": sum(cost[k] for k in keep)})
        print(json.dumps(path[-1]), flush=True)
    floor = path[0]["score"] - TOL
    chosen = min((p for p in path if p["score"] >= floor), key=lambda p: (p["cost"], -p["score"]))
    res = {"from": src, "name": name, "options": opt, "tol": TOL, "splits": SPLITS, "params": params,
           "cost_bytes": cost, "path": path, "full_score": path[0]["score"], "floor": floor,
           "selected": chosen["members"], "selected_score": chosen["score"], "method": "backward-elimination",
           "command": f"python -m kelter.ensemble_select --from {src} --name {name}"}
    (OUT / f"{name}_select.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    res = select(args.src, args.name)
    print(json.dumps({k: res[k] for k in ("full_score", "floor", "selected", "selected_score")}, indent=2))
    return 0


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("ensemble fitting (python -m kelter.ensemble_select)")
    raise SystemExit(main())
