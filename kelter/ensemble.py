"""Score ensemble of trained checkpoints.

python -m kelter.ensemble --name E1 --options r2 --null r3 r5 r7 rL10
python -m kelter.ensemble --install E1     # writes checkpoints/E1/config.json

One checkpoint chooses the option and gives the witness. The null score is
the mean, over the null checkpoints, of each checkpoint's null score as a
quantile among its own null scores on the calibration split's answerable
records; a one-variable logistic fit on calib.jsonl (answerable against
natural nulls) turns it into a probability. Option probabilities are the
option checkpoint's, renormalized over the options and scaled by 1 minus the
null probability. Everything fitted comes from the calibration split only and
is written to results/ensemble/<name>.json.

checkpoints/<name>/config.json marks the directory as an ensemble, so every
tool that loads a checkpoint through kelter.model.load_predictor (engine,
server, latency, calibration, evaluation) serves it like a single checkpoint.
Its sha256 (ensemble_sha256) covers the spec file and every member's model.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ensemble"


def null_score(out: Dict[str, Any]) -> float:
    # same as kelter.eval.run.null_score
    if out.get("null") is not None:
        return float(out["null"])
    return 1.0 - max(out["probs"].values())


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 50):
    """Newton's method for p = sigmoid(a x + b)."""
    a, b = 0.0, 0.0
    X = np.stack([x, np.ones_like(x)], 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        g = X.T @ (p - y)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(H, g)
    return float(a), float(b)


def members(name: str) -> List[str]:
    spec = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
    return [spec["options"]] + list(spec["null"])


def ensemble_sha256(name: str) -> str:
    """sha256 over the spec file's bytes and each member's model.pt sha256, in member order,
    then the stacker file's sha256 when the spec has one."""
    h = hashlib.sha256((OUT / f"{name}.json").read_bytes())
    for c in members(name):
        h.update(hashlib.sha256((ROOT / "checkpoints" / c / "model.pt").read_bytes()).hexdigest().encode())
    st = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8")).get("stacker")
    if st:
        h.update(hashlib.sha256((ROOT / st).read_bytes()).hexdigest().encode())
    return h.hexdigest()


def install(name: str) -> Path:
    """checkpoints/<name>/config.json; keeps calibration already written there."""
    d = ROOT / "checkpoints" / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    opt = json.loads((ROOT / "checkpoints" / members(name)[0] / "config.json").read_text(encoding="utf-8"))
    cfg.update({"ensemble": name, "spec": f"results/ensemble/{name}.json", "members": members(name),
                "model_hash": ensemble_sha256(name), "infer_max_len": opt["infer_max_len"], "max_len": opt["max_len"]})
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


class Ensemble:
    def __init__(self, name: str, device: str | None = None, cfg: Dict[str, Any] | None = None):
        from kelter.device import default_device
        from kelter.model import Predictor
        device = device or default_device()
        from kelter import paths
        self.spec = json.loads(paths.asset(f"results/ensemble/{name}.json").read_text(encoding="utf-8"))
        ck = paths.home() / "checkpoints"
        self.opt = Predictor(str(ck / self.spec["options"]), device=device)
        # a null member that is also the option checkpoint shares its model and its forward pass (E2: r2)
        self.nulls = {c: self.opt if c == self.spec["options"] else Predictor(str(ck / c), device=device)
                      for c in self.spec["null"]}
        self.refs = {c: np.asarray(v) for c, v in self.spec["refs"].items()}
        self.a, self.b = self.spec["logistic"]["a"], self.spec["logistic"]["b"]
        self.stacker = None
        if self.spec.get("stacker"):  # stacked: a classifier over the members' quantiles replaces their mean
            # A pickle can run code when loaded, so the stacker file is unpickled only after its sha256 matches
            # stacker_sha256 in the spec (below); a mismatch raises before pickle.loads is reached.
            import pickle
            st = paths.under_home(self.spec["stacker"])
            if hashlib.sha256(st.read_bytes()).hexdigest() != self.spec["stacker_sha256"]:
                raise ValueError(f"{st} does not match stacker_sha256 in the spec")
            self.stacker = pickle.loads(st.read_bytes())
        self.name = f"kelter:{name}"
        self.cfg = cfg or {}
        self.model_hash = self.cfg.get("model_hash", "")
        self.tok, self.feat, self.null_head = self.opt.tok, self.opt.feat, False
        self.temperature = True  # False: every member scores without its fitted temperature (ECE before temperature)

    def quantize_int8(self) -> None:
        """Dynamic int8 on every member (CPU latency, as for single checkpoints)."""
        import torch
        for p in {id(x): x for x in [self.opt, *self.nulls.values()]}.values():
            p.model = torch.ao.quantization.quantize_dynamic(p.model.float(), {torch.nn.Linear}, dtype=torch.qint8)

    def quantile(self, c: str, s) -> np.ndarray:
        """Where null scores s fall among member c's own null scores on the calibration split's ANSWERABLE
        records: an empirical CDF, the fraction of reference scores at or below s (side="right" counts ties).
        Near 1 means more null-like than almost every answerable record. self.refs[c] is that sorted reference."""
        ref = self.refs[c]
        return np.searchsorted(ref, np.asarray(s, dtype=float), side="right") / len(ref)

    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True, temperature: bool | None = None,
                **_) -> List[Dict[str, Any]]:
        t = self.temperature if temperature is None else temperature
        outs = self.opt.predict(recs, canonical=canonical, temperature=t)
        Q = np.stack([self.quantile(c, [null_score(o) for o in (
            outs if p is self.opt else p.predict(recs, canonical=canonical, temperature=t))])
                      for c, p in self.nulls.items()], 1)
        q = self.stacker.predict_proba(Q)[:, 1] if self.stacker is not None else Q.mean(1)
        res = []
        for o, qi in zip(outs, q):
            pn = 1.0 / (1.0 + math.exp(-(self.a * qi + self.b)))
            tot = sum(o["probs"].values())
            probs = {k: v / tot * (1.0 - pn) for k, v in o["probs"].items()}
            res.append({**o, "probs": probs, "null": pn, "null_quantile": float(qi)})
        return res


def fit_stacked(name: str, options: str, null: List[str], folds: int = 5) -> Dict[str, Any]:
    """Stacked null score: a gradient-boosted classifier over the members' quantiles, trained on
    calib answerable records against the calib-side unknowables (the held-out
    builder applied to calib.jsonl). The logistic map to a probability is fitted
    on out-of-fold classifier scores of calib.jsonl's own records, so the
    classifier's fit to its training rows does not leak into calibration."""
    import pickle

    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    from kelter.device import default_device
    from kelter.eval import unknowable as U
    from kelter.model import Predictor
    calib = [json.loads(l) for l in (ROOT / "data" / "built" / "calib.jsonl").open(encoding="utf-8")]
    m = SentenceTransformer(U.MODEL, device=default_device())
    unk, stats = U.build(calib, lambda t: m.encode(t, normalize_embeddings=True, convert_to_numpy=True, batch_size=128))
    del m
    ans = [r for r in calib if not r["null"]]
    rows = ans + unk
    y = np.r_[np.zeros(len(ans)), np.ones(len(unk))]
    refs, cols = {}, []
    for c in null:
        p = Predictor(str(ROOT / "checkpoints" / c))
        s = np.array([null_score(o) for o in p.predict(rows)])
        refs[c] = np.sort(s[: len(ans)])
        cols.append(np.searchsorted(refs[c], s, side="right") / len(refs[c]))
        del p
    X = np.stack(cols, 1)
    params = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.05, "random_state": 0}
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        oof[te] = HistGradientBoostingClassifier(**params).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    clf = HistGradientBoostingClassifier(**params).fit(X, y)
    # calibration on calib.jsonl's own population: its answerable records and its natural nulls
    nat = np.array([x["meta"]["unk_group"] == "natural_null" for x in unk])
    keep = np.r_[np.ones(len(ans), bool), nat]
    a, b = fit_logistic(oof[keep], y[keep])
    d = ROOT / "checkpoints" / name
    d.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(clf)
    (d / "stacker.pkl").write_bytes(blob)
    spec = {"name": name, "options": options, "null": null, "refs": {c: v.tolist() for c, v in refs.items()},
            "combiner": "gbm", "stacker": f"checkpoints/{name}/stacker.pkl",
            "stacker_sha256": hashlib.sha256(blob).hexdigest(), "stacker_params": params, "folds": folds,
            "logistic": {"a": a, "b": b}, "fit_on": "data/built/calib.jsonl and its unknowables (kelter.eval.unknowable.build)",
            "unknowable_counts": stats["counts"], "n_answerable": len(ans), "n_unknowable": len(unk),
            "method": "stacked",
            "command": f"python -m kelter.ensemble --name {name} --options {options} --null {' '.join(null)} --stacked"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec


def fit(name: str, options: str, null: List[str]) -> Dict[str, Any]:
    from kelter.model import Predictor
    calib = [json.loads(l) for l in (ROOT / "data" / "built" / "calib.jsonl").open(encoding="utf-8")]
    y = np.array([1.0 if r["null"] else 0.0 for r in calib])
    refs, qs = {}, []
    for c in null:
        p = Predictor(str(ROOT / "checkpoints" / c))
        s = np.array([null_score(o) for o in p.predict(calib)])
        refs[c] = np.sort(s[y == 0])  # reference: this member's null scores on the answerable records (y == 0), sorted
        qs.append(np.searchsorted(refs[c], s, side="right") / len(refs[c]))  # empirical CDF against that reference
        del p
    x = np.mean(qs, 0)
    a, b = fit_logistic(x, y)
    spec = {"name": name, "options": options, "null": null, "refs": {c: v.tolist() for c, v in refs.items()},
            "logistic": {"a": a, "b": b}, "fit_on": "data/built/calib.jsonl", "n_answerable": int((y == 0).sum()),
            "n_null": int(y.sum()), "method": "mean",
            "command": f"python -m kelter.ensemble --name {name} --options {options} --null {' '.join(null)}"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--options")
    ap.add_argument("--null", nargs="+")
    ap.add_argument("--stacked", action="store_true", help="stacked null score: a classifier over member quantiles instead of their mean")
    ap.add_argument("--install", help="write checkpoints/<name>/config.json for a fitted ensemble")
    args = ap.parse_args(argv)
    if args.install:
        print(install(args.install))
        return 0
    spec = (fit_stacked if args.stacked else fit)(args.name, args.options, args.null)
    print(json.dumps({k: v for k, v in spec.items() if k != "refs"}, indent=2))
    return 0


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("ensemble fitting (python -m kelter.ensemble)")
    raise SystemExit(main())
