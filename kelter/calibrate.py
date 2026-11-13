"""Fit per-question-type temperature and split-conformal thresholds on the
calib split (held-out sources only), then store both in the checkpoint config.

python -m kelter.calibrate --ckpt checkpoints/r1

Temperature: per question type, the T in a grid over [0.25, 6] minimizing
NLL of the gold label (null included as a label) on calib.
Conformal: nonconformity 1 - p(gold) with the temperature applied; for each
alpha and each domain tag (plus "_global" over all calib items), threshold
q = the ceil((n + 1)(1 - alpha))-th smallest score (split conformal,
Vovk et al.; Angelopoulos and Bates 2021). Writes results/calibration/<run>.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

from kelter.eval import metrics as M
from kelter.eval.run import load, prob_matrix
from kelter.model import load_predictor

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.05, 0.1)


def nll_at(logits_rows, golds, T):
    tot = 0.0
    for lg, g in zip(logits_rows, golds):
        z = np.asarray(lg) / T
        z = z - z.max()
        tot += -(z[g] - math.log(np.exp(z).sum()))
    return tot / len(golds)


# The gold label of a record appears below as `rec["gold"] if isinstance(rec["gold"], str) else max(rec["gold"],
# key=rec["gold"].get)`: gold is either one option name (a string) or a soft distribution {option name: probability}
# (a dict); in the second case the label used here is the most probable name (the first one on a tie). Null
# records are handled before this expression, as the null label, or left out.
def gold_pos(rec, out):
    """Position of the gold label in out['logits'] (canonical option order, null last)."""
    if rec["null"]:
        return len(out["order"])
    g = rec["gold"] if isinstance(rec["gold"], str) else max(rec["gold"], key=rec["gold"].get)
    return out["order"].index(g)


def conformal_q(scores, alpha):
    """Split-conformal threshold: the ceil((n + 1)(1 - alpha))-th smallest nonconformity score among the n
    calibration scores. A label then enters the prediction set when 1 - p(label) <= q, which covers the gold label
    with probability at least 1 - alpha on data like the calibration split. With too few scores (k > n) there is
    no valid quantile, so q = 1.0: every label is in the set and the engine never acts on its own."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return 1.0
    return float(np.sort(scores)[k - 1])


def _recall_at_fpr(scores, is_null, fpr=0.05):
    neg = [s for s, y in zip(scores, is_null) if not y]
    pos = [s for s, y in zip(scores, is_null) if y]
    return M.null_recall_at_fpr(neg, pos, fpr)


def calibrate_null_head(pred, calib, ck, args) -> int:
    """r6 calibration on the calib split only:
    per-type temperature over option logits on answerable items; T_null for
    the null logit by BCE NLL on all items; a logistic stacker over
    [null_logit, 1 - max option prob, option entropy] that replaces the
    T-scaled null score iff its calib null recall at 5% FPR is strictly
    higher; the null threshold at 5% FPR over answerable calib items; split
    conformal over options only on answerable items."""
    from sklearn.linear_model import LogisticRegression

    outs = pred.predict(calib, temperature=False)
    is_null = [bool(r["null"]) for r in calib]
    # Temperature grid: 0.25 to 6.00 in steps of 0.05 (116 values). The upper bound is 6.0001 so 6.0 is included
    # despite float rounding; round(., 2) removes the float noise so the stored T is a clean two-decimal number.
    grid = np.round(np.arange(0.25, 6.0001, 0.05), 2)
    by_type = collections.defaultdict(list)
    for r, o in zip(calib, outs):
        if not r["null"]:
            by_type[r["question"]["type"]].append((o["logits"], gold_pos(r, o)))
    temps, report = {}, {"per_type": {}, "null_head": True}
    for t, rows in by_type.items():
        lg, gs = [x[0] for x in rows], [x[1] for x in rows]
        nlls = [nll_at(lg, gs, T) for T in grid]
        temps[t] = float(grid[int(np.argmin(nlls))])
        report["per_type"][t] = {"n": len(rows), "T": temps[t], "nll_T1": nll_at(lg, gs, 1.0), "nll_best": min(nlls)}
    z = np.array([o["null_logit"] for o in outs])
    y = np.array(is_null, dtype=float)

    def bce(T):
        p = 1 / (1 + np.exp(-np.clip(z / T, -60, 60)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    t_null = float(grid[int(np.argmin([bce(T) for T in grid]))])
    pred.temps = temps
    pred.null_cal = {"T_null": t_null}
    outs_t = pred.predict(calib, temperature=True)
    po = [list(o["option_probs"].values()) for o in outs_t]
    feats = np.array([[o["null_logit"], 1 - max(p), -sum(x * math.log(max(x, 1e-12)) for x in p)] for o, p in zip(outs_t, po)])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats, y)
    s_plain = [o["null"] for o in outs_t]
    s_stack = list(lr.predict_proba(feats)[:, 1])
    r_plain = _recall_at_fpr(s_plain, is_null)
    r_stack = _recall_at_fpr(s_stack, is_null)
    use = r_stack["recall"] > r_plain["recall"]
    chosen = r_stack if use else r_plain
    cal = {"T_null": t_null, "use_stacker": bool(use), "stacker_w": [float(w) for w in lr.coef_[0]],
           "stacker_b": float(lr.intercept_[0]), "threshold": float(chosen["threshold"])}
    report["null"] = {"T_null": t_null, "calib_recall_plain": r_plain, "calib_recall_stacker": r_stack,
                      "use_stacker": bool(use), "rule": "stacker iff strictly higher calib null recall at 5% FPR",
                      "n_null": int(y.sum()), "n": len(y)}
    pred.null_cal = cal
    outs_f = pred.predict(calib, temperature=True)
    ans = [(r, o) for r, o in zip(calib, outs_f) if not r["null"]]
    scores = collections.defaultdict(list)
    for r, o in ans:
        s = 1.0 - o["option_probs"][r["gold"] if isinstance(r["gold"], str) else max(r["gold"], key=r["gold"].get)]
        scores[r["domain"]].append(s)
        scores["_global"].append(s)
    conf = {str(a): {d: conformal_q(np.array(v), a) for d, v in scores.items()} for a in ALPHAS}
    P1, G1 = prob_matrix(calib, outs_f)
    report["calib_ece_after"] = M.ece(P1, G1, 15)
    P0, G0 = prob_matrix(calib, outs)
    report["calib_ece_before"] = M.ece(P0, G0, 15)
    report["conformal"] = conf
    report["conformal_n"] = {d: len(v) for d, v in scores.items()}
    cfg_path = ck / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["temperature"], cfg["conformal"], cfg["null_calibration"] = temps, conf, cal
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    out = ROOT / "results" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    report["ckpt"] = str(ck)
    report["command"] = f"python -m kelter.calibrate --ckpt {args.ckpt}"
    (out / f"{ck.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def calibrate_ensemble(pred, calib, ck, args) -> int:
    """Ensemble: members keep their own temperatures and the
    null probability is fitted in kelter.ensemble, so only split conformal
    is fitted here, as for a single checkpoint without a null head."""
    outs = pred.predict(calib)
    report = {"ensemble": pred.spec["name"], "temperature": "members' own"}
    P, G = prob_matrix(calib, outs)
    report["calib_ece_after"] = M.ece(P, G, 15)
    scores = collections.defaultdict(list)
    for r, o in zip(calib, outs):
        p = o["null"] if r["null"] else o["probs"][r["gold"] if isinstance(r["gold"], str) else max(r["gold"], key=r["gold"].get)]
        scores[r["domain"]].append(1.0 - p)
        scores["_global"].append(1.0 - p)
    conf = {str(a): {d: conformal_q(np.array(v), a) for d, v in scores.items()} for a in ALPHAS}
    report["conformal"] = conf
    report["conformal_n"] = {d: len(v) for d, v in scores.items()}
    cfg_path = ck / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["conformal"] = conf
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    out = ROOT / "results" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    report["ckpt"] = str(ck).replace("\\", "/")
    report["command"] = f"python -m kelter.calibrate --ckpt {args.ckpt}"
    (out / f"{ck.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args(argv)
    ck = Path(args.ckpt)
    pred = load_predictor(str(ck))
    calib = load("calib")
    if pred.cfg.get("ensemble"):
        return calibrate_ensemble(pred, calib, ck, args)
    if pred.null_head:
        return calibrate_null_head(pred, calib, ck, args)
    outs = pred.predict(calib, temperature=False)
    by_type = collections.defaultdict(list)
    for r, o in zip(calib, outs):
        by_type[r["question"]["type"]].append((o["logits"], gold_pos(r, o)))
    grid = np.round(np.arange(0.25, 6.0001, 0.05), 2)  # same grid as calibrate_null_head: 0.25 to 6.00, step 0.05
    temps, report = {}, {"per_type": {}}
    for t, rows in by_type.items():
        lg = [x[0] for x in rows]
        gs = [x[1] for x in rows]
        nlls = [nll_at(lg, gs, T) for T in grid]
        best = float(grid[int(np.argmin(nlls))])
        temps[t] = best
        report["per_type"][t] = {"n": len(rows), "T": best, "nll_T1": nll_at(lg, gs, 1.0), "nll_best": min(nlls)}
    pred.temps = temps
    outs_t = pred.predict(calib, temperature=True)
    P0, G0 = prob_matrix(calib, outs)
    P1, G1 = prob_matrix(calib, outs_t)
    report["calib_ece_before"] = M.ece(P0, G0, 15)
    report["calib_ece_after"] = M.ece(P1, G1, 15)

    scores = collections.defaultdict(list)
    for r, o in zip(calib, outs_t):
        g = gold_pos(r, o)
        p = o["null"] if g == len(o["order"]) else o["probs"][o["order"][g]]
        s = 1.0 - p
        scores[r["domain"]].append(s)
        scores["_global"].append(s)
    conf = {str(a): {d: conformal_q(np.array(v), a) for d, v in scores.items()} for a in ALPHAS}
    report["conformal"] = conf
    report["conformal_n"] = {d: len(v) for d, v in scores.items()}

    cfg_path = ck / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["temperature"] = temps
    cfg["conformal"] = conf
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    out = ROOT / "results" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    report["ckpt"] = str(ck)
    report["command"] = f"python -m kelter.calibrate --ckpt {args.ckpt}"
    (out / f"{ck.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("calibration (python -m kelter.calibrate)")
    raise SystemExit(main())
