"""Evaluation harness: every system through the same sets and metrics.

python -m kelter.eval.run --system ours --ckpt checkpoints/r1 [--limit N] [--tag r1]
python -m kelter.eval.run --system tfidf

Writes results/eval/<tag or system>.json. Every metric is defined in
kelter/eval/metrics.py. Sets come from data/built (see data_card.md);
JevBench is evaluated separately through its own harness (P6).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from kelter.data.augment import mask_spans, witness_only
from kelter.eval import metrics as M
from kelter.eval.adapters import get_system

ROOT = Path(__file__).resolve().parents[2]
BUILT = ROOT / "data" / "built"
NULL = "__null__"


def load(name: str, limit=None, seed=0) -> List[Dict[str, Any]]:
    recs = [json.loads(l) for l in (BUILT / f"{name}.jsonl").open(encoding="utf-8")]
    if limit and len(recs) > limit:
        rng = random.Random(seed)
        rng.shuffle(recs)
        recs = recs[:limit]
    return recs


def verdict(out: Dict[str, Any]) -> str:
    """The predicted label of one model output, for scoring: the best option, or NULL when the null wins.
    With a separate null head (r6 and later) the null wins when its calibrated score clears the stored
    threshold; otherwise when the null probability beats the best option. An output with no "null" (a baseline
    without a null option) can never give NULL. Same rule as kelter.engine.verdict_of."""
    best = max(out["probs"], key=out["probs"].get)
    if "null_threshold" in out:  # separate null head (r6+): null iff the calibrated score clears its threshold
        return NULL if out["null"] > out["null_threshold"] else best
    if out.get("null") is not None and out["null"] > out["probs"][best]:
        return NULL
    return best


def forced(out: Dict[str, Any]) -> str:
    """The best option even when the null would win: the forced-choice answer, for accuracy on answerable items."""
    return max(out["probs"], key=out["probs"].get)


def null_score(out: Dict[str, Any]) -> float:
    """How strongly the model says the item is unanswerable, for null recall at a false-positive rate. The
    model's own null probability when it has one; for a model without a null (a baseline), 1 minus its top
    option probability, so a low-confidence answer counts as a null signal."""
    if out.get("null") is not None:
        return float(out["null"])
    return 1.0 - max(out["probs"].values())


def gold_label(r: Dict[str, Any]) -> str:
    if r["null"]:
        return NULL
    g = r["gold"]
    return g if isinstance(g, str) else max(g, key=g.get)


def prob_matrix(recs, outs):
    """Rows over [options..., null] with the gold index, for ECE and Brier.

    Silent behaviours, all deliberate so a baseline can be scored the same way:
      - an option missing from out["probs"] counts as probability 0.0;
      - a missing null (out["null"] is None) counts as 0.0;
      - every row is renormalised to sum to 1, so a system whose probabilities do not sum to 1 is scored on the
        shares it gave;
      - a row that sums to 0 becomes uniform over its options and the null;
      - rows are padded with 0.0 up to K = (most options in any record) + 1 columns, so a record with fewer
        options has its null at column len(options), not in the last column, and G holds that index for a
        null-gold record.
    """
    P, G = [], []
    K = max(len(r["options"]) for r in recs) + 1
    for r, o in zip(recs, outs):
        names = [x["name"] for x in r["options"]]
        row = [o["probs"].get(n, 0.0) for n in names]
        nul = o.get("null")
        row.append(0.0 if nul is None else nul)
        s = sum(row)
        row = [x / s for x in row] if s > 0 else [1.0 / len(row)] * len(row)
        row += [0.0] * (K - len(row))
        P.append(row)
        g = gold_label(r)
        G.append(len(names) if g == NULL else names.index(g))
    return np.array(P), np.array(G)


def shuffled(r: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    if r["question"]["type"] != "choice":
        return r
    opts = list(r["options"])
    rng.shuffle(opts)
    return dict(r, options=opts, question=dict(r["question"], criteria={o["name"]: o["description"] for o in opts}))


def random_mask_spans(state: str, budget: int, rng: random.Random, protect=("Premise:", "Hypothesis:")):
    words, pos = [], 0
    for w in state.split(" "):
        if w and w not in protect and "\n" not in w:
            words.append([pos, pos + len(w)])
        pos += len(w) + 1
    rng.shuffle(words)
    out, used = [], 0
    for s, e in words:
        if used >= budget:
            break
        out.append([s, e])
        used += e - s
    return out


def corrected_null(system, answerable, outs) -> Dict[str, Any]:
    """Null recall at 5% FPR on the corrected unknowable set, overall and per
    group, against the same answerable items' null scores."""
    p = BUILT / "test_unknowable_corrected.jsonl"
    if not p.exists():
        return {}
    ids = {r["id"] for r in answerable}
    unk = [r for r in load("test_unknowable_corrected")
           if r["meta"]["unk_group"] == "natural_null" or r["id"].split("~", 1)[0] in ids]
    # Note: split("~", 1)[0] cuts at the FIRST "~" here, while the nudge, forge and null filters below use
    # rsplit("~", 1)[0], which cuts at the LAST. They give the same base id on every id in the current data (each id
    # has zero or one "~"); they would differ for an id with two (a chained augmentation) or a "~" in the base id.
    uo = system.predict(unk)
    neg = [null_score(o) for o in outs]
    pos = [null_score(o) for o in uo]
    res = {"null_recall_at_5fpr_corrected": M.null_recall_at_fpr(neg, pos, 0.05),
           "null_n_unknowable_corrected": len(unk)}
    t = res["null_recall_at_5fpr_corrected"]["threshold"]
    groups = collections.defaultdict(list)
    for r, s in zip(unk, pos):
        groups[r["meta"]["unk_group"]].append(s > t)
        groups[f"{r['source']}/{r['meta']['unk_group']}"].append(s > t)
    res["null_recall_corrected_by_group"] = {k: {"recall": float(np.mean(v)), "n": len(v)} for k, v in sorted(groups.items())}
    return res


def run(system, limit=None, seed=0) -> Dict[str, Any]:
    res: Dict[str, Any] = {"system": system.name, "seed": seed, "limit": limit}
    t0 = time.time()

    test = load("test", limit, seed)
    answerable = [r for r in test if not r["null"]]
    natural_null = [r for r in test if r["null"]]
    outs = system.predict(answerable)
    pred = [verdict(o) for o in outs]
    gold = [gold_label(r) for r in answerable]
    res["heldout_accuracy"] = M.accuracy(pred, gold)
    res["heldout_accuracy_forced"] = M.accuracy([forced(o) for o in outs], gold)
    res["heldout_n"] = len(answerable)
    by_src = collections.defaultdict(list)
    for r, p, g in zip(answerable, pred, gold):
        by_src[r["source"]].append(p == g)
    res["heldout_accuracy_by_source"] = {k: {"acc": float(np.mean(v)), "n": len(v)} for k, v in sorted(by_src.items())}
    P, G = prob_matrix(answerable, outs)
    res["ece_15"] = M.ece(P, G, 15)
    res["brier"] = M.brier(P, G)
    if hasattr(system, "temperature"):
        old = system.temperature
        system.temperature = False
        raw = system.predict(answerable)
        system.temperature = old
        P0, G0 = prob_matrix(answerable, raw)
        res["ece_15_before_temperature"] = M.ece(P0, G0, 15)

    # Flip rate under a random permutation of choice options.
    rng = random.Random(seed + 1)
    choice = [r for r in answerable if r["question"]["type"] == "choice"]
    perm = [shuffled(r, rng) for r in choice]
    a = [verdict(o) for o in system.predict(choice, canonical=True)]
    b = [verdict(o) for o in system.predict(perm, canonical=True)]
    res["flip_rate_canonical"] = M.paired_flip_rate(a, b)
    try:
        a2 = [verdict(o) for o in system.predict(choice, canonical=False)]
        b2 = [verdict(o) for o in system.predict(perm, canonical=False)]
        res["flip_rate_no_canonical"] = M.paired_flip_rate(a2, b2)
    except TypeError:
        res["flip_rate_no_canonical"] = None
    res["flip_n"] = len(choice)

    # Injection: clean verdict vs verdict with planted text. An id with a "~" suffix is a derived copy of the item
    # whose id comes before the "~" (kelter.data.augment adds ~perm, ~rm, ~dist, ~plant, ~mask, ~wmask, ~wonly, ~all,
    # ~untr and ~rmask; engine_eval adds ~auth); rsplit("~", 1)[0] cuts off that last suffix to find the clean item.
    clean = {r["id"]: p for r, p in zip(answerable, pred)}
    for kind in ("nudge", "forge"):
        att = [r for r in load(f"test_{kind}") if r["id"].rsplit("~", 1)[0] in clean]
        ao = system.predict(att)
        cp = [clean[r["id"].rsplit("~", 1)[0]] for r in att]
        ap = [verdict(o) for o in ao]
        tg = [r["meta"]["attack"]["target"] for r in att]
        res[f"{kind}_asr"] = M.attack_success_rate(cp, ap, tg)

    # Null recall at 5% FPR: unknowable = answer-removed copies plus natural nulls.
    tn = load("test_null")
    ids = {r["id"] for r in answerable}
    tn = [r for r in tn if r["id"].rsplit("~", 1)[0] in ids]
    unk = tn + natural_null
    uo = system.predict(unk)
    res["null_recall_at_5fpr"] = M.null_recall_at_fpr([null_score(o) for o in outs], [null_score(o) for o in uo], 0.05)
    res["null_n_unknowable"] = len(unk)
    res["null_predicted_rate_unknowable"] = float(np.mean([verdict(o) == NULL for o in uo]))
    res["null_predicted_rate_answerable"] = float(np.mean([p == NULL for p in pred]))
    if limit is None:
        res.update(corrected_null(system, answerable, outs))

    # Witness quality, necessity and sufficiency (e-SNLI and SQuAD 2.0 test items).
    wt = load("witness_test", limit, seed)
    wo = system.predict(wt)
    if all("witness" in o for o in wo):
        for src in ("esnli", "squad2"):
            idx = [i for i, r in enumerate(wt) if r["source"] == src]
            res[f"witness_f1_{src}"] = M.mean_span_f1([wo[i]["witness"] for i in idx], [wt[i]["witness"] for i in idx])
        es = [i for i, r in enumerate(wt) if r["source"] == "esnli"]
        recs = [wt[i] for i in es]
        full = [verdict(wo[i]) for i in es]
        correct = [f == gold_label(r) for f, r in zip(full, recs)]
        # Four e-SNLI variants of each item, each re-run through the model:
        #   masked: the model's own witness masked out (necessity: the verdict should change)
        #   only:   the model's own witness kept, everything else masked (sufficiency: the verdict should hold)
        #   rnd:    a random span of the same total length masked (control for necessity: should flip far less)
        #   gm:     the human rationale (the item's gold witness) masked (what a human-marked necessity gives)
        masked = [mask_spans(r, wo[i]["witness"]) if wo[i]["witness"] else r for i, r in zip(es, recs)]
        mo = [verdict(o) for o in system.predict(masked)]
        only = []
        for i, r in zip(es, recs):
            w = wo[i]["witness"]
            # an empty witness keeps nothing: the witness-only pass sees a fully masked state
            only.append(witness_only(dict(r, witness=w)) if w else mask_spans(r, [[0, len(r["state"])]]))
        oo = [verdict(o) for o in system.predict(only)]
        rr = random.Random(seed + 2)
        rnd = [mask_spans(r, random_mask_spans(r["state"], sum(e - s for s, e in (wo[i]["witness"] or [])), rr))
               if wo[i]["witness"] else r for i, r in zip(es, recs)]
        ro = [verdict(o) for o in system.predict(rnd)]
        gm = [mask_spans(r, r["witness"]) for r in recs]
        go = [verdict(o) for o in system.predict(gm)]
        res["esnli_accuracy"] = float(np.mean(correct))
        res["witness_necessity_esnli"] = M.witness_necessity(correct, full, mo)
        res["witness_sufficiency_esnli"] = M.witness_sufficiency(correct, full, oo)
        res["random_mask_flip_esnli"] = M.witness_necessity(correct, full, ro)
        res["human_rationale_mask_flip_esnli"] = M.witness_necessity(correct, full, go)
        res["witness_empty_rate_esnli"] = float(np.mean([not wo[i]["witness"] for i in es]))
        res["witness_n_esnli"] = len(es)

    an = load("anli_test", limit, seed)
    ao = system.predict(an)
    res["anli_accuracy"] = M.accuracy([verdict(o) for o in ao], [gold_label(r) for r in an])
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-only", action="store_true", help="add corrected-set null metrics to an existing results file")
    ap.add_argument("--before-temperature-only", action="store_true",
                    help="add ece_15_before_temperature to an existing results file")
    args = ap.parse_args(argv)
    system = get_system(args.system, ckpt=args.ckpt, model_id=args.model_id)
    tag = args.tag or args.system
    out = ROOT / "results" / "eval"
    if args.before_temperature_only:
        res = json.loads((out / f"{tag}.json").read_text(encoding="utf-8"))
        answerable = [r for r in load("test") if not r["null"]]
        system.temperature = False
        P0, G0 = prob_matrix(answerable, system.predict(answerable))
        res["ece_15_before_temperature"] = M.ece(P0, G0, 15)
        res["command_before_temperature"] = (f"python -m kelter.eval.run --system {args.system} --tag {tag} "
                                             f"--before-temperature-only" + (f" --ckpt {args.ckpt}" if args.ckpt else ""))
        (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(res["ece_15_before_temperature"])
        return 0
    if args.null_only:
        # add the corrected-set null metrics to an existing full results file
        res = json.loads((out / f"{tag}.json").read_text(encoding="utf-8"))
        answerable = [r for r in load("test") if not r["null"]]
        res.update(corrected_null(system, answerable, system.predict(answerable)))
        res["command_null_corrected"] = f"python -m kelter.eval.run --system {args.system} --tag {tag} --null-only" + (
            f" --ckpt {args.ckpt}" if args.ckpt else "")
        (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if "corrected" in k}, indent=2))
        return 0
    res = run(system, args.limit, args.seed)
    res["command"] = "python -m kelter.eval.run " + " ".join(
        f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items() if v is not None and v is not False)
    from kelter.llm import prefill_chunk_setting
    res["prefill_chunk"] = prefill_chunk_setting()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("evaluation (python -m kelter.eval.run)")
    raise SystemExit(main())
