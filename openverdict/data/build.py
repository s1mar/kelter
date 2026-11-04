"""Build the training corpus and evaluation sets.

python -m openverdict.data.build [--small]

Roles
- Train sources: their train split -> train, their validation split -> dev.
- New-source holdout (never trained on): clinc150, arc_challenge, and the
  synthetic policy family HOLDOUT_FAMILY. Their items are split by a hash of
  the id into calib (temperature and conformal fitting) and test (reporting).
- Eval-only: anli (CC BY-NC 4.0; never trained on).
- Witness eval: esnli and squad2 test splits (sources seen in training, items
  not seen).
- JevBench is never read here, so no JevBench text reaches any split.

Outputs under data/built/: train.jsonl, dev.jsonl, calib.jsonl, test.jsonl,
test_null.jsonl, test_nudge.jsonl, test_forge.jsonl, witness_test.jsonl,
anli_test.jsonl, manifest.json; and the data card under docs/data_cards/.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openverdict.data import augment, synthetic, synthetic_score, synthetic_shapes
from openverdict.data.converters import (
    anli, arc, banking77, boolq, civil_comments, clinc150, esnli, massive, mmlu, mnli, paws, squad2,
)
from openverdict.data.schema import validate

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "docs" / "data_cards"


def card_path(name: str) -> Path:
    CARDS.mkdir(parents=True, exist_ok=True)
    return CARDS / name
OUT = ROOT / "data" / "built"

TRAIN_SOURCES = {
    "banking77": banking77.convert,
    "massive": massive.convert,
    "paws": paws.convert,
    "civil_comments": civil_comments.convert,
    "boolq": boolq.convert,
    "mnli": mnli.convert,
    "esnli": esnli.convert,
    "squad2": squad2.convert,
    "arc_easy": arc.convert_easy,
    "mmlu": mmlu.convert,
}
HOLDOUT_SOURCES = {
    "clinc150": clinc150.convert,
    "arc_challenge": arc.convert_challenge,
}
HOLDOUT_FAMILY = "sharing"
N_SYNTHETIC = 8000
FILTER_TRAIN_NULLS = False  # shipped r2 trained on unfiltered nulls; r6 and r7 used True
N_SHAPES = 0  # shipped r2 used none; r4, r5, r7 used 9000
MAX_OPTIONS_TRAIN = 16
EVAL_OPTIONS = 10
SEED = 0

# Training augmentation probabilities, applied independently per record.
AUG_P = {
    "remove_answer": 0.15,  # shipped r2; r3 0.35, r4 and r5 0.25
    "nudge": 0.10,
    "distractors": 0.10,
    "mask_witness": 0.10,
    "witness_only": 0.10,
    "mask_untrusted": 0.10,
    "mask_whole_state": 0.02,
    "swap_state": 0.03,  # shipped r2; r4, r5, r7 used 0.05
    "mask_random_nonwitness": 0.25,
}

# r7 (now a null-score member of the shipped ensemble) was trained
# on this recipe. `--recipe r7 --out data/built_r7` rebuilds that corpus beside
# the shipped r2 data so its card and leakage check stay reproducible.
# E2 adds r1 (before mask_random_nonwitness), r4 (r5's corpus)
# and r6 (r2's recipe with filtered nulls). E1 also draws on r3 and r5; `--recipe r3 --out data/built_r3`
# and `--recipe r5 --out data/built_r5` rebuild their corpora for the card and the
# leakage check (r5 trained on r4's data).
RECIPES = {
    "r1": {"N_SHAPES": 0, "FILTER_TRAIN_NULLS": False, "swap_state": 0.03, "mask_random_nonwitness": 0.0},
    "r2": {"N_SHAPES": 0, "FILTER_TRAIN_NULLS": False, "swap_state": 0.03},
    "r3": {"N_SHAPES": 0, "FILTER_TRAIN_NULLS": False, "swap_state": 0.03, "remove_answer": 0.35},
    "r5": {"N_SHAPES": 9000, "FILTER_TRAIN_NULLS": False, "swap_state": 0.05, "remove_answer": 0.25},
    "r6": {"N_SHAPES": 0, "FILTER_TRAIN_NULLS": True, "swap_state": 0.03},
    "r7": {"N_SHAPES": 9000, "FILTER_TRAIN_NULLS": True, "swap_state": 0.05},
}


def mask_random_nonwitness(rec: Dict[str, Any], rng: random.Random):
    """Mask random words outside the witness, with a character budget equal to
    the witness length; label unchanged. Breaks the shortcut "any [MASK] means
    null"."""
    wit = rec["witness"] or []
    budget = sum(e - s for s, e in wit)
    words, pos = [], 0
    st = rec["state"]
    for w in st.split(" "):
        s, e = pos, pos + len(w)
        if w.strip() and "\n" not in w and not any(s < we and e > ws for ws, we in wit):
            words.append([s, e])
        pos = e + 1
    rng.shuffle(words)
    spans, used = [], 0
    for s, e in words:
        if used >= budget:
            break
        spans.append([s, e])
        used += e - s
    if not spans:
        return None
    r = augment.mask_spans(rec, spans)
    r["id"] = rec["id"] + "~rmask"
    meta = r.setdefault("meta", {})
    prev = meta.get("aug")
    meta["aug"] = [x for x in ([prev] if isinstance(prev, str) else (prev or [])) if x] + ["mask_random_nonwitness"]
    return r


def h01(s: str) -> float:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def subsample_options(rec: Dict[str, Any], k: int, rng: random.Random) -> Dict[str, Any]:
    """Keep gold plus k-1 random others for choice records with more than k options."""
    if rec["question"]["type"] != "choice" or len(rec["options"]) <= k:
        return rec
    r = copy.deepcopy(rec)
    names = [o["name"] for o in r["options"]]
    keep = set()
    if isinstance(r["gold"], str):
        keep.add(r["gold"])
    others = [n for n in names if n not in keep]
    rng.shuffle(others)
    keep.update(others[: k - len(keep)])
    r["options"] = [o for o in r["options"] if o["name"] in keep]
    r["question"]["criteria"] = {o["name"]: o["description"] for o in r["options"]}
    return r


def swap_state(rec: Dict[str, Any], donor: Dict[str, Any]) -> Dict[str, Any]:
    """State from an unrelated source: no option fits."""
    r = copy.deepcopy(rec)
    r["state"] = donor["state"]
    r["provenance"] = copy.deepcopy(donor["provenance"])
    r["null"], r["gold"], r["witness"] = True, None, None
    r["id"] = rec["id"] + "~swap"
    r.setdefault("meta", {})["aug"] = "swap_state"
    r["meta"]["donor"] = donor["id"]
    return r


def opt_text_sim(o: Dict[str, str]) -> str:
    return f"{o['name'].replace('_', ' ')}: {o['description']}"


def filter_answer_removed(recs: List[Dict[str, Any]], sim_max: float = 0.80) -> tuple:
    """Drop answer-removed records whose remaining options include one with
    cosine >= sim_max to the removed gold (all-mpnet-base-v2 on "name:
    description"), the same rule as the corrected unknowable test set
    Returns (kept records, number removed)."""
    flagged = [r for r in recs if "_filter" in r]
    if not flagged:
        return recs, 0
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from openverdict.device import default_device
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=default_device())
    texts = sorted({t for r in flagged for t in [r["_filter"][0], *r["_filter"][1]]})
    vec = dict(zip(texts, m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256)))
    out, removed = [], 0
    for r in recs:
        f = r.pop("_filter", None)
        if f is not None:
            g = vec[f[0]]
            if max(float(np.dot(g, vec[t])) for t in f[1]) >= sim_max:
                removed += 1
                continue
        out.append(r)
    return out, removed


def augment_train(recs: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    by_domain = collections.defaultdict(list)
    for r in recs:
        if r["question"]["type"] == "choice":
            by_domain[r["domain"]].extend(r["options"])
    out = []
    for r in recs:
        out.append(r)
        cands = []
        if rng.random() < AUG_P["remove_answer"]:
            rm = augment.remove_answer(r, rng)
            if rm is not None and FILTER_TRAIN_NULLS:
                gold = next(o for o in r["options"] if o["name"] == r["gold"])
                rm["_filter"] = (opt_text_sim(gold), [opt_text_sim(o) for o in rm["options"]])
            cands.append(rm)
        if rng.random() < AUG_P["nudge"]:
            cands.append(augment.plant_instruction(r, rng, kind="nudge"))
        if rng.random() < AUG_P["distractors"]:
            pool = by_domain[r["domain"]]
            sample = [pool[rng.randrange(len(pool))] for _ in range(20)] if pool else []
            cands.append(augment.add_distractors(r, rng, sample, k=rng.randint(1, 3)))
        if rng.random() < AUG_P["mask_witness"]:
            cands.append(augment.mask_witness(r))
        if rng.random() < AUG_P["witness_only"]:
            cands.append(augment.witness_only(r))
        if r["witness"] and AUG_P["mask_random_nonwitness"] > 0 and rng.random() < AUG_P["mask_random_nonwitness"]:  # r1: no draw
            cands.append(mask_random_nonwitness(r, rng))
        if rng.random() < AUG_P["mask_untrusted"]:
            cands.append(augment.mask_untrusted(r))
        if rng.random() < AUG_P["mask_whole_state"]:
            cands.append(augment.mask_whole_state(r))
        if rng.random() < AUG_P["swap_state"]:
            donor = recs[rng.randrange(len(recs))]
            if donor["source"] != r["source"]:
                cands.append(swap_state(r, donor))
        out.extend(c for c in cands if c is not None)
    return out


def write_jsonl(path: Path, recs: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            validate(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


CARD_HEAD = {
    "r2": ["# Data card: r2 corpus (option member of the shipped ensemble; also rL10)", "",
           "Generated by `python -m openverdict.data.build`. Counts are of records written.", "",
           "The default variant is the stacked score ensemble E3 (data_card_E3.md). This card "
           "describes the corpus r2 and rL10 were trained on; r2 chooses the option and witness and is also a "
           "null-score member, as is rL10. The other trained checkpoints' corpora are in data_card_r1.md, data_card_r3.md, "
           "data_card_r5.md (r4 and r5), data_card_r6.md and data_card_r7.md. Held-out calib and test splits are "
           "identical across all of them.", ""],
    "r7": ["# Data card: r7 corpus (null member of the shipped ensemble)", "",
           "Generated by `python -m openverdict.data.build --recipe r7 --out data/built_r7`. Counts are of records written.", "",
           "Differs from the r2 corpus (data_card.md) in three ways: 9000 synthetic_shapes records, swap_state at 0.05, "
           "and training answer-removed nulls filtered by option similarity (cosine >= 0.80, all-mpnet-base-v2). "
           "Held-out calib and test splits are identical to r2's.", ""],
    "r1": ["# Data card: r1 corpus (null member of the earlier ensemble E2)", "",
           "Generated by `python -m openverdict.data.build --recipe r1 --out data/built_r1`. Counts are of records written.", "",
           "r2's corpus (data_card.md) without the mask_random_nonwitness augmentation, which was added after r1. "
           "r1 was a null-score member of the ensemble E2 (data_card_E2.md), which E3 replaced. Held-out calib and test "
           "splits are identical to r2's.", ""],
    "r6": ["# Data card: r6 corpus (null member of the earlier ensemble E2)", "",
           "Generated by `python -m openverdict.data.build --recipe r6 --out data/built_r6`. Counts are of records written.", "",
           "r2's corpus (data_card.md) with training answer-removed nulls filtered by option similarity (cosine >= 0.80, "
           "all-mpnet-base-v2), as the r6 recipe set. r6 was a null-score member of the ensemble E2, which E3 replaced "
           "(data_card_E2.md). Held-out calib and test splits are identical to r2's.", ""],
    "r3": ["# Data card: r3 corpus (null member of the shipped ensemble)", "",
           "Generated by `python -m openverdict.data.build --recipe r3 --out data/built_r3`. Counts are of records written.", "",
           "r2's corpus (data_card.md) with remove_answer at 0.35 instead of 0.15. r3 is one of the null-score members of the lower-latency ensemble E1 (data_card_E1.md). Held-out calib and test splits are identical to r2's.", ""],
    "r5": ["# Data card: r5 corpus (null member of the shipped ensemble)", "",
           "Generated by `python -m openverdict.data.build --recipe r5 --out data/built_r5`. Counts are of records written.", "",
           "r4's corpus, which r5 trained on: r2's corpus with 9000 synthetic_shapes records, swap_state at 0.05, "
           "remove_answer at 0.25, and unfiltered training nulls. r5 and r4 trained on it; both are null-score members of the default ensemble E3 (data_card_E3.md). Held-out calib and test splits are identical to r2's.", ""],
}


def card(splits: Dict[str, List[Dict[str, Any]]], recipe: str = "r2") -> str:
    lines = list(CARD_HEAD[recipe])
    lines += ["## Per split", "", "| split | records | null rate | witness coverage |", "|---|---|---|---|"]
    for name, recs in splits.items():
        n = len(recs)
        nulls = sum(r["null"] for r in recs)
        wit = sum(r["witness"] is not None for r in recs)
        lines.append(f"| {name} | {n} | {nulls / max(n, 1):.3f} | {wit / max(n, 1):.3f} |")
    lines += ["", "## Per source and split", "", "| split | source | records |", "|---|---|---|"]
    for name, recs in splits.items():
        c = collections.Counter(r["source"] for r in recs)
        for src, k in sorted(c.items()):
            lines.append(f"| {name} | {src} | {k} |")
    lines += ["", "## Per question type and split", "", "| split | type | records |", "|---|---|---|"]
    for name, recs in splits.items():
        c = collections.Counter(r["question"]["type"] for r in recs)
        for t, k in sorted(c.items()):
            lines.append(f"| {name} | {t} | {k} |")
    lines += ["", "## Augmentations in train", "", "| augmentation | records |", "|---|---|"]
    c = collections.Counter()
    for r in splits.get("train", []):
        a = (r.get("meta") or {}).get("aug")
        for x in ([a] if isinstance(a, str) else (a or [])):
            c[x] += 1
    for a, k in sorted(c.items()):
        lines.append(f"| {a} | {k} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="tiny caps for a smoke build")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="r2")
    ap.add_argument("--card-only", action="store_true", help="rewrite the data card from an existing build")
    ap.add_argument("--out", default=None, help="output directory (default data/built); "
                    "a non-default directory writes its card there and skips the corrected unknowable set")
    args = ap.parse_args(argv)
    global N_SHAPES, FILTER_TRAIN_NULLS, OUT
    rc = RECIPES[args.recipe]
    N_SHAPES, FILTER_TRAIN_NULLS = rc["N_SHAPES"], rc["FILTER_TRAIN_NULLS"]
    AUG_P["swap_state"] = rc["swap_state"]
    AUG_P["remove_answer"] = rc.get("remove_answer", AUG_P["remove_answer"])
    AUG_P["mask_random_nonwitness"] = rc.get("mask_random_nonwitness", AUG_P["mask_random_nonwitness"])
    default_out = args.out is None
    if not default_out:
        OUT = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    if args.card_only:
        names = ["train", "dev", "calib", "test", "test_null", "test_nudge", "test_forge", "witness_test", "anli_test"]
        splits = {n: [json.loads(x) for x in (OUT / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for n in names}
        (card_path("data_card.md" if default_out else f"data_card_{args.recipe}.md")).write_text(card(splits, args.recipe), encoding="utf-8")
        return 0
    caps = {"train": 200, "validation": 50, "test": 50} if args.small else None
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    train, dev = [], []
    for name, fn in TRAIN_SOURCES.items():
        for r in fn(caps, seed=SEED):
            if name == "civil_comments" and h01(r["id"] + "score") < 0.33:
                r = synthetic_score.civil_to_score(r)
            if r["split"] == "train" or (name == "mmlu" and r["split"] == "test"):
                train.append(dict(r, split="train"))
            elif r["split"] == "validation":
                dev.append(r)
    witness_test = []
    for r in esnli.convert({"train": 0, "validation": 0, "test": 50 if args.small else 2000}, seed=SEED):
        if r["split"] == "test":
            witness_test.append(r)
    for r in squad2.convert({"train": 0, "validation": 0, "test": 50 if args.small else 2000}, seed=SEED):
        if r["split"] == "test" and not r["null"]:
            witness_test.append(r)

    syn = synthetic.generate(400 if args.small else N_SYNTHETIC, seed=SEED)
    held = []
    for r in syn:
        fam = r["meta"]["family"]
        if fam == HOLDOUT_FAMILY:
            r = dict(r, source=f"synthetic_policy_{fam}", split="test")
            held.append(r)
        elif h01(r["id"]) < 0.9:
            train.append(dict(r, split="train"))
        else:
            dev.append(dict(r, split="validation"))
    for r in (synthetic_shapes.generate(300 if args.small else N_SHAPES, seed=SEED) if N_SHAPES else []):
        (train if h01(r["id"]) < 0.9 else dev).append(dict(r, split="train" if h01(r["id"]) < 0.9 else "validation"))
    for r in synthetic_score.generate(200 if args.small else 4000, seed=SEED):
        (train if h01(r["id"]) < 0.9 else dev).append(dict(r, split="train" if h01(r["id"]) < 0.9 else "validation"))
    for name, fn in HOLDOUT_SOURCES.items():
        for r in fn(caps, seed=SEED):
            if r["split"] in ("test", "validation"):
                held.append(r)

    # Option subsampling: training draws a random size, eval uses a fixed size.
    train = [subsample_options(r, rng.randint(4, MAX_OPTIONS_TRAIN), rng) for r in train]
    dev = [subsample_options(r, EVAL_OPTIONS, random.Random(r["id"])) for r in dev]
    held = [subsample_options(r, EVAL_OPTIONS, random.Random(r["id"])) for r in held]

    # Drop held-out items whose normalized state also occurs in train or dev
    # (short generic utterances repeat across intent datasets).
    seen = {norm_text(r["state"]) for r in train + dev}
    n_held = len(held)
    held = [r for r in held if norm_text(r["state"]) not in seen]
    print(f"held-out items dropped as duplicates of train or dev states: {n_held - len(held)} of {n_held}")

    calib = [r for r in held if h01(r["id"]) < 0.5]
    test = [r for r in held if h01(r["id"]) >= 0.5]
    test_null = [x for x in (augment.remove_answer(r, random.Random(r["id"])) for r in test) if x]
    test_nudge = [x for x in (augment.plant_instruction(r, random.Random(r["id"] + "n"), kind="nudge") for r in test) if x]
    test_forge = [x for x in (augment.plant_instruction(r, random.Random(r["id"] + "f"), kind="forge") for r in test) if x]
    anli_test = [r for r in anli.convert({"train": 0, "validation": 0, "test": 50 if args.small else 2000}, seed=SEED)
                 if r["split"] == "test"]

    train = augment_train(train, rng)
    train, n_filtered = filter_answer_removed(train)
    print(f"training answer-removed nulls dropped by the similarity filter: {n_filtered}")
    rng.shuffle(train)

    splits = {
        "train": train, "dev": dev, "calib": calib, "test": test,
        "test_null": test_null, "test_nudge": test_nudge, "test_forge": test_forge,
        "witness_test": witness_test, "anli_test": anli_test,
    }
    manifest = {"small": args.small, "seed": SEED, "recipe": args.recipe, "holdout_sources": sorted(HOLDOUT_SOURCES) + [f"synthetic_policy_{HOLDOUT_FAMILY}"],
                "train_sources": sorted(TRAIN_SOURCES) + ["synthetic_policy", "synthetic_score", "synthetic_shapes"], "aug_p": AUG_P, "counts": {},
                "train_nulls_filtered": n_filtered, "train_answer_removed_kept": sum(1 for r in train if "remove_answer" in str((r.get("meta") or {}).get("aug")))}
    for name, recs in splits.items():
        manifest["counts"][name] = write_jsonl(OUT / f"{name}.jsonl", recs)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.small:
        (card_path("data_card.md" if default_out else f"data_card_{args.recipe}.md")).write_text(
            card(splits, args.recipe), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    if not args.small and default_out:
        from openverdict.eval import unknowable  # corrected unknowable set
        unknowable.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
