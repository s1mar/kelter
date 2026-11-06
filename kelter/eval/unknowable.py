"""Corrected unknowable set for null recall.

python -m kelter.eval.unknowable

Reads data/built/test.jsonl and writes data/built/test_unknowable_corrected.jsonl
plus data/built/test_unknowable_corrected.stats.json. Groups (meta.unk_group):
  answer_removed_filtered  gold option deleted; kept only if no remaining option
                           has cosine >= SIM_MAX to the deleted gold
                           ("name: description", all-mpnet-base-v2)
  swap_state               question and options kept, state taken from a
                           held-out test item of a different source (seeded)
  natural_null             CLINC150 out-of-scope items (null in the source)
Base ids are kept as the prefix before "~" so the harness can pair copies
with their answerable originals.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np

from kelter.data.augment import remove_answer
from kelter.data.schema import validate

ROOT = Path(__file__).resolve().parents[2]
BUILT = ROOT / "data" / "built"
SIM_MAX = 0.80
MODEL = "sentence-transformers/all-mpnet-base-v2"
SEED = 0


def opt_text(o):
    return f"{o['name'].replace('_', ' ')}: {o['description']}"


def build(test, embed):
    answerable = [r for r in test if not r["null"]]
    out, stats = [], {"answer_removed_candidates": 0, "answer_removed_kept": 0, "dropped_by_source": {}}
    texts = sorted({opt_text(o) for r in answerable for o in r["options"]})
    vec = dict(zip(texts, embed(texts)))
    for r in answerable:
        if r["question"]["type"] != "choice":
            continue
        rm = remove_answer(r, random.Random(r["id"]))
        if rm is None:
            continue
        stats["answer_removed_candidates"] += 1
        gold = next(o for o in r["options"] if o["name"] == r["gold"])
        g = vec[opt_text(gold)]
        sims = [float(np.dot(g, vec[opt_text(o)])) for o in rm["options"]]
        if max(sims) >= SIM_MAX:
            d = stats["dropped_by_source"]
            d[r["source"]] = d.get(r["source"], 0) + 1
            continue
        rm["meta"]["unk_group"] = "answer_removed_filtered"
        rm["meta"]["max_sim_to_removed_gold"] = round(max(sims), 4)
        out.append(rm)
        stats["answer_removed_kept"] += 1
    rng = random.Random(SEED)
    for r in answerable:
        donors = [d for d in answerable if d["source"] != r["source"]]
        d = donors[rng.randrange(len(donors))]
        x = copy.deepcopy(r)
        x["id"] = r["id"] + "~swapstate"
        x["state"], x["provenance"] = d["state"], copy.deepcopy(d["provenance"])
        x["null"], x["gold"], x["witness"] = True, None, None
        x.setdefault("meta", {})["unk_group"] = "swap_state"
        x["meta"]["donor"] = d["id"]
        out.append(x)
    for r in test:
        if r["null"]:
            x = copy.deepcopy(r)
            x.setdefault("meta", {})["unk_group"] = "natural_null"
            out.append(x)
    for x in out:
        validate(x)
    stats["counts"] = {}
    for x in out:
        k = f"{x['source']}/{x['meta']['unk_group']}"
        stats["counts"][k] = stats["counts"].get(k, 0) + 1
    return out, stats


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from kelter.device import default_device
    m = SentenceTransformer(MODEL, device=default_device())
    test = [json.loads(l) for l in (BUILT / "test.jsonl").open(encoding="utf-8")]
    out, stats = build(test, lambda t: m.encode(t, normalize_embeddings=True, convert_to_numpy=True, batch_size=128))
    stats.update({"sim_max": SIM_MAX, "model": MODEL, "seed": SEED})
    with (BUILT / "test_unknowable_corrected.jsonl").open("w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (BUILT / "test_unknowable_corrected.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (ROOT / "results" / "unknowable_corrected_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
