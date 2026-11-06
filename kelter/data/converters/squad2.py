"""SQuAD 2.0 converter as 4-way multiple choice over answers borrowed from sibling questions."""
from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

from kelter.data.schema import options_from_criteria, whole_state_provenance

SOURCE = "squad2"
HF_ID = "rajpurkar/squad_v2"
LICENSE = "Card metadata: cc-by-sa-4.0. Licensing Information: 'The dataset is distributed under the CC BY-SA 4.0 license.'"

DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
NAMES = ("A", "B", "C", "D")
N_OPTIONS = 4


def _texts(row: Dict[str, Any]) -> List[str]:
    ans = row.get("answers") or {}
    texts = ans.get("text", []) if hasattr(ans, "get") else []
    return [str(t) for t in list(texts)]


def other_answer_texts(rows: List[Dict[str, Any]], k: int) -> List[str]:
    """Distinct (case-insensitive) answer texts of the other questions in rows, minus
    anything equal to a gold text of question k. Sorted for determinism."""
    banned = {t.strip().casefold() for t in _texts(rows[k])}
    seen = set(banned)
    out: List[str] = []
    for j, other in enumerate(rows):
        if j == k:
            continue
        for t in _texts(other):
            key = t.strip().casefold()
            if t.strip() and key not in seen:
                seen.add(key)
                out.append(t)
    return sorted(out)


def is_eligible(row: Dict[str, Any], others: List[str]) -> bool:
    need = N_OPTIONS - 1 if _texts(row) else N_OPTIONS
    return len(others) >= need


def row_to_record(
    row: Dict[str, Any], split: str, index: int, others: List[str], seed: int = 0
) -> Optional[Dict[str, Any]]:
    """Pure mapping of one SQuAD 2.0 row given the candidate texts from sibling questions.

    Returns None when there are not enough distinct other answers.
    """
    gold_texts = _texts(row)
    answerable = bool(gold_texts)
    need = N_OPTIONS - 1 if answerable else N_OPTIONS
    if len(others) < need:
        return None
    rng = random.Random(f"{seed}:{SOURCE}:{split}:{index}")
    picked = rng.sample(list(others), need)
    context = str(row["context"])
    gold_text = None
    start = None
    if answerable:
        gold_text = gold_texts[0]
        starts = list(row["answers"]["answer_start"])
        start = int(starts[0])
        picked.append(gold_text)
    rng.shuffle(picked)
    criteria = {NAMES[i]: t for i, t in enumerate(picked)}
    gold_name = None
    witness = None
    if answerable:
        gold_name = next(n for n, t in criteria.items() if t == gold_text)
        if context[start : start + len(gold_text)] == gold_text and gold_text:
            witness = [[start, start + len(gold_text)]]
    return {
        "id": f"{SOURCE}-{split}-{index:06d}",
        "source": SOURCE,
        "split": split,
        "state": context,
        "provenance": whole_state_provenance(context, "retrieved_doc"),
        "question": {"type": "choice", "instructions": str(row["question"]), "criteria": criteria},
        "options": options_from_criteria("choice", criteria),
        "gold": gold_name,
        "null": not answerable,
        "witness": witness,
        "domain": "qa",
        "meta": {"squad_id": row.get("id"), "title": row.get("title"), "answerable": answerable},
    }


def split_contexts(contexts: List[str], seed: int) -> Dict[str, str]:
    """Deterministically assign each distinct context to validation or test (half each)."""
    uniq = sorted(set(contexts))
    random.Random(f"{seed}:{SOURCE}:ctxsplit").shuffle(uniq)
    half = len(uniq) // 2
    return {c: ("validation" if i < half else "test") for i, c in enumerate(uniq)}


def _eligible_indices(rows: List[Dict[str, Any]], pick) -> List[tuple]:
    """(index, others) for every eligible row whose index passes pick."""
    by_ctx: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_ctx.setdefault(r["context"], []).append(i)
    out = []
    for idxs in by_ctx.values():
        group = [rows[i] for i in idxs]
        for k, i in enumerate(idxs):
            if not pick(i, rows[i]):
                continue
            others = other_answer_texts(group, k)
            if is_eligible(rows[i], others):
                out.append((i, others))
    out.sort(key=lambda t: t[0])
    return out


def _cap(items: List[tuple], cap: Optional[int], seed: int, salt: str) -> List[tuple]:
    if cap is None or len(items) <= cap:
        return items
    keep = set(random.Random(f"{seed}:{salt}").sample(range(len(items)), cap))
    return [it for j, it in enumerate(items) if j in keep]


def convert(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    caps = {**DEFAULT_CAPS, **(max_per_split or {})}

    def load(name: str) -> List[Dict[str, Any]]:
        path = hf_hub_download(HF_ID, f"squad_v2/{name}-00000-of-00001.parquet", repo_type="dataset")
        return pd.read_parquet(path).to_dict("records")

    train = load("train")
    items = _cap(_eligible_indices(train, lambda i, r: True), caps.get("train"), seed, f"{SOURCE}:train")
    for i, others in items:
        rec = row_to_record(train[i], "train", i, others, seed)
        if rec is not None:
            yield rec

    val = load("validation")
    assign = split_contexts([r["context"] for r in val], seed)
    for split in ("validation", "test"):
        items = _eligible_indices(val, lambda i, r, s=split: assign[r["context"]] == s)
        items = _cap(items, caps.get(split), seed, f"{SOURCE}:{split}")
        for i, others in items:
            rec = row_to_record(val[i], split, i, others, seed)
            if rec is not None:
                yield rec
