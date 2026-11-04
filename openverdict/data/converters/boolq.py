"""BoolQ converter.

Hub id: google/boolq (parquet files in the repo). BoolQ has no test labels, so only
train and validation are emitted.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterator, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from openverdict.data.schema import options_from_criteria, validate, whole_state_provenance

SOURCE = "boolq"
LICENSE = "cc-by-sa-3.0"
HF_ID = "google/boolq"
DEFAULT_CAPS = {"train": 20000, "validation": 2000}
CRITERIA = {
    "true": "The passage supports answering yes.",
    "false": "The passage supports answering no.",
}
_FILES = {s: f"data/{s}-00000-of-00001.parquet" for s in ("train", "validation")}


def question_to_instructions(question: str) -> str:
    q = question.strip()
    q = q[:1].upper() + q[1:]
    if not q.endswith("?"):
        q += "?"
    return q


def row_to_record(row: Dict[str, Any], split: str, index: int, label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    state = row["passage"]
    criteria = dict(CRITERIA)
    return {
        "id": f"{SOURCE}-{split}-{index:06d}",
        "source": SOURCE,
        "split": split,
        "state": state,
        "provenance": whole_state_provenance(state, "retrieved_doc"),
        "question": {
            "type": "noul",
            "instructions": question_to_instructions(row["question"]),
            "criteria": criteria,
        },
        "options": options_from_criteria("noul", criteria),
        "gold": "yes" if row["answer"] else "no",
        "null": False,
        "witness": None,
        "domain": "qa",
        "meta": {"row_index": index},
    }


def _pick(indices: List[int], cap: Optional[int], seed: int, split: str) -> List[int]:
    if cap is None or len(indices) <= cap:
        return list(indices)
    rng = random.Random(f"{seed}:{SOURCE}:{split}")
    return sorted(rng.sample(list(indices), cap))


def convert(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    caps = dict(DEFAULT_CAPS)
    if max_per_split:
        caps.update(max_per_split)
    for split, fname in _FILES.items():
        path = hf_hub_download(HF_ID, fname, repo_type="dataset")
        cols = pq.read_table(path, columns=["question", "answer", "passage"]).to_pydict()
        for i in _pick(list(range(len(cols["answer"]))), caps.get(split), seed, split):
            rec = row_to_record({k: cols[k][i] for k in cols}, split, i)
            validate(rec)
            yield rec
