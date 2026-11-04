"""PAWS (labeled_final) converter.

Hub id: google-research-datasets/paws, config labeled_final (parquet files in the repo).
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterator, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from openverdict.data.schema import options_from_criteria, validate, whole_state_provenance

SOURCE = "paws"
LICENSE = "other"
HF_ID = "google-research-datasets/paws"
CONFIG = "labeled_final"
DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
INSTRUCTIONS = "Do the two sentences have the same meaning?"
CRITERIA = {
    "true": "The sentences are paraphrases with the same meaning.",
    "false": "The sentences differ in meaning.",
}
_FILES = {s: f"{CONFIG}/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")}


def row_to_record(row: Dict[str, Any], split: str, index: int, label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    state = f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}"
    criteria = dict(CRITERIA)
    return {
        "id": f"{SOURCE}-{split}-{index:06d}",
        "source": SOURCE,
        "split": split,
        "state": state,
        "provenance": whole_state_provenance(state, "user"),
        "question": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": criteria},
        "options": options_from_criteria("noul", criteria),
        "gold": "yes" if int(row["label"]) == 1 else "no",
        "null": False,
        "witness": None,
        "domain": "paraphrase",
        "meta": {"row_index": index, "paws_id": row.get("id")},
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
        cols = pq.read_table(path, columns=["id", "sentence1", "sentence2", "label"]).to_pydict()
        for i in _pick(list(range(len(cols["label"]))), caps.get(split), seed, split):
            row = {k: cols[k][i] for k in cols}
            rec = row_to_record(row, split, i)
            validate(rec)
            yield rec
