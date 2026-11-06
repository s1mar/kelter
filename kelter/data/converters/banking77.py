"""BANKING77 converter.

Hub id: legacy-datasets/banking77 (parquet-backed copy of PolyAI/banking77, whose own
repo only ships a loading script). The source has train and test splits only, so a
validation split is carved deterministically from train (10 percent, seeded).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from kelter.data.schema import options_from_criteria, validate, whole_state_provenance

SOURCE = "banking77"
LICENSE = "cc-by-4.0"
HF_ID = "legacy-datasets/banking77"
DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
VAL_FRACTION = 0.1
INSTRUCTIONS = "Which banking intent does the customer's message express?"
_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def humanize(name: str) -> str:
    s = name.replace("_", " ")
    return s[:1].upper() + s[1:]


def row_to_record(row: Dict[str, Any], split: str, index: int, label_names: List[str]) -> Dict[str, Any]:
    state = row["text"]
    label = label_names[row["label"]]
    criteria = {n: humanize(n) for n in label_names}
    rec = {
        "id": f"{SOURCE}-{split}-{index:06d}",
        "source": SOURCE,
        "split": split,
        "state": state,
        "provenance": whole_state_provenance(state, "user"),
        "question": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria},
        "options": options_from_criteria("choice", criteria),
        "gold": label,
        "null": False,
        "witness": None,
        "domain": "intent",
        "meta": {"row_index": index, "label_id": row["label"]},
    }
    return rec


def _label_names(path: str, column: str) -> List[str]:
    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    feats = json.loads(md[b"huggingface"])["info"]["features"]
    return list(feats[column]["names"])


def _pick(indices: List[int], cap: Optional[int], seed: int, split: str) -> List[int]:
    if cap is None or len(indices) <= cap:
        return list(indices)
    rng = random.Random(f"{seed}:{SOURCE}:{split}")
    return sorted(rng.sample(list(indices), cap))


def convert(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    caps = dict(DEFAULT_CAPS)
    if max_per_split:
        caps.update(max_per_split)
    paths = {s: hf_hub_download(HF_ID, f, repo_type="dataset") for s, f in _FILES.items()}
    names = _label_names(paths["train"], "label")
    tables = {s: pq.read_table(p, columns=["text", "label"]).to_pydict() for s, p in paths.items()}

    n_train = len(tables["train"]["text"])
    perm = random.Random(f"{seed}:{SOURCE}:carve").sample(range(n_train), n_train)
    val_set = set(perm[: int(n_train * VAL_FRACTION)])
    eligible = {
        "train": [i for i in range(n_train) if i not in val_set],
        "validation": sorted(val_set),
        "test": list(range(len(tables["test"]["text"]))),
    }
    source_table = {"train": tables["train"], "validation": tables["train"], "test": tables["test"]}

    for split in ("train", "validation", "test"):
        cols = source_table[split]
        for i in _pick(eligible[split], caps.get(split), seed, split):
            rec = row_to_record({"text": cols["text"][i], "label": cols["label"][i]}, split, i, names)
            validate(rec)
            yield rec
