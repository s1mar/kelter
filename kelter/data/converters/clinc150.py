"""CLINC150 converter ("plus" config).

Hub id: clinc/clinc_oos (parquet files in the repo, config "plus"). The 150 in-scope
intents are the options. The out-of-scope class ("oos") is not an option: oos rows
become null records with gold None.
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

SOURCE = "clinc150"
LICENSE = "cc-by-3.0"
HF_ID = "clinc/clinc_oos"
CONFIG = "plus"
OOS_NAME = "oos"
DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
INSTRUCTIONS = "Which intent does the user's request express?"
_FILES = {s: f"{CONFIG}/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")}


def humanize(name: str) -> str:
    s = name.replace("_", " ")
    return s[:1].upper() + s[1:]


def row_to_record(row: Dict[str, Any], split: str, index: int, label_names: List[str]) -> Dict[str, Any]:
    state = row["text"]
    label = label_names[row["intent"]]
    criteria = {n: humanize(n) for n in label_names if n != OOS_NAME}
    is_oos = label == OOS_NAME
    return {
        "id": f"{SOURCE}-{split}-{index:06d}",
        "source": SOURCE,
        "split": split,
        "state": state,
        "provenance": whole_state_provenance(state, "user"),
        "question": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria},
        "options": options_from_criteria("choice", criteria),
        "gold": None if is_oos else label,
        "null": is_oos,
        "witness": None,
        "domain": "intent",
        "meta": {"row_index": index, "label_id": row["intent"], "oos": is_oos},
    }


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
    names = _label_names(paths["train"], "intent")
    for split in ("train", "validation", "test"):
        cols = pq.read_table(paths[split], columns=["text", "intent"]).to_pydict()
        for i in _pick(list(range(len(cols["text"]))), caps.get(split), seed, split):
            rec = row_to_record({"text": cols["text"][i], "intent": cols["intent"][i]}, split, i, names)
            validate(rec)
            yield rec
