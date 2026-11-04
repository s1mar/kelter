"""MASSIVE English (en-US) intent converter.

Hub id: AmazonScience/massive, read from its auto-generated parquet branch
(revision "refs/convert/parquet", config en-US), because the main branch only ships a
loading script. The 60 intents are the options.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from openverdict.data.schema import options_from_criteria, validate, whole_state_provenance

SOURCE = "massive"
LICENSE = "cc-by-4.0"
HF_ID = "AmazonScience/massive"
REVISION = "refs/convert/parquet"
CONFIG = "en-US"
DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
INSTRUCTIONS = "Which intent does the user's request express?"
_FILES = {s: f"{CONFIG}/{s}/0000.parquet" for s in ("train", "validation", "test")}


def humanize(name: str) -> str:
    s = name.replace("_", " ")
    return s[:1].upper() + s[1:]


def row_to_record(row: Dict[str, Any], split: str, index: int, label_names: List[str]) -> Dict[str, Any]:
    state = row["utt"]
    label = label_names[row["intent"]]
    criteria = {n: humanize(n) for n in label_names}
    return {
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
        "meta": {"row_index": index, "label_id": row["intent"], "locale": CONFIG},
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
    paths = {s: hf_hub_download(HF_ID, f, repo_type="dataset", revision=REVISION) for s, f in _FILES.items()}
    names = _label_names(paths["train"], "intent")
    for split in ("train", "validation", "test"):
        cols = pq.read_table(paths[split], columns=["utt", "intent"]).to_pydict()
        for i in _pick(list(range(len(cols["utt"]))), caps.get(split), seed, split):
            rec = row_to_record({"utt": cols["utt"][i], "intent": cols["intent"][i]}, split, i, names)
            validate(rec)
            yield rec
