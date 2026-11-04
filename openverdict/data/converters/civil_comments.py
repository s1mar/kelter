"""Civil Comments converter.

Hub id: google/civil_comments (parquet files in the repo). Gold is a distribution over
yes/no built from the toxicity fraction. Empty comments are skipped before sampling.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterator, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from openverdict.data.schema import options_from_criteria, validate, whole_state_provenance

SOURCE = "civil_comments"
LICENSE = "cc0-1.0"
HF_ID = "google/civil_comments"
DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
INSTRUCTIONS = "Is this comment toxic?"
CRITERIA = {
    "true": "The comment is rude, disrespectful or unreasonable.",
    "false": "The comment is civil.",
}
_FILES = {
    "train": [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
    ],
    "validation": ["data/validation-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"],
}


def toxicity_gold(t: float) -> Dict[str, float]:
    """Return {"yes": t, "no": 1 - t}, 4 decimals, summing to exactly 1.0 in floats."""
    t = min(max(float(t), 0.0), 1.0)
    y = round(round(t, 4) * 10000)
    yes = y / 10000
    no = (10000 - y) / 10000
    if yes + no != 1.0:
        no = 1.0 - yes
    return {"yes": yes, "no": no}


def is_usable(row: Dict[str, Any]) -> bool:
    text = row.get("text")
    tox = row.get("toxicity")
    if not isinstance(text, str) or not text.strip():
        return False
    return tox is not None and tox == tox


def row_to_record(row: Dict[str, Any], split: str, index: int, label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    state = row["text"]
    criteria = dict(CRITERIA)
    return {
        "id": f"{SOURCE}-{split}-{index:06d}",
        "source": SOURCE,
        "split": split,
        "state": state,
        "provenance": whole_state_provenance(state, "user"),
        "question": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": criteria},
        "options": options_from_criteria("noul", criteria),
        "gold": toxicity_gold(row["toxicity"]),
        "null": False,
        "witness": None,
        "domain": "toxicity",
        "meta": {"row_index": index, "toxicity": float(row["toxicity"])},
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
    for split, fnames in _FILES.items():
        parts = [
            pq.read_table(hf_hub_download(HF_ID, f, repo_type="dataset"), columns=["text", "toxicity"])
            for f in fnames
        ]
        table = pa.concat_tables(parts) if len(parts) > 1 else parts[0]
        text = pc.fill_null(table["text"], "")
        tox = pc.fill_null(table["toxicity"], float("nan"))
        non_empty = pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(text)), 0)
        mask = pc.and_(non_empty, pc.invert(pc.is_nan(tox)))
        eligible = pc.indices_nonzero(mask).to_pylist()
        for i in _pick(eligible, caps.get(split), seed, split):
            row = {"text": text[i].as_py(), "toxicity": tox[i].as_py()}
            rec = row_to_record(row, split, i)
            validate(rec)
            yield rec
