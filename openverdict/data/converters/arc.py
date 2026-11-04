"""ARC (Easy and Challenge) converter. One module, two sources."""
from __future__ import annotations

import random
from typing import Any, Dict, Iterator, Optional

from openverdict.data.schema import options_from_criteria, whole_state_provenance

SOURCE = "arc"  # family name; the emitted sources are in SOURCES
SOURCE_EASY = "arc_easy"
SOURCE_CHALLENGE = "arc_challenge"
SOURCES = (SOURCE_EASY, SOURCE_CHALLENGE)
HF_ID = "allenai/ai2_arc"
LICENSE = (
    "Card metadata: cc-by-sa-4.0. The card's Licensing Information section is a placeholder "
    "('More Information Needed')."
)

DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
CONFIGS = {SOURCE_EASY: "ARC-Easy", SOURCE_CHALLENGE: "ARC-Challenge"}
INSTRUCTIONS = "Which option correctly answers the question?"
_LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def normalize_label(label: Any) -> str:
    s = str(label).strip()
    return _LABEL_MAP.get(s, s.upper())


def row_to_record(row: Dict[str, Any], source: str, split: str, index: int) -> Optional[Dict[str, Any]]:
    """Pure mapping of one ARC row. None when the answer key is not among the choices."""
    labels = [normalize_label(x) for x in list(row["choices"]["label"])]
    texts = [str(x) for x in list(row["choices"]["text"])]
    if len(labels) != len(texts) or len(set(labels)) != len(labels) or len(labels) < 2:
        return None
    gold = normalize_label(row["answerKey"])
    if gold not in labels:
        return None
    question = str(row["question"]).strip()
    if not question:
        return None
    criteria = dict(zip(labels, texts))
    return {
        "id": f"{source}-{split}-{index:06d}",
        "source": source,
        "split": split,
        "state": question,
        "provenance": whole_state_provenance(question, "user"),
        "question": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria},
        "options": options_from_criteria("choice", criteria),
        "gold": gold,
        "null": False,
        "witness": None,
        "domain": "mcq",
        "meta": {"arc_id": row.get("id"), "original_labels": [str(x) for x in list(row["choices"]["label"])]},
    }


def _convert(source: str, max_per_split: Optional[dict], seed: int) -> Iterator[Dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    caps = {**DEFAULT_CAPS, **(max_per_split or {})}
    cfg = CONFIGS[source]
    for split in ("train", "validation", "test"):
        path = hf_hub_download(HF_ID, f"{cfg}/{split}-00000-of-00001.parquet", repo_type="dataset")
        rows = pd.read_parquet(path).to_dict("records")
        valid = [i for i, r in enumerate(rows) if row_to_record(r, source, split, i) is not None]
        cap = caps.get(split)
        if cap is not None and len(valid) > cap:
            valid = sorted(random.Random(f"{seed}:{source}:{split}").sample(valid, cap))
        for i in valid:
            yield row_to_record(rows[i], source, split, i)


def convert_easy(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    return _convert(SOURCE_EASY, max_per_split, seed)


def convert_challenge(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    return _convert(SOURCE_CHALLENGE, max_per_split, seed)


def convert(
    max_per_split: Optional[dict] = None, seed: int = 0, config: str = "arc_easy"
) -> Iterator[Dict[str, Any]]:
    """Dispatch on config: arc_easy / easy / ARC-Easy or arc_challenge / challenge / ARC-Challenge."""
    key = str(config).lower().replace("-", "_")
    if key in ("arc_easy", "easy"):
        return convert_easy(max_per_split, seed)
    if key in ("arc_challenge", "challenge"):
        return convert_challenge(max_per_split, seed)
    raise ValueError(f"unknown ARC config {config!r}")
