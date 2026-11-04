"""MMLU converter: seeded test subset plus the full dev split (as validation)."""
from __future__ import annotations

import random
from typing import Any, Dict, Iterator, Optional

from openverdict.data.schema import options_from_criteria, whole_state_provenance

SOURCE = "mmlu"
HF_ID = "cais/mmlu"
LICENSE = "Card Licensing Information: '[MIT License](https://github.com/hendrycks/test/blob/master/LICENSE)'; card metadata: mit."

DEFAULT_CAPS = {"validation": 2000, "test": 2000}
# unified split -> source split in config "all"
SOURCE_SPLITS = {"validation": "dev", "test": "test"}
INSTRUCTIONS = "Which option correctly answers the question?"
NAMES = ("A", "B", "C", "D")


def row_to_record(row: Dict[str, Any], split: str, index: int) -> Optional[Dict[str, Any]]:
    """Pure mapping of one MMLU row. Needs exactly four choices and an answer in 0..3."""
    choices = [str(c) for c in list(row["choices"])]
    if len(choices) != len(NAMES):
        return None
    ans = int(row["answer"])
    if not 0 <= ans < len(NAMES):
        return None
    question = str(row["question"]).strip()
    if not question:
        return None
    criteria = dict(zip(NAMES, choices))
    return {
        "id": f"{SOURCE}-{split}-{index:06d}",
        "source": SOURCE,
        "split": split,
        "state": question,
        "provenance": whole_state_provenance(question, "user"),
        "question": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria},
        "options": options_from_criteria("choice", criteria),
        "gold": NAMES[ans],
        "null": False,
        "witness": None,
        "domain": "mcq",
        "meta": {"subject": row.get("subject")},
    }


def convert(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    caps = {**DEFAULT_CAPS, **(max_per_split or {})}
    for split, src in SOURCE_SPLITS.items():
        path = hf_hub_download(HF_ID, f"all/{src}-00000-of-00001.parquet", repo_type="dataset")
        rows = pd.read_parquet(path).to_dict("records")
        valid = [i for i, r in enumerate(rows) if row_to_record(r, split, i) is not None]
        cap = caps.get(split)
        if cap is not None and len(valid) > cap:
            valid = sorted(random.Random(f"{seed}:{SOURCE}:{split}").sample(valid, cap))
        for i in valid:
            yield row_to_record(rows[i], split, i)
