"""ANLI converter: three rounds pooled per split, round kept in meta."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from openverdict.data.converters.mnli import label_name_from_int, nli_record, sample_indices

SOURCE = "anli"
HF_ID = "facebook/anli"
LICENSE = (
    "Card metadata: cc-by-nc-4.0. Licensing Information section: "
    "'cc-4 Attribution-NonCommercial' (linking github.com/facebookresearch/anli LICENSE)"
)

DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}

# unified split -> source file prefix
SPLIT_PREFIX = {"train": "train", "validation": "dev", "test": "test"}
ROUNDS = (1, 2, 3)


def row_to_record(row: Dict[str, Any], split: str, index: int, round_no: int) -> Optional[Dict[str, Any]]:
    """Pure mapping of one ANLI row. index is the position in the pooled split."""
    name = label_name_from_int(row["label"])
    if name is None:
        return None
    premise = str(row["premise"]).strip()
    hypothesis = str(row["hypothesis"]).strip()
    if not premise or not hypothesis:
        return None
    meta = {"round": round_no, "uid": row.get("uid")}
    return nli_record(SOURCE, split, index, premise, hypothesis, name, None, meta)


def convert(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    caps = {**DEFAULT_CAPS, **(max_per_split or {})}
    for split, prefix in SPLIT_PREFIX.items():
        rows: List[Tuple[int, Dict[str, Any]]] = []
        for r in ROUNDS:
            path = hf_hub_download(HF_ID, f"plain_text/{prefix}_r{r}-00000-of-00001.parquet", repo_type="dataset")
            df = pd.read_parquet(path)
            rows.extend((r, d) for d in df.to_dict("records"))
        valid = [i for i, (_, d) in enumerate(rows) if label_name_from_int(d["label"]) is not None]
        for i in sample_indices(valid, caps.get(split), seed, f"{SOURCE}:{split}"):
            r, d = rows[i]
            rec = row_to_record(d, split, i, r)
            if rec is not None:
                yield rec
