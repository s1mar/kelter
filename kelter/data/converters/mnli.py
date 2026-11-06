"""MultiNLI converter. Also hosts the NLI record shape shared by anli and esnli."""
from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional

from kelter.data.schema import options_from_criteria, whole_state_provenance

SOURCE = "mnli"
HF_ID = "nyu-mll/multi_nli"
LICENSE = (
    "Card lists license: cc-by-3.0, cc-by-sa-3.0, mit, other. Licensing Information: "
    "'The majority of the corpus is released under the OANC's license, which allows all "
    "content to be freely used, modified, and shared under permissive terms.' (FICTION "
    "section under several permissive licenses)"
)

DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}

# source split name -> unified split name
SPLIT_FILES = {
    "train": ("train", "data/train-00000-of-00001.parquet"),
    "validation_matched": ("validation", "data/validation_matched-00000-of-00001.parquet"),
    "validation_mismatched": ("test", "data/validation_mismatched-00000-of-00001.parquet"),
}

NLI_INSTRUCTIONS = "Does the premise entail the hypothesis, contradict it, or neither?"
NLI_CRITERIA = {
    "entailment": "The premise entails the hypothesis.",
    "neutral": "The premise neither entails nor contradicts the hypothesis.",
    "contradiction": "The premise contradicts the hypothesis.",
}
NLI_LABELS = ("entailment", "neutral", "contradiction")  # HF int label order
PREMISE_PREFIX = "Premise: "
HYPOTHESIS_PREFIX = "\nHypothesis: "


def nli_state(premise: str, hypothesis: str) -> str:
    return f"{PREMISE_PREFIX}{premise}{HYPOTHESIS_PREFIX}{hypothesis}"


def nli_record(
    source: str,
    split: str,
    index: int,
    premise: str,
    hypothesis: str,
    label_name: str,
    witness: Optional[List[List[int]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = nli_state(premise, hypothesis)
    criteria = dict(NLI_CRITERIA)
    return {
        "id": f"{source}-{split}-{index:06d}",
        "source": source,
        "split": split,
        "state": state,
        "provenance": whole_state_provenance(state, "user"),
        "question": {"type": "choice", "instructions": NLI_INSTRUCTIONS, "criteria": criteria},
        "options": options_from_criteria("choice", criteria),
        "gold": label_name,
        "null": False,
        "witness": witness,
        "domain": "nli",
        "meta": dict(meta or {}),
    }


def label_name_from_int(label: Any) -> Optional[str]:
    """Map an HF int label to its name; None for -1 or anything unknown."""
    try:
        i = int(label)
    except (TypeError, ValueError):
        return None
    if 0 <= i < len(NLI_LABELS):
        return NLI_LABELS[i]
    return None


def row_to_record(row: Dict[str, Any], split: str, index: int) -> Optional[Dict[str, Any]]:
    """Pure mapping of one MultiNLI row. Returns None for rows with label -1."""
    name = label_name_from_int(row["label"])
    if name is None:
        return None
    premise = str(row["premise"]).strip()
    hypothesis = str(row["hypothesis"]).strip()
    if not premise or not hypothesis:
        return None
    meta = {"genre": row.get("genre"), "pairID": row.get("pairID")}
    return nli_record(SOURCE, split, index, premise, hypothesis, name, None, meta)


def sample_indices(valid: List[int], cap: Optional[int], seed: int, salt: str) -> List[int]:
    """Deterministic seeded subset of valid, returned in ascending order."""
    if cap is None or len(valid) <= cap:
        return list(valid)
    rng = random.Random(f"{seed}:{salt}")
    return sorted(rng.sample(valid, cap))


def convert(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    caps = {**DEFAULT_CAPS, **(max_per_split or {})}
    for _src_split, (split, fname) in SPLIT_FILES.items():
        path = hf_hub_download(HF_ID, fname, repo_type="dataset")
        df = pd.read_parquet(path, columns=["premise", "hypothesis", "genre", "pairID", "label"])
        valid = [i for i, lab in enumerate(df["label"].tolist()) if label_name_from_int(lab) is not None]
        for i in sample_indices(valid, caps.get(split), seed, f"{SOURCE}:{split}"):
            rec = row_to_record(df.iloc[i].to_dict(), split, i)
            if rec is not None:
                yield rec
