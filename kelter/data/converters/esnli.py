"""e-SNLI converter with human highlight rationales (annotator 1) as the witness.

The Hub copy of e-SNLI is a loading script without highlights, so the original CSVs
are downloaded from the authors' GitHub repo into data/cache/esnli/.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from kelter.data.converters.mnli import (
    HYPOTHESIS_PREFIX,
    PREMISE_PREFIX,
    nli_record,
    sample_indices,
)

SOURCE = "esnli"
HF_ID = "https://github.com/OanaMariaCamburu/e-SNLI"
LICENSE = (
    "Repo LICENSE file (MIT terms, no license name in the text): 'Copyright <2019> <Oana-Maria "
    "Camburu> Permission is hereby granted, free of charge, to any person obtaining a copy of this "
    "software ... to deal in the Software without restriction'. Underlying SNLI sentences are "
    "CC BY-SA 4.0 per the stanfordnlp/snli Hub card."
)

RAW_BASE = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/"
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "esnli"
# unified split -> CSV file. Train uses only the first of the two train CSVs.
SPLIT_FILES = {
    "train": "esnli_train_1.csv",
    "validation": "esnli_dev.csv",
    "test": "esnli_test.csv",
}
DEFAULT_CAPS = {"train": 20000, "validation": 2000, "test": 2000}
LABELS = ("entailment", "neutral", "contradiction")

_WORD = re.compile(r"\S+")


def highlight_spans(text: str, marked: Any) -> List[List[int]]:
    """Character spans in text of the words highlighted in marked (asterisk markup).

    Words are whitespace-delimited. Marked and plain text must tokenize to the same words
    once asterisks are removed, otherwise no spans are returned. Adjacent highlighted
    words merge into one span running from the first word's start to the last word's end.
    """
    if not isinstance(marked, str) or "*" not in marked:
        return []
    words = [(m.start(), m.end(), m.group()) for m in _WORD.finditer(text)]
    toks = marked.split()
    if len(toks) != len(words):
        return []
    flags: List[bool] = []
    inside = False
    for tok, (_, _, word) in zip(toks, words):
        starts = tok.startswith("*")
        ends = tok.endswith("*") and (len(tok) > 1 or inside)
        if tok.replace("*", "") != word:
            return []
        hl = inside or starts
        flags.append(hl)
        if starts and not ends:
            inside = True
        elif ends:
            inside = False
    spans: List[List[int]] = []
    cur: Optional[List[int]] = None
    for flag, (s, e, _) in zip(flags, words):
        if flag:
            if cur is None:
                cur = [s, e]
            else:
                cur[1] = e
        elif cur is not None:
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append(cur)
    return spans


def row_to_record(row: Dict[str, Any], split: str, index: int) -> Optional[Dict[str, Any]]:
    """Pure mapping of one e-SNLI CSV row. Witness is annotator 1's highlights or None."""
    label = row.get("gold_label")
    if label not in LABELS:
        return None
    premise = str(row["Sentence1"]).strip()
    hypothesis = str(row["Sentence2"]).strip()
    if not premise or not hypothesis:
        return None
    p_off = len(PREMISE_PREFIX)
    h_off = len(PREMISE_PREFIX) + len(premise) + len(HYPOTHESIS_PREFIX)
    witness = [[s + p_off, e + p_off] for s, e in highlight_spans(premise, row.get("Sentence1_marked_1"))]
    witness += [[s + h_off, e + h_off] for s, e in highlight_spans(hypothesis, row.get("Sentence2_marked_1"))]
    meta = {"pairID": row.get("pairID")}
    return nli_record(SOURCE, split, index, premise, hypothesis, label, witness or None, meta)


def ensure_csv(name: str) -> Path:
    """Download one CSV into the dedicated cache folder if it is not there yet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(RAW_BASE + name, timeout=120) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    return dest


def convert(max_per_split: Optional[dict] = None, seed: int = 0) -> Iterator[Dict[str, Any]]:
    import pandas as pd

    caps = {**DEFAULT_CAPS, **(max_per_split or {})}
    cols = ["pairID", "gold_label", "Sentence1", "Sentence2", "Sentence1_marked_1", "Sentence2_marked_1"]
    for split, fname in SPLIT_FILES.items():
        df = pd.read_csv(ensure_csv(fname), usecols=cols, keep_default_na=False, dtype=str)
        valid = [i for i, lab in enumerate(df["gold_label"].tolist()) if lab in LABELS]
        for i in sample_indices(valid, caps.get(split), seed, f"{SOURCE}:{split}"):
            rec = row_to_record(df.iloc[i].to_dict(), split, i)
            if rec is not None:
                yield rec
