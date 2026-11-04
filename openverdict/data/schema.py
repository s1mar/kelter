"""Unified training record and its validator.

Every converter emits records of this shape, one JSON object per line:

{
  "id": "banking77-train-000123",           unique across the corpus
  "source": "banking77",                    dataset name; the holdout unit
  "split": "train" | "validation" | "test", the source dataset's own split
  "state": "text the decision is about",
  "provenance": [[start, end, type], ...],  char spans covering state exactly,
                                            type in PROVENANCE_TYPES
  "question": {"type": "choice" | "score" | "noul",
               "instructions": "...",
               "criteria": ...},            TypeSafe shape: choice -> {name: description},
                                            noul -> {"true": ..., "false": ...},
                                            score -> [level0 text, level1 text, ...]
  "options": [{"name": ..., "description": ...}, ...],
                                            derived from criteria; noul -> names "no","yes";
                                            score -> names "0","1",...
  "gold": "option name" | {name: prob} | null,   null iff null is True
  "null": false,                            True when no option fits (answer removed)
  "witness": [[start, end], ...] | null,    char spans in state that justify gold
  "domain": "intent" | "nli" | "qa" | "mcq" | "paraphrase" | "toxicity" | "policy",
  "meta": {...}                             free-form, never read by the model
}

Option names are what the API returns; descriptions are what the model reads.
"""
from __future__ import annotations

from typing import Any, Dict, List

PROVENANCE_TYPES = ("user", "tool_output", "retrieved_doc", "system", "unknown")
QUESTION_TYPES = ("choice", "score", "noul")
DOMAINS = ("intent", "nli", "qa", "mcq", "paraphrase", "toxicity", "policy")
SPLITS = ("train", "validation", "test")


def options_from_criteria(qtype: str, criteria: Any) -> List[Dict[str, str]]:
    """Derive the option list from TypeSafe-shaped criteria."""
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a non-empty dict")
        return [{"name": str(k), "description": str(v)} for k, v in criteria.items()]
    if qtype == "noul":
        if not isinstance(criteria, dict):
            raise ValueError("noul criteria must be a dict with true/false")
        return [
            {"name": "no", "description": str(criteria.get("false", "no"))},
            {"name": "yes", "description": str(criteria.get("true", "yes"))},
        ]
    if qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must be a list of at least 2 levels")
        return [{"name": str(i), "description": str(t)} for i, t in enumerate(criteria)]
    raise ValueError(f"unknown question type {qtype}")


def whole_state_provenance(state: str, ptype: str) -> List[List[Any]]:
    return [[0, len(state), ptype]]


def validate(rec: Dict[str, Any]) -> None:
    """Raise ValueError describing the first problem found."""
    for k in ("id", "source", "split", "state", "provenance", "question", "options", "gold", "null", "witness", "domain"):
        if k not in rec:
            raise ValueError(f"missing key {k}")
    if rec["split"] not in SPLITS:
        raise ValueError(f"bad split {rec['split']}")
    if rec["domain"] not in DOMAINS:
        raise ValueError(f"bad domain {rec['domain']}")
    state = rec["state"]
    if not isinstance(state, str) or not state.strip():
        raise ValueError("empty state")
    # provenance covers state exactly, in order, no gaps or overlaps
    pos = 0
    for seg in rec["provenance"]:
        s, e, t = seg
        if s != pos or e <= s or t not in PROVENANCE_TYPES:
            raise ValueError(f"bad provenance segment {seg} at pos {pos}")
        pos = e
    if pos != len(state):
        raise ValueError(f"provenance ends at {pos}, state has {len(state)} chars")
    q = rec["question"]
    if q.get("type") not in QUESTION_TYPES:
        raise ValueError(f"bad question type {q.get('type')}")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        raise ValueError("empty instructions")
    opts = rec["options"]
    if opts != options_from_criteria(q["type"], q.get("criteria")):
        raise ValueError("options do not match criteria")
    names = [o["name"] for o in opts]
    if len(set(names)) != len(names):
        raise ValueError("duplicate option names")
    gold = rec["gold"]
    if rec["null"]:
        if gold is not None:
            raise ValueError("null record must have gold None")
    else:
        if isinstance(gold, str):
            if gold not in names:
                raise ValueError(f"gold {gold!r} not in options")
        elif isinstance(gold, dict):
            if set(gold) - set(names):
                raise ValueError("gold distribution has unknown names")
            tot = sum(gold.values())
            if abs(tot - 1.0) > 1e-6 or min(gold.values()) < 0:
                raise ValueError(f"gold distribution sums to {tot}")
        else:
            raise ValueError("gold must be a name or a distribution")
    w = rec["witness"]
    if w is not None:
        for s, e in w:
            if not (0 <= s < e <= len(state)):
                raise ValueError(f"witness span {(s, e)} out of range")
