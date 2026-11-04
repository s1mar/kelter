"""Test doubles for the server."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class FakeEngine:
    """Deterministic engine. Records every decide() call in `calls`.

    Default probabilities: option i gets weight (k - i), scaled to 0.9 total,
    and 0.1 goes to the null, so probs sum below 1 like the real engine.
    Pass `probs_fn(rec) -> {name: p}` to override per record.
    """

    def __init__(self, model_id: str = "fake-verdict-1",
                 probs_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None):
        self.model_id = model_id
        self.probs_fn = probs_fn
        self.calls: List[List[Dict[str, Any]]] = []

    def decide(self, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.calls.append(recs)
        return [self._one(r) for r in recs]

    def _one(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        names = [o["name"] for o in rec["options"]]
        if self.probs_fn is not None:
            probs = self.probs_fn(rec)
        else:
            k = len(names)
            tot = k * (k + 1) / 2
            probs = {n: 0.9 * (k - i) / tot for i, n in enumerate(names)}
        top = max(probs, key=probs.get)
        state = rec["state"]
        return {
            "probs": probs,
            "null_probability": max(0.0, 1.0 - sum(probs.values())),
            "witness": [{"start": 0, "end": min(5, len(state)), "provenance": rec["provenance"][0][2], "text": state[:5]}],
            "verifier": {"agrees": True, "score": 0.9},
            "set": [top],
            "coverage_level": 0.95,
            "action": "act",
            "receipt": {"engine": "fake", "record_id": rec["id"]},
            "input_tokens": len(state.split()) + 3,
        }
