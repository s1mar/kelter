"""Laya (Convai Innovations, Apache-2.0) as a baseline system.

Package: `laya` 0.4.15 from PyPI (author Convai Innovations, repo
github.com/NandhaKishorM/laya, weights convaiinnovations/laya on the Hub).
Loaded with `laya.load(model_id, device=...)`; scored with
`Agent.predict_batch(states, questions)`, which returns per state
{"answers": {qid: {"probabilities": {option: p}, ...}}}.

Every record is posed as a `choice` question over its options (names as keys,
descriptions as criteria), so noul and score records are scored the same way as
choice records and the returned names are exactly the record's option names.
Records that share instructions and options are batched together.

Order invariance: not guaranteed. The checkpoint loads with parallel_options
False, so options are encoded as one sequence in the order given. This adapter
does not canonicalise or average over permutations and ignores `canonical`.

Default checkpoint: the general English model `convaiinnovations/laya` (ModernBERT-large,
421M parameters, 512-token context). `laya-typed-decisions` is tuned to four
specific workflows and is not the default.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

DEFAULT_MODEL = "convaiinnovations/laya"


class LayaAdapter:
    def __init__(self, model_id: str | None = None, device: str | None = None, batch_size: int = 16):
        self.model_id = model_id or DEFAULT_MODEL
        self.name = "laya:" + self.model_id.split("/")[-1]
        self.device = device
        self.batch_size = batch_size
        self._agent = None

    def _load(self):
        if self._agent is not None:
            return
        try:
            import laya
        except ImportError as e:
            raise RuntimeError("laya is not installed; run `pip install laya` (0.4.15 was used, "
                               "its requirements are met by the project's torch and transformers)") from e
        self._agent = laya.load(self.model_id, device=self.device)

    @staticmethod
    def _question(r: Dict[str, Any]) -> Dict[str, Any]:
        return {"q": {"type": "choice", "instructions": r["question"]["instructions"],
                      "criteria": {o["name"]: o["description"] for o in r["options"]}}}

    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True) -> List[Dict[str, Any]]:
        """Not canonicalised; `canonical` is ignored (see module docstring)."""
        self._load()
        groups = defaultdict(list)
        for i, r in enumerate(recs):
            groups[json.dumps(self._question(r), sort_keys=False)].append(i)
        out: List[Dict[str, Any] | None] = [None] * len(recs)
        for key, idx in groups.items():
            q = json.loads(key)
            for s in range(0, len(idx), self.batch_size):
                part = idx[s:s + self.batch_size]
                res = self._agent.predict_batch([recs[i]["state"] for i in part], q)
                for i, a in zip(part, res):
                    probs = a["answers"]["q"]["probabilities"]
                    names = [o["name"] for o in recs[i]["options"]]
                    p = np.array([float(probs[n]) for n in names], dtype=np.float64)
                    p = p / p.sum() if p.sum() > 0 else np.full(len(names), 1.0 / len(names))
                    out[i] = {"probs": dict(zip(names, map(float, p))), "null": None}
        return out
