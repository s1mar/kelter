"""Systems evaluated through one interface.

Every system implements `predict(recs, canonical=True) -> list[dict]` with,
per record: "probs" {option name: probability} (may sum to less than 1 when
mass goes to null), "null" (probability of no option fitting, or None when
the system has no null output), optional "witness" [[start, end], ...].

When a system has no null output, the harness uses 1 - max option
probability as its null score.
"""
from __future__ import annotations

from typing import Any, Dict, List


class OursAdapter:
    name = "openverdict"

    def __init__(self, ckpt: str, temperature: bool = True, max_len: int | None = None):
        from openverdict.model import load_predictor
        self.pred = load_predictor(ckpt, max_len=max_len)
        self.temperature = temperature
        self.name = f"openverdict:{ckpt.replace(chr(92), '/').rstrip('/').split('/')[-1]}"

    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True) -> List[Dict[str, Any]]:
        return self.pred.predict(recs, canonical=canonical, temperature=self.temperature)


def get_system(name: str, **kw):
    if name == "ours":
        return OursAdapter(kw["ckpt"], temperature=kw.get("temperature", True))
    if name == "ensemble":  # --ckpt names results/ensemble/<name>.json
        from openverdict.ensemble import Ensemble
        return Ensemble(kw["ckpt"])
    if name == "tfidf":
        from openverdict.eval.baselines.tfidf import TfidfAdapter
        return TfidfAdapter()
    if name == "qwen":
        from openverdict.eval.baselines.qwen_logprob import QwenLogprobAdapter
        return QwenLogprobAdapter(kw.get("model_id"))
    if name == "laya":
        from openverdict.eval.baselines.laya import LayaAdapter
        return LayaAdapter()
    if name == "kev":
        from openverdict.eval.baselines.kev import KevAdapter
        return KevAdapter(kw.get("model_id"))
    raise ValueError(f"unknown system {name}")
