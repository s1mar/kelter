"""Zero-shot letter readout from a small open Qwen instruct model.

The prompt holds the instructions, the state and the options labelled A, B, C, ...
and asks for the single letter. The next-token logits of the letter tokens
(bare and leading-space variants) are softmaxed over the letters.

Order sensitivity: the model is NOT order-invariant. With canonical=True the
option list is evaluated under K = min(4, n) cyclic rotations and the letter
probabilities are mapped back to option names and averaged, which cancels a
pure position bias. With canonical=False only the given order is used (K = 1).
Records with more than 26 options are not supported (the dev, test and train
sets have at most 10).
"""
from __future__ import annotations

import string
from typing import Any, Dict, List, Sequence

import numpy as np

from kelter.model import option_text

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LETTERS = string.ascii_uppercase
MAX_PROMPT_TOKENS = 2048


def rotation_order(n: int, k: int) -> List[int]:
    """Option indices shown in slots 0..n-1 under rotation k."""
    return [(i + k) % n for i in range(n)]


def n_rotations(n: int, canonical: bool, cap: int = 4) -> int:
    return min(cap, n) if canonical else 1


def aggregate_rotations(slot_probs: Sequence[Sequence[float]], n: int) -> np.ndarray:
    """Average per-slot probabilities over rotations, mapped back to option indices.

    slot_probs[k][s] is the probability the model gave to slot s when the
    options were shown in rotation_order(n, k). Pure function; no model needed.
    """
    total = np.zeros(n, dtype=np.float64)
    for k, sp in enumerate(slot_probs):
        sp = np.asarray(sp, dtype=np.float64)
        sp = sp / sp.sum()
        for slot, opt in enumerate(rotation_order(n, k)):
            total[opt] += sp[slot]
    return total / len(slot_probs)


def letter_probs(letter_logits: Sequence[Sequence[float]]) -> np.ndarray:
    """letter_logits[s] = logits of the token variants of slot s's letter; logsumexp over
    the variants, then softmax over slots."""
    z = np.array([np.logaddexp.reduce(np.asarray(v, dtype=np.float64)) for v in letter_logits])
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class QwenLogprobAdapter:
    def __init__(self, model_id: str | None = None, device: str | None = None, dtype=None,
                 max_prompt_tokens: int = MAX_PROMPT_TOKENS):
        import torch

        self.model_id = model_id or DEFAULT_MODEL
        self.name = "qwen:" + self.model_id.split("/")[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.max_prompt_tokens = max_prompt_tokens
        self._tok = None
        self._model = None
        self._letter_ids: List[List[int]] | None = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=self.dtype)
        model.to(self.device).eval()
        ids = []
        for ch in LETTERS:
            variants = set()
            for s in (ch, " " + ch):
                t = tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    variants.add(t[0])
            ids.append(sorted(variants))
        self._tok, self._model, self._letter_ids = tok, model, ids

    def _prompt(self, r: Dict[str, Any], order: Sequence[int]) -> str:
        tok = self._tok
        opts = "\n".join(f"{LETTERS[s]}. {option_text(r['options'][i])}" for s, i in enumerate(order))

        def build(state: str) -> str:
            user = (f"{r['question']['instructions']}\n\nText:\n{state}\n\nOptions:\n{opts}\n\n"
                    "Answer with the single letter of the best option.")
            return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True)

        text = build(r["state"])
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n > self.max_prompt_tokens:
            state_ids = tok(r["state"], add_special_tokens=False)["input_ids"]
            keep = max(16, len(state_ids) - (n - self.max_prompt_tokens) - 8)
            text = build(tok.decode(state_ids[:keep]) + " [...]")
        return text

    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True) -> List[Dict[str, Any]]:
        """Not order-invariant; canonical=True averages over cyclic rotations (see module docstring)."""
        import torch

        self._load()
        out = []
        for r in recs:
            n = len(r["options"])
            if n > len(LETTERS):
                raise ValueError(f"{n} options exceed the {len(LETTERS)} letters this adapter supports")
            K = n_rotations(n, canonical)
            prompts = [self._prompt(r, rotation_order(n, k)) for k in range(K)]
            enc = self._tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :].float().cpu().numpy()
            slot_probs = [letter_probs([[logits[k, t] for t in self._letter_ids[s]] for s in range(n)])
                          for k in range(K)]
            p = aggregate_rotations(slot_probs, n)
            out.append({"probs": {o["name"]: float(x) for o, x in zip(r["options"], p)}, "null": None})
        return out
