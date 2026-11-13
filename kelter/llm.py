"""LLM-backed predictor behind the same engine contract.

A causal LM (Qwen family) reads one prompt per record: the instructions, the
state, the options labelled A, B, C, ... and one more letter for "none of the
options", the explicit null choice. The next-token logits of those letters are
the record's logits, in the order [shown options..., null], the same layout as
an encoder checkpoint without a null head, so kelter.calibrate fits
per-type temperature and split conformal on it unchanged.

Order: choice and noul options are shown in canonical order (sha256 of the
rendered option, kelter.model.canonical_order), so any permutation of the
request gives the same prompt and the same answer. Score levels keep their
given order, since their order carries their meaning.

Witness: the encoder checkpoint named in the config ("witness_ckpt", r2 by
default) supplies the witness spans; the LLM has no span output.

The fine-tune is a LoRA adapter (kelter.llm_train). Serving loads the base
in bf16 and merges the adapter.
"""
from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LETTERS = string.ascii_uppercase
NULL_TEXT = "None of the options: the text does not settle the question."
ANSWER_LINE = "Answer with the single letter of the best option."
PREFILL_CHUNK = 1024  # tokens per prefill chunk; 0 is off


def prefill_chunk_setting(cfg=None) -> int:
    """Effective prefill chunk: KELTER_PREFILL_CHUNK, else the checkpoint config, else PREFILL_CHUNK."""
    import os
    return int(os.environ.get("KELTER_PREFILL_CHUNK", (cfg or {}).get("prefill_chunk", PREFILL_CHUNK)))


def show_order(r: Dict[str, Any], canonical: bool) -> List[int]:
    from kelter.model import canonical_order
    n = len(r["options"])
    if canonical and r["question"]["type"] != "score":
        return canonical_order(r["options"])
    return list(range(n))


def user_text(r: Dict[str, Any], order: Sequence[int], state: str) -> str:
    from kelter.model import option_text
    lines = [f"{LETTERS[s]}. {option_text(r['options'][i])}" for s, i in enumerate(order)]
    lines.append(f"{LETTERS[len(order)]}. {NULL_TEXT}")
    return (f"{r['question']['instructions']}\n\nText:\n{state}\n\nOptions:\n" + "\n".join(lines)
            + f"\n\n{ANSWER_LINE}")


def build_prompt(tok, r: Dict[str, Any], order: Sequence[int], max_tokens: int) -> List[int]:
    """Chat-templated prompt ids ending where the answer letter goes. A state too
    long for max_tokens keeps its first two thirds and last third of the token
    budget, joined by ' [...] '."""
    def ids_for(state: str) -> List[int]:
        text = tok.apply_chat_template([{"role": "user", "content": user_text(r, order, state)}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    ids = ids_for(r["state"])
    if len(ids) <= max_tokens:
        return ids
    s = tok(r["state"], add_special_tokens=False)["input_ids"]
    keep = max(32, len(s) - (len(ids) - max_tokens) - 16)
    head = (2 * keep) // 3
    state = tok.decode(s[:head]) + " [...] " + tok.decode(s[len(s) - (keep - head):])
    return ids_for(state)[-max_tokens:]


def letter_ids(tok) -> List[int]:
    out = []
    for ch in LETTERS:
        t = tok.encode(ch, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"letter {ch!r} is not a single token")
        out.append(t[0])
    return out


def gold_slot(r: Dict[str, Any], order: Sequence[int]) -> int:
    if r["null"]:
        return len(order)
    g = r["gold"] if isinstance(r["gold"], str) else max(r["gold"], key=r["gold"].get)
    names = [r["options"][i]["name"] for i in order]
    return names.index(g)


def adapter_hash(ckpt: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((ckpt / "adapter").glob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    h.update(json.dumps(json.loads((ckpt / "config.json").read_text(encoding="utf-8"))["llm"], sort_keys=True).encode())
    return h.hexdigest()


class LLMPredictor:
    null_head = False

    def __init__(self, ckpt_dir: str, device: Optional[str] = None, witness: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from kelter.device import default_device
        self.device = device or default_device()
        ck = Path(ckpt_dir)
        self.cfg = json.loads((ck / "config.json").read_text(encoding="utf-8"))
        L = self.cfg["llm"]
        self.max_tokens = int(self.cfg.get("infer_max_len", L.get("max_len", 2048)))
        self.token_budget = int(L.get("token_budget", 8192))
        self.prefill_chunk = prefill_chunk_setting(self.cfg)
        self.temps = self.cfg.get("temperature", {})
        self.tok = AutoTokenizer.from_pretrained(L["base"])
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(L["base"], dtype=dtype)
        if (ck / "adapter").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ck / "adapter")).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.letters = letter_ids(self.tok)
        self.model_hash = adapter_hash(ck) if (ck / "adapter").exists() else hashlib.sha256(L["base"].encode()).hexdigest()
        self.witness_pred = None
        if witness and self.cfg.get("witness_ckpt"):
            from kelter.model import Predictor
            from kelter import paths
            self.witness_pred = Predictor(str(paths.under_home(self.cfg["witness_ckpt"])), device=self.device)

    def _chunked(self, p: List[int]):
        """Last-position letter logits of one long prompt, prefilled in chunks of
        prefill_chunk tokens through the KV cache, so
        attention memory grows with chunk x length, not length x length."""
        import torch
        from transformers import DynamicCache
        ids = torch.tensor([p], device=self.device)
        cache = DynamicCache()
        with torch.inference_mode():
            for s in range(0, len(p), self.prefill_chunk):
                lg = self.model(input_ids=ids[:, s:s + self.prefill_chunk], past_key_values=cache,
                                use_cache=True, logits_to_keep=1).logits[:, -1, :]
        del cache
        return lg[0, self.letters].float().cpu()

    def logits_for(self, prompts: List[List[int]]):
        """Last-position logits (float32, on CPU) for each prompt. Prompts longer than
        prefill_chunk run one at a time, chunked; the rest are batched under token_budget."""
        import torch
        out = [None] * len(prompts)
        if self.prefill_chunk:
            for i, p in enumerate(prompts):
                if len(p) > self.prefill_chunk:
                    out[i] = self._chunked(p)
        order = sorted((i for i in range(len(prompts)) if out[i] is None), key=lambda i: len(prompts[i]))
        groups, cur = [], []
        for i in order:
            L = len(prompts[i])
            if cur and L * (len(cur) + 1) > self.token_budget:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        pad = self.tok.pad_token_id
        for g in groups:
            L = max(len(prompts[i]) for i in g)
            ids = torch.full((len(g), L), pad, dtype=torch.long)
            att = torch.zeros((len(g), L), dtype=torch.long)
            for row, i in enumerate(g):
                p = prompts[i]
                ids[row, L - len(p):] = torch.tensor(p)
                att[row, L - len(p):] = 1
            with torch.inference_mode():
                lg = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                                logits_to_keep=1).logits[:, -1, :]
                lg = lg[:, self.letters].float().cpu()
            for row, i in enumerate(g):
                out[i] = lg[row]
        return out

    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True, temperature: bool = True,
                **_) -> List[Dict[str, Any]]:
        import torch
        orders = [show_order(r, canonical) for r in recs]
        for r in recs:
            if len(r["options"]) + 1 > len(LETTERS):
                raise ValueError(f"{len(r['options'])} options exceed the letters this predictor supports")
        prompts = [build_prompt(self.tok, r, o, self.max_tokens) for r, o in zip(recs, orders)]
        lgs = self.logits_for(prompts)
        wit = self.witness_pred.predict(recs, canonical=canonical) if self.witness_pred else [None] * len(recs)
        out = []
        for r, o, p, lg, w in zip(recs, orders, prompts, lgs, wit):
            k = len(o)
            z = lg[:k + 1]
            T = self.temps.get(r["question"]["type"], 1.0) if temperature else 1.0
            pr = torch.softmax(z / T, -1).tolist()
            out.append({
                "probs": {r["options"][o[j]]["name"]: pr[j] for j in range(k)},
                "null": pr[k],
                "logits": z.tolist(),
                "order": [r["options"][j]["name"] for j in o],
                "witness": (w or {}).get("witness") or [],
                "n_tokens": len(p),
            })
        return out
