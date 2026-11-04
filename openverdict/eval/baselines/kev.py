"""Kev-0.8B (jaredpalmer/kev, Apache-2.0) in-process, without the kev repo or peft.

Kev is Qwen3.5-0.8B-Base plus a rank-16 LoRA plus a pointer head. This file
re-implements only the minimal scoring path, following these files of
github.com/jaredpalmer/kev (read, not executed):
  kev/model.py      SPECIAL tokens, user_tokens, encode (record packing),
                    PointerHead, DecisionModel._rows_hidden / probs path
  kev/checkpoint.py head.pt layout (keys base, base_revision, head, temperature),
                    adapter loading (PeftModel on the bare backbone)
  kev/api.py        option_text and the per-type option rendering (to_record)

Sequence for one question:
  <|fim_prefix|> state-tokens  <|fim_middle|> instructions
  <|box_start|> option <|box_end|> ... <|fim_suffix|>
The backbone is run causally over the whole sequence (the hybrid-backbone "row"
form, valid for one question per record); the last-layer hidden state at
<|fim_suffix|> (decide) is scored against the hidden state at each option's
<|box_end|> by the pointer head; logits are divided by the temperature stored in
head.pt (2.351) and softmaxed.

The LoRA is merged into the base weights by hand: W += (alpha / r) * B @ A
(alpha 32, r 16), in fp32 (the adapter and head are fp32 on the Hub).

Order invariance: none by construction (options are one causal sequence, as in
Kev's default non-isolated mode). `canonical` is ignored.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import torch

KEV_REPO = "jaredpalmer/kev-0.8b"
KEV_REVISION = "9a45d25eb2ab761841196625383fa1dff0e56c1e"   # Kev 1.0 weights (main is a model-card-only commit later)
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE = 8192            # Kev's validated serving context in the model card
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text: str) -> List[int]:
    """kev.model.user_tokens: caller text can never produce delimiter tokens."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def option_strings(r: Dict[str, Any]) -> List[str]:
    """kev.api.to_record: choice and noul render `name: description`, score renders the level text."""
    t = r["question"]["type"]
    if t == "score":
        return [o["description"] for o in r["options"]]
    return [o["name"] if not o["description"] else f"{o['name']}: {o['description']}" for o in r["options"]]


def encode(tok, state: str, instr: str, options: List[str], max_state: int = MAX_STATE):
    """kev.model.encode for one question. Returns (ids, decide_idx, option_idx)."""
    ids_special = [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    s_id, q_id, o_id, c_id, d_id = ids_special
    ids = [s_id] + user_tokens(tok, state)[: max_state - 1]
    ids += [q_id] + user_tokens(tok, instr)
    opt_idx = []
    for o in options:
        ids += [o_id] + user_tokens(tok, o) + [c_id]
        opt_idx.append(len(ids) - 1)
    ids.append(d_id)
    return ids, len(ids) - 1, opt_idx


def merge_lora(lm: torch.nn.Module, adapter_path: str, r: int, alpha: int) -> int:
    """Fold a peft LoRA safetensors file into lm's weights in place. Returns the number of modules merged."""
    from safetensors.torch import load_file

    w = load_file(adapter_path)
    params = dict(lm.named_parameters())
    scale = alpha / r
    n = 0
    for k in list(w):
        if not k.endswith(".lora_A.weight"):
            continue
        base = k[: -len(".lora_A.weight")]
        name = base.removeprefix("base_model.model.") + ".weight"
        A, B = w[k], w[base + ".lora_B.weight"]
        if name not in params:
            raise KeyError(f"adapter module {name} not found in backbone")
        p = params[name]
        with torch.no_grad():
            p.copy_((p.float() + scale * (B.float() @ A.float())).to(p.dtype))
        n += 1
    return n


class KevAdapter:
    def __init__(self, model_id: str | None = None, device: str | None = None, dtype=None,
                 temperature: float | None = None):
        self.model_id = model_id or KEV_REPO
        self.name = "kev:" + self.model_id.split("/")[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.temperature_override = temperature
        self._lm = None

    def _load(self):
        if self._lm is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import json
        import os

        rev = KEV_REVISION if self.model_id == KEV_REPO else None
        if os.path.isdir(self.model_id):
            path = self.model_id
        else:
            path = snapshot_download(self.model_id, revision=rev,
                                     allow_patterns=["adapter_config.json", "adapter_model.safetensors", "head.pt"])
        meta = torch.load(os.path.join(path, "head.pt"), map_location="cpu", weights_only=True)
        cfg = json.load(open(os.path.join(path, "adapter_config.json"), encoding="utf-8"))
        base, base_rev = meta["base"], meta.get("base_revision")
        tok = AutoTokenizer.from_pretrained(base, revision=base_rev)
        missing = [t for t in SPECIAL if tok.convert_tokens_to_ids(t) in (None, tok.unk_token_id)]
        if missing:
            raise RuntimeError(f"tokenizer lacks the delimiter tokens {missing}")
        lm = AutoModelForCausalLM.from_pretrained(base, revision=base_rev, dtype=torch.float32,
                                                  attn_implementation="eager").model
        merged = merge_lora(lm, os.path.join(path, "adapter_model.safetensors"), cfg["r"], cfg["lora_alpha"])
        if merged != 372 // 2:
            raise RuntimeError(f"merged {merged} LoRA modules, expected 186")
        lm = lm.to(self.dtype).to(self.device).eval()
        d = meta["head"]["q.weight"].shape[1]
        self._q = torch.nn.Linear(d, meta["head_dim"]).to(self.device)
        self._k = torch.nn.Linear(d, meta["head_dim"]).to(self.device)
        self._q.load_state_dict({"weight": meta["head"]["q.weight"], "bias": meta["head"]["q.bias"]})
        self._k.load_state_dict({"weight": meta["head"]["k.weight"], "bias": meta["head"]["k.bias"]})
        self._scale = 1.0 / meta["head_dim"] ** 0.5
        self._temp = float(meta["temperature"]) if self.temperature_override is None else self.temperature_override
        self._tok, self._lm = tok, lm

    @torch.no_grad()
    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True) -> List[Dict[str, Any]]:
        """Not order-invariant; `canonical` is ignored (see module docstring)."""
        self._load()
        out = []
        for r in recs:
            ids, dec, opts = encode(self._tok, r["state"], r["question"]["instructions"], option_strings(r))
            x = torch.tensor([ids], device=self.device)
            h = self._lm(input_ids=x).last_hidden_state[0].float()
            z = (self._k(h[opts]) @ self._q(h[dec])) * self._scale / self._temp
            p = torch.softmax(z, -1).cpu().tolist()
            out.append({"probs": {o["name"]: float(v) for o, v in zip(r["options"], p)}, "null": None})
        return out
