"""ModernBERT cross-encoder with three heads: option scorer (with a null
option), option-conditioned witness tagger, and per-type temperature.

Input layout, built token by token so state offsets are exact:

  [CLS] instructions [SEP] [OPT] option 1 ... [OPT] option k [NULL] none of the options fit [SEP] state [SEP]

Every token also carries a provenance id (0 for question and option tokens,
1.. for the state segment types), added to the token embedding through a
learned table. The option logit is read at each [OPT] marker, the null logit
at [NULL], through the same scorer, so null competes as a real option.
The witness head scores every state token conditioned on one option marker's
hidden state; at inference that is the winning option.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from openverdict.data.schema import PROVENANCE_TYPES

OPT, NULL = "[OPT]", "[NULL]"
NULL_TEXT = "none of the options fit"
PROV_IDS = {t: i + 1 for i, t in enumerate(PROVENANCE_TYPES)}  # 0 = not state


def option_text(o: Dict[str, str]) -> str:
    """Names that carry meaning (e.g. deny_vacancy_exclusion) are shown with the
    description; bare letters and digits are not."""
    name, desc = o["name"], o["description"]
    if re.fullmatch(r"[A-Za-z]|\d+", name) or name.lower() in ("yes", "no"):
        return desc
    human = name.replace("_", " ")
    if human.lower() == desc.lower():
        return desc
    return f"{human}: {desc}"


def canonical_order(options: Sequence[Dict[str, str]]) -> List[int]:
    """Indices sorting options by the sha256 of their rendered text."""
    keys = [hashlib.sha256((o["name"] + "\x1f" + o["description"]).encode("utf-8")).hexdigest() for o in options]
    return sorted(range(len(options)), key=lambda i: keys[i])


@dataclass
class Encoded:
    input_ids: List[int]
    prov_ids: List[int]
    opt_pos: List[int]          # marker positions, in the order options were given
    null_pos: int
    state_start: int            # first state token position
    state_offsets: List[List[int]]  # char offsets of state tokens kept
    truncated: bool = False
    witness_labels: Optional[List[float]] = None


class Featurizer:
    def __init__(self, tokenizer, max_len: int = 512, max_option_tokens: int = 48, max_instr_tokens: int = 96):
        self.tok = tokenizer
        self.max_len = max_len
        self.max_option_tokens = max_option_tokens
        self.max_instr_tokens = max_instr_tokens
        self.opt_id = tokenizer.convert_tokens_to_ids(OPT)
        self.null_id = tokenizer.convert_tokens_to_ids(NULL)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.null_ids = tokenizer(NULL_TEXT, add_special_tokens=False)["input_ids"]

    def pieces_batch(self, recs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tokenize instructions, option texts and states once, in batches.
        Option order is applied later by `assemble`, so per-epoch shuffling
        needs no re-tokenization."""
        t = self.tok
        instr = t([r["question"]["instructions"] for r in recs], add_special_tokens=False)["input_ids"]
        flat = [option_text(o) for r in recs for o in r["options"]]
        oids = t(flat, add_special_tokens=False)["input_ids"] if flat else []
        st = t([r["state"] for r in recs], add_special_tokens=False, return_offsets_mapping=True)
        out, k = [], 0
        for i, r in enumerate(recs):
            n = len(r["options"])
            out.append({
                "instr": instr[i][: self.max_instr_tokens],
                "opts": [ids[: self.max_option_tokens] for ids in oids[k:k + n]],
                "sids": st["input_ids"][i],
                "offs": [list(o) for o in st["offset_mapping"][i]],
            })
            k += n
        return out

    def encode(self, state: str, provenance, instructions: str, options: Sequence[Dict[str, str]],
               witness=None) -> Encoded:
        rec = {"state": state, "question": {"instructions": instructions}, "options": list(options)}
        return self.assemble(self.pieces_batch([rec])[0], provenance, None, witness)

    def assemble(self, pc: Dict[str, Any], provenance, perm: Optional[Sequence[int]] = None,
                 witness=None) -> Encoded:
        """Build the sequence from pieces; perm orders the options (marker
        positions are returned in perm order)."""
        perm = list(range(len(pc["opts"]))) if perm is None else list(perm)
        ids = [self.cls_id] + pc["instr"] + [self.sep_id]
        opt_pos = []
        for j in perm:
            opt_pos.append(len(ids))
            ids += [self.opt_id] + pc["opts"][j]
        null_pos = len(ids)
        ids += [self.null_id] + self.null_ids + [self.sep_id]
        prov = [0] * len(ids)
        sids, offs = pc["sids"], pc["offs"]
        budget = self.max_len - len(ids) - 1
        truncated = len(sids) > budget
        if budget < 1:
            raise ValueError("question and options exceed max_len")
        sids, offs = sids[:budget], offs[:budget]
        seg_types = []
        for s, e in offs:
            mid = s if e <= s else (s + e - 1) // 2
            ptype = "unknown"
            for ss, se, st in provenance:
                if ss <= mid < se:
                    ptype = st
                    break
            seg_types.append(PROV_IDS[ptype])
        state_start = len(ids)
        ids += sids + [self.sep_id]
        prov += seg_types + [0]
        wl = None
        if witness:
            wl = [1.0 if e > s and any(s < we and e > ws for ws, we in witness) else 0.0 for s, e in offs]
        return Encoded(ids, prov, opt_pos, null_pos, state_start, [list(o) for o in offs], truncated, wl)


class VerdictModel(nn.Module):
    def __init__(self, backbone: str, vocab_size: int, n_prov: int = len(PROVENANCE_TYPES) + 1,
                 config_dir: Optional[str] = None, null_head: bool = False):
        """With config_dir, the encoder is built from a saved config (weights come
        from the checkpoint), so loading needs no network and no backbone download."""
        super().__init__()
        self.backbone_name = backbone
        if config_dir is not None:
            conf = AutoConfig.from_pretrained(config_dir)
            conf._attn_implementation = "sdpa"
            self.encoder = AutoModel.from_config(conf)
        else:
            self.encoder = AutoModel.from_pretrained(backbone, attn_implementation="sdpa")
            self.encoder.resize_token_embeddings(vocab_size)
        d = self.encoder.config.hidden_size
        self.prov = nn.Embedding(n_prov, d)
        nn.init.zeros_(self.prov.weight)
        self.scorer = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.w_tok = nn.Linear(d, d)
        self.w_opt = nn.Linear(d, d)
        self.w_out = nn.Sequential(nn.GELU(), nn.Linear(d, 1))
        # r6: abstention as its own binary head on the [NULL] marker,
        # outside the option softmax. Older checkpoints have no null head.
        self.null_head = null_head
        if null_head:
            self.null_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def null_logit(self, m):
        """m (B, K+1, d) marker states with the [NULL] marker last."""
        return self.null_mlp(m[:, -1]).squeeze(-1).float()

    def hidden(self, input_ids, prov_ids, attention_mask):
        emb = self.encoder.get_input_embeddings()(input_ids) + self.prov(prov_ids)
        return self.encoder(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state

    def option_logits(self, h, marker_pos, marker_mask):
        """marker_pos (B, K+1) with the null marker last; padded slots masked."""
        b = torch.arange(h.shape[0], device=h.device).unsqueeze(1)
        m = h[b, marker_pos]  # (B, K+1, d)
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, float("-inf")), m

    def witness_logits(self, h, cond):
        """cond (B, d): hidden state of the conditioning option marker."""
        x = self.w_tok(h) + self.w_opt(cond).unsqueeze(1)
        return self.w_out(x).squeeze(-1).float()


def collate(encs: List[Encoded], pad_id: int, device: str):
    B = len(encs)
    L = max(len(e.input_ids) for e in encs)
    K = max(len(e.opt_pos) for e in encs) + 1
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    prov = torch.zeros((B, L), dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    mpos = torch.zeros((B, K), dtype=torch.long)
    mmask = torch.zeros((B, K), dtype=torch.bool)
    smask = torch.zeros((B, L), dtype=torch.bool)
    wlab = torch.zeros((B, L), dtype=torch.float)
    has_w = torch.zeros(B, dtype=torch.bool)
    for i, e in enumerate(encs):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        prov[i, :n] = torch.tensor(e.prov_ids)
        att[i, :n] = 1
        k = len(e.opt_pos)
        mpos[i, :k] = torch.tensor(e.opt_pos)
        mmask[i, :k] = True
        mpos[i, K - 1] = e.null_pos
        mmask[i, K - 1] = True
        ns = len(e.state_offsets)
        smask[i, e.state_start:e.state_start + ns] = True
        if e.witness_labels is not None:
            wlab[i, e.state_start:e.state_start + ns] = torch.tensor(e.witness_labels)
            has_w[i] = True
    t = lambda x: x.to(device, non_blocking=True)
    return {"input_ids": t(ids), "prov_ids": t(prov), "attention_mask": t(att), "marker_pos": t(mpos),
            "marker_mask": t(mmask), "state_mask": t(smask), "witness_labels": t(wlab), "has_witness": t(has_w)}


def spans_from_tokens(probs: Sequence[float], offsets: Sequence[Sequence[int]], thresh: float = 0.5,
                      max_spans: int = 3, state: Optional[str] = None) -> List[List[int]]:
    """Contiguous runs of tokens above thresh become char spans; the max_spans
    runs with the highest mean probability are kept, in text order. Runs whose
    gap in the text is only whitespace are merged."""
    runs, cur = [], None
    for p, (s, e) in zip(probs, offsets):
        if e <= s:
            continue
        if p > thresh:
            if cur is None:
                cur = [s, e, [p]]
            else:
                gap = state[cur[1]:s] if state is not None else " "
                if gap.strip() == "":
                    cur[1] = e
                    cur[2].append(p)
                else:
                    runs.append(cur)
                    cur = [s, e, [p]]
        else:
            if cur is not None:
                runs.append(cur)
                cur = None
    if cur is not None:
        runs.append(cur)
    runs.sort(key=lambda r: -sum(r[2]) / len(r[2]))
    keep = sorted(runs[:max_spans], key=lambda r: r[0])
    return [[r[0], r[1]] for r in keep]


def load_predictor(ckpt_dir: str, device: Optional[str] = None, max_len: Optional[int] = None):
    """A Predictor, or an Ensemble when the directory's config names one.
    device None: openverdict.device.default_device()."""
    from openverdict.device import default_device
    device = device or default_device()
    cfg = json.loads((Path(ckpt_dir) / "config.json").read_text(encoding="utf-8"))
    if cfg.get("ensemble"):
        from openverdict.ensemble import Ensemble
        return Ensemble(cfg["ensemble"], device=device, cfg=cfg)
    return Predictor(ckpt_dir, device=device, max_len=max_len)


class Predictor:
    """Loads a trained checkpoint and scores records (unified schema or the
    equivalent request pieces). Returns probabilities in the caller's option
    order plus null probability and the witness for the winning option."""

    def __init__(self, ckpt_dir: str, device: Optional[str] = None, max_len: Optional[int] = None):
        from openverdict.device import default_device
        device = device or default_device()
        ckpt = Path(ckpt_dir)
        cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"))
        self.feat = Featurizer(self.tok, max_len=max_len or cfg.get("infer_max_len", cfg["max_len"]))
        enc_dir = ckpt / "encoder_config"
        if not enc_dir.exists():  # checkpoints written before encoder_config was saved
            AutoConfig.from_pretrained(cfg["backbone"]).save_pretrained(str(enc_dir))
            c = AutoConfig.from_pretrained(str(enc_dir))
            c.vocab_size = len(self.tok)
            c.save_pretrained(str(enc_dir))
        self.null_head = bool(cfg.get("null_head", False))
        self.null_cal = cfg.get("null_calibration") or {}
        self.model = VerdictModel(cfg["backbone"], len(self.tok), config_dir=str(enc_dir), null_head=self.null_head)
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval().to(torch.bfloat16)
        self.device = device
        self.temps: Dict[str, float] = cfg.get("temperature", {})
        self.model_hash = cfg.get("model_hash", "")
        self.token_budget = int(cfg.get("infer_token_budget", 16384))

    @torch.inference_mode()
    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True, temperature: bool = True,
                batch_size: int = 16, witness_thresh: float = 0.5) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = [None] * len(recs)  # type: ignore
        order = sorted(range(len(recs)), key=lambda i: len(recs[i]["state"]))
        for bi in range(0, len(order), batch_size):
            idx = order[bi:bi + batch_size]
            encs, perms = [], []
            pcs = self.feat.pieces_batch([recs[i] for i in idx])
            for i, pc in zip(idx, pcs):
                r = recs[i]
                perm = canonical_order(r["options"]) if canonical else list(range(len(r["options"])))
                encs.append(self.feat.assemble(pc, r["provenance"], perm))
                perms.append(perm)
            # split the chunk so padded tokens per forward stay under token_budget
            groups, cur = [], []
            for j in range(len(idx)):
                L = max([len(encs[x].input_ids) for x in cur] + [len(encs[j].input_ids)])
                if cur and L * (len(cur) + 1) > self.token_budget:
                    groups.append(cur)
                    cur = []
                cur.append(j)
            groups.append(cur)
            for g in groups:
                self._score_group([idx[j] for j in g], [encs[j] for j in g], [perms[j] for j in g],
                                  recs, out, temperature, witness_thresh)
        return out

    def _score_group(self, idx, encs, perms, recs, out, temperature, witness_thresh):
        b = collate(encs, self.tok.pad_token_id, self.device)
        h = self.model.hidden(b["input_ids"], b["prov_ids"], b["attention_mask"])
        logits, m = self.model.option_logits(h, b["marker_pos"], b["marker_mask"])
        nl = self.model.null_logit(m) if self.null_head else None
        for row, i in enumerate(idx):
            r, e, perm = recs[i], encs[row], perms[row]
            k = len(e.opt_pos)
            T = self.temps.get(r["question"]["type"], 1.0) if temperature else 1.0
            extra = {}
            if self.null_head:
                lg = logits[row, :k]
                po = torch.softmax(lg / T, -1).cpu().tolist()
                z = float(nl[row])
                pn = self.null_score(z, po, temperature)
                probs = {r["options"][perm[j]]["name"]: po[j] * (1.0 - pn) for j in range(k)}
                p = po + [pn]
                extra = {"option_probs": {r["options"][perm[j]]["name"]: po[j] for j in range(k)},
                         "null_logit": z, "null_threshold": self.null_cal.get("threshold", 0.5) if temperature else 0.5}
            else:
                lg = torch.cat([logits[row, :k], logits[row, -1:]])
                p = torch.softmax(lg / T, -1).cpu().tolist()
                probs = {r["options"][perm[j]]["name"]: p[j] for j in range(k)}
            win = int(torch.argmax(lg[:k]))
            cond = m[row, win].unsqueeze(0)
            wl = self.model.witness_logits(h[row:row + 1], cond)[0]
            ns = len(e.state_offsets)
            wp = torch.sigmoid(wl[e.state_start:e.state_start + ns]).cpu().tolist()
            out[i] = {
                "probs": probs,
                "null": p[k],
                "logits": lg.cpu().tolist(),
                "order": [r["options"][j]["name"] for j in perm],
                "witness": spans_from_tokens(wp, e.state_offsets, witness_thresh, 3, r["state"]),
                "truncated": e.truncated,
                "n_tokens": len(e.input_ids),
                **extra,
            }

    def null_score(self, z: float, po: List[float], calibrated: bool = True) -> float:
        """Null probability from the null head: the fitted stacker over
        [null_logit, 1 - max option prob, option entropy] when calibration chose
        it, else sigmoid(z / T_null). Uncalibrated: sigmoid(z)."""
        import math
        c = self.null_cal if calibrated else {}
        if c.get("use_stacker"):
            ent = -sum(x * math.log(max(x, 1e-12)) for x in po)
            w, b0 = c["stacker_w"], c["stacker_b"]
            u = w[0] * z + w[1] * (1.0 - max(po)) + w[2] * ent + b0
        else:
            u = z / c.get("T_null", 1.0)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, u))))
