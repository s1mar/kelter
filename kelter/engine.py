"""Decision engine: model, four-pass verifier, policy table, conformal layer.

For every record the engine runs one batched forward over four variants:
  full            the request as given
  witness_only    everything except the witness union masked
  witness_masked  the witness union masked
  untrusted_masked every untrusted segment masked (policy table untrusted_types)

A witness is admissible when it is non-empty, every span lies inside the
state, every span lies entirely in segments whose provenance type the policy
table allows for the verdict's class, the verdict survives the witness-only
pass (sufficiency), and the verdict changes in the witness-masked pass
(necessity; a class whose policy says necessity "report" skips this check).
`untrusted_influence` is the probability the winning option loses when
untrusted segments are masked, floored at 0.

Action: null wins -> defer; no admissible witness -> defer; untrusted
influence over the class budget -> escalate; conformal set not a singleton
-> escalate; otherwise act. These are ERASER's sufficiency and
comprehensiveness (DeYoung et al., 2020) run per decision at inference time.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from kelter.data.augment import mask_spans

ROOT = Path(__file__).resolve().parents[1]
NULL = "__null__"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the policy table and check it against its signature file
    (<name>.sha256 holding the sha256 of the canonical JSON)."""
    path = path or ROOT / "policy_table.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    sig_path = path.with_suffix(".sha256")
    digest = sha256(canonical_json(table))
    if not sig_path.exists() or sig_path.read_text(encoding="utf-8").strip() != digest:
        raise ValueError(f"policy table {path.name} does not match its signature {sig_path.name}")
    table["_sha256"] = digest
    return table


def sign_policy(path: Optional[Path] = None) -> str:
    path = path or ROOT / "policy_table.json"
    digest = sha256(canonical_json(json.loads(path.read_text(encoding="utf-8"))))
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def merge(spans: Sequence[Sequence[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def complement(spans, n: int) -> List[List[int]]:
    out, pos = [], 0
    for s, e in merge(spans):
        if s > pos:
            out.append([pos, s])
        pos = max(pos, e)
    if pos < n:
        out.append([pos, n])
    return out


def split_by_segments(span, provenance) -> List[Dict[str, Any]]:
    s, e = span
    out = []
    for ss, se, t in provenance:
        a, b = max(s, ss), min(e, se)
        if a < b:
            out.append({"start": a, "end": b, "provenance": t})
    return out


def verdict_of(out: Dict[str, Any]) -> str:
    best = max(out["probs"], key=out["probs"].get)
    if "null_threshold" in out:  # separate null head (r6+)
        return NULL if out["null"] > out["null_threshold"] else best
    return NULL if out["null"] > out["probs"][best] else best


def conformal_set(out: Dict[str, Any], q: float) -> List[str]:
    """Labels whose nonconformity 1 - p is at most q (null included as NULL).
    With a separate null head the set is over options only (option softmax)
    and NULL joins it only when the null decision fires."""
    if "option_probs" in out:
        po = out["option_probs"]
        s = sorted([k for k, p in po.items() if 1.0 - p <= q] or [max(po, key=po.get)], key=lambda k: -po[k])
        return ([NULL] + s) if out["null"] > out["null_threshold"] else s
    labels = dict(out["probs"])
    labels[NULL] = out["null"]
    s = [k for k, p in labels.items() if 1.0 - p <= q]
    if not s:
        s = [max(labels, key=labels.get)]
    return sorted(s, key=lambda k: -labels[k])


class Engine:
    def __init__(self, predictor, policy: Optional[Dict[str, Any]] = None, alpha: float = 0.05):
        self.pred = predictor
        self.policy = policy or load_policy()
        self.alpha = alpha
        self.model_id = "kelter-" + getattr(predictor, "model_hash", "")[:12]
        cfg = getattr(predictor, "cfg", {}) or {}
        self.conformal = (cfg.get("conformal") or {}).get(str(alpha), {})

    def _class_of(self, rec: Dict[str, Any], verdict: str) -> str:
        vc = (rec.get("meta") or {}).get("verdict_class") or {}
        c = vc.get(verdict, "default")
        return c if c in self.policy["classes"] else "default"

    def _q(self, rec) -> tuple:
        dom = rec.get("domain", "")
        if dom in self.conformal:
            return dom, self.conformal[dom]
        if "_global" in self.conformal:
            return "_global", self.conformal["_global"]
        return "none", 1.0  # no calibration: the set is every label, so nothing acts

    def decide(self, recs: List[Dict[str, Any]], canonical: bool = True) -> List[Dict[str, Any]]:
        full = self.pred.predict(recs, canonical=canonical)
        untrusted = set(self.policy["untrusted_types"])
        variants, index = [], []
        for i, (r, o) in enumerate(zip(recs, full)):
            w = merge(o["witness"])
            n = len(r["state"])
            if w:
                variants.append(mask_spans(r, complement(w, n)))
                variants.append(mask_spans(r, w))
            else:
                variants.extend([None, None])
            um = [[s, e] for s, e, t in r["provenance"] if t in untrusted]
            variants.append(mask_spans(r, um) if um else None)
            index.append(i)
        todo = [v for v in variants if v is not None]
        outs = iter(self.pred.predict(todo, canonical=canonical)) if todo else iter([])
        passes = [next(outs) if v is not None else None for v in variants]

        results = []
        for i, (r, o) in enumerate(zip(recs, full)):
            wo, wm, um = passes[3 * i:3 * i + 3]
            v = verdict_of(o)
            win_p = o["null"] if v == NULL else (o.get("option_probs") or o["probs"])[v]
            w = merge(o["witness"])
            cls = self._class_of(r, v) if v != NULL else "default"
            rule = self.policy["classes"][cls]
            pieces = [p for s in w for p in split_by_segments(s, r["provenance"])]
            in_bounds = all(0 <= s < e <= len(r["state"]) for s, e in w)
            prov_ok = bool(pieces) and all(p["provenance"] in rule["allowed_witness_types"] for p in pieces)
            suff = wo is not None and verdict_of(wo) == v
            nec = wm is not None and verdict_of(wm) != v
            nec_needed = rule["necessity"] == "gate"
            admissible = bool(w) and in_bounds and prov_ok and suff and (nec or not nec_needed)

            def p_of(x):
                if x is None:
                    return None
                if v == NULL:
                    return x["null"]
                return (x.get("option_probs") or x["probs"]).get(v, 0.0)

            ui = 0.0 if um is None else max(0.0, win_p - p_of(um))
            dom, q = self._q(r)
            cset = conformal_set(o, q)
            if v == NULL:
                action = "defer"
            elif not admissible:
                action = "defer"
            elif ui > rule["untrusted_influence_budget"]:
                action = "escalate"
            elif len(cset) != 1:
                action = "escalate"
            else:
                action = "act"
            opt_hash = sha256(canonical_json(sorted([x["name"], x["description"]] for x in r["options"])))
            results.append({
                "probs": o["probs"],
                "null_probability": o["null"],
                "verdict": v,
                "witness": [dict(p, text=r["state"][p["start"]:p["end"]]) for p in pieces],
                "verifier": {
                    "sufficiency": suff,
                    "necessity": nec,
                    "necessity_required": nec_needed,
                    "witness_only_p": p_of(wo),
                    "witness_masked_p": p_of(wm),
                    "untrusted_masked_p": p_of(um),
                    "untrusted_influence": ui,
                    "provenance_allowed": prov_ok,
                    "verdict_class": cls,
                    "admissible": admissible,
                },
                "set": cset,
                "coverage_level": 1.0 - self.alpha if dom != "none" else None,
                "action": action,
                "receipt": {
                    "model_hash": getattr(self.pred, "model_hash", ""),
                    "state_hash": sha256(r["state"]),
                    "option_set_hash": opt_hash,
                    "policy_table_id": self.policy["id"],
                    "policy_table_sha256": self.policy["_sha256"],
                    "conformal": {"alpha": self.alpha, "domain": dom, "threshold": q},
                    "logits": {
                        "full": o.get("logits"),
                        "witness_only": None if wo is None else wo.get("logits"),
                        "witness_masked": None if wm is None else wm.get("logits"),
                        "untrusted_masked": None if um is None else um.get("logits"),
                    },
                    "order": o.get("order"),
                },
                "input_tokens": sum(int(x.get("n_tokens", 0)) for x in (o, wo, wm, um) if x is not None),
            })
        return results


class BareEngine:
    """Probabilities only, from a baseline adapter: no null, verifier, conformal
    set or action. For diagnosis runs behind the same server."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.model_id = "kelter-bare-" + adapter.name.replace(":", "-")

    def decide(self, recs: List[Dict[str, Any]], canonical: bool = True) -> List[Dict[str, Any]]:
        return [{"probs": o["probs"]} for o in self.adapter.predict(recs, canonical=canonical)]


def load_engine(ckpt: str, alpha: float = 0.05):
    """KELTER_DEVICE overrides the device (default: cuda, else mps, else cpu; kelter.device).
    ckpt "qwen:<hf model id>" serves the zero-shot letter-logprob baseline as a BareEngine."""
    from kelter.device import default_device
    if ckpt.startswith("qwen:"):
        from kelter.eval.baselines.qwen_logprob import QwenLogprobAdapter
        return BareEngine(QwenLogprobAdapter(ckpt[len("qwen:"):], device=default_device()))
    from kelter.model import load_predictor
    return Engine(load_predictor(ckpt, device=default_device()), load_policy(), alpha)
