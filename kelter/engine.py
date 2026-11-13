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
    from kelter import paths
    path = path or paths.asset("policy_table.json")
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
    """The predicted label for one model output: the best option, or NULL when the null wins.
    Output keys read: "probs" (option name -> probability), "null" (null score) and, for models with a
    separate null head (r6 and later), "null_threshold". With a null head the null wins when its score
    passes the threshold; without one it wins when it beats the best option's probability."""
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
        """The policy-table class of a verdict: the request's meta.verdict_class maps option name -> class
        name. A verdict with no entry, or a class the policy table does not define, is "default"."""
        vc = (rec.get("meta") or {}).get("verdict_class") or {}
        c = vc.get(verdict, "default")
        return c if c in self.policy["classes"] else "default"

    def _q(self, rec) -> tuple:
        """The conformal threshold q for a request, as (domain, q): the request's own domain if the
        checkpoint was calibrated for it, else the "_global" threshold, else ("none", 1.0)."""
        dom = rec.get("domain", "")
        if dom in self.conformal:
            return dom, self.conformal[dom]
        if "_global" in self.conformal:
            return "_global", self.conformal["_global"]
        return "none", 1.0  # no calibration: the set is every label, so nothing acts

    def decide(self, recs: List[Dict[str, Any]], canonical: bool = True) -> List[Dict[str, Any]]:
        """Decide a batch of requests. In: records with state, provenance [start, end, type] segments,
        options and optional meta.verdict_class and domain. Out: one result per record, in order, with
        probs, verdict, witness pieces, the verifier block, the conformal set, the action and the receipt.

        Four model passes per record, in this order (passes 2 to 4 are skipped when there is nothing to mask):
          1. full input: gives the verdict and the witness (the evidence spans)
          2. evidence only: everything except the witness is masked (sufficiency check)
          3. evidence masked: the witness is masked (necessity check)
          4. untrusted text masked: every segment of an untrusted provenance type is masked
        Action ladder, first rule that fits wins:
          null wins, or no admissible witness                     -> defer
          untrusted influence over the class budget                   -> escalate
          conformal set is not a single label (more than one option left) -> escalate
          otherwise                                                   -> act
        """
        full = self.pred.predict(recs, canonical=canonical)
        untrusted = set(self.policy["untrusted_types"])
        variants = []
        for rec, full_out in zip(recs, full):
            witness_spans = merge(full_out["witness"])
            n = len(rec["state"])
            if witness_spans:
                variants.append(mask_spans(rec, complement(witness_spans, n)))
                variants.append(mask_spans(rec, witness_spans))
            else:
                variants.extend([None, None])
            untrusted_spans = [[s, e] for s, e, t in rec["provenance"] if t in untrusted]
            variants.append(mask_spans(rec, untrusted_spans) if untrusted_spans else None)
        todo = [variant for variant in variants if variant is not None]
        outs = iter(self.pred.predict(todo, canonical=canonical)) if todo else iter([])
        passes = [next(outs) if variant is not None else None for variant in variants]

        results = []
        for i, (rec, full_out) in enumerate(zip(recs, full)):
            # Relies on exactly three variants per record (evidence only, evidence masked, untrusted masked),
            # appended in that order above; None marks a pass that was skipped.
            witness_only, witness_masked, untrusted_masked = passes[3 * i:3 * i + 3]
            verdict = verdict_of(full_out)
            win_p = full_out["null"] if verdict == NULL else (full_out.get("option_probs") or full_out["probs"])[verdict]
            witness_spans = merge(full_out["witness"])
            cls = self._class_of(rec, verdict) if verdict != NULL else "default"
            rule = self.policy["classes"][cls]
            pieces = [p for span in witness_spans for p in split_by_segments(span, rec["provenance"])]
            in_bounds = all(0 <= s < e <= len(rec["state"]) for s, e in witness_spans)
            prov_ok = bool(pieces) and all(p["provenance"] in rule["allowed_witness_types"] for p in pieces)
            suff = witness_only is not None and verdict_of(witness_only) == verdict
            nec = witness_masked is not None and verdict_of(witness_masked) != verdict
            nec_needed = rule["necessity"] == "gate"
            admissible = bool(witness_spans) and in_bounds and prov_ok and suff and (nec or not nec_needed)
            basis = "witness" if admissible else None  # what an act rests on; None when nothing is admissible

            def p_of(x):
                """Probability of the verdict in a pass output (the null score when the verdict is NULL); None if the pass was skipped."""
                if x is None:
                    return None
                if verdict == NULL:
                    return x["null"]
                return (x.get("option_probs") or x["probs"]).get(verdict, 0.0)

            ui = 0.0 if untrusted_masked is None else max(0.0, win_p - p_of(untrusted_masked))
            dom, q = self._q(rec)
            cset = conformal_set(full_out, q)
            if verdict == NULL:
                action = "defer"
            elif not admissible:
                action = "defer"
            elif ui > rule["untrusted_influence_budget"]:
                action = "escalate"
            elif len(cset) != 1:
                action = "escalate"
            else:
                action = "act"
            opt_hash = sha256(canonical_json(sorted([x["name"], x["description"]] for x in rec["options"])))
            results.append({
                "probs": full_out["probs"],
                "null_probability": full_out["null"],
                "verdict": verdict,
                "witness": [dict(p, text=rec["state"][p["start"]:p["end"]]) for p in pieces],
                "verifier": {
                    "sufficiency": suff,
                    "necessity": nec,
                    "necessity_required": nec_needed,
                    "witness_only_p": p_of(witness_only),
                    "witness_masked_p": p_of(witness_masked),
                    "untrusted_masked_p": p_of(untrusted_masked),
                    "untrusted_influence": ui,
                    "provenance_allowed": prov_ok,
                    "verdict_class": cls,
                    "admissible": admissible,
                    "basis": basis,
                },
                "set": cset,
                "coverage_level": 1.0 - self.alpha if dom != "none" else None,
                "action": action,
                "receipt": {
                    "model_hash": getattr(self.pred, "model_hash", ""),
                    "state_hash": sha256(rec["state"]),
                    "option_set_hash": opt_hash,
                    "policy_table_id": self.policy["id"],
                    "policy_table_sha256": self.policy["_sha256"],
                    "conformal": {"alpha": self.alpha, "domain": dom, "threshold": q},
                    "logits": {
                        "full": full_out.get("logits"),
                        "witness_only": None if witness_only is None else witness_only.get("logits"),
                        "witness_masked": None if witness_masked is None else witness_masked.get("logits"),
                        "untrusted_masked": None if untrusted_masked is None else untrusted_masked.get("logits"),
                    },
                    "order": full_out.get("order"),
                },
                "input_tokens": sum(int(x.get("n_tokens", 0)) for x in (full_out, witness_only, witness_masked, untrusted_masked) if x is not None),
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
    """KELTER_DEVICE overrides the device (default: cuda, else cpu; kelter.device).
    ckpt "qwen:<hf model id>" serves the zero-shot letter-logprob baseline as a BareEngine."""
    from kelter.device import default_device
    if ckpt.startswith("qwen:"):
        from kelter.eval.baselines.qwen_logprob import QwenLogprobAdapter
        return BareEngine(QwenLogprobAdapter(ckpt[len("qwen:"):], device=default_device()))
    from kelter.model import load_predictor
    return Engine(load_predictor(ckpt, device=default_device()), load_policy(), alpha)
