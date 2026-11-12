"""HTTP server for the TypeSafe System One contract.

POST /v1/systemone   TypeSafe request and response, plus optional extension fields
GET  /v1/models      TypeSafe model list
See notes/typesafe-contract.md for the sources behind every field.

The engine is injected: `create_app(engine, variant)`. The module-level `app`
builds the real engine lazily on first request.

Variants (results/final.json lists the same runs): "default", Q1 (Qwen3-4B
with a LoRA adapter, r2 for the witness), and "lower-latency", the score
ensemble E1. Select one with --variant or env KELTER_VARIANT. An explicit --ckpt or env
KELTER_CKPT overrides the variant; a checkpoint that is not a shipped
variant is served as "custom:<name>". Every answer carries the variant name in
the extension field "variant".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from kelter.data import schema
from kelter.state import render_state

MAX_CHOICES = 255
MAX_LEVELS = 10
VARIANTS = {"default": "checkpoints/Q1", "lower-latency": "checkpoints/E1"}
DEFAULT_VARIANT = "default"
EXTENSION_FIELDS = ("null_probability", "witness", "verifier", "set", "coverage_level", "action", "receipt")


class ContractError(Exception):
    """A request problem reported as a TypeSafe style 422."""

    def __init__(self, loc: List[Union[str, int]], msg: str):
        super().__init__(msg)
        self.loc = loc
        self.msg = msg


class EngineError(Exception):
    """The engine returned something the server cannot map. Reported as 500."""


# ---------------------------------------------------------------- request models
class _Q(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instructions: Any = None


class NoulQ(_Q):
    type: Literal["noul"]
    criteria: Optional[Dict[str, Any]] = None


class ChoiceQ(_Q):
    type: Literal["choice"]
    criteria: Dict[str, Any] = Field(min_length=1, max_length=MAX_CHOICES)


class ScoreQ(_Q):
    type: Literal["score"]
    criteria: List[Any] = Field(min_length=1, max_length=MAX_LEVELS)


Question = Annotated[Union[NoulQ, ChoiceQ, ScoreQ], Field(discriminator="type")]


class ProvSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int
    end: int
    type: str


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    state: Union[str, Dict[str, Any], List[Any]]
    questions: Dict[str, Question] = Field(min_length=1)
    assets: Optional[Dict[str, Any]] = None
    provenance: Optional[List[ProvSpan]] = None  # extension; TypeSafe clients never send it
    # extension: {question_id: {option_name: verdict class}}, classes from policy_table.json
    verdict_classes: Optional[Dict[str, Dict[str, str]]] = None


# ---------------------------------------------------------------- request to records
def _text(v: Any) -> str:
    """Instruction and description text. Strings pass through, None is empty,
    other JSON values are rendered as JSON."""
    if v is None:
        return ""
    return render_state(v)


def build_provenance(spans: Optional[List[ProvSpan]], n: int) -> List[List[Any]]:
    """Segments covering [0, n) exactly. Gaps become 'unknown'. Overlaps and
    out-of-range spans raise ContractError."""
    if not spans:
        return [[0, n, "unknown"]]
    for i, sp in enumerate(spans):
        loc = ["body", "provenance", i]
        if sp.type not in schema.PROVENANCE_TYPES:
            raise ContractError(loc + ["type"], f"provenance type {sp.type!r} not in {list(schema.PROVENANCE_TYPES)}")
        if not (0 <= sp.start < sp.end <= n):
            raise ContractError(loc, f"provenance range [{sp.start}, {sp.end}) must satisfy 0 <= start < end <= {n} (rendered state length)")
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    out: List[List[Any]] = []
    pos = 0
    for sp in ordered:
        if sp.start < pos:
            raise ContractError(["body", "provenance"], f"provenance ranges overlap near offset {sp.start}")
        if sp.start > pos:
            out.append([pos, sp.start, "unknown"])
        out.append([sp.start, sp.end, sp.type])
        pos = sp.end
    if pos < n:
        out.append([pos, n, "unknown"])
    return out


def build_record(req_id: str, qid: str, q: Any, state: str, provenance: List[List[Any]],
                 verdict_classes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    loc = ["body", "questions", qid]
    instructions = _text(q.instructions)
    if not instructions.strip():
        raise ContractError(loc + ["instructions"], "instructions are required (non-empty) by this server")
    if q.type == "choice":
        criteria: Any = {str(k): _text(v) for k, v in q.criteria.items()}
    elif q.type == "score":
        criteria = [_text(v) for v in q.criteria]
    else:
        src = q.criteria or {}
        criteria = {k: _text(src[k]) for k in ("true", "false") if src.get(k) is not None}
    try:
        options = schema.options_from_criteria(q.type, criteria)
    except ValueError as e:
        raise ContractError(loc + ["criteria"], str(e))
    rec = {
        "id": f"{req_id}-{qid}",
        "source": "request",
        "split": "test",
        "state": state,
        "provenance": provenance,
        "question": {"type": q.type, "instructions": instructions, "criteria": criteria},
        "options": options,
        "gold": None,
        "null": False,
        "witness": None,
        "domain": "policy",
        "meta": {"verdict_class": dict(verdict_classes)} if verdict_classes else {},
    }
    unknown = set(verdict_classes or {}) - {o["name"] for o in options}
    if unknown:
        raise ContractError(["body", "verdict_classes", qid], f"verdict_classes name unknown options {sorted(unknown)}")
    try:
        # schema.validate demands a gold unless null is set. Requests have none, so
        # check a copy with a placeholder gold; the engine gets the record as built.
        schema.validate({**rec, "gold": options[0]["name"]})
    except ValueError as e:
        raise ContractError(loc, f"invalid question: {e}")
    return rec


# ---------------------------------------------------------------- engine result to answer
def _normalized(result: Dict[str, Any], names: List[str]) -> List[float]:
    probs = result.get("probs")
    if not isinstance(probs, dict):
        raise EngineError("engine result has no 'probs' dict")
    unknown = set(probs) - set(names)
    if unknown:
        raise EngineError(f"engine probs name unknown options {sorted(unknown)}")
    vals = [float(probs.get(n, 0.0)) for n in names]
    if min(vals) < 0:
        raise EngineError("engine returned a negative probability")
    tot = sum(vals)
    if tot <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / tot for v in vals]


def choice_confidence(p: List[float]) -> float:
    """(p_max - 1/n) / (1 - 1/n); 1.0 for a single option."""
    n = len(p)
    if n == 1:
        return 1.0
    return max(0.0, (max(p) - 1.0 / n) / (1.0 - 1.0 / n))


def score_confidence(p: List[float]) -> float:
    """max(0, 1 - sum_i p_i |i - m| / MAD_unif), m the first most likely level,
    MAD_unif = (1/n) sum_i |i - (n-1)/2|."""
    n = len(p)
    if n == 1:
        return 1.0
    m = max(range(n), key=p.__getitem__)
    mad_unif = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - sum(pi * abs(i - m) for i, pi in enumerate(p)) / mad_unif)


def build_answer(q: Any, rec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    names = [o["name"] for o in rec["options"]]
    p = _normalized(result, names)
    if q.type == "choice":
        ans: Dict[str, Any] = {
            "type": "choice",
            "choice": names[max(range(len(p)), key=p.__getitem__)],
            "confidence": choice_confidence(p),
            "probabilities": dict(zip(names, p)),
        }
    elif q.type == "noul":
        ans = {"type": "noul", "noul": p[names.index("yes")]}
    else:
        ans = {
            "type": "score",
            "score": sum(i * pi for i, pi in enumerate(p)),
            "confidence": score_confidence(p),
            "legend": {str(i): v for i, v in enumerate(q.criteria)},
            "probabilities": dict(zip(names, p)),
        }
    for k in EXTENSION_FIELDS:
        if k in result:
            ans[k] = result[k]
    return ans


# ---------------------------------------------------------------- app
def resolve_variant(variant: Optional[str] = None, ckpt: Optional[str] = None) -> tuple:
    """(variant name, checkpoint path). An explicit ckpt wins over the variant."""
    if ckpt:
        norm = Path(ckpt).as_posix().rstrip("/")
        for name, path in VARIANTS.items():
            if norm == path or norm.endswith("/" + path):
                return name, ckpt
        return f"custom:{Path(ckpt).name}", ckpt
    name = variant or DEFAULT_VARIANT
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose one of {sorted(VARIANTS)}")
    return name, VARIANTS[name]


class MissingWeights(SystemExit):
    """Raised before loading when a checkpoint's files are absent, with what to do next."""


def check_weights(ckpt: str, root: Optional[Path] = None) -> None:
    """A one-line error naming the missing files and the next step, instead of a traceback."""
    if ckpt.startswith("qwen:"):
        return
    base = Path(root) if root else Path.cwd()
    ck = Path(ckpt) if Path(ckpt).is_absolute() else base / ckpt
    missing = []

    def shown(x: Path) -> str:
        try:
            return x.relative_to(base).as_posix()
        except ValueError:
            return x.as_posix()

    cfg = ck / "config.json"
    if not cfg.exists():
        missing.append(shown(ck / "config.json"))
    else:
        c = json.loads(cfg.read_text(encoding="utf-8"))
        if "llm" in c and not (ck / "adapter").exists():
            missing.append(shown(ck / "adapter"))
        wit = c.get("witness_ckpt")
        if wit and not ((base / wit) / "config.json").exists():
            missing.append(shown(base / wit / "config.json"))
    if missing:
        raise MissingWeights(f"kelter: weights not found ({', '.join(missing)}). Download the release weights into "
                             f"checkpoints/ as the README's quick start says, or pass --ckpt <checkpoint dir>.")


LLM_GPU_GB = 9.0  # free GPU memory an LLM variant needs: peak 8.82 GB at 8,000 tokens, plus margin


def _cuda_free_gb():
    """(free, total) GiB on GPU 0, or None without a usable CUDA GPU. Device-wide use comes from
    nvidia-smi, since torch.cuda.mem_get_info can report memory other processes hold as free when
    the Windows driver oversubscribes it; torch's figure is the fallback."""
    import subprocess
    import torch
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
                              "-i", "0"], capture_output=True, text=True, timeout=10).stdout
        used, total = (float(x) / 1024 for x in out.strip().split(","))
        return total - used, total
    except (OSError, ValueError, subprocess.SubprocessError):
        free, total = torch.cuda.mem_get_info(0)
        return free / 2**30, total / 2**30


def check_device(ckpt: str, root: Optional[Path] = None) -> None:
    """Before loading an LLM variant: a one-line error when there is no usable GPU or too little
    free GPU memory, saying what to run instead. KELTER_DEVICE set explicitly skips the check."""
    if ckpt.startswith("qwen:") or os.environ.get("KELTER_DEVICE"):
        return
    base = Path(root) if root else Path.cwd()
    ck = Path(ckpt) if Path(ckpt).is_absolute() else base / ckpt
    cfg = json.loads((ck / "config.json").read_text(encoding="utf-8"))
    if "llm" not in cfg:
        return
    gpu = _cuda_free_gb()
    if gpu is None:
        raise MissingWeights(f"kelter: {ckpt} runs a 4B language model and needs an NVIDIA GPU with about "
                             f"{LLM_GPU_GB:.0f} GB free, and no usable CUDA GPU was found. Run the CPU variant with "
                             "`python -m kelter.server --variant lower-latency`, or set KELTER_DEVICE=cpu to run this one "
                             "on the CPU anyway (much slower).")
    free, total = gpu
    if free < LLM_GPU_GB:
        raise MissingWeights(f"kelter: {ckpt} needs about {LLM_GPU_GB:.1f} GB of free GPU memory; {free:.1f} GB of "
                             f"{total:.1f} GB is free. Close other programs using the GPU (`nvidia-smi` lists them) or run "
                             "`python -m kelter.server --variant lower-latency`.")


def _load(ckpt: str):
    """check_weights, check_device, then load_engine, turning a GPU out-of-memory error into one line."""
    check_weights(ckpt)
    check_device(ckpt)
    from kelter.engine import load_engine
    try:
        return load_engine(ckpt)
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        raise MissingWeights(f"kelter: the GPU ran out of memory while loading {ckpt}. Close other programs using the "
                             "GPU (`nvidia-smi` lists them) or run `python -m kelter.server --variant lower-latency`.") from None


def create_app(engine: Any, variant: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="kelter", description="TypeSafe System One compatible server")
    lock = threading.Lock()  # one decide() at a time; the engine owns the GPU
    release_date = str(getattr(engine, "release_date", None) or date.today().isoformat())

    @app.exception_handler(ContractError)
    async def _contract_error(_, exc: ContractError):
        return JSONResponse({"detail": [{"loc": exc.loc, "msg": exc.msg, "type": "value_error"}]}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_, exc: RequestValidationError):
        detail = [{"loc": list(e.get("loc", [])), "msg": str(e.get("msg", "")), "type": str(e.get("type", "value_error"))}
                  for e in exc.errors()]
        return JSONResponse({"detail": detail}, status_code=422)

    @app.exception_handler(EngineError)
    async def _engine_error(_, exc: EngineError):
        return JSONResponse({"detail": f"engine error: {exc}"}, status_code=500)

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        if req.assets:
            raise ContractError(["body", "assets"], "assets (images) are not supported by this server")
        req_id = uuid.uuid4().hex
        state = render_state(req.state)
        provenance = build_provenance(req.provenance, len(state))
        qids = list(req.questions)
        recs = [build_record(req_id, qid, req.questions[qid], state, provenance, (req.verdict_classes or {}).get(qid)) for qid in qids]
        with lock:
            results = engine.decide(recs)
        if not isinstance(results, list) or len(results) != len(recs):
            raise EngineError("engine returned a different number of results than questions")
        answers = {qid: build_answer(req.questions[qid], rec, res) for qid, rec, res in zip(qids, recs, results)}
        if variant:
            for a in answers.values():
                a["variant"] = variant
        input_tokens = sum(int(r.get("input_tokens", 0)) for r in results)
        body = {
            "model": engine.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        return JSONResponse(jsonable_encoder(body), headers={"x-typesafe-request-id": req_id})

    @app.get("/v1/models")
    def models():
        return {"models": [{
            "name": engine.model_id,
            "description": "kelter System One decision model with evidence, certified null and action.",
            "release_date": release_date,
        }]}

    return app


class _LazyApp:
    """ASGI app that loads the real engine on first use."""

    def __init__(self) -> None:
        self._app: Optional[FastAPI] = None
        self._lock = threading.Lock()

    def _build(self) -> FastAPI:
        with self._lock:
            if self._app is None:
                name, ckpt = resolve_variant(os.environ.get("KELTER_VARIANT"), os.environ.get("KELTER_CKPT"))  # lazy: tests never load
                self._app = create_app(_load(ckpt), name)
            return self._app

    async def __call__(self, scope, receive, send):
        app = self._app or await asyncio.to_thread(self._build)
        await app(scope, receive, send)


app = _LazyApp()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Serve the TypeSafe System One contract.")
    ap.add_argument("--variant", default=os.environ.get("KELTER_VARIANT"), choices=sorted(VARIANTS))
    ap.add_argument("--ckpt", default=os.environ.get("KELTER_CKPT"), help="overrides --variant")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    import uvicorn
    name, ckpt = resolve_variant(args.variant, args.ckpt)
    uvicorn.run(create_app(_load(ckpt), name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
