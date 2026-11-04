"""Latency of a single request: 1k-token state, 5 options.

python -m openverdict.eval.latency --ckpt checkpoints/r1 [--runs 50] [--cpu-int8]

Measures (a) one model forward through the predictor and (b) a full engine
decision (four verifier passes in two batched forwards plus conformal and
policy). GPU (CUDA, or MPS on Apple Silicon; openverdict.device) in bf16; optionally CPU with dynamic int8 quantization of the
Linear layers. Writes results/eval/latency_<run>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from openverdict.data.schema import options_from_criteria
from openverdict.device import default_device, device_name, synchronize
from openverdict.engine import Engine, load_policy
from openverdict.eval.metrics import latency_percentiles
from openverdict.model import load_predictor

ROOT = Path(__file__).resolve().parents[2]
DEVICE = default_device()
SENTENCE = ("The customer reported that the order arrived late and the box was damaged, "
            "and the support log shows two earlier contacts about the same shipment. ")


def make_request(tok, target_tokens=1000):
    text = ""
    while len(tok(text, add_special_tokens=False)["input_ids"]) < target_tokens:
        text += SENTENCE
    ids = tok(text, add_special_tokens=False)["input_ids"][:target_tokens]
    state = tok.decode(ids)
    half = len(state) // 2
    crit = {
        "refund": "Issue a refund for the order.",
        "replace": "Send a replacement item.",
        "apologize": "Apologize and close the ticket.",
        "escalate": "Escalate to a human agent.",
        "ignore": "Take no action.",
    }
    return {
        "id": "latency", "source": "request", "split": "test", "state": state,
        "provenance": [[0, half, "user"], [half, len(state), "tool_output"]],
        "question": {"type": "choice", "instructions": "What should support do next?", "criteria": crit},
        "options": options_from_criteria("choice", crit), "gold": None, "null": False, "witness": None,
        "domain": "policy", "meta": {},
    }


def bench(fn, runs, warmup=5, sync=True):
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(runs):
        if sync:
            synchronize(DEVICE)
        t0 = time.perf_counter()
        fn()
        if sync:
            synchronize(DEVICE)
        ms.append((time.perf_counter() - t0) * 1000)
    return latency_percentiles(ms)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--cpu-int8", action="store_true")
    args = ap.parse_args(argv)
    res = {"ckpt": args.ckpt, "runs": args.runs, "state_tokens": 1000, "options": 5}
    pred = load_predictor(args.ckpt)
    req = make_request(pred.tok)
    res["sequence_tokens"] = pred.predict([req])[0]["n_tokens"]
    eng = Engine(pred, load_policy())
    res["device"] = DEVICE
    res["gpu"] = device_name(DEVICE)
    res["gpu_forward_ms"] = bench(lambda: pred.predict([req]), args.runs)
    res["gpu_engine_ms"] = bench(lambda: eng.decide([req]), args.runs)
    if args.cpu_int8:
        cpu = load_predictor(args.ckpt, device="cpu")
        if hasattr(cpu, "quantize_int8"):  # ensemble: every member
            cpu.quantize_int8()
        else:
            cpu.model = torch.ao.quantization.quantize_dynamic(cpu.model.float(), {torch.nn.Linear}, dtype=torch.qint8)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        res["cpu_threads"] = torch.get_num_threads()
        res["cpu_int8_forward_ms"] = bench(lambda: cpu.predict([req]), max(5, args.runs // 10), warmup=2, sync=False)
        ceng = Engine(cpu, load_policy())
        res["cpu_int8_engine_ms"] = bench(lambda: ceng.decide([req]), max(5, args.runs // 10), warmup=1, sync=False)
    res["command"] = f"python -m openverdict.eval.latency --ckpt {args.ckpt} --runs {args.runs}" + (
        " --cpu-int8" if args.cpu_int8 else "")
    out = ROOT / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"latency_{Path(args.ckpt).name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
