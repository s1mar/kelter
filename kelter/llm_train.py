"""LoRA fine-tune of a causal LM as an kelter predictor.

python -m kelter.llm_train --name Q1 --base Qwen/Qwen3-4B-Instruct-2507 [--qlora]

Data: the train split of --corpus only (data/built by default). Records whose state carries the encoder's
[MASK] augmentation are left out. A seeded stratified sample takes at most
--cap-policy synthetic_policy records, --cap-score synthetic_score records, --cap-shapes synthetic_shapes records
(present in data/built_r5) and
--cap records of every other source (uniform within a source, so its null
share is kept). Each example shows its options in a random order (seeded) with
the null letter last (kelter.llm.user_text). Loss: cross entropy over the
example's n + 1 letter logits at the answer position. One pass, AdamW, cosine
schedule with warmup. Writes checkpoints/<name>/adapter, config.json and
results/train/<name>.json (loss curve, sample counts, timing).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sample(args):
    by_src = collections.defaultdict(list)
    for line in (ROOT / args.corpus / "train.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if "[MASK]" in r["state"]:
            continue
        if len(r["options"]) + 1 > 26:
            continue
        by_src[r["source"]].append(r)
    rng = random.Random(args.seed)
    out, counts = [], {}
    for src in sorted(by_src):
        cap = {"synthetic_policy": args.cap_policy, "synthetic_score": args.cap_score,
               "synthetic_shapes": args.cap_shapes}.get(src, args.cap)
        rows = by_src[src]
        pick = rng.sample(rows, min(cap, len(rows)))
        counts[src] = {"available": len(rows), "taken": len(pick), "null": sum(bool(r["null"]) for r in pick)}
        out += pick
    rng.shuffle(out)
    if args.long_frac:
        out = pad_long(out, by_src, args, counts)
    return out, counts


def pad_long(out, by_src, args, counts):
    """A seeded args.long_frac of the sample gets unrelated train-split passages
    (states of records from other sources) placed before and after its state until the state
    reaches a length drawn from args.long_chars; options, question and gold are unchanged.
    This trains length handling with distractor text, not long single documents."""
    lo, hi = (int(x) for x in args.long_chars.split(","))
    rng = random.Random(args.seed + 3)
    pool = {src: rows for src, rows in by_src.items()}
    srcs = sorted(pool)
    picks = set(rng.sample(range(len(out)), int(args.long_frac * len(out))))
    res, padded = [], collections.Counter()
    for i, r in enumerate(out):
        if i not in picks:
            res.append(r)
            continue
        target = rng.randint(lo, hi)
        before, after, n = [], [], len(r["state"])
        while n < target:
            src = rng.choice([x for x in srcs if x != r["source"]])
            t = rng.choice(pool[src])["state"]
            (before if rng.random() < 0.5 else after).append(t)
            n += len(t) + 2
        r = dict(r, state="\n\n".join(before + [r["state"]] + after))
        res.append(r)
        padded[r["source"]] += 1
    counts["_long_padded"] = dict(sorted(padded.items()))
    return res


SENTINEL = "checkpoints/r2/model.pt"  # a fixed large file whose sha256 is recorded in its config.json


def _hash_cached(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_unbuffered(path: Path):
    """sha256 read straight from disk, bypassing the file cache (Windows FILE_FLAG_NO_BUFFERING); None elsewhere.
    Opening a file this way also drops its stale cached pages."""
    import os
    if os.name != "nt":
        return None
    import ctypes
    import hashlib
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = wintypes.HANDLE
    k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                              wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    k.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                           ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    size = Path(path).stat().st_size
    hnd = k.CreateFileW(str(path), 0x80000000, 0x1, None, 3, 0x20000000 | 0x08000000, None)  # read, no buffering
    if hnd is None or hnd == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), "CreateFileW failed", str(path))
    n = 1 << 20
    buf = k.VirtualAlloc(None, n, 0x3000, 0x04)  # page-aligned, which meets the sector alignment rule
    try:
        h, left, got = hashlib.sha256(), size, wintypes.DWORD(0)
        while left > 0:
            if not k.ReadFile(hnd, buf, n, ctypes.byref(got), None):
                raise OSError(ctypes.get_last_error(), "ReadFile failed", str(path))
            if got.value == 0:
                break
            take = min(got.value, left)
            h.update(ctypes.string_at(buf, take))
            left -= take
        return h.hexdigest()
    finally:
        k.VirtualFree(buf, 0, 0x8000)
        k.CloseHandle(hnd)


def sentinel_probe(path: Path, expected: str) -> dict:
    """Hash the file through the file cache; on a mismatch also hash it
    straight from disk and then through the cache again, so a stale cached page (disk right, re-read right) can be
    told from a live flip in memory (disk right, re-read still wrong) and from a bad disk (disk wrong)."""
    cached = _hash_cached(path)
    rec = {"cached": cached[:12], "ok": cached == expected}
    if rec["ok"]:
        return rec
    disk = _hash_unbuffered(path)
    again = _hash_cached(path)
    rec.update({"unbuffered": disk[:12] if disk else None, "cached_again": again[:12]})
    if disk is None:
        rec["kind"] = "unknown (no unbuffered read on this platform)"
    elif disk != expected:
        rec["kind"] = "disk"
    elif again == expected:
        rec["kind"] = "stale cache, dropped by the unbuffered read"
    else:
        rec["kind"] = "memory (live flip)"
    return rec


def sentinel_check(path: Path, expected: str) -> bool:
    """Re-hash a fixed file; a mismatch means memory or disk returned bad data."""
    return _hash_cached(path) == expected


def save_state(path: Path, model, opt, sched, extra: dict) -> None:
    """Trainable parameters, optimizer, scheduler, RNG state and loop counters in
    one file, written to a temporary name and renamed so an interrupted save leaves the last one."""
    import os
    import torch
    state = {"params": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
             "opt": opt.state_dict(), "sched": sched.state_dict(), "extra": extra,
             "rng_cpu": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_state(path: Path, model, opt, sched) -> dict:
    """Inverse of save_state; returns the loop counters."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state["params"].items():
            named[n].copy_(v.to(named[n].device))
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    torch.set_rng_state(state["rng_cpu"])
    if state["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["extra"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base during training")
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--cap-policy", type=int, default=5000)
    ap.add_argument("--cap-score", type=int, default=3000)
    ap.add_argument("--cap-shapes", type=int, default=9000)
    ap.add_argument("--corpus", default="data/built", help="built corpus whose train split is sampled")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--long-frac", type=float, default=0.0, help="share of the sample padded to long inputs")
    ap.add_argument("--long-chars", default="4000,11000", help="padded state length range in characters")
    ap.add_argument("--infer-max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-tokens", type=int, default=4096, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug only: examples")
    ap.add_argument("--witness-ckpt", default="checkpoints/r2")
    ap.add_argument("--ckpt-every", type=int, default=50, help="optimizer steps between resume states; 0 is off")
    ap.add_argument("--power-mode", choices=["quiet", "balance"], default=None,
                    help="laptop power mode as read back before the run; recorded in the results file")
    ap.add_argument("--sentinel", default=SENTINEL, help="file re-hashed at every resume save; '' is off")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>/resume/state.pt")
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from kelter.llm import build_prompt, gold_slot, letter_ids

    torch.manual_seed(args.seed)
    recs, counts = sample(args)
    if args.limit:
        recs = recs[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.base)
    letters = letter_ids(tok)
    rng = random.Random(args.seed + 1)
    ex = []
    for r in recs:
        n = len(r["options"])
        order = list(range(n))
        if r["question"]["type"] != "score":
            rng.shuffle(order)
        ex.append((build_prompt(tok, r, order, args.max_len), gold_slot(r, order), n + 1))
    # length-bucketed micro-batches under batch_tokens, in a shuffled batch order
    idx = sorted(range(len(ex)), key=lambda i: len(ex[i][0]))
    batches, cur = [], []
    for i in idx:
        L = len(ex[i][0])
        if cur and L * (len(cur) + 1) > args.batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    random.Random(args.seed + 2).shuffle(batches)

    kw = {"dtype": torch.bfloat16}
    if args.qlora:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base, device_map={"": 0}, **kw)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(batches) / args.accum)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    letters_t = torch.tensor(letters, device="cuda")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    log, t0, run_loss, run_n, step = [], time.time(), 0.0, 0, 0
    state_path = ROOT / "checkpoints" / args.name / "resume" / "state.pt"
    sent_path = ROOT / args.sentinel if args.sentinel else None
    sent_hash = json.loads((sent_path.parent / "config.json").read_text(encoding="utf-8"))["model_hash"] if sent_path else None
    if sent_path:
        pr = sentinel_probe(sent_path, sent_hash)
        if not pr["ok"]:
            raise SystemExit(f"sentinel {args.sentinel} does not hash to its recorded sha256 before training ({pr}); not starting")
    start_bi, resumed = 0, []
    if args.resume and state_path.exists():
        x = load_state(state_path, model, opt, sched)
        start_bi, step, run_loss, run_n, log = x["next_bi"], x["step"], x["run_loss"], x["run_n"], x["log"]
        t0 = time.time() - x["elapsed_s"]
        resumed = x.get("resumed", []) + [step]
    (ROOT / "notes").mkdir(exist_ok=True)
    notes = (ROOT / "notes" / f"train_{args.name}.log").open("a" if resumed else "w", encoding="utf-8")
    if resumed:
        notes.write(json.dumps({"resumed_at_step": step, "next_batch": start_bi}) + "\n")
        notes.flush()
    for bi, b in enumerate(batches):
        if bi < start_bi:
            continue
        L = max(len(ex[i][0]) for i in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for row, i in enumerate(b):
            p = ex[i][0]
            ids[row, L - len(p):] = torch.tensor(p)
            att[row, L - len(p):] = 1
        out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), logits_to_keep=1)
        lg = out.logits[:, -1, :].float()[:, letters_t]
        mask = torch.full_like(lg, float("-inf"))
        for row, i in enumerate(b):
            mask[row, : ex[i][2]] = 0.0
        gold = torch.tensor([ex[i][1] for i in b], device="cuda")
        loss = torch.nn.functional.cross_entropy(lg + mask, gold)
        (loss / args.accum).backward()
        run_loss += float(loss) * len(b)
        run_n += len(b)
        if (bi + 1) % args.accum == 0 or bi == len(batches) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == steps:
                rec = {"step": step, "of": steps, "loss": run_loss / run_n, "examples": sum(len(x) for x in batches[: bi + 1]),
                       "elapsed_s": round(time.time() - t0, 1), "lr": sched.get_last_lr()[0],
                       "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                log.append(rec)
                notes.write(json.dumps(rec) + "\n")
                notes.flush()
                run_loss, run_n = 0.0, 0
            if args.ckpt_every and step % args.ckpt_every == 0 and step < steps:
                if sent_path:
                    pr = sentinel_probe(sent_path, sent_hash)
                    notes.write(json.dumps({"step": step, "sentinel": "ok" if pr["ok"] else "MISMATCH", **pr}) + "\n")
                    notes.flush()
                    if not pr["ok"]:
                        notes.close()
                        raise SystemExit(f"sentinel {args.sentinel} hashed differently at step {step} ({pr.get('kind')}): memory or disk returned "
                                         "bad data. Stopped without writing a resume state; the last good one is kept.")
                save_state(state_path, model, opt, sched,
                           {"next_bi": bi + 1, "step": step, "run_loss": run_loss, "run_n": run_n, "log": log,
                            "elapsed_s": time.time() - t0, "resumed": resumed})
    notes.close()
    ck = ROOT / "checkpoints" / args.name
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ck / "adapter"))
    if state_path.exists():
        state_path.unlink()  # the finished adapter supersedes the resume state
    cfg = {"llm": {"base": args.base, "qlora_train": bool(args.qlora), "max_len": args.max_len, "lora_r": args.r,
                   "lora_alpha": args.alpha, "lora_dropout": args.dropout, "token_budget": 8192},
           "infer_max_len": args.infer_max_len, "corpus": args.corpus, "witness_ckpt": args.witness_ckpt}
    (ck / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    res = {"name": args.name, "args": vars(args), "counts": counts, "n_examples": len(ex), "n_batches": len(batches),
           "steps": steps, "log": log, "resumed_at_steps": resumed, "train_s": round(time.time() - t0, 1),
           "power_mode": args.power_mode or "unrecorded",
           "command": "python -m kelter.llm_train " + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items()
                                                                     if v not in (None, False) and k not in ("qlora", "resume")) + (" --qlora" if args.qlora else "")}
    (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "train" / f"{args.name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_examples", "steps", "train_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
