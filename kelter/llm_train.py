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
    (ROOT / "notes").mkdir(exist_ok=True)
    notes = (ROOT / "notes" / f"train_{args.name}.log").open("w", encoding="utf-8")
    for bi, b in enumerate(batches):
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
    notes.close()
    ck = ROOT / "checkpoints" / args.name
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ck / "adapter"))
    cfg = {"llm": {"base": args.base, "qlora_train": bool(args.qlora), "max_len": args.max_len, "lora_r": args.r,
                   "lora_alpha": args.alpha, "lora_dropout": args.dropout, "token_budget": 8192},
           "infer_max_len": args.infer_max_len, "corpus": args.corpus, "witness_ckpt": args.witness_ckpt}
    (ck / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    res = {"name": args.name, "args": vars(args), "counts": counts, "n_examples": len(ex), "n_batches": len(batches),
           "steps": steps, "log": log, "train_s": round(time.time() - t0, 1),
           "command": "python -m kelter.llm_train " + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items()
                                                                     if v not in (None, False) and k != "qlora") + (" --qlora" if args.qlora else "")}
    (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "train" / f"{args.name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_examples", "steps", "train_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
