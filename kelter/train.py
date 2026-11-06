"""Train the three-head verdict model.

python -m kelter.train --run r1 [--epochs 1] [--max-train N] [--max-tokens 2048 --accum-steps 4]

Writes checkpoints/<run>/{model.pt, config.json, tokenizer/} and
results/train/<run>.json (config, loss curve summary, dev metrics).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from kelter.model import NULL, OPT, Featurizer, VerdictModel, collate

ROOT = Path(__file__).resolve().parents[1]
BUILT = ROOT / "data" / "built"


def load_jsonl(path: Path, limit=None, seed=0):
    recs = [json.loads(l) for l in path.open(encoding="utf-8")]
    if limit and len(recs) > limit:
        random.Random(seed).shuffle(recs)
        recs = recs[:limit]
    return recs


def target_for(rec, perm):
    """Soft target over K options (in perm order) plus null, last."""
    k = len(perm)
    y = np.zeros(k + 1, dtype=np.float32)
    if rec["null"]:
        y[k] = 1.0
    elif isinstance(rec["gold"], str):
        names = [rec["options"][j]["name"] for j in perm]
        y[names.index(rec["gold"])] = 1.0
    else:
        names = [rec["options"][j]["name"] for j in perm]
        for n, p in rec["gold"].items():
            y[names.index(n)] = p
    return y


def gold_index(rec, perm):
    """Index (in perm order) of the option the witness is conditioned on."""
    if rec["null"]:
        return len(perm)
    names = [rec["options"][j]["name"] for j in perm]
    if isinstance(rec["gold"], str):
        return names.index(rec["gold"])
    return names.index(max(rec["gold"], key=rec["gold"].get))


def make_batches(lengths, max_tokens, rng):
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    chunk = 4096
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in part:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > max_tokens:
                batches.append(cur)
                cur, m = [], lengths[i]
            cur.append(i)
            cur_max = m
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def optimizer_groups(plan, accum_steps):
    """Consecutive micro-batches, accum_steps per optimizer step; the last group may be short."""
    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")
    return [plan[i:i + accum_steps] for i in range(0, len(plan), accum_steps)]


def denominators(micro):
    """Counts over a whole accumulated group: records, answerable records and
    witness tokens. Each loss term is a sum divided by its group count, so the
    group's loss is the mean over the group, whatever the micro-batch split."""
    n_rec = sum(int(m["T"].shape[0]) for m in micro)
    n_ans = sum(int((m["T"][:, -1] < 0.5).sum()) for m in micro)
    n_wit = sum(int((m["b"]["state_mask"] & m["b"]["has_witness"].unsqueeze(1)).sum()) for m in micro)
    return n_rec, n_ans, n_wit


def step_loss(model, b, targets, cond_idx, w_witness, denoms=None):
    """denoms = (records, answerable records, witness tokens) of the group this
    micro-batch belongs to; None means the micro-batch is the whole group."""
    if denoms is None:
        denoms = denominators([{"b": b, "T": targets}])
    n_rec, n_ans, n_wit = denoms
    h = model.hidden(b["input_ids"], b["prov_ids"], b["attention_mask"])
    logits, m = model.option_logits(h, b["marker_pos"], b["marker_mask"])
    if model.null_head:
        # r6: options-only softmax on answerable records; BCE null head on every record
        y_null = targets[:, -1]
        ans = y_null < 0.5
        lo = logits[:, :-1]
        logp = torch.log_softmax(lo, -1)
        logp = torch.where(torch.isinf(logp), torch.zeros_like(logp), logp)
        per = -(targets[:, :-1] * logp).sum(-1)
        l_opt = per[ans].sum() / n_ans if ans.any() else per.sum() * 0.0
        l_null = F.binary_cross_entropy_with_logits(model.null_logit(m), y_null, reduction="sum") / n_rec
        l_opt = l_opt + l_null
    else:
        logp = torch.log_softmax(logits, -1)
        logp = torch.where(torch.isinf(logp), torch.zeros_like(logp), logp)
        l_opt = -(targets * logp).sum(-1).sum() / n_rec
    l_wit = torch.zeros((), device=h.device)
    hw = b["has_witness"]
    if hw.any():
        rows = torch.arange(h.shape[0], device=h.device)
        cond = m[rows, cond_idx]
        wl = model.witness_logits(h, cond)
        mask = b["state_mask"] & hw.unsqueeze(1)
        l_wit = F.binary_cross_entropy_with_logits(wl[mask], b["witness_labels"][mask], reduction="sum") / n_wit
    return l_opt + w_witness * l_wit, l_opt.detach(), l_wit.detach(), logits.detach()


def prepare(recs, pcs, bidx, feat, pad_id, device, rng=None):
    """One micro-batch: encodings, soft targets and witness conditioning index.
    rng shuffles each record's option order; None keeps the given order."""
    encs, tg, ci = [], [], []
    for i in bidx:
        r, pc = recs[i], pcs[i]
        perm = list(range(len(r["options"])))
        if rng is not None:
            rng.shuffle(perm)
        try:
            e = feat.assemble(pc, r["provenance"], perm, r["witness"])
        except ValueError:
            continue
        encs.append(e)
        tg.append(target_for(r, perm))
        ci.append(gold_index(r, perm) if r["witness"] else 0)
    if not encs:
        return None
    b = collate(encs, pad_id, device)
    K = b["marker_pos"].shape[1]
    T = torch.zeros((len(encs), K))
    for j, y in enumerate(tg):
        T[j, : len(y) - 1] = torch.from_numpy(y[:-1])
        T[j, K - 1] = float(y[-1])
    return {"b": b, "T": T.to(device), "ci": torch.tensor(ci, device=device)}


def accum_step(model, opt, micro, w_witness, clip=1.0):
    """One optimizer step over a group of micro-batches; gradients of each
    micro-batch's loss (already divided by the group counts) are summed."""
    denoms = denominators(micro)
    device_type = next(model.parameters()).device.type
    opt.zero_grad(set_to_none=True)
    lo_sum, lw_sum = 0.0, 0.0
    for mb in micro:
        with torch.autocast(device_type, dtype=torch.bfloat16):
            loss, lo, lw, _ = step_loss(model, mb["b"], mb["T"], mb["ci"], w_witness, denoms)
        loss.backward()
        lo_sum += float(lo)
        lw_sum += float(lw)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return lo_sum, lw_sum


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--w-witness", type=float, default=0.5)
    ap.add_argument("--null-head", action="store_true", help="r6: separate null head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infer-max-len", type=int, default=2048)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="micro-batches of --max-tokens per optimizer step")
    ap.add_argument("--spill-factor", type=float, default=2.0,
                    help="stop when a 200-step window runs slower than this times the first 200 steps")
    ap.add_argument("--built", default=str(BUILT), help="built corpus directory (data/built_r7 for r7's recipe)")
    ap.add_argument("--dev-batch", type=int, default=32,
                    help="records per batch in the end-of-training dev pass (8 for large)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    ck = ROOT / "checkpoints" / args.run
    ck.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    tok.add_special_tokens({"additional_special_tokens": [OPT, NULL]})
    tok.save_pretrained(str(ck / "tokenizer"))
    feat = Featurizer(tok, max_len=args.max_len)

    t0 = time.time()
    built = Path(args.built)
    train = load_jsonl(built / "train.jsonl", args.max_train, args.seed)
    dev = load_jsonl(built / "dev.jsonl", 3000, args.seed)
    pcs = []
    for i in range(0, len(train), 2048):
        pcs.extend(feat.pieces_batch(train[i:i + 2048]))
    dev_pcs = feat.pieces_batch(dev)
    print(f"tokenized {len(train)} train, {len(dev)} dev in {time.time() - t0:.0f}s", flush=True)

    model = VerdictModel(args.backbone, len(tok), null_head=args.null_head).cuda()
    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
                            weight_decay=0.01)

    def length(pc):
        return min(args.max_len, 4 + len(pc["instr"]) + sum(len(o) + 1 for o in pc["opts"]) + 8 + len(pc["sids"]))

    lengths = [length(pc) for pc in pcs]
    n_epochs_full = math.ceil(args.epochs)
    plan = []
    for ep in range(n_epochs_full):
        bs = make_batches(lengths, args.max_tokens, rng)
        frac = min(1.0, args.epochs - ep)
        plan.extend(bs[: int(len(bs) * frac)])
    # one optimizer step = accum_steps micro-batches; schedule, warmup, logging
    # and the recorded step count are in optimizer steps
    groups = optimizer_groups(plan, args.accum_steps)
    total = len(groups)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    print(f"{total} steps ({len(plan)} micro-batches, accum {args.accum_steps})", flush=True)

    model.train()
    log = []
    t0 = time.time()
    t_win, profiled = t0, None
    run_lo, run_lw = 0.0, 0.0
    for step, group in enumerate(groups):
        micro = [prepare(train, pcs, bidx, feat, tok.pad_token_id, "cuda", rng) for bidx in group]
        micro = [mb for mb in micro if mb is not None]
        if not micro:
            continue
        lo, lw = accum_step(model, opt, micro, args.w_witness)
        sched.step()
        run_lo = 0.98 * run_lo + 0.02 * lo if step else lo
        run_lw = 0.98 * run_lw + 0.02 * lw if step else lw
        if step % 200 == 0 or step == total - 1:
            now = time.time()
            el = now - t0
            win = (now - t_win) / 200 if step else None
            t_win = now
            peak = torch.cuda.max_memory_reserved() / 2**20
            print(f"step {step}/{total} opt_loss {run_lo:.4f} wit_loss {run_lw:.4f} "
                  f"{el / 60:.1f} min, eta {(total - step - 1) * el / max(step, 1) / 60:.1f} min, "
                  f"window {win if win is None else round(win, 3)} s/step, peak reserved {peak:.0f} MiB", flush=True)
            log.append({"step": step, "opt_loss": run_lo, "wit_loss": run_lw, "minutes": el / 60,
                        "window_s_per_step": win, "peak_reserved_mib": peak})
            if step == 200:
                profiled = win
                print(f"profiled {profiled:.3f} s/step over the first 200 steps", flush=True)
            elif profiled and win and step < total - 1 and win > args.spill_factor * profiled:
                # a slowdown past spill_factor x profile is shared-memory spill
                print(f"SPILL: window {win:.3f} s/step > {args.spill_factor} x profiled {profiled:.3f}; stopping",
                      flush=True)
                (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
                (ROOT / "results" / "train" / f"{args.run}_spill.json").write_text(json.dumps(
                    {"run": args.run, "step": step, "profiled_s_per_step": profiled, "window_s_per_step": win,
                     "log": log}, indent=2), encoding="utf-8")
                return 3

    model.eval()
    correct, n = 0, 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(dev), args.dev_batch):
            encs, golds = [], []
            for r, pc in zip(dev[i:i + args.dev_batch], dev_pcs[i:i + args.dev_batch]):
                e = feat.assemble(pc, r["provenance"])
                encs.append(e)
                golds.append(int(np.argmax(target_for(r, list(range(len(r["options"])))))))
            b = collate(encs, tok.pad_token_id, "cuda")
            h = model.hidden(b["input_ids"], b["prov_ids"], b["attention_mask"])
            logits, mm = model.option_logits(h, b["marker_pos"], b["marker_mask"])
            nlog = model.null_logit(mm) if model.null_head else None
            for j, e in enumerate(encs):
                k = len(e.opt_pos)
                if model.null_head:
                    pred = k if float(nlog[j]) > 0 else int(torch.argmax(logits[j, :k]))
                else:
                    lg = torch.cat([logits[j, :k], logits[j, -1:]])
                    pred = int(torch.argmax(lg))
                correct += pred == golds[j]
                n += 1
    dev_acc = correct / n
    torch.save(model.state_dict(), ck / "model.pt")
    model.encoder.config.save_pretrained(str(ck / "encoder_config"))
    mh = hashlib.sha256((ck / "model.pt").read_bytes()).hexdigest()
    cfg = {**vars(args), "model_hash": mh, "infer_max_len": args.infer_max_len, "temperature": {},
           "n_train": len(train), "steps": total, "micro_batches": len(plan), "minutes": (time.time() - t0) / 60,
           "profiled_s_per_step": profiled, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    (ck / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    res = {"run": args.run, "config": cfg, "dev_accuracy_incl_null_3000": dev_acc, "log": log}
    (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "train" / f"{args.run}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"dev_acc": dev_acc, "model_hash": mh}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
