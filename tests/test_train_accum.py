"""Gradient accumulation equivalence.

One group of 32 records takes one optimizer step two ways from the same
initial weights: as a single 8192-token batch, and as four 2048-token
micro-batches with accum_steps 4. Both run under bf16 autocast, as training
does. The parameter updates must agree within the bf16 tolerance below, for
the option-softmax path and the null-head path. A mis-weighted accumulation
(each micro-batch averaged over its own records, then divided by 4) is run
as a control and must miss the same tolerance, so the test can fail.

The backbone is a 2-layer ModernBERT built from a config with random
weights, so the test needs no download and runs on CPU.
"""
from __future__ import annotations

import copy
import random

import pytest
import torch
from transformers import ModernBertConfig

from kelter import train as T
from kelter.model import Featurizer, VerdictModel

# bf16 keeps 8 mantissa bits (relative step 2**-8 = 0.0039); summing four
# micro-batch gradients in a different order than one batch moves each
# gradient by a few such steps. Tolerance on the relative L2 difference of
# the full parameter update vector:
UPDATE_RTOL = 2e-2
# fp32 gradient check (autocast off): reordering sums only
GRAD_RTOL_FP32 = 1e-5

MAX_LEN = 256
N_MICRO, PER_MICRO = 4, 8      # 4 x (8 x 256) = 2048 tokens each; 32 x 256 = 8192 in one batch
VOCAB = 300
OPT_ID, NULL_ID, CLS_ID, SEP_ID, PAD_ID = 1, 2, 3, 4, 0


class _Tok:
    cls_token_id, sep_token_id, pad_token_id = CLS_ID, SEP_ID, PAD_ID

    def convert_tokens_to_ids(self, t):
        return {"[OPT]": OPT_ID, "[NULL]": NULL_ID}[t]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10, 11, 12, 13, 14]}


def _records(rng):
    """32 records with mixed lengths (every micro-batch holds one full-length
    record), uneven nulls and uneven witness counts across micro-batches, so
    per-micro-batch means and group means differ."""
    recs, pcs = [], []
    nulls_per_micro = [1, 0, 3, 6]
    wits_per_micro = [6, 2, 0, 4]
    for g in range(N_MICRO):
        for j in range(PER_MICRO):
            k = rng.randint(2, 4)
            opts = [{"name": f"o{g}_{j}_{q}", "description": "d"} for q in range(k)]
            instr = [rng.randrange(20, VOCAB) for _ in range(6)]
            opt_ids = [[rng.randrange(20, VOCAB) for _ in range(4)] for _ in range(k)]
            fixed = 1 + len(instr) + 1 + sum(len(o) + 1 for o in opt_ids) + 1 + 5 + 1  # cls instr sep, options, null block
            n_state = MAX_LEN - fixed - 1 if j == 0 else rng.randint(40, MAX_LEN - fixed - 1)
            sids = [rng.randrange(20, VOCAB) for _ in range(n_state)]
            offs = [[5 * t, 5 * t + 4] for t in range(n_state)]
            is_null = j < nulls_per_micro[g]
            wit = [[0, 5 * min(10, n_state)]] if j >= PER_MICRO - wits_per_micro[g] and not is_null else None
            recs.append({"options": opts, "null": is_null, "gold": None if is_null else opts[j % k]["name"],
                         "provenance": [[0, 5 * n_state, "retrieved_doc"]], "witness": wit})
            pcs.append({"instr": instr, "opts": opt_ids, "sids": sids, "offs": offs})
    return recs, pcs


def _model(null_head):
    conf = ModernBertConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=4, max_position_embeddings=MAX_LEN, pad_token_id=PAD_ID,
                            bos_token_id=CLS_ID, eos_token_id=SEP_ID, cls_token_id=CLS_ID, sep_token_id=SEP_ID,
                            global_attn_every_n_layers=1, local_attention=MAX_LEN)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        conf.save_pretrained(d)
        torch.manual_seed(0)
        m = VerdictModel("tiny", VOCAB, config_dir=d, null_head=null_head)
    # the provenance table starts at zero; give it values so its gradient path is exercised too
    torch.nn.init.normal_(m.prov.weight, std=0.02)
    return m.train()


def _update(model0, micro_lists, recs, pcs, feat, mis_weighted=False):
    m = copy.deepcopy(model0)
    before = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    micro = [T.prepare(recs, pcs, idx, feat, PAD_ID, "cpu") for idx in micro_lists]
    if mis_weighted:
        opt.zero_grad(set_to_none=True)
        for mb in micro:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss, *_ = T.step_loss(m, mb["b"], mb["T"], mb["ci"], 0.5)  # own counts
            (loss / len(micro)).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    else:
        T.accum_step(m, opt, micro, 0.5)
    return torch.cat([(p.detach() - b).flatten().float() for p, b in zip(m.parameters(), before)])


def _rel(a, b):
    return float((a - b).norm() / a.norm())


@pytest.mark.parametrize("null_head", [False, True], ids=["option_softmax", "null_head"])
def test_accumulated_step_equals_single_batch(null_head):
    rng = random.Random(0)
    recs, pcs = _records(rng)
    feat = Featurizer(_Tok(), max_len=MAX_LEN)
    groups = [list(range(g * PER_MICRO, (g + 1) * PER_MICRO)) for g in range(N_MICRO)]
    model0 = _model(null_head)

    # the batch shapes the decision names: padded token count of each micro-batch
    # is 2048, of the single batch 8192 (every micro-batch has one full-length record)
    for g in groups:
        assert T.prepare(recs, pcs, g, feat, PAD_ID, "cpu")["b"]["input_ids"].numel() == 2048
    assert T.prepare(recs, pcs, sum(groups, []), feat, PAD_ID, "cpu")["b"]["input_ids"].numel() == 8192

    single = _update(model0, [sum(groups, [])], recs, pcs, feat)
    accum = _update(model0, groups, recs, pcs, feat)
    wrong = _update(model0, groups, recs, pcs, feat, mis_weighted=True)

    assert single.norm() > 0
    assert _rel(single, accum) < UPDATE_RTOL, _rel(single, accum)
    # the control: averaging inside each micro-batch is not the group mean
    assert _rel(single, wrong) > UPDATE_RTOL, _rel(single, wrong)


class _NoStep:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


@pytest.mark.parametrize("null_head", [False, True], ids=["option_softmax", "null_head"])
def test_accumulated_gradient_equals_single_batch_fp32(null_head, monkeypatch):
    """Stricter companion: with autocast off, the summed gradient before clipping
    and Adam equals the single-batch gradient to fp32 rounding (GRAD_RTOL_FP32).
    Adam's first step divides by |g|, so tiny gradients make the update check
    above looser than this one."""
    real = torch.autocast
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: real(*a, **{**k, "enabled": False}))
    rng = random.Random(0)
    recs, pcs = _records(rng)
    feat = Featurizer(_Tok(), max_len=MAX_LEN)
    groups = [list(range(g * PER_MICRO, (g + 1) * PER_MICRO)) for g in range(N_MICRO)]
    model0 = _model(null_head)
    grads = []
    for parts in ([sum(groups, [])], groups):
        m = copy.deepcopy(model0)
        micro = [T.prepare(recs, pcs, idx, feat, PAD_ID, "cpu") for idx in parts]
        T.accum_step(m, _NoStep(), micro, 0.5, clip=float("inf"))
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert _rel(grads[0], grads[1]) < GRAD_RTOL_FP32, _rel(grads[0], grads[1])


def test_denominators_count_the_whole_group():
    rng = random.Random(1)
    recs, pcs = _records(rng)
    feat = Featurizer(_Tok(), max_len=MAX_LEN)
    groups = [list(range(g * PER_MICRO, (g + 1) * PER_MICRO)) for g in range(N_MICRO)]
    micro = [T.prepare(recs, pcs, idx, feat, PAD_ID, "cpu") for idx in groups]
    n_rec, n_ans, n_wit = T.denominators(micro)
    assert n_rec == 32 and n_ans == 32 - (1 + 0 + 3 + 6)
    # the witness loss runs over every state token of a record that has a witness
    assert n_wit == sum(len(pc["sids"]) for r, pc in zip(recs, pcs) if r["witness"])
    assert n_wit < sum(len(pc["sids"]) for pc in pcs)


def test_schedule_counts_optimizer_steps():
    """10 micro-batches at accum 4 are 3 optimizer steps; accum 1 keeps one step per batch."""
    plan = [[i] for i in range(10)]
    groups = T.optimizer_groups(plan, 4)
    assert [len(g) for g in groups] == [4, 4, 2] and sum(groups, []) == plan
    assert T.optimizer_groups(plan, 1) == [[b] for b in plan]
    with pytest.raises(ValueError):
        T.optimizer_groups(plan, 0)
