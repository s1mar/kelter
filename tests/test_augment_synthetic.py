"""Offline tests for openverdict.data.synthetic and openverdict.data.augment."""
import copy
import random
from collections import Counter

import pytest

from openverdict.data import augment as A
from openverdict.data import synthetic as S
from openverdict.data.schema import options_from_criteria, validate

POOL = [{"name": "maybe", "description": "Cannot tell from the text."},
        {"name": "other", "description": "None of the above."},
        {"name": "allow", "description": "collides by name"},
        {"name": "unsure", "description": "Not enough information to say."}]


def make(segs, qtype, criteria, gold, null=False, instructions="Pick one."):
    state, prov, wit = S.assemble(segs)
    return {"id": "hand-1", "source": "hand", "split": "train", "state": state, "provenance": prov,
            "question": {"type": qtype, "instructions": instructions, "criteria": criteria},
            "options": options_from_criteria(qtype, criteria), "gold": gold, "null": null,
            "witness": wit or None, "domain": "policy", "meta": {"k": 1}}


def hand_records():
    segs = [("system", "Policy: refunds need a receipt."), ("user", "Refund please."),
            ("tool_output", f"Lookup: {S.mark('receipt: none')}; {S.mark('12 days ago')}.")]
    choice = make(segs, "choice", {"allow": "ok", "deny": "no", "escalate": "ask"}, "deny")
    noul = make(segs, "noul", {"true": "yes it is", "false": "no it is not"}, "no")
    score = make(segs, "score", ["bad", "meh", "good"], "2")
    nowit = make([("user", "plain text without witness")], "choice", {"a": "A", "b": "B", "c": "C"}, "a")
    return [choice, noul, score, nowit]


@pytest.fixture(scope="module")
def sample():
    return S.generate(200, seed=1)


def all_augs(rec, rng):
    outs = [A.permute_options(rec, rng), A.remove_answer(rec, rng),
            A.plant_instruction(rec, rng, "nudge"), A.plant_instruction(rec, rng, "forge"),
            A.add_distractors(rec, rng, POOL, 2), A.mask_witness(rec), A.witness_only(rec),
            A.mask_whole_state(rec), A.mask_untrusted(rec)]
    if rec["witness"]:
        outs.append(A.mask_spans(rec, [tuple(rec["witness"][0])]))
    return outs


def test_generated_validate_and_all_augs_validate(sample):
    rng = random.Random(0)
    fired = Counter()
    for rec in sample:
        validate(rec)
        for i, out in enumerate(all_augs(rec, rng)):
            if out is not None:
                validate(out)
                fired[i] += 1
    assert all(fired[i] > 0 for i in range(10))


def test_hand_records_validate_all_augs():
    rng = random.Random(1)
    for rec in hand_records():
        validate(rec)
        for out in all_augs(rec, rng):
            if out is not None:
                validate(out)


def test_inputs_not_mutated(sample):
    rng = random.Random(2)
    for rec in sample[:60] + hand_records():
        before = copy.deepcopy(rec)
        all_augs(rec, rng)
        assert rec == before


def test_plant_instruction_offsets(sample):
    rng = random.Random(3)
    n = 0
    for rec in sample + hand_records():
        for kind in ("nudge", "forge"):
            out = A.plant_instruction(rec, rng, kind)
            if out is None:
                continue
            n += 1
            atk = out["meta"]["attack"]
            s, e = atk["span"]
            assert atk["target"] != rec["gold"]
            assert atk["target"] in out["state"][s:e]
            assert out["state"][s:e].count(atk["target"]) >= 1
            assert out["gold"] == rec["gold"] and out["null"] == rec["null"]
            seg = [g for g in out["provenance"] if g[0] <= s < g[1]][0]
            assert seg[1] >= e
            assert seg[2] in (("tool_output", "retrieved_doc") if kind == "nudge" else ("user", "system"))
            assert out["meta"]["aug"] == "plant_instruction"
            old = [rec["state"][a:b] for a, b in (rec["witness"] or [])]
            new = [out["state"][a:b] for a, b in (out["witness"] or [])]
            assert old == new
    assert n > 100


def test_attack_text_exact(sample):
    rng = random.Random(4)
    for rec in sample[:80]:
        out = A.plant_instruction(rec, rng)
        if out is None:
            continue
        s, e = out["meta"]["attack"]["span"]
        text = out["state"][s:e]
        assert any(text == t.format(t=out["meta"]["attack"]["target"]) for t in A.ATTACK_TEMPLATES)
        assert out["state"].count(text) == 1


def test_mask_spans_witness_shift():
    rec = hand_records()[0]
    w0, w1 = rec["witness"]
    texts = [rec["state"][a:b] for a, b in rec["witness"]]
    # mask the policy segment (before witness): both witnesses survive
    out = A.mask_spans(rec, [(0, rec["provenance"][0][1])])
    validate(out)
    assert [out["state"][a:b] for a, b in out["witness"]] == texts
    assert out["state"].startswith(A.MASK)
    assert out["provenance"][0] == [0, len(A.MASK), "system"]
    # mask over the first witness only: it is dropped, second survives
    out = A.mask_spans(rec, [tuple(w0)])
    validate(out)
    assert [out["state"][a:b] for a, b in out["witness"]] == [texts[1]]
    # overlapping spans merge into one MASK
    out = A.mask_spans(rec, [(0, 10), (5, 20)])
    assert out["state"].count(A.MASK) == 1
    assert [out["state"][a:b] for a, b in out["witness"]] == texts


def test_mask_spans_on_samples(sample):
    rng = random.Random(5)
    for rec in sample[:100]:
        L = len(rec["state"])
        a = rng.randrange(0, L - 5)
        b = a + rng.randrange(1, 40)
        out = A.mask_spans(rec, [(a, min(b, L))])
        validate(out)
        surv = [rec["state"][s:e] for s, e in rec["witness"] or [] if e <= a or s >= min(b, L)]
        got = [out["state"][s:e] for s, e in out["witness"] or []]
        for t in surv:
            assert t in got


def test_mask_witness_and_only(sample):
    for rec in sample:
        if not rec["witness"]:
            assert A.mask_witness(rec) is None and A.witness_only(rec) is None
            continue
        mw = A.mask_witness(rec)
        validate(mw)
        assert mw["null"] and mw["gold"] is None and mw["witness"] is None and mw["meta"]["mask"] == "witness"
        for a, b in rec["witness"]:
            assert rec["state"][a:b] not in mw["state"] or rec["state"][a:b] in (
                rec["state"][:a] + rec["state"][b:])
        wo = A.witness_only(rec)
        validate(wo)
        assert wo["gold"] == rec["gold"] and wo["meta"]["mask"] == "witness_only"
        assert [wo["state"][a:b] for a, b in wo["witness"]] == [rec["state"][a:b] for a, b in rec["witness"]]
        kept = wo["state"]
        for a, b in wo["witness"]:
            kept = kept.replace(wo["state"][a:b], "", 1)
        assert set(kept.replace(A.MASK, "")) <= {"", "\n"} or kept.replace(A.MASK, "").strip() == ""


def test_mask_whole_state_and_untrusted():
    for rec in hand_records():
        w = A.mask_whole_state(rec)
        validate(w)
        assert w["state"] == A.MASK and w["null"] and w["witness"] is None and w["meta"]["mask"] == "all"
    choice = hand_records()[0]  # witness lives in tool_output
    mu = A.mask_untrusted(choice)
    validate(mu)
    assert mu["null"] and mu["gold"] is None and mu["witness"] is None
    assert A.mask_untrusted(hand_records()[3]) is None
    # witness in a user segment, untrusted tool_output elsewhere: label survives
    rec = make([("system", "Rule text."), ("user", f"I say {S.mark('the key fact')} here."),
                ("tool_output", "Noise from a tool.")], "choice", {"a": "A", "b": "B"}, "a")
    mu = A.mask_untrusted(rec)
    validate(mu)
    assert mu["gold"] == "a" and not mu["null"]
    assert [mu["state"][a:b] for a, b in mu["witness"]] == ["the key fact"]
    assert mu["state"].endswith(A.MASK) and mu["provenance"][-1][2] == "tool_output"


def test_option_augs():
    rec = hand_records()[0]
    rng = random.Random(7)
    p = A.permute_options(rec, rng)
    assert p["gold"] == rec["gold"] and set(p["question"]["criteria"]) == set(rec["question"]["criteria"])
    assert [o["name"] for o in p["options"]] == list(p["question"]["criteria"])
    assert list(p["question"]["criteria"]) != list(rec["question"]["criteria"])
    assert A.permute_options(hand_records()[1], rng) is None
    assert A.permute_options(hand_records()[2], rng) is None
    r = A.remove_answer(rec, rng)
    assert r["null"] and r["gold"] is None and r["witness"] is None
    assert r["meta"]["removed_gold"] == "deny" and "deny" not in r["question"]["criteria"]
    two = make([("user", "x y z")], "choice", {"a": "A", "b": "B"}, "a")
    assert A.remove_answer(two, rng) is None
    d = A.add_distractors(rec, rng, POOL, 5)
    names = [o["name"] for o in d["options"]]
    assert names[:3] == list(rec["question"]["criteria"]) and len(names) == 6
    assert names.count("allow") == 1 and d["gold"] == rec["gold"]
    assert A.add_distractors(hand_records()[1], rng, POOL, 2) is None
    assert d["id"].endswith("~dist")


def test_aug_meta_appends():
    rec = hand_records()[0]
    once = A.mask_whole_state(rec)
    twice = A.plant_instruction(A.permute_options(rec, random.Random(0)), random.Random(0))
    assert twice["meta"]["aug"] == ["permute_options", "plant_instruction"]
    assert once["meta"]["aug"] == "mask_whole_state"


# ---------------------------------------------------------------- synthetic

def test_decide_refund():
    base = {"limit": 30, "receipt": True, "days": 12, "defective": False}
    assert S.decide("refund", base) == "allow"
    assert S.decide("refund", {**base, "days": 45}) == "deny"
    assert S.decide("refund", {**base, "days": 45, "defective": True}) == "allow"
    assert S.decide("refund", {**base, "days": 30}) == "allow"
    assert S.decide("refund", {**base, "receipt": False}) == "deny"
    assert S.decide("refund", {**base, "receipt": None}) == "escalate"
    assert S.decide("refund", {**base, "days": None}) == "escalate"
    assert S.decide("refund", {**base, "days": None, "defective": True}) == "allow"
    assert S.decide("refund", {**base, "receipt": False, "days": None}) == "deny"


def test_decide_access():
    f = {"allowed": ["engineer", "sre", "analyst"], "role": "sre", "approval": True, "oncall": False,
         "resource": "x"}
    assert S.decide("access", f) == "allow"
    assert S.decide("access", {**f, "role": "intern"}) == "deny"
    assert S.decide("access", {**f, "role": "intern", "oncall": True}) == "allow"
    assert S.decide("access", {**f, "approval": False}) == "deny"
    assert S.decide("access", {**f, "approval": None}) == "escalate"


def test_decide_expense():
    f = {"limit": 100, "prohibited": ["gifts", "alcohol"], "amount": 100, "receipt": True, "category": "meals"}
    assert S.decide("expense", f) == "allow"
    assert S.decide("expense", {**f, "amount": 101}) == "deny"
    assert S.decide("expense", {**f, "category": "gifts"}) == "deny"
    assert S.decide("expense", {**f, "receipt": None}) == "escalate"
    assert S.decide("expense", {**f, "receipt": None, "amount": 500}) == "deny"


def test_decide_shell():
    ws = "/srv/work/projA"
    f = {"workspace": ws, "command": "ls -l", "path": ws + "/src", "confirm": None}
    assert S.decide("shell", f) == "allow"
    assert S.decide("shell", {**f, "command": "rm -rf"}) == "escalate"
    assert S.decide("shell", {**f, "command": "rm -rf", "confirm": True}) == "allow"
    assert S.decide("shell", {**f, "command": "rm -rf", "confirm": False}) == "deny"
    assert S.decide("shell", {**f, "path": "/etc/hosts"}) == "deny"
    assert S.decide("shell", {**f, "path": ws + "-old/data"}) == "deny"
    assert S.decide("shell", {**f, "path": ws + "/../secrets/keys"}) == "deny"


def test_decide_sharing():
    f = {"domain": "acme.example", "restricted": ["health records", "access credentials"],
         "recipient": "a@acme.example", "dataclass": "product roadmap", "dpa": None}
    assert S.decide("sharing", f) == "allow"
    assert S.decide("sharing", {**f, "recipient": "a@acme.example.net"}) == "escalate"
    assert S.decide("sharing", {**f, "recipient": "a@acme.example.net", "dpa": False}) == "deny"
    assert S.decide("sharing", {**f, "recipient": "a@partner.example", "dpa": True}) == "allow"
    assert S.decide("sharing", {**f, "dataclass": "health records", "dpa": False}) == "deny"


def test_decide_deletion():
    assert S.decide("deletion", {"identity": True, "balance": 0}) == "allow"
    assert S.decide("deletion", {"identity": True, "balance": 40}) == "deny"
    assert S.decide("deletion", {"identity": False, "balance": None}) == "deny"
    assert S.decide("deletion", {"identity": True, "balance": None}) == "escalate"


def test_generate_deterministic():
    assert S.generate(60, seed=3) == S.generate(60, seed=3)
    assert S.generate(60, seed=3) != S.generate(60, seed=4)


def test_generated_structure(sample):
    assert len({r["id"] for r in sample}) == len(sample)
    fams = Counter(r["meta"]["family"] for r in sample)
    assert set(fams) == set(S.FAMILIES) and len(fams) >= 6
    for r in sample:
        assert r["source"] == S.SOURCE and r["domain"] == "policy" and r["split"] == "train"
        m = r["meta"]
        assert set(m["verdict_class"]) == {o["name"] for o in r["options"]}
        permit = "yes" if r["question"]["type"] == "noul" else "allow"
        want = "destructive" if m["family"] in ("shell", "deletion") else "benign"
        assert m["verdict_class"][permit] == want
        types = {t for _, _, t in r["provenance"]}
        assert "user" in types and "tool_output" in types
        if m["outcome"] == "escalate":
            assert r["witness"] is None and r["question"]["type"] == "choice" and r["gold"] == "escalate"
        else:
            assert r["witness"]
            assert m["decisive_conditions"]
            for a, b in r["witness"]:
                seg = [g for g in r["provenance"] if g[0] <= a < g[1]][0]
                assert seg[2] == "tool_output" and b <= seg[1]
        if r["question"]["type"] == "noul":
            assert r["gold"] == ("yes" if m["outcome"] == "allow" else "no")


def test_label_balance_n2000():
    recs = S.generate(2000, seed=0)
    for fam in S.FAMILIES:
        noul = [r for r in recs if r["meta"]["family"] == fam and r["question"]["type"] == "noul"]
        yes = sum(r["gold"] == "yes" for r in noul) / len(noul)
        assert 0.3 <= yes <= 0.7, (fam, yes)
        ch = [r for r in recs if r["meta"]["family"] == fam and r["question"]["type"] == "choice"]
        assert {r["gold"] for r in ch} == {"allow", "deny", "escalate"}
