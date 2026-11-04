"""Build helpers."""
import random

from openverdict.data import build
from openverdict.data.schema import options_from_criteria, validate


def _rec(i, source="banking77", state=None, n_opts=20, gold="o0"):
    crit = {f"o{k}": f"Option {k}" for k in range(n_opts)}
    st = state or f"state number {i}"
    return {
        "id": f"{source}-train-{i:06d}", "source": source, "split": "train", "state": st,
        "provenance": [[0, len(st), "user"]],
        "question": {"type": "choice", "instructions": "Pick.", "criteria": crit},
        "options": options_from_criteria("choice", crit), "gold": gold, "null": False,
        "witness": None, "domain": "intent", "meta": {},
    }


def test_subsample_keeps_gold_and_validates():
    rng = random.Random(0)
    for i in range(50):
        r = build.subsample_options(_rec(i, gold=f"o{i % 20}"), 5, rng)
        validate(r)
        assert len(r["options"]) == 5
        assert r["gold"] in [o["name"] for o in r["options"]]


def test_subsample_noop_when_small():
    r = _rec(0, n_opts=3)
    assert build.subsample_options(r, 5, random.Random(0)) == r


def test_swap_state_is_null():
    r = build.swap_state(_rec(0), _rec(1, source="paws", state="other text"))
    validate(r)
    assert r["null"] and r["gold"] is None and r["state"] == "other text"


def test_synthetic_score_levels_and_validity():
    from openverdict.data import synthetic_score as ss
    assert ss.priority_level(False, False, 1, True) == 0
    assert ss.priority_level(False, False, 1, False) == 1
    assert ss.priority_level(False, False, 5, True) == 1
    assert ss.priority_level(False, False, 5, False) == 2
    assert ss.priority_level(True, False, 1, True) == 3
    assert ss.priority_level(False, True, 1, True) == 3
    assert ss.risk_level("public", "read", False) == 0
    assert ss.risk_level("internal", "read", False) == 1
    assert ss.risk_level("public", "write", False) == 2
    assert ss.risk_level("restricted", "read", False) == 3
    assert ss.risk_level("public", "read", True) == 3
    for r in ss.generate(300, seed=1):
        validate(r)
        for s, e in r["witness"] or []:
            assert r["state"][s:e].strip()


def test_civil_to_score_bins():
    from openverdict.data import synthetic_score as ss
    base = {"id": "c-1", "source": "civil_comments", "split": "train", "state": "hi",
            "provenance": [[0, 2, "user"]],
            "question": {"type": "noul", "instructions": "Toxic?", "criteria": {"true": "t", "false": "f"}},
            "options": options_from_criteria("noul", {"true": "t", "false": "f"}),
            "null": False, "witness": None, "domain": "toxicity", "meta": {}}
    for t, lvl in ((0.0, "0"), (0.19, "0"), (0.2, "1"), (0.5, "2"), (0.79, "2"), (0.8, "3"), (1.0, "3")):
        r = ss.civil_to_score({**base, "gold": {"yes": t, "no": 1 - t}})
        validate(r)
        assert r["gold"] == lvl


def test_mask_random_nonwitness_keeps_label_and_witness():
    st = "Premise: a dog runs fast. Hypothesis: an animal moves quickly."
    i = st.index("dog")
    r = _rec(0, state=st)
    r["witness"] = [[i, i + 3]]
    out = build.mask_random_nonwitness(r, random.Random(3))
    validate(out)
    assert out["gold"] == r["gold"] and not out["null"]
    assert "[MASK]" in out["state"]
    assert [out["state"][s:e] for s, e in out["witness"]] == ["dog"]
