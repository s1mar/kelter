"""Split routing and its hash, the conformal threshold, and the evaluation helpers. No GPU, no network."""
import ast
import math
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---- 2: h01 and the split routing
def test_h01_is_pinned_on_fixed_ids():
    from kelter.data.build import h01
    pinned = {"a": 0.7913742705228213, "clinc150-test-001218": 0.7891508100529087, "boolq-train-000001": 0.9850918916019359,
              "synthetic_policy-0007": 0.238628656658956, "xscore": 0.7312560369566213,
              "civil_comments-train-000042score": 0.5134217726330789}
    for s, v in pinned.items():
        assert h01(s) == v, s
    assert 0.0 <= min(h01(f"i{k}") for k in range(2000)) and max(h01(f"i{k}") for k in range(2000)) <= 1.0


def test_h01_cuts_land_on_the_stated_side():
    """Ids whose h01 sits just under or over each cut: below 0.9 train, else dev; below 0.5 calib, else test; below
    0.33 (with the "score" salt) a civil_comments item becomes a score item."""
    from kelter.data.build import h01
    assert h01("id-547") < 0.33 <= h01("id-175706")        # 0.329995 and 0.330001
    assert h01("id-227092") < 0.5 <= h01("id-220853")      # 0.499998 and 0.500000
    assert h01("id-247614") < 0.9 <= h01("id-20273")       # 0.899999 and 0.900002


def test_the_routing_thresholds_in_build_are_the_stated_ones():
    """The routing is inline in build._build, so the cuts are checked where they are written: every comparison of an
    h01(...) call with a number uses 0.33, 0.5 or 0.9, in the expected direction. (Not covered: which records reach which
    branch, which needs the converters and their downloads.)"""
    tree = ast.parse((ROOT / "kelter" / "data" / "build.py").read_text(encoding="utf-8"))
    cuts = [(type(n.ops[0]).__name__, n.comparators[0].value) for n in ast.walk(tree)
            if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call) and getattr(n.left.func, "id", "") == "h01"]
    assert sorted(set(cuts)) == [("GtE", 0.5), ("Lt", 0.33), ("Lt", 0.5), ("Lt", 0.9)]
    assert [c for c in cuts if c == ("Lt", 0.9)] and len(cuts) == 8


def test_holdout_sources_and_family_are_fixed_and_disjoint_from_training():
    from kelter.data import build, synthetic
    assert build.HOLDOUT_FAMILY == "sharing"
    assert sorted(build.HOLDOUT_SOURCES) == ["arc_challenge", "clinc150"]
    assert sorted(build.TRAIN_SOURCES) == ["arc_easy", "banking77", "boolq", "civil_comments", "esnli", "massive", "mmlu", "mnli",
                                           "paws", "squad2"]
    assert not set(build.HOLDOUT_SOURCES) & set(build.TRAIN_SOURCES)
    assert build.HOLDOUT_FAMILY in synthetic.FAMILIES  # a real family, so held-out synthetic records exist
    source = (ROOT / "kelter" / "data" / "build.py").read_text(encoding="utf-8")
    # the manifest names the held-out sources the same way the routing does
    assert 'sorted(HOLDOUT_SOURCES) + [f"synthetic_policy_{HOLDOUT_FAMILY}"]' in source


# ---- 3: conformal_q
def test_conformal_q_by_hand():
    from kelter.calibrate import conformal_q
    scores = list(range(1, 20))                       # n = 19
    assert conformal_q(scores, 0.1) == 18             # ceil(20 * 0.9) = 18: the 18th smallest
    assert conformal_q(scores[::-1], 0.1) == 18       # order does not matter
    assert conformal_q(list(range(1, 10)), 0.05) == 1.0  # n = 9: k = ceil(10 * 0.95) = 10 > n, so no valid quantile
    assert conformal_q([0.3, 0.3, 0.3, 0.9], 0.25) == 0.9   # n = 4: k = ceil(5 * 0.75) = 4, the largest
    assert conformal_q([0.3, 0.3, 0.3, 0.3], 0.25) == 0.3   # ties
    s = [0.05 * i for i in range(1, 101)]
    qs = [conformal_q(s, a) for a in (0.01, 0.05, 0.1, 0.5)]
    assert qs == sorted(qs, reverse=True)             # a smaller alpha asks for more coverage: a larger threshold


def test_conformal_set_keeps_a_label_at_exactly_the_threshold():
    from kelter.engine import NULL, conformal_set
    out = {"probs": {"a": 0.75, "b": 0.25}, "null": 0.0}
    assert conformal_set(out, 0.25) == ["a"]                  # 1 - 0.75 <= 0.25 keeps a; 1 - 0.25 = 0.75 does not keep b
    assert set(conformal_set(out, 0.75)) == {"a", "b"}        # 1 - 0.25 = 0.75 <= 0.75: the equal case is included
    assert set(conformal_set(out, 1.0)) == {"a", "b", NULL}   # and null (p = 0, 1 - p = 1) joins at q = 1
    assert conformal_set(out, 0.2499) == ["a"]                 # nothing passes: the fallback keeps the most likely label


# ---- 4: evaluation metrics
def _random_out(rng, null_head):
    names = ["a", "b", "c", "d"][: rng.randint(2, 4)]
    po = [rng.random() for _ in names]
    s = sum(po)
    out = {"probs": {n: p / s * (1 - 0.0) for n, p in zip(names, po)}, "null": rng.random()}
    if null_head:
        out["option_probs"] = dict(out["probs"])
        out["null_threshold"] = rng.random()
    return out


def test_run_verdict_agrees_with_the_engines():
    from kelter.engine import verdict_of
    from kelter.eval import run
    rng = random.Random(5)
    for null_head in (False, True):
        for _ in range(500):
            o = _random_out(rng, null_head)
            assert run.verdict(o) == verdict_of(o), o
    no_null = {"probs": {"a": 0.2, "b": 0.8}}  # a baseline without a null option can never answer null
    assert run.verdict(no_null) == "b" and run.verdict({"probs": {"a": 0.2, "b": 0.8}, "null": None}) == "b"
    assert run.forced({"probs": {"a": 0.1, "b": 0.2}, "null": 0.99}) == "b"


def test_null_score_gold_label_and_prob_matrix_by_hand():
    from kelter.eval import run
    assert run.null_score({"probs": {"a": 0.3, "b": 0.6}, "null": 0.4}) == 0.4
    assert run.null_score({"probs": {"a": 0.3, "b": 0.6}}) == pytest.approx(0.4)      # no null: 1 - top option
    assert run.null_score({"probs": {"a": 0.3, "b": 0.6}, "null": None}) == pytest.approx(0.4)
    assert run.gold_label({"null": True, "gold": None}) == run.NULL
    assert run.gold_label({"null": False, "gold": "b"}) == "b"
    assert run.gold_label({"null": False, "gold": {"a": 0.3, "b": 0.7}}) == "b"
    recs = [{"options": [{"name": "a"}, {"name": "b"}], "null": False, "gold": "b"},
            {"options": [{"name": "a"}, {"name": "b"}, {"name": "c"}], "null": True, "gold": None}]
    outs = [{"probs": {"a": 0.2, "b": 0.2}, "null": 0.1}, {"probs": {}, "null": None}]
    P, G = run.prob_matrix(recs, outs)
    P = [list(map(float, row)) for row in P]
    G = [int(g) for g in G]
    assert [round(x, 6) for x in P[0]] == [0.4, 0.4, 0.2, 0.0]   # renormalised to 1, padded to K = 3 options + 1
    assert P[1] == [0.25] * 4 and G == [1, 3]                     # a zero row is uniform; null gold is at len(options)
    P, _ = run.prob_matrix(recs[:1], [{"probs": {"a": 0.3}, "null": None}])
    assert [float(x) for x in P[0]] == [1.0, 0.0, 0.0]            # a missing option counts as 0; a missing null as 0


def test_attack_ids_pair_with_their_clean_item_through_rsplit():
    """eval.run pairs an attacked item with its clean one by cutting the last "~" suffix; the ids the real
    augmentations make carry exactly one."""
    from kelter.data import augment
    from kelter.data.schema import options_from_criteria
    rec = {"id": "boolq-test-000007", "source": "boolq", "split": "test", "state": "Customer wants a refund today.",
           "provenance": [[0, 30, "user"]], "question": {"type": "choice", "instructions": "Route it.",
                                                         "criteria": {"billing": "money", "other": "else", "tech": "bug"}},
           "options": options_from_criteria("choice", {"billing": "money", "other": "else", "tech": "bug"}), "gold": "billing",
           "null": False, "witness": [[0, 8]], "domain": "intent", "meta": {}}
    rng = random.Random(1)
    for new in (augment.plant_instruction(rec, rng, kind="nudge"), augment.plant_instruction(rec, rng, kind="forge"),
                augment.remove_answer(rec, rng)):
        assert new is not None and new["id"].count("~") == 1
        assert new["id"].rsplit("~", 1)[0] == rec["id"] == new["id"].split("~", 1)[0]
    assert "a~b~c".rsplit("~", 1)[0] == "a~b" != "a~b~c".split("~", 1)[0]  # the two cuts differ only for chained ids
