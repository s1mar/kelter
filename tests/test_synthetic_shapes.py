"""Offline tests for kelter.data.synthetic_shapes."""
import json
import re
import statistics
from collections import Counter

import pytest

from kelter.data import synthetic_shapes as ss
from kelter.data.schema import validate

N = 300


@pytest.fixture(scope="module")
def items():
    return {f: ss.generate_family(f, N, seed=1) for f in ss.FAMILIES}


def test_families_tuple():
    assert set(ss.FAMILIES) == {"routing", "intent_negation", "policy_exemption", "adequacy", "long_policy", "conflict"}
    assert ss.SOURCE == "synthetic_shapes"


@pytest.mark.parametrize("fam", ss.FAMILIES)
def test_validate_and_shape(items, fam):
    ids = set()
    for r in items[fam]:
        validate(r)
        assert r["source"] == ss.SOURCE and r["split"] == "train" and r["domain"] == "policy"
        assert r["meta"]["family"] == fam
        assert set(r["meta"]["verdict_class"]) == {o["name"] for o in r["options"]}
        assert set(r["meta"]["verdict_class"].values()) == {"benign"}
        ids.add(r["id"])
    assert len(ids) == N


@pytest.mark.parametrize("fam", ss.FAMILIES)
def test_witness_spans_nonempty(items, fam):
    for r in items[fam]:
        if r["witness"] is not None:
            assert r["witness"]
            for s, e in r["witness"]:
                assert r["state"][s:e].strip()


@pytest.mark.parametrize("fam", ss.FAMILIES)
def test_determinism(fam):
    a = ss.generate_family(fam, 20, seed=5)
    b = ss.generate_family(fam, 20, seed=5)
    c = ss.generate_family(fam, 20, seed=6)
    assert a == b
    assert a != c


def test_generate_round_robin_and_determinism():
    a = ss.generate(60, seed=3)
    assert a == ss.generate(60, seed=3)
    assert [r["meta"]["family"] for r in a[:12]] == list(ss.FAMILIES) * 2
    for r in a:
        validate(r)


@pytest.mark.parametrize("fam", ["policy_exemption", "adequacy"])
def test_binary_balance(items, fam):
    c = Counter(r["gold"] for r in items[fam])
    assert set(c) == {"yes", "no"}
    frac = c["yes"] / N
    assert 0.3 <= frac <= 0.7, c


def test_routing_catch_all_rate_and_layout(items):
    rs = items["routing"]
    rate = sum(r["gold"] == r["meta"]["catch_all"] for r in rs) / N
    assert 0.08 <= rate <= 0.25, rate
    for r in rs:
        n = len(r["options"])
        assert 6 <= n <= 9
        assert (r["witness"] is None) == (r["gold"] == r["meta"]["catch_all"])
    listed = sum(r["meta"]["handlers_in_state"] for r in rs) / N
    assert 0.25 <= listed <= 0.55
    for r in rs:
        if r["meta"]["handlers_in_state"]:
            assert '"handlers"' in r["state"]
            assert r["provenance"][0][2] == "system"


def test_intent_negation(items):
    rs = items["intent_negation"]
    other = sum(r["gold"] == "other" for r in rs) / N
    assert 0.15 <= other <= 0.45
    for r in rs:
        assert (r["witness"] is None) == (r["gold"] == "other")
        assert r["options"][-1]["name"] == "other"


def _words(s):
    return len(s.split())


def _json_ok(kind, resp, keys):
    try:
        v = json.loads(resp)
    except ValueError:
        return False
    if kind == "format_array":
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if kind == "format_object":
        return isinstance(v, dict) and set(v) == set(keys)
    if kind == "format_number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    raise AssertionError(kind)


def test_adequacy_independent_recheck(items):
    seen = Counter()
    for r in items["adequacy"]:
        m = r["meta"]
        resp = m["response"]
        assert r["state"].rstrip().endswith(resp)
        chk = m["check"]
        seen[chk] += 1
        if chk == "list_all":
            tail = resp.rsplit(":", 1)[-1].strip().rstrip(".")
            got = sorted(x.strip().lower() for x in tail.split(",") if x.strip())
            want = sorted(x.lower() for x in m["reference_items"])
            ok = got == want
        elif chk.startswith("format_"):
            ok = _json_ok(chk, resp, m["required_keys"])
        elif chk == "word_count":
            ok = _words(resp) == m["target_words"]
            assert f"exactly {m['target_words']} word" in r["state"]
        elif chk == "yes_no":
            first = re.match(r"\s*(yes|no)\b", resp.lower()).group(1)
            ok = (first == "yes") == m["truth"]
        else:
            raise AssertionError(chk)
        assert r["gold"] == ("yes" if ok else "no"), (chk, resp)
    assert set(seen) == {"list_all", "format_array", "format_object", "format_number", "word_count", "yes_no"}


def test_long_policy(items):
    rs = items["long_policy"]
    long_enough = sum(len(r["state"]) >= 4000 for r in rs) / N
    assert long_enough >= 0.9, long_enough
    assert max(len(r["state"]) for r in rs) <= 12000
    with_amend = 0
    for r in rs:
        dec = r["meta"]["decisive_amendment"]
        if dec is not None:
            with_amend += 1
            assert any(dec in r["state"][s:e] for s, e in r["witness"])
        assert "AMENDMENT" in r["state"].upper()
        assert r["state"].rstrip().splitlines()[-1].startswith("- ")
        assert r["provenance"][0][2] == "retrieved_doc" and r["provenance"][-1][2] == "tool_output"
        assert 4 <= len(r["options"]) <= 6
    assert with_amend / N > 0.3
    c = Counter(r["gold"] for r in rs)
    assert len(c) >= 4 and "prohibited" in c


def test_conflict(items):
    rs = items["conflict"]
    c = Counter(r["gold"] == "cannot_determine" for r in rs)
    assert c[True] > 0 and c[False] > 0
    rule = sum(r["meta"]["has_rule"] for r in rs) / N
    assert 0.45 <= rule <= 0.75
    for r in rs:
        assert r["options"][-1]["name"] == "cannot_determine"
        assert (r["witness"] is None) == (r["gold"] == "cannot_determine")
        if not r["meta"]["has_rule"]:
            assert r["gold"] == "cannot_determine"


def test_no_banned_words_or_dashes(items):
    banned = [chr(0x2014), "furthermore", "moreover", "notably", "delve"]
    for fam, rs in items.items():
        for r in rs:
            low = r["state"].lower()
            assert not any(b in low for b in banned)
