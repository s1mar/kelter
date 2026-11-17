"""Faster masking, non-finite numbers and question ids. No GPU, no network."""
import json
import random
import time
from pathlib import Path

import pytest

from kelter.data import augment
from kelter.data.augment import MASK, _merge, _type_at

ROOT = Path(__file__).resolve().parents[1]


def _mask_reference(rec, spans):
    """The earlier implementation, verbatim: a rescan of all provenance for every merged span."""
    import copy
    state = rec["state"]
    L = len(state)
    prov = rec["provenance"]
    merged = _merge([(max(0, s), min(L, e)) for s, e in spans])
    pieces = []

    def emit(a, b):
        for s, e, t in prov:
            lo, hi = max(a, s), min(b, e)
            if lo < hi:
                pieces.append((state[lo:hi], t))

    pos = 0
    for a, b in merged:
        emit(pos, a)
        pieces.append((MASK, _type_at(prov, a)))
        pos = b
    emit(pos, L)
    new_state = "".join(t for t, _ in pieces)
    new_prov, p = [], 0
    for text, t in pieces:
        new_prov.append([p, p + len(text), t])
        p += len(text)

    def delta(x):
        return sum(len(MASK) - (b - a) for a, b in merged if b <= x)

    new_wit = None
    if rec.get("witness") is not None:
        new_wit = []
        for s, e in rec["witness"]:
            cur = s
            keep = []
            for a, b in merged:
                if b <= cur:
                    continue
                if a >= e:
                    break
                if a > cur:
                    keep.append((cur, a))
                cur = max(cur, b)
            if cur < e:
                keep.append((cur, e))
            for x, y in keep:
                new_wit.append([x + delta(x), y + delta(y)])
        new_wit = new_wit or None
    out = copy.deepcopy(rec)
    out["state"] = new_state
    out["provenance"] = new_prov
    out["witness"] = new_wit
    return out


def _random_rec(rng, canonical):
    n = rng.randint(1, 120)
    state = "".join(rng.choice("abcdef \n") for _ in range(n))
    types_ = ["user", "tool_output", "system", "retrieved_doc", "unknown"]
    if canonical:  # contiguous, sorted, sometimes with gaps and empty spans
        cuts = sorted(rng.sample(range(n + 1), min(n + 1, rng.randint(1, 9))))
        prov = [[cuts[i], cuts[i + 1], rng.choice(types_)] for i in range(len(cuts) - 1)] or [[0, n, "user"]]
        if rng.random() < 0.3:
            prov.insert(rng.randint(0, len(prov)), [prov[0][0], prov[0][0], "system"])
            prov.sort(key=lambda x: x[0])
    else:  # any order, overlapping
        prov = [[a, a + rng.randint(0, 30), rng.choice(types_)] for a in (rng.randint(0, n) for _ in range(rng.randint(1, 6)))]
        rng.shuffle(prov)
    wit = None
    if rng.random() < 0.7:
        wit = [[a, a + rng.randint(1, 25)] for a in (rng.randint(0, n) for _ in range(rng.randint(0, 4)))]
    return {"state": state, "provenance": prov, "witness": wit, "id": "x"}


def test_mask_is_byte_identical_to_the_old_implementation():
    rng = random.Random(7)
    for i in range(1500):
        rec = _random_rec(rng, canonical=i % 3 != 0)
        spans = [(rng.randint(-5, len(rec["state"]) + 5), rng.randint(-5, len(rec["state"]) + 5)) for _ in range(rng.randint(0, 8))]
        got, want = augment._mask(rec, spans), _mask_reference(rec, spans)
        assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), (rec, spans)


def test_mask_is_fast_at_the_documented_limits():
    from kelter import server
    n = server.MAX_PROVENANCE_SPANS  # 10,000 spans over a state they tile exactly
    state = "w" * (20 * n)
    types_ = ["user", "tool_output"]
    prov = [[20 * i, 20 * i + 20, types_[i % 2]] for i in range(n)]
    rec = {"state": state, "provenance": prov, "witness": [[40 * i, 40 * i + 25] for i in range(n // 2)], "id": "x"}
    spans = [(40 * i + 3, 40 * i + 11) for i in range(n // 2)]  # 5,000 masked spans
    t0 = time.perf_counter()
    out = augment.mask_spans(rec, spans)
    took = time.perf_counter() - t0
    assert took < 3.0, f"{took:.1f} s"  # about 28 s before the change; well under a second after
    assert out["state"].count(MASK) == n // 2 and out["provenance"][-1][1] == len(out["state"])


# ---- 2 and 3: non-finite numbers and question ids
CHOICE = {"type": "choice", "instructions": "What is the tone?", "criteria": {"angry": "upset", "calm": "neutral"}}


def _client():
    from fastapi.testclient import TestClient

    from kelter import server
    from tests.fakes import FakeEngine
    return TestClient(server.create_app(FakeEngine()))


def _raw(c, body: bytes):
    return c.post("/v1/systemone", content=body, headers={"Content-Type": "application/json"})


@pytest.mark.parametrize("where", ["score", "state", "instructions", "nested"])
def test_non_finite_numbers_are_a_422(where):
    c = _client()
    for tok in (b"NaN", b"Infinity", b"-Infinity"):
        score = b'{"type":"score","instructions":"i","criteria":[%s,"b"]}' % (tok if where == "score" else b'"a"')
        question = {"score": score,
                    "state": json.dumps(CHOICE).encode(),
                    "instructions": b'{"type":"choice","instructions":%s,"criteria":{"a":"1","b":"2"}}' % tok,
                    "nested": b'{"type":"choice","instructions":"i","criteria":{"a":{"x":[1,{"y":%s}]},"b":"2"}}' % tok}[where]
        state = b"[1, " + tok + b"]" if where == "state" else b'"hello"'
        r = _raw(c, b'{"model":"m","state":' + state + b',"questions":{"q":' + question + b"}}")
        assert r.status_code == 422, (where, tok, r.status_code)
        assert "finite" in r.json()["detail"][0]["msg"]
    ok = _raw(c, b'{"model":"m","state":{"n":1.5},"questions":{"q":{"type":"score","instructions":"i","criteria":["a","b"]}}}')
    assert ok.status_code == 200


def test_question_ids_are_checked_and_errors_cannot_raise():
    from kelter import server
    c = _client()
    q = json.dumps(CHOICE).encode()
    r = _raw(c, b'{"model":"m","state":"x","questions":{"a\\ud800":' + q + b"}}")
    assert r.status_code == 422 and "UTF-8" in r.json()["detail"][0]["msg"]
    # a pydantic validation error whose loc holds the bad id: the handler must not raise either
    r = _raw(c, b'{"model":"m","state":"x","questions":{"a\\ud800":{"type":"nope"}}}')
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][:2] == ["body", "questions"]
    r = _raw(c, b'{"model":"m","state":"x","questions":{"":' + q + b"}}")
    assert r.status_code == 422
    r = c.post("/v1/systemone", json={"model": "m", "state": "x", "questions": {"i" * (server.MAX_ID_CHARS + 1): CHOICE}})
    assert r.status_code == 422 and str(server.MAX_ID_CHARS) in r.json()["detail"][0]["msg"]
    assert c.post("/v1/systemone", json={"model": "m", "state": "x", "questions": {"i" * server.MAX_ID_CHARS: CHOICE}}).status_code == 200
    # verdict_classes naming a bad id
    r = _raw(c, b'{"model":"m","state":"x","questions":{"q":' + q + b'},"verdict_classes":{"a\\ud800":{"angry":"destructive"}}}')
    assert r.status_code == 422
    assert "128" in (ROOT / "docs" / "typesafe-contract.md").read_text(encoding="utf-8")
