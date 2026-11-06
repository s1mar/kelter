"""Contract tests for the TypeSafe compatible server. No model or GPU is loaded."""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kelter.server import create_app
from tests.fakes import FakeEngine

ROOT = Path(__file__).resolve().parents[1]
JEVBENCH = ROOT / "third_party" / "jevbench"

STATE = "I was charged twice for my subscription this month. Please refund the extra charge."
CHOICE_Q = {"type": "choice", "instructions": "What is the tone?",
            "criteria": {"angry": "upset", "calm": "neutral", "excited": "eager"}}
SCORE_Q = {"type": "score", "instructions": "How urgent?",
           "criteria": ["Can wait", "This week", "Today"]}
NOUL_Q = {"type": "noul", "instructions": "Is this about billing?"}


@pytest.fixture()
def engine():
    return FakeEngine()


@pytest.fixture()
def client(engine):
    return TestClient(create_app(engine))


def post(client, questions, state=STATE, **extra):
    return client.post("/v1/systemone", json={"state": state, "model": "jev-latest", "questions": questions, **extra})


def assert_extensions(ans):
    assert isinstance(ans["null_probability"], float) and 0.0 <= ans["null_probability"] <= 1.0
    assert isinstance(ans["witness"], list)
    for w in ans["witness"]:
        assert set(w) == {"start", "end", "provenance", "text"}
    assert isinstance(ans["verifier"], dict)
    assert isinstance(ans["set"], list)
    assert 0.0 <= ans["coverage_level"] <= 1.0
    assert ans["action"] in ("act", "escalate", "defer")
    assert isinstance(ans["receipt"], dict)


# ------------------------------------------------------------------ per type
def test_choice_answer(client, engine):
    r = post(client, {"tone": CHOICE_Q})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == engine.model_id
    ans = body["answers"]["tone"]
    assert ans["type"] == "choice"
    assert ans["choice"] == "angry"  # FakeEngine puts the most weight on the first option
    probs = ans["probabilities"]
    assert list(probs) == ["angry", "calm", "excited"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in probs.values())
    n = 3
    expected = (max(probs.values()) - 1 / n) / (1 - 1 / n)
    assert ans["confidence"] == pytest.approx(expected)
    assert 0.0 <= ans["confidence"] <= 1.0
    assert ans["choice"] == max(probs, key=probs.get)
    assert_extensions(ans)
    assert "x-typesafe-request-id" in r.headers


def test_choice_confidence_extremes():
    certain = FakeEngine(probs_fn=lambda rec: {"angry": 1.0, "calm": 0.0, "excited": 0.0})
    uniform = FakeEngine(probs_fn=lambda rec: {"angry": 0.3, "calm": 0.3, "excited": 0.3})
    a = post(TestClient(create_app(certain)), {"q": CHOICE_Q}).json()["answers"]["q"]
    b = post(TestClient(create_app(uniform)), {"q": CHOICE_Q}).json()["answers"]["q"]
    assert a["confidence"] == pytest.approx(1.0)
    assert b["confidence"] == pytest.approx(0.0)
    assert b["probabilities"]["angry"] == pytest.approx(1 / 3)  # renormalized when engine mass is below 1


def test_choice_docs_example():
    eng = FakeEngine(probs_fn=lambda rec: {"angry": 0.6, "calm": 0.3, "excited": 0.1})
    ans = post(TestClient(create_app(eng)), {"q": CHOICE_Q}).json()["answers"]["q"]
    assert ans["confidence"] == pytest.approx(0.4)  # confidence page: (0.6, 0.3, 0.1) gives 0.4


def test_noul_answer(client):
    ans = post(client, {"billing": NOUL_Q}).json()["answers"]["billing"]
    assert ans["type"] == "noul"
    assert isinstance(ans["noul"], float) and 0.0 <= ans["noul"] <= 1.0
    assert "confidence" not in ans  # docs: noul answers carry no confidence
    # FakeEngine option order is [no, yes] with weights 2:1, so P(yes) = 1/3
    assert ans["noul"] == pytest.approx(1 / 3)
    assert_extensions(ans)


def test_noul_with_criteria_and_renormalization():
    eng = FakeEngine(probs_fn=lambda rec: {"no": 0.1, "yes": 0.3})  # null took 0.6
    q = {"type": "noul", "instructions": "Spam?", "criteria": {"true": "ads", "false": "chat"}}
    ans = post(TestClient(create_app(eng)), {"q": q}).json()["answers"]["q"]
    assert ans["noul"] == pytest.approx(0.75)
    assert ans["null_probability"] == pytest.approx(0.6)
    rec = eng.calls[0][0]
    assert rec["options"] == [{"name": "no", "description": "chat"}, {"name": "yes", "description": "ads"}]


def test_score_answer(client):
    ans = post(client, {"urgency": SCORE_Q}).json()["answers"]["urgency"]
    assert ans["type"] == "score"
    assert ans["legend"] == {"0": "Can wait", "1": "This week", "2": "Today"}
    probs = ans["probabilities"]
    assert list(probs) == ["0", "1", "2"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert ans["score"] == pytest.approx(sum(int(k) * v for k, v in probs.items()))
    assert 0.0 <= ans["score"] <= 2.0
    assert 0.0 <= ans["confidence"] <= 1.0
    assert_extensions(ans)


def test_score_docs_example():
    # confidence page and score page: (0, 0.57, 0.43) gives score 1.43, confidence about 0.35
    eng = FakeEngine(probs_fn=lambda rec: {"0": 0.0, "1": 0.57, "2": 0.43})
    ans = post(TestClient(create_app(eng)), {"q": SCORE_Q}).json()["answers"]["q"]
    assert ans["score"] == pytest.approx(1.43)
    assert ans["confidence"] == pytest.approx(1 - 0.43 / (2 / 3))
    assert round(ans["confidence"], 2) == 0.35


def test_score_confidence_extremes():
    peak = FakeEngine(probs_fn=lambda rec: {"0": 0.0, "1": 1.0, "2": 0.0})
    flat = FakeEngine(probs_fn=lambda rec: {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})
    assert post(TestClient(create_app(peak)), {"q": SCORE_Q}).json()["answers"]["q"]["confidence"] == pytest.approx(1.0)
    assert post(TestClient(create_app(flat)), {"q": SCORE_Q}).json()["answers"]["q"]["confidence"] == pytest.approx(0.0, abs=1e-9)


def test_score_structured_legend_is_echoed():
    q = {"type": "score", "instructions": "Rate",
         "criteria": [{"desc": "low", "examples": ["a"]}, "high"]}
    ans = post(TestClient(create_app(FakeEngine())), {"q": q}).json()["answers"]["q"]
    assert ans["legend"] == {"0": {"desc": "low", "examples": ["a"]}, "1": "high"}


# ------------------------------------------------------------------ requests
def test_multi_question_request(client, engine):
    r = post(client, {"tone": CHOICE_Q, "urgency": SCORE_Q, "billing": NOUL_Q})
    assert r.status_code == 200
    body = r.json()
    assert list(body["answers"]) == ["tone", "urgency", "billing"]
    assert [body["answers"][k]["type"] for k in body["answers"]] == ["choice", "score", "noul"]
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 3
    expected_tokens = sum(len(r_["state"].split()) + 3 for r_ in engine.calls[0])
    assert body["usage"] == {"input_tokens": expected_tokens, "output_tokens": 0}


def test_unknown_model_is_served_with_our_id(client, engine):
    r = client.post("/v1/systemone", json={"state": STATE, "model": "jev-1.14.0", "questions": {"q": NOUL_Q}})
    assert r.status_code == 200 and r.json()["model"] == engine.model_id


def test_authorization_header_is_ignored(client):
    r = client.post("/v1/systemone", headers={"Authorization": "Bearer anything"},
                    json={"state": STATE, "model": "m", "questions": {"q": NOUL_Q}})
    assert r.status_code == 200


def test_json_object_state(client, engine):
    state = {"subject": "Duplicate charge", "message": "Please help", "n": 3}
    assert post(client, {"q": NOUL_Q}, state=state).status_code == 200
    rec = engine.calls[0][0]
    from kelter.state import render_state
    assert rec["state"] == render_state(state)
    assert '"subject": "Duplicate charge"' in rec["state"]
    assert rec["provenance"] == [[0, len(rec["state"]), "unknown"]]


def test_array_state(client, engine):
    assert post(client, {"q": NOUL_Q}, state=["a", {"b": 1}]).status_code == 200
    assert engine.calls[0][0]["state"].startswith("[")


def test_record_shape(client, engine):
    post(client, {"tone": CHOICE_Q, "urgency": SCORE_Q})
    from kelter.data import schema
    for rec in engine.calls[0]:
        schema.validate({**rec, "gold": rec["options"][0]["name"]})  # validate() wants a gold; requests have none
        assert rec["gold"] is None and rec["null"] is False and rec["witness"] is None
        assert rec["split"] == "test" and rec["domain"] == "policy" and rec["source"] == "request"
    assert [o["name"] for o in engine.calls[0][1]["options"]] == ["0", "1", "2"]
    assert len({r["id"] for r in engine.calls[0]}) == 2


def test_provenance_reaches_record(client, engine):
    state = "user said hi. TOOL OUTPUT HERE. tail"
    start = state.index("TOOL")
    end = start + len("TOOL OUTPUT HERE.")
    r = post(client, {"q": NOUL_Q}, state=state, provenance=[{"start": 0, "end": 13, "type": "user"},
                                                            {"start": start, "end": end, "type": "tool_output"}])
    assert r.status_code == 200
    prov = engine.calls[0][0]["provenance"]
    assert prov == [[0, 13, "user"], [13, start, "unknown"], [start, end, "tool_output"], [end, len(state), "unknown"]]


def test_provenance_unordered_and_touching(client, engine):
    r = post(client, {"q": NOUL_Q}, state="abcdef", provenance=[
        {"start": 3, "end": 6, "type": "retrieved_doc"}, {"start": 0, "end": 3, "type": "system"}])
    assert r.status_code == 200
    assert engine.calls[0][0]["provenance"] == [[0, 3, "system"], [3, 6, "retrieved_doc"]]


def test_no_provenance_is_one_unknown_segment(client, engine):
    post(client, {"q": NOUL_Q})
    assert engine.calls[0][0]["provenance"] == [[0, len(STATE), "unknown"]]


def test_provenance_over_rendered_json_state(client, engine):
    state = {"a": "x"}
    from kelter.state import render_state
    n = len(render_state(state))
    assert post(client, {"q": NOUL_Q}, state=state, provenance=[{"start": 0, "end": n, "type": "tool_output"}]).status_code == 200
    assert engine.calls[0][0]["provenance"] == [[0, n, "tool_output"]]


# ------------------------------------------------------------------ 422
def _detail(r):
    assert r.status_code == 422, r.text
    d = r.json()["detail"]
    assert isinstance(d, list) and d and {"loc", "msg", "type"} <= set(d[0])
    return d


def test_unknown_question_type_is_422(client):
    _detail(post(client, {"q": {"type": "ranking", "instructions": "x", "criteria": ["a", "b"]}}))


def test_box_question_is_422(client):
    _detail(post(client, {"q": {"type": "box", "instructions": "find it"}}))


def test_empty_criteria_is_422(client):
    _detail(post(client, {"q": {"type": "choice", "instructions": "x", "criteria": {}}}))
    _detail(post(client, {"q": {"type": "score", "instructions": "x", "criteria": []}}))


def test_missing_criteria_is_422(client):
    _detail(post(client, {"q": {"type": "choice", "instructions": "x"}}))


def test_single_score_level_is_422(client):
    d = _detail(post(client, {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}))
    assert "2 levels" in d[0]["msg"]


def test_too_many_score_levels_is_422(client):
    _detail(post(client, {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}}))


def test_missing_instructions_is_422(client):
    d = _detail(post(client, {"q": {"type": "noul"}}))
    assert "instructions" in d[0]["msg"]


def test_missing_fields_are_422(client):
    _detail(client.post("/v1/systemone", json={"model": "m", "questions": {"q": NOUL_Q}}))
    _detail(client.post("/v1/systemone", json={"model": "m", "state": "s"}))
    _detail(client.post("/v1/systemone", json={"model": "m", "state": "s", "questions": {}}))
    _detail(client.post("/v1/systemone", json={"state": "s", "questions": {"q": NOUL_Q}}))


def test_empty_state_is_422(client):
    _detail(post(client, {"q": NOUL_Q}, state="   "))


def test_assets_are_422(client):
    _detail(post(client, {"q": NOUL_Q}, assets={"x": {"type": "image", "format": "png", "data_base64": "AA=="}}))


@pytest.mark.parametrize("prov", [
    [{"start": 0, "end": 5, "type": "user"}, {"start": 4, "end": 8, "type": "system"}],  # overlap
    [{"start": 0, "end": 999, "type": "user"}],                                          # out of range
    [{"start": -1, "end": 3, "type": "user"}],                                           # negative
    [{"start": 3, "end": 3, "type": "user"}],                                            # empty range
    [{"start": 0, "end": 3, "type": "wizard"}],                                          # bad type
    [{"start": 0, "end": 3}],                                                            # missing type
])
def test_bad_provenance_is_422(client, engine, prov):
    _detail(post(client, {"q": NOUL_Q}, state="abcdefgh", provenance=prov))
    assert engine.calls == []  # never reaches the engine


# ------------------------------------------------------------------ models
def test_models(client, engine):
    r = client.get("/v1/models", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    models = r.json()["models"]
    assert [m["name"] for m in models] == [engine.model_id]
    for m in models:
        assert set(m) == {"name", "description", "release_date"}
        assert len(m["release_date"]) == 10 and m["release_date"][4] == "-"


# ------------------------------------------------------------------ JevBench round trip
def _jevbench_items():
    if not (JEVBENCH / "jevbench").exists():
        pytest.skip("third_party/jevbench is not cloned")
    if str(JEVBENCH) not in sys.path:
        sys.path.insert(0, str(JEVBENCH))
    from jevbench.tasks import load_jsonl
    items = []
    for name in ("original", "hard", "easy"):
        tasks = load_jsonl(str(JEVBENCH / "datasets" / "public" / f"{name}.jsonl"))
        picked, seen = [], set()
        for t in tasks:  # one per question type first, so choice, noul and score are all covered
            if t.question["type"] not in seen:
                seen.add(t.question["type"])
                picked.append(t)
        for t in tasks:
            if len(picked) >= 3:
                break
            if t not in picked:
                picked.append(t)
        items.extend((name, t) for t in picked[:3])
    return items


def test_jevbench_adapter_round_trip(monkeypatch):
    items = _jevbench_items()
    assert len(items) == 9
    from jevbench.adapters import typesafe as ts_mod
    tc = TestClient(create_app(FakeEngine()))

    def fake_post(url, body, headers, timeout_s=120.0):
        assert url == "http://testserver/v1/systemone"
        r = tc.post("/v1/systemone", json=body, headers=headers)
        return r.status_code, r.json(), 0.01

    monkeypatch.setattr(ts_mod, "http_post_json", fake_post)
    adapter = ts_mod.TypeSafeAdapter(endpoint="http://testserver", model="jev-latest", key_env="")
    types = set()
    for name, task in items:
        res = adapter.run(task)
        assert res.ok, f"{name}/{task.id}: {res.error}"
        types.add(task.question["type"])
        assert set(res.probs) == set(task.labels), f"{name}/{task.id}"
        assert sum(res.probs.values()) == pytest.approx(1.0)
        assert res.model == "fake-verdict-1"
        assert res.usage["output_tokens"] == 0 and res.usage["input_tokens"] > 0
    assert types == {"choice", "noul", "score"}


# ------------------------------------------------------------------ official SDK
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_server():
    import uvicorn
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(FakeEngine()), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=10)


def test_typesafe_python_sdk_against_server(live_server):
    sdk = pytest.importorskip("typesafe_sdk")
    with sdk.TypeSafeClient(api_key="test-key", base_url=live_server) as c:
        resp = c.system_one(
            state={"document": "I was charged twice. Please fix this ASAP."},
            questions={
                "billing": sdk.Noul(instructions="Is this ticket about billing?"),
                "tone": sdk.Choice(instructions="Tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
                "urgency": sdk.Score(instructions="How urgent?", criteria=["can wait", "this week", "today"]),
            },
        )
        models = c.list_models() if hasattr(c, "list_models") else None
    assert 0.0 <= resp.nouls["billing"].noul <= 1.0
    assert resp.choices["tone"].choice == "calm"
    assert sum(resp.choices["tone"].probabilities.values()) == pytest.approx(1.0)
    assert 0.0 <= resp.scores["urgency"].score <= 2.0
    assert set(resp.scores["urgency"].probabilities) == {0, 1, 2}
    assert resp.model == "fake-verdict-1"
    assert resp.usage.output_tokens == 0 and resp.usage.input_tokens > 0
    if models is not None:
        assert "fake-verdict-1" in str(models)


def test_verdict_classes_extension_reaches_record():
    from fastapi.testclient import TestClient
    from kelter.server import create_app
    from tests.fakes import FakeEngine
    eng = FakeEngine()
    c = TestClient(create_app(eng))
    body = {"model": "jev-latest", "state": "rm -rf build", "questions": {"q": {"type": "choice", "instructions": "Run it?",
            "criteria": {"allow": "Run.", "deny": "Do not run."}}},
            "verdict_classes": {"q": {"allow": "destructive"}}}
    r = c.post("/v1/systemone", json=body)
    assert r.status_code == 200
    assert eng.calls[-1][0]["meta"]["verdict_class"] == {"allow": "destructive"}
    body["verdict_classes"] = {"q": {"nope": "destructive"}}
    assert c.post("/v1/systemone", json=body).status_code == 422


# ------------------------------------------------------------------ variants
def test_default_variant_is_e2():
    from kelter.server import DEFAULT_VARIANT, VARIANTS, resolve_variant
    assert DEFAULT_VARIANT == "default"
    assert resolve_variant() == ("default", "checkpoints/Q1")
    assert VARIANTS == {"default": "checkpoints/Q1", "lower-latency": "checkpoints/E1"}


def test_variants_match_final_json():
    import json
    from kelter.server import DEFAULT_VARIANT, VARIANTS
    fin = json.loads((ROOT / "results" / "final.json").read_text(encoding="utf-8"))
    assert {v["name"]: v["ckpt"] for v in fin["variants"]} == VARIANTS
    default = [v for v in fin["variants"] if v["role"] == "default"]
    assert [v["name"] for v in default] == [DEFAULT_VARIANT] and default[0]["run"] == fin["run"]


@pytest.mark.parametrize("name,ckpt", [("default", "checkpoints/Q1"), ("lower-latency", "checkpoints/E1")])
def test_variant_named_in_every_answer(name, ckpt):
    from kelter.server import resolve_variant
    v, path = resolve_variant(name)
    assert (v, path) == (name, ckpt)
    assert resolve_variant(None, ckpt)[0] == name
    body = post(TestClient(create_app(FakeEngine(), v)), {"a": CHOICE_Q, "b": NOUL_Q, "c": SCORE_Q}).json()
    assert {a["variant"] for a in body["answers"].values()} == {name}


def test_unknown_variant_and_custom_ckpt():
    from kelter.server import resolve_variant
    with pytest.raises(ValueError):
        resolve_variant("fastest")
    assert resolve_variant("default", "checkpoints/r5") == ("custom:r5", "checkpoints/r5")
    assert resolve_variant(None, "checkpoints/r2") == ("custom:r2", "checkpoints/r2")
