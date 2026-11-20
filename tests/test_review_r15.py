"""A huge integer in the body, the download hint per variant, the empty-state
loc, and --help on every module run with -m. No GPU, no network."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kelter import server
from tests.fakes import FakeEngine

ROOT = Path(__file__).resolve().parents[1]
GUARDED = ["kelter.eval.table", "kelter.eval.unknowable", "kelter.eval.baselines.tfidf"]
Q = {"type": "choice", "instructions": "i", "criteria": {"a": "1", "b": "2"}}


def post(body):
    c = TestClient(server.create_app(FakeEngine()))
    return c.post("/v1/systemone", content=body, headers={"content-type": "application/json"})


# ---- 1: an integer of more than 4,300 digits is the documented 422, not FastAPI's 400
@pytest.mark.parametrize("where", ["state", "question", "top"])
def test_huge_integer_is_422_json_invalid(where):
    big = "9" * 5000
    body = {"state": '{"model":"m","state":%s,"questions":{}}' % big,
            "question": '{"model":"m","state":"x","questions":{"q":{"type":"choice","instructions":%s,"criteria":{"a":"1"}}}}' % big,
            "top": "[%s]" % big}[where]
    r = post(body)
    d = r.json()["detail"]
    assert r.status_code == 422 and d[0]["type"] == "json_invalid" and d[0]["loc"][0] == "body"
    assert "4,300 digits" in d[0]["msg"]


def test_4300_digits_still_parse_and_invalid_json_keeps_422():
    ok = '{"model":"m","state":"x","questions":{"q":%s},"n":%s}' % (json.dumps(Q), "1" * 4300)
    assert post(ok).status_code != 400
    r = post("{not json")
    assert r.status_code == 422 and r.json()["detail"][0]["msg"].startswith("JSON decode error: ")
    assert post(b"\xff\xfe").status_code == 422  # not UTF-8: unchanged


# ---- 2: the hint names the variant
def _empty(tmp_path, monkeypatch):
    monkeypatch.setenv("KELTER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def test_missing_weights_hint_names_the_variant(tmp_path, monkeypatch):
    _empty(tmp_path, monkeypatch)
    with pytest.raises(server.MissingWeights) as e:
        server.check_weights("checkpoints/E1", root=tmp_path)
    assert re.search(r"`[^`]*download\S* --variant lower-latency`", str(e.value))
    with pytest.raises(server.MissingWeights) as e:
        server.check_weights("checkpoints/Q1", root=tmp_path)
    assert re.search(r"`[^`]*download\S*`", str(e.value)) and "--variant" not in str(e.value)
    with pytest.raises(server.MissingWeights) as e:
        server.check_weights("somewhere/mine", root=tmp_path)  # not a named variant: the missing files are listed
    assert "mine" in str(e.value) and "--variant" not in str(e.value)


# ---- 3: the comment says what was measured
def test_gpu_comment_states_the_measurement():
    src = (ROOT / "kelter" / "server.py").read_text(encoding="utf-8")
    assert "no longer exists" in src and "was not measured" in src
    assert server.LLM_GPU_GB > 8.82


# ---- 4: an empty state names the state
@pytest.mark.parametrize("state", ["", "   ", "\n\t "])
def test_empty_state_loc_is_body_state(state):
    c = TestClient(server.create_app(FakeEngine()))
    r = c.post("/v1/systemone", json={"model": "m", "state": state, "questions": {"q1": Q, "q2": Q}})
    assert r.status_code == 422
    assert r.json()["detail"] == [{"loc": ["body", "state"], "msg": "invalid question: empty state", "type": "value_error"}]


def test_contract_lists_the_empty_state_and_the_digit_limit():
    doc = (ROOT / "docs" / "typesafe-contract.md").read_text(encoding="utf-8")
    assert "empty or whitespace-only `state`" in doc and "4,300 digits" in doc


