"""Input limits, truncation flag and build-record fixes. No GPU, no network."""
import json
import random
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHOICE = {"type": "choice", "instructions": "What is the tone?", "criteria": {"angry": "upset", "calm": "neutral"}}


def _post_raw(client, body: bytes):
    return client.post("/v1/systemone", content=body, headers={"Content-Type": "application/json"})


def _client(engine=None):
    from fastapi.testclient import TestClient

    from kelter import server
    from tests.fakes import FakeEngine
    return TestClient(server.create_app(engine or FakeEngine()))


def _post(c, question=None, state="hello", **kw):
    body = {"model": "m", "state": state, "questions": {"q": question or CHOICE}}
    body.update(kw)
    return c.post("/v1/systemone", json=body)


# ---- A1 to A5: the server answers 422
def test_null_sentinel_name_is_rejected():
    r = _post(_client(), dict(CHOICE, criteria={"__null__": "x", "calm": "neutral"}))
    assert r.status_code == 422 and "__null__" in r.json()["detail"][0]["msg"]


def test_deep_nesting_is_rejected_not_a_500():
    c = _client()
    deep = lambda n: b"[" * n + b"]" * n
    q = json.dumps(CHOICE).encode()
    r = _post_raw(c, b'{"model":"m","state":' + deep(100) + b',"questions":{"q":' + q + b"}}")
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "state"]
    r = _post_raw(c, b'{"model":"m","state":' + deep(1_500) + b',"questions":{"q":' + q + b"}}")  # near the interpreter's limit
    assert r.status_code == 422 and "nested" in r.json()["detail"][0]["msg"]
    r = _post_raw(c, b'{"model":"m","state":' + deep(20_000) + b',"questions":{"q":' + q + b"}}")  # unparseable: FastAPI's 400
    assert r.status_code == 400
    nested_instr = json.loads(b'{"a":' * 70 + b"1" + b"}" * 70)
    assert _post(c, dict(CHOICE, instructions=nested_instr)).status_code == 422
    assert _post(c, state={"a": [1, {"b": 2}]}).status_code == 200


def test_lone_surrogate_is_rejected():
    c = _client()
    q = json.dumps(CHOICE).encode()
    for body in (b'{"model":"m","state":"a\\ud800b","questions":{"q":' + q + b"}}",
                 b'{"model":"m","state":"x","questions":{"q":{"type":"choice","instructions":"i\\ud800",'
                 b'"criteria":{"a":"1","b":"2"}}}}',
                 b'{"model":"m","state":"x","questions":{"q":{"type":"choice","instructions":"i",'
                 b'"criteria":{"a":"\\udc00","b":"2"}}}}'):
        r = _post_raw(c, body)
        assert r.status_code == 422 and "UTF-8" in r.json()["detail"][0]["msg"], body


def test_question_too_long_for_the_encoder_is_a_422():
    from kelter.data.schema import QuestionTooLong
    from tests.fakes import FakeEngine

    class Raising(FakeEngine):
        def decide(self, recs, canonical=True):
            raise QuestionTooLong("question and options exceed max_len")
    r = _post(_client(Raising()))
    assert r.status_code == 422 and "too long" in r.json()["detail"][0]["msg"]
    assert issubclass(QuestionTooLong, ValueError)


def test_featurizer_raises_question_too_long():
    from kelter.data.schema import QuestionTooLong
    from kelter.model import Featurizer
    f = Featurizer.__new__(Featurizer)
    f.cls_id, f.sep_id, f.opt_id, f.null_id, f.null_ids, f.max_len = 1, 2, 3, 4, [5, 6], 12
    pc = {"instr": [7] * 5, "opts": [[8] * 4, [9] * 4], "sids": [10] * 3, "offs": [[0, 1], [1, 2], [2, 3]]}
    with pytest.raises(QuestionTooLong):
        f.assemble(pc, [[0, 3, "user"]])


def test_field_caps():
    from kelter import server
    c = _client()
    assert _post(c, dict(CHOICE, instructions="x" * server.MAX_INSTRUCTION_CHARS)).status_code == 200
    assert _post(c, dict(CHOICE, instructions="x" * (server.MAX_INSTRUCTION_CHARS + 1))).status_code == 422
    big = {"a": "x" * (server.MAX_DESCRIPTION_CHARS + 1), "b": "y"}
    assert _post(c, dict(CHOICE, criteria=big)).status_code == 422
    assert _post(c, dict(CHOICE, criteria={"n" * (server.MAX_NAME_CHARS + 1): "x", "b": "y"})).status_code == 422
    text = (ROOT / "docs" / "typesafe-contract.md").read_text(encoding="utf-8")
    for n in ("10,000", "4,000", "256", "64 levels"):
        assert n in text


# ---- A6: the sorted lookup gives the scan's answer
def test_provenance_finder_matches_a_scan():
    from kelter.model import _provenance_finder
    rng = random.Random(0)
    types_ = ["user", "tool_output", "system", "retrieved_doc", "unknown"]

    def scan(prov, mid):
        for s, e, t in prov:
            if s <= mid < e:
                return t
        return "unknown"
    for _ in range(300):
        n = 60
        if rng.random() < 0.5:  # non-overlapping, with gaps and an empty span
            cuts = sorted(rng.sample(range(n + 1), 8))
            prov = [[cuts[i], cuts[i + 1], rng.choice(types_)] for i in range(0, 7, 2)] + [[5, 5, "user"]]
        else:  # overlapping, unsorted
            prov = [[a, a + rng.randint(0, 20), rng.choice(types_)] for a in (rng.randint(0, n) for _ in range(5))]
        rng.shuffle(prov)
        find = _provenance_finder(prov)
        assert all(find(m) == scan(prov, m) for m in range(-1, n + 25)), prov


# ---- B7: a blank description on a bare name shows the name
def test_option_text_blank_description():
    from kelter.model import option_text
    assert option_text({"name": "yes", "description": ""}) == "yes"
    assert option_text({"name": "A", "description": "  "}) == "A"
    assert option_text({"name": "3", "description": "three apples"}) == "three apples"  # unchanged
    assert option_text({"name": "no", "description": "No refund"}) == "No refund"
    assert option_text({"name": "deny_x", "description": "d"}) == "deny x: d"


# ---- C8: the encoder's instruction and option cuts set truncated
class _Tok:
    cls_token_id, sep_token_id = 1, 2

    def convert_tokens_to_ids(self, t):
        return {"[OPT]": 3, "[NULL]": 4}[t]

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False, split_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[10 + i for i, _ in enumerate(t.split())] for t in texts]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [[(j, j + 1) for j in range(len(i))] for i in ids]
        return out


def test_encoder_cuts_set_truncated():
    from kelter.model import Featurizer
    f = Featurizer(_Tok(), max_len=512, max_option_tokens=3, max_instr_tokens=4)
    rec = lambda instr, desc: {"state": "a b c", "question": {"instructions": instr},
                               "options": [{"name": "x_y", "description": desc}], "provenance": [[0, 5, "user"]]}
    for instr, desc, want in (("one two three", "d", False), ("one two three four five", "d", True),
                              ("one two", "p q r s t u", True), ("one two", "d", False)):
        r = rec(instr, desc)
        pc = f.pieces_batch([r])[0]
        assert f.assemble(pc, r["provenance"]).truncated is want, (instr, desc)


# ---- C9 to C13
def test_llm_train_revision_and_command():
    from kelter import llm_train as lt
    assert lt.pick_revision(lt.BASE, None) == lt.BASE_REVISION
    assert lt.pick_revision(lt.BASE, "abc") == "abc" and lt.pick_revision("other/base", None) is None
    q1 = ROOT / "checkpoints" / "Q1" / "config.json"
    if q1.exists():
        assert json.loads(q1.read_text(encoding="utf-8"))["base_revision"] == lt.BASE_REVISION
    a = types.SimpleNamespace(name="Q", seed=0, limit=0, qlora=True, resume=False, base_revision=None, dropout=0.05, power_mode=None)
    cmd = lt.command_line(a)
    assert "--seed 0" in cmd and "--limit 0" in cmd and "--qlora" in cmd and "--resume" not in cmd
    assert "base-revision" not in cmd and "power-mode" not in cmd


def test_corrected_null_missing_is_recorded(tmp_path, monkeypatch, capsys):
    from kelter.eval import run
    monkeypatch.setattr(run, "BUILT", tmp_path)
    assert run.corrected_null(None, [], []) == {"null_corrected_set_missing": True}
    assert "not found" in capsys.readouterr().out


def test_ensemble_member_calibration_recorded_and_warned():
    from kelter.ensemble import member_calibration
    ok = types.SimpleNamespace(temps={"choice": 1.4}, null_cal={"T_null": 2.0})
    assert member_calibration("m", ok) == {"temperature": {"choice": 1.4}, "null_calibration": {"T_null": 2.0}}
    bare = types.SimpleNamespace(temps={}, null_cal={})
    with pytest.warns(UserWarning, match="no temperature"):
        assert member_calibration("m2", bare) == {"temperature": {}, "null_calibration": {}}


def test_build_records_source_file_hashes(tmp_path, monkeypatch):
    import hashlib

    import huggingface_hub

    from kelter.data import build
    f = tmp_path / "part.parquet"
    f.write_bytes(b"data")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, **k: str(f))
    sink = []
    undo = build._record_downloads(sink)
    try:
        assert huggingface_hub.hf_hub_download("o/ds", "part.parquet", repo_type="dataset") == str(f)
    finally:
        undo()
    assert sink == [{"repo": "o/ds", "file": "part.parquet", "revision": None, "sha256": hashlib.sha256(b"data").hexdigest()}]


# ---- D14: truncation from the full pass alone, and from the prompt through predict
def test_truncated_from_the_full_pass_alone():
    from kelter.engine import Engine, load_policy
    from tests.test_engine import SequencedPredictor, _out, _user_rec
    r, wit = _user_rec()
    pred = SequencedPredictor([_out(0.9, wit, truncated=True)], [_out(0.9), _out(0.1), _out(0.9)])
    assert Engine(pred, load_policy()).decide([r])[0]["verifier"]["truncated"] is True


def test_prompt_cut_reaches_predict_output():
    import torch

    from kelter import llm

    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids):
            return "".join(chr(i) for i in ids)
    p = llm.LLMPredictor.__new__(llm.LLMPredictor)
    p.tok, p.temps, p.witness_pred = Tok(), {}, None
    p.logits_for = lambda prompts: [torch.zeros(26) for _ in prompts]
    rec = lambda n: {"state": "s" * n, "question": {"type": "choice", "instructions": "i"},
                     "options": [{"name": "a", "description": "A"}, {"name": "b", "description": "B"}]}
    p.max_tokens = 200
    assert [o["truncated"] for o in p.predict([rec(10), rec(900)])] == [False, True]
