"""Tests for prompt-window refusal, ensemble weight checks, a negative prefill chunk and non-finite probabilities."""
import pytest


# ---- the LLM never cuts the header, the instructions or an option line
class _CharTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<S>" + msgs[0]["content"] + "<E>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _rec(instr="Pick one.", opt="d" * 20, state="s" * 50, n=3):
    return {"id": "abc123-q", "state": state, "question": {"type": "choice", "instructions": instr},
            "options": [{"name": f"o{i}", "description": opt} for i in range(n)]}


def test_prompt_that_cannot_hold_the_question_is_refused():
    from kelter import llm
    from kelter.data.schema import QuestionTooLong
    rec = _rec(instr="i" * 600, opt="d" * 300)
    with pytest.raises(QuestionTooLong) as e:
        llm.build_prompt(_CharTok(), rec, [0, 1, 2], 500)
    assert e.value.record_id == "abc123-q" and "500" in str(e.value)


def test_a_long_state_is_cut_but_the_header_and_every_option_line_stay():
    from kelter import llm
    tok = _CharTok()
    rec = _rec(state="word " * 2000)
    info = {}
    ids = llm.build_prompt(tok, rec, [0, 1, 2], 600, info)
    text = tok.decode(ids)
    assert info == {"truncated": True} and len(ids) <= 600 and text.startswith("<S>Pick one.") and text.endswith("<E>")
    for i in range(3):
        assert f"{llm.LETTERS[i]}. o{i} 'x'".replace(" 'x'", "") in text or f"{llm.LETTERS[i]}. " in text
    assert text.count("Options:") == 1 and "[...]" in text and llm.ANSWER_LINE in text


def test_a_prompt_that_fits_is_unchanged():
    from kelter import llm
    tok = _CharTok()
    rec = _rec()
    text = tok.apply_chat_template([{"content": llm.user_text(rec, [0, 1, 2], rec["state"])}])
    assert tok.decode(llm.build_prompt(tok, rec, [0, 1, 2], 10_000)) == text


def test_server_names_the_question_that_does_not_fit():
    from fastapi.testclient import TestClient

    from kelter import server
    from kelter.data.schema import QuestionTooLong
    from tests.fakes import FakeEngine

    class Raising(FakeEngine):
        def decide(self, recs, canonical=True):
            raise QuestionTooLong("too long", recs[0]["id"])
    c = TestClient(server.create_app(Raising()))
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "1", "b": "2"}}
    r = c.post("/v1/systemone", json={"model": "m", "state": "x", "questions": {"route": q}})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "questions", "route"]


# ---- an ensemble is checked through its members
def test_ensemble_members_are_checked_by_check_weights_and_present(tmp_path, monkeypatch):
    import json

    from kelter import download, paths, server
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))
    ck = tmp_path / "checkpoints"
    for name, cfg in (("r2", {"backbone": "b"}), ("E1", {"ensemble": "E1", "members": ["r2", "r3"]})):
        (ck / name).mkdir(parents=True)
        (ck / name / "config.json").write_text(json.dumps(cfg))
    (ck / "r2" / "model.pt").write_bytes(b"w")
    assert paths.checkpoint_missing(ck / "E1") == ["../r3/config.json"]
    with pytest.raises(server.MissingWeights) as e:
        server.check_weights(str(ck / "E1"))
    msg = str(e.value)
    assert "\n" not in msg and "r3/config.json" in msg and "weights not found" in msg
    assert download.present({"variants": {"ll": ["r2", "r3", "E1"]}}, "ll", tmp_path) is False
    (ck / "r3").mkdir()
    (ck / "r3" / "config.json").write_text(json.dumps({"backbone": "b"}))
    (ck / "r3" / "model.pt").write_bytes(b"w")
    assert paths.checkpoint_missing(ck / "E1") == []
    server.check_weights(str(ck / "E1"))


# ---- other input checks
def test_negative_prefill_chunk_is_refused_at_start(monkeypatch):
    from kelter import llm
    monkeypatch.setenv("KELTER_PREFILL_CHUNK", "-1")
    with pytest.raises(ValueError, match="KELTER_PREFILL_CHUNK"):
        llm.prefill_chunk_setting({})
    monkeypatch.setenv("KELTER_PREFILL_CHUNK", "0")
    assert llm.prefill_chunk_setting({}) == 0
    monkeypatch.setenv("KELTER_PREFILL_CHUNK", "256")
    assert llm.prefill_chunk_setting({}) == 256


def test_non_finite_probabilities_give_the_engine_error_body():
    from fastapi.testclient import TestClient

    from kelter import server
    from tests.fakes import FakeEngine
    for bad in (float("nan"), float("inf")):
        eng = FakeEngine(probs_fn=lambda rec, bad=bad: {o["name"]: bad for o in rec["options"]})
        r = TestClient(server.create_app(eng)).post("/v1/systemone", json={
            "model": "m", "state": "x", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "1", "b": "2"}}}})
        assert r.status_code == 500 and r.json()["detail"].startswith("engine error:") and "non-finite" in r.json()["detail"]
