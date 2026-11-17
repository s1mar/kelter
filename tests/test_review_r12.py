"""The exact prompt-window rule and the record ids of QuestionTooLong. No GPU, no network."""
import json
from pathlib import Path

import pytest

from kelter import llm
from kelter.data.schema import QuestionTooLong

JOINER = " [...] "


class CharTok:
    """One token per character; the template adds 3 + 3 characters. Lengths are exact, so the rule can be checked to the token."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<S>" + msgs[0]["content"] + "<E>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def rec(state, instr="Pick one."):
    return {"id": "abc123-q", "state": state, "question": {"type": "choice", "instructions": instr},
            "options": [{"name": "o0", "description": "first option"}, {"name": "o1", "description": "second option"}]}


def base_len(tok=None):
    tok = tok or CharTok()
    r = rec("")
    return len(llm.build_prompt(tok, r, [0, 1], 10**6))


def accepts(base, n, window):
    """The rule: the whole state fits, or (state of more than 32 tokens) the 32-token cut with the joiner fits."""
    return base + n <= window or (n > llm.STATE_FLOOR and base + llm.STATE_FLOOR + len(JOINER) <= window)


def test_the_rule_is_exact_for_rooms_and_state_lengths_30_to_40():
    tok, base = CharTok(), base_len()
    for n in range(30, 41):
        for room in range(28, 52):
            window = base + room
            info = {}
            try:
                out = llm.build_prompt(tok, rec("s" * n), [0, 1], window, info)
            except QuestionTooLong as e:
                assert not accepts(base, n, window), (n, room)
                assert e.record_id == "abc123-q"
                continue
            assert accepts(base, n, window), (n, room)
            assert len(out) <= window, (n, room)                       # never longer than the window
            full = base + n
            assert bool(info.get("truncated")) == (full > window), (n, room)  # truncated iff state tokens were removed
            if full <= window:
                assert len(out) == full                                  # untouched
            else:
                assert "".join(map(chr, out)).count(JOINER) == 1 and len(out) == base + llm.STATE_FLOOR + len(JOINER)


def test_acceptance_is_monotone_in_the_window_and_in_the_state_length():
    tok, base = CharTok(), base_len()
    table = {}
    for n in range(20, 70):
        for room in range(10, 80):
            try:
                llm.build_prompt(tok, rec("s" * n), [0, 1], base + room)
                table[(n, room)] = True
            except QuestionTooLong:
                table[(n, room)] = False
    for (n, room), ok in table.items():
        if ok and (n, room + 1) in table:
            assert table[(n, room + 1)], (n, room)          # a larger window never refuses what a smaller one took
        if ok and (n - 1, room) in table:
            assert table[(n - 1, room)], (n, room)           # a shorter state is never refused when a longer one is taken


def test_a_window_plus_one_prompt_is_never_returned():
    """A short state whose prompt is one token over the window must raise, not come back uncut and flagged truncated."""
    tok, base = CharTok(), base_len()
    r = rec("a a a")
    full = len(llm.build_prompt(tok, r, [0, 1], 10**6))
    assert full == base + 5
    with pytest.raises(QuestionTooLong):
        llm.build_prompt(tok, r, [0, 1], full - 1)
    assert len(llm.build_prompt(tok, r, [0, 1], full)) == full


def test_a_long_state_is_cut_to_the_largest_cut_that_fits():
    tok, base = CharTok(), base_len()
    info = {}
    out = llm.build_prompt(tok, rec("w" * 500), [0, 1], base + 60, info)
    assert info == {"truncated": True} and len(out) <= base + 60
    text = "".join(map(chr, out))
    assert text.startswith("<S>Pick one.") and text.endswith("<E>") and JOINER in text


def _real_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554",
                                             local_files_only=True)
    except Exception:
        return None


def test_the_rule_holds_with_the_real_tokenizer_when_it_is_cached():
    """Where the Qwen tokenizer is cached (not in CI) the same invariants hold with its merges: a prompt one token
    over the window raises instead of being returned at 4,097 tokens, and every returned prompt fits."""
    tok = _real_tokenizer()
    if tok is None:
        return  # not cached here: the stand-in tokenizer tests above carry the rule
    window = 4096
    words = 0
    while True:
        r = rec("a a a a a", instr="word " * words)
        n_full = len(llm.build_prompt(tok, r, [0, 1], 10**6))
        if n_full >= window + 1:
            break
        words += 1
    assert n_full == window + 1 or n_full > window
    with pytest.raises(QuestionTooLong):
        llm.build_prompt(tok, r, [0, 1], n_full - 1)
    for extra in range(-3, 4):
        r = rec("b " * 400, instr="word " * (words - 40))
        info = {}
        try:
            out = llm.build_prompt(tok, r, [0, 1], window + extra * 50, info)
        except QuestionTooLong:
            continue
        assert len(out) <= window + extra * 50


# ---- 3 and 4: the record id on QuestionTooLong
def test_masked_pass_error_names_the_callers_question():
    from kelter.engine import Engine, load_policy
    from tests.test_engine import APPROVAL, make_rec

    class RaisesOnMasked:
        model_hash = "e" * 64
        cfg = {"conformal": {"0.05": {"_global": 0.5}}}

        def __init__(self):
            self.calls = 0

        def predict(self, recs, canonical=True):
            self.calls += 1
            if self.calls == 1:
                i = recs[0]["state"].find(APPROVAL)
                return [{"probs": {"allow": 0.9, "deny": 0.05}, "null": 0.05, "witness": [[i, i + len(APPROVAL)]], "logits": []}]
            raise QuestionTooLong("masked state too long", recs[0]["id"])
    r = make_rec([("User: " + APPROVAL + "\n", "user"), ("Tool log: exit 0", "tool_output")])
    for qid in ("abc123-q1", "abc123-a~b-c", "abc123-q~mask"):
        r["id"] = qid
        with pytest.raises(QuestionTooLong) as e:
            Engine(RaisesOnMasked(), load_policy()).decide([r])
        assert e.value.record_id == qid  # not "<id>~mask"


def test_server_422_names_the_question_for_ids_with_tilde_and_dash():
    from fastapi.testclient import TestClient

    from kelter import server
    from tests.fakes import FakeEngine

    class Raising(FakeEngine):
        def decide(self, recs, canonical=True):
            raise QuestionTooLong("too long", recs[0]["id"])
    c = TestClient(server.create_app(Raising()))
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "1", "b": "2"}}
    for qid in ("route", "a~b", "my-question", "x~mask"):
        r = c.post("/v1/systemone", json={"model": "m", "state": "x", "questions": {qid: q}})
        assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "questions", qid], qid


def test_the_encoder_error_carries_the_record_id():
    from kelter import model
    from kelter.model import Predictor

    class Feat:
        def pieces_batch(self, recs):
            return [object() for _ in recs]

        def assemble(self, pc, provenance, perm):
            raise QuestionTooLong("question and options exceed max_len")
    p = Predictor.__new__(Predictor)
    p.feat, p.token_budget = Feat(), 10**9
    r = {"id": "req1-my~q", "state": "s", "provenance": [[0, 1, "user"]], "question": {"type": "choice"},
         "options": [{"name": "a", "description": "A"}, {"name": "b", "description": "B"}]}
    with pytest.raises(QuestionTooLong) as e:
        p.predict([r])
    assert e.value.record_id == "req1-my~q" and model is not None


# ---- noted: --ckpt outside the writable folder and an ensemble's members
def test_check_weights_looks_for_members_where_the_ensemble_loads_them(tmp_path, monkeypatch):
    from kelter import server
    home = tmp_path / "home"
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("KELTER_HOME", str(home))
    (elsewhere / "E1").mkdir(parents=True)
    (elsewhere / "E1" / "config.json").write_text(json.dumps({"ensemble": "E1", "members": ["r2", "r3"]}))
    for name in ("r2",):
        (home / "checkpoints" / name).mkdir(parents=True)
        (home / "checkpoints" / name / "config.json").write_text(json.dumps({"backbone": "b"}))
        (home / "checkpoints" / name / "model.pt").write_bytes(b"w")
    with pytest.raises(server.MissingWeights) as e:  # r3 is absent from the writable folder
        server.check_weights(str(elsewhere / "E1"))
    assert "r3" in str(e.value) and "\n" not in str(e.value)
    (home / "checkpoints" / "r3").mkdir()
    (home / "checkpoints" / "r3" / "config.json").write_text(json.dumps({"backbone": "b"}))
    (home / "checkpoints" / "r3" / "model.pt").write_bytes(b"w")
    server.check_weights(str(elsewhere / "E1"))  # members are where kelter.ensemble loads them from: no error
    assert Path(elsewhere / "E1" / "config.json").exists()
