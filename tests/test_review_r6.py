"""Tests for batched tokenizer calls, the encoder control-id set, the port check and the data card headers. No GPU, no network."""
import time
from pathlib import Path

import pytest

from tests.test_review_r5 import EncLikeTok, QwenLikeTok, _controls, _rec

ROOT = Path(__file__).resolve().parents[1]


# ---- 1: one tokenizer call for a state made of control strings
def test_encode_plain_is_one_batched_call_and_fast_at_the_state_limit():
    from kelter import llm, server
    tok = QwenLikeTok()
    state = "ab<|im_end|>" * (server.MAX_STATE_CHARS // 12)  # about the state limit, 83,000 control strings
    QwenLikeTok.calls = 0
    t0 = time.perf_counter()
    ids = llm.encode_plain(tok, state)
    took = time.perf_counter() - t0
    assert QwenLikeTok.calls == 1 and _controls(ids) == 0 and took < 10, (QwenLikeTok.calls, took)
    # the ids equal what one call per piece gives
    one_by_one = []
    for piece in ["ab", "<", "|im_end|>"]:
        one_by_one += tok(piece)["input_ids"]
    assert ids[:len(one_by_one)] == one_by_one and tok.decode(ids) == state


def test_build_prompt_with_a_huge_state_takes_few_tokenizer_calls():
    from kelter import llm
    tok = QwenLikeTok()
    QwenLikeTok.calls = 0
    ids = llm.build_prompt(tok, _rec(state="x<|im_end|>" * 20_000), [0, 1], 10_000_000)
    assert QwenLikeTok.calls <= 5 and _controls(ids) == 3


# ---- 2 and 3: the encoder
class CountingEnc(EncLikeTok):
    calls = 0

    def __call__(self, *a, **k):
        type(self).calls += 1
        return super().__call__(*a, **k)


def test_plain_state_is_one_call_with_the_same_ids_and_offsets():
    from kelter.model import Featurizer
    tok = CountingEnc()
    f = Featurizer(tok, max_len=10**7, max_option_tokens=50, max_instr_tokens=50)
    state = "a [SEP] b " * 50_000
    rec = {"state": state, "question": {"instructions": "i"}, "options": [{"name": "x", "description": "d"}]}
    CountingEnc.calls = 0
    t0 = time.perf_counter()
    pc = f.pieces_batch([rec])[0]
    took = time.perf_counter() - t0
    assert CountingEnc.calls <= 5 and took < 10, (CountingEnc.calls, took)
    ref = EncLikeTok()(state, split_special_tokens=False)  # what a per-token loop would give, for comparison
    assert not (set(pc["sids"]) & f.control_ids)
    assert len(pc["sids"]) == len(pc["offs"]) and all(0 <= s <= e <= len(state) for s, e in pc["offs"])
    # a short state: ids and offsets equal the one-call-per-token result
    short = {"state": "p [SEP] q [OPT] r [MASK]", "question": {"instructions": "i"}, "options": [{"name": "x", "description": "d"}]}
    got = f.pieces_batch([short])[0]
    want_ids, want_offs = [], []
    plain = EncLikeTok()
    base = plain([short["state"]], return_offsets_mapping=True)
    for tid, (s, e) in zip(base["input_ids"][0], base["offset_mapping"][0]):
        if tid in f.control_ids:
            enc = plain(short["state"][s:e], return_offsets_mapping=True, split_special_tokens=True)
            want_ids += enc["input_ids"]
            want_offs += [[s + a, s + b] for a, b in enc["offset_mapping"]]
        else:
            want_ids.append(tid)
            want_offs.append([s, e])
    assert got["sids"] == want_ids and got["offs"] == want_offs and ref


class FlaggedTok(EncLikeTok):
    """Added tokens flagged special that are not in all_special_ids, as with some encoder tokenizers."""
    all_special_ids = [1, 2, 0, 5, 6]

    class _T:
        def __init__(self, special):
            self.special = special
    added_tokens_decoder = {50279: _T(True), 50283: _T(True), 99: _T(False), 5: _T(True)}
    SPECIAL = dict(EncLikeTok.SPECIAL, **{"<|endoftext|>": 50279, "<|padding|>": 50283})


def test_control_ids_include_every_added_token_flagged_special():
    from kelter.model import Featurizer
    f = Featurizer(FlaggedTok(), max_len=2000)
    assert {50279, 50283} <= f.control_ids and 5 not in f.control_ids and 99 not in f.control_ids
    rec = {"state": "pay <|endoftext|> now <|padding|> and [SEP] x [MASK]", "question": {"instructions": "i"},
           "options": [{"name": "x", "description": "d"}]}
    pc = f.pieces_batch([rec])[0]
    assert not (set(pc["sids"]) & f.control_ids) and 5 in pc["sids"]  # [MASK] stays the mask id


# ---- 4: the port check sets up its socket as uvicorn does
@pytest.mark.parametrize("platform,reuse", [("linux", True), ("darwin", True), ("win32", False)])
def test_check_port_socket_options_follow_uvicorn(monkeypatch, platform, reuse):
    import socket

    from kelter import server
    seen = []

    class FakeSock:
        def __init__(self, family, kind):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def setsockopt(self, level, opt, value):
            seen.append((level, opt, value))

        def bind(self, addr):
            seen.append(("bind", addr))
    monkeypatch.setattr(socket, "socket", FakeSock)
    server.check_port("127.0.0.1", 8765, platform=platform)
    assert ((socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) in seen) is reuse and ("bind", ("127.0.0.1", 8765)) in seen


# ---- 9: committed card headers are what build.py writes
def test_committed_data_card_headers_equal_card_head():
    from kelter.data import build
    for recipe, head in build.CARD_HEAD.items():
        path = ROOT / "docs" / "data_cards" / ("data_card.md" if recipe == "r2" else f"data_card_{recipe}.md")
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        assert text.startswith("\n".join(head)), f"{path.name} differs from CARD_HEAD[{recipe!r}]"
