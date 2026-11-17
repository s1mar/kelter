"""Option order with colliding sort keys, and source hashes in the build manifest. No GPU, no network."""
import hashlib
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

A = {"name": "a\x1fb", "description": "c"}   # joins to "a\x1fb\x1fc"
B = {"name": "a", "description": "b\x1fc"}   # joins to "a\x1fb\x1fc" as well: the same key
OTHER = {"name": "z", "description": "last"}


def _names(options, order):
    return [options[i]["name"] for i in order]


def test_colliding_options_get_the_same_order_for_both_request_orders():
    from kelter.model import canonical_order
    keys = [hashlib.sha256((o["name"] + "\x1f" + o["description"]).encode("utf-8")).hexdigest() for o in (A, B)]
    assert keys[0] == keys[1]  # the collision is real
    one, two = [A, B, OTHER], [B, A, OTHER]
    assert _names(one, canonical_order(one)) == _names(two, canonical_order(two))


def test_show_order_and_the_prompt_do_not_depend_on_request_order():
    from kelter import llm
    def rec(options):
        return {"state": "s", "question": {"type": "choice", "instructions": "i"}, "options": options}
    r1, r2 = rec([A, B, OTHER]), rec([B, A, OTHER])
    o1, o2 = llm.show_order(r1, True), llm.show_order(r2, True)
    assert llm.user_text(r1, o1, "s") == llm.user_text(r2, o2, "s")


def test_the_encoder_path_scores_the_collision_in_one_order(monkeypatch):
    from kelter import model
    seen = []

    class Feat:
        def pieces_batch(self, recs):
            return [object() for _ in recs]

        def assemble(self, pc, provenance, perm):
            seen.append(perm)
            return types_ns(input_ids=[0])

    class types_ns:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    p = model.Predictor.__new__(model.Predictor)
    p.feat, p.token_budget = Feat(), 10**9
    p._score_group = lambda *a, **k: None
    recs = [{"state": "s", "provenance": [[0, 1, "user"]], "options": opts, "question": {"type": "choice"}}
            for opts in ([A, B, OTHER], [B, A, OTHER])]
    p.predict(recs, canonical=True)
    shown = [[recs[k]["options"][i]["name"] for i in perm] for k, perm in enumerate(seen)]
    assert shown[0] == shown[1]


def test_distinct_keys_keep_the_old_order():
    from kelter.model import canonical_order
    rng = random.Random(3)
    for _ in range(300):
        opts = [{"name": f"n{rng.randint(0, 99)}_{i}", "description": "d" * rng.randint(0, 5) + str(i)} for i in range(rng.randint(2, 12))]
        keys = [hashlib.sha256((o["name"] + "\x1f" + o["description"]).encode("utf-8")).hexdigest() for o in opts]
        old = sorted(range(len(opts)), key=lambda i: keys[i])
        assert canonical_order(opts) == old


def test_requirements_lock_pins_peft_and_bitsandbytes():
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    for name in ("peft", "bitsandbytes"):
        pin = next(l.split("#")[0].strip() for l in req if l.startswith(name + "=="))
        assert pin in [l.strip() for l in lock], name


def test_esnli_csvs_are_recorded_in_the_build_manifest(tmp_path, monkeypatch):
    from kelter.data import build
    from kelter.data.converters import esnli
    f = tmp_path / "esnli_dev.csv"
    f.write_bytes(b"a,b\n1,2\n")
    monkeypatch.setattr(esnli, "ensure_csv", lambda name: f)
    sink = []
    undo = build._record_downloads(sink)
    try:
        assert esnli.ensure_csv("esnli_dev.csv") == f
    finally:
        undo()
    assert sink == [{"repo": esnli.RAW_BASE, "file": "esnli_dev.csv", "revision": None, "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest()}]
    assert esnli.ensure_csv("x") == f  # the fake is back in place after undo
