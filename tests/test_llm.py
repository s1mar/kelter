"""Pure helpers of the LLM-backed predictor; no model loaded."""
from openverdict.llm import LETTERS, NULL_TEXT, gold_slot, show_order, user_text


def _rec(qtype="choice", null=False, gold="b"):
    opts = [{"name": n, "description": f"option {n}"} for n in ("a", "b", "c")]
    return {"question": {"type": qtype, "instructions": "Pick one."}, "options": opts, "state": "s",
            "null": null, "gold": None if null else gold}


def test_canonical_order_ignores_request_order():
    r = _rec()
    r2 = dict(r, options=list(reversed(r["options"])))
    names = lambda x: [x["options"][i]["name"] for i in show_order(x, True)]
    assert names(r) == names(r2)


def test_score_levels_keep_their_order():
    assert show_order(_rec("score"), True) == [0, 1, 2]


def test_null_letter_follows_the_options():
    r = _rec(null=True)
    order = [0, 1, 2]
    assert f"{LETTERS[3]}. {NULL_TEXT}" in user_text(r, order, r["state"])
    assert gold_slot(r, order) == 3
    assert gold_slot(_rec(), [2, 1, 0]) == 1
