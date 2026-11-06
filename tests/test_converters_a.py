"""Offline unit tests for the row-to-record mappings of converters group A."""
from kelter.data.converters import banking77, boolq, civil_comments, clinc150, massive, paws
from kelter.data.schema import validate

BANK_NAMES = ["activate_my_card", "age_limit", "apple_pay_or_google_pay"]
CLINC_NAMES = ["alarm_set", "oos", "time_zone"]
MASSIVE_NAMES = ["alarm_set", "iot_hue_lightchange", "qa_stock"]


def test_banking77_rows():
    rows = [
        {"text": "I am still waiting on my card?", "label": 0},
        {"text": "What is the age requirement?", "label": 1},
        {"text": "Can I use Google Pay?", "label": 2},
    ]
    for i, row in enumerate(rows):
        rec = banking77.row_to_record(row, "train", 10 + i, BANK_NAMES)
        validate(rec)
        assert rec["id"] == f"banking77-train-{10 + i:06d}"
        assert rec["source"] == "banking77" and rec["domain"] == "intent"
        assert rec["gold"] == BANK_NAMES[row["label"]]
        assert rec["null"] is False and rec["witness"] is None
        assert rec["question"]["type"] == "choice"
        assert rec["question"]["instructions"] == "Which banking intent does the customer's message express?"
        assert rec["provenance"] == [[0, len(row["text"]), "user"]]
    crit = banking77.row_to_record(rows[2], "test", 0, BANK_NAMES)["question"]["criteria"]
    assert crit == {
        "activate_my_card": "Activate my card",
        "age_limit": "Age limit",
        "apple_pay_or_google_pay": "Apple pay or google pay",
    }


def test_clinc150_in_scope_and_oos():
    inscope = clinc150.row_to_record({"text": "wake me at six", "intent": 0}, "validation", 3, CLINC_NAMES)
    validate(inscope)
    assert inscope["gold"] == "alarm_set" and inscope["null"] is False
    assert [o["name"] for o in inscope["options"]] == ["alarm_set", "time_zone"]
    assert inscope["question"]["instructions"] == "Which intent does the user's request express?"
    oos = clinc150.row_to_record({"text": "who won the game last night", "intent": 1}, "test", 4, CLINC_NAMES)
    validate(oos)
    assert oos["null"] is True and oos["gold"] is None
    assert "oos" not in oos["question"]["criteria"]
    assert oos["provenance"][0][2] == "user" and oos["domain"] == "intent"
    third = clinc150.row_to_record({"text": "what time is it in Rome", "intent": 2}, "train", 5, CLINC_NAMES)
    validate(third)
    assert third["gold"] == "time_zone" and third["question"]["criteria"]["time_zone"] == "Time zone"


def test_massive_rows():
    rows = [
        {"utt": "wake me up at nine am on friday", "intent": 0},
        {"utt": "make the lights blue", "intent": 1},
        {"utt": "what is the stock price of apple", "intent": 2},
    ]
    for i, row in enumerate(rows):
        rec = massive.row_to_record(row, "train", i, MASSIVE_NAMES)
        validate(rec)
        assert rec["gold"] == MASSIVE_NAMES[row["intent"]]
        assert rec["state"] == row["utt"]
        assert rec["domain"] == "intent" and rec["null"] is False
        assert rec["provenance"] == [[0, len(row["utt"]), "user"]]
    crit = massive.row_to_record(rows[1], "test", 0, MASSIVE_NAMES)["question"]["criteria"]
    assert crit["iot_hue_lightchange"] == "Iot hue lightchange"


def test_paws_rows():
    pos = paws.row_to_record({"id": 2, "sentence1": "A b c .", "sentence2": "C b a .", "label": 1}, "train", 1)
    neg = paws.row_to_record({"id": 1, "sentence1": "X .", "sentence2": "Y .", "label": 0}, "test", 0)
    for rec in (pos, neg):
        validate(rec)
        assert rec["domain"] == "paraphrase" and rec["question"]["type"] == "noul"
        assert rec["question"]["instructions"] == "Do the two sentences have the same meaning?"
        assert rec["provenance"] == [[0, len(rec["state"]), "user"]]
    assert pos["gold"] == "yes" and neg["gold"] == "no"
    assert pos["state"] == "Sentence 1: A b c .\nSentence 2: C b a ."
    assert pos["question"]["criteria"] == {
        "true": "The sentences are paraphrases with the same meaning.",
        "false": "The sentences differ in meaning.",
    }
    assert [o["name"] for o in pos["options"]] == ["no", "yes"]


def test_civil_comments_distribution():
    rows = [
        {"text": "Thanks for the clear explanation.", "toxicity": 0.0},
        {"text": "You are an idiot.", "toxicity": 0.8936170340},
        {"text": "Hmm, not sure.", "toxicity": 0.3333333432674408},
    ]
    for i, row in enumerate(rows):
        rec = civil_comments.row_to_record(row, "train", i)
        validate(rec)
        g = rec["gold"]
        assert set(g) == {"yes", "no"}
        assert g["yes"] + g["no"] == 1.0
        assert g["yes"] == round(row["toxicity"], 4)
        assert rec["provenance"] == [[0, len(row["text"]), "user"]]
        assert rec["domain"] == "toxicity" and rec["question"]["instructions"] == "Is this comment toxic?"
    assert civil_comments.row_to_record(rows[1], "test", 0)["gold"] == {"yes": 0.8936, "no": 0.1064}


def test_civil_comments_exact_sum_for_all_four_decimal_values():
    for y in range(10001):
        g = civil_comments.toxicity_gold(y / 10000)
        assert g["yes"] + g["no"] == 1.0
        assert abs(g["yes"] - y / 10000) < 1e-12


def test_civil_comments_usable_filter():
    assert civil_comments.is_usable({"text": "ok", "toxicity": 0.1})
    assert not civil_comments.is_usable({"text": "   ", "toxicity": 0.1})
    assert not civil_comments.is_usable({"text": "", "toxicity": 0.1})
    assert not civil_comments.is_usable({"text": "ok", "toxicity": float("nan")})


def test_boolq_rows():
    rows = [
        {"question": "do iran and afghanistan speak the same language", "answer": True, "passage": "Persian is spoken in both."},
        {"question": "is the sky green?", "answer": False, "passage": "The sky appears blue."},
        {"question": "Can you vote at 17", "answer": True, "passage": "Some places allow it."},
    ]
    expected_q = [
        "Do iran and afghanistan speak the same language?",
        "Is the sky green?",
        "Can you vote at 17?",
    ]
    for i, (row, q) in enumerate(zip(rows, expected_q)):
        rec = boolq.row_to_record(row, "validation", i)
        validate(rec)
        assert rec["question"]["instructions"] == q
        assert rec["state"] == row["passage"]
        assert rec["gold"] == ("yes" if row["answer"] else "no")
        assert rec["provenance"] == [[0, len(row["passage"]), "retrieved_doc"]]
        assert rec["domain"] == "qa" and rec["question"]["type"] == "noul"
    assert rows and boolq.row_to_record(rows[0], "train", 0)["question"]["criteria"] == {
        "true": "The passage supports answering yes.",
        "false": "The passage supports answering no.",
    }
