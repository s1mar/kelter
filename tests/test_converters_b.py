"""Offline unit tests for the mnli, anli, esnli, squad2, arc and mmlu row mappings."""
from __future__ import annotations

import numpy as np

from openverdict.data.converters import anli, arc, esnli, mmlu, mnli, squad2
from openverdict.data.schema import validate


# ---------------------------------------------------------------- NLI

def test_mnli_mapping_and_labels():
    base = {"premise": "A man sleeps.", "hypothesis": "A man is awake.", "genre": "fiction", "pairID": "1c"}
    for lab, name in [(0, "entailment"), (1, "neutral"), (2, "contradiction")]:
        rec = mnli.row_to_record({**base, "label": lab}, "validation", 7)
        validate(rec)
        assert rec["gold"] == name
        assert rec["id"] == "mnli-validation-000007"
        assert rec["state"] == "Premise: A man sleeps.\nHypothesis: A man is awake."
        assert rec["domain"] == "nli" and rec["question"]["type"] == "choice"
        assert rec["provenance"] == [[0, len(rec["state"]), "user"]]
        assert [o["name"] for o in rec["options"]] == ["entailment", "neutral", "contradiction"]
    assert mnli.row_to_record({**base, "label": -1}, "train", 0) is None


def test_anli_mapping_round_in_meta():
    row = {"uid": "u1", "premise": "P text.", "hypothesis": "H text.", "label": 2, "reason": "x"}
    rec = anli.row_to_record(row, "train", 12, 3)
    validate(rec)
    assert rec["gold"] == "contradiction"
    assert rec["meta"]["round"] == 3
    assert rec["id"] == "anli-train-000012"
    assert anli.row_to_record({**row, "label": -1}, "train", 0, 1) is None


# ---------------------------------------------------------------- e-SNLI

def _esnli_row(s1, s2, m1, m2, label="entailment"):
    return {"pairID": "p#1", "gold_label": label, "Sentence1": s1, "Sentence2": s2,
            "Sentence1_marked_1": m1, "Sentence2_marked_1": m2}


def test_esnli_dog_highlight_is_exact_substring():
    row = _esnli_row("A dog runs fast.", "The dog runs.", "A *dog* runs fast.", "The *dog* runs.")
    rec = esnli.row_to_record(row, "train", 3)
    validate(rec)
    subs = [rec["state"][s:e] for s, e in rec["witness"]]
    assert subs == ["dog", "dog"]
    # first span sits in the premise, second after the hypothesis prefix
    assert rec["witness"][0][0] == len("Premise: ") + 2
    assert rec["state"].index("\nHypothesis: ") < rec["witness"][1][0]


def test_esnli_fixture_star_dog_runs():
    row = _esnli_row("dog runs", "it runs", "*dog* runs", "it runs")
    rec = esnli.row_to_record(row, "train", 0)
    validate(rec)
    assert [rec["state"][s:e] for s, e in rec["witness"]] == ["dog"]


def test_esnli_adjacent_words_merge_and_punctuation_kept():
    text = "Two women are embracing while holding to go packages."
    marked = " Two women are embracing while holding *to* *go* *packages.*"
    assert esnli.highlight_spans(text, marked) == [[text.index("to go"), len(text)]]
    row = _esnli_row(text, "Two woman are holding packages.", marked, " Two woman are *holding* *packages.*")
    rec = esnli.row_to_record(row, "validation", 1)
    validate(rec)
    assert [rec["state"][s:e] for s, e in rec["witness"]] == ["to go packages.", "holding packages."]


def test_esnli_non_adjacent_spans_and_multiword_star_form():
    text = "a b c d e"
    assert [text[s:e] for s, e in esnli.highlight_spans(text, "*a* b *c* *d* e")] == ["a", "c d"]
    assert [text[s:e] for s, e in esnli.highlight_spans(text, "a *b c* d e")] == ["b c"]


def test_esnli_no_highlights_gives_none_witness():
    row = _esnli_row("A man sleeps.", "A man is up.", "A man sleeps.", "A man is up.", "contradiction")
    rec = esnli.row_to_record(row, "test", 0)
    validate(rec)
    assert rec["witness"] is None
    assert esnli.row_to_record({**row, "gold_label": "-"}, "test", 0) is None


def test_esnli_misaligned_markup_is_ignored():
    assert esnli.highlight_spans("a b c", "*a* b") == []
    assert esnli.highlight_spans("a b c", "*x* b c") == []


# ---------------------------------------------------------------- SQuAD 2.0

CONTEXT = "The Normans were the people who gave their name to Normandy, a region in France. Rollo led them."


def _sq(qid, question, texts, starts):
    return {"id": qid, "title": "Normans", "context": CONTEXT, "question": question,
            "answers": {"text": np.array(texts, dtype=object), "answer_start": np.array(starts)}}


def _group():
    return [
        _sq("q0", "Who led the Normans?", ["Rollo"], [CONTEXT.index("Rollo")]),
        _sq("q1", "Where is Normandy?", ["France", "france"], [CONTEXT.index("France")] * 2),
        _sq("q2", "Who were the Normans?", ["the people"], [CONTEXT.index("the people")]),
        _sq("q3", "What did they name?", ["Normandy"], [CONTEXT.index("Normandy")]),
        _sq("q4", "Unanswerable one?", [], []),
        _sq("q5", "Another gold FRANCE?", ["FRANCE"], [CONTEXT.index("France")]),
    ]


def test_squad_answerable_witness_and_distractors():
    rows = _group()
    others = squad2.other_answer_texts(rows, 0)
    assert "rollo" not in [o.casefold() for o in others]
    rec = squad2.row_to_record(rows[0], "train", 5, others, seed=1)
    validate(rec)
    assert rec["id"] == "squad2-train-000005"
    assert rec["null"] is False and rec["domain"] == "qa"
    assert rec["provenance"] == [[0, len(CONTEXT), "retrieved_doc"]]
    (s, e), = rec["witness"]
    assert rec["state"][s:e] == "Rollo"
    gold_text = rec["question"]["criteria"][rec["gold"]]
    assert gold_text == "Rollo"
    assert [o["name"] for o in rec["options"]] == ["A", "B", "C", "D"]
    texts = [o["description"] for o in rec["options"]]
    assert len({t.casefold() for t in texts}) == 4
    assert sum(t.casefold() == "rollo" for t in texts) == 1
    assert rec["question"]["instructions"] == "Who led the Normans?"


def test_squad_case_variants_of_gold_never_distractors():
    rows = _group()
    # question q1 has gold "France"/"france"; q5 gold "FRANCE" is another question's answer
    others = squad2.other_answer_texts(rows, 1)
    assert "france" not in [o.casefold() for o in others]
    rec = squad2.row_to_record(rows[1], "train", 1, others, seed=3)
    validate(rec)
    texts = [o["description"].casefold() for o in rec["options"]]
    assert texts.count("france") == 1


def test_squad_unanswerable_is_null():
    rows = _group()
    others = squad2.other_answer_texts(rows, 4)
    rec = squad2.row_to_record(rows[4], "validation", 4, others, seed=0)
    validate(rec)
    assert rec["null"] is True and rec["gold"] is None and rec["witness"] is None
    assert len(rec["options"]) == 4
    assert len({o["description"].casefold() for o in rec["options"]}) == 4


def test_squad_skips_when_not_enough_distinct_answers():
    rows = _group()[:3]  # q0 has only 2 other distinct answers (France, the people)
    others = squad2.other_answer_texts(rows, 0)
    assert squad2.is_eligible(rows[0], others) is False
    assert squad2.row_to_record(rows[0], "train", 0, others) is None


def test_squad_deterministic_and_context_split_disjoint():
    rows = _group()
    others = squad2.other_answer_texts(rows, 0)
    a = squad2.row_to_record(rows[0], "train", 0, others, seed=9)
    b = squad2.row_to_record(rows[0], "train", 0, others, seed=9)
    assert a == b
    ctxs = [f"context {i}" for i in range(10)] * 2
    assign = squad2.split_contexts(ctxs, seed=0)
    assert set(assign.values()) == {"validation", "test"}
    assert sum(v == "validation" for v in assign.values()) == 5
    assert assign == squad2.split_contexts(list(reversed(ctxs)), seed=0)


# ---------------------------------------------------------------- ARC

def test_arc_label_normalization_numeric():
    row = {"id": "x1", "question": "Which is a planet?", "answerKey": "2",
           "choices": {"text": np.array(["sun", "mars", "moon", "star"], dtype=object),
                       "label": np.array(["1", "2", "3", "4"], dtype=object)}}
    rec = arc.row_to_record(row, "arc_easy", "train", 4)
    validate(rec)
    assert rec["gold"] == "B"
    assert rec["question"]["criteria"] == {"A": "sun", "B": "mars", "C": "moon", "D": "star"}
    assert rec["id"] == "arc_easy-train-000004" and rec["source"] == "arc_easy"
    assert rec["domain"] == "mcq" and rec["state"] == "Which is a planet?"
    assert rec["provenance"] == [[0, len(rec["state"]), "user"]]


def test_arc_letter_labels_and_three_choices_and_bad_key():
    row = {"id": "x2", "question": "Q?", "answerKey": "C",
           "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]}}
    rec = arc.row_to_record(row, "arc_challenge", "test", 0)
    validate(rec)
    assert rec["gold"] == "C" and len(rec["options"]) == 3
    assert arc.row_to_record({**row, "answerKey": "E"}, "arc_challenge", "test", 0) is None
    assert arc.normalize_label("5") == "E"


# ---------------------------------------------------------------- MMLU

def test_mmlu_mapping():
    row = {"question": "2+2?", "subject": "elementary_mathematics",
           "choices": np.array(["3", "4", "5", "6"], dtype=object), "answer": 1}
    rec = mmlu.row_to_record(row, "test", 10)
    validate(rec)
    assert rec["gold"] == "B" and rec["meta"]["subject"] == "elementary_mathematics"
    assert rec["id"] == "mmlu-test-000010"
    assert [o["name"] for o in rec["options"]] == ["A", "B", "C", "D"]
    assert mmlu.row_to_record({**row, "answer": 7}, "test", 0) is None
