import math

import numpy as np
import pytest

from kelter.eval import metrics as m


# ---------------------------------------------------------------- accuracy
def test_accuracy_hand_case():
    assert m.accuracy(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == 0.5


def test_accuracy_perfect_model():
    gold = [1, 2, 3, 2, 1]
    assert m.accuracy(list(gold), gold) == 1.0


# --------------------------------------------------------------------- ECE
def test_ece_overconfident_half_right():
    probs = np.array([[1.0, 0.0]] * 10)
    gold = np.array([0] * 5 + [1] * 5)
    assert m.ece(probs, gold) == pytest.approx(0.5)


def test_ece_hand_calibrated_is_zero():
    probs = np.array([[0.5, 0.5]] * 4)
    gold = np.array([0, 1, 0, 1])
    assert m.ece(probs, gold, n_bins=10) == pytest.approx(0.0, abs=1e-12)


def test_ece_synthetic_calibrated_close_to_zero():
    rng = np.random.default_rng(0)
    n = 20000
    conf = rng.uniform(0.5, 1.0, size=n)
    probs = np.stack([conf, 1.0 - conf], axis=1)
    correct = rng.uniform(size=n) < conf
    gold = np.where(correct, 0, 1)
    assert m.ece(probs, gold) < 0.02


def test_ece_zero_confidence_goes_to_bin_zero():
    # K=1 with prob 0 edge: confidence 0 must not index out of range.
    probs = np.array([[0.0]])
    assert m.ece(probs, np.array([0]), n_bins=5) == pytest.approx(1.0 - 0.0)


# ------------------------------------------------------------------- Brier
def test_brier_perfect_is_zero():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert m.brier(probs, np.array([0, 1])) == 0.0


def test_brier_uniform_two_class():
    probs = np.full((4, 2), 0.5)
    assert m.brier(probs, np.array([0, 1, 1, 0])) == pytest.approx(0.5)


def test_brier_distribution_gold():
    probs = np.array([[0.6, 0.4], [0.2, 0.8]])
    gold = np.array([[0.5, 0.5], [0.0, 1.0]])
    # item1: 0.01 + 0.01 = 0.02 ; item2: 0.04 + 0.04 = 0.08 ; mean 0.05
    assert m.brier(probs, gold) == pytest.approx(0.05)


# --------------------------------------------------------------- flip rate
LABELS = ["A", "B", "C"]


def _position_biased(options):
    return options[0]


def _content_model(options):
    return max(options)  # depends only on label content


def _flip(model):
    items = [list(LABELS) for _ in range(6)]
    orig = [model(o) for o in items]
    permuted = [model(list(reversed(o))) for o in items]  # labels already in caller space
    return m.paired_flip_rate(orig, permuted)


def test_flip_rate_position_biased_nonzero():
    assert _flip(_position_biased) > 0.0
    assert _flip(_position_biased) == 1.0


def test_flip_rate_order_invariant_zero():
    assert _flip(_content_model) == 0.0


def test_flip_rate_hand_case():
    assert m.paired_flip_rate([1, 2, 3, 4], [1, 9, 3, 9]) == 0.5


# --------------------------------------------------------------------- ASR
def test_asr_always_obeys():
    clean = ["A", "A", "B"]
    target = ["B", "C", "A"]
    r = m.attack_success_rate(clean, list(target), target)
    assert r == {"asr": 1.0, "n": 3, "hits": 3}


def test_asr_ignores_instruction():
    clean = ["A", "A", "B"]
    target = ["B", "C", "A"]
    r = m.attack_success_rate(clean, list(clean), target)
    assert r == {"asr": 0.0, "n": 3, "hits": 0}


def test_asr_excludes_target_equal_clean():
    clean = ["A", "A", "B", "B"]
    target = ["A", "C", "B", "A"]
    attacked = ["A", "C", "B", "B"]
    r = m.attack_success_rate(clean, attacked, target)
    assert r["n"] == 2 and r["hits"] == 1 and r["asr"] == 0.5


def test_asr_zero_denominator_is_nan():
    r = m.attack_success_rate(["A"], ["A"], ["A"])
    assert r["n"] == 0 and math.isnan(r["asr"])


def test_asr_aliases_same_math():
    args = (["A", "B"], ["C", "B"], ["C", "A"])
    base = m.attack_success_rate(*args)
    assert m.nudge_asr(*args) == base
    assert m.forge_asr(*args) == base


# -------------------------------------------------------------- null recall
def test_null_recall_never_nulls():
    r = m.null_recall_at_fpr([0.3] * 40, [0.3] * 10)
    assert r["recall"] == 0.0
    assert r["realized_fpr"] == 0.0


def test_null_recall_perfect_separator():
    ans = np.linspace(0.0, 0.4, 100)
    unk = np.linspace(0.6, 1.0, 50)
    r = m.null_recall_at_fpr(ans, unk, fpr=0.05)
    assert r["recall"] == 1.0
    assert r["realized_fpr"] == 0.0 or r["realized_fpr"] <= 0.05


def test_null_recall_hand_threshold():
    ans = list(range(20))  # 0..19, fpr .1 -> k = 3, t = 3rd largest = 17
    r = m.null_recall_at_fpr(ans, [18, 10, 25], fpr=0.1)
    assert r["threshold"] == 17.0
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["realized_fpr"] == pytest.approx(2 / 20)


def test_null_recall_small_n_uses_neg_inf():
    r = m.null_recall_at_fpr([0.1, 0.2], [0.0], fpr=1.0)
    assert r["threshold"] == float("-inf")
    assert r["realized_fpr"] == 1.0


def test_null_recall_fpr_bound_random_trials():
    rng = np.random.default_rng(123)
    for _ in range(200):
        n_a = int(rng.integers(1, 300))
        n_u = int(rng.integers(1, 100))
        fpr = float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.3]))
        ans = rng.normal(size=n_a)
        unk = rng.normal(loc=1.0, size=n_u)
        r = m.null_recall_at_fpr(ans, unk, fpr=fpr)
        assert r["realized_fpr"] <= fpr + 1e-12
        assert (r["realized_fpr"] * n_a) <= math.floor(fpr * n_a) + 1e-9


# ------------------------------------------------- necessity / sufficiency
def test_necessity_ignores_witness_is_zero():
    correct = [True, True, True]
    full = ["A", "B", "C"]
    assert m.witness_necessity(correct, full, list(full)) == 0.0


def test_necessity_depends_only_on_witness_is_one():
    correct = [True, True, True]
    full = ["A", "B", "C"]
    masked = ["Z", "Z", "Z"]
    assert m.witness_necessity(correct, full, masked) == 1.0


def test_necessity_only_counts_correct_items():
    correct = [True, False, True, False]
    full = ["A", "B", "C", "D"]
    masked = ["A", "X", "Q", "X"]
    assert m.witness_necessity(correct, full, masked) == 0.5


def test_necessity_no_correct_is_nan():
    assert math.isnan(m.witness_necessity([False, False], ["A", "B"], ["A", "B"]))


def test_sufficiency_witness_only_matches_is_one():
    correct = [True, True]
    full = ["A", "B"]
    assert m.witness_sufficiency(correct, full, list(full)) == 1.0


def test_sufficiency_witness_irrelevant_is_zero():
    correct = [True, True]
    assert m.witness_sufficiency(correct, ["A", "B"], ["Z", "Z"]) == 0.0


def test_sufficiency_only_counts_correct_items():
    correct = [True, False, True]
    full = ["A", "B", "C"]
    only = ["A", "B", "X"]
    assert m.witness_sufficiency(correct, full, only) == 0.5


def test_sufficiency_no_correct_is_nan():
    assert math.isnan(m.witness_sufficiency([False], ["A"], ["A"]))


# ----------------------------------------------------------------- span F1
def test_span_f1_exact_match():
    assert m.span_f1([(0, 5)], [(0, 5)]) == 1.0


def test_span_f1_partial_overlap():
    # pred 0..10, gold 5..15 -> inter 5, P=.5, R=.5, F1=.5
    assert m.span_f1([(0, 10)], [(5, 15)]) == pytest.approx(0.5)


def test_span_f1_union_of_multiple_spans():
    # pred union = 0..4 and 4..8 -> 8 chars ; gold 0..4 -> 4 chars
    # inter 4, P=.5, R=1, F1 = 2*.5/1.5 = 2/3
    assert m.span_f1([(0, 4), (4, 8)], [(0, 4)]) == pytest.approx(2 / 3)


def test_span_f1_overlapping_spans_not_double_counted():
    assert m.span_f1([(0, 6), (3, 9)], [(0, 9)]) == 1.0


def test_span_f1_disjoint_and_empty_cases():
    assert m.span_f1([(0, 3)], [(5, 8)]) == 0.0
    assert m.span_f1([], []) == 1.0
    assert m.span_f1([(0, 3)], []) == 0.0
    assert m.span_f1([], [(0, 3)]) == 0.0


def test_mean_span_f1():
    preds = [[(0, 10)], []]
    golds = [[(5, 15)], []]
    assert m.mean_span_f1(preds, golds) == pytest.approx(0.75)


# ----------------------------------------------------------------- latency
def test_latency_percentiles_known_list():
    r = m.latency_percentiles([10, 20, 30, 40, 50])
    assert r["p50"] == 30.0
    assert r["p95"] == pytest.approx(48.0)  # 40 + 0.8 * 10
    assert r["mean"] == 30.0
    assert r["n"] == 5


# --------------------------------------------------------------- conformal
def test_conformal_coverage_hand_case():
    sets = [{"A"}, {"A", "B"}, {"C"}, {"A", "B", "C"}]
    gold = ["A", "B", "A", "C"]
    r = m.conformal_coverage(sets, gold)
    assert r["coverage"] == 0.75
    assert r["mean_set_size"] == 1.75
    assert r["singleton_rate"] == 0.5


# ------------------------------------------------------- validation errors
MISMATCH_CALLS = [
    lambda: m.accuracy([1, 2], [1]),
    lambda: m.ece(np.full((3, 2), 0.5), np.array([0, 1])),
    lambda: m.brier(np.full((3, 2), 0.5), np.array([0, 1])),
    lambda: m.paired_flip_rate([1, 2], [1]),
    lambda: m.attack_success_rate([1], [1, 2], [1]),
    lambda: m.witness_necessity([True], [1, 2], [1, 2]),
    lambda: m.witness_sufficiency([True, True], [1, 2], [1]),
    lambda: m.mean_span_f1([[(0, 1)]], []),
    lambda: m.conformal_coverage([{1}], [1, 2]),
]

EMPTY_CALLS = [
    lambda: m.accuracy([], []),
    lambda: m.ece(np.zeros((0, 2)), np.zeros(0, dtype=int)),
    lambda: m.brier(np.zeros((0, 2)), np.zeros(0, dtype=int)),
    lambda: m.paired_flip_rate([], []),
    lambda: m.attack_success_rate([], [], []),
    lambda: m.null_recall_at_fpr([], [1.0]),
    lambda: m.null_recall_at_fpr([1.0], []),
    lambda: m.witness_necessity([], [], []),
    lambda: m.witness_sufficiency([], [], []),
    lambda: m.mean_span_f1([], []),
    lambda: m.latency_percentiles([]),
    lambda: m.conformal_coverage([], []),
]


@pytest.mark.parametrize("call", MISMATCH_CALLS)
def test_length_mismatch_raises(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("call", EMPTY_CALLS)
def test_empty_input_raises(call):
    with pytest.raises(ValueError):
        call()
