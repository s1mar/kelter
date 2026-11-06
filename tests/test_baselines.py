"""Offline tests for the baseline adapters: no downloads, no models."""
import numpy as np

from kelter.eval.baselines.qwen_logprob import (
    aggregate_rotations, letter_probs, n_rotations, rotation_order)
from kelter.eval.baselines.tfidf import TfidfAdapter, softmax_groups


def _rec(i, state, instr, opts, gold):
    options = [{"name": n, "description": d} for n, d in opts]
    return {"id": f"t{i}", "source": "toy", "split": "train", "state": state, "provenance": [[0, len(state), "user"]],
            "question": {"type": "choice", "instructions": instr, "criteria": dict(opts)},
            "options": options, "gold": gold, "null": False, "witness": None, "domain": "intent", "meta": {}}


WEATHER = [("rain", "it is raining outside"), ("sun", "the sun is shining"), ("snow", "snow is falling")]
FOOD = [("pizza", "an italian pizza dish"), ("sushi", "japanese sushi rolls"), ("salad", "fresh green salad"), ("soup", "hot soup")]
CORPUS = [
    _rec(0, "water drops fall and everything is wet, umbrella needed", "What is the weather?", WEATHER, "rain"),
    _rec(1, "bright and warm, a clear sky with sunshine", "What is the weather?", WEATHER, "sun"),
    _rec(2, "white flakes cover the ground, very cold and icy", "What is the weather?", WEATHER, "snow"),
    _rec(3, "cheese and tomato baked in an oven, italian", "Which dish is described?", FOOD, "pizza"),
    _rec(4, "raw fish and rice wrapped in seaweed, japanese", "Which dish is described?", FOOD, "sushi"),
    _rec(5, "lettuce and cucumber with green leaves, fresh", "Which dish is described?", FOOD, "salad"),
]


def _fit():
    return TfidfAdapter(cache=None, max_neg=8, max_features=5000).fit(CORPUS)


def test_pair_feature_shapes():
    ad = _fit()
    X, counts = ad._feat.pairs(CORPUS)
    assert counts == [3, 3, 3, 4, 4, 4]
    V = len(ad.vec.vocabulary_)
    assert X.shape == (sum(counts), 2 * V)
    # the product block is zero wherever the option block is zero
    prod, opt = X[:, :V], X[:, V:]
    assert ((prod != 0).astype(int) - (opt != 0).astype(int)).max() <= 0
    # picking a subset of options yields the matching rows
    Xs, cs = ad._feat.pairs(CORPUS[:1], picks=[[2, 0]])
    assert cs == [2] and Xs.shape == (2, 2 * V)
    assert abs(Xs[0] - X[2]).sum() < 1e-6 and abs(Xs[1] - X[0]).sum() < 1e-6


def test_softmax_normalization_and_interface():
    ad = _fit()
    outs = ad.predict(CORPUS)
    assert len(outs) == len(CORPUS)
    for r, o in zip(CORPUS, outs):
        assert o["null"] is None
        assert list(o["probs"]) == [x["name"] for x in r["options"]]
        assert abs(sum(o["probs"].values()) - 1.0) < 1e-9
        assert min(o["probs"].values()) >= 0
    # trained on these 6 records, the gold option gets more than uniform mass on average
    gold_p = np.mean([o["probs"][r["gold"]] for o, r in zip(outs, CORPUS)])
    uniform = np.mean([1 / len(r["options"]) for r in CORPUS])
    assert gold_p > uniform


def test_tfidf_order_invariant():
    ad = _fit()
    r = CORPUS[3]
    swapped = dict(r, options=list(reversed(r["options"])))
    a, b = ad.predict([r])[0]["probs"], ad.predict([swapped], canonical=False)[0]["probs"]
    for k in a:
        assert abs(a[k] - b[k]) < 1e-9


def test_softmax_groups():
    ps = softmax_groups(np.array([1.0, 2.0, 3.0, 0.0, 0.0]), [3, 2])
    assert [len(p) for p in ps] == [3, 2]
    assert all(abs(p.sum() - 1) < 1e-12 for p in ps)
    assert np.allclose(ps[1], [0.5, 0.5])


def test_rotation_order_and_counts():
    assert rotation_order(4, 1) == [1, 2, 3, 0]
    assert sorted(rotation_order(5, 3)) == list(range(5))
    assert n_rotations(10, True) == 4 and n_rotations(2, True) == 2 and n_rotations(10, False) == 1


def test_position_bias_cancels_over_rotations():
    n = 4
    # fake model that always prefers slot A regardless of content
    biased = [[0.85, 0.05, 0.05, 0.05] for _ in range(n)]
    avg = aggregate_rotations(biased, n)
    assert np.allclose(avg, 0.25) and abs(avg.sum() - 1) < 1e-12


def test_aggregate_recovers_content_signal():
    n = 4
    true_opt = 2
    slot_probs = []
    for k in range(n):
        order = rotation_order(n, k)
        p = np.full(n, 0.1)
        p[order.index(true_opt)] += 0.5          # content signal
        p[0] += 0.3                              # plus a slot-A bias
        slot_probs.append(p / p.sum())
    avg = aggregate_rotations(slot_probs, n)
    assert avg.argmax() == true_opt and abs(avg.sum() - 1) < 1e-12
    # a single rotation (canonical=False) keeps the bias: option in slot 0 gets inflated
    single = aggregate_rotations(slot_probs[:1], n)
    assert single[rotation_order(n, 0)[0]] > avg[rotation_order(n, 0)[0]]


def test_letter_probs_variants():
    # two token variants per slot are combined by logsumexp, then softmaxed over slots
    p = letter_probs([[0.0, 0.0], [0.0], [-50.0]])
    assert abs(p.sum() - 1) < 1e-12
    assert np.allclose(p[:2], [2 / 3, 1 / 3], atol=1e-6)
