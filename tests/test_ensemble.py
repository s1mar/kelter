"""Score ensemble pieces: logistic fit and probability assembly."""
from __future__ import annotations

import numpy as np

from openverdict import ensemble as E
from openverdict import ensemble_select as S


def test_fit_logistic_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, x.size) < 1 / (1 + np.exp(-(6 * x - 4)))).astype(float)
    a, b = E.fit_logistic(x, y)
    assert abs(a - 6) < 0.3 and abs(b + 4) < 0.2


def test_predict_scales_options_and_keeps_witness():
    class P:
        def __init__(self, outs):
            self.outs = outs

        def predict(self, recs, canonical=True, temperature=True):
            return self.outs

    ens = E.Ensemble.__new__(E.Ensemble)
    ens.opt = P([{"probs": {"a": 0.6, "b": 0.2}, "null": 0.2, "witness": [[0, 3]]}])
    ens.nulls = {"m1": P([{"probs": {"a": 0.9}, "null": 0.5}]), "m2": P([{"probs": {"a": 0.9}, "null": 0.1}])}
    ens.refs = {"m1": np.array([0.1, 0.2, 0.3, 0.4]), "m2": np.array([0.1, 0.2, 0.3, 0.4])}
    ens.a, ens.b = 0.0, 0.0  # null probability 0.5 regardless of quantile
    ens.temperature, ens.stacker = True, None
    (o,) = ens.predict([{}])
    assert o["null_quantile"] == (1.0 + 0.25) / 2
    assert abs(o["null"] - 0.5) < 1e-12
    assert abs(o["probs"]["a"] - 0.75 * 0.5) < 1e-12 and abs(o["probs"]["b"] - 0.25 * 0.5) < 1e-12
    assert o["witness"] == [[0, 3]]


def test_stacked_combiner_feeds_member_quantiles_in_order():
    class P:
        def __init__(self, null):
            self.null = null

        def predict(self, recs, canonical=True, temperature=True):
            return [{"probs": {"a": 0.5, "b": 0.5}, "null": self.null}]

    class Stacker:
        def predict_proba(self, X):
            self.seen = X
            return np.array([[0.3, 0.7]])

    ens = E.Ensemble.__new__(E.Ensemble)
    ens.opt = P(0.0)
    ens.nulls = {"m1": P(0.35), "m2": P(0.05)}
    ens.refs = {"m1": np.array([0.1, 0.2, 0.3, 0.4]), "m2": np.array([0.1, 0.2, 0.3, 0.4])}
    ens.a, ens.b, ens.temperature = 1.0, 0.0, True
    ens.stacker = Stacker()
    (o,) = ens.predict([{}])
    assert ens.stacker.seen.tolist() == [[0.75, 0.0]]
    assert o["null_quantile"] == 0.7
    assert abs(o["null"] - 1 / (1 + np.exp(-0.7))) < 1e-12


def test_recall_at_5fpr_thresholds_on_answerable_scores():
    y = np.r_[np.zeros(100), np.ones(4)]
    s = np.r_[np.arange(100) / 100.0, [0.99, 0.95, 0.5, 0.1]]
    # 95th percentile of the answerable scores is 0.9405; two unknowables lie above it
    assert S.recall_at_5fpr(s, y) == 0.5
