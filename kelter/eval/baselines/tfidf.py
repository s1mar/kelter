"""TF-IDF pairwise relevance baseline (the dumb baseline).

For every (record, option) pair the features are
    hstack[ tfidf(instructions + " " + state) * tfidf(option_text), tfidf(option_text) ]
using one shared TfidfVectorizer, and a logistic regression is trained to tell
the gold option from the others. Per-record probabilities are a softmax over the
decision function of the record's options.

Order invariance: the score of an option depends only on that option and the
record text, never on its position, so this system is order-invariant by
construction and ignores `canonical`.

Training negatives: records can have up to 150 options, so a pair for every
option of 60000 records would not fit in memory. Each training record
contributes its gold option and at most `max_neg` randomly sampled other
options (seeded).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from kelter.model import option_text

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "data" / "built" / "train.jsonl"
CACHE = ROOT / "checkpoints" / "tfidf" / "model.joblib"


def record_text(r: Dict[str, Any]) -> str:
    return r["question"]["instructions"] + " " + r["state"]


def gold_name(r: Dict[str, Any]) -> str:
    g = r["gold"]
    return g if isinstance(g, str) else max(g, key=g.get)


def sample_train(n: int = 60000, seed: int = 0, path: Path = TRAIN) -> List[Dict[str, Any]]:
    """Seeded sample of non-null training records, read lazily line by line."""
    with open(path, "rb") as f:
        total = sum(1 for _ in f)
    rng = random.Random(seed)
    order = list(range(total))
    rng.shuffle(order)
    want = set(order[: int(n * 1.15) + 100])
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                if not r["null"]:
                    out.append(r)
    rng.shuffle(out)
    return out[:n]


def softmax_groups(scores: np.ndarray, sizes: Sequence[int]) -> List[np.ndarray]:
    """Softmax over consecutive groups of `scores` with the given sizes."""
    out, i = [], 0
    for k in sizes:
        z = np.asarray(scores[i:i + k], dtype=np.float64)
        z = z - z.max()
        e = np.exp(z)
        out.append(e / e.sum())
        i += k
    return out


class _Featurizer:
    """Pair features from one fitted shared vectorizer."""

    def __init__(self, vec):
        self.vec = vec
        self._opt_cache: Dict[str, Any] = {}

    def option_matrix(self, texts: Sequence[str]):
        missing = [t for t in dict.fromkeys(texts) if t not in self._opt_cache]
        if missing:
            m = self.vec.transform(missing).tocsr()
            for i, t in enumerate(missing):
                self._opt_cache[t] = m[i]
        return sp.vstack([self._opt_cache[t] for t in texts], format="csr")

    def pairs(self, recs: Sequence[Dict[str, Any]], picks: Sequence[Sequence[int]] | None = None
              ) -> Tuple[sp.csr_matrix, List[int]]:
        """Features for the chosen options of each record (all options when picks is None).
        Returns (matrix, per-record counts); rows are record-major."""
        qx = self.vec.transform([record_text(r) for r in recs]).tocsr()
        opt_texts, owner, counts = [], [], []
        for i, r in enumerate(recs):
            idx = range(len(r["options"])) if picks is None else picks[i]
            counts.append(len(idx))
            for j in idx:
                opt_texts.append(option_text(r["options"][j]))
                owner.append(i)
        ox = self.option_matrix(opt_texts)
        prod = qx[owner].multiply(ox).tocsr()
        return sp.hstack([prod, ox], format="csr"), counts


class TfidfAdapter:
    name = "tfidf"

    def __init__(self, n_train: int = 60000, seed: int = 0, C: float = 4.0, max_neg: int = 8,
                 max_features: int = 200000, cache: Path | None = CACHE):
        self.n_train, self.seed, self.C, self.max_neg = n_train, seed, C, max_neg
        self.max_features = max_features
        self.cache = Path(cache) if cache else None
        self.vec = None
        self.clf = None
        self._feat: _Featurizer | None = None

    # -- training ---------------------------------------------------------
    def fit(self, recs: Sequence[Dict[str, Any]]) -> "TfidfAdapter":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        recs = [r for r in recs if not r["null"]]
        corpus = [record_text(r) for r in recs]
        opts = list(dict.fromkeys(option_text(o) for r in recs for o in r["options"]))
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=self.max_features,
                                   dtype=np.float32)
        self.vec.fit(corpus + opts)
        self._feat = _Featurizer(self.vec)
        rng = random.Random(self.seed)
        picks, labels = [], []
        for r in recs:
            names = [o["name"] for o in r["options"]]
            g = names.index(gold_name(r))
            neg = [j for j in range(len(names)) if j != g]
            if len(neg) > self.max_neg:
                neg = rng.sample(neg, self.max_neg)
            idx = [g] + neg
            picks.append(idx)
            labels += [1] + [0] * len(neg)
        X, _ = self._feat.pairs(recs, picks)
        self.clf = LogisticRegression(C=self.C, solver="liblinear", max_iter=300)
        self.clf.fit(X, np.array(labels))
        return self

    def _ensure(self):
        if self.clf is not None:
            return
        import joblib
        if self.cache is not None and self.cache.exists():
            blob = joblib.load(self.cache)
            self.vec, self.clf = blob["vec"], blob["clf"]
            self._feat = _Featurizer(self.vec)
            return
        self.fit(sample_train(self.n_train, self.seed))
        if self.cache is not None:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"vec": self.vec, "clf": self.clf}, self.cache)

    # -- inference --------------------------------------------------------
    def predict(self, recs: List[Dict[str, Any]], canonical: bool = True, chunk: int = 200) -> List[Dict[str, Any]]:
        """Order-invariant, so `canonical` has no effect."""
        self._ensure()
        out: List[Dict[str, Any]] = []
        for s in range(0, len(recs), chunk):
            part = recs[s:s + chunk]
            X, counts = self._feat.pairs(part)
            scores = self.clf.decision_function(X)
            for r, p in zip(part, softmax_groups(scores, counts)):
                out.append({"probs": {o["name"]: float(x) for o, x in zip(r["options"], p)}, "null": None})
        return out


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("evaluation (python -m kelter.eval.baselines.tfidf)")
    # Smoke test: fit on 2000 records, predict 50 dev records.
    dev = [json.loads(l) for _, l in zip(range(400), open(ROOT / "data" / "built" / "dev.jsonl", encoding="utf-8"))]
    dev = [r for r in dev if not r["null"]][:50]
    ad = TfidfAdapter(cache=None).fit(sample_train(2000, 0))
    outs = ad.predict(dev)
    acc = np.mean([max(o["probs"], key=o["probs"].get) == gold_name(r) for o, r in zip(outs, dev)])
    print("n=%d acc=%.3f sums=%s" % (len(dev), acc, [round(sum(o["probs"].values()), 6) for o in outs[:3]]))
