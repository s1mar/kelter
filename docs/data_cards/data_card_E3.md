# Data card: shipped variant E3 (stacked score ensemble, default)

E3 trains no network of its own. It is E2 with three null members removed by a selection on the calibration split. It combines five trained checkpoints, and its training data is theirs:

| member | role in E3 | corpus | card |
|---|---|---|---|
| r2 | chooses the option, gives the witness; null score | data/built | data_card.md |
| r4 | null score | data/built_r5 (r4's corpus, which r5 also trained on) | data_card_r5.md |
| r5 | null score | data/built_r5 | data_card_r5.md |
| r7 | null score | data/built_r7 | data_card_r7.md |
| rL10 | null score | data/built | data_card.md |

E3's own fitted parts come from the calibration split only: the member selection (results/ensemble/E3_select.json), the quantile references, the gradient-boosted stacker over the members' null-score quantiles (trained on data/built/calib.jsonl's answerable records against unknowables built from calib.jsonl with openverdict.eval.unknowable.build; stored in checkpoints/E3/stacker.pkl, sha256 in results/ensemble/E3.json), the logistic map to a null probability, and the conformal thresholds (results/calibration/E3.json). No test or JevBench item is used to fit anything.

Every member corpus has the same sources, the same new-source holdout and byte-identical held-out splits (calib, test, test_null, test_nudge, test_forge, witness_test, anli_test). For every member, its card matches its corpus manifest, the corpus train count equals the checkpoint's n_train, and no JevBench public item is in its train, dev or calib splits.
