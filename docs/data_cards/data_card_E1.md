# Data card: shipped variant E1 (score ensemble)

E1 trains nothing of its own. It combines five trained checkpoints, and its data is theirs:

| member | role in E1 | corpus | card |
|---|---|---|---|
| r2 | chooses the option, gives the witness | data/built | data_card.md |
| r3 | null score | data/built_r3 | data_card_r3.md |
| r5 | null score | data/built_r5 | data_card_r5.md |
| r7 | null score | data/built_r7 | data_card_r7.md |
| rL10 | null score | data/built | data_card.md |

The only fitted part of E1 itself, the quantile references and the logistic map from the mean null quantile to a null probability, is fitted on the calibration split (data/built/calib.jsonl) and stored in results/ensemble/E1.json; the conformal thresholds are fitted on the same split (results/calibration/E1.json).

Every member corpus has the same sources, the same new-source holdout and byte-identical held-out splits (calib, test, test_null, test_nudge, test_forge, witness_test, anli_test). For every member, its card matches its corpus manifest, the corpus train count equals the checkpoint's n_train, and no JevBench public item is in its train, dev or calib splits.
