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

The four null-score members (r3, r5, r7, rL10) were chosen on the calibration split, before E1 was scored on the test split: unknowable items were built from calib.jsonl with the same builder as the test set, six checkpoints were scored on them, and this set was the best of 57 combinations (recall 0.770 at a 5% false-alarm rate, 0.773 and 0.768 on two halves). Picking the best of 57 biases that figure upward. The scripts and outputs of that selection are not in this repository, so it cannot be rerun from here. The same four are also the four single checkpoints with the highest null recall on the test split, r2 aside (results/eval); the selection did not read the test split.

Every member corpus has the same sources, the same new-source holdout and byte-identical held-out splits (calib, test, test_null, test_nudge, test_forge, witness_test, anli_test). For every member, its card matches its corpus manifest, the corpus train count equals the checkpoint's n_train, and no JevBench public item is in its train, dev or calib splits.
