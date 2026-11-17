# Data card: variant E2 (stacked score ensemble; E3 replaced it)

E2 trains no network of its own. It combines eight trained checkpoints (r2 serves twice), and its training data is theirs:

| member | role in E2 | corpus | card |
|---|---|---|---|
| r2 | chooses the option, gives the witness; null score | data/built | data_card.md |
| r1 | null score | data/built_r1 | data_card_r1.md |
| r3 | null score | data/built_r3 | data_card_r3.md |
| r4 | null score | data/built_r5 (r4's corpus, which r5 also trained on) | data_card_r5.md |
| r5 | null score | data/built_r5 | data_card_r5.md |
| r6 | null score | data/built_r6 | data_card_r6.md |
| r7 | null score | data/built_r7 | data_card_r7.md |
| rL10 | null score | data/built | data_card.md |

E2's own fitted parts come from the calibration split only: the quantile references, the gradient-boosted stacker over the members' null-score quantiles (trained on data/built/calib.jsonl's answerable records against unknowables built from calib.jsonl with kelter.eval.unknowable.build; stored in checkpoints/E2/stacker.pkl, sha256 in results/ensemble/E2.json), the logistic map to a null probability, and the conformal thresholds (results/calibration/E2.json). No test or JevBench item is used to fit anything.

Every member corpus has the same sources, the same new-source holdout and byte-identical held-out splits (calib, test, test_null, test_nudge, test_forge, witness_test, anli_test).
