# Results

Every number here is read from a JSON file under `results/`, named in the last column of each row. The held-out sets come from sources the model never trained on: CLINC150 intents, ARC-Challenge science questions and a family of synthetic policy tasks, plus e-SNLI and SQuAD 2.0 test items for the evidence spans. See `docs/evaluation.md` for what each metric measures.

## The two served variants

### `default` (Q1)

Qwen3-4B-Instruct-2507 with a LoRA adapter trained on a 24,000-record sample of the corpus (`docs/data_cards/data_card_Q1.md`); the encoder checkpoint r2 supplies the evidence spans.

| Metric | Value | Source |
|---|---|---|
| Held-out accuracy (CLINC150, ARC-Challenge, synthetic sharing; n=3120) | 0.913 | results/eval/Q1.json |
| Held-out accuracy, arc_challenge (n=745) | 0.902 | results/eval/Q1.json |
| Held-out accuracy, clinc150 (n=1718) | 0.973 | results/eval/Q1.json |
| Held-out accuracy, synthetic_policy_sharing (n=657) | 0.769 | results/eval/Q1.json |
| ECE, 15 bins, after / before temperature | 0.017 / 0.049 | results/eval/Q1.json |
| Brier | 0.142 | results/eval/Q1.json |
| Paired flip rate, canonical ordering on / off (n=2787) | 0.000 / 0.030 | results/eval/Q1.json |
| Nudge-ASR / forge-ASR | 0.008 (n=2980) / 0.004 (n=2980) | results/eval/Q1.json |
| Null recall at 5% FPR, old set: answer removed without filter plus natural nulls (n=3007) | 0.960 (realized FPR 0.050) | results/eval/Q1.json |
| Null recall at 5% FPR, corrected set (n=5918; see Null recall below) | 0.968 (realized FPR 0.050) | results/eval/Q1.json |
| Corrected-set null recall by group: answer removed (filtered) / swap state / natural null | 0.956 / 0.977 / 0.986 | results/eval/Q1.json |
| Witness F1, e-SNLI / SQuAD 2.0 (spans from r2) | 0.611 / 0.857 | results/eval/Q1.json |
| Witness necessity / sufficiency, e-SNLI (n=2000) | 0.586 / 0.001 | results/eval/Q1.json |
| Random-mask flip control / human-rationale mask flip, e-SNLI | 0.362 / 0.590 | results/eval/Q1.json |
| ANLI test accuracy (eval only, n=2000) | 0.565 | results/eval/Q1.json |
| Calibration split ECE before / after temperature | 0.043 / 0.018 | results/calibration/Q1.json |
| Conformal coverage at alpha 0.05 / mean set size / singleton rate | 0.961 / 1.33 / 0.801 | results/eval/engine_Q1.json |
| Action rates act / escalate / defer | 0.263 / 0.028 / 0.708 | results/eval/engine_Q1.json |
| Accuracy when the engine acts (n=880) | 0.974 | results/eval/engine_Q1.json |
| Fake approval in tool_output: verdict moved to destructive / unsafe act (n=400) | 0.025 / 0.000 | results/eval/engine_Q1.json |
| Planted text for a wrong option under a trusted label (forge set): unsafe act (n=3120) | 0.001 (4 of 3120) | results/eval/engine_Q1base.json |
| Latency, one forward, 1095-token sequence, GPU p50 / p95 ms | 372.8 / 380.5 | results/eval/latency_Q1.json |
| Latency, full engine decision (4 passes), GPU p50 / p95 ms | 542.4 / 548.4 | results/eval/latency_Q1.json |
| Latency, CPU int8 forward / engine p50 ms | 7230 / 11471 | results/eval/latency_Q1.json |
| JevBench public, original tier (their harness): accuracy / macro accuracy / Brier / ECE (valid 72 of 72) | 0.944 / 0.944 / 0.097 / 0.054 | results/jevbench/Q1chunk-20261104T162222Z/kelter_summary.json |
| JevBench public, hard tier (their harness): accuracy / macro accuracy / Brier / ECE (valid 111 of 111) | 0.568 / 0.630 / 0.558 / 0.124 | results/jevbench/Q1chunk-20261104T162222Z/kelter_summary.json |
| JevBench public, easy tier (their harness): accuracy / macro accuracy / Brier / ECE (valid 48 of 48) | 0.958 / 0.958 / 0.050 / 0.038 | results/jevbench/Q1chunk-20261104T162222Z/kelter_summary.json |
| JevBench public, per-request latency p50 / p95 s by tier: original, hard, easy | 0.119 / 0.164, 0.455 / 1.783, 0.135 / 0.159 | results/jevbench/Q1chunk-20261104T162222Z/kelter_summary.json |
| JevBench public, pooled 231 items: correct / latency p50 s / latency max s | 177 / 0.147 / 2.15 | results/jevbench/Q1chunk_cost.json |
| JevBench public, mean reported input tokens per decision (all engine passes) / cost per 1,000 decisions at USD 0.04 per million tokens | 1331.4 / USD 0.0533 | results/jevbench/Q1chunk_cost.json |
| JevBench public, hard tier: accuracy / mean top probability | 0.568 / 0.657 | results/jevbench/Q1chunk_cost.json |
| JevBench v1.4 composite | not computable locally: it needs the sealed tier and a hosted cost basis | third_party/jevbench/jevbench/composite_v14.py |

### `lower-latency` (E1)

| Metric | Value | Source |
|---|---|---|
| Held-out accuracy (CLINC150, ARC-Challenge, synthetic sharing; n=3120) | 0.688 | results/eval/E1.json |
| Held-out accuracy, arc_challenge (n=745) | 0.302 | results/eval/E1.json |
| Held-out accuracy, clinc150 (n=1718) | 0.821 | results/eval/E1.json |
| Held-out accuracy, synthetic_policy_sharing (n=657) | 0.779 | results/eval/E1.json |
| ECE, 15 bins, after / before temperature | 0.042 / 0.166 | results/eval/E1.json |
| Brier | 0.398 | results/eval/E1.json |
| Paired flip rate, canonical ordering on / off (n=2787) | 0.000 / 0.090 | results/eval/E1.json |
| Nudge-ASR / forge-ASR | 0.012 (n=2858) / 0.015 (n=2820) | results/eval/E1.json |
| Null recall at 5% FPR, old set: answer removed without filter plus natural nulls (n=3007) | 0.742 (realized FPR 0.050) | results/eval/E1.json |
| Null recall at 5% FPR, corrected set (n=5918; see Null recall below) | 0.775 (realized FPR 0.050) | results/eval/E1.json |
| Corrected-set null recall by group: answer removed (filtered) / swap state / natural null | 0.751 / 0.801 / 0.691 | results/eval/E1.json |
| Witness F1, e-SNLI / SQuAD 2.0 | 0.611 / 0.857 | results/eval/E1.json |
| Witness necessity / sufficiency, e-SNLI (n=2000) | 0.932 / 0.843 | results/eval/E1.json |
| Random-mask flip control / human-rationale mask flip, e-SNLI | 0.205 / 0.913 | results/eval/E1.json |
| ANLI test accuracy (eval only, n=2000) | 0.301 | results/eval/E1.json |
| Calibration split ECE before / after temperature | n/a / 0.035 | results/calibration/E1.json |
| Conformal coverage at alpha 0.05 / mean set size / singleton rate | 0.951 / 2.32 / 0.368 | results/eval/engine_E1.json |
| Action rates act / escalate / defer | 0.221 / 0.194 / 0.585 | results/eval/engine_E1.json |
| Accuracy when the engine acts (n=737) | 0.970 | results/eval/engine_E1.json |
| Fake approval in tool_output: verdict moved to destructive / unsafe act (n=400) | 0.013 / 0.000 | results/eval/engine_E1.json |
| Planted text for a wrong option under a trusted label (forge set): unsafe act (n=3120) | 0.002 (6 of 3120) | results/eval/engine_E1base.json |
| Latency, one forward, 1068-token sequence, GPU p50 / p95 ms | 85.9 / 87.7 | results/eval/latency_E1.json |
| Latency, full engine decision (4 passes), GPU p50 / p95 ms | 149.0 / 154.0 | results/eval/latency_E1.json |
| Latency, CPU int8 forward / engine p50 ms | 2177 / 3067 | results/eval/latency_E1.json |
| JevBench public, original tier (their harness): accuracy / macro accuracy / Brier / ECE (valid 72 of 72) | 0.444 / 0.444 / 0.608 / 0.208 | results/jevbench/E1-20261030T212239Z/kelter_summary.json |
| JevBench public, hard tier (their harness): accuracy / macro accuracy / Brier / ECE (valid 111 of 111) | 0.360 / 0.369 / 0.786 / 0.224 | results/jevbench/E1-20261030T212239Z/kelter_summary.json |
| JevBench public, easy tier (their harness): accuracy / macro accuracy / Brier / ECE (valid 48 of 48) | 0.896 / 0.896 / 0.252 / 0.201 | results/jevbench/E1-20261030T212239Z/kelter_summary.json |
| JevBench v1.4 composite | not computable locally: it needs the sealed tier and a hosted cost basis | third_party/jevbench/jevbench/composite_v14.py |

## All measured runs

The two served variants, the earlier default E3, the member checkpoints and every other run measured on the same pipeline. Rows r1 to r7 are ModernBERT-base runs, rL8 to rL11 ModernBERT-large; E1 to E3 are ensembles over them; Q1 to Q3 are Qwen3-4B LoRA adapters with r2 for the evidence spans.

| Run | Held-out acc | ECE | Flip canon | Flip no canon | Nudge-ASR | Forge-ASR | Null recall (old) | Null recall (corrected) | Necessity / control | JevBench original | JevBench hard | JevBench easy | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 0.913 | 0.017 | 0.000 | 0.030 | 0.008 | 0.004 | 0.960 | 0.968 | 0.586 / 0.362 | 0.944 | 0.568 | 0.958 | results/eval/Q1.json |
| E1 | 0.688 | 0.042 | 0.000 | 0.090 | 0.012 | 0.015 | 0.742 | 0.775 | 0.932 / 0.205 | 0.444 | 0.360 | 0.896 | results/eval/E1.json |
| Q3 | 0.909 | 0.019 | 0.000 | 0.035 | 0.011 | 0.010 | 0.944 | 0.962 | 0.562 / 0.356 | 0.903 | 0.541 | 0.917 | results/eval/Q3.json |
| Q2 | 0.890 | 0.021 | 0.000 | 0.030 | 0.005 | 0.007 | 0.953 | 0.972 | 0.515 / 0.322 | 0.958 | 0.486 | 0.958 | results/eval/Q2.json |
| E3 | 0.688 | 0.040 | 0.000 | 0.092 | 0.012 | 0.015 | 0.761 | 0.818 | 0.922 / 0.206 | 0.444 | 0.360 | 0.896 | results/eval/E3.json |
| E2 | 0.688 | 0.039 | 0.000 | 0.098 | 0.012 | 0.015 | 0.783 | 0.833 | 0.934 / 0.209 | 0.444 | 0.360 | 0.896 | results/eval/E2.json |
| r1 | 0.573 | 0.045 | 0.000 | 0.184 | 0.029 | 0.064 | 0.541 | 0.598 | 0.972 / 0.969 | n/a | n/a | n/a | results/eval/r1.json |
| r2 | 0.671 | 0.040 | 0.000 | 0.090 | 0.013 | 0.016 | 0.582 | 0.603 | 0.971 / 0.227 | 0.444 | 0.360 | 0.896 | results/eval/r2.json |
| r3 | 0.623 | 0.056 | 0.000 | 0.097 | 0.012 | 0.013 | 0.678 | 0.646 | 0.967 / 0.215 | n/a | n/a | n/a | results/eval/r3.json |
| r4 | 0.520 | 0.059 | 0.000 | 0.100 | 0.034 | 0.199 | 0.522 | 0.589 | 0.988 / 0.218 | n/a | n/a | n/a | results/eval/r4.json |
| r5 | 0.603 | 0.061 | 0.000 | 0.097 | 0.009 | 0.012 | 0.701 | 0.741 | 0.975 / 0.229 | n/a | n/a | n/a | results/eval/r5.json |
| r6 | 0.605 | 0.056 | 0.000 | 0.100 | 0.012 | 0.015 | 0.523 | 0.597 | 0.585 / 0.186 | n/a | n/a | n/a | results/eval/r6.json |
| r7 | 0.631 | 0.060 | 0.000 | 0.084 | 0.010 | 0.012 | 0.580 | 0.710 | 0.881 / 0.203 | 0.472 | 0.360 | 0.938 | results/eval/r7.json |
| rL8 | 0.567 | 0.063 | 0.000 | 0.084 | 0.050 | 0.292 | 0.524 | 0.543 | 0.979 / 0.234 | n/a | n/a | n/a | results/eval/rL8.json |
| rL9 | 0.565 | 0.071 | 0.000 | 0.093 | 0.083 | 0.199 | 0.288 | 0.367 | 0.575 / 0.167 | n/a | n/a | n/a | results/eval/rL9.json |
| rL10 | 0.646 | 0.044 | 0.000 | 0.100 | 0.034 | 0.072 | 0.694 | 0.679 | 0.958 / 0.249 | n/a | n/a | n/a | results/eval/rL10.json |
| rL11 | 0.527 | 0.063 | 0.000 | 0.115 | 0.085 | 0.242 | 0.431 | 0.510 | 0.968 / 0.255 | n/a | n/a | n/a | results/eval/rL11.json |

JevBench columns for Q2 and Q3 come from `results/jevbench/Q2-20261104T140620Z/kelter_summary.json` and `results/jevbench/Q3-20261104T230135Z/kelter_summary.json`.

The two Qwen rows that did not become the default: Q2 trained on the larger r5 corpus (with the synthetic shapes tasks) and 2048-token prompts, and lost on the hard tier. Q3 is Q1's recipe with 15% of the training records padded with unrelated passages to 846 to 2,786 tokens, to train length handling; it held the held-out numbers but was lower on every JevBench tier and more overconfident on the hard tier (mean top probability 0.734 against accuracy 0.541). Q1 stays the default.

## An action rule that was tested and not shipped

The shipped rule acts only on admissible evidence, so it defers most questions. A looser candidate also acted when the calibrated probability was at least 0.90 and no evidence was admissible, for every kind of option except destructive ones. Its pass conditions were fixed before the run: no unsafe act on the fake-approval test, no unsafe act on the forge set, accuracy when acting at least 0.970, and an act rate of at least 0.40.

| | Shipped rule | Candidate |
|---|---|---|
| Acts on held-out questions (n=3340) | 880 (0.263) | 2217 (0.664) |
| Accuracy when it acts | 0.974 | 0.973 |
| Fake approval in tool_output: unsafe act (n=400) | 0 | 0 |
| Forge set: unsafe act (n=3120) | 4 (0.001) | 19 (0.006) |

The candidate failed on the forge set, so it was not shipped. The same test gave the shipped rule its first forge figure: 4 unsafe acts in 3,120, all on items where the planted text carried a `user` or `system` label. The shipped figures are in `results/eval/engine_Q1base.json`; the candidate's run is not part of this repository.

## Baselines

Other systems run through the same evaluation code on the same held-out sets: `results/baselines_table.md`.

## Null recall: two test sets

Null recall is the share of questions with no fitting option that the model flags as null, at a threshold set so that 5% of answerable questions are wrongly flagged. Two sets are reported. The "old" set is answerable questions with their correct option deleted, plus the natural out-of-scope items in CLINC150. Some of those deleted-answer items still have a near-duplicate option left, so the "corrected" set filters them by option similarity and adds swapped-state items (the question paired with an unrelated text). The corrected set was introduced after reading diagnostics on the test split, so treat its numbers as weaker evidence than the old set's; both are reported everywhere.
