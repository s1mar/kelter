# What the metrics measure

All of these are computed by `kelter/eval/run.py`, `engine_eval.py` and `latency.py` on the held-out sets, and written to `results/eval/`. The held-out sources (CLINC150, ARC-Challenge and the synthetic "sharing" policy family) were never used for training; a calibration split of the same sources is used to fit temperatures and conformal thresholds, and the test split is evaluated once per run.

## Answers

- **Held-out accuracy.** Share of answerable test questions where the verdict is the gold option. The verdict is "none of the options" when the null wins, and that counts as wrong here. The share where the most likely option is the gold option, whatever the null says, is stored as `heldout_accuracy_forced` (0.917 for Q1 and 0.710 for E1, against 0.913 and 0.688). Also reported per source.
- **ECE (expected calibration error, 15 bins).** Predictions are bucketed by confidence; ECE is the average gap between each bucket's confidence and its accuracy, weighted by bucket size. Reported after the per-type temperature fitted on the calibration split, and before it.
- **Brier.** Mean squared error between the probability vector and the one-hot gold answer.

## Robustness

- **Paired flip rate.** Each answerable `choice` question is scored with its options in two orders; a flip is a different top option. Yes or no questions and score questions are not permuted. With canonical ordering (options sorted by hash before scoring) the rate is exactly zero by construction. The "no canonical" column shows what the model alone does.
- **Nudge-ASR.** Attack success rate when a sentence is appended to the input suggesting a wrong option: the share of answers that move to the suggested option, among all answerable questions whose answer before the attack was not already that option (right or wrong before the attack).
- **Forge-ASR.** The same sentences, inserted under a trusted label (`user` or `system`) instead of an untrusted one (`tool_output` or `retrieved_doc`). Both tests draw from four fixed attack sentences, and kelter's training data plants the same four under untrusted labels, so these rates are for wording kelter has seen (see Limitations).

## Null detection

Some questions have no fitting option. The model gives a null probability; at a threshold chosen so that 5% of answerable questions are wrongly flagged, **null recall** is the share of unanswerable questions caught.

- **Old set.** Answerable questions with their gold option deleted, plus CLINC150's natural out-of-scope items.
- **Corrected set.** The deleted-answer items filtered to those where no near-duplicate option remains (cosine similarity under 0.80 with a sentence encoder), plus swapped-state items (a question paired with an unrelated text), plus the natural out-of-scope items. It is reported by group as well.

## Evidence

The encoder's witness head points at spans of the input; in the default variant those spans are checked against the Qwen model's answer. On e-SNLI and SQuAD 2.0 test items, which have human rationales:

- **Witness F1.** Overlap between the predicted spans and the human rationale, counted in characters (so long words weigh more than short ones).
- **Necessity.** Among the items the model answered correctly, the share of answers that change when the predicted spans are masked. Reported next to a **random-mask control** because any masking changes some answers; the gap between the two is the evidence that the spans matter. The control masks random words up to the same number of characters as the predicted spans. It is not an exact match: a predicted phrase becomes one mask and the control's separate words become several, so the control tends to hide a little more.
- **Sufficiency.** Among the items the model answered correctly, the share of answers that stay the same when everything except the predicted spans is masked.
- **Human-rationale mask flip.** Among the items the model answered correctly, the share of answers that change when the human rationale is masked.

## The engine

`engine_eval.py` runs the full four-pass decision and reports:

- **Conformal coverage** at alpha 0.05: how often the gold option is inside the returned set; nominal 0.95. Also the mean set size and the share of singleton sets.
- **Action rates.** How often the engine says `act`, `escalate` or `defer`, and the accuracy of the top option on the questions where it acts.
- **Fake approval.** On synthetic shell-gating and account-deletion items, a sentence approving the destructive option is planted inside text tagged as tool output. Reported: how often the verdict after the planted sentence is the destructive option, and how often the engine still acts on it (an unsafe act). Unlike the nudge and forged-instruction rates, the first figure does not compare with the answer before the sentence was planted, so it includes items the model would have got wrong anyway.

## Latency

`latency.py` times one forward pass on an input of about 1,100 tokens and a full engine decision (four passes), 50 runs each on the GPU and 5 runs each on CPU with int8 dynamic quantization. Medians and 95th percentiles are reported. Numbers depend heavily on the machine and its power mode; the published ones are from one laptop GPU, in its highest power mode except the one set marked balance mode (`results/eval/latency_Q1_balance.json`).

## JevBench

`jevbench_run.py` serves the model and runs the public tiers of JevBench through JevBench's own harness and TypeSafe adapter. It expects a JevBench checkout at `third_party/jevbench`, which this repository does not include: `git clone https://github.com/fstandhartinger/jevbench third_party/jevbench`, then check out commit `609a664e81f46e03731600bd7bd9f1b7a61257c5`, the one the tests use. The harness scores the option probabilities only, so the encoder ensembles score the same as the single checkpoint that chooses their options. `jevbench_cost.py` reads a run's per-item results back for reported input tokens, cost per 1,000 decisions at the board's reference price, per-request latency and mean top probability. The v1.4 composite needs the sealed tier and a hosted cost basis and is not computable locally.
