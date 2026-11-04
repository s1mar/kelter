# What the metrics measure

All of these are computed by `openverdict/eval/run.py`, `engine_eval.py` and `latency.py` on the held-out sets, and written to `results/eval/`. The held-out sources (CLINC150, ARC-Challenge and the synthetic "sharing" policy family) were never used for training; a calibration split of the same sources is used to fit temperatures and conformal thresholds, and the test split is evaluated once per run.

## Answers

- **Held-out accuracy.** Share of answerable test questions where the top option is the gold option. Also reported per source.
- **ECE (expected calibration error, 15 bins).** Predictions are bucketed by confidence; ECE is the average gap between each bucket's confidence and its accuracy, weighted by bucket size. Reported after the per-type temperature fitted on the calibration split, and before it.
- **Brier.** Mean squared error between the probability vector and the one-hot gold answer.

## Robustness

- **Paired flip rate.** Each question is scored with its options in two orders; a flip is a different top option. With canonical ordering (options sorted by hash before scoring) the rate is exactly zero by construction. The "no canonical" column shows what the model alone does.
- **Nudge-ASR.** Attack success rate when a sentence is appended to the input suggesting a wrong option: the share of previously correct answers that move to the suggested option.
- **Forge-ASR.** The same, when the appended sentence imitates an authoritative instruction ("the correct answer is...").

## Null detection

Some questions have no fitting option. The model gives a null probability; at a threshold chosen so that 5% of answerable questions are wrongly flagged, **null recall** is the share of unanswerable questions caught.

- **Old set.** Answerable questions with their gold option deleted, plus CLINC150's natural out-of-scope items.
- **Corrected set.** The deleted-answer items filtered to those where no near-duplicate option remains (cosine similarity under 0.80 with a sentence encoder), plus swapped-state items (a question paired with an unrelated text), plus the natural out-of-scope items. It is reported by group as well.

## Evidence

The encoder's witness head points at spans of the input; in the default variant those spans are checked against the Qwen model's answer. On e-SNLI and SQuAD 2.0 test items, which have human rationales:

- **Witness F1.** Token overlap between the predicted spans and the human rationale.
- **Necessity.** Share of answers that change when the predicted spans are masked. Reported next to a **random-mask control** (masking the same number of random tokens) because any masking changes some answers; the gap between the two is the evidence that the spans matter.
- **Sufficiency.** Share of answers that stay the same when everything except the predicted spans is masked.
- **Human-rationale mask flip.** Share of answers that change when the human rationale is masked.

## The engine

`engine_eval.py` runs the full four-pass decision and reports:

- **Conformal coverage** at alpha 0.05: how often the gold option is inside the returned set; nominal 0.95. Also the mean set size and the share of singleton sets.
- **Action rates.** How often the engine says `act`, `escalate` or `defer`, and the accuracy of the top option on the questions where it acts.
- **Fake approval.** On synthetic shell-gating and account-deletion items, a sentence approving the destructive option is planted inside text tagged as tool output. Reported: how often the verdict moves to the destructive option, and how often the engine still acts on it (an unsafe act).

## Latency

`latency.py` times one forward pass on an input of about 1,100 tokens and a full engine decision (four passes), 50 runs each, on the GPU and on CPU with int8 dynamic quantization. Medians and 95th percentiles are reported. Numbers depend heavily on the machine and its power mode; the published ones are from one laptop GPU in its highest power mode, measured back to back for the variants being compared.

## JevBench

`jevbench_run.py` serves the model and runs the public tiers of JevBench through JevBench's own harness and TypeSafe adapter. The harness scores the option probabilities only, so the encoder ensembles score the same as the single checkpoint that chooses their options. `jevbench_cost.py` reads a run's per-item results back for reported input tokens, cost per 1,000 decisions at the board's reference price, per-request latency and mean top probability. The v1.4 composite needs the sealed tier and a hosted cost basis and is not computable locally.
