# Baselines (measured by kelter/eval/run.py on the same held-out sets)

| System | Held-out acc | ECE | Brier | Flip (canon) | Flip (no canon) | Nudge-ASR | Forge-ASR | Null recall @5% FPR (old set) | Null recall @5% FPR (corrected set) | Witness F1 e-SNLI | Necessity e-SNLI | Random-mask flip | ANLI acc | Source file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kelter r2 (ModernBERT-base, this repo) | 0.671 | 0.040 | 0.423 | 0.000 | 0.090 | 0.013 (n=2867) | 0.016 (n=2829) | 0.582 | 0.603 | 0.611 | 0.971 | 0.227 | 0.301 | results/eval/r2.json |
| TF-IDF pairwise logistic regression (trained on our train split) | 0.257 | 0.114 | 0.830 | 0.009 | 0.009 | 0.034 (n=2525) | 0.032 (n=2509) | 0.043 | 0.052 | n/a | n/a | n/a | 0.317 | results/eval/tfidf.json |
| Laya 0.4.15 (convaiinnovations/laya, zero-shot on our sets) | 0.720 | 0.076 | 0.339 | 0.112 | 0.112 | 0.689 (n=2696) | 0.675 (n=2711) | 0.165 | 0.310 | n/a | n/a | n/a | 0.514 | results/eval/laya.json |
| Kev 0.8B (jaredpalmer/kev-0.8b @9a45d25, zero-shot on our sets) | 0.771 | 0.050 | 0.319 | 0.082 | 0.082 | 0.151 (n=2857) | 0.168 (n=2835) | 0.237 | 0.286 | n/a | n/a | n/a | 0.360 | results/eval/kev.json |
| Qwen2.5-1.5B-Instruct letter logprobs, 4-rotation averaging | 0.813 | 0.090 | 0.310 | 0.070 | 0.181 | 0.278 (n=2846) | 0.284 (n=2842) | 0.351 | 0.438 | n/a | n/a | n/a | 0.420 | results/eval/qwen.json |

Systems without a null output use 1 - max probability as the null score.

## Quoted, not measured here

| System | Numbers | Source and date |
|---|---|---|
| Jev 1.13 (hosted, not run here) | ECE 0.045, paired flip rate 0.021, injection ASR 0.063 | Respan's published run (read 2026-10-25); different items from ours |
| Jev 1.13 (hosted, not run here) | JevBench v1.4 composite 63.3 | JevBench v1.4 published results (read 2026-10-25) |
