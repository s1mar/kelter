# Data card: shipped variant Q1 (LLM-backed, default)

Q1 is Qwen/Qwen3-4B-Instruct-2507 with a LoRA adapter. The adapter was trained on a seeded stratified sample of data/built's train split; r2 (data_card.md), trained on the whole of that corpus, supplies the witness. The base model's own pretraining data is its publisher's and is not described here.

Sample (kelter/llm_train.py, results/train/Q1.json): records whose state carries the encoder's [MASK] augmentation are left out, then at most 5000 synthetic_policy, 3000 synthetic_score and 1600 of each other source are drawn with seed 0.

| sample | 24000 |
|---|---|

| source | available after the [MASK] filter | taken | of which null |
|---|---|---|---|
| arc_easy | 2854 | 1600 | 208 |
| banking77 | 12431 | 1600 | 206 |
| boolq | 10669 | 1600 | 39 |
| civil_comments | 20515 | 1600 | 46 |
| esnli | 25492 | 1600 | 219 |
| massive | 15988 | 1600 | 224 |
| mmlu | 2582 | 1600 | 247 |
| mnli | 25417 | 1600 | 210 |
| paws | 22592 | 1600 | 33 |
| squad2 | 24062 | 1600 | 581 |
| synthetic_policy | 7187 | 5000 | 450 |
| synthetic_score | 3728 | 3000 | 94 |

Calibration (per-type temperature, split conformal) uses data/built/calib.jsonl only (results/calibration/Q1.json). No test or JevBench item is used to train, select or calibrate anything.

## Corpus the sample is drawn from (data/built/manifest.json)

| split | records |
|---|---|
| train | 199502 |
| dev | 16967 |
| calib | 3439 |
| test | 3340 |
| test_null | 2787 |
| test_nudge | 3120 |
| test_forge | 3120 |
| witness_test | 2999 |
| anli_test | 2000 |
