# How kelter works

What happens inside one decision, where the code lives, and how to train the model yourself.

## How it works

One model pass scores the options, and the encoder points at the evidence spans. The engine then runs three more passes: with the evidence masked, with a random control masked, and with untrusted provenance masked. From those it decides whether the evidence was necessary, whether it is admissible under a small policy table (`policy_table.json`), and how much of the answer depended on text that should not be trusted. A split-conformal step turns the probabilities into the option set and the action. Option order cannot change the answer because options are sorted by a hash before scoring. Prompts longer than 1,024 tokens are prefilled in chunks through the KV cache, which keeps long inputs within the GPU's memory.

The main pieces:

- `kelter/llm.py`: the Qwen predictor, its prompt and the letter readout.
- `kelter/model.py`: the encoder with option, witness and null heads.
- `kelter/engine.py`: the four-pass verifier and the action rule.
- `kelter/server.py`: the FastAPI server.
- `kelter/data/`: dataset converters, the synthetic policy tasks and the corpus builder.
- `kelter/llm_train.py`, `train.py`, `calibrate.py`, `ensemble.py`: adapter training, encoder training, calibration and the ensembles.
- `kelter/eval/`: the evaluation harness, baselines, latency and the JevBench runner.

## Training it yourself

Training needs an NVIDIA GPU with 12 GB or more. The adapter trains in 4-bit (QLoRA) on a 24,000-record sample of the corpus in a little over an hour on a laptop GPU.

```bash
python -m kelter.data.build
python -m kelter.llm_train --name Q1 --qlora
python -m kelter.calibrate --ckpt checkpoints/Q1
python -m kelter.eval.run --system ours --ckpt checkpoints/Q1 --tag Q1
```

The first command downloads the public datasets listed in `THIRD-PARTY.md` and writes `data/built/` and a data card under `docs/data_cards/`. The encoder checkpoints train with `python -m kelter.train --run r1 --epochs 1 --max-tokens 8192`; `docs/results.md` lists every run and `docs/data_cards/` records what each trained on.
