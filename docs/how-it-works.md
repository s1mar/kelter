# How kelter works

What happens inside one decision, where the code lives, and how to train the model yourself.

## One request, step by step

A request is some text (the *state*) plus one or more questions. Here is what happens to it, with the function that does each step.

1. **The server receives it.** `systemone` in `kelter/server.py` turns the state into one string and checks the request.
2. **The text is labelled by source.** `build_provenance` cuts the state into segments, each marked `user`, `system`, `tool_output`, `retrieved_doc` or `unknown`. Anything you did not label becomes `unknown`.
3. **Each question becomes a record.** `build_record` puts the state, the labels, the question and its options into one dictionary. Everything after this works on records.
4. **The model answers four times.** `Engine.decide` in `kelter/engine.py` asks the model about:
   - the full text;
   - the evidence only (everything else hidden);
   - the text with the evidence hidden;
   - the text with the untrusted segments hidden.
5. **The verdict is picked.** The verdict is the most likely option, unless "none of the options" (the *null*) wins.
6. **The evidence is checked.** The evidence is *admissible* when all of these hold:
   - there is some, and it lies inside the text;
   - its source is allowed for this kind of option (the *verdict class*);
   - it is *sufficient*: the evidence alone gives the same verdict;
   - it is *necessary*: hiding it changes the verdict (required for most classes).
7. **Untrusted influence is measured.** This is how much the verdict's probability drops when the untrusted segments are hidden.
8. **The shortlist is built.** `conformal_set` keeps every option that cannot be ruled out at 95% coverage. A clear answer leaves exactly one. The server gives every request the `policy` type, so it uses the threshold fitted for that type (see Limitations): 0.991 for the default model (`results/calibration/Q1.json`) and 0.947 for the lower-latency variant (`results/calibration/E1.json`). The top option needs a probability of about that much to stand alone.
9. **The action is chosen**, in this order:
   - the null won, or the evidence is not admissible: `defer`;
   - untrusted text moved the verdict more than the class allows, or more than one option is left: `escalate`;
   - otherwise: `act`.
10. **The answer is sent back.** `build_answer` adds the choice, the probabilities and a confidence to the engine's fields.

## Glossary

| Term | Meaning |
|---|---|
| state | the text the question is about |
| record | one question with its state, source labels and options; the unit the engine works on |
| option | one of the answers the question allows |
| verdict | the option the model picks, or the null |
| null | "none of the options fits"; `null_probability` is how likely that is |
| provenance | where a piece of the state came from: `user`, `system`, `tool_output`, `retrieved_doc`, `unknown` |
| untrusted | provenance the policy does not trust: `tool_output`, `retrieved_doc`, `unknown` |
| witness | the spans of the state the answer rests on; "evidence" in this page |
| sufficiency | the witness alone gives the same verdict |
| necessity | hiding the witness changes the verdict |
| admissible | the witness passed every check in step 6 |
| verdict class | how risky an option is: `default`, `benign`, `low_stakes` or `destructive`. You set it per option with `verdict_classes`; unset options are `default` |
| policy table | `policy_table.json`: for each class, which sources may serve as evidence and how much untrusted influence is allowed |
| untrusted influence | the drop in the verdict's probability when untrusted text is hidden |
| conformal set | the options that cannot be ruled out at the chosen coverage (95%); `set` in the answer |
| calibration | fitting the model's probabilities on held-out data so that "90% sure" is right about 90% of the time |
| temperature | the single number per question type that calibration fits |
| act / escalate / defer | go ahead / probably right, get a second opinion / do not rely on this answer |
| choice, score, noul | the three question types: pick one option, rate on a scale, yes or no |
| canonical order | options are sorted by a hash before scoring, so their order in the request cannot change the answer. The default model keeps the levels of a `score` question in the order given, because the order is the scale |

## Model names

The names in the code and the results are the released checkpoints.

| Name | What it is |
|---|---|
| Q1 | the default model: Qwen3-4B with a LoRA adapter |
| r2 | the encoder (ModernBERT) that points at the evidence for Q1, and picks the option in E1 |
| E1 | the `lower-latency` variant: r2 picks the option, and the null scores of r3, r5, r7 and rL10 are averaged |
| E3 | an earlier ensemble with a small stacking model over its members' null scores |
| r1 to r7, rL10 | encoder training runs; `docs/results.md` lists each one |

## Where the code lives

- `kelter/server.py`: the web server, the request checks and the answer format.
- `kelter/engine.py`: the four passes, the evidence checks and the action rule.
- `kelter/llm.py`: the Qwen model, its prompt and how its letter scores become probabilities.
- `kelter/model.py`: the encoder with its option, witness and null outputs.
- `kelter/ensemble.py`: the encoder ensembles.
- `kelter/paths.py`, `cli.py`, `download.py`: where files live, the `kelter` command and the weights download.
- `kelter/data/`: dataset converters, the synthetic tasks and the corpus builder.
- `kelter/llm_train.py`, `train.py`, `calibrate.py`: adapter training, encoder training and calibration.
- `kelter/eval/`: the evaluation harness, the baselines, latency and the JevBench runner.

Long prompts (over 1,024 tokens) are fed to the Qwen model in chunks, which keeps long inputs within the GPU's memory.

## Training it yourself

Training needs an NVIDIA GPU with 12 GB or more. The adapter trains in 4-bit (QLoRA) on a 24,000-record sample of the corpus in a little over an hour on a laptop GPU.

```bash
python -m kelter.data.build
python -m kelter.llm_train --name Q1 --qlora
python -m kelter.calibrate --ckpt checkpoints/Q1
python -m kelter.eval.run --system ours --ckpt checkpoints/Q1 --tag Q1
```

The first command downloads the public datasets listed in `THIRD-PARTY.md` and writes `data/built/` and a data card under `docs/data_cards/`. Q1 needs the r2 encoder, because the default model's config points at `checkpoints/r2` for its evidence spans. Without r2, calibrating or evaluating a freshly trained Q1 fails. Get r2 in one of two ways. Run `python scripts/download_weights.py`, which fetches r2 and Q1. Or train it:

```bash
python -m kelter.train --run r2 --epochs 3 --max-tokens 8192 --lr 8e-05 --head-lr 0.0003 --w-witness 0.5 --seed 0 --infer-max-len 2048
```

This command was reconstructed from the settings recorded in `results/train/r2.json`, because that file predates recording the command line. Other encoder runs are trained the same way with their own flags. `docs/results.md` lists every run and `docs/data_cards/` records what most of them trained on: `data_card.md` covers r2, rL8, rL10 and rL11, `data_card_r1.md` covers r1, `data_card_r3.md` covers r3, `data_card_r5.md` covers r4 and r5, `data_card_r6.md` covers r6, `data_card_r7.md` covers r7 and rL9, `data_card_Q1.md` covers Q1, and `data_card_E1.md`, `data_card_E2.md` and `data_card_E3.md` list the members of the ensembles E1, E2 and E3. Q2 and Q3 have no card yet.
