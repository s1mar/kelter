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
8. **The shortlist is built.** `conformal_set` keeps every option that cannot be ruled out at 95% coverage. A clear answer leaves exactly one.
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
| canonical order | options are sorted by a hash before scoring, so their order in the request cannot change the answer |

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

The first command downloads the public datasets listed in `THIRD-PARTY.md` and writes `data/built/` and a data card under `docs/data_cards/`. The encoder checkpoints train with `python -m kelter.train --run r1 --epochs 1 --max-tokens 8192`; `docs/results.md` lists every run and `docs/data_cards/` records what each trained on.
