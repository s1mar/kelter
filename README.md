# openverdict

A small decision model you run on your own machine. Give it a piece of text and a multiple-choice question, and it returns an answer, a calibrated probability for each option, the passages that support the answer, how likely it is that no option fits, and whether you should act on the answer or hand the question to something bigger.

It is a 149M-parameter encoder (ModernBERT), not a language model. It answers in about a quarter of a second on a laptop GPU and needs no API key.

The server speaks the same `/v1/systemone` API as TypeSafe's Jev, so clients written for that API only need a new base URL.

## What it does

Agents and apps make small decisions all the time: is this ticket about billing, is this command safe to run, how urgent is this message. A large model can answer, but slowly, at a cost per call, and with confidence numbers that are often wrong. openverdict is built for exactly these decisions:

- **Calibrated.** When it says 90%, it is right about 90% of the time (expected calibration error 0.04 on held-out data).
- **Shows its evidence.** Every answer comes with the spans of the input it relied on, and a check that masking those spans changes the answer.
- **Knows when nothing fits.** A null probability says how likely it is that none of the options is right.
- **Resists injected text.** You can tag which parts of the input came from the user, a tool, a document or the system. Text from a tool's output cannot approve a destructive action on its own. In 400 test cases with a planted fake approval, it never acted unsafely.
- **Tells you what to do next.** Each answer carries an action: `act`, `escalate` or `defer`. On held-out data it acts on about a fifth of questions and is right 97% of the time when it does.

It is not a knowledge model. On science exam questions it scores near chance, and the action field says so by escalating.

## Quick start

Windows or Linux with an NVIDIA GPU, or a Mac with Apple Silicon (slower, untested). Python 3.12.

```bash
git clone https://github.com/s1mar/openverdict.git
cd openverdict
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/download_weights.py
.venv/Scripts/python -m openverdict.server --port 8765
```

On Linux and macOS use `.venv/bin/python`. On a Mac, install torch from PyPI without the CUDA index. The weights download is about 5 GB for the default model.

Then ask it something:

```bash
curl -s http://127.0.0.1:8765/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "I was charged twice for my subscription this month. Please refund the extra charge.",
  "questions": {
    "tone":    {"type": "choice", "instructions": "What is the tone?",
                "criteria": {"angry": "upset or hostile", "calm": "neutral", "excited": "eager"}},
    "urgency": {"type": "score",  "instructions": "How urgent is this?",
                "criteria": ["Can wait", "This week", "Today"]},
    "billing": {"type": "noul",   "instructions": "Is this about billing?"}
  }
}'
```

Each answer has the standard TypeSafe fields (the chosen option, a probability per option, a confidence) plus these:

| Field | Meaning |
|---|---|
| `action` | `act`, `escalate` or `defer` |
| `null_probability` | chance that no option fits |
| `witness` | the input spans the answer rests on, with their provenance |
| `verifier` | whether that evidence is allowed to count, given where it came from |
| `set`, `coverage_level` | the options that cannot be ruled out at 95% coverage |
| `receipt`, `variant` | a record of the decision and which model answered |

Clients that do not know these fields ignore them. To tag provenance, add a `provenance` list of `{"start", "end", "type"}` character ranges to the request; types are `user`, `tool_output`, `retrieved_doc`, `system` and `unknown`. The full field-by-field contract is in `docs/typesafe-contract.md`.

## Two variants

| | `default` | `lower-latency` |
|---|---|---|
| Held-out accuracy | 0.688 | 0.688 |
| Null recall at 5% false positives | 0.818 | 0.775 |
| Calibration error (15 bins) | 0.040 | 0.042 |
| Accuracy when it acts | 0.967 | 0.970 |
| Decision latency, laptop GPU (p50) | 251 ms | 149 ms |
| Weights | 5.0 GB | 4.0 GB |

Both use the same checkpoint to choose the option and its evidence. They differ in how they estimate the null probability: `default` combines five checkpoints' null scores with a small gradient-boosted model, `lower-latency` averages four. Start the second one with `--variant lower-latency` and fetch its weights with `scripts/download_weights.py --variant lower-latency`.

Full metric tables, every measured run and the baselines are in [docs/results.md](docs/results.md).

## How it works

One encoder pass scores the options and points at the evidence spans. The engine then runs three more passes: with the evidence masked, with a random control masked, and with untrusted provenance masked. From those it decides whether the evidence was necessary, whether it is admissible under a small policy table (`policy_table.json`), and how much of the answer depended on text that should not be trusted. A split-conformal step turns the probabilities into the option set and the action. Option order cannot change the answer because options are sorted by a hash before scoring.

The main pieces:

- `openverdict/model.py`: the encoder with option, witness and null heads.
- `openverdict/engine.py`: the four-pass verifier and the action rule.
- `openverdict/server.py`: the FastAPI server.
- `openverdict/data/`: dataset converters, the synthetic policy tasks and the corpus builder.
- `openverdict/train.py`, `calibrate.py`, `ensemble.py`: training, calibration and the two ensembles.
- `openverdict/eval/`: the evaluation harness, baselines, latency and the JevBench runner.

## Training it yourself

Training needs an NVIDIA GPU with 12 GB or more.

```bash
python -m openverdict.data.build
python -m openverdict.train --run r1 --epochs 1 --max-tokens 8192
python -m openverdict.calibrate --ckpt checkpoints/r1
python -m openverdict.eval.run --system ours --ckpt checkpoints/r1 --tag r1
```

The first command downloads the public datasets listed in `THIRD-PARTY.md` and writes `data/built/` and a data card under `docs/data_cards/`. The shipped ensembles are built from several such runs; `docs/results.md` lists them and `docs/data_cards/` records what each trained on.

## Limitations

- Knowledge questions are near chance: ARC-Challenge 0.30, MMLU and ANLI similar. Larger open models beat it on raw accuracy (see the baselines in `docs/results.md`); it leads on calibration, order robustness, injection resistance and null detection.
- Null recall passes 0.80 only on the corrected test set, which was defined after looking at test diagnostics; on the original set it is 0.76. Read `docs/results.md` for how the two sets differ.
- The action rule is conservative and its thresholds were not tuned, so it defers most held-out questions.
- The default variant runs five encoders per request, two of them ModernBERT-large, which is why it is slower than a single checkpoint.
- Calibration on JevBench's public tiers is worse than on our held-out data; the temperature was fitted on different kinds of text.
- Latency was measured on one laptop GPU shared with the desktop, in its highest power mode.
- The injection guarantee was tested on synthetic shell-gating and account-deletion items, not on real agent traces.

## License

Code: Apache 2.0 (see `LICENSE`). The training data carries its own licenses, listed in `THIRD-PARTY.md`; some of it is share-alike, which may extend to the weights. Check that file before redistributing the weights.
