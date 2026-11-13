# License for the model weights

The model weights published with this repository are licensed under the Creative Commons Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0).

- Full text: https://creativecommons.org/licenses/by-sa/4.0/legalcode
- Summary: https://creativecommons.org/licenses/by-sa/4.0/

The summary below is a plain-language guide. The full text is what applies.

## What it covers

Every checkpoint archive attached to this repository's releases and listed in `weights/MANIFEST.json`: the Q1 LoRA adapter, the encoder checkpoints r2, r3, r4, r5, r7 and rL10, and the ensemble files E1 and E3.

It does not cover:

- the code in this repository, which is Apache 2.0 (see `LICENSE`);
- the Qwen3-4B base model, which is downloaded from its own source under Apache 2.0;
- the training datasets, which keep their own licenses (see `THIRD-PARTY.md`).

## What it asks of you

- You may use, copy and change the weights, including for commercial work.
- If you share the weights, or weights made from them (for example fine-tuned, merged or quantized copies), give credit and release them under this same license.
- Credit line: "kelter weights, https://github.com/s1mar/kelter, CC BY-SA 4.0". Say so if you changed them.

## Why this license

Some of the training data is share-alike: BoolQ (CC BY-SA 3.0), SQuAD 2.0, ARC-Easy and the SNLI sentences in e-SNLI (CC BY-SA 4.0). Whether share-alike terms on data reach weights trained on it is not settled. Releasing the weights under CC BY-SA 4.0 complies either way.

Two datasets do not carry a single standard license: PAWS (its card says "other") and MultiNLI (several licenses by text source). `THIRD-PARTY.md` lists every dataset and model with its license as stated by its source. Check those two at their source if your use depends on them.
