# Installing kelter

Two ways to install: from a clone (the README's quick start) or as a package. This page covers the package, the platforms, and the two model sizes in detail.

## Platforms

Python 3.12 and an NVIDIA GPU with 12 GB of memory (the default model needs about 9 GB of it free). Tested on Windows 11. On Linux, installation and the CPU test suite run on Ubuntu 24.04 in CI, but the server has not been run end to end there. A Mac with Apple Silicon should work, slower, and is untested: install torch from PyPI without the CUDA index. Without a GPU, start the smaller model with `--variant lower-latency`.

## Install as a package

From version 0.3.0 kelter also installs as a package, so you can serve it without keeping a clone around. It is not on PyPI yet. Build the wheel once from a clone:

```bash
python -m pip install build
python -m build
```

That writes `dist/kelter-0.3.0-py3-none-any.whl`. In a new environment, install a CUDA build of torch first (the torch that PyPI offers for Windows is CPU-only), then the wheel:

```bash
python -m venv kelter-env
kelter-env/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
kelter-env/Scripts/python -m pip install dist/kelter-0.3.0-py3-none-any.whl
```

Then use the `kelter` command, found in `kelter-env/Scripts/` (`kelter-env/bin/` on Linux and macOS):

| Command | What it does |
|---|---|
| `kelter` | says where its folder is, which weights are there, whether a GPU was found, and what to run next |
| `kelter download` | fetches the default variant's weights; add `--variant lower-latency` for the smaller model, which also runs without a GPU |
| `kelter serve` | starts the server, with the same flags as `python -m kelter.server` |
| `kelter examples` | sends the two quick-start examples to the running server and saves the responses |
| `kelter --version` | prints the version |

The installed package keeps weights and outputs in one folder: `%LOCALAPPDATA%\kelter` on Windows, `~/.local/share/kelter` on Linux and `~/Library/Application Support/kelter` on macOS. Set `KELTER_HOME` to use another. If a download fails, `kelter download` says where to put a weight file you fetched yourself; a file already there with the right checksum is used as it is.

In a clone nothing changes: the folder is the clone, and `python scripts/download_weights.py` does what `kelter download` does. Training and evaluation run from a clone only, and say so in one line if you try them from the installed package.

## Two variants

| | `default` | `lower-latency` |
|---|---|---|
| Model | Qwen3-4B with a LoRA adapter, plus the encoder for evidence | encoder ensemble (four checkpoints) |
| Held-out accuracy | 0.913 | 0.688 |
| Null recall at 5% false positives | 0.968 | 0.775 |
| Calibration error (15 bins) | 0.017 | 0.042 |
| Accuracy when it acts | 0.974 | 0.970 |
| Wrong option acted on, planted text under a trusted label (3,120 cases) | 4 (0.001) | 6 (0.002) |
| Decision latency, laptop GPU (p50), highest power mode | 542 ms | 149 ms |
| Decision latency, laptop GPU (p50), balanced power mode | 688 ms | not measured |
| Weights | 126 MB adapter + 607 MB encoder, plus the 8 GB base download | 4.0 GB |

In the default variant the Qwen model reads the text and the options, labelled with letters plus one more letter for "none of the options", and the probabilities come from its next-token logits for those letters. The encoder checkpoint r2 supplies the evidence spans. The lower-latency variant is encoder-only: r2 chooses the option and four checkpoints' null scores are averaged. Start it with `--variant lower-latency` and fetch its weights with `scripts/download_weights.py --variant lower-latency`.

Full metric tables, every measured run and the baselines are in [docs/results.md](docs/results.md).
