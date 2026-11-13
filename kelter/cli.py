"""The `kelter` command.

  kelter                 what is set up and what to run next
  kelter serve [...]     the HTTP server (flags of python -m kelter.server)
  kelter download [...]  the release weights (flags of kelter.download)
  kelter examples [...]  the quick-start examples against a running server
  kelter --version
"""
from __future__ import annotations

import sys

from kelter import __version__, paths

CHECKOUT_ONLY = {
    "train": ("training", "python -m kelter.train or python -m kelter.llm_train"),
    "eval": ("evaluation", "python -m kelter.eval.run, kelter.eval.engine_eval or kelter.eval.jevbench_run"),
}


def _gpu() -> str:
    try:
        from kelter.server import _cuda_free_gb
        g = _cuda_free_gb()
    except Exception as e:  # torch missing or broken: say so instead of a traceback
        return f"not checked ({e.__class__.__name__}: is torch installed?)"
    if g is None:
        return "no CUDA GPU found"
    free, total = g
    import torch
    return f"{torch.cuda.get_device_name(0)}, {free:.1f} of {total:.1f} GB free"


def status() -> list:
    from kelter.download import load_manifest, present
    from kelter.server import LLM_GPU_GB
    root, home = paths.checkout_root(), paths.home()
    m = load_manifest()
    have = {v: present(m, v, home) for v in ("default", "lower-latency")}
    gpu = _gpu()
    has_gpu = "free" in gpu
    lines = [f"kelter {__version__} ({'clone at ' + str(root) if root else 'installed package'})",
             f"Folder: {home} (set KELTER_HOME to use another)",
             "Weights: " + "; ".join(f"{v} {'present' if ok else 'not downloaded'}" for v, ok in have.items()),
             f"GPU: {gpu}"]
    serve = paths.command("serve")
    if not have["default"] and not have["lower-latency"]:
        nxt = (f"{paths.command('download')}" if has_gpu
               else f"{paths.command('download')} --variant lower-latency (the default needs an NVIDIA GPU with "
                    f"about {LLM_GPU_GB:.0f} GB free)")
    elif have["default"] and has_gpu:
        nxt = f"{serve}, then in another shell {paths.command('examples')}"
    elif have["lower-latency"]:
        nxt = f"{serve} --variant lower-latency, then in another shell {paths.command('examples')}"
    else:
        nxt = (f"{paths.command('download')} --variant lower-latency (no usable GPU for the default; "
               f"KELTER_DEVICE=cpu runs it on the CPU, much slower)")
    lines += [f"Next: {nxt}", "Commands: serve, download, examples, --version; each takes --help."]
    return lines


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("\n".join(status()))
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in ("--version", "-V", "version"):
        print(f"kelter {__version__}")
        return 0
    if cmd in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    if cmd == "serve":
        from kelter.server import main as serve
        serve(rest)
        return 0
    if cmd == "download":
        from kelter.download import main as download
        return download(rest)
    if cmd == "examples":
        from kelter.examples import main as examples
        return examples(rest)
    if cmd in CHECKOUT_ONLY:
        what, how = CHECKOUT_ONLY[cmd]
        paths.require_checkout(what)
        print(f"kelter: {what} has no kelter subcommand; in this clone run {how} (see the README).")
        return 0
    print(f"kelter: unknown command {cmd!r}. Run `kelter` with no arguments to see what to run next, or "
          "`kelter --help`.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
