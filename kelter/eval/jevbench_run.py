"""Run the JevBench public tiers through JevBench's own harness against the
local /v1/systemone server.

python -m kelter.eval.jevbench_run --ckpt checkpoints/r1 [--port 8765]

Starts `python -m kelter.server` on 127.0.0.1, waits for GET /v1/models,
then for each tier (original, hard, easy) runs
  python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:<port> --key-env ''
      --cost-basis local_self_hosted_gpu --reserve-usd 0 ...
and `jevbench.cli summarize`, with the working directory at
third_party/jevbench. Every run writes a new directory
results/jevbench/<run>-<timestamp>/ (the harness refuses to overwrite).
JevBench items are test only: nothing here trains or calibrates on them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from kelter.llm import prefill_chunk_setting

ROOT = Path(__file__).resolve().parents[2]
JB = ROOT / "third_party" / "jevbench"
TIERS = ("original", "hard", "easy")


def wait_ready(url: str, timeout_s: float = 300.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"server not ready at {url}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--limit", type=int, default=None, help="debug only: items per tier")
    ap.add_argument("--tag", default=None, help="results directory prefix (default: the run name)")
    args = ap.parse_args(argv)
    run = args.tag or Path(args.ckpt).name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / "jevbench" / f"{run}-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1")
    log = (out / "server.log").open("w", encoding="utf-8")
    srv = subprocess.Popen([sys.executable, "-m", "kelter.server", "--ckpt", args.ckpt, "--port", str(args.port)],
                           cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    summary = {"run": run, "ckpt": args.ckpt, "endpoint": f"http://127.0.0.1:{args.port}", "tiers": {},
               "prefill_chunk": prefill_chunk_setting()}
    try:
        wait_ready(f"http://127.0.0.1:{args.port}/v1/models")
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(JB), capture_output=True, text=True).stdout.strip()
        summary["jevbench_commit"] = rev
        jb_env = dict(os.environ)
        if os.name == "nt":  # JevBench's ledger imports POSIX fcntl; see kelter/compat/fcntl_shim
            shim = str(ROOT / "kelter" / "compat" / "fcntl_shim")
            jb_env["PYTHONPATH"] = shim + os.pathsep + jb_env.get("PYTHONPATH", "")
            summary["fcntl_shim"] = True
        for tier in TIERS:
            tdir = out / tier
            tasks = JB / "datasets" / "public" / f"{tier}.jsonl"
            cmd = [sys.executable, "-m", "jevbench.cli", "run", "--tasks", str(tasks), "--adapter", "typesafe",
                   "--endpoint", f"http://127.0.0.1:{args.port}", "--key-env", "", "--model", f"kelter-{run}",
                   "--cost-basis", "local_self_hosted_gpu", "--reserve-usd", "0",
                   "--results", str(tdir / "results.jsonl"), "--raw-dir", str(tdir / "raw"),
                   "--ledger", str(out / "ledger.jsonl"), "--manifest", str(tdir / "manifest.json"),
                   "--run-label", f"kelter-{run}-{tier}"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            tdir.mkdir()
            r = subprocess.run(cmd, cwd=str(JB), capture_output=True, text=True, env=jb_env)
            (tdir / "run.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
            s = subprocess.run([sys.executable, "-m", "jevbench.cli", "summarize", "--tasks", str(tasks),
                                "--results", str(tdir / "results.jsonl"), "--public-export", str(tdir / "summary.json")],
                               cwd=str(JB), capture_output=True, text=True, env=jb_env)
            (tdir / "summarize.log").write_text(s.stdout + "\n" + s.stderr, encoding="utf-8")
            summary["tiers"][tier] = {"run_exit": r.returncode, "summarize_exit": s.returncode,
                                      "command": " ".join(cmd)}
            sp = tdir / "summary.json"
            if sp.exists():
                summary["tiers"][tier]["summary"] = json.loads(sp.read_text(encoding="utf-8"))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()
    (out / "kelter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({t: {k: v for k, v in d.items() if k != "summary"} for t, d in summary["tiers"].items()}, indent=2))
    print(f"results in {out}")
    return 0 if all(d["run_exit"] == 0 and d["summarize_exit"] == 0 for d in summary["tiers"].values()) else 1


if __name__ == "__main__":
    from kelter import paths
    paths.require_checkout("evaluation (python -m kelter.eval.jevbench_run)")
    raise SystemExit(main())
