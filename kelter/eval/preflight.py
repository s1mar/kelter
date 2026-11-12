"""Two-way read check of the files a GPU job reads, run right before it starts.

Each file is hashed through the file cache and then straight from disk (FILE_FLAG_NO_BUFFERING, which also drops
its cached pages). A difference means stale or corrupted pages were in the cache; the disk read clears them. A
second pass must then be clean, or the job does not start.

  python -m kelter.eval.preflight --tag Q4 [--base Qwen/Qwen3-4B-Instruct-2507] [--paths ...]

Writes results/eval/preflight/<tag>.json; exit code 1 when pass 2 is not clean.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kelter.llm_train import _hash_cached, _hash_unbuffered

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = ["checkpoints/r2", "checkpoints/Q1", "data/built"]


def base_files(repo_id: str) -> list[Path]:
    from huggingface_hub import scan_cache_dir
    out = []
    for r in scan_cache_dir().repos:
        if r.repo_id == repo_id:
            for rev in r.revisions:
                out += [Path(f.file_path) for f in rev.files]
    return sorted(set(out))


def collect(paths: list[str], base: str | None) -> list[Path]:
    files = []
    for p in paths:
        q = ROOT / p
        files += [q] if q.is_file() else sorted(x for x in q.rglob("*") if x.is_file())
    if base:
        files += base_files(base)
    return files


def one_pass(files: list[Path]) -> list[dict]:
    bad = []
    for f in files:
        c, d = _hash_cached(f), _hash_unbuffered(f)
        if d is not None and c != d:
            bad.append({"file": str(f), "cached": c[:12], "disk": d[:12]})
    return bad


def _label(f: Path) -> str:
    try:
        return str(f.relative_to(ROOT)).replace(chr(92), "/")
    except ValueError:
        parts = f.parts
        return "/".join(parts[parts.index("hub"):]) if "hub" in parts else f.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507", help="Hugging Face repo whose cached files are checked; '' is off")
    ap.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    a = ap.parse_args(argv)
    files = collect(a.paths, a.base or None)
    t0 = time.time()
    p1 = one_pass(files)
    p2 = one_pass(files)
    for x in p1 + p2:
        x["file"] = _label(Path(x["file"]))
    res = {"tag": a.tag, "n_files": len(files), "bytes": sum(f.stat().st_size for f in files),
           "pass1_mismatches": p1, "pass2_mismatches": p2, "clean": not p2, "seconds": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "command": f"python -m kelter.eval.preflight --tag {a.tag}"}
    d = ROOT / "results" / "eval" / "preflight"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_files", "bytes", "clean", "seconds")} | {"pass1": len(p1), "pass2": len(p2)}))
    return 0 if res["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
