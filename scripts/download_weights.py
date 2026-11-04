"""Download the released model weights into checkpoints/.

Usage:
    python scripts/download_weights.py                  # the default variant
    python scripts/download_weights.py --variant lower-latency
    python scripts/download_weights.py --variant all

The weights are attached to a GitHub release of this repository as one tar
file per checkpoint directory. The release tag, the asset names and their
SHA-256 checksums are in weights/MANIFEST.json. Each file is checked against
its checksum before it is unpacked; a mismatch aborts.

The repository is read from the git remote of this checkout. Pass --repo
owner/name to override it, for example when working from a copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights" / "MANIFEST.json"
CHECKPOINTS = ROOT / "checkpoints"


def repo_from_git() -> str | None:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    url = url.removesuffix(".git")
    for prefix in ("https://github.com/", "git@github.com:", "ssh://git@github.com/"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {dest.name}: {got / 1e6:8.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        print()
    tmp.replace(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", default="default", help="default, lower-latency or all")
    ap.add_argument("--repo", default=None, help="GitHub owner/name; defaults to this checkout's origin")
    ap.add_argument("--keep-archives", action="store_true", help="keep the downloaded tar files")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    repo = args.repo or manifest.get("repo") or repo_from_git()
    if not repo:
        print("cannot determine the GitHub repository; pass --repo owner/name", file=sys.stderr)
        return 2
    variants = manifest["variants"]
    if args.variant == "all":
        wanted = sorted({c for v in variants.values() for c in v})
    elif args.variant in variants:
        wanted = variants[args.variant]
    else:
        print(f"unknown variant {args.variant!r}; choose one of {sorted(variants)} or all", file=sys.stderr)
        return 2

    base = f"https://github.com/{repo}/releases/download/{manifest['tag']}"
    archives = ROOT / "weights" / "archives"
    for name in wanted:
        if (CHECKPOINTS / name / "config.json").exists():
            print(f"  {name}: already present, skipped")
            continue
        asset = manifest["assets"][name]
        path = archives / asset["file"]
        if not path.exists() or sha256(path) != asset["sha256"]:
            print(f"downloading {asset['file']} ({asset['size_mb']} MB)")
            download(f"{base}/{asset['file']}", path)
        digest = sha256(path)
        if digest != asset["sha256"]:
            print(f"checksum mismatch for {asset['file']}: {digest}", file=sys.stderr)
            return 1
        with tarfile.open(path) as tar:
            tar.extractall(CHECKPOINTS, filter="data")
        print(f"  {name}: unpacked into {CHECKPOINTS / name}")
        if not args.keep_archives:
            path.unlink()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
