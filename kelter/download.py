"""Download the released model weights.

Usage:
    kelter download                          # the default variant
    kelter download --variant lower-latency
    kelter download --variant all
    python scripts/download_weights.py ...   # the same, from a clone

The weights are attached to a GitHub release of the repository as one tar file per checkpoint directory. The release
tags, the asset names and their SHA-256 checksums are in weights/MANIFEST.json (inside the installed package, or in a
clone). Each file is checked against its checksum before it is unpacked; a mismatch aborts.

Files go under the writable folder (kelter.paths.home): the clone itself, or a per-user folder for an installed
package, or KELTER_HOME. Archives are kept in <folder>/weights/archives while downloading and checkpoints are unpacked
into <folder>/checkpoints. An archive already in weights/archives with the right checksum (for example one fetched in a
browser) is used without downloading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from kelter import paths


def repo_from_git(root: Path) -> str | None:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.strip()
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


def fetch(url: str, dest: Path) -> None:
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


def wanted(manifest: dict, variant: str) -> list | None:
    variants = manifest["variants"]
    if variant == "all":
        return sorted({c for v in variants.values() for c in v})
    return variants.get(variant)


def present(manifest: dict, variant: str, home: Path | None = None) -> bool:
    """Whether every checkpoint of a variant is unpacked under the writable folder."""
    home = home or paths.home()
    names = wanted(manifest, variant) or []
    return bool(names) and all((home / "checkpoints" / n / "config.json").exists() for n in names)


def load_manifest() -> dict:
    return json.loads(paths.asset("weights/MANIFEST.json").read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kelter download", description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", default="default", help="default, lower-latency or all")
    ap.add_argument("--repo", default=None, help="GitHub owner/name; defaults to the manifest's repository")
    ap.add_argument("--keep-archives", action="store_true", help="keep the downloaded tar files")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    root = paths.checkout_root()
    repo = args.repo or manifest.get("repo") or (repo_from_git(root) if root else None)
    if not repo:
        print("kelter: cannot determine the GitHub repository; pass --repo owner/name", file=sys.stderr)
        return 2
    names = wanted(manifest, args.variant)
    if names is None:
        print(f"kelter: unknown variant {args.variant!r}; choose one of {sorted(manifest['variants'])} or all",
              file=sys.stderr)
        return 2

    home = paths.home()
    checkpoints, archives = home / "checkpoints", home / "weights" / "archives"
    print(f"kelter: weights folder {checkpoints}")
    for name in names:
        if (checkpoints / name / "config.json").exists():
            print(f"  {name}: already present, skipped")
            continue
        asset = manifest["assets"][name]
        path = archives / asset["file"]
        if not path.exists() or sha256(path) != asset["sha256"]:
            print(f"downloading {asset['file']} ({asset['size_mb']} MB)")
            tag = asset.get("tag", manifest["tag"])
            url = f"https://github.com/{repo}/releases/download/{tag}/{asset['file']}"
            try:
                fetch(url, path)
            except urllib.error.HTTPError as e:
                print(f"kelter: could not download {asset['file']}: the server answered {e.code}. Check that release "
                      f"{tag} of {repo} exists and is public. You can also fetch {url} in a browser, put the file in "
                      f"{archives} and run this command again.", file=sys.stderr)
                return 1
            except urllib.error.URLError as e:
                print(f"kelter: could not reach github.com to download {asset['file']} ({e.reason}). Check your "
                      f"connection and run this command again; files already downloaded are kept.", file=sys.stderr)
                return 1
        digest = sha256(path)
        if digest != asset["sha256"]:
            print(f"kelter: checksum mismatch for {path}: {digest}", file=sys.stderr)
            return 1
        checkpoints.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as tar:
            tar.extractall(checkpoints, filter="data")
        print(f"  {name}: unpacked into {checkpoints / name}")
        if not args.keep_archives:
            path.unlink()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
