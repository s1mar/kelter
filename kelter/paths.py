"""Where kelter's files live, in a checkout and in an installed wheel.

Checkout mode: the package's parent directory holds a pyproject.toml naming kelter (a clone, or an editable install).
Everything stays where it was: read-only files and the writable folder are the checkout root.

Installed mode (a wheel in site-packages): read-only files (policy table, weights manifest, ensemble specs) come from
kelter/_assets inside the package, copied there at build time (setup.py), and the writable folder (checkpoints,
downloaded archives, example outputs) is a per-user folder.

KELTER_HOME overrides the writable folder in both modes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
ASSETS = PKG / "_assets"

# read-only files: path in a checkout -> name inside kelter/_assets (setup.py copies them)
ASSET_FILES = {
    "policy_table.json": "policy_table.json",
    "policy_table.sha256": "policy_table.sha256",
    "weights/MANIFEST.json": "weights/MANIFEST.json",
    "results/ensemble/E1.json": "ensemble/E1.json",
    "results/ensemble/E3.json": "ensemble/E3.json",
}


def checkout_root() -> Path | None:
    """The checkout this package runs from, or None when it is installed."""
    root = PKG.parent
    p = root / "pyproject.toml"
    try:
        return root if p.is_file() and 'name = "kelter"' in p.read_text(encoding="utf-8") else None
    except OSError:
        return None


def is_checkout() -> bool:
    return checkout_root() is not None


def user_home(env=None, platform=None) -> Path:
    """The per-user folder an installed kelter writes to."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "kelter"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "kelter"
    base = env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "kelter"


def home() -> Path:
    """The writable folder: KELTER_HOME, else the checkout root, else the per-user folder."""
    h = os.environ.get("KELTER_HOME")
    if h:
        return Path(h).expanduser()
    return checkout_root() or user_home()


def asset(rel: str) -> Path:
    """A read-only file by its checkout path (a key of ASSET_FILES)."""
    root = checkout_root()
    if root is not None:
        return root / rel
    return ASSETS / ASSET_FILES[rel] if rel in ASSET_FILES else home() / rel


def under_home(p) -> Path:
    """A path relative to the writable folder (absolute paths are kept)."""
    p = Path(p)
    return p if p.is_absolute() else home() / p


def checkpoint_missing(ck, members_root=None) -> list:
    """What a checkpoint folder lacks, as names relative to it; an empty list means it is complete. config.json marks
    a checkpoint, but an interrupted unpack can leave it alone, so also: an LLM checkpoint ("llm" in its config) needs
    adapter/ with adapter_model.*, an encoder checkpoint ("backbone") needs model.pt, an ensemble ("ensemble") needs each of its members complete. Used by server.check_weights and
    by the download command, so "present" means the same in both."""
    ck = Path(ck)
    cfg = ck / "config.json"
    if not cfg.exists():
        return ["config.json"]
    try:
        c = json.loads(cfg.read_text(encoding="utf-8"))
    except ValueError:
        return ["config.json (unreadable)"]
    missing = []
    if "llm" in c and not any((ck / "adapter").glob("adapter_model.*")):
        missing.append("adapter")
    if "backbone" in c and not (ck / "model.pt").exists():
        missing.append("model.pt")
    if "ensemble" in c:  # an ensemble is its members: follow its config to their checkpoint folders (siblings of this one)
        root = Path(members_root) if members_root else ck.parent  # kelter.ensemble loads them from home()/checkpoints
        for m in c.get("members", []):
            try:
                shown = os.path.relpath(root / m, ck).replace(chr(92), "/")
            except ValueError:  # another drive on Windows: show the full path
                shown = (root / m).as_posix()
            missing += [f"{shown}/{x}" for x in checkpoint_missing(root / m)]
    return missing


def command(sub: str) -> str:
    """How to run a kelter command in this mode, for messages: `kelter serve` installed,
    `python -m kelter.server` in a checkout."""
    if is_checkout():
        return {"serve": "python -m kelter.server", "download": "python scripts/download_weights.py",
                "examples": "python -m kelter.examples"}[sub]
    return f"kelter {sub}"


def no_arguments(module: str, doc: Optional[str]) -> None:
    """For a module run with -m that takes no options: `--help` prints the module's docstring as usage and exits 0, any other
    argument is an argparse error, and nothing else happens. Call it first in the __main__ block, before the work starts."""
    import argparse
    argparse.ArgumentParser(prog=f"python -m {module}", description=doc,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()


def require_checkout(what: str) -> None:
    """Exit with one plain line when a checkout-only command runs from an installed wheel."""
    if not is_checkout():
        print(f"kelter: {what} runs from a clone of the repository, not from the installed package. "
              "Clone the repository and run it there (see the README); the installed package serves and "
              "downloads weights only.", file=sys.stderr)
        raise SystemExit(2)
