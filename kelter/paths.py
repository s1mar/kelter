"""Where kelter's files live, in a checkout and in an installed wheel.

Checkout mode: the package's parent directory holds a pyproject.toml naming kelter (a clone, or an editable install).
Everything stays where it was: read-only files and the writable folder are the checkout root.

Installed mode (a wheel in site-packages): read-only files (policy table, weights manifest, ensemble specs) come from
kelter/_assets inside the package, copied there at build time (setup.py), and the writable folder (checkpoints,
downloaded archives, example outputs) is a per-user folder.

KELTER_HOME overrides the writable folder in both modes.
"""
from __future__ import annotations

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


def command(sub: str) -> str:
    """How to run a kelter command in this mode, for messages: `kelter serve` installed,
    `python -m kelter.server` in a checkout."""
    if is_checkout():
        return {"serve": "python -m kelter.server", "download": "python scripts/download_weights.py",
                "examples": "python -m kelter.examples"}[sub]
    return f"kelter {sub}"


def require_checkout(what: str) -> None:
    """Exit with one plain line when a checkout-only command runs from an installed wheel."""
    if not is_checkout():
        print(f"kelter: {what} runs from a clone of the repository, not from the installed package. "
              "Clone the repository and run it there (see the README); the installed package serves and "
              "downloads weights only.", file=sys.stderr)
        raise SystemExit(2)
