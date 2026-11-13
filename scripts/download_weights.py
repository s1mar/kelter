"""Download the released model weights into checkpoints/ (a thin wrapper around kelter.download).

Usage:
    python scripts/download_weights.py                  # the default variant
    python scripts/download_weights.py --variant lower-latency
    python scripts/download_weights.py --variant all

The release tags, the asset names and their SHA-256 checksums are in weights/MANIFEST.json; each file is checked
against its checksum before it is unpacked. Installed from a wheel, `kelter download` does the same.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kelter.download import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
