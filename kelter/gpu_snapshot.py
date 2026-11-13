"""GPU state for a before-note, with machine paths replaced (no user profile paths in tracked files).

  python -m kelter.gpu_snapshot >> notes/<job>_before.txt

Prints utilisation and memory, then the nvidia-smi compute-apps list, with the LOCALAPPDATA, APPDATA and
USERPROFILE prefixes (and HOME elsewhere) written as %LOCALAPPDATA%, %APPDATA%, %USERPROFILE% and ~.
"""
from __future__ import annotations

import os
import subprocess


def scrub(text: str, env=None) -> str:
    env = os.environ if env is None else env
    for var, label in (("LOCALAPPDATA", "%LOCALAPPDATA%"), ("APPDATA", "%APPDATA%"),
                       ("USERPROFILE", "%USERPROFILE%"), ("HOME", "~")):
        v = env.get(var)
        if v and len(v) > 3:
            for form in {v, v.replace(chr(92), "/")}:
                text = text.replace(form, label)
                text = text.replace(form.lower(), label)
    return text


def snapshot() -> str:
    def run(*args):
        try:
            return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            return f"nvidia-smi unavailable ({e.__class__.__name__})"
    gpu = run("--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader")
    apps = run("--query-compute-apps=pid,process_name,used_memory", "--format=csv")
    return scrub(f"GPU: {gpu}\n{apps}")


if __name__ == "__main__":
    print(snapshot())
