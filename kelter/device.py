"""Device choice for inference: KELTER_DEVICE if set, else CUDA (NVIDIA), else CPU.
The Apple GPU (MPS) is used only when KELTER_DEVICE=mps asks for it: in an end-to-end test on an
Apple Silicon machine it failed with an index error inside torch, while the CPU gave the expected
answers. Training stays CUDA only."""
from __future__ import annotations

import os
import platform

import torch


def default_device() -> str:
    env = os.environ.get("KELTER_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize(device: str) -> None:
    """Wait for queued GPU work, so timings cover it."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    if device.startswith("mps"):
        return f"Apple GPU via MPS ({platform.machine()}, macOS {platform.mac_ver()[0]})"
    return f"CPU ({platform.processor() or platform.machine()})"
