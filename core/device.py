"""
Pick compute device for local models (Whisper, FaceNet, SpeechBrain, Kokoro).

ROBOT_DEVICE in .env:
  auto — CUDA if available, else CPU
  cuda — force GPU (falls back to CPU with a warning)
  cpu  — force CPU
"""

from __future__ import annotations

import os
from typing import Literal

import config.settings as cfg

DevicePref = Literal["auto", "cuda", "cpu"]


def get_device_preference() -> DevicePref:
    raw = (getattr(cfg, "ROBOT_DEVICE", "auto") or "auto").strip().lower()
    if raw in ("cuda", "gpu"):
        return "cuda"
    if raw == "cpu":
        return "cpu"
    return "auto"


def pick_torch_device() -> str:
    """Return 'cuda' or 'cpu' for PyTorch-backed models."""
    import torch

    pref = get_device_preference()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print(
            "[Device] ROBOT_DEVICE=cuda but CUDA unavailable — falling back to CPU."
        )
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def whisper_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"
