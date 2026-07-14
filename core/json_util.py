"""JSON-safe serialization for API responses (strip tensors, numpy, etc.)."""

from __future__ import annotations

from typing import Any, Optional


_SKIP_KEYS = frozenset(
    {
        "face_embedding",
        "voice_embedding",
        "embedding",
        "track_id",
    }
)


def sanitize_for_api(obj: Any) -> Any:
    """Return a structure FastAPI can json-encode."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _SKIP_KEYS:
                continue
            out[str(k)] = sanitize_for_api(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_api(x) for x in obj]
    # torch.Tensor, numpy, ObjectId, etc.
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    return str(obj)


def has_embedding(value: Optional[Any]) -> bool:
    """True if a face/voice embedding is present (safe for torch tensors)."""
    if value is None:
        return False
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.numel() > 0
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.size > 0
    except ImportError:
        pass
    return True
