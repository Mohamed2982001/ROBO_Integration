"""JSON-encode MongoDB user docs and other objects for LLM prompts."""

from __future__ import annotations

import json
from typing import Any


def json_safe(obj: Any, **kwargs) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False, **kwargs)
