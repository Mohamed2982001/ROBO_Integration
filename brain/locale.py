"""
Bilingual input helpers — robot SPEAKS English only; understands Arabic confirmations.
"""

from __future__ import annotations

import re
from collections import Counter

_CONFIRM_EXACT = re.compile(
    r"^("
    r"yes|yeah|yep|yup|correct|right|sure|ok|okay|confirm|confirmed|"
    r"that's right|that is right|that's correct|that is correct|"
    r"sounds good|sounds right|absolutely|exactly|perfect"
    r")[\s.!?,]*$",
    re.IGNORECASE,
)

_CONFIRM_WORDS = (
    "yes",
    "yeah",
    "yep",
    "yup",
    "correct",
    "right",
    "sure",
    "okay",
    "ok",
    "confirm",
    "confirmed",
    "absolutely",
    "exactly",
    "perfect",
    "that's right",
    "that is right",
    "that's correct",
    "that is correct",
    "sounds good",
)

_DENY_RE = re.compile(
    r"^(no|nope|wrong|incorrect|not right|cancel)[\s.!?,]*$",
    re.IGNORECASE,
)


def is_repetitive_garbage(text: str) -> bool:
    """Whisper loop hallucinations (e.g. ايوه x50)."""
    words = (text or "").split()
    if len(words) < 6:
        return False
    counts = Counter(words)
    top_word, top_n = counts.most_common(1)[0]
    if top_n / len(words) >= 0.45:
        return True
    if len(counts) <= 2 and len(words) >= 8:
        return True
    return False


def is_confirmation(text: str) -> bool:
    t = (text or "").strip()
    if not t or is_repetitive_garbage(t):
        return False
    lower = t.lower().strip(" .,!?")
    if _CONFIRM_EXACT.match(lower):
        return True
    # Short replies: "yes that's right", "yeah correct", etc.
    if len(lower.split()) <= 10:
        for w in _CONFIRM_WORDS:
            if w in lower:
                return True
    return False


def is_denial(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and bool(_DENY_RE.match(t.lower()))


def overlap_interrupt_message() -> str:
    return "Hey — one at a time please, I can only focus on one person at a time."
