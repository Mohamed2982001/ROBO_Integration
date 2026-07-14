"""
Extract name/age from speech (English + Arabic script). Used without LLM tools.
"""

from __future__ import annotations

import re
from typing import Any, Dict

# Common Arabic given names → Latin for TTS / DB consistency
_AR_TO_EN_NAMES = {
    "محمود": "Mahmoud",
    "عبدالله": "Abdullah",
    "عبد الله": "Abdullah",
    "علا": "Ola",
    "عمر": "Omar",
    "احمد": "Ahmed",
    "أحمد": "Ahmed",
    "محمد": "Mohamed",
    "يوسف": "Youssef",
    "علي": "Ali",
    "مصطفى": "Mustafa",
    "ابراهيم": "Ibrahim",
    "إبراهيم": "Ibrahim",
    "سارة": "Sarah",
    "فاطمة": "Fatima",
    "مريم": "Maryam",
    "هدى": "Hoda",
    "منى": "Mona",
    "نور": "Nour",
    "خالد": "Khaled",
    "طارق": "Tarek",
}

_NAME_PATTERNS = [
    re.compile(
        r"(?:my name is|my name's|i am|i'm|im|call me|name is|this is)\s+"
        r"([A-Za-z][A-Za-z'\-]{1,31})",
        re.IGNORECASE,
    ),
    re.compile(r"^([A-Za-z][A-Za-z'\-]{1,31})$"),
    re.compile(r"(?:اسمي|أنا|انا|اسمي هو)\s+([\u0621-\u064A]+)"),
    re.compile(r"^([\u0621-\u064A]+)$"),
]

_AGE_PATTERNS = [
    re.compile(r"(?:i am|i'm|im|i have|i've got|im)\s+(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*,?\s*(?:years?\s+old)?\s*$", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s+years?\s+old", re.IGNORECASE),
    re.compile(r"age\s+(?:is\s+)?(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"(?:عندي|عمري|سن)\s+(\d{1,3})"),
    re.compile(r"(\d{1,3})\s+(?:سنة|سنوات|عام|أعوام)"),
]


def _normalize_name(raw: str) -> str:
    s = raw.strip()
    # Check if the name is in Arabic and can be translated
    if s in _AR_TO_EN_NAMES:
        return _AR_TO_EN_NAMES[s]
    return s.title()


def extract_profile_facts(text: str) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {}
    t = str(text).strip()
    t_clean = t.strip(".,!?\"'")
    out: Dict[str, Any] = {}

    for pat in _NAME_PATTERNS:
        matches = pat.finditer(t_clean)
        for m in matches:
            name = _normalize_name(m.group(1))
            if name.lower() not in (
                "a", "the", "and", "yes", "no", "hi", "hey", "hello",
                "from", "here", "there", "fine", "good", "not", "nothing",
                "ok", "okay", "am", "is", "are", "an", "welcome",
                "egypt", "syria", "yemen", "cairo", "dubai", "london",
                "libya", "sudan", "algeria", "tunisia", "morocco", "iraq",
                "jordan", "lebanon", "palestine", "saudi", "kuwait", "qatar",
                "dubai", "emirates", "oman", "bahrain"
            ):
                out["name"] = name
                break
        if "name" in out:
            break

    for pat in _AGE_PATTERNS:
        m = pat.search(t_clean)
        if m:
            age = int(m.group(1))
            if 5 <= age <= 120:
                out["age"] = age
                break

    return out
