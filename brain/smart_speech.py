"""
Short, natural English lines for enrollment — no LLM needed (fast, consistent).
"""

from __future__ import annotations

from typing import List, Optional


def ask_for_missing(missing: List[str], name: Optional[str] = None) -> str:
    if "name" in missing:
        return "Hey — I didn't catch your name. What should I call you?"
    if "age" in missing:
        nm = name or "there"
        return f"Thanks {nm}. How old are you?"
    if "face_on_camera" in missing:
        return "Can you look straight at the camera for a second? I want to remember your face."
    if "voice" in missing:
        return "Please say a longer, clear sentence in a quiet voice so I can register your voice print."
    return "Tell me a bit about yourself — name and age?"
 
 
def confirm_prompt(name: str, age: int) -> str:
    return f"So you're {name}, {age} years old — is that right?"
 
 
def after_confirm(name: str) -> str:
    return (
        f"Perfect — nice to meet you, {name}. I've got your face and voice saved. "
        f"What are you up to today?"
    )
 
 
def confirm_still_missing(missing: List[str]) -> str:
    if "face_on_camera" in missing:
        return "Almost there — I still need a clear look at your face on camera."
    if "voice" in missing:
        return "Almost — please say one more long, clear sentence so I can fully lock in your voice print."
    if "age" in missing:
        return "I still need your age — how old are you?"
    return ask_for_missing(missing)
