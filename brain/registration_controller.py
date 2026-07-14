"""
Deterministic onboarding — no LLM tools required for register flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from brain.locale import is_confirmation, is_denial
from brain.profile_extract import extract_profile_facts
from brain.smart_speech import (
    after_confirm,
    ask_for_missing,
    confirm_prompt,
    confirm_still_missing,
)
from core.json_util import has_embedding


@dataclass
class RegistrationTurnResult:
    facts_extracted: Dict[str, Any] = field(default_factory=dict)
    stage_message: Optional[str] = None
    confirmed: bool = False
    confirm_message: Optional[str] = None
    missing: List[str] = field(default_factory=list)
    pending_profile: Optional[Dict] = None
    display_name: Optional[str] = None
    speak_text: Optional[str] = None
    skip_llm: bool = False


def _effective_bio(session, pr: Optional[dict]) -> dict:
    bio = dict(session.get_biometric_capture() if session else {})
    if pr:
        if has_embedding(pr.get("face_embedding")):
            bio["face_embedding"] = pr["face_embedding"]
        if has_embedding(pr.get("voice_embedding")):
            bio["voice_embedding"] = pr["voice_embedding"]
    return bio


def _missing_requirements(
    profile: dict, bio: dict, has_voice_turn: bool
) -> List[str]:
    missing = []
    if not profile.get("name") or str(profile.get("name")).lower() == "unknown":
        missing.append("name")
    if profile.get("age") is None:
        missing.append("age")
    if not has_embedding(bio.get("face_embedding")):
        missing.append("face_on_camera")
    has_voice = has_voice_turn or has_embedding(bio.get("voice_embedding"))
    if not has_voice:
        missing.append("voice")
    return missing


def process_registration_turn(
    text: str,
    *,
    crud,
    session,
    has_voice_embedding: bool,
) -> RegistrationTurnResult:
    facts = extract_profile_facts(text)
    # Check if a name is provided and matches an existing registered user
    name_fact = facts.get("name")
    if name_fact:
        try:
            existing_user = crud.memory.mongo.get_user_by_name(name_fact)
            if existing_user:
                user_id = str(existing_user["_id"])
                session.current_user_id = user_id
                session.cached_user_info = existing_user
                session.session_owner_id = user_id
                session.clear_pending_registration()
                session.confirm_prompt_sent = False
                print(f"[RegistrationController] Logged in existing user by name: {name_fact} ({user_id})")
                return RegistrationTurnResult(
                    facts_extracted=facts,
                    display_name=name_fact,
                    skip_llm=False,
                )
        except Exception as e:
            print(f"[RegistrationController] Error checking existing user by name: {e}")

    stage_msg = None
    if facts and not is_confirmation(text):
        stage_msg = crud.stage_user_profile(json.dumps(facts))

    pr = session.get_pending_registration()
    profile = dict((pr or {}).get("profile_data") or {})
    bio = _effective_bio(session, pr)
    display = profile.get("name") or facts.get("name")
    name = profile.get("name")

    if is_denial(text) and pr:
        session.clear_pending_registration()
        session.confirm_prompt_sent = False
        return RegistrationTurnResult(
            facts_extracted=facts,
            speak_text="No problem — tell me your correct name and age.",
            skip_llm=True,
            pending_profile=None,
            display_name=None,
        )

    if pr:
        missing = _missing_requirements(profile, bio, has_voice_embedding)
        if not missing:
            # AUTO-CONFIRM IMMEDIATELY in one shot!
            confirm_msg = crud.confirm_registration(confirm=True)
            session.confirm_prompt_sent = False
            nm = profile.get("name") or "friend"
            return RegistrationTurnResult(
                facts_extracted=facts,
                confirmed=True,
                confirm_message=confirm_msg,
                pending_profile=None,
                display_name=nm,
                speak_text=after_confirm(nm),
                skip_llm=True,
            )

        # Still missing some biometrics or details; ask for them directly
        session.confirm_prompt_sent = False
        return RegistrationTurnResult(
            facts_extracted=facts,
            stage_message=stage_msg,
            missing=missing,
            pending_profile=profile,
            display_name=display,
            speak_text=ask_for_missing(missing, name=name),
            skip_llm=True,
        )

    return RegistrationTurnResult(
        facts_extracted=facts,
        stage_message=stage_msg,
        pending_profile=None,
        display_name=display or facts.get("name"),
    )


def registration_hint_for_llm(result: RegistrationTurnResult) -> str:
    if result.confirmed:
        return (
            "[Registration: user was just saved. Continue chatting in ENGLISH ONLY. "
            "Do NOT ask to confirm again.]"
        )
    return (
        "[Registration is handled in code. Reply in ENGLISH ONLY. "
        "Do NOT call stage_user_profile or confirm_registration.]"
    )
