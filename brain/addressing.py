"""
brain/addressing.py
Cheap heuristics for whether an utterance is likely meant for Musa vs background chat.
(No extra LLM latency — complements the personality rules in prompts.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Set


_STOPWORDS_COMMON = frozenset(
    "yeah yep uh-huh hmm mhm ok okay sure right exactly totally uh um".split()
)


def _norm_tokens(text: str) -> list[str]:
    return [
        w
        for w in re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        if len(w) > 1 or w.isdigit()
    ]


def _keyword_set(csv: str) -> Set[str]:
    raw = (csv or "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


@dataclass
class DirectednessAssessment:
    likely_for_robot: bool
    score: float
    rationale: str


def directedness_notes_for_prompt(assessment: DirectednessAssessment) -> str:
    return (
        f"Estimated directedness score={assessment.score:.2f}; "
        f"{assessment.rationale}"
    )


def assess_directedness(
    user_text: str,
    *,
    known_participant_names: Sequence[str],
    participant_count_estimate: int,
    wake_patterns: Iterable[re.Pattern],
) -> DirectednessAssessment:
    """
    Returns a coarse prior the LLM uses to soften or withhold long replies when people
    are probably talking beside the robot rather than invoking it.

    Higher score ⇒ more confident the utterance involves the companion.
    """
    text = (user_text or "").strip()
    tl = text.lower()
    tokens = _norm_tokens(text)

    robot_hit = any(pat.search(tl) for pat in wake_patterns)

    explicit_directives = [
        "tell me",
        "let me know",
        "do you",
        "can you",
        "could you",
        "would you",
        "show me",
        "answer",
        "find out",
        "give me",
        "what is",
        "what are",
        "who is",
        "who are",
        "where is",
        "where are",
        "when is",
        "when are",
        "why is",
        "why are",
    ]
    direct_phrase = any(phrase in tl for phrase in explicit_directives)
    interrogative_or_imperative = bool(
        tl.endswith("?")
        or tl.startswith(("what ", "why ", "how ", "when ", "who ", "where ", "tell ", "can you", "could you", "please ", "show ", "answer ", "let me"))
        or direct_phrase
    )

    # Very short fillers or repeated stutters while several people are visible — often ambient.
    is_backchannel_like = False
    if participant_count_estimate >= 2:
        junk = bool(tokens) and len(tokens) <= 3 and all(
            t in _STOPWORDS_COMMON for t in tokens
        )
        repeats = (
            bool(tokens)
            and len(tokens) >= 6
            and tokens.count(tokens[0]) == len(tokens)
        )
        is_backchannel_like = junk or repeats

    lc_names = {
        _n.strip().lower()
        for _n in known_participant_names
        if _n and _n != "Unknown"
    }
    names_only = bool(lc_names and any(n in tl for n in lc_names))

    score = 0.45
    if robot_hit:
        score += 0.42
        reason = "addressing pattern / wake marker present"
        return DirectednessAssessment(True, min(1.0, score), reason)

    if interrogative_or_imperative and not names_only:
        score += 0.28
        reason = "question or direct request without other speaker names"
        return DirectednessAssessment(True, min(1.0, score), reason)

    if participant_count_estimate >= 2:
        if names_only:
            score -= 0.24
            reason = "mentions another visible person in a multi-party scene"
            return DirectednessAssessment(score >= 0.48, max(0.0, score), reason)
        if is_backchannel_like:
            score -= 0.22
            reason = "short backchannel phrase with multiple people visible"
            return DirectednessAssessment(score >= 0.45, max(0.0, score), reason)
        score -= 0.08
        reason = "multiple people visible; no explicit robot cue"
        return DirectednessAssessment(score >= 0.50, score, reason)

    if names_only:
        score -= 0.10
        reason = "mentions a nearby person by name rather than addressing robot"
        return DirectednessAssessment(score >= 0.45, min(1.0, max(0.0, score)), reason)

    reason = "single-person scene or neutral phrasing — assume normal turn-taking"
    return DirectednessAssessment(True, min(1.0, score + 0.12), reason)


def build_wake_patterns(csv: str) -> list[re.Pattern]:
    phrases: list[str] = []
    for p in _keyword_set(csv):
        if len(p) > 1:
            phrases.append(re.escape(p))
    if phrases:
        return [re.compile(rf"\b(?:{'|'.join(phrases)})\b", re.I)]

    return [
        re.compile(
            r"\b(musa|robot|buddy|assistant|listen|tell me|ask you|hey you)\b",
            re.I,
        )
    ]
