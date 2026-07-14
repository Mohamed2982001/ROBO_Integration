"""
core/events.py
Typed events for the robot event bus + PriorityQueue wrapper.

Priority levels:
  0 (SPEECH)  — user speech, highest priority
  1 (VISION)  — scene changes, lower priority
"""
import queue
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


# ── Priority levels ───────────────────────────────────────────────────────────
class Priority(IntEnum):
    SPEECH = 0
    VISION = 1


# ── Event data classes ────────────────────────────────────────────────────────
@dataclass
class SpeechEvent:
    """Emitted by HearingEngine when the user finishes a spoken utterance."""
    text: str
    voice_embedding: Optional[Any] = None
    audio_bytes: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class VisionChangeEvent:
    """
    Emitted by VisionPipeline when something meaningful happens.
    change_type: one of
        "new_person"    — a new face appeared
        "person_left"   — a tracked face disappeared
        "emotion_shift" — emotion changed on a known face
        "gesture"       — gesture detected (future)
        "unknown_idle"  — unknown face idle > UNKNOWN_IDLE_SECONDS
        "unknown_appeared" — unknown face seen for the first time (immediate greet)
        "multi_unknown_appeared" — 2+ unknown faces first seen together
    """
    change_type: str
    details: Dict[str, Any]
    scene_snapshot: Any           # VisionState object
    timestamp: float = field(default_factory=time.time)


@dataclass
class RegistrationEvent:
    """Emitted when an unknown speaker says their name."""
    name: str
    embedding: Any                # 512-dim torch tensor
    timestamp: float = field(default_factory=time.time)


# ── Internal priority wrapper ──────────────────────────────────────────────────
@dataclass(order=True)
class _PrioritizedItem:
    priority: int
    timestamp: float              # tiebreaker: earlier = higher priority
    event: Any = field(compare=False)


# ── Public EventQueue ─────────────────────────────────────────────────────────
class EventQueue:
    """
    Thread-safe priority queue.
    Speech events (priority=0) are always processed before vision events (priority=1).
    Within the same priority, earlier events are processed first.
    """

    def __init__(self):
        self._q: queue.PriorityQueue = queue.PriorityQueue()

    # ── push helpers ──────────────────────────────────────────────────────────
    def push_speech(self, text: str, voice_embedding: Optional[Any] = None, audio_bytes: Optional[bytes] = None) -> None:
        """Enqueue user utterance — SPEECH beats VISION whenever both are queued."""
        evt = SpeechEvent(text=text, voice_embedding=voice_embedding, audio_bytes=audio_bytes)
        self._q.put(_PrioritizedItem(Priority.SPEECH, evt.timestamp, evt))

    def push_vision_change(
        self,
        change_type: str,
        details: Dict[str, Any],
        scene_snapshot: Any,
    ) -> None:
        evt = VisionChangeEvent(change_type, details, scene_snapshot)
        self._q.put(_PrioritizedItem(Priority.VISION, evt.timestamp, evt))

    def push_registration(self, name: str, embedding: Any) -> None:
        evt = RegistrationEvent(name=name, embedding=embedding)
        self._q.put(_PrioritizedItem(Priority.SPEECH, evt.timestamp, evt))

    def push_raw(self, priority: int, event: Any) -> None:
        self._q.put(_PrioritizedItem(priority, time.time(), event))

    # ── consume ───────────────────────────────────────────────────────────────
    def get(self, timeout: Optional[float] = None) -> Any:
        """
        Block until an event is available.
        Raises queue.Empty if timeout elapses.
        Returns the unwrapped event object.
        """
        item = self._q.get(timeout=timeout)
        return item.event

    def task_done(self) -> None:
        self._q.task_done()

    @property
    def empty(self) -> bool:
        return self._q.empty()
