"""
Shared speech-activity state between STT and vision.

While the user is speaking, vision accumulates per-track lip-motion scores.
At end of utterance, the track with the highest average score is treated as
the speaker for that turn (more reliable than a single frame at reply time).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional


class SpeechActivityTracker:
    def __init__(
        self,
        active_window_seconds: float = 1.2,
        overlap_frame_threshold: int = 2,
    ):
        self._lock = threading.Lock()
        self._active_window = active_window_seconds
        self._overlap_frame_threshold = overlap_frame_threshold
        self._last_activity: float = 0.0
        self._lip_scores: Dict[int, List[float]] = defaultdict(list)
        self._utterance_speaker_track_id: Optional[int] = None
        self._overlap_frames: int = 0

    def mark_speech_chunk(self) -> None:
        """Called by STT when mic energy indicates the user is talking."""
        with self._lock:
            self._last_activity = time.time()

    @property
    def is_user_speaking(self) -> bool:
        with self._lock:
            return (time.time() - self._last_activity) < self._active_window

    def record_lip_motion(self, track_id: int, motion_score: float) -> None:
        """Called by vision each frame while user speech is active."""
        if motion_score <= 0:
            return
        with self._lock:
            if (time.time() - self._last_activity) >= self._active_window:
                return
            self._lip_scores[int(track_id)].append(float(motion_score))

    def record_multi_speaker_frame(self, active_track_count: int) -> None:
        """Increment when 2+ faces show strong lip motion during user speech."""
        if active_track_count < 2:
            return
        with self._lock:
            if (time.time() - self._last_activity) < self._active_window:
                self._overlap_frames += 1

    def consume_overlap_interrupt(self) -> bool:
        """True once if multiple people were talking over the last utterance."""
        with self._lock:
            if self._overlap_frames >= self._overlap_frame_threshold:
                self._overlap_frames = 0
                return True
            self._overlap_frames = 0
            return False

    def finalize_utterance_speaker(self) -> Optional[int]:
        """
        Pick the track with highest mean lip motion for the utterance just ended.
        Clears accumulated scores for the next turn.
        """
        with self._lock:
            best_tid: Optional[int] = None
            best_avg = -1.0
            for tid, scores in self._lip_scores.items():
                if not scores:
                    continue
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_tid = int(tid)
            self._lip_scores.clear()
            self._utterance_speaker_track_id = best_tid
            return best_tid

    def get_utterance_speaker_track_id(self) -> Optional[int]:
        with self._lock:
            return self._utterance_speaker_track_id

    def clear_utterance_speaker(self) -> None:
        with self._lock:
            self._utterance_speaker_track_id = None
