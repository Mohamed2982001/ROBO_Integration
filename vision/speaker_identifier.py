"""
vision/speaker_identifier.py
Speaker selection when multiple faces are visible: motion (lower-face change vs last frame),
frame centrality, and face area, with temporal stickiness to reduce jitter.
"""

from __future__ import annotations

from typing import List, Optional, Any

import cv2
import numpy as np


class SpeakerIdentifier:
    """
    Pluggable interface. Default single-face behavior is unused when SmartSpeakerIdentifier exists.
    """

    def identify_speaker(
        self,
        frame: np.ndarray,
        faces: List[Any],
        audio_chunk: Optional[np.ndarray] = None,
        *,
        only_when_speaking: bool = False,
        speech_active: bool = False,
    ) -> Optional[int]:
        if len(faces) == 1:
            return int(faces[0].track_id)
        return None

    def score_faces(self, frame: np.ndarray, faces: List[Any]) -> dict[int, float]:
        if len(faces) == 1:
            return {int(faces[0].track_id): 1.0}
        return {}


class LipMovementSpeakerIdentifier(SpeakerIdentifier):
    """
    Identifies the active speaker by measuring motion explicitly in the lower 35%
    of the face bounding box (the mouth region).

    When only_when_speaking=True, the pick is frozen until speech_active is True
    (avoids head-motion false positives while everyone is silent).
    """

    def __init__(self, sticky_ratio: float = 0.88):
        self._sticky_ratio = sticky_ratio
        self._prev_crops: dict[int, np.ndarray] = {}
        self._last_pick: Optional[int] = None

    def score_faces(self, frame: np.ndarray, faces: List[Any]) -> dict[int, float]:
        if not faces:
            return {}
        if len(faces) == 1:
            tid = int(faces[0].track_id)
            self._cache_crop(frame, faces[0])
            return {tid: 1.0}

        h, w = frame.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores: dict[int, float] = {}

        for f in faces:
            tid = int(f.track_id)
            box = getattr(f, "box", None)
            if box is None or len(box) < 4:
                scores[tid] = 0.0
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                scores[tid] = 0.0
                continue

            mouth_y1 = y1 + int((y2 - y1) * 0.65)
            if mouth_y1 >= y2:
                scores[tid] = 0.0
                continue

            roi = gray[mouth_y1:y2, x1:x2]
            if roi.size == 0:
                scores[tid] = 0.0
                continue

            small = cv2.resize(roi, (48, 24), interpolation=cv2.INTER_AREA)
            prev = self._prev_crops.get(tid)
            motion = 0.0
            if prev is not None and prev.shape == small.shape:
                motion = float(
                    np.mean(np.abs(small.astype(np.float32) - prev.astype(np.float32)))
                )
            self._prev_crops[tid] = small

            fcx = (x1 + x2) * 0.5
            fcy = (y1 + y2) * 0.5
            max_d = float(np.hypot(w, h)) * 0.5 or 1.0
            center_score = max(
                0.0, min(1.0, 1.0 - float(np.hypot(fcx - cx, fcy - cy)) / max_d)
            )
            face_area = max(1, (x2 - x1) * (y2 - y1))
            area_frac = face_area / float(w * h)
            area_score = min(1.0, area_frac * 45.0)
            motion_score = min(1.0, motion / 15.0)
            scores[tid] = 0.75 * motion_score + 0.15 * center_score + 0.10 * area_score

        alive = {int(f.track_id) for f in faces}
        for k in list(self._prev_crops.keys()):
            if k not in alive:
                del self._prev_crops[k]
        return scores

    def identify_speaker(
        self,
        frame: np.ndarray,
        faces: List[Any],
        audio_chunk: Optional[np.ndarray] = None,
        *,
        only_when_speaking: bool = False,
        speech_active: bool = False,
    ) -> Optional[int]:
        if not faces:
            self._prev_crops.clear()
            self._last_pick = None
            return None
        if len(faces) == 1:
            tid = int(faces[0].track_id)
            self._cache_crop(frame, faces[0])
            self._last_pick = tid
            return tid

        if only_when_speaking and not speech_active:
            return self._last_pick

        scores = self.score_faces(frame, faces)
        if not scores:
            return None

        best_tid = max(scores.keys(), key=lambda t: scores[t])
        lp = self._last_pick
        if lp is not None and lp in scores and best_tid != lp:
            try:
                if scores[lp] >= self._sticky_ratio * scores[best_tid]:
                    best_tid = lp
            except (TypeError, KeyError):
                pass

        self._last_pick = best_tid
        return best_tid

    def _cache_crop(self, frame: np.ndarray, face: Any) -> None:
        box = getattr(face, "box", None)
        if box is None or len(box) < 4:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouth_y1 = y1 + int((y2 - y1) * 0.65)
        if mouth_y1 >= y2:
            return
        roi = gray[mouth_y1:y2, x1:x2]
        if roi.size == 0:
            return

        self._prev_crops[int(face.track_id)] = cv2.resize(
            roi, (48, 24), interpolation=cv2.INTER_AREA
        )
