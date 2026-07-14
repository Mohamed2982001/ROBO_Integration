"""
vision/object_recognition.py
ObjectRecognizer class — YOLOv8n with ByteTrack.
Refactored from vision_model.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config.settings as cfg


@dataclass
class ObjectInfo:
    track_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]    # x1, y1, x2, y2
    first_seen: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class ObjectRecognizer:
    """
    Detects and tracks objects in a frame using YOLOv8n.
    Maintains a registry of unique objects seen.
    """

    def __init__(self):
        obj_model_path = str(cfg.ROOT_DIR / cfg.OBJECT_MODEL_PATH)
        self.model = YOLO(obj_model_path)
        self._seen_ids: Dict[int, ObjectInfo] = {}  # track_id → first ObjectInfo
        self._synth_track: int = -1  # ephemeral ids when ByteTrack gives no IDs
        print("[ObjectRecognizer] Ready ✓")

    def detect(self, frame_bgr: np.ndarray) -> List[ObjectInfo]:
        """
        Run YOLOv8 object detection + ByteTrack on a frame.
        Returns list of ObjectInfo for all currently visible objects.
        Excludes 'person' class since faces are detected separately.
        """
        results = self.model.track(
            frame_bgr,
            conf=0.4,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        current: List[ObjectInfo] = []

        def append_object(oid: int, class_name: str, conf: float, bbox):
            if oid not in self._seen_ids:
                self._seen_ids[oid] = ObjectInfo(
                    track_id=oid,
                    class_name=class_name,
                    confidence=conf,
                    box=bbox,
                )
            current.append(
                ObjectInfo(
                    track_id=oid,
                    class_name=class_name,
                    confidence=conf,
                    box=bbox,
                    first_seen=self._seen_ids[oid].first_seen,
                )
            )

        if results.boxes is not None and len(results.boxes) > 0:
            if results.boxes.id is not None:
                for box, obj_id in zip(results.boxes, results.boxes.id):
                    oid = int(obj_id)
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    # Skip 'person' class — faces are handled separately by FaceRecognizer
                    if class_name.lower() == "person":
                        continue
                    conf = float(box.conf[0])
                    bbox = tuple(map(int, box.xyxy[0].tolist()))
                    append_object(oid, class_name, conf, bbox)
            else:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    # Skip 'person' class — faces are handled separately by FaceRecognizer
                    if class_name.lower() == "person":
                        continue
                    conf = float(box.conf[0])
                    bbox = tuple(map(int, box.xyxy[0].tolist()))
                    oid = self._synth_track
                    self._synth_track -= 1
                    append_object(oid, class_name, conf, bbox)

        return current

    def total_unique_seen(self) -> int:
        return len(self._seen_ids)

    def reset(self):
        self._seen_ids.clear()

    @staticmethod
    def draw(frame: np.ndarray, objects: List[ObjectInfo]) -> np.ndarray:
        for obj in objects:
            x1, y1, x2, y2 = obj.box
            label = f"{obj.class_name} ID:{obj.track_id} ({int(obj.confidence*100)}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return frame
