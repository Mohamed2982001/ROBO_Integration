"""
vision/vision_pipeline.py
Threaded vision pipeline:
 - Camera capture loop
 - Face detection + identification every VISION_INTERVAL seconds
 - Object detection
 - Meaningful-change filter → pushes VisionChangeEvent to EventQueue
 - Updates shared latest_scene snapshot always
 - Proactive behavior (greet known, ask unknown idle 3s+)
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import cv2
import numpy as np

import config.settings as cfg
from core.events import EventQueue, VisionChangeEvent
from vision.face_recognition import FaceInfo, FaceRecognizer
from vision.object_recognition import ObjectInfo, ObjectRecognizer
from vision.speaker_identifier import LipMovementSpeakerIdentifier, SpeakerIdentifier


def _open_video_capture(index: int) -> cv2.VideoCapture:
    """Open a camera; on Windows the default MSMF stack often leaves the LED off until DShow is used."""
    b = cfg.CAMERA_BACKEND
    if sys.platform != "win32" or b in ("", "default"):
        try:
            return cv2.VideoCapture(index)
        except Exception:
            pass
    if b == "dshow":
        try:
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
        except Exception:
            pass
    if b == "msmf":
        try:
            return cv2.VideoCapture(index, cv2.CAP_MSMF)
        except Exception:
            pass
    # auto
    try:
        cap_ds = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap_ds.isOpened():
            print("[VisionPipeline] Camera backend: DirectShow (CAMERA_BACKEND=auto)")
            return cap_ds
        cap_ds.release()
    except Exception as e:
        print(f"[VisionPipeline] DirectShow failed, falling back: {e}")
        
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print("[VisionPipeline] Camera backend: default (DirectShow did not open this index)")
        return cap
    except Exception:
        pass
        
    return cv2.VideoCapture(index)
    return cap


def dedupe_faces_for_scene_json(faces: List[FaceInfo]) -> List[FaceInfo]:
    """One entry per known identity for LLM context; keep separate Unknown tracks."""
    by_uid: Dict[str, FaceInfo] = {}
    by_name_fallback: Dict[str, FaceInfo] = {}
    by_unknown_tid: Dict[str, FaceInfo] = {}

    for f in faces:
        if f.name == "Unknown":
            key = f"unknown:{f.track_id}"
            old = by_unknown_tid.get(key)
            if old is None or f.confidence > old.confidence:
                by_unknown_tid[key] = f
        elif f.user_id:
            uid = str(f.user_id)
            old = by_uid.get(uid)
            if old is None or f.confidence > old.confidence:
                by_uid[uid] = f
        else:
            nm = f.name
            old = by_name_fallback.get(nm)
            if old is None or f.confidence > old.confidence:
                by_name_fallback[nm] = f

    for identified in by_uid.values():
        by_name_fallback.pop(identified.name, None)

    merged = (
        list(by_uid.values())
        + list(by_name_fallback.values())
        + list(by_unknown_tid.values())
    )
    return sorted(merged, key=lambda x: -x.confidence)


@dataclass
class VisionState:
    """Snapshot of the current scene — updated every VISION_INTERVAL seconds."""

    timestamp: float = field(default_factory=time.time)
    faces: List[FaceInfo] = field(default_factory=list)
    objects: List[ObjectInfo] = field(default_factory=list)
    current_speaker_track_id: Optional[int] = None
    current_speaker_name: Optional[str] = None
    current_speaker_id: Optional[str] = None  # MongoDB _id
    user_info: Optional[Dict] = None  # full MongoDB user document
    current_speaker_embedding: Optional[Any] = None  # for auto-registration

    def to_dict(self) -> Dict:
        faces_out = dedupe_faces_for_scene_json(self.faces)
        return {
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)
            ),
            "faces": [
                {
                    "track_id": f.track_id,
                    "name": f.name,
                    "confidence": round(f.confidence, 3),
                    **({"user_id": f.user_id} if f.user_id else {}),
                    "emotion": f.emotion,
                }
                for f in faces_out
            ],
            "participant_names": [
                f.name for f in faces_out if f.name and f.name != "Unknown"
            ],
            "participant_ids": [
                f.user_id for f in faces_out if f.user_id
            ],
            "objects": [
                {
                    "class": o.class_name,
                    "confidence": round(o.confidence, 3),
                    "track_id": o.track_id,
                    "bbox": [int(o.box[0]), int(o.box[1]), int(o.box[2]), int(o.box[3])],
                }
                for o in self.objects
            ],
            "participant_count_estimate": len(faces_out),
            "group_context": (
                "single"
                if len(faces_out) <= 1
                else ("pair" if len(faces_out) == 2 else "group")
            ),
            # Audio + vision heuristic: whom the pipeline treats as addressing the microphone.
            "active_speaker_track_id": self.current_speaker_track_id,
            "active_speaker_name": self.current_speaker_name,
            "active_speaker_id": self.current_speaker_id,
            "current_speaker": self.current_speaker_name,
            "current_speaker_confidence": (
                round(next((f.confidence for f in faces_out if f.track_id == self.current_speaker_track_id), 0.0), 3)
                if self.current_speaker_track_id is not None
                else None
            ),
        }


class VisionPipeline:
    """
    Runs in a dedicated thread.
    Produces:
      - latest_scene   : VisionState  (always updated, thread-safe read)
      - EventQueue     : VisionChangeEvent (only on meaningful changes)
    """

    def __init__(
        self,
        event_queue: EventQueue,
        memory_manager,  # MemoryManager
        speaker_identifier: Optional[SpeakerIdentifier] = None,
        on_frame_processed: Optional[Callable[[np.ndarray], None]] = None,
        speech_activity: Optional[Any] = None,
        robot: Optional[Any] = None,
        robot_config: Optional[Any] = None,
    ):
        self.event_queue = event_queue
        self.db = memory_manager
        self.speaker_id = speaker_identifier or LipMovementSpeakerIdentifier()
        self.on_frame_processed = on_frame_processed
        self.speech_activity = speech_activity
        self._display_name_for_track: Optional[Callable[[int], Optional[str]]] = None
        self._skip_unknown_greet: bool = False

        self.robot = robot
        self._face_tracker_detector = None
        if self.robot:
            import mediapipe as mp
            from face_tracker import FaceTrackerConfig
            self.robot_cfg = robot_config or FaceTrackerConfig()
            self._face_tracker_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.robot_cfg.model_selection,
                min_detection_confidence=self.robot_cfg.min_detection_confidence,
            )
            # Center pan/tilt targets initially
            self.pan = float((self.robot_cfg.pan_min + self.robot_cfg.pan_max) / 2)
            self.tilt = float((self.robot_cfg.tilt_min + self.robot_cfg.tilt_max) / 2)
            self._last_sent_pan = None
            self._last_sent_tilt = None
            self._last_send_time = 0.0

        # Sub-models
        self.face_recognizer = FaceRecognizer()
        self.obj_recognizer = ObjectRecognizer()

        # Shared state
        self._latest_scene = VisionState()
        self._latest_annotated_frame: Optional[bytes] = None
        self._scene_lock = threading.Lock()

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Change tracking
        self._prev_face_ids: Set[int] = set()
        self._prev_face_names: Dict[int, str] = {}
        self._unknown_first_seen: Dict[int, float] = {}  # track_id → first_seen time
        self._unknown_idle_alerted: Set[int] = set()  # already alerted for this track
        self._unknown_appeared_alerted: Set[int] = set()  # greeted on first sight
        self._multi_unknown_greeted: bool = False

        # Name-based debounced tracking
        self._active_names: Set[str] = set()  # names we have greeted
        self._last_seen: Dict[str, float] = {}  # name → last detected timestamp
        self._last_rolling_reg: Dict[str, float] = {}  # user_id → timestamp
        
        # Robust tracking helpers
        self._recent_departed: Dict[str, Dict] = {} # name -> {embedding, timestamp}
        self._REAPPEAR_GRACE = 20.0 # seconds to keep embedding in cache
        self._PERSON_LEFT_TIMEOUT = cfg.PERSON_LEFT_TIMEOUT
        # Last known embedding while face was visible (fixes empty stash on person_left)
        self._last_known_embedding_by_name: Dict[str, Dict] = {}
        self._last_new_person_emit_time: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self):
        """Start the vision pipeline thread."""
        # Load all face embeddings into local memory (fast in-memory ID)
        self.face_recognizer.load_local_cache(self.db.qdrant, self.db.mongo)

        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="VisionPipeline"
        )
        self._thread.start()
        print("[VisionPipeline] Started ✓")


    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[VisionPipeline] Stopped")

    def get_latest_scene(self) -> VisionState:
        """Thread-safe read of the latest scene snapshot."""
        with self._scene_lock:
            return self._latest_scene

    def get_video_frame(self) -> Optional[bytes]:
        """Get the latest JPEG-encoded annotated frame."""
        with self._scene_lock:
            return self._latest_annotated_frame

    def get_current_speaker_embedding(self) -> Optional[Any]:
        with self._scene_lock:
            return self._latest_scene.current_speaker_embedding

    def refresh_face_cache(self) -> None:
        """Reload local identity cache after a new user has been registered."""
        self.face_recognizer.refresh_cache(self.db.qdrant, self.db.mongo)

    def set_display_name_resolver(
        self, resolver: Optional[Callable[[int], Optional[str]]]
    ) -> None:
        """Map track_id → display name while registration is pending (UI + overlay)."""
        self._display_name_for_track = resolver

    def set_skip_unknown_greet(self, skip: bool) -> None:
        """When True, do not emit unknown_appeared (enrollment already in progress)."""
        self._skip_unknown_greet = skip

    # ── Main Loop ─────────────────────────────────────────────────────────────
    def _loop(self):
        idx = cfg.CAMERA_INDEX
        cap = _open_video_capture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(
                f"[VisionPipeline] ⚠ Cannot open camera index {idx} — running without vision. "
                "Close other apps using the camera; set CAMERA_INDEX in .env if the webcam is not device 0."
            )
            return

        print(f"[VisionPipeline] Camera opened ✓ (device index {idx}; no preview window — capture is background-only)")
        # Wake the sensor — activity LED usually follows first successful grabs, not isOpened().
        warmup_ok = False
        warm_frame = None
        for _ in range(40):
            ret, warm_frame = cap.read()
            if ret and warm_frame is not None:
                warmup_ok = True
                break
            time.sleep(0.05)
        if not warmup_ok:
            print(
                "[VisionPipeline] ⚠ Device opened but no frames — LED may stay off. "
                "Try CAMERA_BACKEND=dshow or msmf, or CAMERA_INDEX=1. "
                "Disable other apps using the camera."
            )

        # First frame after connect should not hit the vision throttle (avoids dropping the warmup grab).
        next_process_time = 0.0
        read_fail_streak = 0
        _warned_read_fail = False

        while self._running:
            if warmup_ok and warm_frame is not None:
                ret, frame = True, warm_frame
                warm_frame = None
            else:
                ret, frame = cap.read()
            if not ret:
                read_fail_streak += 1
                if read_fail_streak >= 60 and not _warned_read_fail:
                    print(
                        "[VisionPipeline] ⚠ Many failed frame reads — check camera privacy settings, "
                        "drivers, or try another CAMERA_INDEX."
                    )
                    _warned_read_fail = True
                time.sleep(0.05)
                continue
            read_fail_streak = 0
            _warned_read_fail = False

            # High-frequency face tracking on every frame
            if self.robot is not None and self._face_tracker_detector is not None:
                try:
                    # MediaPipe requires RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self._face_tracker_detector.process(rgb)
                    if results and results.detections:
                        bbox = results.detections[0].location_data.relative_bounding_box
                        w, h = self.robot_cfg.frame_width, self.robot_cfg.frame_height
                        x_min = int(bbox.xmin * w)
                        y_min = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        target_x = x_min + width // 2
                        target_y = y_min + height // 2
                        
                        center_x = w // 2
                        center_y = h // 2
                        error_x = target_x - center_x
                        error_y = target_y - center_y
                        
                        import math
                        distance = math.hypot(error_x, error_y)
                        if distance > self.robot_cfg.deadzone_radius:
                            step_x = max(-self.robot_cfg.max_step_deg, min(self.robot_cfg.max_step_deg, error_x * self.robot_cfg.pan_gain))
                            step_y = max(-self.robot_cfg.max_step_deg, min(self.robot_cfg.max_step_deg, error_y * self.robot_cfg.tilt_gain))
                            
                            if self.robot_cfg.invert_pan:
                                step_x = -step_x
                            if self.robot_cfg.invert_tilt:
                                step_y = -step_y
                                
                            raw_pan = self.pan - step_x
                            raw_tilt = self.tilt - step_y
                            
                            a = self.robot_cfg.smoothing
                            self.pan = a * self.pan + (1 - a) * raw_pan
                            self.tilt = a * self.tilt + (1 - a) * raw_tilt
                            
                            self.pan = max(self.robot_cfg.pan_min, min(self.robot_cfg.pan_max, self.pan))
                            self.tilt = max(self.robot_cfg.tilt_min, min(self.robot_cfg.tilt_max, self.tilt))
                            
                            now_time = time.time()
                            if now_time - self._last_send_time >= self.robot_cfg.send_interval:
                                pan_i = int(round(self.pan))
                                tilt_i = int(round(self.tilt))
                                moved_enough = (
                                    self._last_sent_pan is None
                                    or abs(pan_i - self._last_sent_pan) >= self.robot_cfg.min_delta_deg
                                    or self._last_sent_tilt is None
                                    or abs(tilt_i - self._last_sent_tilt) >= self.robot_cfg.min_delta_deg
                                )
                                if moved_enough:
                                    import threading
                                    def _send_pose(p, t):
                                        try:
                                            self.robot.set_servos({
                                                self.robot_cfg.pan_channel: p,
                                                self.robot_cfg.tilt_channel: t,
                                            })
                                        except Exception:
                                            pass
                                    threading.Thread(target=_send_pose, args=(pan_i, tilt_i), daemon=True).start()
                                    self._last_sent_pan = pan_i
                                    self._last_sent_tilt = tilt_i
                                    self._last_send_time = now_time
                except Exception as exc:
                    print(f"[VisionPipeline] Face tracking exception: {exc}")

            now = time.time()
            if now < next_process_time:
                time.sleep(0.01)
                continue

            next_process_time = now + cfg.VISION_INTERVAL

            try:
                self._process_frame(frame, now)
            except Exception as e:
                print(f"[VisionPipeline] Error: {e}")

        cap.release()

    def _process_frame(self, frame: np.ndarray, ts: float):
        """Run detection, identification, change detection for one frame."""
        # ── Face detection ────────────────────────────────────────────────────
        raw_faces = self.face_recognizer.detect(frame)
        face_infos: List[FaceInfo] = []
        embedding_by_track: Dict[int, Any] = {}

        for rf in raw_faces:
            tid = rf["track_id"]
            emb = self.face_recognizer.get_embedding(frame, rf["box"])
            if emb is None:
                continue

            embedding_by_track[tid] = emb
            name, conf, uid = self.face_recognizer.identify(
                emb, tid, self.db.qdrant, self.db.mongo
            )

            # Detect emotion
            emotion = self.face_recognizer.detect_emotion(frame, rf["box"])

            face_infos.append(
                FaceInfo(
                    track_id=tid,
                    name=name,
                    confidence=conf,
                    user_id=uid,
                    box=rf["box"],
                    embedding=emb,
                    emotion=emotion,
                )
            )

        # ── Object detection ──────────────────────────────────────────────────
        obj_infos = self.obj_recognizer.detect(frame)

        # ── Speaker identification ────────────────────────────────────────────
        speech_active = bool(
            self.speech_activity and self.speech_activity.is_user_speaking
        )
        only_when_speaking = getattr(cfg, "SPEAKER_ONLY_WHEN_SPEAKING", True)
        lip_scores = {}
        if hasattr(self.speaker_id, "score_faces"):
            lip_scores = self.speaker_id.score_faces(frame, face_infos)
            if self.speech_activity and speech_active:
                motion_thr = float(getattr(cfg, "LIP_SPEAKER_MOTION_THRESHOLD", 0.22))
                active_count = sum(1 for s in lip_scores.values() if s >= motion_thr)
                self.speech_activity.record_multi_speaker_frame(active_count)
                for tid, score in lip_scores.items():
                    self.speech_activity.record_lip_motion(tid, score)

        speaker_tid = self.speaker_id.identify_speaker(
            frame,
            face_infos,
            only_when_speaking=only_when_speaking,
            speech_active=speech_active,
        )
        speaker_face = next((f for f in face_infos if f.track_id == speaker_tid), None)
        user_info = None
        if speaker_face and speaker_face.user_id:
            user_info = self.db.mongo.get_user(speaker_face.user_id)

        # ── Build new scene ───────────────────────────────────────────────────
        new_scene = VisionState(
            timestamp=ts,
            faces=face_infos,
            objects=obj_infos,
            current_speaker_track_id=speaker_tid,
            current_speaker_name=speaker_face.name if speaker_face else None,
            current_speaker_id=speaker_face.user_id if speaker_face else None,
            user_info=user_info,
            current_speaker_embedding=embedding_by_track.get(speaker_tid),
        )

        # ── Update shared state ───────────────────────────────────────────────
        annotated = self._annotate_frame(frame.copy(), new_scene)
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        with self._scene_lock:
            self._latest_scene = new_scene
            self._latest_annotated_frame = buffer.tobytes()

        if self.on_frame_processed is not None:
            self.on_frame_processed(frame)

        # ── Rolling Registration (Angular Robustness) ────────────────────────
        for face in face_infos:
            if face.user_id and face.confidence > 0.8 and face.embedding is not None:
                last_reg = self._last_rolling_reg.get(face.user_id, 0)
                if (ts - last_reg) > 10.0:  # Every 10 seconds per user
                    self.db.qdrant.store_face_embedding(face.embedding, face.user_id)
                    self._last_rolling_reg[face.user_id] = ts
                    # print(f"[Vision] Rolling registration for {face.name}")

        # ── Change detection → emit events ────────────────────────────────────
        self._detect_and_emit_changes(face_infos, new_scene, ts)

    def _detect_and_emit_changes(
        self, faces: List[FaceInfo], scene: VisionState, ts: float
    ):
        current_ids = {f.track_id for f in faces}
        current_names = {f.track_id: f.name for f in faces}

        # Update last_seen timestamp + stash embeddings for every detected name
        for face in faces:
            name = face.name or "Unknown"
            self._last_seen[name] = ts
            if face.name != "Unknown" and face.embedding is not None:
                emb = face.embedding
                if hasattr(emb, "numel") and emb.numel() == 0:
                    pass
                else:
                    self._last_known_embedding_by_name[name] = {
                        "embedding": face.embedding,
                        "timestamp": ts,
                    }

            # 1. NEW NAME — fire new_person only once
            if name != "Unknown" and name not in self._active_names:
                # Check if they were recently departed (bridge the gap)
                was_recent = name in self._recent_departed
                self._recent_departed.pop(name, None) # remove from cache regardless
                
                self._active_names.add(name)
                
                # Only fire event if they aren't "returning" from a very brief gap
                if was_recent:
                    print(f"[Vision] 🔄 Person returned: {name} (re-id)")
                elif (
                    ts - self._last_new_person_emit_time.get(name, 0)
                    < cfg.KNOWN_REAPPEAR_GREET_DEBOUNCE_SECONDS
                ):
                    print(f"[Vision] (debounce) skip new_person greet for {name}")
                else:
                    self.event_queue.push_vision_change(
                        "new_person",
                        {
                            "track_id": face.track_id,
                            "name": name,
                            "user_id": face.user_id,
                            "confidence": face.confidence,
                        },
                        scene,
                    )
                    self._last_new_person_emit_time[name] = ts
                    print(f"[Vision] 👤 New person: {name} (track {face.track_id})")

        # 2. DEBOUNCED person_left — only after PERSON_LEFT_TIMEOUT seconds of absence
        for name in list(self._active_names):
            last = self._last_seen.get(name, 0)
            if (ts - last) > self._PERSON_LEFT_TIMEOUT:
                self._active_names.discard(name)
                self._last_seen.pop(name, None)

                departed = self._last_known_embedding_by_name.pop(name, None)
                if departed and departed.get("embedding") is not None:
                    self._recent_departed[name] = {
                        "embedding": departed["embedding"],
                        "timestamp": ts,
                    }

                self.event_queue.push_vision_change(
                    "person_left",
                    {"name": name, "user_id": next((f.user_id for f in faces if f.name == name), None)},
                    scene,
                )
                print(f"[Vision] 🚪 Person left: {name}")

        # Cleanup old re-id cache
        for name in list(self._recent_departed.keys()):
            if (ts - self._recent_departed[name]["timestamp"]) > self._REAPPEAR_GRACE:
                self._recent_departed.pop(name)

        # 3. Per-track-ID cleanup for unknown idle tracking
        for tid in self._prev_face_ids - current_ids:
            self._unknown_first_seen.pop(tid, None)
            self._unknown_idle_alerted.discard(tid)
            self._unknown_appeared_alerted.discard(tid)
            self.face_recognizer.clear_track(tid)

        unknown_faces_now = [f for f in faces if f.name == "Unknown"]
        if len(unknown_faces_now) < 2:
            self._multi_unknown_greeted = False

        # 3b. FIRST SIGHT — greet after face is stable (not on first noisy frame)
        stabilize = float(getattr(cfg, "UNKNOWN_APPEAR_STABILIZE_SECONDS", 1.0))
        newly_seen_unknowns = []
        # Skip re-greet if we already have a pending name (track ids churn on ByteTrack)
        skip_greet = bool(getattr(self, "_skip_unknown_greet", False))
        for face in unknown_faces_now:
            if face.track_id not in self._unknown_first_seen:
                self._unknown_first_seen[face.track_id] = ts
            if face.track_id in self._unknown_appeared_alerted:
                continue
            stable_for = ts - self._unknown_first_seen[face.track_id]
            if stable_for < stabilize:
                continue
            self._unknown_appeared_alerted.add(face.track_id)
            newly_seen_unknowns.append(face)

        if skip_greet:
            newly_seen_unknowns = []
        if len(newly_seen_unknowns) >= 2 and not self._multi_unknown_greeted:
            self._multi_unknown_greeted = True
            self.event_queue.push_vision_change(
                "multi_unknown_appeared",
                {"count": len(unknown_faces_now)},
                scene,
            )
            print(
                f"[Vision] 👋 First sight: {len(unknown_faces_now)} unknown people — group greet."
            )
        elif len(newly_seen_unknowns) == 1 and len(unknown_faces_now) == 1:
            face = newly_seen_unknowns[0]
            self.event_queue.push_vision_change(
                "unknown_appeared",
                {"track_id": face.track_id},
                scene,
            )
            print(f"[Vision] 👋 First sight: unknown person (track {face.track_id}).")

        # 3. EMOTION SHIFT (placeholder — requires emotion detector)
        for face in faces:
            if face.track_id in self._prev_face_ids:
                # Compare emotions when you add emotion detection
                pass

        # 4. UNKNOWN IDLE > UNKNOWN_IDLE_SECONDS
        newly_idle_unknowns = []
        for face in faces:
            if face.name == "Unknown":
                if face.track_id not in self._unknown_first_seen:
                    self._unknown_first_seen[face.track_id] = ts
                idle_time = ts - self._unknown_first_seen[face.track_id]
                if (
                    idle_time >= cfg.UNKNOWN_IDLE_SECONDS
                    and face.track_id not in self._unknown_idle_alerted
                ):
                    self._unknown_idle_alerted.add(face.track_id)
                    newly_idle_unknowns.append(face)
            else:
                # Known person — reset unknown tracking
                self._unknown_first_seen.pop(face.track_id, None)
                self._unknown_idle_alerted.discard(face.track_id)

        # Emit group or single unknown event
        if len(newly_idle_unknowns) > 1:
            self.event_queue.push_vision_change(
                "multi_unknown_idle",
                {
                    "count": len(newly_idle_unknowns),
                    "idle_seconds": cfg.UNKNOWN_IDLE_SECONDS,
                },
                scene,
            )
            print(f"[Vision] ❓ Multiple unknown people ({len(newly_idle_unknowns)}) idle.")
        elif len(newly_idle_unknowns) == 1:
            face = newly_idle_unknowns[0]
            idle_time = ts - self._unknown_first_seen[face.track_id]
            self.event_queue.push_vision_change(
                "unknown_idle",
                {
                    "track_id": face.track_id,
                    "idle_seconds": round(idle_time, 1),
                },
                scene,
            )
            print(f"[Vision] ❓ Unknown person idle {idle_time:.1f}s")

        # Update previous state
        self._prev_face_ids = current_ids
        self._prev_face_names = current_names

    def _annotate_frame(self, frame: np.ndarray, scene: VisionState) -> np.ndarray:
        """Draw bounding boxes and labels on the frame for visualization."""
        # 1. Draw Objects
        for obj in scene.objects:
            x1, y1, x2, y2 = map(int, obj.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{obj.class_name} {obj.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Draw Faces
        for face in scene.faces:
            x1, y1, x2, y2 = map(int, face.box)
            
            # Color: Cyan for speaker, Red for known, Gray for unknown
            if face.track_id == scene.current_speaker_track_id:
                color = (255, 255, 0) # Cyan
                thickness = 3
            elif face.name != "Unknown":
                color = (0, 255, 255) # Yellow/Gold
                thickness = 2
            else:
                color = (150, 150, 150) # Gray
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            display = face.name
            if display == "Unknown" and self._display_name_for_track:
                override = self._display_name_for_track(int(face.track_id))
                if override:
                    display = f"{override} *"
            name_label = f"{display} ({face.confidence:.2f})"
            if face.emotion:
                name_label += f" [{face.emotion}]"
                
            cv2.putText(frame, name_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. System Overlay
        known_count = len([f for f in scene.faces if f.name and f.name != "Unknown"])
        status_text = f"Users: {known_count} | Speaker: {scene.current_speaker_name or 'None'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame
