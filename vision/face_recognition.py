"""
vision/face_recognition.py
FaceRecognizer — detection, embedding, and identification.

Identification approach (from reference code):
  - At startup, ALL face embeddings are loaded from Qdrant into a local
    in-memory dict  {user_id → {name, embedding}}.
  - Per-frame identification is pure in-memory cosine similarity — no HTTP.
  - A cache-refresh method lets callers reload after a new user is registered.
  - Falls back to Qdrant live search if the cache hasn't been loaded yet.

Tracking / smoothing (unchanged):
  - deque-based name_history + sim_history per track_id  (majority vote)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from bson import ObjectId
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

import config.settings as cfg
from core.device import pick_torch_device

try:
    try:
        from fer import FER
    except ImportError:
        from fer.fer import FER
except ImportError:
    FER = None


@dataclass
class FaceInfo:
    track_id: int
    name: str                          # "Unknown" if not identified
    confidence: float                  # cosine similarity score (0–1)
    user_id: Optional[str]            # MongoDB _id if identified
    box: Tuple[int, int, int, int]    # x1, y1, x2, y2
    embedding: Optional[torch.Tensor]
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    emotion: str = "neutral"           # placeholder for emotion detector


class FaceRecognizer:
    """
    Detects, tracks, and identifies faces in a frame.
    Thread-safe for reading; call from a single processing thread.
    """

    def __init__(self):
        self.device = torch.device(pick_torch_device())
        print(f"[FaceRecognizer] Using device: {self.device}")

        # Detection model
        face_model_path = str(cfg.ROOT_DIR / cfg.FACE_MODEL_PATH)
        self.detector = YOLO(face_model_path)

        # Alignment + embedding model
        self.mtcnn = MTCNN(image_size=160, margin=20, device=self.device)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Emotion detection (optional)
        self.emotion_detector = None
        if FER is not None:
            try:
                self.emotion_detector = FER(
                    mtcnn=False,
                )
                print("[FaceRecognizer] Emotion detector (FER) loaded ✓")
            except Exception as e:
                print(f"[FaceRecognizer] Emotion detector failed to load: {e}; emotions will be 'neutral'")
        else:
            print("[FaceRecognizer] FER not installed. Emotions will default to 'neutral'. Install via: pip install fer")

        # Track history: track_id → {name_history, sim_history, user_id, first_seen}
        self._track_history: Dict[int, dict] = {}

        # ── Local identity cache (old-code style) ─────────────────────────────
        # {user_id: {"name": str, "embedding": torch.Tensor}}
        self._local_db: Dict[str, dict] = {}
        self._cache_ready = False        # True once load_local_cache() succeeds

        print("[FaceRecognizer] Ready ✓")

    # ── Cache management ──────────────────────────────────────────────────────
    def load_local_cache(self, qdrant_manager, mongo_manager) -> None:
        """
        Pull all identity embeddings from Qdrant into memory.
        Call once at startup and after every new registration.
        """
        raw = qdrant_manager.get_all_face_embeddings()   # [{user_id, vector}]
        new_db: Dict[str, dict] = {}

        for item in raw:
            uid = item["user_id"]
            # Skip orphaned / test payloads (Mongo user lookups require ObjectId)
            if uid is None or not ObjectId.is_valid(str(uid)):
                continue
            vec = item["vector"]

            # Average multiple embeddings for the same user (more robust)
            emb = F.normalize(
                torch.tensor(vec, dtype=torch.float32, device=self.device),
                p=2, dim=0
            )

            if uid not in new_db:
                user = mongo_manager.get_user(uid)
                name = user.get("name", "Unknown") if user else "Unknown"
                new_db[uid] = {"name": name, "embedding": emb, "count": 1}
            else:
                # Accumulate for averaging
                new_db[uid]["embedding"] = new_db[uid]["embedding"] + emb
                new_db[uid]["count"] += 1

        # Normalize averaged embeddings
        for uid, data in new_db.items():
            if data["count"] > 1:
                data["embedding"] = F.normalize(data["embedding"], p=2, dim=0)
            data.pop("count")

        self._local_db = new_db
        self._cache_ready = True
        print(f"[FaceRecognizer] Local cache loaded: {len(self._local_db)} user(s)")

    def refresh_cache(self, qdrant_manager, mongo_manager) -> None:
        """Reload identity cache and flush per-track smoothing history.

        Without clearing _track_history the majority-vote deque keeps old
        "Unknown" entries, so a freshly-registered name can't win the vote
        for many frames — the label stays "Unknown" on the UI even though
        the DB already knows the person.
        """
        self.load_local_cache(qdrant_manager, mongo_manager)
        self._track_history.clear()  # Force re-identification on next frame

    # ── Embedding ─────────────────────────────────────────────────────────────
    def get_embedding(self, frame_bgr: np.ndarray, box: Tuple) -> Optional[torch.Tensor]:
        """Extract normalized 512-dim face embedding from a bounding box region."""
        x1, y1, x2, y2 = map(int, box)
        h, w = frame_bgr.shape[:2]
        margin = 30
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)

        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        aligned = self.mtcnn(face_rgb)
        if aligned is None:
            try:
                # Fallback: Resize YOLO-cropped face to 160x160 and normalize manually (MTCNN range)
                face_resized = cv2.resize(face_rgb, (160, 160))
                face_tensor = torch.tensor(face_resized, dtype=torch.float32, device=self.device).permute(2, 0, 1)
                face_tensor = (face_tensor - 127.5) / 128.0
                aligned = face_tensor
            except Exception as e:
                print(f"[FaceRecognizer] Fallback alignment failed: {e}")
                return None

        aligned = aligned.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.facenet(aligned)[0]
        return F.normalize(emb, p=2, dim=0)

    @staticmethod
    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def detect_emotion(self, frame_bgr: np.ndarray, box: Tuple) -> str:
        """
        Detect emotion from a face crop using FER if available.
        Returns emotion label (angry, happy, neutral, etc.) or "neutral" if not available.
        """
        if self.emotion_detector is None:
            return "neutral"

        try:
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame_bgr[y1:y2, x1:x2]
            if face_crop.size == 0:
                return "neutral"
            
            # FER expects RGB, input is BGR
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Predict emotions
            emotion, score = self.emotion_detector.top_emotion(face_rgb)
            if emotion:
                return str(emotion).lower()
        except Exception as e:
            pass
        
        return "neutral"

    # ── Detection + Tracking ──────────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        """
        Run YOLOv8 face detection + ByteTrack on a frame.
        Returns list of dicts: {track_id, box (xyxy), conf}
        Filters out false positives (small boxes, low confidence, partial faces).
        """
        results = self.detector.track(
            frame_bgr,
            conf=0.6,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        faces = []
        h, w = frame_bgr.shape[:2]
        min_face_size = max(20, int(min(h, w) * 0.05))  # Min 5% of smallest dimension
        
        if results.boxes.id is not None:
            for box, tid in zip(results.boxes, results.boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # Filter: face must be reasonably large
                face_width = x2 - x1
                face_height = y2 - y1
                if face_width < min_face_size or face_height < min_face_size:
                    continue
                
                # Filter: face must not be cut off at frame edges (partial faces)
                edge_margin = 10
                if x1 < edge_margin or y1 < edge_margin or x2 > (w - edge_margin) or y2 > (h - edge_margin):
                    # Allow some edge presence, but reject heavily cropped faces
                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        continue
                
                # Filter: aspect ratio should be roughly face-like (0.6–1.7)
                aspect_ratio = face_width / max(face_height, 1)
                if aspect_ratio < 0.6 or aspect_ratio > 1.7:
                    continue
                
                faces.append({
                    "track_id": int(tid),
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                })
        return faces

    # ── Identify ──────────────────────────────────────────────────────────────
    def _identify_from_cache(
        self, embedding: torch.Tensor
    ) -> Tuple[str, float, Optional[str]]:
        """
        Fast in-memory identification (old-code approach).
        Iterates over the local dict and picks the best cosine match.
        """
        best_name = "Unknown"
        best_sim = -1.0
        best_uid = None

        for uid, data in self._local_db.items():
            sim = self.cosine_sim(embedding, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_name = data["name"]
                best_uid = uid

        if best_sim < cfg.FACE_THRESHOLD:
            best_name = "Unknown"
            best_uid = None

        return best_name, max(best_sim, 0.0), best_uid

    def _identify_from_qdrant(
        self,
        embedding: torch.Tensor,
        qdrant_manager,
        mongo_manager,
    ) -> Tuple[str, float, Optional[str]]:
        """Fallback: live Qdrant search (used only before cache is ready)."""
        results = qdrant_manager.search_face(embedding, top_k=1)
        best_name = "Unknown"
        best_sim = 0.0
        user_id = None

        if results:
            top = results[0]
            best_sim = top["score"]
            if best_sim >= cfg.FACE_THRESHOLD:
                uid = top["payload"].get("user_id")
                if uid:
                    user = mongo_manager.get_user(uid)
                    if user:
                        best_name = user.get("name", "Unknown")
                        user_id = uid
        return best_name, best_sim, user_id

    def identify(
        self,
        embedding: torch.Tensor,
        track_id: int,
        qdrant_manager,
        mongo_manager,
    ) -> Tuple[str, float, Optional[str]]:
        """
        Identify a face embedding, then apply temporal smoothing.
        Uses local in-memory cache when ready; falls back to Qdrant otherwise.
        Returns: (smoothed_name, avg_confidence, user_id)
        """
        # ── 1. Raw identification ─────────────────────────────────────────────
        if self._cache_ready:
            best_name, best_sim, user_id = self._identify_from_cache(embedding)
        else:
            best_name, best_sim, user_id = self._identify_from_qdrant(
                embedding, qdrant_manager, mongo_manager
            )

        # ── 2. Update track history ───────────────────────────────────────────
        if track_id not in self._track_history:
            self._track_history[track_id] = {
                "name_history": deque(maxlen=cfg.FACE_SIM_HISTORY),
                "sim_history":  deque(maxlen=cfg.FACE_SIM_HISTORY),
                "user_id":      None,
                "first_seen":   time.time(),
            }

        hist = self._track_history[track_id]
        if best_name != "Unknown" and best_sim >= cfg.FACE_THRESHOLD:
            hist["name_history"].clear()
            hist["sim_history"].clear()
        hist["name_history"].append(best_name)
        hist["sim_history"].append(best_sim)
        if user_id:
            hist["user_id"] = user_id

        # ── 3. Majority vote + average similarity ─────────────────────────────
        smoothed_name = max(
            set(hist["name_history"]),
            key=hist["name_history"].count,
        )
        avg_sim = float(np.mean(hist["sim_history"]))
        smoothed_uid = hist["user_id"]

        return smoothed_name, avg_sim, smoothed_uid

    # ── Misc ──────────────────────────────────────────────────────────────────
    def get_idle_time(self, track_id: int) -> float:
        hist = self._track_history.get(track_id)
        if hist is None:
            return 0.0
        return time.time() - hist["first_seen"]

    def clear_track(self, track_id: int):
        self._track_history.pop(track_id, None)

    def active_track_ids(self) -> set:
        return set(self._track_history.keys())

    # ── Draw ──────────────────────────────────────────────────────────────────
    @staticmethod
    def draw(frame: np.ndarray, faces: List[FaceInfo]) -> np.ndarray:
        for f in faces:
            x1, y1, x2, y2 = f.box
            color = (0, 255, 0) if f.name != "Unknown" else (0, 0, 255)
            label = f"{f.name} ({int(f.confidence * 100)}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
