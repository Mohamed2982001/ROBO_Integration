"""Central configuration loader — reads from .env"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root. Accept "env" as a local fallback because this
# project has been run with both names.
_ROOT = Path(__file__).parent.parent
if not load_dotenv(_ROOT / ".env"):
    load_dotenv(_ROOT / "env")

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Databases ─────────────────────────────────────────────────────────────────
MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "robot_memory")

QDRANT_URI: str = os.getenv("QDRANT_URI", "http://localhost:6333")

# Qdrant collections
COLLECTION_IDENTITY = "identity_embeddings"  # 512-dim face embeddings
COLLECTION_LTM = "ltm_memories"  # 384-dim text embeddings (bge-small)
COLLECTION_VOICE = "voice_embeddings" # 192-dim voice embeddings

# ── Models ────────────────────────────────────────────────────────────────────
FACE_MODEL_PATH: str = os.getenv("FACE_MODEL_PATH", "models/yolov8n-face.pt")
OBJECT_MODEL_PATH: str = os.getenv("OBJECT_MODEL_PATH", "models/yolov8l.pt")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
FACE_EMBEDDING_DIM: int = 512
TEXT_EMBEDDING_DIM: int = 384  # Updated for bge-small
VOICE_EMBEDDING_DIM: int = 192  # spkrec-ecapa-voxceleb
VOICE_EMBEDDING_MIN_SECONDS: float = float(os.getenv("VOICE_EMBEDDING_MIN_SECONDS", "1.5"))
VOICE_THRESHOLD: float = float(os.getenv("VOICE_THRESHOLD", "0.60"))

# ── Vision ────────────────────────────────────────────────────────────────────
# OpenCV device index (0 = default camera; try 1 if an external USB webcam is primary).
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
# Windows only: webcam capture API. Many USB / laptop cams only turn on the activity LED under DirectShow.
#   auto — try DirectShow first, then MSMF/OpenCV default
#   default — OpenCV pick (often MSMF)
#   dshow — cv2.CAP_DSHOW   msmf — cv2.CAP_MSMF
CAMERA_BACKEND: str = os.getenv("CAMERA_BACKEND", "auto").strip().lower()
# Comma-separated words treated as addressing the companion (intent / wake heuristics).
ROBOT_WAKE_WORDS: str = os.getenv(
    "ROBOT_WAKE_WORDS",
    "musa,hey musa",
)
VISION_INTERVAL: float = float(os.getenv("VISION_INTERVAL", "0.5"))
FACE_THRESHOLD: float = float(os.getenv("FACE_THRESHOLD", "0.45"))
FACE_SIM_HISTORY: int = 5
UNKNOWN_IDLE_SECONDS: float = float(os.getenv("UNKNOWN_IDLE_SECONDS", "3.0"))
# Wait for a stable face before first greet (seconds).
UNKNOWN_APPEAR_STABILIZE_SECONDS: float = float(
    os.getenv("UNKNOWN_APPEAR_STABILIZE_SECONDS", "1.0")
)
# Lip-motion scores above this count as "actively speaking" for overlap detection.
LIP_SPEAKER_MOTION_THRESHOLD: float = float(
    os.getenv("LIP_SPEAKER_MOTION_THRESHOLD", "0.22")
)
# Lip-motion active speaker only updates while mic detects user speech.
SPEAKER_ONLY_WHEN_SPEAKING: bool = os.getenv(
    "SPEAKER_ONLY_WHEN_SPEAKING", "true"
).strip().lower() in ("1", "true", "yes", "on")
# auto | cuda | cpu — local models (Whisper, FaceNet, ECAPA, Kokoro).
ROBOT_DEVICE: str = os.getenv("ROBOT_DEVICE", "auto").strip().lower()
# Skip "unknown_idle" prompts if someone spoke recently (face reco often flickers Unknown).
UNKNOWN_IDLE_VOICE_GATE_SECONDS: float = float(
    os.getenv("UNKNOWN_IDLE_VOICE_GATE_SECONDS", "45.0")
)
# Minimum gap between proactive vision spoken replies (name ask / greet).
PROACTIVE_VISION_DEBOUNCE_SECONDS: float = float(
    os.getenv("PROACTIVE_VISION_DEBOUNCE_SECONDS", "25.0")
)
# OCR: minimum EasyOCR confidence per box (0–1 scale).
OCR_MIN_CONFIDENCE: float = float(os.getenv("OCR_MIN_CONFIDENCE", "0.35"))
# Web snippets max length per result (characters) for LLM context.
WEB_SEARCH_SNIPPET_MAX_CHARS: int = int(
    os.getenv("WEB_SEARCH_SNIPPET_MAX_CHARS", "450")
)
SEARCH_CONTEXT_MAX_CHARS: int = int(
    os.getenv("SEARCH_CONTEXT_MAX_CHARS", "420")
)
SEARCH_EXPANSION_MAX_LENGTH: int = int(
    os.getenv("SEARCH_EXPANSION_MAX_LENGTH", "220")
)
# Comma-separated DDG regions (retry order), e.g. wt-wt,us-en,eg-en
_WEB_SEARCH_FALLBACK_REGIONS_RAW: str = os.getenv(
    "WEB_SEARCH_FALLBACK_REGIONS",
    "wt-wt,us-en,eg-en",
).strip()
WEB_SEARCH_FALLBACK_REGIONS: tuple[str, ...] = tuple(
    r.strip() for r in _WEB_SEARCH_FALLBACK_REGIONS_RAW.split(",") if r.strip()
)

# ── STT — faster-whisper ────────────────────────────────────────────────────────
# small | medium | large-v3 — larger improves accent/noise robustness (slower / more VRAM).
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_TASK: str = os.getenv("WHISPER_TASK", "transcribe").strip().lower()
WHISPER_LOG_PROB_THRESHOLD: float = float(
    os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.0")
)
WHISPER_COMPRESSION_RATIO_THRESHOLD: float = float(
    os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")
)
WHISPER_NO_SPEECH_THRESHOLD: float = float(
    os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.72")
)
WHISPER_AVG_LOGPROB_MIN: float = float(
    os.getenv("WHISPER_AVG_LOGPROB_MIN", "-0.85")
)
# Optional path to a plain-text file appended to STT hints (names, jargon, locales).
WHISPER_VOCAB_FILE: str = os.getenv("WHISPER_VOCAB_FILE", "").strip()
# auto = Whisper detects Arabic/English per utterance; en/ar force a single language.
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "auto").strip()
WHISPER_VAD_MIN_SIL_MS: int = int(os.getenv("WHISPER_VAD_MIN_SIL_MS", "450"))

# ── STT — capture loop ──────────────────────────────────────────────────────────
SILENCE_THRESHOLD: float = float(os.getenv("SILENCE_THRESHOLD", "0.02"))
# RMS above SILENCE_THRESHOLD: user barge-in during TTS or post-playback cooldown; 0 = auto (max(STT_BARGE_MIN_RMS, 1.5×SILENCE_THRESHOLD))
_raw_barge = float(os.getenv("BARGE_IN_RMS", "0").strip() or "0")
STT_BARGE_MIN_RMS: float = float(os.getenv("STT_BARGE_MIN_RMS", "0.055"))
BARGE_IN_RMS: float = (
    _raw_barge if _raw_barge > 0 else max(STT_BARGE_MIN_RMS, SILENCE_THRESHOLD * 1.5)
)
# Discard quiet mic briefly after robots stops speaking (suppress speaker tail bleed).
POST_TTS_COOLDOWN_SECONDS: float = float(os.getenv("POST_TTS_COOLDOWN_SECONDS", "0.65"))
SILENCE_DURATION: float = float(os.getenv("SILENCE_DURATION", "0.8"))
MAX_CHUNK_DURATION: float = float(os.getenv("MAX_CHUNK_DURATION", "30.0"))
SAMPLE_RATE: int = 16000

# ── Memory ────────────────────────────────────────────────────────────────────
STM_MAX_MESSAGES: int = int(os.getenv("STM_MAX_MESSAGES", "5"))
SESSION_TIMEOUT_SECONDS: int = int(os.getenv("SESSION_TIMEOUT_SECONDS", "60"))
# Seconds without a detection before emitting person_left for a tracked name.
PERSON_LEFT_TIMEOUT: float = float(os.getenv("PERSON_LEFT_TIMEOUT", "12.0"))
# Suppress redundant new_person greeting if same known name returned within this window.
KNOWN_REAPPEAR_GREET_DEBOUNCE_SECONDS: float = float(
    os.getenv("KNOWN_REAPPEAR_GREET_DEBOUNCE_SECONDS", "8.0")
)

# ── Long-term memory (prompt prefetch on speech turns) ─────────────────────────
LTM_PREFETCH_ENABLED: bool = os.getenv("LTM_PREFETCH_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
LTM_PREFETCH_TOP_K: int = int(os.getenv("LTM_PREFETCH_TOP_K", "3"))

# ── Hardware / Robot Client ───────────────────────────────────────────────────
ROBOT_IP: str = os.getenv("ROBOT_IP", "192.168.1.18").strip()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = _ROOT
MODELS_DIR = _ROOT / "models"
