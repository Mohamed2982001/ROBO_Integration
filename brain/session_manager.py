"""
brain/session_manager.py
Manages conversation sessions, timeouts, and flushes STM to LTM on end.
Also holds pending-registration state for the two-step identity confirmation flow.
"""

import threading
import time
from typing import Any, Optional

import config.settings as cfg

# How long a pending registration stays valid before auto-expiring (seconds).
# Increased to 300s (5 mins) since profile building can span multiple conversation turns.
_PENDING_REG_TTL = 300.0


class SessionManager:
    TIMEOUT_SECONDS = getattr(cfg, "SESSION_TIMEOUT_SECONDS", 60)

    def __init__(self, stm, ltm):
        self.stm = stm
        self.ltm = ltm
        self.last_interaction = 0.0
        self.current_user_id: Optional[str] = None
        self.cached_user_info: Optional[dict] = None  # last known user info
        self.session_owner_id: Optional[str] = None  # The person who started the session
        self.last_proactive_reply_time: float = 0.0  # unix time; debounce vision cues
        self.confirm_prompt_sent: bool = False
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # ── Voice embedding from the latest speech turn ──────────────────────
        # Set by the main event loop on every SpeechEvent so that tools can
        # access the voice embedding of the current utterance.
        self.last_voice_embedding: Optional[Any] = None

        # Latest biometrics captured in the background while an unknown speaks.
        self._biometric_capture: dict = {}

        # ── Pending registration (two-step confirmation) ─────────────────────
        # Holds {name, face_embedding, voice_embedding, timestamp} until the
        # user confirms.  Auto-expires after _PENDING_REG_TTL seconds.
        self._pending_registration: Optional[dict] = None

    # ── Pending registration helpers ─────────────────────────────────────────

    def update_biometric_capture(
        self,
        *,
        face_embedding: Any = None,
        voice_embedding: Any = None,
        track_id: Optional[int] = None,
    ) -> None:
        """Background face/voice capture while an unknown person is speaking."""
        with self._lock:
            if face_embedding is not None:
                self._biometric_capture["face_embedding"] = face_embedding
            if voice_embedding is not None:
                self._biometric_capture["voice_embedding"] = voice_embedding
            if track_id is not None:
                self._biometric_capture["track_id"] = track_id

    def get_biometric_capture(self) -> dict:
        with self._lock:
            return dict(self._biometric_capture)

    def clear_biometric_capture(self) -> None:
        with self._lock:
            self._biometric_capture = {}

    def set_pending_registration(
        self,
        profile_data: dict,
        face_embedding: Any,
        voice_embedding: Optional[Any] = None,
    ) -> None:
        """Stage a registration that needs user confirmation before committing."""
        with self._lock:
            bio = self._biometric_capture
            if face_embedding is None:
                face_embedding = bio.get("face_embedding")
            if voice_embedding is None:
                voice_embedding = bio.get("voice_embedding")

            if self._pending_registration:
                current_data = self._pending_registration.get("profile_data", {})
                current_data.update(profile_data)
                self._pending_registration["profile_data"] = current_data
                self._pending_registration["timestamp"] = time.time()
                if face_embedding is not None:
                    self._pending_registration["face_embedding"] = face_embedding
                if voice_embedding is not None:
                    self._pending_registration["voice_embedding"] = voice_embedding
                name = current_data.get("name", "Unknown")
                print(f"[Session] Pending profile updated for '{name}'")
            else:
                self._pending_registration = {
                    "profile_data": profile_data,
                    "face_embedding": face_embedding,
                    "voice_embedding": voice_embedding,
                    "track_id": bio.get("track_id"),
                    "timestamp": time.time(),
                }
                name = profile_data.get("name", "Unknown")
                print(f"[Session] Pending profile staged for '{name}'")

    def get_pending_registration(self) -> Optional[dict]:
        """Return the pending registration if it hasn't expired, else None."""
        with self._lock:
            pr = self._pending_registration
            if pr is None:
                return None
            if (time.time() - pr["timestamp"]) > _PENDING_REG_TTL:
                print("[Session] Pending registration expired.")
                self._pending_registration = None
                return None
            return pr

    def clear_pending_registration(self) -> None:
        with self._lock:
            self._pending_registration = None
            self._biometric_capture = {}
            self.confirm_prompt_sent = False

    @property
    def has_pending_registration(self) -> bool:
        return self.get_pending_registration() is not None

    # ── Core session management (unchanged) ──────────────────────────────────

    @property
    def has_active_session(self) -> bool:
        return self.current_user_id is not None or self.session_owner_id is not None

    def on_interaction(
        self, user_id: Optional[str] = None, user_info: Optional[dict] = None
    ):
        with self._lock:
            self.last_interaction = time.time()
            
            # If no owner yet, set one
            if not self.session_owner_id and user_id:
                self.session_owner_id = user_id
                print(f"[Session] New session owner: {user_id}")

            # Update current user if it's a valid ID (ignore None from brief tracker drops)
            if user_id:
                self.current_user_id = user_id
                if user_info:
                    self.cached_user_info = user_info

            self._reset_timer_locked()

    def on_new_person(self, name: str, user_id: Optional[str] = None):
        """Called when a new person is detected during an active session."""
        print(f"[Session] {name} (ID: {user_id}) joined the session.")
        # We don't change the owner, just acknowledge the presence

    def mark_proactive_reply(self):
        """Call after speaking a proactive vision response (debounce overlapping cues)."""
        with self._lock:
            self.last_proactive_reply_time = time.time()

    def seconds_since_voice_interaction(self) -> float:
        """Wall time since last speech turn touched the session (STT pipeline)."""
        with self._lock:
            return time.time() - self.last_interaction

    def on_speaker_left(self, user_id: Optional[str] = None):
        """Only end session if the OWNER leaves."""
        with self._lock:
            if user_id and user_id == self.session_owner_id:
                print(f"[Session] Owner {user_id} left. Ending session.")
                self._end_session_locked()
            elif not user_id and self.session_owner_id:
                # If we don't know who left but we have an owner, we might want a timeout
                # for now, we rely on the interaction timer
                pass

    def _reset_timer_locked(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(SessionManager.TIMEOUT_SECONDS, self._end_session_thread)
        self._timer.daemon = True
        self._timer.start()

    def _end_session_thread(self):
        with self._lock:
            self._end_session_locked()

    def _end_session_locked(self):
        summary = self.stm.summarize_and_flush()
        if summary:
            self.ltm.store_memory(summary, user_id=self.session_owner_id or self.current_user_id)

        # Keep pending registration across idle timeout while user is still enrolling
        keep_pending = self._pending_registration is not None

        self.current_user_id = None
        self.session_owner_id = None
        self.cached_user_info = None
        if not keep_pending:
            self._pending_registration = None
            self._biometric_capture = {}
        if self._timer:
            self._timer.cancel()
            self._timer = None
        print("[Session] Session ended.")
