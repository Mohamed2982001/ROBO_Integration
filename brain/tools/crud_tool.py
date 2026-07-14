"""
brain/tools/crud_tool.py
CRUD operations exposed as LLM tools.

Identity registration follows a two-step confirmation flow:
  1. initiate_registration(name)  — stages a pending registration
  2. confirm_registration()       — commits face + voice + profile to DB

Profile changes for locked (registered) users require explicit request
via request_profile_change(field, new_value).
"""

from typing import Dict, Any, Optional
import json
import time
from colorama import Fore, Style

from core.json_util import has_embedding


class CRUDTool:
    def __init__(
        self,
        memory_manager,
        vision_pipeline: Optional[object] = None,
        session_manager: Optional[object] = None,
        ui_state: Optional[dict] = None,
        tts: Optional[object] = None,
    ):
        self.memory = memory_manager
        self.mongo = memory_manager.mongo
        self.vision = vision_pipeline
        self.session = session_manager
        self.ui_state = ui_state
        self.tts = tts

    # ── Generic DB CRUD ──────────────────────────────────────────────────────

    def db_create(self, collection: str, data: Dict[str, Any]) -> str:
        """Insert a record into the database."""
        try:
            doc_id = self.mongo.insert_one(collection, data)
            return f"Successfully created record with id {doc_id}"
        except Exception as e:
            return f"Error creating record: {e}"

    def db_read(self, collection: str, query: Dict[str, Any]) -> str:
        """Query a record from the database."""
        try:
            results = self.mongo.find_many(collection, query, limit=5)
            if not results:
                return "No records found."
            return str(results)
        except Exception as e:
            return f"Error reading records: {e}"

    def db_update(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> str:
        """Update a record in the database."""
        try:
            success = self.mongo.update_one(collection, query, update)
            if success:
                return "Successfully updated record."
            return "No matching record found to update."
        except Exception as e:
            return f"Error updating record: {e}"

    def db_delete(self, collection: str, query: Dict[str, Any]) -> str:
        """Delete a record from the database."""
        try:
            success = self.mongo.delete_one(collection, query)
            if success:
                return "Successfully deleted record."
            return "No matching record found to delete."
        except Exception as e:
            return f"Error deleting record: {e}"

    # ── Identity management (admin) ──────────────────────────────────────────

    def delete_identity(self, user_id: str) -> str:
        """Completely delete a user identity from BOTH MongoDB and Qdrant Identity Embeddings."""
        try:
            success = self.memory.delete_user(user_id)
            if success:
                return f"Successfully wiped identity (ID: {user_id}) from MongoDB and Vector DB."
            return "Error: User ID not found."
        except Exception as e:
            return f"Error deleting identity: {e}"

    def update_identity(self, user_id: str, updates: Dict[str, Any]) -> str:
        """Update the metadata (like name) of an identity in MongoDB and refresh face cache."""
        try:
            success = self.memory.update_user(user_id, updates)
            if success:
                # Refresh face recognizer cache to pick up name changes
                if self.vision:
                    try:
                        self.vision.refresh_face_cache()
                    except Exception as e:
                        pass  # Non-critical; cache refresh failures don't block the response
                return f"Successfully updated identity (ID: {user_id})."
            return "Error: User ID not found."
        except Exception as e:
            return f"Error updating identity: {e}"

    # ── Two-step identity registration ───────────────────────────────────────

    def stage_user_profile(self, profile_data_json: str = "", **kwargs) -> str:
        """
        Stage a pending profile when gathering information about a user.
        Does NOT commit to the database — waits for confirm_registration().
        """
        if not self.session:
            return "Error: Session manager not available."

        profile_data = {}
        raw = profile_data_json
        if isinstance(raw, dict):
            profile_data = raw
        elif isinstance(raw, str) and raw.strip() and raw.strip() not in ("{", "}", "["):
            try:
                profile_data = json.loads(raw)
            except Exception:
                # LLM often sends broken JSON — try to salvage age/name keys
                if "age" in kwargs:
                    profile_data["age"] = kwargs["age"]
                if kwargs.get("name"):
                    profile_data["name"] = kwargs["name"]
                if not profile_data:
                    return (
                        "SYSTEM: Invalid profile JSON. Pass e.g. "
                        '\'{"name": "Mahmoud", "age": 25}\'.'
                    )
        if kwargs:
            profile_data.update({k: v for k, v in kwargs.items() if k in ("name", "age", "nickname", "location", "sport", "hobby", "interests")})
        if not profile_data:
            return "SYSTEM: No profile data provided."

        name = profile_data.get("name", "Unknown")

        # Guard: refuse only if the *active speaker* is already a locked known user.
        scene = self.vision.get_latest_scene() if self.vision else None
        active_uid = scene.current_speaker_id if scene else None
        if active_uid:
            active_user = self.memory.get_user(active_uid)
            if active_user and active_user.get("profile_locked"):
                known_name = active_user.get("name", "Unknown")
                return (
                    f"SYSTEM: The person speaking already has a locked profile as '{known_name}'. "
                    f"Do NOT create a new profile for them. If they want to change their name, "
                    f"use request_profile_change."
                )

        # Get face embedding from vision
        face_emb = None
        if self.vision:
            scene = self.vision.get_latest_scene()
            unknown_faces = [
                f for f in scene.faces
                if f.name == "Unknown" and f.embedding is not None
            ]

            if len(unknown_faces) == 1:
                face_emb = unknown_faces[0].embedding
            elif len(unknown_faces) > 1:
                bio_tid = (self.session.get_biometric_capture() or {}).get("track_id")
                pick_tid = bio_tid or scene.current_speaker_track_id
                speaker_face = next(
                    (
                        f
                        for f in scene.faces
                        if f.track_id == pick_tid
                        and f.name == "Unknown"
                        and f.embedding is not None
                    ),
                    None,
                )
                if speaker_face:
                    face_emb = speaker_face.embedding
                else:
                    return (
                        "SYSTEM: Multiple unknown faces visible and I can't tell which one is speaking. "
                        "Ask them to introduce themselves one at a time, then try again when only one is talking."
                    )
            elif not unknown_faces:
                bio = self.session.get_biometric_capture() if self.session else {}
                face_emb = bio.get("face_embedding") or self.vision.get_current_speaker_embedding()
        else:
            bio = self.session.get_biometric_capture() if self.session else {}
            face_emb = bio.get("face_embedding")

        # Get voice embedding from the current speech turn
        voice_emb = getattr(self.session, "last_voice_embedding", None)

        # Stage the pending registration
        self.session.set_pending_registration(
            profile_data=profile_data,
            face_embedding=face_emb,
            voice_embedding=voice_emb,
        )

        missing = []
        if not profile_data.get("name") or str(profile_data.get("name")).lower() == "unknown":
            missing.append("name")
        if profile_data.get("age") is None:
            missing.append("age")
        if not has_embedding(voice_emb):
            missing.append("voice (keep them talking)")
        if not has_embedding(face_emb):
            missing.append("face (ask them to look at camera)")

        ready = not missing
        return (
            f"SYSTEM: Pending profile staged. Current data: {profile_data}. "
            f"Still needed before confirm: {', '.join(missing) if missing else 'nothing — ready to summarize'}. "
            + (
                "Summarize name + age in their language and ask to confirm, then confirm_registration()."
                if ready
                else "Keep chatting; gather missing items, then summarize and confirm."
            )
        )

    def confirm_registration(self, confirm: bool = True) -> str:
        """
        Commit the pending registration to the database.
        Called after the user confirms their name.
        Registers face + voice + profile and locks the profile.
        """
        if not self.session:
            return "Error: Session manager not available."

        pr = self.session.get_pending_registration()
        if pr is None:
            return (
                "SYSTEM: No pending registration found (it may have expired). "
                "Ask the user to introduce themselves again."
            )

        profile_data = pr.get("profile_data", {})
        name = profile_data.get("name", "Unknown")
        face_emb = pr.get("face_embedding")
        voice_emb = pr.get("voice_embedding")

        bio = self.session.get_biometric_capture() if self.session else {}
        if face_emb is None:
            face_emb = bio.get("face_embedding")
        if voice_emb is None:
            voice_emb = bio.get("voice_embedding")

        if not name or str(name).strip().lower() in ("unknown", ""):
            return (
                "SYSTEM: Cannot confirm yet — need their name. "
                "Ask for their name (Arabic or English) and call stage_user_profile again."
            )
        if profile_data.get("age") is None:
            return (
                "SYSTEM: Cannot confirm yet — need their age. "
                "Ask how old they are and call stage_user_profile with age."
            )
        if not has_embedding(voice_emb):
            return (
                "SYSTEM: Cannot confirm yet — no voice print. "
                "Ask them to say a full sentence so you can learn their voice."
            )
        if not has_embedding(face_emb):
            return (
                "SYSTEM: Cannot confirm yet — no face visible. "
                "Ask them to look at the camera, then try again after they speak."
            )

        try:
            if self.tts:
                msg = "Please wait a moment while I save your data."
                if self.ui_state is not None:
                    self.ui_state["conversation_history"].append({
                        "role": "Robot",
                        "text": msg,
                        "timestamp": time.time()
                    })
                    if len(self.ui_state["conversation_history"]) > 15:
                        self.ui_state["conversation_history"].pop(0)
                    self.ui_state["last_update"] = time.time()
                self.tts.say(msg)

            # 1. Register user with face embedding
            extra_data = {"profile_locked": True}
            extra_data.update(profile_data)
            
            uid = self.memory.register_user(name, face_emb, extra_data=extra_data)

            # 2. Register voice embedding if available
            if voice_emb is not None:
                self.memory.register_voice(uid, voice_emb)
                print(f"{Fore.GREEN}[CRUDTool] ✓ Registered voice for {name}{Style.RESET_ALL}")

            # 3. Update session
            user_info = self.memory.get_user(uid)
            self.session.on_interaction(uid, user_info)
            self.session.clear_pending_registration()

            # 4. Refresh face cache so the label updates immediately
            if self.vision:
                self.vision.refresh_face_cache()

            if self.ui_state is not None:
                self.ui_state["last_registration"] = {
                    "name": name,
                    "user_id": str(uid),
                    "time": time.time(),
                }
                self.ui_state["pending_profile"] = None
                self.ui_state["face_display_names"] = {}
                self.ui_state["last_update"] = time.time()

            return (
                f"SYSTEM: Successfully registered '{name}' with face and voice. "
                f"Greet them warmly by name, then continue the conversation — "
                f"ask about hobbies, work, or what they're up to today, or anything natural. "
                f"Do NOT stop after one sentence; stay social and curious."
            )

        except Exception as e:
            self.session.clear_pending_registration()
            return f"Error during registration: {e}"

    def cancel_registration(self, cancel: bool = True) -> str:
        """Cancel the pending registration if the user says no."""
        if not self.session:
            return "Error: Session manager not available."

        self.session.clear_pending_registration()
        return "SYSTEM: Registration cancelled. Tell the user that's fine and ask their correct name if they want."

    # ── Profile changes for locked users ─────────────────────────────────────

    def request_profile_change(self, field: str, new_value: str) -> str:
        """
        Change a field on a known, locked user's profile.
        Only works if the person is already recognized by the system.
        """
        if not self.session:
            return "Error: Session manager not available."

        current_uid = self.session.current_user_id
        if not current_uid:
            return (
                "SYSTEM: I don't know who this person is yet. "
                "They need to be registered first before changing their profile."
            )

        current_user = self.memory.get_user(current_uid)
        if not current_user:
            return f"Error: User (ID: {current_uid}) not found in database."

        if not current_user.get("profile_locked"):
            return (
                "SYSTEM: This person's profile isn't locked/confirmed yet. "
                "They need to complete registration first."
            )

        # Sanitize field name — only allow safe fields
        allowed_fields = {"name", "nickname", "location", "occupation", "project", "interests"}
        if field not in allowed_fields:
            return f"SYSTEM: Cannot change field '{field}'. Allowed fields: {', '.join(sorted(allowed_fields))}."

        old_value = current_user.get(field, "(not set)")

        try:
            success = self.memory.update_user(current_uid, {field: new_value})
            if success:
                # Update cached info
                current_user[field] = new_value
                self.session.on_interaction(current_uid, current_user)

                # Refresh face cache if name changed
                if field == "name" and self.vision:
                    self.vision.refresh_face_cache()

                return (
                    f"SYSTEM: Updated {field} from '{old_value}' to '{new_value}'. "
                    f"Confirm to the user: 'Done, I've updated your {field} to {new_value}.'"
                )
            return "Error updating profile."
        except Exception as e:
            return f"Error updating profile: {e}"

    # ── Third-party registration ─────────────────────────────────────────────

    def register_new_person(self, name: str) -> str:
        """
        Use this when a known user introduces an unknown person (e.g. 'This is my friend Omar').
        This registers the unknown face in the scene as the new person.
        """
        if self.vision:
            # Get all faces currently in the scene
            scene = self.vision.get_latest_scene()
            unknown_faces = [f for f in scene.faces if f.name == "Unknown" and f.embedding is not None]
            if unknown_faces:
                # Register the first unknown face (with locked profile)
                emb = unknown_faces[0].embedding
                uid = self.memory.register_user(name, emb, extra_data={"profile_locked": True})
                self.vision.refresh_face_cache()
                return (
                    f"SYSTEM: Registered the unknown person as '{name}' with a locked profile. "
                    f"Now greet {name} and ask them to say something so you can learn their voice."
                )
            return "SYSTEM: I don't see any unknown faces clearly right now."
        return "SYSTEM: Vision not available."
