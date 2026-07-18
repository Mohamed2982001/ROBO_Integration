"""
database/memory_manager.py
High-level bridge between Qdrant (face embeddings) and MongoDB (user profiles).
Handles user registration and identification.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from colorama import Fore, Style, init

import config.settings as cfg
from database.mongo_manager import MongoManager
from database.qdrant_manager import QdrantManager

init(autoreset=True)


class MemoryManager:
    """
    Orchestrates MongoDB + Qdrant for user identity management.

    - Registration: stores face embedding in Qdrant + profile in MongoDB
    - Identification: searches Qdrant with face embedding → fetches MongoDB profile
    """

    def __init__(self):
        self.mongo = MongoManager()
        self.qdrant = QdrantManager()

    # ── Registration ──────────────────────────────────────────────────────────
    def register_user(self, name: str, face_embedding: Any, extra_data: Optional[Dict] = None) -> str:
        """
        Register a new user.
        1. Insert profile into MongoDB → get user_id
        2. Store face embedding in Qdrant linked to user_id
        Returns: mongo user_id (string)
        """
        user_data = {"name": name}
        if extra_data:
            user_data.update(extra_data)

        # Check if name already exists
        existing = self.mongo.get_user_by_name(name)
        if existing:
            user_id = existing["_id"]
            print(f"{Fore.YELLOW}[MemoryManager] User '{name}' already exists (id={user_id}). Updating embedding.{Style.RESET_ALL}")
        else:
            user_id = self.mongo.store_user(user_data)
            print(f"{Fore.GREEN}[MemoryManager] Registered '{name}' -> MongoDB id={user_id}{Style.RESET_ALL}")

        # Store / update face embedding in Qdrant
        self.qdrant.store_face_embedding(face_embedding, user_id)
        print(f"{Fore.GREEN}[MemoryManager] Face embedding stored in Qdrant for user_id={user_id}{Style.RESET_ALL}")
        return user_id

    def register_voice(self, user_id: str, voice_embedding: Any) -> bool:
        """Link a voice embedding to an existing user."""
        user = self.mongo.get_user(user_id)
        if not user:
            print(f"{Fore.RED}[MemoryManager] User ID {user_id} not found.{Style.RESET_ALL}")
            return False
        
        self.qdrant.store_voice_embedding(voice_embedding, user_id)
        print(f"{Fore.GREEN}[MemoryManager] Voice embedding stored for user_id={user_id}{Style.RESET_ALL}")
        return True

    # ── Identification ────────────────────────────────────────────────────────
    def identify_user(self, face_embedding: Any) -> Optional[Dict]:
        """
        Search Qdrant with a face embedding.
        Returns the MongoDB user document if found, else None.
        Document includes: _id, name, + any extra fields.
        """
        results = self.qdrant.search_face(face_embedding, top_k=1)
        if not results:
            return None

        top = results[0]
        user_id = top["payload"].get("user_id")
        confidence = top["score"]

        if not user_id:
            return None

        user = self.mongo.get_user(user_id)
        if user:
            user["face_confidence"] = round(confidence, 4)
        return user

    def identify_voice(self, voice_embedding: Any) -> Optional[Dict]:
        """
        Search Qdrant with a voice embedding.
        Returns the MongoDB user document if found, else None.
        """
        results = self.qdrant.search_voice(voice_embedding, top_k=1)
        if not results:
            return None

        top = results[0]
        user_id = top["payload"].get("user_id")
        confidence = top["score"]

        if not user_id:
            return None

        user = self.mongo.get_user(user_id)
        if user:
            user["voice_confidence"] = round(confidence, 4)
        return user

    # ── User CRUD helpers ─────────────────────────────────────────────────────
    def get_user(self, user_id: str) -> Optional[Dict]:
        return self.mongo.get_user(user_id)

    def update_user(self, user_id: str, update: Dict) -> bool:
        return self.mongo.update_user(user_id, update)

    def delete_user(self, user_id: str) -> bool:
        """Remove user from both MongoDB and Qdrant (face + voice)."""
        ok = self.mongo.delete_user(user_id)
        self.qdrant.delete_by_user_id(cfg.COLLECTION_IDENTITY, user_id)
        self.qdrant.delete_by_user_id(cfg.COLLECTION_VOICE, user_id)
        return ok

    def merge_identities(self, source_user_id: str, target_user_id: str) -> bool:
        """Move all embeddings from source to target, then delete source user."""
        try:
            # 1. Update payloads in Qdrant
            self.qdrant.update_user_id_in_payload(cfg.COLLECTION_IDENTITY, source_user_id, target_user_id)
            self.qdrant.update_user_id_in_payload(cfg.COLLECTION_VOICE, source_user_id, target_user_id)
            
            # 2. Merge LTM memories (Qdrant payload update)
            self.qdrant.update_user_id_in_payload(cfg.COLLECTION_LTM, source_user_id, target_user_id)
            
            # 3. Delete source user from MongoDB
            self.mongo.delete_user(source_user_id)
            print(f"{Fore.GREEN}[MemoryManager] Merged {source_user_id} into {target_user_id}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}[MemoryManager] Error merging: {e}{Style.RESET_ALL}")
            return False

    def list_users(self) -> list:
        return self.mongo.find_all("users")
