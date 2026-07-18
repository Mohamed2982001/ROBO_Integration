"""
database/mongo_manager.py
Full CRUD wrapper around MongoDB using pymongo.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

import config.settings as cfg


class MongoManager:
    """Thread-safe MongoDB manager (pymongo client is thread-safe by default)."""

    def __init__(self):
        self.client = MongoClient(cfg.MONGO_URI)
        self.db = self.client[cfg.MONGO_DB_NAME]
        print(f"[MongoDB] Connected -> {cfg.MONGO_URI} / {cfg.MONGO_DB_NAME}")

    # ── internal helpers ──────────────────────────────────────────────────────
    def _col(self, name: str) -> Collection:
        return self.db[name]

    @staticmethod
    def _oid(id_val: Any) -> ObjectId:
        return id_val if isinstance(id_val, ObjectId) else ObjectId(str(id_val))

    # ── CREATE ────────────────────────────────────────────────────────────────
    def insert_one(self, collection: str, document: Dict) -> str:
        """Insert a document and return its string id."""
        result = self._col(collection).insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, collection: str, documents: List[Dict]) -> List[str]:
        result = self._col(collection).insert_many(documents)
        return [str(i) for i in result.inserted_ids]

    # ── READ ──────────────────────────────────────────────────────────────────
    def find_by_id(self, collection: str, doc_id: Any) -> Optional[Dict]:
        doc = self._col(collection).find_one({"_id": self._oid(doc_id)})
        return self._serialize(doc)

    def find_one(self, collection: str, query: Dict) -> Optional[Dict]:
        doc = self._col(collection).find_one(query)
        return self._serialize(doc)

    def find_many(self, collection: str, query: Dict, limit: int = 50) -> List[Dict]:
        cursor = self._col(collection).find(query).limit(limit)
        return [self._serialize(d) for d in cursor]

    def find_all(self, collection: str) -> List[Dict]:
        return [self._serialize(d) for d in self._col(collection).find()]

    # ── UPDATE ────────────────────────────────────────────────────────────────
    def update_by_id(self, collection: str, doc_id: Any, update: Dict) -> bool:
        result = self._col(collection).update_one(
            {"_id": self._oid(doc_id)}, {"$set": update}
        )
        return result.modified_count > 0

    def update_one(self, collection: str, query: Dict, update: Dict) -> bool:
        result = self._col(collection).update_one(query, {"$set": update})
        return result.modified_count > 0

    def upsert_one(self, collection: str, query: Dict, update: Dict) -> str:
        result = self._col(collection).update_one(query, {"$set": update}, upsert=True)
        uid = result.upserted_id or self._col(collection).find_one(query)["_id"]
        return str(uid)

    # ── DELETE ────────────────────────────────────────────────────────────────
    def delete_by_id(self, collection: str, doc_id: Any) -> bool:
        result = self._col(collection).delete_one({"_id": self._oid(doc_id)})
        return result.deleted_count > 0

    def delete_one(self, collection: str, query: Dict) -> bool:
        result = self._col(collection).delete_one(query)
        return result.deleted_count > 0

    def delete_many(self, collection: str, query: Dict) -> int:
        result = self._col(collection).delete_many(query)
        return result.deleted_count

    # ── User helpers ─────────────────────────────────────────────────────────
    def store_user(self, user_data: Dict) -> str:
        """Insert a user document, return string _id."""
        return self.insert_one("users", user_data)

    def get_user(self, user_id: Any) -> Optional[Dict]:
        return self.find_by_id("users", user_id)

    def get_user_by_name(self, name: str) -> Optional[Dict]:
        return self.find_one("users", {"name": name})

    def update_user(self, user_id: Any, update: Dict) -> bool:
        return self.update_by_id("users", user_id, update)

    def delete_user(self, user_id: Any) -> bool:
        return self.delete_by_id("users", user_id)

    # ── Serialization ─────────────────────────────────────────────────────────
    @staticmethod
    def _serialize(doc: Optional[Dict]) -> Optional[Dict]:
        """Convert ObjectId fields to strings so the doc is JSON-serializable."""
        if doc is None:
            return None
        doc["_id"] = str(doc["_id"])
        return doc

    def close(self):
        self.client.close()
