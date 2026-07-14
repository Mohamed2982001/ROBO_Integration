"""
Utility: wipe ALL data from Qdrant and MongoDB.
Usage: python reset_db.py
"""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from qdrant_client import QdrantClient
from pymongo import MongoClient
import config.settings as cfg

print("=== Resetting ALL databases ===\n")

# ── Qdrant ──────────────────────────────────────────────────────────
qdrant = QdrantClient(url=cfg.QDRANT_URI)

collections = [c.name for c in qdrant.get_collections().collections]
for col in collections:
    qdrant.delete_collection(col)
    print(f"[Qdrant] Deleted collection: {col}")

print(f"[Qdrant] Done. All {len(collections)} collections removed.\n")

# ── MongoDB ─────────────────────────────────────────────────────────
mongo = MongoClient(cfg.MONGO_URI)
db = mongo[cfg.MONGO_DB_NAME]

col_names = db.list_collection_names()
for col in col_names:
    db.drop_collection(col)
    print(f"[MongoDB] Dropped collection: {col}")

print(f"[MongoDB] Done. All {len(col_names)} collections dropped.\n")

print("=== Reset complete. Restart the robot to recreate collections. ===")
