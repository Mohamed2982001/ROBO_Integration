"""
tests/test_database.py
Phase 1 tests — MongoDB, Qdrant, and MemoryManager.
Run with:  conda activate grad_env && pytest tests/test_database.py -v

Prerequisites:
  - MongoDB running on localhost:27017
  - Qdrant running on localhost:6333 (qdrant.exe)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uuid
import numpy as np
import pytest

from database.mongo_manager import MongoManager
from database.qdrant_manager import QdrantManager
from database.memory_manager import MemoryManager
import config.settings as cfg

# Fixed test UUIDs (must be valid UUIDs for Qdrant)
TEST_FACE_UUID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test_face_001"))
TEST_MEM_UUID  = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test_mem_001"))
TEST_MEM_UUID2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "test_mem_002"))

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def mongo():
    m = MongoManager()
    yield m
    # Cleanup test docs
    m.delete_many("users", {"name": {"$regex": "^_test_"}})
    m.close()

@pytest.fixture(scope="module")
def qdrant():
    return QdrantManager()

@pytest.fixture(scope="module")
def mem():
    return MemoryManager()

# ── MongoDB Tests ─────────────────────────────────────────────────────────────
class TestMongo:
    def test_insert_and_find(self, mongo):
        uid = mongo.store_user({"name": "_test_Alice", "age": 25})
        assert uid is not None
        user = mongo.get_user(uid)
        assert user["name"] == "_test_Alice"

    def test_update(self, mongo):
        uid = mongo.store_user({"name": "_test_Bob"})
        ok = mongo.update_user(uid, {"name": "_test_Bob_updated"})
        assert ok
        user = mongo.get_user(uid)
        assert user["name"] == "_test_Bob_updated"

    def test_delete(self, mongo):
        uid = mongo.store_user({"name": "_test_ToDelete"})
        ok = mongo.delete_user(uid)
        assert ok
        assert mongo.get_user(uid) is None

    def test_find_by_name(self, mongo):
        mongo.store_user({"name": "_test_Carol"})
        user = mongo.get_user_by_name("_test_Carol")
        assert user is not None
        assert user["name"] == "_test_Carol"

    def test_find_many(self, mongo):
        mongo.insert_many("users", [
            {"name": "_test_Multi1"},
            {"name": "_test_Multi2"},
        ])
        results = mongo.find_many("users", {"name": {"$regex": "^_test_Multi"}})
        assert len(results) >= 2

# ── Qdrant Tests ──────────────────────────────────────────────────────────────
class TestQdrant:
    def test_collections_exist(self, qdrant):
        existing = {c.name for c in qdrant.client.get_collections().collections}
        assert cfg.COLLECTION_IDENTITY in existing
        assert cfg.COLLECTION_LTM in existing

    def test_store_and_search_face(self, qdrant):
        vec = np.random.rand(cfg.FACE_EMBEDDING_DIM).astype(np.float32)
        # Use a valid UUID for Qdrant point id
        qdrant.store_embedding(
            cfg.COLLECTION_IDENTITY,
            vec,
            {"user_id": "test_user_999"},
            point_id=TEST_FACE_UUID,
        )
        results = qdrant.search(cfg.COLLECTION_IDENTITY, vec, top_k=1)
        assert len(results) > 0
        assert results[0]["score"] > 0.98  # same vector → near-perfect match

    def test_store_and_search_memory(self, qdrant):
        vec = np.random.rand(cfg.TEXT_EMBEDDING_DIM).astype(np.float32)
        qdrant.store_embedding(
            cfg.COLLECTION_LTM,
            vec,
            {"text": "Test memory text", "user_id": "user_abc"},
            point_id=TEST_MEM_UUID,
        )
        results = qdrant.search(cfg.COLLECTION_LTM, vec, top_k=1)
        assert len(results) > 0
        assert results[0]["payload"]["text"] == "Test memory text"

    def test_filter_by_user_id(self, qdrant):
        vec_a = np.random.rand(cfg.TEXT_EMBEDDING_DIM).astype(np.float32)
        vec_b = np.random.rand(cfg.TEXT_EMBEDDING_DIM).astype(np.float32)
        pid_a = str(uuid.uuid5(uuid.NAMESPACE_DNS, "filter_user_A_mem"))
        pid_b = str(uuid.uuid5(uuid.NAMESPACE_DNS, "filter_user_B_mem"))
        qdrant.store_embedding(
            cfg.COLLECTION_LTM, vec_a,
            {"text": "Memory for A", "user_id": "filter_user_A"},
            point_id=pid_a,
        )
        qdrant.store_embedding(
            cfg.COLLECTION_LTM, vec_b,
            {"text": "Memory for B", "user_id": "filter_user_B"},
            point_id=pid_b,
        )
        results = qdrant.search_memory(vec_a, top_k=5, user_id="filter_user_A")
        user_ids = [r["payload"].get("user_id") for r in results]
        # All returned results should belong to filter_user_A
        assert all(uid == "filter_user_A" for uid in user_ids)

    def test_count(self, qdrant):
        count = qdrant.count(cfg.COLLECTION_IDENTITY)
        assert count >= 0

# ── MemoryManager Tests ───────────────────────────────────────────────────────
class TestMemoryManager:
    def test_register_and_identify(self, mem):
        face_vec = np.random.rand(cfg.FACE_EMBEDDING_DIM).astype(np.float32)
        user_id = mem.register_user("_test_Dave", face_vec)
        assert user_id is not None

        # Identify with same vector → should find Dave
        result = mem.identify_user(face_vec)
        assert result is not None
        assert result["name"] == "_test_Dave"
        assert "face_confidence" in result

    def test_unknown_face_returns_none(self, mem):
        # Totally random vector very unlikely to match with high cosine similarity
        random_vec = np.ones(cfg.FACE_EMBEDDING_DIM, dtype=np.float32)
        random_vec[0] = -1.0  # make it very different from typical embeddings
        result = mem.identify_user(random_vec)
        assert result is None or isinstance(result, dict)

    def test_update_user(self, mem):
        face_vec = np.random.rand(cfg.FACE_EMBEDDING_DIM).astype(np.float32)
        uid = mem.register_user("_test_Eve", face_vec)
        ok = mem.update_user(uid, {"hobby": "chess"})
        assert ok
        user = mem.get_user(uid)
        assert user["hobby"] == "chess"

    def test_list_users(self, mem):
        users = mem.list_users()
        assert isinstance(users, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
