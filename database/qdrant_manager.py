"""
database/qdrant_manager.py
Qdrant vector database manager using the official qdrant-client SDK.
Supports multiple collections with different vector sizes.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config.settings as cfg


class QdrantManager:
    """Thread-safe Qdrant manager using qdrant-client SDK."""

    def __init__(self):
        self.client = QdrantClient(url=cfg.QDRANT_URI)
        self.specs = {
            cfg.COLLECTION_IDENTITY: cfg.FACE_EMBEDDING_DIM,
            cfg.COLLECTION_LTM: cfg.TEXT_EMBEDDING_DIM,
            cfg.COLLECTION_VOICE: cfg.VOICE_EMBEDDING_DIM,
        }
        self._init_collections()
        print(f"[Qdrant] Connected → {cfg.QDRANT_URI}")

    # ── Collection bootstrap ──────────────────────────────────────────────────
    def _init_collections(self):
        existing = {c.name for c in self.client.get_collections().collections}
        for name, dim in self.specs.items():
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=dim,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                print(f"[Qdrant] Created collection '{name}' (dim={dim})")
            else:
                print(f"[Qdrant] Collection '{name}' already exists")

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_list(vec: Any) -> List[float]:
        """Accept torch tensors, numpy arrays, or plain lists."""
        if hasattr(vec, "cpu"):  # torch tensor
            return vec.cpu().tolist()
        if hasattr(vec, "tolist"):  # numpy array
            return vec.tolist()
        return list(vec)

    @staticmethod
    def _make_uuid(reference_id: Any) -> str:
        """Deterministic UUID from a MongoDB ObjectId string."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(reference_id)))

    # ── STORE ─────────────────────────────────────────────────────────────────
    def store_embedding(
        self,
        collection: str,
        vector: Any,
        payload: Dict,
        point_id: Optional[str] = None,
    ) -> str:
        """
        Upsert a single vector with payload.
        Returns the point UUID used.
        """
        vec_list = self._to_list(vector)
        pid = point_id or str(uuid.uuid4())
        self.client.upsert(
            collection_name=collection,
            points=[
                qmodels.PointStruct(
                    id=pid,
                    vector=vec_list,
                    payload=payload,
                )
            ],
        )
        return pid

    _MAX_FACE_EMBEDDINGS_PER_USER: int = 20

    def store_face_embedding(self, vector: Any, mongo_user_id: str) -> str:
        """Store a face embedding linked to a MongoDB user id.

        Uses a random UUID so that multiple embeddings accumulate per user
        (angular robustness). Caps at _MAX_FACE_EMBEDDINGS_PER_USER; oldest
        points are pruned when the cap is exceeded.
        """
        pid = str(uuid.uuid4())
        result = self.store_embedding(
            cfg.COLLECTION_IDENTITY,
            vector,
            {"user_id": mongo_user_id},
            point_id=pid,
        )
        # Prune oldest embeddings beyond cap
        self._prune_face_embeddings(mongo_user_id)
        return result

    def _prune_face_embeddings(self, mongo_user_id: str) -> None:
        """Keep at most _MAX_FACE_EMBEDDINGS_PER_USER points per user."""
        try:
            hits = self.client.scroll(
                collection_name=cfg.COLLECTION_IDENTITY,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="user_id",
                            match=qmodels.MatchValue(value=mongo_user_id),
                        )
                    ]
                ),
                limit=200,
                with_payload=False,
                with_vectors=False,
            )[0]
            if len(hits) > self._MAX_FACE_EMBEDDINGS_PER_USER:
                # IDs are UUIDv4 (random), so sort by string for stable ordering;
                # remove the earliest ones (first in sorted order).
                ids_sorted = sorted(str(p.id) for p in hits)
                to_delete = ids_sorted[: len(ids_sorted) - self._MAX_FACE_EMBEDDINGS_PER_USER]
                self.client.delete(
                    collection_name=cfg.COLLECTION_IDENTITY,
                    points_selector=qmodels.PointIdsList(points=to_delete),
                )
        except Exception:
            pass  # Non-critical; next store will retry

    def store_voice_embedding(self, vector: Any, mongo_user_id: str) -> str:
        """Store a voice embedding linked to a MongoDB user id."""
        pid = self._make_uuid(f"voice_{mongo_user_id}")
        return self.store_embedding(
            cfg.COLLECTION_VOICE,
            vector,
            {"user_id": mongo_user_id},
            point_id=pid,
        )

    def store_memory(
        self,
        vector: Any,
        text: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store a text memory embedding in LTM collection."""
        payload: Dict = {"text": text}
        if user_id:
            payload["user_id"] = user_id
        if metadata:
            payload.update(metadata)
        return self.store_embedding(cfg.COLLECTION_LTM, vector, payload)

    # ── SEARCH ────────────────────────────────────────────────────────────────
    def search(
        self,
        collection: str,
        vector: Any,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_payload: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Nearest-neighbour search. Returns list of dicts:
            {id, score, payload}
        """
        vec_list = self._to_list(vector)

        qfilter = None
        if filter_payload:
            must = [
                qmodels.FieldCondition(
                    key=k,
                    match=qmodels.MatchValue(value=v),
                )
                for k, v in filter_payload.items()
            ]
            qfilter = qmodels.Filter(must=must)

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=collection,
                query_vector=vec_list,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                query_filter=qfilter,
            )
        else:
            # Fallback to modern query_points API
            results = self.client.query_points(
                collection_name=collection,
                query=vec_list,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                query_filter=qfilter,
            ).points

        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def search_face(self, vector: Any, top_k: int = 1) -> List[Dict]:
        return self.search(
            cfg.COLLECTION_IDENTITY,
            vector,
            top_k,
            score_threshold=cfg.FACE_THRESHOLD,
        )

    def search_voice(self, vector: Any, top_k: int = 1) -> List[Dict]:
        return self.search(
            cfg.COLLECTION_VOICE,
            vector,
            top_k,
            score_threshold=cfg.VOICE_THRESHOLD,
        )

    def search_memory(
        self, vector: Any, top_k: int = 3, user_id: Optional[str] = None
    ) -> List[Dict]:
        filt = {"user_id": user_id} if user_id else None
        return self.search(cfg.COLLECTION_LTM, vector, top_k, filter_payload=filt)

    # ── DELETE ────────────────────────────────────────────────────────────────
    def delete_point(self, collection: str, point_id: str) -> None:
        self.client.delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=[point_id]),
        )

    def delete_by_user_id(self, collection: str, user_id: str) -> None:
        self.client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="user_id",
                            match=qmodels.MatchValue(value=user_id),
                        )
                    ]
                )
            ),
        )
    def update_user_id_in_payload(self, collection: str, old_id: str, new_id: str) -> None:
        """Update all points with old_id to new_id in their payload."""
        self.client.set_payload(
            collection_name=collection,
            payload={"user_id": new_id},
            points=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=old_id),
                    )
                ]
            ),
        )


    # ── COUNT ─────────────────────────────────────────────────────────────────
    def count(self, collection: str) -> int:
        return self.client.count(collection_name=collection).count

    # ── BULK FETCH ────────────────────────────────────────────────────────────
    def get_all_face_embeddings(self) -> List[Dict]:
        """
        Fetch ALL points from the identity collection.
        Returns list of dicts: {user_id, vector (list[float])}
        Used to build a local in-memory cache for fast per-frame identification.
        """
        all_points = []
        offset = None
        limit = 100

        while True:
            result = self.client.scroll(
                collection_name=cfg.COLLECTION_IDENTITY,
                limit=limit,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            points, next_offset = result
            for p in points:
                uid = (p.payload or {}).get("user_id")
                if uid and p.vector:
                    all_points.append({"user_id": uid, "vector": p.vector})
            if next_offset is None:
                break
            offset = next_offset

        return all_points
