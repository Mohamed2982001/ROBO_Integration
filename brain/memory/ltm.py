"""
brain/memory/ltm.py
Long-Term Memory using SentenceTransformers and Qdrant.
Per-user partitioned RAG.
"""
from __future__ import annotations

from typing import List, Dict, Optional
import datetime
from sentence_transformers import SentenceTransformer

import config.settings as cfg
from database.qdrant_manager import QdrantManager

class LongTermMemory:
    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant = qdrant_manager
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[LTM] Loading embedding model: {cfg.EMBEDDING_MODEL_NAME} on {device}")
        self.embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL_NAME, device=device)

    def store_memory(self, text: str, user_id: Optional[str] = None):
        """Embed text and store in Qdrant with optional user_id filter."""
        embedding = self.embed_model.encode(text).tolist()
        metadata = {
            "timestamp": datetime.datetime.now().isoformat()
        }
        pid = self.qdrant.store_memory(embedding, text, user_id=user_id, metadata=metadata)
        print(f"[LTM] Stored memory (user_id={user_id}): {text}")
        return pid

    def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 3,
        user_profile_hint: Optional[str] = None,
    ) -> List[str]:
        """
        Vector search with optional profile-conditioned query fusion (better pronoun / entity recall).
        """
        q = (query or "").strip()
        if not q:
            return []

        hint = (user_profile_hint or "").strip()
        vec_q = self.embed_model.encode(q)
        embeddings = [vec_q]

        if hint and len(hint) > 8:
            blended = f"{hint}\nTopic: {q}"[:2000]
            vec_b = self.embed_model.encode(blended)
            embeddings.append((vec_q.astype("float64") + vec_b.astype("float64")) * 0.5)

        best_score: Dict[str, float] = {}
        hit_by_id: Dict[str, Dict] = {}
        wide_k = max(top_k + 4, top_k * 2)
        for emb in embeddings:
            emb_list = emb.tolist()
            hits = self.qdrant.search_memory(emb_list, top_k=wide_k, user_id=user_id)
            for r in hits:
                pid = str(r["id"])
                sc = float(r.get("score", 0.0))
                if sc >= best_score.get(pid, -1.0):
                    best_score[pid] = sc
                    hit_by_id[pid] = r

        ordered = sorted(hit_by_id.keys(), key=lambda i: best_score[i], reverse=True)[
            :top_k
        ]

        memories: List[str] = []
        for pid in ordered:
            hit = hit_by_id[pid]
            payload = hit["payload"]
            mem_id = hit["id"]
            memories.append(
                f"[ID: {mem_id}] [{payload.get('timestamp')}] {payload.get('text')}"
            )

        return memories

    def delete_memory(self, memory_id: str):
        """Delete a specific memory by its ID."""
        self.qdrant.delete_point(cfg.COLLECTION_LTM, memory_id)
        print(f"[LTM] Deleted memory ID: {memory_id}")
        return True

    def clear_user_memories(self, user_id: str):
        """Delete all memories for a specific user."""
        self.qdrant.delete_by_user_id(cfg.COLLECTION_LTM, user_id)
        print(f"[LTM] Cleared all memories for user: {user_id}")
        return True
