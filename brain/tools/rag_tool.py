"""
brain/tools/rag_tool.py
RAG tool to search Long-Term Memory.
"""
import re


class RAGTool:
    def __init__(self, ltm_module, session_manager):
        self.ltm = ltm_module
        self.session = session_manager

    def _profile_hint(self) -> str:
        ui = getattr(self.session, "cached_user_info", None) or {}
        if not isinstance(ui, dict) or not ui:
            return ""
        bits = []
        for key in ("name", "nickname", "location", "occupation", "project", "interests"):
            v = ui.get(key)
            if v:
                bits.append(f"{key}: {v}")
        return "; ".join(bits)[:900]

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """Extract lowercase alphanumeric keywords from text (length >= 3)."""
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return {t for t in tokens if len(t) >= 3}

    @staticmethod
    def _has_keyword_overlap(query: str, context: str, min_overlap: int = 1) -> bool:
        """Check if context contains at least min_overlap keywords from query."""
        query_kws = RAGTool._extract_keywords(query)
        context_kws = RAGTool._extract_keywords(context)
        overlap = query_kws & context_kws
        return len(overlap) >= min_overlap

    def search_memory(self, query: str) -> str:
        """
        Search the user's long-term memory.
        Verify retrieved results contain matching keywords to prevent hallucination.
        """
        user_id = self.session.current_user_id if self.session else None
        if not user_id:
            try:
                from database.mongo_manager import MongoManager
                mongo = MongoManager()
                users = mongo.find_all("users")
                mongo.close()
                if len(users) == 1:
                    user_id = users[0]["_id"]
                    if self.session:
                        self.session.current_user_id = user_id
                        self.session.cached_user_info = users[0]
                elif len(users) > 1:
                    mohamed_user = next((u for u in users if u.get("name", "").lower() == "mohamed"), None)
                    if mohamed_user:
                        user_id = mohamed_user["_id"]
                        if self.session:
                            self.session.current_user_id = user_id
                            self.session.cached_user_info = mohamed_user
            except Exception as e:
                print(f"[RAGTool] Fallback user resolution failed: {e}")
                
        if not user_id:
            return "Error: No user currently recognized."

        results = self.ltm.retrieve(
            query,
            user_id=user_id,
            top_k=7,
            user_profile_hint=self._profile_hint(),
        )
        
        if not results:
            return "No relevant memories found."
        
        # Filter results by keyword overlap to ensure relevance
        verified_results = []
        for result in results:
            if self._has_keyword_overlap(query, result):
                verified_results.append(result)
        
        if not verified_results:
            # Fallback to returning all retrieved semantic results if keyword filter is too strict
            return "\n".join(results)
        
        return "\n".join(verified_results)

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory by its ID."""
        try:
            self.ltm.delete_memory(memory_id)
            return f"Successfully deleted memory with ID: {memory_id}"
        except Exception as e:
            return f"Error deleting memory: {str(e)}"

    def clear_all_memories(self) -> str:
        """Delete ALL long-term memories for the current user."""
        user_id = self.session.current_user_id
        if not user_id:
            return "Error: No user currently recognized."
            
        try:
            self.ltm.clear_user_memories(user_id)
            return "Successfully cleared all memories for the current user."
        except Exception as e:
            return f"Error clearing memories: {str(e)}"
