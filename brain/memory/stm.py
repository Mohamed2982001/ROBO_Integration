"""
brain/memory/stm.py
Short-Term Memory: keeps last N exchanges as structured chat history.
Summarizes older messages into a single context summary when full.
"""
from __future__ import annotations

import time
from typing import List, Dict, Optional
import config.settings as cfg

class ShortTermMemory:
    def __init__(self, brain_engine):
        self.messages: List[Dict[str, str]] = []  # [{role, content}, ...]
        self.brain_engine = brain_engine
        self.summary = ""

    def add_message(self, user_text: str, robot_text: str):
        """Add a user/robot exchange. Summarize if buffer is full."""
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": robot_text})

        # 1 exchange = 2 messages. Summarize when we exceed MAX_MESSAGES exchanges.
        max_items = cfg.STM_MAX_MESSAGES * 2
        if len(self.messages) > max_items:
            self._summarize_and_trim()

    def get_context(self) -> dict:
        """Return structured context: summary + recent messages list."""
        return {
            "summary": self.summary,
            "messages": list(self.messages),  # copy
        }

    def compact_snippet_for_tools(self, max_chars: int = 800) -> str:
        """Condensed transcript tail for contextual web search query expansion."""
        parts: List[str] = []
        if self.summary:
            parts.append(f"(summary) {self.summary.strip()}")
        for m in self.messages[-10:]:
            role = "user" if m["role"] == "user" else "assistant"
            parts.append(f"{role}: {(m['content'] or '').strip()}")
        blob = "\n".join(parts).strip()
        if len(blob) <= max_chars:
            return blob
        return blob[-max_chars:]

    def _summarize_and_trim(self):
        """Summarize all current messages into self.summary, keep last 2 pairs."""
        # Build prompt for summarization
        lines = ""
        for m in self.messages:
            role = "User" if m["role"] == "user" else "Aria"
            lines += f"{role}: {m['content']}\n"

        prompt = (
            f"You are summarizing a conversation between a user and an AI robot named Aria.\n"
            f"Previous summary: {self.summary or 'None'}\n\n"
            f"New messages:\n{lines}\n"
            f"Write a single concise paragraph summarizing the full conversation so far. "
            f"Focus on topics discussed and key facts mentioned. Be brief."
        )
        try:
            # Since brain_engine.client is now a LangChain ChatModel:
            response = self.brain_engine.client.invoke([{"role": "user", "content": prompt}])
            self.summary = response.content.strip()
            # Keep only the last 2 exchanges (4 messages) for continuity
            self.messages = self.messages[-4:]
        except Exception as e:
            print(f"[STM] Summarization failed: {e}")
            # Fallback: just trim without summarizing
            self.messages = self.messages[-4:]

    def summarize_and_flush(self) -> Optional[str]:
        """Summarize all remaining messages and clear. Returns final summary for LTM."""
        if not self.messages and not self.summary:
            return None

        self._summarize_and_trim()
        final_summary = self.summary
        self.messages.clear()
        self.summary = ""
        return final_summary
