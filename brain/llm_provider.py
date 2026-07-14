"""
LLM access via Groq with multiple API keys rotation support.
"""

from __future__ import annotations

from typing import Any

from colorama import Fore, Style
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

import config.settings as cfg


class RotatorChatModel:
    """Wrapper that rotates through multiple Groq API keys when one fails."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("No Groq API keys provided.")
        self.keys = keys
        self.current_index = 0
        self.tools_schemas: list[dict[str, Any]] | None = None
        self.current = self._build_model()

    def _build_model(self) -> BaseChatModel:
        key = self.keys[self.current_index]
        masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "..."
        print(
            f"{Fore.MAGENTA}[Brain] LLM: Initializing Groq / {cfg.GROQ_MODEL} "
            f"using key index {self.current_index} ({masked_key}){Style.RESET_ALL}"
        )
        model = ChatGroq(
            api_key=key,
            model_name=cfg.GROQ_MODEL,
            max_retries=1,
            temperature=0.4,
        )
        if self.tools_schemas:
            model = model.bind_tools(self.tools_schemas)
        return model

    def bind_tools(self, schemas: list[dict[str, Any]]) -> "RotatorChatModel":
        self.tools_schemas = schemas
        self.current = self.current.bind_tools(schemas)
        return self

    def rotate_key(self) -> None:
        self.current_index = (self.current_index + 1) % len(self.keys)
        self.current = self._build_model()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        attempts = len(self.keys)
        for i in range(attempts):
            try:
                return self.current.invoke(*args, **kwargs)
            except Exception as exc:
                if len(self.keys) > 1 and i < attempts - 1:
                    print(
                        f"{Fore.YELLOW}[Brain] Groq call failed on key index {self.current_index}. "
                        f"Rotating to next key... Error: {exc}{Style.RESET_ALL}"
                    )
                    self.rotate_key()
                else:
                    raise
        raise RuntimeError("All Groq API keys failed.")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current, name)


def build_chat_model() -> BaseChatModel:
    raw_keys = cfg.GROQ_API_KEY
    groq_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not groq_keys:
        raise ValueError(
            "No Groq API key configured. Set GROQ_API_KEY in your .env file."
        )

    return RotatorChatModel(groq_keys)