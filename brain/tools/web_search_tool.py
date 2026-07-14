"""
brain/tools/web_search_tool.py
Web search via DuckDuckGo (`ddgs` package).
Keeps it simple: one direct call, news fallback, no region loops.
"""
from __future__ import annotations

import re
import warnings

_DDGS_IMPORT_ERROR: Exception | None = None

# Suppress the deprecation warning from the old package if it is still installed
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    try:
        from ddgs import DDGS  # pip install ddgs
    except ImportError:  # pragma: no cover
        try:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover
            DDGS = None  # type: ignore[misc, assignment]
            _DDGS_IMPORT_ERROR = exc

import config.settings as cfg


def _squash_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _format_block(r: dict, snippets_max: int) -> str:
    title = _squash_spaces(str(r.get("title", "") or ""))
    body = _squash_spaces(str(r.get("body", "") or ""))
    if len(body) > snippets_max:
        body = body[: snippets_max - 3] + "..."
    link = _squash_spaces(str(r.get("href", "") or r.get("url", "") or ""))
    extras = f"\nLink: {link}" if link else ""
    return f"Title: {title}\nSnippet: {body}{extras}"


def search_web(
    query: str,
    max_results: int = 3,
    *,
    conversation_context: str | None = None,
) -> str:
    """Search the web via DuckDuckGo and return formatted snippets."""
    if DDGS is None:  # pragma: no cover
        detail = repr(_DDGS_IMPORT_ERROR)
        return f"Search unavailable (install ddgs: pip install ddgs). Detail: {detail}"

    trimmed = (query or "").strip()
    if not trimmed:
        return "No query provided."

    snippets_max = int(getattr(cfg, "WEB_SEARCH_SNIPPET_MAX_CHARS", 450))
    last_error: Exception | None = None
    rows: list[dict] = []

    # ── 1. Text search (single call, no region) ──────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with DDGS() as ddgs:
                rows = list(ddgs.text(trimmed, max_results=max_results))
    except Exception as e:
        last_error = e

    # ── 2. News fallback ─────────────────────────────────────────────────────
    if not rows:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with DDGS() as ddgs:
                    news_fn = getattr(ddgs, "news", None)
                    if callable(news_fn):
                        rows = list(news_fn(trimmed, max_results=max_results))
        except Exception as e:
            last_error = e

    if not rows:
        detail = repr(last_error) if last_error else "no_results"
        return f"No results returned. Retry with shorter keywords. Detail: {detail}"

    out_lines = [_format_block(r, snippets_max) for r in rows[:max_results]]
    return "\n\n".join(out_lines)
