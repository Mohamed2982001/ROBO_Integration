"""
brain/tools/tool_registry.py
Registry for LLM tools. Maps tool names to executable functions and provides JSON schemas for Groq.
"""
import json
from typing import Callable, Dict, Any, Optional


def _normalize_tool_call(tool_call) -> Optional[Dict[str, Any]]:
    """LangChain may return dicts or ToolCall objects; Groq needs valid ids."""
    if tool_call is None:
        return None
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        tid = tool_call.get("id") or tool_call.get("tool_call_id")
    else:
        name = getattr(tool_call, "name", None)
        args = getattr(tool_call, "args", {})
        tid = getattr(tool_call, "id", None)
    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {
        "name": name,
        "args": args,
        "id": tid or f"call_{name}",
    }


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: list[Dict[str, Any]] = []

    def register(self, name: str, description: str, func: Callable, parameters: Dict[str, Any]):
        """Register or replace a tool schema (later bindings can supersede stubs)."""
        self.tools[name] = func
        self.schemas = [
            s for s in self.schemas if s.get("function", {}).get("name") != name
        ]
        self.schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    def execute(self, tool_call) -> Any:
        """Execute a tool call from the LangChain AIMessage."""
        norm = _normalize_tool_call(tool_call)
        if norm is None:
            return "Error: invalid tool call."
        name = norm["name"]
        if name not in self.tools:
            return f"Error: Tool '{name}' not found."

        kwargs = norm["args"]
        print(f"[Tool] Executing {name} with args {kwargs}")
        try:
            result = self.tools[name](**kwargs)
            return str(result)
        except TypeError as e:
            print(f"[Tool] Error executing {name}: {e}")
            return f"Error executing tool: {e} — check required arguments."
        except Exception as e:
            print(f"[Tool] Error executing {name}: {e}")
            return f"Error executing tool: {e}"
