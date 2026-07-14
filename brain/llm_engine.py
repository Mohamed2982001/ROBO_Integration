"""
brain/llm_engine.py
LangChain-based LLM Engine. Handles tool-calling loops and streaming response.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional, Callable

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from colorama import Fore, Style

import config.settings as cfg
from brain.llm_provider import build_chat_model
from brain.addressing import (
    assess_directedness,
    build_wake_patterns,
    directedness_notes_for_prompt,
)
from brain.json_safe import json_safe
from brain.prompts import LTM_PROACTIVE_NOTE, STM_CONTEXT_NOTE, SYSTEM_PROMPT
from brain.tools.tool_registry import ToolRegistry, _normalize_tool_call

# Registration tools — handled in code; Groq 8b often breaks these.
_REGISTRATION_TOOL_NAMES = frozenset(
    {
        "stage_user_profile",
        "confirm_registration",
        "cancel_registration",
    }
)


class BrainEngine:
    def __init__(self, config, tool_registry: ToolRegistry):
        self.client = build_chat_model()
        self.tools = tool_registry

        self.last_response_text = ""
        self.last_save_to_memory = False
        self.last_summary = ""

    def _tools_guide_text(self) -> str:
        lines = []
        for spec in self.tools.schemas:
            fn = spec.get("function", {})
            name = fn.get("name", "?")
            desc = (fn.get("description") or "").strip()
            lines.append(f"- `{name}`: {desc}")
        return "\n".join(lines) if lines else "(no tools registered)"

    def _ltm_block(self, ltm_prefetch: Optional[str]) -> str:
        if ltm_prefetch is None:
            return "None (long-term prefetch disabled or no user id)."
        if not str(ltm_prefetch).strip():
            return "None retrieved for this utterance."
        return str(ltm_prefetch).strip()

    def _build_messages(
        self,
        user_text,
        vision_state,
        stm_context,
        user_info=None,
        *,
        ltm_prefetch: Optional[str] = None,
        append_final_human: bool = True,
        user_message_display: Optional[str] = None,
        assess_addressing: bool = True,
    ) -> list:
        display = (
            user_message_display if user_message_display is not None else user_text
        )
        vision_json = (
            json.dumps(vision_state.to_dict(), indent=2) if vision_state else "None"
        )
        addressing_guidance = (
            "Automated cue — obey the behavioural instructions elsewhere in this prompt."
        )
        if (
            assess_addressing
            and append_final_human
            and isinstance(user_text, str)
            and user_text.strip()
        ):
            vd = vision_state.to_dict() if vision_state else {}
            faces = vd.get("faces", []) or []
            names = [
                str(f.get("name", "") or "").strip()
                for f in faces
                if isinstance(f, dict)
            ]
            pc_raw = vd.get("participant_count_estimate")
            participant_count = int(pc_raw) if isinstance(pc_raw, int) else len(faces)
            asm = assess_directedness(
                user_text,
                known_participant_names=names,
                participant_count_estimate=max(participant_count, len(faces)),
                wake_patterns=build_wake_patterns(cfg.ROBOT_WAKE_WORDS),
            )
            diag = directedness_notes_for_prompt(asm)
            if asm.likely_for_robot:
                addressing_guidance = (
                    "Likely meant for Musa — answer normally while staying terse. "
                    + diag
                )
            else:
                addressing_guidance = (
                    "May be overheard chatter between humans — acknowledge lightly if at all "
                    "(one short clause or silence); do not take over their thread unless they steer to you. "
                    + diag
                )

        system_content = SYSTEM_PROMPT.format(
            vision_input=vision_json,
            user_message=display,
            stm_context=STM_CONTEXT_NOTE,
            user_info=json_safe(user_info) if user_info else "Unknown User",
            tools_guide=self._tools_guide_text(),
            relevant_long_term_memory=self._ltm_block(ltm_prefetch),
            addressing_guidance=addressing_guidance,
        )

        messages = [SystemMessage(content=system_content)]

        if stm_context:
            # Inject summary as a system message if present
            summary = (
                stm_context.get("summary", "") if isinstance(stm_context, dict) else ""
            )
            history = (
                stm_context.get("messages", []) if isinstance(stm_context, dict) else []
            )

            if summary:
                messages.append(
                    SystemMessage(content=f"Conversation summary so far: {summary}")
                )

            # Inject recent messages as proper chat history
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        if append_final_human:
            messages.append(HumanMessage(content=user_text))
        return messages

    def _call_no_tools(self, messages: list) -> str:
        """Fallback: call the LLM with no tools and return plain text."""
        messages_copy = messages + [
            SystemMessage(content="Respond naturally in English. Keep it short.")
        ]
        try:
            resp = self.client.invoke(messages_copy)
            return resp.content or ""
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                print(
                    f"{Fore.RED}[Brain] Rate limit hit — API quota exhausted.{Style.RESET_ALL}"
                )
                return "my brain's a bit slow right now, give me a sec"
            print(f"{Fore.RED}[Brain] Fallback error: {e}{Style.RESET_ALL}")
            return "sorry, my brain is having some trouble."

    def _schemas_for_turn(self, enable_tools: bool, registration_mode: bool) -> list:
        if not enable_tools:
            return []
        if registration_mode:
            return [
                s
                for s in self.tools.schemas
                if s.get("function", {}).get("name") not in _REGISTRATION_TOOL_NAMES
            ]
        return self.tools.schemas

    def think(
        self,
        user_text: str,
        vision_state: Any,
        stm_context: Any,
        user_info: Optional[Dict] = None,
        on_tool_call: Optional[Callable[[str], None]] = None,
        ltm_prefetch: Optional[str] = None,
        *,
        enable_tools: bool = True,
        registration_mode: bool = False,
    ) -> str:
        """
        Main logic. Returns the full text.
        Handles tool calls synchronously before returning final text.
        Falls back gracefully if the LLM produces malformed tool syntax.
        """
        self.last_response_text = ""
        self.last_save_to_memory = False
        self.last_summary = ""

        messages = self._build_messages(
            user_text,
            vision_state,
            stm_context,
            user_info,
            ltm_prefetch=ltm_prefetch,
        )

        schemas = self._schemas_for_turn(enable_tools, registration_mode)
        llm_with_tools = (
            self.client.bind_tools(schemas) if schemas else self.client
        )

        try:
            ai_msg = llm_with_tools.invoke(messages)

            tool_rounds = 0
            max_tool_rounds = 4
            while tool_rounds < max_tool_rounds:
                tool_calls = getattr(ai_msg, "tool_calls", None) or []
                if not tool_calls and ai_msg.content:
                    match = re.search(
                        r"<function=(\w+)>?(.*?)</function>",
                        ai_msg.content,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if match:
                        func_name = match.group(1)
                        try:
                            args = json.loads(match.group(2))
                        except Exception:
                            args = {}
                        tool_calls = [
                            {
                                "name": func_name,
                                "args": args,
                                "id": "call_manual_" + func_name,
                            }
                        ]
                        ai_msg.content = ai_msg.content[: match.start()].strip()

                if not tool_calls:
                    break

                messages.append(ai_msg)
                tool_rounds += 1

                for raw_tc in tool_calls:
                    tc = _normalize_tool_call(raw_tc)
                    if tc is None:
                        continue
                    if registration_mode and tc["name"] in _REGISTRATION_TOOL_NAMES:
                        result = (
                            "SYSTEM: Registration is handled automatically. "
                            "Do not call this tool; speak to the user in English."
                        )
                    else:
                        if on_tool_call:
                            on_tool_call(tc["name"])
                        result = self.tools.execute(tc)

                    messages.append(
                        ToolMessage(
                            tool_call_id=tc["id"],
                            name=tc["name"],
                            content=str(result)[:4000],
                        )
                    )

                ai_msg = llm_with_tools.invoke(messages)

            # 3. Final output
            full_text = ai_msg.content or ""

            # Edge case: model returned no tools but empty content
            if not full_text:
                messages.append(ai_msg)
                messages.append(
                    SystemMessage(
                        content="Now provide the final response to the user. DO NOT use tools. SPEAK ENGLISH ONLY."
                    )
                )
                final_msg = self.client.invoke(messages)
                full_text = final_msg.content or ""

        except Exception as e:
            err_low = str(e).lower()
            if any(x in err_low for x in ("rate limit", "429", "quota", "exhausted")):
                print(
                    f"{Fore.RED}[Brain] API limit — {type(e).__name__}{Style.RESET_ALL}"
                )
                full_text = "Give me a second — my connection hiccuped. Say that again?"
            else:
                # Attempt to recover tool call from failed generation (400 Bad Request)
                failed_gen = None
                body = getattr(e, "body", None)
                if body:
                    if isinstance(body, dict):
                        failed_gen = body.get("error", {}).get("failed_generation")
                    elif isinstance(body, str):
                        try:
                            data = json.loads(body)
                            failed_gen = data.get("error", {}).get("failed_generation")
                        except Exception:
                            pass
                if not failed_gen:
                    # Match pattern from the string exception representation
                    match_fg = re.search(r"['\"]failed_generation['\"]\s*:\s*(['\"])(.*?)\1", str(e), re.DOTALL)
                    if match_fg:
                        failed_gen = match_fg.group(2)
                
                recovered_tc = None
                match_tool = None
                if failed_gen:
                    match_tool = re.search(
                        r"<function=(\w+)>?(.*?)</function>",
                        failed_gen,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if match_tool:
                        func_name = match_tool.group(1)
                        try:
                            args = json.loads(match_tool.group(2))
                        except Exception:
                            args = {}
                        recovered_tc = {
                            "name": func_name,
                            "args": args,
                            "id": "call_recovered_" + func_name,
                        }

                if recovered_tc and match_tool:
                    print(
                        f"{Fore.GREEN}[Brain] Recovered tool call from failed generation: "
                        f"{recovered_tc['name']} with args {recovered_tc['args']}{Style.RESET_ALL}"
                    )
                    try:
                        # Extract clean conversational content preceding the tool call
                        spoken_text = failed_gen[:match_tool.start()].strip()
                        # Clean any tags
                        spoken_text = re.sub(r"<function=.*", "", spoken_text, flags=re.IGNORECASE | re.DOTALL)
                        spoken_text = re.sub(r"</function>", "", spoken_text, flags=re.IGNORECASE | re.DOTALL).strip()
                        
                        # Mock the AIMessage with tool_calls set
                        ai_msg = AIMessage(
                            content=spoken_text,
                            tool_calls=[{
                                "name": recovered_tc["name"],
                                "args": recovered_tc["args"],
                                "id": recovered_tc["id"],
                                "type": "tool_call",
                            }]
                        )
                        messages.append(ai_msg)

                        # Execute
                        if registration_mode and recovered_tc["name"] in _REGISTRATION_TOOL_NAMES:
                            result = (
                                "SYSTEM: Registration is handled automatically. "
                                "Do not call this tool; speak to the user in English."
                            )
                        else:
                            if on_tool_call:
                                on_tool_call(recovered_tc["name"])
                            result = self.tools.execute(recovered_tc)

                        messages.append(
                            ToolMessage(
                                tool_call_id=recovered_tc["id"],
                                name=recovered_tc["name"],
                                content=str(result)[:4000],
                            )
                        )

                        # Resume generation loop
                        ai_msg = llm_with_tools.invoke(messages)
                        
                        max_tool_rounds = 4
                        tool_rounds = 1
                        while tool_rounds < max_tool_rounds:
                            tool_calls = getattr(ai_msg, "tool_calls", None) or []
                            if not tool_calls and ai_msg.content:
                                match = re.search(
                                    r"<function=(\w+)>?(.*?)</function>",
                                    ai_msg.content,
                                    re.IGNORECASE | re.DOTALL,
                                )
                                if match:
                                    func_name = match.group(1)
                                    try:
                                        args = json.loads(match.group(2))
                                    except Exception:
                                        args = {}
                                    tool_calls = [
                                        {
                                            "name": func_name,
                                            "args": args,
                                            "id": "call_manual_" + func_name,
                                        }
                                    ]
                                    ai_msg.content = ai_msg.content[: match.start()].strip()

                            if not tool_calls:
                                break

                            messages.append(ai_msg)
                            tool_rounds += 1

                            for raw_tc in tool_calls:
                                tc = _normalize_tool_call(raw_tc)
                                if tc is None:
                                    continue
                                if registration_mode and tc["name"] in _REGISTRATION_TOOL_NAMES:
                                    result = (
                                        "SYSTEM: Registration is handled automatically. "
                                        "Do not call this tool; speak to the user in English."
                                    )
                                else:
                                    if on_tool_call:
                                        on_tool_call(tc["name"])
                                    result = self.tools.execute(tc)

                                messages.append(
                                    ToolMessage(
                                        tool_call_id=tc["id"],
                                        name=tc["name"],
                                        content=str(result)[:4000],
                                    )
                                )

                            ai_msg = llm_with_tools.invoke(messages)

                        full_text = ai_msg.content or ""
                        if not full_text:
                            messages.append(ai_msg)
                            messages.append(
                                SystemMessage(
                                    content="Now provide the final response to the user. DO NOT use tools. SPEAK ENGLISH ONLY."
                                )
                            )
                            final_msg = self.client.invoke(messages)
                            full_text = final_msg.content or ""

                    except Exception as loop_err:
                        print(f"{Fore.RED}[Brain] Recovered execution failed: {loop_err}{Style.RESET_ALL}")
                        full_text = self._call_no_tools(
                            self._build_messages(
                                user_text,
                                vision_state,
                                stm_context,
                                user_info,
                                ltm_prefetch=ltm_prefetch,
                            )
                        )
                else:
                    err_detail = str(e)[:500]
                    print(
                        f"{Fore.YELLOW}[Brain] Tool call failed ({type(e).__name__}): {err_detail}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.YELLOW}[Brain] Retrying without tools...{Style.RESET_ALL}"
                    )
                    full_text = self._call_no_tools(
                        self._build_messages(
                            user_text,
                            vision_state,
                            stm_context,
                            user_info,
                            ltm_prefetch=ltm_prefetch,
                        )
                    )

        # 4. Parse memory tag
        self._parse_memory_tag(full_text)
        return self.last_response_text

    def react_to_vision(
        self,
        event,
        stm_context: Optional[Dict] = None,
        user_info: Optional[Dict] = None,
    ) -> Optional[str]:
        """Proactive response to vision event."""
        prompt = ""
        if (
            event.change_type == "new_person"
            and event.details.get("name")
            and event.details.get("name") != "Unknown"
        ):
            name = event.details.get("name")
            prompt = f"System: A known user named {name} just walked into view. Greet them very casually and briefly in English (like a human friend). Use vision JSON for context."
        elif event.change_type in ("multi_unknown_idle", "multi_unknown_appeared"):
            count = event.details.get("count", 2)
            prompt = (
                f"System: {count} unknown people are in front of you. "
                "Greet them warmly IN ENGLISH and ask each person to introduce themselves "
                "one at a time — only one should talk at a time. Use vision JSON."
            )
        elif event.change_type == "unknown_appeared":
            prompt = (
                "System: You just see an unknown person (face stable in frame). "
                "ALWAYS greet them in ENGLISH — even if they already spoke. "
                "Brief hello, then ask their name. Do NOT call tools yet. Use vision JSON."
            )
        elif event.change_type == "unknown_idle":
            idle = event.details.get("idle_seconds", "?")
            if user_info and user_info.get("name"):
                known_name = user_info.get("name")
                prompt = (
                    f"System: You are mid-conversation with {known_name}. "
                    f"An unknown face track has been standing there for (~{idle}s). "
                    f"Casually ask {known_name} who their friend is or to introduce them briefly in English. Use vision JSON."
                )
            else:
                prompt = (
                    f"System: An unknown face track has been idle (~{idle}s). "
                    "Say hi casually in English and ask who they are or their name briefly. "
                    "If the user profile or vision JSON suggests you already know who is there, "
                    "do NOT re-ask for their name."
                )

        if not prompt:
            return None

        scene = getattr(event, "scene_snapshot", None)
        eff_user_info = user_info or (
            getattr(scene, "user_info", None) if scene is not None else None
        )
        messages = self._build_messages(
            prompt,
            scene,
            stm_context or {},
            eff_user_info,
            ltm_prefetch=LTM_PROACTIVE_NOTE,
            append_final_human=True,
            user_message_display="(proactive vision cue — reply out loud briefly)",
            assess_addressing=False,
        )

        try:
            ai_msg = self.client.invoke(messages)
            full_text = ai_msg.content or ""
            self._parse_memory_tag(full_text)
            return self.last_response_text
        except Exception:
            return None

    def react_to_new_person_during_session(
        self,
        event,
        current_user_info: Optional[Dict],
        stm_context: Optional[Dict] = None,
    ) -> Optional[str]:
        """React to a newcomer while already in a session with someone else."""
        new_name = event.details.get("name", "Unknown")
        current_name = (
            current_user_info.get("name", "friend")
            if current_user_info
            else "the person I'm talking to"
        )

        prompt = (
            f"System: You are mid-conversation with {current_name}. "
            f"A new person named {new_name} just walked in. Acknowledge {new_name} briefly and naturally "
            f"(like 'hey {new_name}'), but don't stop the conversation with {current_name}. "
            "Respond very briefly. Use vision JSON."
        )

        scene = getattr(event, "scene_snapshot", None)
        messages = self._build_messages(
            prompt,
            scene,
            stm_context or {},
            current_user_info,
            ltm_prefetch=LTM_PROACTIVE_NOTE,
            append_final_human=True,
            user_message_display="(proactive vision cue — brief acknowledgement only)",
            assess_addressing=False,
        )

        try:
            ai_msg = self.client.invoke(messages)
            full_text = ai_msg.content or ""
            self._parse_memory_tag(full_text)
            return self.last_response_text
        except Exception:
            return None

    def extract_name(self, text: str) -> Optional[str]:
        """Extract a person's name from an introduction."""
        prompt = f"Extract only the person's name from this text. If no name, return exactly 'None'. Text: '{text}'"
        try:
            ai_msg = self.client.invoke([HumanMessage(content=prompt)])
            name = (ai_msg.content or "").strip()
            if name.lower() == "none":
                return None
            return name
        except Exception:
            return None

    def _parse_memory_tag(self, text: str):
        self.last_save_to_memory = False
        self.last_summary = ""

        # Try to parse a complete tag first
        match = re.search(
            r'<save_to_memory\s+summary=["\'](.*?)["\']\s*/>',
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            self.last_save_to_memory = True
            self.last_summary = match.group(1).strip()

        # Clean the text (removes both complete and incomplete tags at the end)
        clean_text = re.sub(
            r"<save_to_memory.*", "", text, flags=re.IGNORECASE | re.DOTALL
        ).strip()

        # Clean any manual XML function tags and trailing arguments
        clean_text = re.sub(
            r"<function=.*", "", clean_text, flags=re.IGNORECASE | re.DOTALL
        )
        clean_text = re.sub(
            r"</function>", "", clean_text, flags=re.IGNORECASE | re.DOTALL
        ).strip()

        self.last_response_text = clean_text
