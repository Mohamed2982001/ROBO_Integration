"""
main.py
AI Robot System Orchestrator.
"""

import json
import time
import os
import signal
import sys
import uuid
import wave
from colorama import Fore, Style, init

import config.settings as cfg
from core.events import EventQueue, SpeechEvent, VisionChangeEvent
from core.speech_activity import SpeechActivityTracker
from database.memory_manager import MemoryManager
from vision.vision_pipeline import VisionPipeline
from voice.stt_engine import HearingEngine
from voice.tts_engine import VoiceEngine
from brain.llm_engine import BrainEngine
from brain.session_manager import SessionManager
from brain.memory.stm import ShortTermMemory
from brain.memory.ltm import LongTermMemory
from brain.tools.tool_registry import ToolRegistry
from brain.tools.ocr_tool import OCRTool
from brain.tools.web_search_tool import search_web
from brain.tools.rag_tool import RAGTool
from brain.tools.crud_tool import CRUDTool
from brain.tools.stm_tool import STMTool
from brain.locale import overlap_interrupt_message
from brain.registration_controller import (
    process_registration_turn,
    registration_hint_for_llm,
)
from core.json_util import sanitize_for_api
from vision.vision_server import VisionServer

init(autoreset=True)

running = True


def signal_handler(sig, frame):
    global running
    print(f"\n{Fore.RED}[Main] Shutting down...{Style.RESET_ALL}")
    running = False


signal.signal(signal.SIGINT, signal_handler)


UI_STATE = {
    "conversation_history": [],
    "system_status": "Active",
    "last_update": time.time(),
    "voice_events": [],
    "last_registration": None,
    "pending_profile": None,
    "face_display_names": {},
    "pending_display_name": None,
}


def _face_for_track(scene, track_id):
    if track_id is None or scene is None:
        return None
    return next((f for f in scene.faces if f.track_id == track_id), None)


def _resolve_speaker_from_scene(scene, speech_activity, voice_user_info):
    """Prefer voice ID but perform multi-modal fusion when face is also active/speaking."""
    # Find active face if any (either matching the utterance lip-speaker or current speaker track)
    utter_tid = (
        speech_activity.get_utterance_speaker_track_id()
        if speech_activity
        else None
    )
    face = _face_for_track(scene, utter_tid)
    if face is None and scene is not None:
        face = _face_for_track(scene, scene.current_speaker_track_id)

    # Multi-modal Fusion: Resolve voice and face identifier mismatch
    if voice_user_info and face and face.user_id:
        voice_uid = str(voice_user_info["_id"])
        face_uid = str(face.user_id)

        if voice_uid != face_uid:
            voice_conf = voice_user_info.get("voice_confidence", 0)
            if face.name != "Unknown" and (utter_tid == face.track_id or voice_conf < 0.70):
                print(
                    f"{Fore.YELLOW}[MultiModalFusion] Conflict detected! "
                    f"Voice={voice_user_info.get('name')} (conf={voice_conf}), "
                    f"Face={face.name} (conf={face.confidence}). "
                    f"Preferring Face due to visual presence.{Style.RESET_ALL}"
                )
                info = None
                if scene.user_info and str(scene.user_info.get("_id")) == face_uid:
                    info = scene.user_info
                return face_uid, info, face

    # Default hierarchy
    if voice_user_info:
        return str(voice_user_info["_id"]), voice_user_info, None

    if face is not None:
        uid = str(face.user_id) if face.user_id else None
        info = None
        if (
            face.user_id
            and scene.user_info
            and str(scene.user_info.get("_id")) == str(face.user_id)
        ):
            info = scene.user_info
        return uid, info, face
    return scene.current_speaker_id, scene.user_info, None

def push_ui_dialogue(role: str, text: str):
    """Push a message to the UI conversation history."""
    UI_STATE["conversation_history"].append({
        "role": role,
        "text": text,
        "timestamp": time.time()
    })
    # Keep last 15 messages
    if len(UI_STATE["conversation_history"]) > 15:
        UI_STATE["conversation_history"].pop(0)
    UI_STATE["last_update"] = time.time()


def _vision_stm_user_line(event: VisionChangeEvent) -> str:
    """Synthetic STM user slot for proactive vision (no microphone turn)."""

    def _idle_sec() -> str:
        raw = event.details.get("idle_seconds", "?")
        return str(round(raw, 1)) if isinstance(raw, (int, float)) else str(raw)

    if event.change_type == "new_person":
        nm = event.details.get("name", "Unknown")
        return (
            f"(Proactive vision — user did not speak yet) Known user {nm} entered the "
            "frame; you greeted them out loud."
        )
    if event.change_type == "unknown_idle":
        return (
            f"(Proactive vision — user did not speak yet) An unknown face stayed idle "
            f"for ~{_idle_sec()}s in frame; greet briefly."
        )
    if event.change_type == "unknown_appeared":
        return (
            "(Proactive vision — user did not speak yet) You just saw an unknown person "
            "for the first time; greet them and ask their name."
        )
    if event.change_type == "multi_unknown_appeared":
        return (
            "(Proactive vision — user did not speak yet) You see multiple unknown people; "
            "greet them and ask each to introduce themselves one at a time."
        )
    return f"(Proactive vision — user did not speak yet) cue: {event.change_type}"


def _ltm_user_profile_fragment(user_doc) -> str:
    if not user_doc or not isinstance(user_doc, dict):
        return ""
    bits = []
    for key in ("name", "nickname", "location", "occupation", "project", "interests"):
        v = user_doc.get(key)
        if v:
            bits.append(f"{key}: {v}")
    return "; ".join(bits)[:900]


def main():
    print(f"{Fore.GREEN}=== Starting AI Robot System ==={Style.RESET_ALL}")

    # 1. Init Core & Database
    event_queue = EventQueue()
    db = MemoryManager()
    speech_activity = SpeechActivityTracker()

    # 1.1 Init Robot client
    from robot_client import Robot
    from face_tracker import FaceTrackerConfig
    print(f"[Main] Connecting to robot at {cfg.ROBOT_IP}...")
    bot = Robot(host=cfg.ROBOT_IP, auto_fetch_limits=True)
    if bot.ping():
        print(f"[Main] Connected to robot at {cfg.ROBOT_IP} ✓")
        bot.set_speed(160)
        bot.center_all_servos()
    else:
        print(f"[Main] WARNING: Robot unreachable at {cfg.ROBOT_IP}. Running in mock/unconnected mode.")

    # 2. Init Voice
    tts = VoiceEngine(robot=bot)

    # 3. Init Vision (OCR registered below; pass frame hook after ocr exists)
    ocr = OCRTool()

    def _on_frame_processed(frame):
        ocr.update_frame(frame)

    vision = VisionPipeline(
        event_queue,
        db,
        on_frame_processed=_on_frame_processed,
        speech_activity=speech_activity,
        robot=bot,
        robot_config=FaceTrackerConfig(),
    )

    def _display_name_for_track(track_id: int):
        return UI_STATE.get("face_display_names", {}).get(str(track_id))

    vision.set_display_name_resolver(_display_name_for_track)

    def _sync_vision_greet_skip():
        vision.set_skip_unknown_greet(
            bool(UI_STATE.get("pending_display_name"))
            or bool(session.get_pending_registration())
        )

    # 3.1. Init Vision Dashboard Server
    viz_server = VisionServer(vision, port=8080, ui_state=UI_STATE)
    viz_server.start()

    # 4. Init LTM (لا تحتاج brain)
    ltm = LongTermMemory(db.qdrant)

    # 5. Init Tool Registry وسجل الـ tools اللي مش محتاجة session أو stm
    registry = ToolRegistry()

    registry.register(
        "perform_ocr",
        "Extract text from the current camera view.",
        ocr.perform_ocr,
        {"type": "object", "properties": {}},
    )

    def move_robot(direction: str, duration: float) -> str:
        try:
            bot.move_for(direction, duration, block=True)
            return f"[Hardware] Moved {direction} for {duration} seconds."
        except Exception as e:
            return f"[Hardware] Error moving: {e}"

    def set_robot_pose(pose_name: str) -> str:
        try:
            if pose_name == "wave":
                import threading
                threading.Thread(target=bot.wave_arm, daemon=True).start()
                return "[Hardware] Waving arm..."
            else:
                bot.set_pose(pose_name)
                return f"[Hardware] Robot pose set to {pose_name}."
        except Exception as e:
            return f"[Hardware] Error setting pose: {e}"

    def control_gripper(claw: str, action: str) -> str:
        try:
            if claw == "right":
                if action == "open":
                    bot.open_gripper_r()
                else:
                    bot.close_gripper_r()
            else:
                if action == "open":
                    bot.open_gripper_l()
                else:
                    bot.close_gripper_l()
            return f"[Hardware] {claw.capitalize()} gripper {action}ed."
        except Exception as e:
            return f"[Hardware] Error controlling gripper: {e}"

    def center_head() -> str:
        try:
            bot.set_servos({0: 90, 7: 80})
            return "[Hardware] Head centered."
        except Exception as e:
            return f"[Hardware] Error centering head: {e}"

    registry.register(
        "move_robot",
        "Move/drive the physical Mecanum chassis robot in a specific direction for a set duration. Valid directions: forward, backward, left, right, rotate_left, rotate_right.",
        move_robot,
        {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["forward", "backward", "left", "right", "rotate_left", "rotate_right"]},
                "duration": {"type": "number", "description": "Time in seconds to drive."}
            },
            "required": ["direction", "duration"]
        }
    )

    registry.register(
        "set_robot_pose",
        "Set the physical robot's arms and shoulders to a predefined stance. Stances: home, arms_up, arms_down, wave.",
        set_robot_pose,
        {
            "type": "object",
            "properties": {
                "pose_name": {"type": "string", "enum": ["home", "arms_up", "arms_down", "wave"]}
            },
            "required": ["pose_name"]
        }
    )

    registry.register(
        "control_gripper",
        "Open or close the left or right mechanical claw gripper. Claw: left, right. Action: open, close.",
        control_gripper,
        {
            "type": "object",
            "properties": {
                "claw": {"type": "string", "enum": ["left", "right"]},
                "action": {"type": "string", "enum": ["open", "close"]}
            },
            "required": ["claw", "action"]
        }
    )

    registry.register(
        "center_head",
        "Center the neck pan/tilt servos (Channels 0 and 7) back to default home angles.",
        center_head,
        {"type": "object", "properties": {}}
    )

    crud = CRUDTool(db, vision_pipeline=vision, ui_state=UI_STATE, tts=tts)
    # NOTE: db_read, update_identity, delete_identity are intentionally NOT
    # exposed to the LLM. The small model hallucinates user IDs and bypasses
    # the proper identity registration flow when given raw DB access.

    # 6. Init Brain (registry كاملة بالـ tools الأساسية)
    brain = BrainEngine(cfg, registry)

    # 7. Init STM بعد الـ brain (محتاجه للـ summarization)
    stm = ShortTermMemory(brain)

    # 8. Init Session بعد الـ stm
    session = SessionManager(stm, ltm)
    crud.session = session

    # 9. سجل الـ tools اللي محتاجة session أو stm
    # ── Two-step identity registration tools ──
    registry.register(
        "stage_user_profile",
        (
            "Stage facts about a user in the background as you talk to them. "
            "Does NOT commit yet. Call this repeatedly to aggregate data (name, age, hobbies). "
            "Once you feel you have gathered enough meaningful information, summarize it out loud and ask the user to confirm. "
            "Example string: '{\"name\": \"Abdullah\", \"age\": 28, \"sport\": \"football\"}'."
        ),
        crud.stage_user_profile,
        {
            "type": "object",
            "properties": {
                "profile_data_json": {
                    "type": "string",
                    "description": "A JSON string of facts gathered about the user. Always include 'name' if known.",
                }
            },
            "required": ["profile_data_json"],
        },
    )

    registry.register(
        "confirm_registration",
        (
            "Commit a pending profile after the user confirmed your summary. "
            "Requires staged name, age, captured voice, and visible face. "
            "Only call AFTER stage_user_profile AND the user said yes (English or Arabic)."
        ),
        crud.confirm_registration,
        {
            "type": "object", 
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Pass true to confirm the registration."
                }
            },
            "required": ["confirm"],
        },
    )

    registry.register(
        "cancel_registration",
        "Cancel a pending registration if the user said no or gave a different name.",
        crud.cancel_registration,
        {
            "type": "object", 
            "properties": {
                "cancel": {
                    "type": "boolean",
                    "description": "Pass true to cancel the registration."
                }
            },
            "required": ["cancel"],
        },
    )

    registry.register(
        "request_profile_change",
        (
            "Change a field on a known, locked user's profile. "
            "Only works for already-registered users who explicitly ask to change something."
        ),
        crud.request_profile_change,
        {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "The field to change: name, nickname, location, occupation, project, or interests.",
                },
                "new_value": {
                    "type": "string",
                    "description": "The new value for the field.",
                },
            },
            "required": ["field", "new_value"],
        },
    )

    registry.register(
        "register_new_person",
        "Register an unknown face in the scene as a new person. Use when a known user introduces a third person (e.g. 'This is my friend Omar').",
        crud.register_new_person,
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the new person to register.",
                }
            },
            "required": ["name"],
        },
    )
    # (الـ registry بـ reference فالـ brain هيشوفها أوتوماتيك)
    rag = RAGTool(ltm, session)
    registry.register(
        "search_memory",
        "Search the user's long-term memory for past conversations or facts.",
        rag.search_memory,
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    registry.register(
        "delete_memory",
        "Delete a specific memory by its ID.",
        rag.delete_memory,
        {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The exact ID of the memory to delete.",
                }
            },
            "required": ["memory_id"],
        },
    )

    registry.register(
        "clear_all_memories",
        "Delete ALL long-term memories for the current user.",
        rag.clear_all_memories,
        {"type": "object", "properties": {}},
    )

    stm_tool = STMTool(stm)
    registry.register(
        "get_conversation_history",
        "Get the recent conversation history.",
        stm_tool.get_conversation_history,
        {"type": "object", "properties": {}},
    )

    def search_web_binding(query: str) -> str:
        return search_web(
            query, conversation_context=stm.compact_snippet_for_tools(900)
        )

    registry.register(
        "search_web",
        (
            "Search the web for factual or up-to-date information. "
            "Use concise, keyword-focused queries — recent dialogue is attached automatically "
            "to improve relevance."
        ),
        search_web_binding,
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search keywords, not conversational filler.",
                }
            },
            "required": ["query"],
        },
    )

    # 10. Init STT بعد ما كل حاجة جاهزة
    stt = HearingEngine(event_queue, tts, speech_activity=speech_activity)

    # 11. Start Threads
    vision.start()
    stt.start()

    print(f"{Fore.GREEN}=== System Ready ==={Style.RESET_ALL}")

    # 12. Main Event Loop
    _last_response_time = 0.0  # Debounce multiple responses within 2 seconds
    try:
        while running:
            
            try:
                event = event_queue.get(timeout=0.5)
            except Exception:  # queue.Empty
                continue

            # User speech is priority 0 in EventQueue — always dequeue before vision and
            # stop TTS immediately so cognition never waits on lingering playback.
            if isinstance(event, SpeechEvent):
                # Strict Wake Word Gate
                user_text = (event.text or "").strip()

                push_ui_dialogue("User", event.text)
                tts.interrupt()
                scene = vision.get_latest_scene()

                if speech_activity.consume_overlap_interrupt():
                    overlap_msg = overlap_interrupt_message()
                    tts.say(overlap_msg)
                    push_ui_dialogue("Robot", overlap_msg)
                    stm.add_message(event.text, overlap_msg)
                    speech_activity.clear_utterance_speaker()
                    _last_response_time = time.time()
                    event_queue.task_done()
                    continue

                # Store voice embedding on session so tools can access it
                session.last_voice_embedding = event.voice_embedding

                # Voice Identification (no auto-registration for unknowns — handled by LLM confirmation flow)
                voice_user_info = None
                if event.voice_embedding is not None:
                    identified = db.identify_voice(event.voice_embedding)
                    if identified:
                        voice_user_info = identified
                        conf = identified.get("voice_confidence", 0)
                        print(
                            f"[VoiceAuth] ✓ Verified speaker: {identified.get('name')} "
                            f"(confidence={conf})"
                        )
                        UI_STATE["voice_events"].append({
                            "type": "verified",
                            "name": identified.get("name"),
                            "confidence": conf,
                            "time": time.time(),
                        })
                        UI_STATE["voice_events"] = UI_STATE["voice_events"][-10:]
                    elif (
                        scene.current_speaker_id
                        and scene.current_speaker_name != "Unknown"
                    ):
                        import threading
                        def _bg_register(uid, name, emb):
                            try:
                                db.register_voice(uid, emb)
                                print(f"[VoiceAuth] 🎙 Auto-registered voice for {name}")
                            except Exception as e:
                                print(f"[VoiceAuth] Background voice registration failed: {e}")
                        
                        threading.Thread(
                            target=_bg_register,
                            args=(scene.current_speaker_id, scene.current_speaker_name, event.voice_embedding),
                            daemon=True
                        ).start()

                        UI_STATE["voice_events"].append({
                            "type": "registered",
                            "name": scene.current_speaker_name,
                            "confidence": 0,
                            "time": time.time(),
                        })
                        UI_STATE["voice_events"] = UI_STATE["voice_events"][-10:]
                    else:
                        UI_STATE["voice_events"].append({
                            "type": "captured",
                            "name": UI_STATE.get("pending_display_name") or "Unknown",
                            "confidence": 0,
                            "time": time.time(),
                        })
                        UI_STATE["voice_events"] = UI_STATE["voice_events"][-10:]

                current_speaker_id, effective_user_info, speaker_face = (
                    _resolve_speaker_from_scene(scene, speech_activity, voice_user_info)
                )
                if effective_user_info is None and current_speaker_id:
                    effective_user_info = db.get_user(current_speaker_id)
                effective_user_info = (
                    effective_user_info or session.cached_user_info
                )

                # Background biometric capture for unknown speakers
                if speaker_face is None and speech_activity:
                    utter_tid = speech_activity.get_utterance_speaker_track_id()
                    speaker_face = _face_for_track(scene, utter_tid)
                if speaker_face and speaker_face.name == "Unknown":
                    session.update_biometric_capture(
                        face_embedding=speaker_face.embedding,
                        voice_embedding=event.voice_embedding,
                        track_id=speaker_face.track_id,
                    )
                elif event.voice_embedding is not None and voice_user_info is None:
                    session.update_biometric_capture(
                        voice_embedding=event.voice_embedding,
                    )

                is_unknown_speaker = (
                    voice_user_info is None
                    and (
                        speaker_face is None
                        or speaker_face.name == "Unknown"
                        or current_speaker_id is None
                    )
                )

                reg_turn = process_registration_turn(
                    event.text,
                    crud=crud,
                    session=session,
                    has_voice_embedding=event.voice_embedding is not None,
                )
                if reg_turn.display_name:
                    UI_STATE["pending_display_name"] = reg_turn.display_name
                    _sync_vision_greet_skip()
                    tid = (
                        speaker_face.track_id
                        if speaker_face
                        else speech_activity.get_utterance_speaker_track_id()
                    )
                    if tid is not None:
                        UI_STATE.setdefault("face_display_names", {})[
                            str(tid)
                        ] = reg_turn.display_name

                pr = session.get_pending_registration()
                if pr:
                    UI_STATE["pending_profile"] = sanitize_for_api(
                        pr.get("profile_data")
                    )
                elif reg_turn.pending_profile:
                    UI_STATE["pending_profile"] = sanitize_for_api(
                        reg_turn.pending_profile
                    )

                _sync_vision_greet_skip()

                if reg_turn.confirmed:
                    UI_STATE["pending_display_name"] = None
                    UI_STATE["face_display_names"] = {}
                    UI_STATE["pending_profile"] = None
                    nm = reg_turn.display_name or "friend"
                    UI_STATE["voice_events"].append({
                        "type": "verified",
                        "name": nm,
                        "confidence": 1.0,
                        "time": time.time(),
                    })
                    UI_STATE["voice_events"] = UI_STATE["voice_events"][-10:]
                    vision.refresh_face_cache()

                session.on_interaction(current_speaker_id, effective_user_info)
                speech_activity.clear_utterance_speaker()

                if reg_turn.skip_llm and reg_turn.speak_text:
                    reply = reg_turn.speak_text
                    tts.say(reply)
                    push_ui_dialogue("Robot", reply)
                    stm.add_message(event.text, reply)
                    if reg_turn.confirmed:
                        session.on_interaction(
                            session.current_user_id, session.cached_user_info
                        )
                    _last_response_time = time.time()
                    event_queue.task_done()
                    continue

                speech_prefix = registration_hint_for_llm(reg_turn) + "\n"
                if (
                    event.voice_embedding is not None
                    and voice_user_info is None
                    and len(scene.faces) == 0
                ):
                    speech_prefix += (
                        "[System: Voice heard but no face in camera. "
                        "Reply in ENGLISH ONLY. Ask them to look at the camera.]\n"
                    )
                else:
                    speech_prefix += "[System: Reply in ENGLISH ONLY. Never use Arabic script in your reply.]\n"

                user_text = speech_prefix + event.text
                registration_mode = is_unknown_speaker or bool(
                    session.get_pending_registration()
                )

                # Filler phrase callback
                def on_tool(tool_name: str):
                    if tool_name == "search_web":
                        tts.say("hold on, let me look that up real quick.")
                    elif tool_name == "search_memory":
                        tts.say("give me a second, let me check my memory.")
                    else:
                        tts.say("hold on a second.")

                ltm_prefetch = None
                if cfg.LTM_PREFETCH_ENABLED:
                    prefetch_uid = current_speaker_id or session.current_user_id
                    if prefetch_uid:
                        snippets = ltm.retrieve(
                            event.text,
                            user_id=prefetch_uid,
                            top_k=cfg.LTM_PREFETCH_TOP_K,
                            user_profile_hint=_ltm_user_profile_fragment(
                                effective_user_info
                            ),
                        )
                        ltm_prefetch = "\n".join(snippets) if snippets else ""

                # Think
                full_response = brain.think(
                    user_text,
                    scene,
                    stm.get_context(),
                    user_info=effective_user_info,
                    on_tool_call=on_tool,
                    ltm_prefetch=ltm_prefetch,
                    enable_tools=True,
                    registration_mode=registration_mode,
                )

                tts.say(full_response)
                push_ui_dialogue("Robot", full_response)

                stm.add_message(event.text, full_response)

                if brain.last_save_to_memory:
                    uid = current_speaker_id or session.current_user_id
                    ltm.store_memory(brain.last_summary, user_id=uid)
                
                _last_response_time = time.time()

            elif isinstance(event, VisionChangeEvent):
                now = time.time()
                is_first_greet = event.change_type in (
                    "unknown_appeared",
                    "multi_unknown_appeared",
                )
                if not is_first_greet and (now - _last_response_time) < 2.0:
                    event_queue.task_done()
                    continue

                if event.change_type == "person_left":
                    # details might contain user_id from vision_pipeline
                    left_uid = event.details.get("user_id")
                    session.on_speaker_left(left_uid)
                elif event.change_type in (
                    "unknown_idle",
                    "multi_unknown_idle",
                    "unknown_appeared",
                    "multi_unknown_appeared",
                ):
                    snap = getattr(event, "scene_snapshot", None)
                    scene_user_info = getattr(snap, "user_info", None) if snap else None
                    skip = False
                    # First greet always fires (even if user spoke first); idle uses voice gate.
                    if not is_first_greet and (
                        session.seconds_since_voice_interaction()
                        < cfg.UNKNOWN_IDLE_VOICE_GATE_SECONDS
                    ):
                        skip = True
                    if (
                        session.last_proactive_reply_time > 0
                        and (now - session.last_proactive_reply_time)
                        < cfg.PROACTIVE_VISION_DEBOUNCE_SECONDS
                    ):
                        skip = True
                    if not skip:
                        response_text = brain.react_to_vision(
                            event,
                            stm.get_context(),
                            user_info=(session.cached_user_info or scene_user_info),
                        )
                        if response_text:
                            tts.say(response_text)
                            push_ui_dialogue("Robot", response_text)
                            session.mark_proactive_reply()
                            stm.add_message(_vision_stm_user_line(event), response_text)
                            if brain.last_save_to_memory:
                                uid = (
                                    event.details.get("user_id")
                                    or session.current_user_id
                                )
                                if uid:
                                    ltm.store_memory(brain.last_summary, user_id=uid)
                elif event.change_type == "new_person":
                    if session.has_active_session:
                        # Someone new joined mid-conversation
                        new_name = event.details.get("name", "someone")
                        session.on_new_person(new_name, event.details.get("user_id"))
                        response_text = brain.react_to_new_person_during_session(
                            event,
                            session.cached_user_info,
                            stm.get_context(),
                        )
                    else:
                        response_text = brain.react_to_vision(
                            event,
                            stm.get_context(),
                            user_info=session.cached_user_info,
                        )

                    if response_text:
                        tts.say(response_text)
                        push_ui_dialogue("Robot", response_text)
                        session.mark_proactive_reply()
                        stm.add_message(_vision_stm_user_line(event), response_text)
                        if brain.last_save_to_memory:
                            uid = (
                                event.details.get("user_id") or session.current_user_id
                            )
                            if uid:
                                ltm.store_memory(brain.last_summary, user_id=uid)
                else:
                    response_text = brain.react_to_vision(
                        event,
                        stm.get_context(),
                        user_info=session.cached_user_info,
                    )
                    if response_text:
                        tts.say(response_text)
                        push_ui_dialogue("Robot", response_text)
                        session.mark_proactive_reply()
                        stm.add_message(_vision_stm_user_line(event), response_text)
                        if brain.last_save_to_memory:
                            uid = (
                                event.details.get("user_id") or session.current_user_id
                            )
                            if uid:
                                ltm.store_memory(brain.last_summary, user_id=uid)



            event_queue.task_done()

    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up...")
        session._end_session_locked()
        stt.stop()
        vision.stop()
        try:
            bot.stop()
            bot.center_all_servos()
            print("[Main] Robot safely disarmed.")
        except Exception as e:
            print(f"[Main] Robot teardown error: {e}")


if __name__ == "__main__":
    main()
