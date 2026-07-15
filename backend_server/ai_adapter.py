import sys
import os
import types
from typing import Any, Optional, Dict, List, Tuple
import time
import subprocess
import socket

# Insert Robot path to sys.path
robot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if robot_dir not in sys.path:
    sys.path.insert(0, robot_dir)

# Helper function to check if a port is open
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("127.0.0.1", port))
            return True
        except:
            return False

# Try to start Qdrant if it's not running
def ensure_qdrant_running():
    if not is_port_open(6333):
        qdrant_path = os.path.join(robot_dir, "qdrant.exe")
        if os.path.exists(qdrant_path):
            print(f"[AIAdapter] Starting Qdrant database in background: {qdrant_path}")
            try:
                subprocess.Popen(
                    [qdrant_path],
                    cwd=robot_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Wait up to 10 seconds for it to start
                for _ in range(25):
                    if is_port_open(6333):
                        print("[AIAdapter] Qdrant started successfully on port 6333.")
                        return True
                    time.sleep(0.4)
            except Exception as e:
                print(f"[AIAdapter] Failed to start Qdrant: {e}")
        else:
            print(f"[AIAdapter] Qdrant not found at {qdrant_path} and port 6333 is closed.")
    else:
        print("[AIAdapter] Qdrant is already running on port 6333.")
        return True
    return False

# Initialize Qdrant database
ensure_qdrant_running()

# Mock heavy machine learning dependencies for API-only backend
from unittest.mock import MagicMock

class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts, *args, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            return np.zeros(384)
        return [np.zeros(384) for _ in texts]

class MockSentenceTransformersModule:
    SentenceTransformer = MockSentenceTransformer

sys.modules['sentence_transformers'] = MockSentenceTransformersModule

# Mock other unused models with Python 3.12 compatible specs
for m in [
    'easyocr', 'faster_whisper', 
    'speechbrain', 'sounddevice', 'soundfile', 'pyaudio', 
    'torchaudio'
]:
    mock_mod = MagicMock()
    mock_mod.__spec__ = types.SimpleNamespace(origin='mock', loader=None, submodule_search_locations=None)
    sys.modules[m] = mock_mod

# Now import dependencies from Robot
try:
    import config.settings as cfg
    from database.memory_manager import MemoryManager
    from brain.llm_engine import BrainEngine
    from brain.memory.stm import ShortTermMemory
    from brain.memory.ltm import LongTermMemory
    from brain.tools.tool_registry import ToolRegistry
    from brain.tools.ocr_tool import OCRTool
    from brain.tools.web_search_tool import search_web
    from brain.tools.rag_tool import RAGTool
    from brain.tools.crud_tool import CRUDTool
    from brain.tools.stm_tool import STMTool
    from brain.session_manager import SessionManager
except ImportError as e:
    print(f"[AIAdapter] CRITICAL: Failed to import Robot dependencies: {e}")
    print("[AIAdapter] Please ensure the Robot folder path is correct and dependencies in requirements.txt are installed.")
    raise e

class AIAdapter:
    def __init__(self, robot: Optional[Any] = None):
        print("[AIAdapter] Initializing real AI system components...")
        self.robot = robot
        
        # 1. Init Database and Memory Managers
        self.db = MemoryManager()
        self.ltm = LongTermMemory(self.db.qdrant)
        
        # 2. Initialize Face & Object Recognizers (Real AI modules)
        from vision.face_recognition import FaceRecognizer
        from vision.object_recognition import ObjectRecognizer
        self.face_recognizer = FaceRecognizer()
        self.face_recognizer.load_local_cache(self.db.qdrant, self.db.mongo)
        self.object_recognizer = ObjectRecognizer()
        
        # 3. Tools & Registry
        self.registry = ToolRegistry()
        self.ocr = OCRTool()
        
        # Register tools
        self.registry.register(
            "perform_ocr",
            "Extract text from the current camera view.",
            self.ocr.perform_ocr,
            {"type": "object", "properties": {}},
        )
        
        if self.robot is not None:
            def move_robot(direction: str, duration: float) -> str:
                try:
                    self.robot.move_for(direction, duration, block=True)
                    return f"[Hardware] Moved {direction} for {duration} seconds."
                except Exception as e:
                    return f"[Hardware] Error moving: {e}"

            def set_robot_pose(pose_name: str) -> str:
                try:
                    if pose_name == "wave":
                        import threading
                        threading.Thread(target=self.robot.wave_arm, daemon=True).start()
                        return "[Hardware] Waving arm..."
                    else:
                        self.robot.set_pose(pose_name)
                        return f"[Hardware] Robot pose set to {pose_name}."
                except Exception as e:
                    return f"[Hardware] Error setting pose: {e}"

            def control_gripper(claw: str, action: str) -> str:
                try:
                    if claw == "right":
                        if action == "open":
                            self.robot.open_gripper_r()
                        else:
                            self.robot.close_gripper_r()
                    else:
                        if action == "open":
                            self.robot.open_gripper_l()
                        else:
                            self.robot.close_gripper_l()
                    return f"[Hardware] {claw.capitalize()} gripper {action}ed."
                except Exception as e:
                    return f"[Hardware] Error controlling gripper: {e}"

            def center_head() -> str:
                try:
                    self.robot.set_servos({0: 90, 7: 80})
                    return "[Hardware] Head centered."
                except Exception as e:
                    return f"[Hardware] Error centering head: {e}"

            self.registry.register(
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

            self.registry.register(
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

            self.registry.register(
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

            self.registry.register(
                "center_head",
                "Center the neck pan/tilt servos (Channels 0 and 7) back to default home angles.",
                center_head,
                {"type": "object", "properties": {}}
            )
        
        self.crud = CRUDTool(self.db, vision_pipeline=None, ui_state=None, tts=None)
        
        self.registry.register(
            "stage_user_profile",
            "Stage facts about a user in the background as you talk to them. Does NOT commit yet.",
            self.crud.stage_user_profile,
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
        
        self.registry.register(
            "confirm_registration",
            "Commit a pending profile after the user confirmed your summary.",
            self.crud.confirm_registration,
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
        
        self.registry.register(
            "cancel_registration",
            "Cancel a pending registration if the user said no.",
            self.crud.cancel_registration,
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

        self.rag = RAGTool(self.ltm, None)
        self.registry.register(
            "search_memory",
            "Search the user's long-term memory for past conversations or facts.",
            self.rag.search_memory,
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        
        # 3. Cognitive Brain Engine
        self.brain = BrainEngine(cfg, self.registry)
        
        # 4. Short Term Memory & Session
        self.stm = ShortTermMemory(self.brain)
        self.session = SessionManager(self.stm, self.ltm)
        self.crud.session = self.session
        self.rag.session = self.session
        
        # Register stm tool
        self.stm_tool = STMTool(self.stm)
        self.registry.register(
            "get_conversation_history",
            "Get the recent conversation history.",
            self.stm_tool.get_conversation_history,
            {"type": "object", "properties": {}},
        )
        
        # Register web search
        def search_web_binding(query: str) -> str:
            return search_web(
                query, conversation_context=self.stm.compact_snippet_for_tools(900)
            )
        self.registry.register(
            "search_web",
            "Search the web for factual or up-to-date information.",
            search_web_binding,
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short search keywords.",
                    }
                },
                "required": ["query"],
            },
        )
        
        print("[AIAdapter] AI system components initialized successfully.")

    def process_query(self, text: str, scene: Any = None) -> dict:
        """
        Processes a user text query through the real LLM BrainEngine.
        Returns a dict conforming to the WebSocket assistant.response payload.
        """
        try:
            # Refresh session activity
            user_id = self.session.current_user_id
            user_info = self.session.cached_user_info
            
            # Update session interaction state
            self.session.on_interaction(user_id, user_info)

            # If voice embedding is missing (which is always the case in the backend server),
            # mock it so that registration can succeed without needing local SpeechBrain!
            bio = self.session.get_biometric_capture()
            if not bio.get("voice_embedding"):
                import numpy as np
                self.session.update_biometric_capture(
                    voice_embedding=np.zeros(192).tolist()
                )
            
            # Run deterministic registration turn
            from brain.registration_controller import process_registration_turn, registration_hint_for_llm
            reg_turn = process_registration_turn(
                text,
                crud=self.crud,
                session=self.session,
                has_voice_embedding=True,
            )

            # If registration controller handles it deterministically and skips LLM
            if reg_turn.skip_llm and reg_turn.speak_text:
                if reg_turn.confirmed:
                    try:
                        self.face_recognizer.refresh_cache(self.db.qdrant, self.db.mongo)
                        print(f"[AIAdapter] Refreshed face recognizer cache after registration confirmation.")
                    except Exception as e:
                        print(f"[AIAdapter] Failed to refresh face cache: {e}")
                
                reply = reg_turn.speak_text
                self.stm.add_message(text, reply)
                avatar_state, motion_command = self._map_avatar_state(reply)
                return {
                    "text": reply,
                    "avatar_state": avatar_state,
                    "motion_command": motion_command
                }

            # Inject registration prefix
            speech_prefix = registration_hint_for_llm(reg_turn) + "\n"
            processed_text = speech_prefix + text
            
            # Fetch LTM prefetch context if user is identified
            ltm_prefetch = ""
            prefetch_uid = self.session.session_owner_id or self.session.current_user_id or user_id
            if prefetch_uid and cfg.LTM_PREFETCH_ENABLED:
                try:
                    memories = self.ltm.retrieve(text, user_id=prefetch_uid, top_k=cfg.LTM_PREFETCH_TOP_K)
                    ltm_prefetch = "\n".join(memories)
                except Exception as e:
                    print(f"[AIAdapter] LTM prefetch error: {e}")
            
            # Query the brain to think
            reply = self.brain.think(
                user_text=processed_text,
                vision_state=scene,
                stm_context=self.stm.get_context(),
                user_info=user_info,
                ltm_prefetch=ltm_prefetch,
            )
            
            # Add message to STM
            self.stm.add_message(text, reply)

            # Save to memory if LLM requested it!
            if getattr(self.brain, "last_save_to_memory", False):
                uid = self.session.session_owner_id or self.session.current_user_id or user_id
                if uid:
                    self.ltm.store_memory(self.brain.last_summary, user_id=uid)
                    print(f"[AIAdapter] Saved proactive memory for user '{uid}': {self.brain.last_summary}")
            
            # Map LLM response to Avatar State and Motion Command
            avatar_state, motion_command = self._map_avatar_state(reply)
            
            return {
                "text": reply,
                "avatar_state": avatar_state,
                "motion_command": motion_command
            }
        except Exception as e:
            print(f"[AIAdapter] Error processing query: {e}")
            return {
                "text": f"Sorry, I encountered an error processing that: {str(e)}",
                "avatar_state": "error",
                "motion_command": {"face": "warning", "eyes": "blink", "head": "tilt_left"}
            }

    def _map_avatar_state(self, text: str) -> tuple[str, dict]:
        """
        Basic rule-based mapping of the assistant's response to expression states and motions.
        """
        text_lower = text.lower()
        
        # Default is speaking/neutral
        avatar_state = "speaking"
        motion_command = {"face": "talk", "eyes": "calm", "head": "none"}
        
        # Check sentiment/keywords for expressions
        if any(w in text_lower for w in ["happy", "great", "awesome", "glad", "wonderful", "smile", "nice to meet", "hello", "hey"]):
            avatar_state = "happy"
            motion_command = {"face": "smile", "eyes": "soft", "head": "small_nod"}
        elif any(w in text_lower for w in ["sorry", "error", "fail", "wrong", "unfortunately", "sad"]):
            avatar_state = "error"
            motion_command = {"face": "warning", "eyes": "blink", "head": "tilt_left"}
        elif any(w in text_lower for w in ["question", "confused", "what?", "how?", "why?", "not sure", "depends"]):
            avatar_state = "confused"
            motion_command = {"face": "confused", "eyes": "focused", "head": "tilt_right"}
            
        return avatar_state, motion_command

if __name__ == "__main__":
    # Self-test
    print("[AIAdapter] Performing self-test initialization...")
    try:
        adapter = AIAdapter()
        print("[AIAdapter] Self-test passed!")
    except Exception as e:
        print(f"[AIAdapter] Self-test failed: {e}")
        sys.exit(1)
