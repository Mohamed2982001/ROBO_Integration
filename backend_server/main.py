import asyncio
import json
import random
import time
import io
import wave
import os
import base64
import numpy as np
import cv2
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load environment variables from parent folder .env
robot_env = Path(__file__).parent.parent / ".env"
if robot_env.exists():
    load_dotenv(robot_env)
else:
    # Try local .env inside backend_server just in case
    load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from typing import Any, List

# Global list to track detected faces and objects from the camera feed
active_faces: List[Any] = []
active_objects: List[Any] = []
is_processing_frame = False

def detect_chest_color(frame_bgr, face_box) -> str:
    """Crops the region below the face bounding box and determines the dominant color."""
    try:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = face_box
        
        # Calculate chest box (below face box)
        face_w = x2 - x1
        face_h = y2 - y1
        
        cx1 = max(0, x1 - int(face_w * 0.2))
        cx2 = min(w, x2 + int(face_w * 0.2))
        cy1 = min(h, y2 + int(face_h * 0.1))
        cy2 = min(h, y2 + int(face_h * 1.5))
        
        if cy2 <= cy1 or cx2 <= cx1:
            return "unknown"
            
        chest_crop = frame_bgr[cy1:cy2, cx1:cx2]
        if chest_crop.size == 0:
            return "unknown"
            
        # Convert to HSV for better color categorization
        hsv = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2HSV)
        avg_hsv = cv2.mean(hsv)
        hue = avg_hsv[0]
        sat = avg_hsv[1]
        val = avg_hsv[2]
        
        if sat < 25: # Low saturation -> greyscale
            if val < 50:
                return "black"
            elif val > 200:
                return "white"
            else:
                return "grey"
        
        # Hue ranges (standard HSV in OpenCV: H is 0-180)
        if hue < 10 or hue > 170:
            return "red"
        elif hue < 25:
            return "orange"
        elif hue < 35:
            return "yellow"
        elif hue < 85:
            return "green"
        elif hue < 130:
            return "blue"
        else:
            return "purple"
            
    except Exception as e:
        print(f"[Vision] Error detecting chest color: {e}")
        return "unknown"

def build_scene_snapshot() -> Any:
    """Assembles a real VisionState object based on active_faces."""
    if ai_adapter is None:
        return None
    
    from vision.vision_pipeline import VisionState
    scene = VisionState()
    scene.timestamp = time.time()
    
    global active_faces, active_objects
    scene.faces = active_faces
    scene.objects = list(active_objects)
    
    if active_faces:
        # Find if there is an identified user (confidence threshold >= 0.75)
        known_faces = [f for f in active_faces if f.name != "Unknown" and f.confidence >= 0.75]
        if known_faces:
            scene.current_speaker_name = known_faces[0].name
            scene.current_speaker_id = known_faces[0].user_id
            scene.current_speaker_track_id = known_faces[0].track_id
            # Log interaction in session
            user_info = ai_adapter.db.mongo.get_user(known_faces[0].user_id)
            ai_adapter.session.on_interaction(known_faces[0].user_id, user_info)
        else:
            scene.current_speaker_name = "Unknown"
            scene.current_speaker_id = None
            scene.current_speaker_track_id = active_faces[0].track_id

        # Generate ObjectInfo for clothing colors
        from vision.object_recognition import ObjectInfo
        for f in active_faces:
            shirt_color = getattr(f, "shirt_color", "unknown")
            if shirt_color and shirt_color != "unknown":
                obj = ObjectInfo(
                    track_id=f.track_id + 1000,
                    class_name=f"{shirt_color} shirt",
                    confidence=0.9,
                    box=(f.box[0], f.box[3], f.box[2], f.box[3] + 100)
                )
                scene.objects.append(obj)
            
    return scene

async def run_face_recognition(payload_b64: str):
    """Decodes base64 frame and runs real FaceRecognizer & ObjectRecognizer in background."""
    global active_faces, active_objects, is_processing_frame
    if ai_adapter is None:
        return
    if is_processing_frame:
        # Drop frame to prevent thread pool exhaustion and CPU starvation
        return
        
    is_processing_frame = True
    try:
        loop = asyncio.get_event_loop()
        def decode_and_run():
            img_bytes = base64.b64decode(payload_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return [], []
                
            # Resize frame to match tracker resolution for consistency
            if face_tracker is not None:
                frame = cv2.resize(frame, (face_tracker.cfg.frame_width, face_tracker.cfg.frame_height))
                
            # 1. Closed-loop head tracking from camera feed (uses MediaPipe FaceTracker)
            detection = None
            if face_tracker is not None:
                try:
                    detection = face_tracker._detect_face_center(frame)
                    if detection is not None:
                        target_x, target_y, (x_min, y_min, width, height) = detection
                        center_x = face_tracker.cfg.frame_width // 2
                        center_y = face_tracker.cfg.frame_height // 2
                        error_x = target_x - center_x
                        error_y = target_y - center_y
                        
                        # Update and send servo commands using the user's exact logic
                        face_tracker._update_target_angles(error_x, error_y)
                        
                        # Run the actual network call in a daemon thread so it doesn't block
                        import threading
                        threading.Thread(target=face_tracker._maybe_send_servo_update, daemon=True).start()
                except Exception as e:
                    print(f"[Main] MediaPipe face tracking error: {e}")
                    
            # 2. Run YOLO face detection
            detected = ai_adapter.face_recognizer.detect(frame)
            faces_out = []
            
            for f in detected:
                box = f["box"]
                tid = f["track_id"]
                emb = ai_adapter.face_recognizer.get_embedding(frame, box)
                if emb is not None:
                    # Identify using local cache / Qdrant
                    name, sim, uid = ai_adapter.face_recognizer.identify(
                        emb, tid, ai_adapter.db.qdrant, ai_adapter.db.mongo
                    )
                    
                    # Detect shirt color
                    shirt_color = detect_chest_color(frame, box)
                    
                    # Update session biometric capture with this embedding
                    ai_adapter.session.update_biometric_capture(face_embedding=emb)
                    
                    from vision.face_recognition import FaceInfo
                    finfo = FaceInfo(
                        track_id=tid,
                        name=name,
                        confidence=sim,
                        user_id=uid,
                        box=box,
                        embedding=emb
                    )
                    # Dynamically set custom attribute for shirt color
                    finfo.shirt_color = shirt_color
                    faces_out.append(finfo)
            
            # 3. Run YOLO object detection
            objects_out = []
            if hasattr(ai_adapter, "object_recognizer") and ai_adapter.object_recognizer is not None:
                try:
                    objects_out = ai_adapter.object_recognizer.detect(frame)
                except Exception as e:
                    print(f"[Vision] Object detection error: {e}")

            # 4. Render who it sees on the PC monitor screen
            try:
                # 4.1 Draw MediaPipe tracking visual aids
                if face_tracker is not None:
                    cx = face_tracker.cfg.frame_width // 2
                    cy = face_tracker.cfg.frame_height // 2
                    # Draw deadzone radius circle
                    cv2.circle(frame, (cx, cy), face_tracker.cfg.deadzone_radius, (255, 255, 255), 1)
                    if detection is not None:
                        tx, ty, _ = detection
                        # Draw green target dot and tracking line
                        cv2.circle(frame, (tx, ty), 5, (0, 255, 0), -1)
                        cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 0), 1)

                # 4.2 Draw YOLO identification bounding boxes & names
                for f in faces_out:
                    x1, y1, x2, y2 = f.box
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw name and confidence label
                    label = f"{f.name} ({int(f.confidence * 100)}%)"
                    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                
                # Show the live annotated camera feed on the PC screen!
                cv2.imshow("MUSA Live Robot Vision (PC Screen)", frame)
                cv2.waitKey(1)
            except Exception as e:
                # Silently catch GUI errors if running headlessly or window was closed
                pass
                    
            return faces_out, objects_out

        faces, objects = await loop.run_in_executor(None, decode_and_run)
        active_faces = faces
        active_objects = objects
        if faces or objects:
            print(f"[Vision] Detected {len(faces)} face(s): {[f.name for f in faces]} | {len(objects)} object(s): {[o.class_name for o in objects]}")
            
    except Exception as e:
        print(f"[Vision] Vision pipeline error: {e}")
    finally:
        is_processing_frame = False

# Initialize FastAPI app
app = FastAPI(title="MUSA Companion Backend Server", version="0.3.0")

# Enable CORS so any local IP client can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load env file to get API keys list
groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
current_groq_key_index = 0

def get_groq_client() -> Groq:
    global current_groq_key_index, groq_keys
    if not groq_keys:
        raise ValueError("No Groq API keys configured.")
    return Groq(api_key=groq_keys[current_groq_key_index])

def rotate_groq_key():
    global current_groq_key_index, groq_keys
    if len(groq_keys) > 1:
        current_groq_key_index = (current_groq_key_index + 1) % len(groq_keys)
        masked_key = f"{groq_keys[current_groq_key_index][:8]}...{groq_keys[current_groq_key_index][-4:]}" if len(groq_keys[current_groq_key_index]) > 12 else "..."
        print(f"[Main] Rotating Whisper Groq API key to index {current_groq_key_index} ({masked_key})")

# Print initial Groq client status
if groq_keys:
    masked_key = f"{groq_keys[0][:8]}...{groq_keys[0][-4:]}" if len(groq_keys[0]) > 12 else "..."
    print(f"[Main] Groq client list loaded with {len(groq_keys)} key(s), starting index 0 ({masked_key})")
else:
    print(f"[Main] WARNING: No Groq API keys loaded.")

# Try loading the real AI Adapter and connect to robot
ai_adapter = None
bot = None

# Initialize Robot Client
try:
    from robot_client import Robot
    robot_ip = os.getenv("ROBOT_IP", "192.168.1.18").strip()
    print(f"[Main] Connecting to robot at {robot_ip}...")
    bot = Robot(host=robot_ip, auto_fetch_limits=True)
    if bot.ping():
        print(f"[Main] Connected to robot at {robot_ip} [OK]")
        try:
            bot.set_speed(160)
            bot.center_all_servos()
        except Exception as e:
            print(f"[Main] Error initializing robot settings: {e}")
    else:
        print(f"[Main] WARNING: Robot unreachable at {robot_ip}. Running in mock/unconnected mode.")
        bot = None
except Exception as e:
    print(f"[Main] WARNING: Could not initialize robot client: {e}")

try:
    from ai_adapter import AIAdapter
    ai_adapter = AIAdapter(robot=bot)
    print("[Main] Real MUSA AI engine adapter loaded successfully!")
except Exception as e:
    print(f"[Main] WARNING: Could not load real AI adapter ({e}). Server will run in Mock Mode.")

# Initialize FaceTracker with user-defined config (uses MediaPipe)
face_tracker_config = None
face_tracker = None
try:
    from face_tracker import FaceTracker, FaceTrackerConfig
    face_tracker_config = FaceTrackerConfig(
        frame_width=320,  # match phone streaming resolution
        frame_height=240,
        deadzone_radius=20,
    )
    # We pass None for camera_url since we stream base64 frames
    face_tracker = FaceTracker(bot, camera_url=0, config=face_tracker_config, show_preview=False)
    print("[Main] MediaPipe FaceTracker initialized successfully [OK]")
except Exception as e:
    print(f"[Main] WARNING: Could not load MediaPipe FaceTracker ({e})")

def trigger_hardware_gesture_for_response(text: str, motion_command: dict = None):
    if bot is None:
        return
        
    try:
        import threading
        def _run():
            # Check motion command first
            if motion_command:
                head_cmd = motion_command.get("head", "none")
                face_cmd = motion_command.get("face", "none")
                if head_cmd == "small_nod":
                    bot.set_servos({7: 90})
                    time.sleep(0.4)
                    bot.set_servos({7: 80})
                    return
                elif head_cmd == "tilt_left":
                    bot.set_servos({0: 75, 7: 85})
                    time.sleep(1.0)
                    bot.set_servos({0: 90, 7: 80})
                    return
                elif head_cmd == "tilt_right":
                    bot.set_servos({0: 105, 7: 85})
                    time.sleep(1.0)
                    bot.set_servos({0: 90, 7: 80})
                    return
                elif face_cmd == "smile" or head_cmd == "wave":
                    bot.wave_arm()
                    return

            # Fallback to text matching
            t_low = text.lower()
            if any(w in t_low for w in ["hi", "hello", "hey", "welcome", "أهلاً", "مرحباً", "مرحبا", "سلام"]):
                bot.wave_arm()
            elif any(w in t_low for w in ["why", "how", "what", "check", "لماذا", "كيف", "ماذا", "هل"]):
                bot.set_servos({7: 95})
                time.sleep(1.2)
                bot.set_servos({7: 80})
            else:
                bot.set_servos({7: 90})
                time.sleep(0.4)
                bot.set_servos({7: 80})
                
        threading.Thread(target=_run, daemon=True).start()
    except Exception as e:
        print(f"[Main] Error triggering hardware gesture: {e}")

from gtts import gTTS
import numpy as np

# Load Kokoro for server-side high-quality audio generation
kokoro_pipeline = None
try:
    from kokoro import KPipeline
    from core.device import pick_torch_device
    device = pick_torch_device()
    print(f"[Main] Loading Kokoro on {device} for server-side audio...")
    kokoro_pipeline = KPipeline(lang_code="a", device=device)
    print("[Main] Kokoro loaded successfully for server-side audio [OK]")
except Exception as e:
    print(f"[Main] WARNING: Could not load Kokoro for server-side audio: {e}")

def text_to_speech_mp3_base64(text: str) -> str:
    """Generates WAV speech audio bytes using Kokoro (or gTTS fallback) and returns them as a base64 string."""
    try:
        # 1. Try Kokoro first for natural high-quality voice
        if kokoro_pipeline is not None:
            # Generate audio chunks
            gen = kokoro_pipeline(
                text,
                voice="af_heart",
                speed=1.0,
                split_pattern=r"\n+",
            )
            audio_chunks = []
            for _, _, audio in gen:
                audio_chunks.append(audio)
            
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                # Convert float32 [-1, 1] to 16-bit PCM
                pcm_bytes = (full_audio * 32767).astype(np.int16).tobytes()
                # Wrap in standard WAV header at 24000 Hz
                wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=24000, channels=1, sample_width=2)
                return base64.b64encode(wav_bytes).decode("utf-8")
        
        # 2. Fallback to gTTS if Kokoro is unavailable
        lang = "ar" if any(0x0600 <= ord(c) <= 0x06FF for c in text) else "en"
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        return base64.b64encode(mp3_buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[TTS] Error generating speech with Kokoro: {e}")
        # Secondary fallback to gTTS
        try:
            lang = "ar" if any(0x0600 <= ord(c) <= 0x06FF for c in text) else "en"
            tts = gTTS(text=text, lang=lang, slow=False)
            mp3_buf = io.BytesIO()
            tts.write_to_fp(mp3_buf)
            return base64.b64encode(mp3_buf.getvalue()).decode("utf-8")
        except:
            return ""

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wraps raw PCM audio bytes in a standard WAV header."""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_buf.getvalue()

async def process_accumulated_audio(websocket: WebSocket, pcm_bytes: bytes, camera_active: bool = False):
    """Transcribes audio using Groq Whisper and queries Musa's brain with the result."""
    if not groq_keys:
        print("[Audio] Error: Groq client not initialized.")
        await websocket.send_json({
            "type": "error",
            "message": "Groq API key is missing. Cannot transcribe voice."
        })
        return

    try:
        # A. Switch MUSA to thinking state
        await websocket.send_json({
            "type": "robot.status",
            "state": "thinking",
            "message": "Transcribing voice...",
            "ai_ready": ai_adapter is not None,
            "media_source": "android_phone"
        })

        # B. Convert PCM to WAV
        wav_bytes = pcm_to_wav(pcm_bytes)
        
        # C. Call Groq Whisper API (run in thread pool executor with key rotation)
        loop = asyncio.get_event_loop()
        def call_whisper():
            attempts = len(groq_keys)
            for i in range(attempts):
                try:
                    client = get_groq_client()
                    transcription = client.audio.transcriptions.create(
                        file=("speech.wav", wav_bytes),
                        model="whisper-large-v3-turbo",
                    )
                    return transcription.text
                except Exception as exc:
                    if len(groq_keys) > 1 and i < attempts - 1:
                        print(f"[Main] Whisper transcription failed on key index {current_groq_key_index}. Rotating...")
                        rotate_groq_key()
                    else:
                        raise exc
            raise RuntimeError("All Groq API keys failed for Whisper STT.")

        user_text = await loop.run_in_executor(None, call_whisper)
        print(f"[Audio] Transcribed text: {user_text}")
        
        if not user_text.strip():
            # Reset back to idle if transcription was empty
            await websocket.send_json({
                "type": "robot.status",
                "state": "idle",
                "message": "Ready (No speech detected)",
                "ai_ready": ai_adapter is not None,
                "media_source": "android_phone"
            })
            return

        # D. Send transcribed text to app as user message
        await websocket.send_json({
            "type": "transcript.final",
            "speaker": "user",
            "text": user_text
        })

        # E. Process query via Musa's AI adapter
        if ai_adapter is not None:
            scene = build_scene_snapshot()
            def run_process_query():
                return ai_adapter.process_query(user_text, scene=scene)
            response = await loop.run_in_executor(None, run_process_query)
            
            # F. Generate TTS speech (run in thread pool executor)
            def generate_tts():
                return text_to_speech_mp3_base64(response["text"])
            voice_payload_b64 = await loop.run_in_executor(None, generate_tts)
            
            # G. Send assistant response with voice payload!
            trigger_hardware_gesture_for_response(response["text"], response["motion_command"])
            await websocket.send_json({
                "type": "assistant.response",
                "text": response["text"],
                "avatar_state": response["avatar_state"],
                "motion_command": response["motion_command"],
                "voice_payload_base64": voice_payload_b64
            })
        else:
            # Fallback mock reply
            await asyncio.sleep(1.0)
            mock_text = f"I transcribed: '{user_text}', but my AI brain is in mock mode."
            def generate_mock_tts():
                return text_to_speech_mp3_base64(mock_text)
            voice_payload_b64 = await loop.run_in_executor(None, generate_mock_tts)
            
            trigger_hardware_gesture_for_response(mock_text, {"face": "talk", "eyes": "calm", "head": "none"})
            await websocket.send_json({
                "type": "assistant.response",
                "text": mock_text,
                "avatar_state": "speaking",
                "motion_command": {"face": "talk", "eyes": "calm", "head": "none"},
                "voice_payload_base64": voice_payload_b64
            })

        # G. Revert back to idle after response complete
        await asyncio.sleep(2.0)
        await websocket.send_json({
            "type": "robot.status",
            "state": "idle",
            "message": "Awaiting next input",
            "ai_ready": ai_adapter is not None,
            "media_source": "android_phone"
        })

    except Exception as e:
        print(f"[Audio] Error processing voice: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to transcribe audio: {str(e)}"
        })

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "musa-backend",
        "version": "0.3.0",
        "ai_ready": ai_adapter is not None
    }

@app.websocket("/ws/robot")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected")
    
    # Initialize connection-specific audio accumulator and camera flag
    audio_buffer = bytearray()
    audio_active = False
    camera_active = False
    
    # 1. Send initial status
    await websocket.send_json({
        "type": "robot.status",
        "state": "idle",
        "message": "Ready",
        "ai_ready": ai_adapter is not None,
        "media_source": "android_phone"
    })

    try:
        while True:
            # Wait for messages from the Flutter client
            data_str = await websocket.receive_text()
            event = json.loads(data_str)
            event_type = event.get("type")

            if event_type == "session.start":
                client_id = event.get("client_id")
                platform = event.get("platform")
                print(f"[Session] Started for client {client_id} on {platform}")
                await websocket.send_json({
                    "type": "robot.status",
                    "state": "idle",
                    "message": f"Session active for client: {client_id}",
                    "ai_ready": ai_adapter is not None,
                    "media_source": "android_phone"
                })

            elif event_type == "text.input":
                user_text = event.get("text", "")
                print(f"[Text] User input: {user_text}")

                # B. Switch MUSA to thinking state
                await websocket.send_json({
                    "type": "robot.status",
                    "state": "thinking",
                    "message": "Processing response",
                    "ai_ready": ai_adapter is not None,
                    "media_source": "android_phone"
                })

                # C. Generate Response (Real AI or Mock Fallback)
                if ai_adapter is not None:
                    # Run the CPU-bound/blocking AI think function in a thread executor
                    loop = asyncio.get_event_loop()
                    scene = build_scene_snapshot()
                    def run_process_query():
                        return ai_adapter.process_query(user_text, scene=scene)
                    response = await loop.run_in_executor(None, run_process_query)
                    
                    # D. Send real assistant response
                    trigger_hardware_gesture_for_response(response["text"], response["motion_command"])
                    await websocket.send_json({
                        "type": "assistant.response",
                        "text": response["text"],
                        "avatar_state": response["avatar_state"],
                        "motion_command": response["motion_command"]
                    })
                else:
                    # Simulate processing delay
                    await asyncio.sleep(1.5)

                    # Mock reply fallback
                    replies = [
                        {
                            "text": f"I hear you loud and clear! You said: '{user_text}'. Everything looks green on my end.",
                            "state": "happy",
                            "motion": {"face": "smile", "eyes": "soft", "head": "small_nod"}
                        },
                        {
                            "text": "Interesting query. Let me scanning my memory database for any records...",
                            "state": "registering",
                            "motion": {"face": "scan", "eyes": "scan", "head": "none"}
                        },
                        {
                            "text": "Hmm, I am slightly unsure about that. Could you clarify your question?",
                            "state": "confused",
                            "motion": {"face": "confused", "eyes": "wide", "head": "tilt_left"}
                        },
                        {
                            "text": "Hello! I am Musa. The PC and your phone are communicating perfectly over WebSockets!",
                            "state": "speaking",
                            "motion": {"face": "talk", "eyes": "calm", "head": "small_nod"}
                        }
                    ]
                    reply = random.choice(replies)

                    # D. Send mock assistant response
                    trigger_hardware_gesture_for_response(reply["text"], reply["motion"])
                    await websocket.send_json({
                        "type": "assistant.response",
                        "text": reply["text"],
                        "avatar_state": reply["state"],
                        "motion_command": reply["motion"]
                    })

                # E. Revert back to idle after response complete
                await asyncio.sleep(2.0)
                await websocket.send_json({
                    "type": "robot.status",
                    "state": "idle",
                    "message": "Awaiting next input",
                    "ai_ready": ai_adapter is not None,
                    "media_source": "android_phone"
                })

            elif event_type == "media.source.set":
                source = event.get("source")
                audio = event.get("audio", False)
                video = event.get("video", False)
                camera_active = video
                print(f"[Media] Source set to {source} (Audio={audio}, Video={video})")
                
                # Clear active faces if camera is disabled
                if not video:
                    global active_faces
                    active_faces = []
                    print("[Vision] Camera disabled. Cleared active faces.")
                
                # Check for audio streaming transitions
                if audio_active and not audio:
                    audio_active = False
                    print(f"[Audio] Stopped recording. Accumulated {len(audio_buffer)} bytes. Transcribing...")
                    if len(audio_buffer) > 0:
                        asyncio.create_task(process_accumulated_audio(websocket, bytes(audio_buffer), camera_active))
                    audio_buffer.clear()
                elif not audio_active and audio:
                    print("[Audio] Started recording voice stream...")
                    audio_active = True
                    audio_buffer.clear()

            elif event_type == "audio.chunk":
                payload_b64 = event.get("payload_base64", "")
                if audio_active and payload_b64:
                    try:
                        chunk_bytes = base64.b64decode(payload_b64)
                        audio_buffer.extend(chunk_bytes)
                    except Exception as e:
                        print(f"[Audio] Error decoding chunk: {e}")

            elif event_type == "video.frame":
                payload_b64 = event.get("payload_base64", "")
                if payload_b64 and ai_adapter is not None:
                    asyncio.create_task(run_face_recognition(payload_b64))

            elif event_type == "hardware.command":
                action = event.get("action")
                print(f"[Hardware] App requested direct action: {action}")
                if bot is not None:
                    try:
                        if action == "home":
                            bot.center_all_servos()
                        elif action == "wave":
                            import threading
                            threading.Thread(target=bot.wave_arm, daemon=True).start()
                        elif action == "pose":
                            pose_name = event.get("pose_name", "home")
                            bot.set_pose(pose_name)
                        elif action == "move":
                            direction = event.get("direction", "stop")
                            duration = event.get("duration", 1.0)
                            bot.move_for(direction, duration, block=False)
                    except Exception as e:
                        print(f"[Hardware] Direct command error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Hardware direct command error: {str(e)}"
                        })
                else:
                    print(f"[Hardware] Robot client is unconnected. Action '{action}' simulated in mock mode.")

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
