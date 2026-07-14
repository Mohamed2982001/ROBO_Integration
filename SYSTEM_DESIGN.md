# 🤖 AI Robot System — System Design Document

**Version**: 1.0  
**Environment**: `grad_env` (Python 3.10, CUDA 12.1)

---

## 1. System Overview

An integrated AI robot that perceives its environment through vision and voice, reasons using a large language model, and responds via speech — with persistent memory per user.

### Core Capabilities
| Capability | Technology |
|-----------|-----------|
| Speech-to-Text | `faster-whisper` (base model) |
| Text-to-Speech | `kokoro` (streaming) |
| Face Recognition | YOLOv8-face + FaceNet (InceptionResnetV1) |
| Object Detection | YOLOv8n with ByteTrack |
| Speaker ID | Pluggable model interface |
| Brain / LLM | Groq — `llama3-20b` |
| Vector DB | Qdrant (local `qdrant.exe`) |
| Document DB | MongoDB (`mongod.exe`) |
| Text Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) |
| Face Embeddings | FaceNet 512-dim |
| Web Search | DuckDuckGo (`duckduckgo-search`) |
| OCR | EasyOCR |

---

## 2. Architecture

### 2.1 Thread Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Process                             │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Thread 1        │  │  Thread 2        │  │  Thread 3    │  │
│  │  VisionPipeline  │  │  HearingEngine   │  │  Brain Loop  │  │
│  │  (0.3s cycle)    │  │  (always-on STT) │  │  (consumer)  │  │
│  │                  │  │                  │  │              │  │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────┐  │  │
│  │  │Camera read │  │  │  │Mic capture │  │  │  │Consume │  │  │
│  │  │Face detect │  │  │  │Whisper STT │  │  │  │Events  │  │  │
│  │  │Obj detect  │  │  │  │Push Speech │  │  │  │from PQ │  │  │
│  │  │Change check│  │  │  │Event→PQ    │  │  │  │        │  │  │
│  │  └─────┬──────┘  │  │  └─────┬──────┘  │  │  └───┬───┘  │  │
│  └────────┼─────────┘  └────────┼─────────┘  └──────┼──────┘  │
│           │                     │                    │         │
│           └──────────┬──────────┘                    │         │
│                      ▼                               │         │
│           ┌──────────────────────┐                   │         │
│           │  Priority Queue (PQ) │◄──────────────────┘         │
│           │  Speech  priority=0  │                             │
│           │  Vision  priority=1  │                             │
│           └──────────────────────┘                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Shared State (thread-safe)                              │  │
│  │  latest_scene: VisionState  (updated every 0.3s)        │  │
│  │  tts_stop_event: threading.Event                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
VOICE INPUT:
  Microphone → faster-whisper → SpeechEvent → PriorityQueue
                                            ↓
  (if speaking detected) → tts_stop_event.set() + sd.stop()

VISION INPUT:
  Camera → YOLOv8-face → FaceNet embedding → Qdrant search → MongoDB user_info
         → YOLOv8n → object list
         → Change detector → VisionChangeEvent → PriorityQueue (only on change)
         → latest_scene updated always

BRAIN PROCESSING:
  PriorityQueue event (Speech or VisionChange)
    → Build context (vision_state + STM + user_info)
    → Groq LLM (streaming) with tool definitions
    → If tool call: execute tool → feed result → continue
    → Stream tokens → TTS SentenceBuffer → kokoro → speaker
    → Update STM, optionally save to LTM

SESSION LIFECYCLE:
  New speaker seen → session.start(user_id)
  60s silence → session.end() → STM summarize → embed → LTM
  Speaker leaves frame → session.end()
```

### 2.3 Memory Architecture

```
┌─────────────────────────────────────────────────┐
│                 Memory System                   │
│                                                 │
│  STM (in-process list)                          │
│  ┌─────────────────────────────────────────┐    │
│  │ [msg1][msg2][msg3][msg4][msg5] → summarize│   │
│  │         ↓ after 5 msgs                   │   │
│  │ [summary] → continues filling            │   │
│  └─────────────────────────────────────────┘    │
│                      ↓ on session end           │
│  LTM (Qdrant — ltm_memories collection)         │
│  ┌─────────────────────────────────────────┐    │
│  │ embedding(summary) + {user_id, date}    │    │
│  │ embedding(summary) + {user_id, date}    │    │
│  │ ...                                     │    │
│  └─────────────────────────────────────────┘    │
│                      ↑ RAG retrieval            │
│                  filtered by user_id            │
│                                                 │
│  Identity (Qdrant — identity_embeddings)        │
│  ┌─────────────────────────────────────────┐    │
│  │ face_embedding → {mongo_user_id}        │    │
│  │ face_embedding → {mongo_user_id}        │    │
│  └─────────────────────────────────────────┘    │
│                      ↑ search with face emb     │
│                                                 │
│  User Profiles (MongoDB — users collection)     │
│  ┌─────────────────────────────────────────┐    │
│  │ {name, preferences, habits, schedule}   │    │
│  │ {name, preferences, habits, schedule}   │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## 3. Qdrant Collections

| Collection | Vector Size | Distance | Purpose |
|-----------|-------------|----------|---------|
| `identity_embeddings` | 512 | Cosine | Face embeddings → user_id |
| `ltm_memories` | 768 | Cosine | Long-term memories per user |

---

## 4. LLM Tool Definitions

| Tool | Description | Parameters |
|------|-------------|-----------|
| `search_web` | DuckDuckGo web search | `query: str` |
| `search_memory` | RAG from LTM | `query: str` |
| `perform_ocr` | Extract text from camera | — |
| `get_conversation_history` | Read STM | — |
| `db_create` | Insert record | `collection, data` |
| `db_read` | Query record | `collection, query` |
| `db_update` | Update record | `collection, query, update` |
| `db_delete` | Delete record | `collection, query` |

---

## 5. Vision Change Events

| Event | Trigger | Brain Action |
|-------|---------|-------------|
| `new_person` | New face track_id | Greet if known, prompt if unknown |
| `person_left` | Face track_id gone | End session, flush STM→LTM |
| `emotion_shift` | Emotion diff on same track | Adapt tone |
| `gesture` | Gesture detected | React (future) |
| `unknown_idle` | Unknown face idle > 3s | Ask "Can I help you?" |

---

## 6. Proactive Behavior

```
Vision detects known person "Ali"
  → VisionChangeEvent("new_person", {"name": "Ali"})
  → Brain: "Hello Ali! Good to see you again."
  → TTS speaks greeting

Vision detects unknown person idle 3s
  → VisionChangeEvent("unknown_idle", {...})
  → Brain: "Hello! I'm your robot assistant. What's your name?"
  → TTS speaks

Unknown person says "My name is Ahmed"
  → SpeechEvent("My name is Ahmed")
  → Brain extracts name "Ahmed"
  → vision.get_current_speaker_embedding() → register in Qdrant + MongoDB
  → TTS: "Nice to meet you, Ahmed! I'll remember you."
```

---

## 7. Session Lifecycle

```
START: New person detected (VisionChangeEvent "new_person")
  → session.start(user_id)
  → Load user LTM context for RAG

ACTIVE: Each interaction
  → Reset 60s timeout timer
  → STM.add_message(user_text, robot_response)

AUTO-END: 60s inactivity
  → session.end()
  → STM.summarize() → embed → LTM.store(user_id)
  → STM.clear()

EVENT-END: Speaker leaves frame
  → VisionChangeEvent("person_left") → session.end()
```

---

## 8. File Structure

```
robot/
├── .env                          # GROQ_API_KEY, DB URIs
├── config/
│   ├── __init__.py
│   └── settings.py               # Central config from .env
├── core/
│   ├── __init__.py
│   └── events.py                 # Event types + PriorityQueue
├── database/
│   ├── __init__.py
│   ├── mongo_manager.py
│   ├── qdrant_manager.py
│   └── memory_manager.py
├── vision/
│   ├── __init__.py
│   ├── face_recognition.py
│   ├── object_recognition.py
│   ├── speaker_identifier.py
│   └── vision_pipeline.py
├── voice/
│   ├── __init__.py
│   ├── stt_engine.py
│   └── tts_engine.py
├── brain/
│   ├── __init__.py
│   ├── llm_engine.py
│   ├── prompts.py
│   ├── session_manager.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── stm.py
│   │   └── ltm.py
│   └── tools/
│       ├── __init__.py
│       ├── tool_registry.py
│       ├── ocr_tool.py
│       ├── web_search_tool.py
│       ├── rag_tool.py
│       ├── crud_tool.py
│       └── stm_tool.py
├── models/                       # .pt weight files
├── tests/
│   ├── test_database.py
│   ├── test_vision.py
│   ├── test_stt.py
│   ├── test_tts.py
│   ├── test_brain.py
│   ├── test_tools.py
│   ├── test_memory.py
│   └── test_integration.py
├── main.py
├── requirements.txt
└── SYSTEM_DESIGN.md
```

---

## 9. Requirements

```
# LLM
groq>=0.37.1
langchain-groq>=1.1.2

# Voice
kokoro>=0.9.4
faster-whisper>=1.2.1
sounddevice>=0.5.1
soundfile

# Vision
opencv-python
ultralytics>=8.4.0
facenet-pytorch>=2.5.3
torch>=2.5.1
torchvision

# Database
pymongo>=4.17.0
qdrant-client>=1.17.1

# Embeddings
sentence-transformers>=5.4.0

# Tools
duckduckgo-search>=8.1.1
easyocr

# Utilities
python-dotenv>=1.2.1
colorama>=0.4.6
numpy
```
