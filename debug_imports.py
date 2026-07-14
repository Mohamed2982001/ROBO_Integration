import time
print("Importing config...")
import config.settings as cfg
print("Importing core.events...")
from core.events import EventQueue, SpeechEvent, VisionChangeEvent, RegistrationEvent
print("Importing database.memory_manager...")
from database.memory_manager import MemoryManager
print("Importing vision.vision_pipeline...")
from vision.vision_pipeline import VisionPipeline
print("Importing voice.stt_engine...")
from voice.stt_engine import HearingEngine
print("Importing voice.tts_engine...")
from voice.tts_engine import VoiceEngine
print("Importing brain.llm_engine...")
from brain.llm_engine import BrainEngine
print("Importing brain.session_manager...")
from brain.session_manager import SessionManager
print("Importing brain.memory.stm...")
from brain.memory.stm import ShortTermMemory
print("Importing brain.memory.ltm...")
from brain.memory.ltm import LongTermMemory
print("Imports complete!")
