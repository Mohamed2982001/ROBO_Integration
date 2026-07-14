import json
import torch
from database.memory_manager import MemoryManager
from brain.session_manager import SessionManager
from brain.tools.crud_tool import CRUDTool
from config.settings import MONGO_URI

# Initialize components
mem = MemoryManager()
sess = SessionManager(None, None)
crud = CRUDTool(mem, vision_pipeline=None, session_manager=sess)

# Mock vision
class MockScene:
    faces = []
    current_speaker_id = None

class MockVision:
    def get_latest_scene(self):
        return MockScene()
    def get_current_speaker_embedding(self):
        return torch.randn(512)
    def refresh_face_cache(self):
        print("Mock face cache refreshed")

crud.vision = MockVision()

# Test 1: stage_user_profile
print("=== Test 1: Stage Profile ===")
res1 = crud.stage_user_profile('{"name": "Abdullah"}')
print(res1)

# Test 2: confirm_registration
print("=== Test 2: Confirm Registration ===")
res2 = crud.confirm_registration(confirm=True)
print(res2)

# Check DB
users = list(mem.mongo._col("users").find())
print("Users in DB:", users)
