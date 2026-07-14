import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from core.json_util import sanitize_for_api
from vision.vision_pipeline import VisionPipeline

app = FastAPI(title="Robot Vision Interface")
os.makedirs("scratch/voices", exist_ok=True)
app.mount("/audio", StaticFiles(directory="scratch/voices"), name="audio")

vision_pipeline_instance: Optional[VisionPipeline] = None
_vision_server_instance: Optional["VisionServer"] = None


def _apply_display_names(scene_dict: dict, ui_state: dict) -> dict:
    overrides = ui_state.get("face_display_names") or {}
    pending_global = ui_state.get("pending_display_name")
    faces = scene_dict.get("faces") or []
    unknowns = [f for f in faces if f.get("name") == "Unknown"]
    for f in faces:
        if f.get("name") != "Unknown":
            continue
        tid = str(f.get("track_id", ""))
        label = overrides.get(tid)
        if not label and pending_global and len(unknowns) == 1:
            label = pending_global
        if label:
            f["name"] = label
            f["registration_pending"] = True
    return scene_dict


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open("vision/templates/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            if vision_pipeline_instance:
                frame = vision_pipeline_instance.get_video_frame()
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
            time.sleep(0.05)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
async def get_status():
    if vision_pipeline_instance:
        scene = vision_pipeline_instance.get_latest_scene()
        data = scene.to_dict()
        if _vision_server_instance and _vision_server_instance.ui_state:
            data = _apply_display_names(data, _vision_server_instance.ui_state)
        return data
    return {"error": "Vision pipeline not initialized"}


class VisionServer:
    def __init__(
        self,
        vision_pipeline: VisionPipeline,
        host: str = "0.0.0.0",
        port: int = 8080,
        ui_state: dict = None,
    ):
        global vision_pipeline_instance, _vision_server_instance
        vision_pipeline_instance = vision_pipeline
        _vision_server_instance = self
        self.host = host
        self.port = port
        self.ui_state = ui_state or {
            "conversation_history": [],
            "system_status": "Starting...",
        }
        self.thread: Optional[threading.Thread] = None

        @app.get("/session_status")
        async def get_session_status():
            return sanitize_for_api(self.ui_state)

        @app.get("/api/users")
        async def get_users():
            if vision_pipeline_instance:
                users = vision_pipeline_instance.db.list_users()
                return {
                    "users": [
                        {"id": str(u["_id"]), "name": u["name"]} for u in users
                    ]
                }
            return {"users": []}

        @app.post("/api/assign_voice")
        async def assign_voice(request: Request):
            data = await request.json()
            voice_id = data.get("voice_id")
            user_id = data.get("user_id")

            voices = self.ui_state.get("unassigned_voices", [])
            target = next((v for v in voices if v["id"] == voice_id), None)

            if not target:
                return {"success": False, "error": "Voice not found"}

            if vision_pipeline_instance and vision_pipeline_instance.db.register_voice(
                user_id, target["embedding"]
            ):
                self.ui_state["unassigned_voices"] = [
                    v for v in voices if v["id"] != voice_id
                ]
                return {"success": True}

            return {"success": False, "error": "Failed to assign voice"}

    def start(self):
        self.thread = threading.Thread(
            target=lambda: uvicorn.run(
                app, host=self.host, port=self.port, log_level="warning"
            ),
            daemon=True,
            name="VisionServer",
        )
        self.thread.start()
        print(f"[VisionServer] Dashboard active at http://localhost:{self.port} ✓")

    def stop(self):
        pass
