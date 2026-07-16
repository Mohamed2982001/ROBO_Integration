"""
    pip install opencv-python mediapipe requests
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import mediapipe.solutions.face_detection
import mediapipe.solutions.drawing_utils

from robot_client import Robot


@dataclass
class FaceTrackerConfig:
    frame_width: int = 640
    frame_height: int = 480
    deadzone_radius: int = 30          # pixels inside this ring treat as "centered"

    pan_channel: int = 0               # head-horizontal servo channel
    tilt_channel: int = 7              # head-vertical servo channel

    pan_min: int = 45                  # min angle for pan servo (degrees)
    pan_max: int = 135                 # max angle for pan servo (degrees)
    tilt_min: int = 70                 # min angle for tilt servo (degrees)
    tilt_max: int = 110                # max angle for tilt servo (degrees)

    pan_gain: float = 1 / 80.0         # pixel error -> degrees/step
    tilt_gain: float = 1 / 50.0
    max_step_deg: float = 2.0          # largest single-step move (degrees)
    invert_pan: bool = False
    invert_tilt: bool = False

    smoothing: float = 0.35            # 0 = no smoothing, closer to 1 = smoother/slower to react
    min_delta_deg: float = 1.0         # don't send unless target moved at least this much
    send_interval: float = 0.12        # seconds between servo commands min ~8 Hz

    min_detection_confidence: float = 0.6
    model_selection: int = 0           # 0 = short range (<2m), 1 = long range


class FaceTracker:
    """
    Usage:
        from robot_client import Robot
        from face_tracker import FaceTracker, FaceTrackerConfig

        bot = Robot("192.168.1.42")   # IP printed by the ESP32 over Serial
        tracker = FaceTracker(bot, camera_url="http://admin:pass@host:8081/video")
        tracker.run()   # blocks; press ESC in the preview window to stop
    """

    def __init__(
        self,
        robot: Robot,
        camera_url: str | int = 0,
        config: Optional[FaceTrackerConfig] = None,
        show_preview: bool = True,
    ):
        self.robot = robot
        self.camera_url = camera_url
        self.cfg = config or FaceTrackerConfig()
        self.show_preview = show_preview

        # Attempt to load MediaPipe
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            import mediapipe.solutions.face_detection
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.cfg.model_selection,
                min_detection_confidence=self.cfg.min_detection_confidence,
            )
            self.use_mediapipe = True
            print("[FaceTracker] MediaPipe face detector loaded successfully ✓")
        except Exception as e:
            print(f"[FaceTracker] MediaPipe solutions not available: {e}. Falling back to OpenCV Haar Cascade...")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                print("[FaceTracker] ERROR: OpenCV Haar Cascade XML could not be loaded!")
            else:
                print("[FaceTracker] OpenCV Haar Cascade face detector loaded successfully ✓")

        # Start centered; will sync toward wherever the servos already are
        # if you set these from robot.get_servo_angle() before run().
        self.pan = float((self.cfg.pan_min + self.cfg.pan_max) / 2)
        self.tilt = float((self.cfg.tilt_min + self.cfg.tilt_max) / 2)

        self._last_sent_pan: Optional[float] = None
        self._last_sent_tilt: Optional[float] = None
        self._last_send_time = 0.0


    def _detect_face_center(self, frame) -> Optional[Tuple[int, int, Tuple[int, int, int, int]]]:
        if self.use_mediapipe:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._face_detector.process(rgb)
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    w, h = self.cfg.frame_width, self.cfg.frame_height
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    target_x = x_min + width // 2
                    target_y = y_min + height // 2
                    return target_x, target_y, (x_min, y_min, width, height)
            except Exception as e:
                # If MediaPipe runtime error occurs, fall back to OpenCV cascade
                pass

        # OpenCV Haar Cascade Fallback
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None
            x, y, w, h = faces[0]
            target_x = x + w // 2
            target_y = y + h // 2
            return target_x, target_y, (x, y, w, h)
        except Exception as e:
            print(f"[FaceTracker] Face detection failed: {e}")
            return None

    def _update_target_angles(self, error_x: float, error_y: float):
        cfg = self.cfg

        step_x = max(-cfg.max_step_deg, min(cfg.max_step_deg, error_x * cfg.pan_gain))
        step_y = max(-cfg.max_step_deg, min(cfg.max_step_deg, error_y * cfg.tilt_gain))

        if cfg.invert_pan:
            step_x = -step_x
        if cfg.invert_tilt:
            step_y = -step_y

        # Raw target before smoothing
        raw_pan = self.pan - step_x
        raw_tilt = self.tilt - step_y

        # Exponential smoothing so the head doesn't snap between values
        a = cfg.smoothing
        self.pan = a * self.pan + (1 - a) * raw_pan
        self.tilt = a * self.tilt + (1 - a) * raw_tilt

        self.pan = max(cfg.pan_min, min(cfg.pan_max, self.pan))
        self.tilt = max(cfg.tilt_min, min(cfg.tilt_max, self.tilt))

    def _maybe_send_servo_update(self):
        """Rate-limited + change-gated + atomic pan/tilt update."""
        now = time.time()
        if now - self._last_send_time < self.cfg.send_interval:
            return

        pan_i, tilt_i = int(round(self.pan)), int(round(self.tilt))

        moved_enough = (
            self._last_sent_pan is None
            or abs(pan_i - self._last_sent_pan) >= self.cfg.min_delta_deg
            or self._last_sent_tilt is None
            or abs(tilt_i - self._last_sent_tilt) >= self.cfg.min_delta_deg
        )
        if not moved_enough:
            return

        try:
            self.robot.set_servos({
                self.cfg.pan_channel: pan_i,
                self.cfg.tilt_channel: tilt_i,
            })
        except Exception as exc:  # keep tracking even if a frame's send fails
            print(f"[FaceTracker] servo update failed: {exc}")
            return

        self._last_sent_pan = pan_i
        self._last_sent_tilt = tilt_i
        self._last_send_time = now


    def run(self):
        cfg = self.cfg
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.camera_url!r}")

        center_x, center_y = cfg.frame_width // 2, cfg.frame_height // 2

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))
                detection = self._detect_face_center(frame)

                if detection is not None:
                    target_x, target_y, (x_min, y_min, width, height) = detection
                    error_x = target_x - center_x
                    error_y = target_y - center_y
                    distance = math.hypot(error_x, error_y)

                    if distance <= cfg.deadzone_radius:
                        circle_color = (0, 255, 0)
                        error_x = 0
                        error_y = 0
                    else:
                        circle_color = (255, 255, 255)

                    if error_x or error_y:
                        self._update_target_angles(error_x, error_y)
                        self._maybe_send_servo_update()

                    if self.show_preview:
                        cv2.circle(frame, (center_x, center_y), cfg.deadzone_radius, circle_color, 2)
                        cv2.circle(frame, (target_x, target_y), 6, (0, 255, 0), -1)
                        cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 2)

                if self.show_preview:
                    cv2.imshow("Face Tracking (rate-limited)", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

        except KeyboardInterrupt:
            print("Face tracking interrupted manually")
        finally:
            cap.release()
            if self.show_preview:
                cv2.destroyAllWindows()
            self.robot.stop()  # chassis-only; harmless if it was already idle



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python robot_client.py <robot_ip>\n"
            "Find the IP in the Arduino Serial Monitor after flashing the "
            "firmware (it prints 'Robot IP address: x.x.x.x')."
        )
        sys.exit(1)

    host = sys.argv[1]
    bot = Robot(host)

    print(f"Connecting to robot at {bot.base_url} ...")
    if not bot.ping():
        print("Could not reach the robot. Check that your computer and the "
              "robot are on the same Wi-Fi network, and that the IP is correct.")
        sys.exit(1)

    print("Connected. Running a short demo Ctrl+C to stop early")
    try:
        bot.set_pose("home")
        bot.move_for("forward", 1)
        bot.move_for("rotate_right", 0.5)
        bot.wave()
        bot.look_around()
        bot.square_patrol(leg_duration=0.8)
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop()
        print("Demo complete, motors stopped.")
