"""
    from robot_client import Robot


    bot = Robot("192.168.1.42")    # the IP printed by the ESP32 over Serial
    bot.forward()
    time.sleep(1)
    bot.stop()

    bot.set_servo(0, 90)                    # single servo
    bot.set_servos({0: 90, 7: 100})         # group of servos, sent atomically
    bot.set_pose("wave")                    # named built-in pose
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union

import requests


# servo configurtion and limits 

@dataclass
class ServoLimit:
    min_angle: int
    max_angle: int
    start_angle: int


DEFAULT_SERVO_LIMITS: Dict[int, ServoLimit] = {
    0: ServoLimit(45, 135, 90),   # head horizontal  - neck pan
    1: ServoLimit(60, 125, 90),   # arm rotate right - shoulder pitch
    2: ServoLimit(60, 125, 125),  # arm right above  - sholder roll
    3: ServoLimit(60, 125, 90),   # arm rotate left  - shoulder pitch
    4: ServoLimit(60, 125, 60),   # arm left above   - shoulder roll
    5: ServoLimit(45, 135, 90),   # arm left under   - elbow pitch
    6: ServoLimit(45, 135, 90),   # arm right under  - elbow pitch
    7: ServoLimit(60, 115, 80),   # head vertical    - neck tilt
    8: ServoLimit(0, 180, 90),    # gripper right
    9: ServoLimit(0, 180, 90),    # gripper left
}

# Friendly aliases for the two new channels. Rename these (and the limits
# above) to match whatever you actually mount on channels 8 and 9.
GRIPPER_CHANNEL_r = 8
GRIPPER_CHANNEL_l = 9

# A few handy named poses out of the box. All angles are within the default
# limits above; add your own with Robot.save_pose() / the `poses` dict.
BUILTIN_POSES: Dict[str, Dict[int, int]] = {
    "home": {i: lim.start_angle for i, lim in DEFAULT_SERVO_LIMITS.items()},
    "arms_up": {1: 60, 2: 125, 3: 125, 4: 125},
    "arms_down": {1: 125, 2: 60, 3: 60, 4: 60},
    "look_left": {0: 45},
    "look_right": {0: 135},
    "look_up": {7: 115},
    "look_down": {7: 60},
    "wave": {1: 90, 2: 125}, 
    "gripper_open": {8: 180},
    "gripper_closed": {8: 0},
    "waist_center": {9: 90},
}


class RobotConnectionError(RuntimeError):
    """Raised when the robot can't be reached or returns an error."""


class RobotError(RuntimeError):
    """Raised for invalid parameters (bad channel, bad angle, etc.)."""


# Main client

class Robot:
    """
    HTTP client for the robot. Every chassis/servo command is a simple GET
    request to the ESP32, mirroring the routes defined in robot_firmware.ino.
    """

    NUM_SERVOS = 10

    def __init__(
        self,
        host: str,
        port: int = 80,
        timeout: float = 3.0,
        servo_limits: Optional[Dict[int, ServoLimit]] = None,
        session: Optional[requests.Session] = None,
        auto_fetch_limits: bool = False,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.servo_limits = dict(servo_limits or DEFAULT_SERVO_LIMITS)
        self.session = session or requests.Session()
        self.poses: Dict[str, Dict[int, int]] = dict(BUILTIN_POSES)
        self._servo_state: Dict[int, int] = {
            ch: lim.start_angle for ch, lim in self.servo_limits.items()
        }
        self._speed = 255
        self._lock = threading.Lock()

        if auto_fetch_limits:
            try:
                self.refresh_status()
            except RobotConnectionError:
                pass  # fall back to defaults silently

    #  context manager convenience 
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __repr__(self):
        return f"Robot(base_url={self.base_url!r})"

    # Low-level transport

    def _get(self, path: str, params: Optional[dict] = None) -> str:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            raise RobotConnectionError(f"Request to {url} failed: {exc}") from exc

    def ping(self) -> bool:
        """Return True if the robot responds, False otherwise"""
        try:
            self._get("/status")
            return True
        except RobotConnectionError:
            return False

    def refresh_status(self) -> dict:
        """
        Query /status on the firmware (if present) and refresh local
        servo_limits + speed from the robot's reported configuration.
        """
        text = self._get("/status")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RobotConnectionError(f"Unexpected /status response: {text!r}") from exc

        self._speed = data.get("speed", self._speed)
        for entry in data.get("servos", []):
            ch = entry["channel"]
            self.servo_limits[ch] = ServoLimit(entry["min"], entry["max"], entry["start"])
        return data

    # Chassis: basic directions

    def forward(self):
        self._get("/forward")

    def backward(self):
        self._get("/backward")

    def strafe_left(self):
        self._get("/left")

    def strafe_right(self):
        self._get("/right")

    def rotate_left(self):
        self._get("/rotateleft")

    def rotate_right(self):
        self._get("/rotateright")

    def stop(self):
        self._get("/stop")

    # Chassis: diagonal mecanum moves

    def diagonal_forward_left(self):
        self._get("/diagfl")

    def diagonal_forward_right(self):
        self._get("/diagfr")

    def diagonal_backward_left(self):
        self._get("/diagbl")

    def diagonal_backward_right(self):
        self._get("/diagbr")


    # Chassis: unified direction dispatcher + timed moves


    _DIRECTIONS = {
        "forward": "forward",
        "backward": "backward",
        "left": "strafe_left",
        "right": "strafe_right",
        "rotate_left": "rotate_left",
        "rotate_right": "rotate_right",
        "forward_left": "diagonal_forward_left",
        "forward_right": "diagonal_forward_right",
        "backward_left": "diagonal_backward_left",
        "backward_right": "diagonal_backward_right",
        "stop": "stop",
    }

    def move(self, direction: str):
        """
        Move in any supported direction by name. Valid values:
        forward, backward, left, right, rotate_left, rotate_right,
        forward_left, forward_right, backward_left, backward_right, stop
        """
        method_name = self._DIRECTIONS.get(direction)
        if method_name is None:
            raise RobotError(
                f"Unknown direction {direction!r}. Valid options: "
                f"{sorted(self._DIRECTIONS)}"
            )
        getattr(self, method_name)()

    def move_for(self, direction: str, duration: float, block: bool = True):
        """
        Move in `direction` for `duration` seconds, then stop.
        If block=False, runs in a background thread and returns immediately.
        """
        def _run():
            self.move(direction)
            time.sleep(duration)
            self.stop()

        if block:
            _run()
        else:
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            return t

    def set_speed(self, value: int):
        """Set chassis motor speed (PWM duty, 0-255). Applies on next move."""
        value = max(0, min(255, int(value)))
        self._speed = value
        self._get("/speed", params={"value": value})

    @property
    def speed(self) -> int:
        return self._speed

    # Servos: single, group, and pose control

    def _clamp(self, channel: int, angle: int) -> int:
        if channel not in self.servo_limits:
            raise RobotError(f"Invalid servo channel {channel}; must be 0-7")
        lim = self.servo_limits[channel]
        return max(lim.min_angle, min(lim.max_angle, int(angle)))
    

    def set_servo(self, channel: int, angle: int):
        """Move a single servo channel (0-7) to `angle` degrees."""
        clamped = self._clamp(channel, angle)
        self._get("/servo", params={"num": channel, "angle": clamped})
        self._servo_state[channel] = clamped


    def set_servos(self, angles: Dict[int, int]):
        """
        Move several servos at once, in a single atomic HTTP request
        (uses the firmware's /pose route so all channels update together).
        """
        params = {}
        for channel, angle in angles.items():
            clamped = self._clamp(channel, angle)
            params[f"s{channel}"] = clamped
            self._servo_state[channel] = clamped
        self._get("/pose", params=params)



    def get_servo_angle(self, channel: int) -> int:
        """Last known angle sent for a channel (client-side cache, not read from hardware)."""
        return self._servo_state.get(channel, self.servo_limits[channel].start_angle)
    


    #  convenience wrappers for the 2 new servos (channels 8 & 9) 

    def set_gripper_r(self, angle: int):
        """Move the new channel-8 servo (default role: gripper/claw)."""
        self.set_servo(GRIPPER_CHANNEL_r, angle)

    def open_gripper_r(self):
        self.set_servo(GRIPPER_CHANNEL_r, self.servo_limits[GRIPPER_CHANNEL_r].max_angle)

    def close_gripper_r(self):
        self.set_servo(GRIPPER_CHANNEL_r, self.servo_limits[GRIPPER_CHANNEL_r].min_angle)

    def set_gripper_l(self, angle: int):
        """Move the new channel-9 servo (default role: gripper/claw)."""
        self.set_servo(GRIPPER_CHANNEL_l, angle)

    def open_gripper_l(self):
        self.set_servo(GRIPPER_CHANNEL_l, self.servo_limits[GRIPPER_CHANNEL_l].max_angle)

    def close_gripper_l(self):
        self.set_servo(GRIPPER_CHANNEL_l, self.servo_limits[GRIPPER_CHANNEL_l].min_angle)

    def center_all_servos(self):
        """Send every servo back to its configured start angle."""
        self.set_servos({ch: lim.start_angle for ch, lim in self.servo_limits.items()})

    # chose a pose by name or by raw {channel: angle} dict

    def set_pose(self, pose: Union[str, Dict[int, int]]):
        """
        Apply a pose. Accepts either the name of a built-in/saved pose
        (see self.poses) or a raw {channel: angle} dict.
        """
        if isinstance(pose, str):
            if pose not in self.poses:
                raise RobotError(f"Unknown pose {pose!r}. Known poses: {sorted(self.poses)}")
            angles = self.poses[pose]
        else:
            angles = pose
        self.set_servos(angles)

    def save_pose(self, name: str, angles: Optional[Dict[int, int]] = None):
        """
        Save a pose under `name`. If `angles` is omitted, captures the
        robot's current (client-cached) servo positions.
        """
        self.poses[name] = dict(angles) if angles is not None else dict(self._servo_state)

    def list_poses(self) -> List[str]:
        return sorted(self.poses.keys())

    def export_poses(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.poses, f, indent=2)

    def import_poses(self, filepath: str, merge: bool = True):
        with open(filepath) as f:
            loaded = json.load(f)
        loaded = {k: {int(ch): a for ch, a in v.items()} for k, v in loaded.items()}
        if merge:
            self.poses.update(loaded)
        else:
            self.poses = loaded

    # move a list of servos together 

    def move_group(self, channels: Iterable[int], angle: int):
        """Send the same angle to a list/set of servo channels at once."""
        self.set_servos({ch: angle for ch in channels})
    # move a single servo smoothly through a range of angles, with small delays

    def sweep_servo(self, channel: int, start: int, end: int, step: int = 5, delay: float = 0.03):
        rng = range(start, end + 1, step) if end >= start else range(start, end - 1, -step)
        for angle in rng:
            self.set_servo(channel, angle)
            time.sleep(delay)
        self.set_servo(channel, end)

    # Simple animated demo behaviors

    def wave(self, channel: int = 1, low: int = 70, high: int = 110, cycles: int = 3, delay: float = 0.4):
        """Animate a servo back and forth (e.g. an arm wave)."""
        for _ in range(cycles):
            self.set_servo(channel, high)
            time.sleep(delay)
            self.set_servo(channel, low)
            time.sleep(delay)

    def wave_arm(self, duration: float = 4.0, delay: float = 0.3):
        """
        Animate the right arm: positions both shoulder joints to a waving stance,
        cycles the elbow back and forth for a fixed duration, and returns all 
        three joints atomically back to their home angles.
        """
        shoulder_pitch_r= 1
        shoulder_roll_r = 2
        elbow_pitch = 6

        target_pitch = 125
        target_roll = 80  
        elbow_low = 80              
        elbow_high = 100            

        self.set_servos({
            shoulder_pitch_r: target_pitch,
            shoulder_roll_r: target_roll
        })
        time.sleep(0.5)

        start_time = time.time()
        toggle = True
        
        while time.time() - start_time < duration:
            target_angle = elbow_high if toggle else elbow_low
            self.set_servo(elbow_pitch, target_angle)
            toggle = not toggle
            time.sleep(delay)

        home_pitch = self.servo_limits[shoulder_pitch_r].start_angle
        home_roll = self.servo_limits[shoulder_roll_r].start_angle
        home_elbow = self.servo_limits[elbow_pitch].start_angle
        
        self.set_servos({
            shoulder_pitch_r: home_pitch,
            shoulder_roll_r: home_roll,
            elbow_pitch: home_elbow
        })

    def look_around(self, pan_channel: int = 0, tilt_channel: int = 7, delay: float = 0.6):
        """Pan/tilt the head through its full range and back to center."""
        pan_lim = self.servo_limits[pan_channel]
        tilt_lim = self.servo_limits[tilt_channel]
        self.set_servo(pan_channel, pan_lim.min_angle+10)
        time.sleep(delay)
        self.set_servo(pan_channel, pan_lim.max_angle-10)
        time.sleep(delay)
        self.set_servo(tilt_channel, tilt_lim.min_angle+10)
        time.sleep(delay)
        self.set_servo(tilt_channel, tilt_lim.max_angle-10)
        time.sleep(delay)
        self.set_servo(pan_channel, pan_lim.start_angle)
        self.set_servo(tilt_channel, tilt_lim.start_angle)

    def square_patrol(self, chasiss_duration: float = 1.0, pause: float = 0.3):
        """Drive in a square using the four cardinal directions."""
        for direction in ("forward", "right", "backward", "left"):
            self.move_for(direction, chasiss_duration)
            time.sleep(pause)
        self.stop()


# CLI demo (python robot_client.py [host])


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
