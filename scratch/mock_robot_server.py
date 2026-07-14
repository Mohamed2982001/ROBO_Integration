#!/usr/bin/env python3
"""
mock_robot_server.py
A zero-dependency HTTP server that simulates the ESP32 robot hardware.
It prints all received chassis movements and servo commands in real-time,
allowing you to test the AI integration offline without the physical robot chassis.

To run:
    python scratch/mock_robot_server.py [port]
Default port: 80 (requires admin privileges on some systems) or 8080.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# Current virtual status of the robot
ROBOT_STATE = {
    "speed": 160,
    "servos": [
        {"channel": 0, "min": 45, "max": 135, "start": 90, "current": 90},
        {"channel": 1, "min": 60, "max": 125, "start": 90, "current": 90},
        {"channel": 2, "min": 60, "max": 125, "start": 125, "current": 125},
        {"channel": 3, "min": 60, "max": 125, "start": 90, "current": 90},
        {"channel": 4, "min": 60, "max": 125, "start": 60, "current": 60},
        {"channel": 5, "min": 45, "max": 135, "start": 90, "current": 90},
        {"channel": 6, "min": 45, "max": 135, "start": 90, "current": 90},
        {"channel": 7, "min": 60, "max": 115, "start": 80, "current": 80},
        {"channel": 8, "min": 0, "max": 180, "start": 90, "current": 90},
        {"channel": 9, "min": 0, "max": 180, "start": 90, "current": 90},
    ]
}


class MockRobotHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default request logging to keep output clean
        return

    def _send_response(self, content: str, content_type: str = "text/plain", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # 1. Status query
        if path == "/status":
            resp_data = {
                "speed": ROBOT_STATE["speed"],
                "servos": ROBOT_STATE["servos"]
            }
            self._send_response(json.dumps(resp_data), "application/json")
            return

        # 2. Speed update
        if path == "/speed":
            val = int(query.get("value", [160])[0])
            ROBOT_STATE["speed"] = val
            print(f"\033[94m[MOCK ESP32] SPEED SET TO: {val}/255\033[0m")
            self._send_response("Speed updated")
            return

        # 3. Single servo move
        if path == "/servo":
            channel = int(query.get("num", [-1])[0])
            angle = int(query.get("angle", [-1])[0])
            if 0 <= channel < len(ROBOT_STATE["servos"]):
                ROBOT_STATE["servos"][channel]["current"] = angle
                print(f"\033[93m[MOCK ESP32] SERVO MOVE: Channel {channel} -> {angle}°\033[0m")
                self._send_response(f"Servo {channel} moved to {angle}")
            else:
                self._send_response("Invalid servo channel", status=400)
            return

        # 4. Atomic multi-servo pose update
        if path == "/pose":
            updates = []
            for ch in range(len(ROBOT_STATE["servos"])):
                param_name = f"s{ch}"
                if param_name in query:
                    angle = int(query[param_name][0])
                    ROBOT_STATE["servos"][ch]["current"] = angle
                    updates.append(f"{ch}:{angle}°")
            print(f"\033[93m[MOCK ESP32] MULTI-SERVO POSE: {{{', '.join(updates)}}}\033[0m")
            self._send_response("Pose updated")
            return

        # 5. Mecanum Chassis locomotion commands
        chassis_routes = {
            "/forward": "FORWARD",
            "/backward": "BACKWARD",
            "/left": "STRAFE LEFT (SIDEWAYS)",
            "/right": "STRAFE RIGHT (SIDEWAYS)",
            "/rotateleft": "ROTATE LEFT (SPIN)",
            "/rotateright": "ROTATE RIGHT (SPIN)",
            "/stop": "STOP ALL MOTORS",
            "/diagfl": "DIAGONAL FORWARD LEFT",
            "/diagfr": "DIAGONAL FORWARD RIGHT",
            "/diagbl": "DIAGONAL BACKWARD LEFT",
            "/diagbr": "DIAGONAL BACKWARD RIGHT",
        }

        if path in chassis_routes:
            action = chassis_routes[path]
            print(f"\033[92m[MOCK ESP32] CHASSIS MOTION: {action}\033[0m")
            self._send_response(f"Motion {action} acknowledged")
            return

        # Unknown route
        self._send_response("Not Found", status=404)


def run_mock_server():
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [port_number]")
            sys.exit(1)

    server_address = ("", port)
    try:
        httpd = HTTPServer(server_address, MockRobotHandler)
        print(f"\033[96m=====================================================")
        print(f"[*] Starting Mock ESP32 Robot Simulator on port {port}...")
        print(f"[*] Point your Integration/.env to ROBOT_IP=127.0.0.1")
        print(f"[*] Press Ctrl+C to stop simulation.")
        print(f"=====================================================\033[0m")
        httpd.serve_forever()
    except PermissionError:
        print(f"\033[91m[!] Permission denied binding to port {port}.\033[0m")
        print(f"\033[93m[!] Try running as administrator or use port 8080: python scratch/mock_robot_server.py 8080\033[0m")
    except KeyboardInterrupt:
        print("\n[*] Mock server shutting down cleanly.")


if __name__ == "__main__":
    run_mock_server()
