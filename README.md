# Robot Hardware Control & Face Tracking System

An end-to-end Python ecosystem designed to interface with an ESP32-controlled multi-servo robotic platform featuring a mecanum chassis. This repository provides low-level network abstractions, high-level behavioral APIs, and a real-time computer vision face-tracking pipeline using OpenCV and MediaPipe.

---

## 🚀 Key Features

* **Omnidirectional Chassis Control:** Native support for standard directions and diagonal Mecanum movements (`forward`, `backward`, `strafe_left`, `strafe_right`, `diagonal_forward_left`, etc.).
* **Atomic Servo Grouping:** Implements synchronous, multi-axis servo positioning over custom HTTP API endpoints to ensure coordinated multi-joint movements (e.g., arms, elbows, grippers).
* **AI-Powered Face Tracking:** Integrated closed-loop vision tracking utilizing **MediaPipe Face Detection** and **OpenCV** to automatically adjust pan-and-tilt neck servos.
* **Pose Management Ecosystem:** System to define, save, export, and import custom multi-servo configurations via JSON files.
* **Failsafe Core Design:** Clean context manager integration ensuring the robot safely executes structural disarming, stops chassis motion, and homes all servos on keyboard interrupts or network termination.

---

## 📦 Project Architecture

* **`robot_client.py`**: The core HTTP client library (`Robot`). Manages the underlying network communication sessions, thread-safe operation locks, hardware angle boundary-clamping, safety limits, and built-in macro routines (`wave`, `look_around`, `square_patrol`).
* **`face_tracker.py`**: Real-time closed-loop control module. Processes camera feeds, applies exponential smoothing filters to avoid servo jitter, filters minor adjustments via deadzone configurations, and rate-limits network payloads.
* **`example_usage.py`**: A clean, deployment-ready entry point demonstrating basic motion sequences, timing profiles, and system cleanup parameters.

---

## 🛠️ Hardware Channel Mapping & Boundaries

The client enforces explicit hardware protection boundaries based on the physical configuration of the robot:

| Channel | Component | Safe Min Angle | Safe Max Angle | Default Home |
| :---: | :--- | :---: | :---: | :---: |
| **0** | Head Pan (Horizontal) | 45° | 135° | 90° |
| **1** | Right Shoulder Pitch | 60° | 125° | 90° |
| **2** | Right Shoulder Roll | 60° | 125° | 125° |
| **3** | Left Shoulder Pitch | 60° | 125° | 90° |
| **4** | Left Shoulder Roll | 60° | 125° | 60° |
| **5** | Left Elbow Pitch | 45° | 135° | 90° |
| **6** | Right Elbow Pitch | 45° | 135° | 90° |
| **7** | Head Tilt (Vertical) | 60° | 115° | 80° |
| **8** | Right Gripper / Claw | 0° | 180° | 90° |
| **9** | Left Gripper / Claw | 0° | 180° | 90° |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure your machine is running **Python 3.11+** and is connected to the **same local Wi-Fi network** as the ESP32 microcontroller.

### 2. Install Dependencies
Install the required computer vision, media pipelines, and networking suites:
```bash
pip install opencv-python mediapipe requests
