# How to Run: Integrated AI Robot System (Musa)

This guide provides a detailed, step-by-step procedure to set up, configure, run, and test the integrated AI Brain and ESP32 Mecanum Chassis project.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your system:
1. **Python**: Python 3.11 or 3.12 (64-bit).
2. **MongoDB**: Local Community Server running on `mongodb://localhost:27017` (Default port).
3. **Hardware (for full setup)**: USB webcam, microphone, and the ESP32 Mecanum Chassis robot.
4. **NVIDIA GPU (Recommended)**: For low-latency voice, vision, and speech processing.

---

## 🛠️ Step 1: Environment Setup

1. **Open a Terminal** (e.g., PowerShell) in the root of the project directory `d:\Work\Dev\Grad Proj\Integration`.
2. **Create a Virtual Environment**:
   ```powershell
   python -m venv .venv
   ```
3. **Activate the Virtual Environment**:
   * **PowerShell**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Command Prompt (CMD)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
4. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   pip install mediapipe opencv-python requests
   ```

---

## ⚙️ Step 2: Configuration

1. **Create the Environment File**:
   Copy `.env.example` to a new file named `.env`:
   ```powershell
   copy .env.example .env
   ```
2. **Configure Robot IP Address**:
   Open the `.env` file and configure `ROBOT_IP` to match the IP address of your ESP32 robot:
   ```env
   ROBOT_IP=192.168.1.18
   GROQ_API_KEY=your_groq_api_key_here
   ```
   *If testing offline without physical hardware, set `ROBOT_IP=127.0.0.1`.*

---

## 💻 Step 3: Offline Hardware Testing (Mock Simulator)

If you do not have the physical robot body, you can simulate all motor and servo movements locally:

1. **Start the Mock Simulator** in a separate terminal:
   ```powershell
   python scratch/mock_robot_server.py
   ```
   *This starts a mock server on port `8080` (or `80` if run as admin). If using port 8080, make sure to adjust `ROBOT_IP` or port settings accordingly.*
2. **Set `.env`**: Ensure `ROBOT_IP` is set to `127.0.0.1` (localhost).
3. **Test the Integration**: Start the main system (Step 4). When the robot thinks, speaks, or runs movement tools, you will see all low-level HTTP requests printed in color inside the mock server terminal:
   * `[MOCK ESP32] CHASSIS MOTION: FORWARD`
   * `[MOCK ESP32] SERVO MOVE: Channel 7 -> 95°`
   * `[MOCK ESP32] MULTI-SERVO POSE: {1:125°, 2:80°}`

---

## 🚀 Step 4: Running the Main System

1. Make sure your **MongoDB service** is running in the background.
2. Run the startup script. This will automatically launch the local vector database (`qdrant.exe`) and start the robot's main orchestrator:
   ```powershell
   .\run.bat
   ```
   *If you encounter execution policy restrictions, run this command instead:*
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\run_robot.ps1
   ```

---

## 👤 Step 5: User Registration & Face Tracking

1. **Camera Sharing**: The main webcam is managed exclusively by the `VisionPipeline`. MediaPipe face tracking runs inside it at 30 fps to control the neck servos without locking conflicts.
2. **Tracking Test**: Move in front of the camera. The neck pan/tilt servos (Channels 0 and 7) will move to track your face center.
3. **Registration Flow**:
   * Stand in front of the camera. Musa will ask for your name and age.
   * Musa will record a voice print.
   * Once confirmed, your profile is saved, and subsequent logins by face/voice are instant.

---

## 🤖 Step 6: Gesture & Action Commands

You can speak to the robot using natural language:
* **Greetings**: "Hi Musa, hello!" -> Triggers arm waving gesture.
* **Questions**: "How do you work?" -> Triggers head tilting.
* **Control Commands**:
  * *"Move forward for 2 seconds and open your right gripper claw."*
  * *"Look up and raise your arms."*
  * Musa will call the registered tools (`move_robot`, `set_robot_pose`, `control_gripper`), executing the low-level HTTP queries to the ESP32.
