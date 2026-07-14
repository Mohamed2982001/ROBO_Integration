# How to Run: AI Robot System (Musa)

This guide provides a detailed, step-by-step procedure to set up, configure, and run the Musa AI Robot Companion project from scratch.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your system:
1. **Python**: Python 3.11 or 3.12 (64-bit recommended).
2. **MongoDB**: Local Community Server running on `mongodb://localhost:27017` (Default port).
3. **Hardware**: A working USB webcam and a microphone. An NVIDIA GPU with CUDA installed is highly recommended for low-latency voice, vision, and speech processing.

---

## 🛠️ Step 1: Environment Setup

1. **Open a Terminal** (e.g., PowerShell) in the root of the project directory `x:\Robot-main\Robot-main`.
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
   ```

---

## ⚙️ Step 2: Configuration

1. **Create the Environment File**:
   Copy `.env.example` to a new file named `.env`:
   ```powershell
   copy .env.example .env
   ```
2. **Add Your API Key**:
   Open the `.env` file and insert your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. *(Optional)* **Configure Custom Settings**:
   You can tweak the threshold values in `.env` or [config/settings.py](file:///x:/Robot-main/Robot-main/config/settings.py). For example:
   * `VOICE_THRESHOLD=0.60` (Voice print verification similarity cut-off).
   * `VOICE_EMBEDDING_MIN_SECONDS=1.5` (Minimum length of speech segment required to lock in/match voice).

---

## 🚀 Step 3: Running the Robot

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

## 👤 Step 4: User Registration (Onboarding)

When the robot runs for the first time or if you reset the database:
1. Stand in front of the camera. The robot will detect an unknown face.
2. Musa will speak and ask you: *"Hey — I didn't catch your name. What should I call you?"*
3. **Provide your name** (e.g., *"My name is Mahmoud"*).
4. Musa will ask for your age. **State your age** (e.g., *"I am 22"*).
5. Musa will ask you to look at the camera to register your face.
6. Musa will ask you to register your voice print: *"Please say a longer, clear sentence in a quiet voice so I can register your voice print."*
   * **Tip**: Speak a complete, continuous sentence under quiet room conditions (e.g., *"Hello Musa, I am testing my voice recording now and configuring my profile"*).
7. Musa will confirm the details and lock in your face, voice embedding, and MongoDB profile.

---

## 🔊 Step 5: Testing Voice-Only Identification (Camera Covered)

To test the newly configured speaker recognition system without using the camera:
1. Cover or disable your webcam.
2. Talk to the robot (e.g., say a complete sentence: *"Hey Musa, can you tell me who is talking to you right now?"*).
3. The console will display:
   ```text
   [VoiceAuth] ✓ Verified speaker: Mahmoud (confidence=0.67)
   [Session] New session owner: <your_user_id>
   ```
4. Because of our multi-modal fusion updates, the robot will identify you as **Mahmoud** entirely by voice and continue the conversation normally without demanding that you look at the camera.
