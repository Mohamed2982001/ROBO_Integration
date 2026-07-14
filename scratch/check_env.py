import sys
import importlib

packages = [
    "colorama",
    "dotenv",
    "numpy",
    "pydantic",
    "scipy",
    "groq",
    "langchain",
    "langchain_groq",
    "kokoro",
    "faster_whisper",
    "sounddevice",
    "soundfile",
    "pyaudio",
    "cv2",
    "ultralytics",
    "facenet_pytorch",
    "torch",
    "pymongo",
    "qdrant_client",
    "sentence_transformers",
    "easyocr",
    "fastapi",
    "uvicorn"
]

missing = []
for p in packages:
    try:
        importlib.import_module(p)
        print(f"[OK] {p}")
    except ImportError as e:
        print(f"[FAIL] {p} (Error: {e})")
        missing.append(p)

if missing:
    print(f"\nMissing packages: {missing}")
    sys.exit(1)
else:
    print("\n[SUCCESS] All packages are installed and ready!")
    sys.exit(0)
