import cv2
import sys

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)

backends = {
    "default": None,
    "CAP_DSHOW": cv2.CAP_DSHOW,
    "CAP_MSMF": cv2.CAP_MSMF,
}

found = False
for name, backend in backends.items():
    print(f"\nScanning backend: {name}...")
    for index in range(8):
        try:
            if backend is not None:
                cap = cv2.VideoCapture(index, backend)
            else:
                cap = cv2.VideoCapture(index)
            
            if cap.isOpened():
                ret, frame = cap.read()
                print(f"  --> INDEX {index} WORKS! (Frame read: {ret})")
                cap.release()
                found = True
            else:
                # Release anyway
                cap.release()
        except Exception as e:
            print(f"  Index {index} error: {e}")

if not found:
    print("\nNo working camera index/backend found. Please check if DroidCam Client preview is locking it, or if it is connected.")
