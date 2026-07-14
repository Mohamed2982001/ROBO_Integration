"""
brain/tools/ocr_tool.py
OCR tool using EasyOCR for robust text detection.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None  # type: ignore[assignment]
    _EASYOCR_AVAILABLE = False

import config.settings as cfg


class OCRTool:
    def __init__(self):
        self.latest_frame: np.ndarray | None = None
        self.reader = None

        if not _EASYOCR_AVAILABLE:
            print("[OCR] WARNING: easyocr not installed. Run: pip install easyocr")
            return

        use_gpu = torch.cuda.is_available()
        print(f"[OCR] Loading EasyOCR model... (gpu={use_gpu})")
        try:
            self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            print("[OCR] EasyOCR loaded OK")
        except Exception as e:
            print(f"[OCR] Failed to load EasyOCR: {e}")
            self.reader = None

    def update_frame(self, frame: np.ndarray) -> None:
        """Called every vision cycle to keep the latest camera frame."""
        self.latest_frame = frame

    @staticmethod
    def _center_roi(frame: np.ndarray, margin_x: float = 0.18, margin_y: float = 0.14) -> np.ndarray:
        """Crop to the centre of the frame to reduce background clutter."""
        h, w = frame.shape[:2]
        x1, x2 = int(w * margin_x), int(w * (1.0 - margin_x))
        y1, y2 = int(h * margin_y), int(h * (1.0 - margin_y))
        return frame[y1:y2, x1:x2]

    def perform_ocr(self) -> str:
        """Perform OCR on the latest camera frame and return detected text."""
        if self.reader is None:
            return "OCR model not available. Install easyocr: pip install easyocr"

        if self.latest_frame is None:
            return "No camera frame available yet."

        thresh = float(getattr(cfg, "OCR_MIN_CONFIDENCE", 0.35))
        aggregated: list[tuple[float, str]] = []

        # Try centre crop first, fall back to full frame
        for label, crop in (
            ("center", self._center_roi(self.latest_frame)),
            ("full", self.latest_frame),
        ):
            try:
                # EasyOCR returns [(bbox, text, confidence), ...]
                results = self.reader.readtext(crop)
            except Exception as e:
                return f"OCR read failed ({label}): {e}"

            for _bbox, text, conf in results:
                raw = str(text).strip()
                if raw and float(conf) >= thresh:
                    aggregated.append((float(conf), raw))

            if aggregated:
                break  # centre crop was enough

        if not aggregated:
            return (
                "No readable text detected. "
                "Try better lighting, hold the document flat, or use larger print."
            )

        # Return text sorted by descending confidence
        aggregated.sort(key=lambda x: -x[0])
        return " ".join(t for _, t in aggregated)
