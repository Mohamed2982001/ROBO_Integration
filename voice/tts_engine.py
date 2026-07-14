"""
voice/tts_engine.py
Kokoro-based TTS engine with:
  - Streaming support: tokens from LLM → SentenceBuffer → kokoro
  - Interrupt: threading.Event + sd.stop() stops audio immediately
  - is_speaking flag for STT to check before interrupting
"""
from __future__ import annotations

import re
import threading
import time
from typing import Generator, Iterator, Optional

import sounddevice as sd
from colorama import Fore, Style, init
from kokoro import KPipeline

init(autoreset=True)


class SentenceBuffer:
    """Accumulates tokens and yields complete sentences."""

    _SPLIT = re.compile(r"(.*?[.?!:\n]+)\s*")

    def __init__(self):
        self.buffer = ""

    def add_token(self, token: str):
        self.buffer += token
        sentences = []
        while True:
            m = self._SPLIT.match(self.buffer)
            if m:
                sentences.append(m.group(1))
                self.buffer = self.buffer[len(m.group(1)):].lstrip()
            else:
                break
        return sentences

    def flush(self):
        result = self.buffer.strip()
        self.buffer = ""
        return [result] if result else []


class VoiceEngine:
    """
    Kokoro TTS engine with streaming and interrupt support.
    """

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, robot: Optional[Any] = None):
        print(f"{Fore.CYAN}[TTS] Loading Kokoro voice engine...{Style.RESET_ALL}")
        from core.device import pick_torch_device

        device = pick_torch_device()
        self.pipeline = KPipeline(lang_code="a", device=device)
        self.voice = voice
        self.speed = speed
        self.robot = robot

        self.stop_event = threading.Event()
        self.is_speaking = False
        self._lock = threading.Lock()
        self.last_output_end_ts: float = 0.0

        print(f"{Fore.GREEN}[TTS] Kokoro ready ✓{Style.RESET_ALL}")

    # ── Interrupt ─────────────────────────────────────────────────────────────
    def interrupt(self):
        """Stop TTS immediately — called by STT on speech detection."""
        with self._lock:
            if self.is_speaking:
                self.stop_event.set()
                sd.stop()
                self.is_speaking = False

    # ── Simple say ────────────────────────────────────────────────────────────
    def say(self, text: str):
        """Speak a complete text string (blocking)."""
        if not text.strip():
            return
        print(f"{Fore.YELLOW}🤖 Robot: {text}{Style.RESET_ALL}")
        self.stop_event.clear()
        with self._lock:
            self.is_speaking = True
        try:
            self._play_text(text)
        finally:
            with self._lock:
                self.is_speaking = False
            self.last_output_end_ts = time.time()

    # ── Streaming speak ───────────────────────────────────────────────────────
    def speak_stream(self, token_generator: Iterator[str]) -> str:
        """
        Stream tokens from LLM → SentenceBuffer → Kokoro TTS.
        Speaks the first sentence as soon as it's complete (low latency).
        Returns the full response text (for STM logging).
        """
        self.stop_event.clear()
        with self._lock:
            self.is_speaking = True
        buf = SentenceBuffer()
        full_text = ""

        print(f"{Fore.YELLOW}🤖 Robot: ", end="", flush=True)

        try:
            for token in token_generator:
                if self.stop_event.is_set():
                    break

                print(token, end="", flush=True)
                full_text += token

                for sentence in buf.add_token(token):
                    if self.stop_event.is_set():
                        break
                    self._play_text(sentence)

            # Flush remaining buffer
            if not self.stop_event.is_set():
                for sentence in buf.flush():
                    if self.stop_event.is_set():
                        break
                    self._play_text(sentence)

        finally:
            print()  # newline after streaming
            with self._lock:
                self.is_speaking = False
            self.last_output_end_ts = time.time()

        return full_text.strip()

    # ── Internal playback ─────────────────────────────────────────────────────
    def _play_text(self, text: str):
        if not text.strip() or self.stop_event.is_set():
            return

        # Trigger speaking gestures if robot is attached
        if self.robot is not None:
            try:
                import threading
                def _trigger_gesture():
                    t_low = text.lower()
                    # 1. Waving gesture for greetings
                    if any(w in t_low for w in ["hi", "hello", "hey", "welcome", "أهلاً", "مرحباً", "مرحبا", "سلام"]):
                        self.robot.wave_arm()
                    # 2. Thinking head tilt for questions
                    elif any(w in t_low for w in ["why", "how", "what", "check", "لماذا", "كيف", "ماذا", "هل"]):
                        self.robot.set_servos({7: 95})  # Tilt head slightly up/side
                        time.sleep(1.2)
                        self.robot.set_servos({7: 80})  # Return to normal
                    # 3. Simple subtle nod for assertions
                    else:
                        self.robot.set_servos({7: 90})  # Nod head slightly down
                        time.sleep(0.4)
                        self.robot.set_servos({7: 80})  # Return to normal
                threading.Thread(target=_trigger_gesture, daemon=True).start()
            except Exception:
                pass

        try:
            gen = self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r"\n+",
            )
            for _, _, audio in gen:
                if self.stop_event.is_set():
                    sd.stop()
                    return
                sd.play(audio, 24000)
                sd.wait()
        except Exception as e:
            print(f"\n{Fore.RED}[TTS] Playback error: {e}{Style.RESET_ALL}")
