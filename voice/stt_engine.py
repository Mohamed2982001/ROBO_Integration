"""
voice/stt_engine.py
Always-on Speech-to-Text engine using PyAudio and faster-whisper.
Includes NVIDIA DLL fix for Windows CUDA environments.
"""

import os
import sys
import threading
import queue
import time
import logging
import numpy as np
import pyaudio
from typing import Optional, Callable
from faster_whisper import WhisperModel
from colorama import Fore, Style, init

import config.settings as cfg
from brain.locale import is_repetitive_garbage
from core.device import pick_torch_device, whisper_compute_type
from voice.speaker_recognition import SpeakerRecognizer

init(autoreset=True)
logger = logging.getLogger("STT")

# --- NVIDIA DLL FIX ---
try:
    import nvidia

    _nvidia_base = list(nvidia.__path__)[0]
    dll_paths = []
    for _pkg in os.listdir(_nvidia_base):
        _bin = os.path.join(_nvidia_base, _pkg, "bin")
        if os.path.isdir(_bin):
            os.add_dll_directory(_bin)
            dll_paths.append(_bin)
    if dll_paths:
        os.environ["PATH"] = ";".join(dll_paths) + ";" + os.environ.get("PATH", "")
except Exception as e:
    # Quietly proceed if nvidia-libs aren't present or handled
    pass


class HearingEngine:
    """
    STT Engine compatible with the Robot's main interaction loop.
    Uses PyAudio for capture and faster-whisper for transcription.
    """

    def __init__(self, event_queue, tts_engine=None, speech_activity=None):
        self.event_queue = event_queue
        self.tts_engine = tts_engine
        self.speech_activity = speech_activity
        self.running = False

        # --- SETTINGS ---
        self.SAMPLE_RATE = 16000
        self.CHUNK = 1024
        self.SILENCE_THRESHOLD = float(getattr(cfg, "SILENCE_THRESHOLD", 0.02))
        self.BARGE_IN_RMS = float(getattr(cfg, "BARGE_IN_RMS", 0.065))
        self.SILENCE_DURATION = getattr(cfg, "SILENCE_DURATION", 0.8)
        self.MAX_CHUNK_DURATION = getattr(cfg, "MAX_CHUNK_DURATION", 30.0)
        self.POST_TTS_COOLDOWN_SECONDS = float(
            getattr(cfg, "POST_TTS_COOLDOWN_SECONDS", 0.65)
        )
        self.STARTUP_GRACE_SECONDS = 0.5

        model_size = getattr(cfg, "WHISPER_MODEL_SIZE", "small")
        print(
            f"{Fore.CYAN}[STT] Loading faster-whisper ({model_size})...{Style.RESET_ALL}"
        )
        self.model = self._load_model(model_size=model_size)
        self._initial_prompt = self._compose_initial_prompt()

        self.speaker_recognizer = SpeakerRecognizer()

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue()

    def _load_model(self, model_size: str) -> WhisperModel:
        """Load Whisper on GPU when available (ROBOT_DEVICE=auto|cuda|cpu)."""
        device = pick_torch_device()
        compute_type = whisper_compute_type(device)

        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(
                f"{Fore.GREEN}[STT] Whisper ready on {device} ({compute_type}) ✓{Style.RESET_ALL}"
            )
            return model
        except Exception as e:
            logger.warning(
                f"Failed to load Whisper on {device}: {e}. Falling back to CPU/int8."
            )
            return WhisperModel(model_size, device="cpu", compute_type="int8")

    def _compose_initial_prompt(self) -> str:
        """Biases recognition toward names and domain terms without hard-coding the full lexicon."""
        chunks = [
            "Mahmoud, Abdullah, Ola, Egypt, Musa, Robot, AI, Hello, Hi, My name is",
        ]
        vf = getattr(cfg, "WHISPER_VOCAB_FILE", "") or ""
        if vf.strip() and os.path.isfile(vf.strip()):
            try:
                with open(vf.strip(), encoding="utf-8", errors="ignore") as fh:
                    extra = " ".join(fh.read().split())
                if extra:
                    chunks.append(extra[:3500])
            except OSError as e:
                logger.warning("WHISPER_VOCAB_FILE read failed: %s", e)
        return ", ".join(c for c in chunks if c)

    def start(self):
        self.running = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="STT_Listen"
        )
        self._listen_thread.start()
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="STT_Process"
        )
        self._process_thread.start()
        print(f"{Fore.GREEN}[STT] Always-on listening started ✓{Style.RESET_ALL}")

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        self.audio.terminate()


    def _listen_loop(self):
        """Captures raw audio from mic."""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            while self.running:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self._audio_queue.put(data)
        except Exception as e:
            logger.error(f"Mic stream error: {e}")

    def _transcribe_chunk(self, audio_np: np.ndarray) -> str:
        """Transcribe buffer to text, filtering hallucinations from silence."""
        try:
            lang_raw = (getattr(cfg, "WHISPER_LANGUAGE", "en") or "").strip().lower()
            language = None if lang_raw in ("", "auto", "detect") else lang_raw
            beam = int(getattr(cfg, "WHISPER_BEAM_SIZE", 5))
            vad_ms = int(getattr(cfg, "WHISPER_VAD_MIN_SIL_MS", 450))
            segments, _ = self.model.transcribe(
                audio_np,
                beam_size=max(1, beam),
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": max(100, vad_ms)},
                temperature=0.0,
                initial_prompt=self._initial_prompt,
                language=language,
                task=getattr(cfg, "WHISPER_TASK", "transcribe"),
                compression_ratio_threshold=float(
                    getattr(cfg, "WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4)
                ),
                log_prob_threshold=float(
                    getattr(cfg, "WHISPER_LOG_PROB_THRESHOLD", -1.0)
                ),
                no_speech_threshold=float(
                    getattr(cfg, "WHISPER_NO_SPEECH_THRESHOLD", 0.6)
                ),
            )
            segments = list(segments)
            if segments:
                text = " ".join(s.text for s in segments).strip()
                if is_repetitive_garbage(text):
                    return ""
                if len(text) > 2:
                    words = text.split()
                    # Reject very short (1-2 word) transcriptions — almost always noise hallucinations
                    if len(words) < 3:
                        lower = text.lower().strip(" .,!?")
                        legit_short = {
                            "hello", "hey", "hi", "yes", "no", "stop", "help",
                            "what", "why", "how", "yeah", "yep", "nope", "correct",
                            "right", "sure", "okay", "ok", "perfect", "absolutely",
                        }
                        confirm_phrases = (
                            "yes that's", "yeah that's", "that's right", "that's correct",
                            "sounds good", "you got it",
                        )
                        intro_prefixes = (
                            "i'm ", "im ", "my name is ", "name is ", "call me ",
                            "this is ", "it's ", "its ",
                        )
                        if lower in legit_short:
                            pass
                        elif any(lower.startswith(p) for p in intro_prefixes):
                            pass
                        elif len(words) == 2 and words[0].lower() in ("i'm", "im", "name's"):
                            pass
                        elif len(words) == 1 and len(lower) >= 2 and lower.isalpha():
                            pass
                        elif any(lower.startswith(p) for p in confirm_phrases):
                            pass
                        else:
                            return ""
                    # Reject common Whisper false positives from ambient noise
                    hallucination_phrases = {
                        "thank you", "thank", "thanks", "okay", "ok", "um", "uh", "hmm",
                        "you", "bye", "the end", "thanks for watching",
                        "subscribe", "like and subscribe",
                    }
                    if text.lower().strip(" .,!?") in hallucination_phrases:
                        return ""
                    # Reject if mostly filler/punctuation
                    alpha_count = sum(1 for c in text if c.isalpha())
                    if alpha_count < len(text) * 0.3:  # Less than 30% alphabetic
                        return ""
                    # Reject low-confidence segments (avg_logprob < -1.0 is very uncertain)
                    avg_confidence = sum(s.avg_logprob for s in segments) / len(segments)
                    conf_floor = float(
                        getattr(cfg, "WHISPER_AVG_LOGPROB_MIN", -0.85)
                    )
                    if avg_confidence < conf_floor:
                        return ""
                    return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        return ""

    def _process_loop(self):
        """Accumulate utterances; discard low-RMS bleed during robot speech and post-TTS cooldown."""
        current_audio = np.array([], dtype=np.float32)
        accumulated_text: str = ""
        silence_start_time: Optional[float] = None
        has_speech = False
        barge_in_active = False
        startup_grace = time.time() + self.STARTUP_GRACE_SECONDS

        print(f"{Fore.CYAN}[STT] Waiting for speech...{Style.RESET_ALL}")

        while self.running:
            try:
                raw = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if time.time() < startup_grace:
                continue

            chunk_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            volume = float(np.sqrt(np.mean(chunk_np**2)))

            tts_eng = self.tts_engine
            speaking = bool(tts_eng and getattr(tts_eng, "is_speaking", False))
            last_end = float(getattr(tts_eng, "last_output_end_ts", 0.0) or 0.0)
            now = time.time()
            in_cooldown = (
                bool(tts_eng)
                and last_end > 0.0
                and not speaking
                and (now - last_end) < self.POST_TTS_COOLDOWN_SECONDS
            )

            if speaking:
                if volume >= self.BARGE_IN_RMS:
                    # User clears TTS immediately; fresh capture for this utterance.
                    if tts_eng:
                        tts_eng.interrupt()
                    barge_in_active = True
                    current_audio = np.array([], dtype=np.float32)
                    accumulated_text = ""
                    silence_start_time = None
                    has_speech = False
                else:
                    # Speaker bleed while robot talks — skip chunk entirely.
                    continue
            elif in_cooldown and not barge_in_active:
                if volume >= self.BARGE_IN_RMS:
                    current_audio = np.array([], dtype=np.float32)
                    accumulated_text = ""
                    silence_start_time = None
                    has_speech = False
                else:
                    continue

            current_audio = np.concatenate((current_audio, chunk_np))
            current_duration = len(current_audio) / self.SAMPLE_RATE

            if volume >= self.SILENCE_THRESHOLD:
                silence_start_time = None
                has_speech = True
                if self.speech_activity:
                    self.speech_activity.mark_speech_chunk()
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                # PREVENT MASSIVE SILENCE ACCUMULATION (Hallucination fix)
                # If we don't have speech yet, we only keep the last 1.0 second of audio as context
                if not has_speech:
                    max_frames = int(self.SAMPLE_RATE * 1.0)
                    if len(current_audio) > max_frames:
                        current_audio = current_audio[-max_frames:]
                    current_duration = len(current_audio) / self.SAMPLE_RATE

            # CASE A: MAX DURATION REACHED (Stitch)
            if current_duration > self.MAX_CHUNK_DURATION:
                print(
                    f"{Fore.MAGENTA}[STT] 🔄 Stitching chunk ({current_duration:.1f}s){Style.RESET_ALL}"
                )
                chunk_text = self._transcribe_chunk(current_audio)
                if chunk_text:
                    accumulated_text += " " + chunk_text
                current_audio = np.array([], dtype=np.float32)
                silence_start_time = None
                has_speech = False

            # CASE B: FINALIZED (Silence timeout)
            elif silence_start_time and (
                time.time() - silence_start_time > self.SILENCE_DURATION
            ):
                if len(current_audio) > self.SAMPLE_RATE * 1.0 and has_speech:
                    last_text = self._transcribe_chunk(current_audio)
                    if last_text:
                        accumulated_text += " " + last_text

                final_output = accumulated_text.strip()
                if final_output:
                    print(f"{Fore.GREEN}[STT] 🗣 Heard: {final_output}{Style.RESET_ALL}")

                    if self.speech_activity:
                        self.speech_activity.finalize_utterance_speaker()

                    if self.speaker_recognizer.classifier is None:
                        print(
                            f"{Fore.YELLOW}[VoiceID] Speaker verification unavailable. Install SpeechBrain to enable voice identification.{Style.RESET_ALL}"
                        )

                    voice_emb = None
                    try:
                        voice_emb = self.speaker_recognizer.get_embedding(current_audio)
                        if voice_emb is None:
                            sample_duration = len(current_audio) / self.SAMPLE_RATE
                            min_seconds = float(getattr(cfg, "VOICE_EMBEDDING_MIN_SECONDS", 0.75))
                            if sample_duration < min_seconds:
                                print(
                                    f"{Fore.YELLOW}[VoiceID] Voice sample too short for verification ({sample_duration:.2f}s). "
                                    f"Need at least {min_seconds:.2f}s of speech.{Style.RESET_ALL}"
                                )
                            else:
                                print(
                                    f"{Fore.YELLOW}[VoiceID] Voice embedding unavailable or model failed to encode the sample.{Style.RESET_ALL}"
                                )
                    except Exception as e:
                        print(f"[VoiceID] Embedding extraction error: {e}")

                    self.event_queue.push_speech(final_output, voice_embedding=voice_emb)

                current_audio = np.array([], dtype=np.float32)
                accumulated_text = ""
                silence_start_time = None
                has_speech = False
                barge_in_active = False
