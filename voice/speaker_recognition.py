import os
import torch
import numpy as np
from colorama import Fore, Style

import config.settings as cfg
from core.device import pick_torch_device

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        EncoderClassifier = None

class SpeakerRecognizer:
    def __init__(self):
        print(f"{Fore.CYAN}[Voice] Loading Speaker Recognition model (SpeechBrain)...{Style.RESET_ALL}")
        if EncoderClassifier is None:
            print(
                f"{Fore.RED}[Voice] SpeechBrain not installed. Speaker verification disabled.{Style.RESET_ALL}\n"
                "Install it with: pip install git+https://github.com/speechbrain/speechbrain.git@develop"
            )
            self.classifier = None
            return

        self.min_seconds = float(getattr(cfg, "VOICE_EMBEDDING_MIN_SECONDS", 0.75))
        device = pick_torch_device()
        save_dir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")

        try:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=save_dir,
                run_opts={"device": device},
            )
            print(f"{Fore.GREEN}[Voice] Speaker Recognition ready on {device}{Style.RESET_ALL}")
        except Exception as e:
            if device == "cuda":
                print(
                    f"{Fore.YELLOW}[Voice] SpeechBrain failed to load on CUDA: {e}{Style.RESET_ALL}"
                )
                try:
                    self.classifier = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir=save_dir,
                        run_opts={"device": "cpu"},
                    )
                    print(
                        f"{Fore.GREEN}[Voice] Speaker Recognition loaded on cpu after CUDA fallback.{Style.RESET_ALL}"
                    )
                except Exception as fallback_err:
                    print(
                        f"{Fore.RED}[Voice] Failed to load Speaker Recognition on cpu: {fallback_err}{Style.RESET_ALL}"
                    )
                    self.classifier = None
            else:
                print(f"{Fore.RED}[Voice] Failed to load Speaker Recognition model: {e}{Style.RESET_ALL}")
                self.classifier = None

    def get_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        """
        Extract 192D voice embedding from a 1D numpy array (16kHz).
        Returns None if model not loaded or audio too short.
        """
        if self.classifier is None:
            return None

        duration = len(audio_np) / 16000.0
        if duration < self.min_seconds:
            return None

        try:
            # Peak amplitude normalization for volume/scaling consistency
            audio = audio_np.copy()
            mean = np.mean(audio)
            audio = audio - mean
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            signal = torch.from_numpy(audio).float().unsqueeze(0)
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(signal)
            return embeddings.squeeze().cpu().numpy()
        except Exception as e:
            print(f"[Voice] Error extracting voice embedding: {e}")
            return None
