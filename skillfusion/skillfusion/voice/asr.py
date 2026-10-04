"""Speech-to-text with NVIDIA open models (NeMo). Lazy import so the rest works without NeMo.

Default: Nemotron 3.5 ASR streaming (0.6B, small enough to share the 10 GB GPU with the sim).
Fallback: Parakeet. Verify exact model ids on Hugging Face / NGC for your NeMo version.
"""
from __future__ import annotations

import os
import tempfile
import wave

import numpy as np

MODELS = ("nvidia/nemotron-3.5-asr-streaming-0.6b", "nvidia/parakeet-tdt-0.6b-v2")


def write_wav(path: str, pcm16: np.ndarray, rate: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.asarray(pcm16, dtype="<i2").tobytes())


class ASR:
    def __init__(self, model: str | None = None, device: str = "cuda"):
        import nemo.collections.asr as nemo_asr
        names = [model or os.environ.get("ASR_MODEL", MODELS[0])] + [m for m in MODELS[1:] if m != model]
        err = None
        for n in names:
            try:
                self.model = nemo_asr.models.ASRModel.from_pretrained(n).to(device).eval()
                self.name = n
                return
            except Exception as e:      # noqa: BLE001 - try next model id
                err = e
        raise RuntimeError(f"no ASR model could be loaded: {err}")

    def transcribe_pcm16(self, pcm16: np.ndarray, rate: int = 16000) -> str:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "u.wav")
            write_wav(p, pcm16, rate)
            out = self.model.transcribe([p])
        first = out[0] if isinstance(out, (list, tuple)) else out
        return getattr(first, "text", first if isinstance(first, str) else str(first)).strip()
