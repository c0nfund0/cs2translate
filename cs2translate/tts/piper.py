"""Piper TTS.

Deliberately CPU-only: the GPU is fully committed to large-v3, and Piper
synthesizes a short callout in well under 100ms on any modern CPU, so moving it
to CUDA would buy nothing while competing for VRAM.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np

from ..config import TtsConfig

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def parse_voice_name(voice: str) -> tuple[str, str, str, str]:
    """'en_US-lessac-medium' -> ('en', 'en_US', 'lessac', 'medium')"""
    try:
        locale, name, quality = voice.split("-", 2)
        lang = locale.split("_")[0]
        return lang, locale, name, quality
    except ValueError as exc:
        raise ValueError(f"malformed piper voice name: {voice!r}") from exc


def ensure_voice(voice: str, voices_dir: Path) -> Path:
    """Download the .onnx and its .json config if not already cached."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    model = voices_dir / f"{voice}.onnx"
    config = voices_dir / f"{voice}.onnx.json"
    if model.exists() and config.exists():
        return model

    lang, locale, name, quality = parse_voice_name(voice)
    base = f"{HF_BASE}/{lang}/{locale}/{name}/{quality}/{voice}.onnx"
    for url, dest in ((base, model), (base + ".json", config)):
        if dest.exists():
            continue
        log.info("downloading piper voice: %s", dest.name)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return model


class PiperTTS:
    """Wraps both the modern (`synthesize`) and legacy
    (`synthesize_stream_raw`) piper-tts APIs, which differ across releases."""

    def __init__(self, cfg: TtsConfig, cache_dir: Path) -> None:
        from piper import PiperVoice

        voices_dir = Path(cfg.voices_dir) if cfg.voices_dir else cache_dir / "piper"
        model_path = ensure_voice(cfg.voice, voices_dir)
        self.cfg = cfg
        self.voice = PiperVoice.load(str(model_path), use_cuda=cfg.use_cuda)
        self.sample_rate = self._detect_sample_rate(model_path)
        self._modern = hasattr(self.voice, "synthesize") and not hasattr(
            self.voice, "synthesize_stream_raw"
        )
        log.info("piper voice %s ready (%d Hz)", cfg.voice, self.sample_rate)

    def _detect_sample_rate(self, model_path: Path) -> int:
        cfg = getattr(self.voice, "config", None)
        rate = getattr(cfg, "sample_rate", None)
        if rate:
            return int(rate)
        with open(str(model_path) + ".json", encoding="utf-8") as fh:
            return int(json.load(fh)["audio"]["sample_rate"])

    def synthesize(self, text: str) -> np.ndarray:
        """Returns mono float32 at self.sample_rate."""
        if hasattr(self.voice, "synthesize_stream_raw"):
            return self._legacy(text)
        return self._modern_synth(text)

    def _legacy(self, text: str) -> np.ndarray:
        chunks = [
            np.frombuffer(b, dtype=np.int16)
            for b in self.voice.synthesize_stream_raw(
                text,
                length_scale=self.cfg.length_scale,
                noise_scale=self.cfg.noise_scale,
                noise_w=self.cfg.noise_w,
            )
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def _modern_synth(self, text: str) -> np.ndarray:
        syn_config = None
        try:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(
                length_scale=self.cfg.length_scale,
                noise_scale=self.cfg.noise_scale,
                noise_w_scale=self.cfg.noise_w,
            )
        except Exception:
            pass

        out: list[np.ndarray] = []
        stream = (
            self.voice.synthesize(text, syn_config=syn_config)
            if syn_config is not None
            else self.voice.synthesize(text)
        )
        for chunk in stream:
            arr = getattr(chunk, "audio_float_array", None)
            if arr is None:
                raw = getattr(chunk, "audio_int16_bytes", None)
                if raw is None:
                    continue
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            out.append(np.asarray(arr, dtype=np.float32))
            rate = getattr(chunk, "sample_rate", None)
            if rate:
                self.sample_rate = int(rate)
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out)


class NullTTS:
    """Silent stand-in for offline pipeline tests."""

    sample_rate = 22050

    def synthesize(self, text: str) -> np.ndarray:
        # Roughly 14 characters per second of speech, so gate timing in tests
        # resembles the real thing.
        return np.zeros(int(self.sample_rate * len(text) / 14.0), dtype=np.float32)
