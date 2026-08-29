"""faster-whisper wrapper: language ID + speech->English in one pass."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..clock import now
from ..config import AsrConfig
from ..vad.segmenter import Utterance
from . import filters

log = logging.getLogger(__name__)


@dataclass
class Translation:
    text: str
    language: str
    language_probability: float
    utterance: Utterance
    asr_ms: float

    @property
    def age(self) -> float:
        return self.utterance.age


def _free_vram_mib() -> int | None:
    """Free VRAM on device 0, or None if we cannot tell."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.free) // (1024 * 1024)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        log.debug("VRAM query failed: %s", exc)
        return None


def choose_compute_type(cfg: AsrConfig) -> str:
    """large-v3 at float16 is ~3GB. On an 8-10GB card that fits, but CS2's own
    allocation moves around, so check free VRAM at load time and step down to
    int8_float16 (~1.6GB) rather than risking an OOM mid-match."""
    if cfg.compute_type != "auto":
        return cfg.compute_type
    if cfg.device != "cuda":
        return "int8"
    free = _free_vram_mib()
    if free is None:
        log.warning("could not read free VRAM; assuming float16 is safe")
        return "float16"
    if free < cfg.vram_guard_mib:
        log.warning(
            "only %d MiB VRAM free (< %d); using int8_float16 instead of float16",
            free,
            cfg.vram_guard_mib,
        )
        return "int8_float16"
    log.info("%d MiB VRAM free; using float16", free)
    return "float16"


class WhisperTranslator:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.compute_type = choose_compute_type(cfg)
        self.model = self._load(self.compute_type)
        # Why the last utterance was dropped. The pipeline aggregates these so
        # a silent app can explain itself without needing DEBUG.
        self.last_reject_reason: str | None = None

    def _load(self, compute_type: str):
        from faster_whisper import WhisperModel

        kwargs = {"device": self.cfg.device, "compute_type": compute_type}
        if self.cfg.download_root:
            kwargs["download_root"] = str(Path(self.cfg.download_root))
        log.info("loading %s (%s, %s)", self.cfg.model, self.cfg.device, compute_type)
        t0 = now()
        try:
            model = WhisperModel(self.cfg.model, **kwargs)
        except Exception as exc:
            if compute_type == "float16" and self.cfg.device == "cuda":
                log.warning("float16 load failed (%s); retrying as int8_float16", exc)
                self.compute_type = "int8_float16"
                kwargs["compute_type"] = "int8_float16"
                model = WhisperModel(self.cfg.model, **kwargs)
            else:
                raise
        log.info("model ready in %.1fs", now() - t0)
        return model

    def warmup(self) -> None:
        """First inference pays CUDA kernel autotuning; do it before the match
        rather than on the first real callout."""
        silence = np.zeros(16000, dtype=np.float32)
        try:
            list(self.model.transcribe(silence, task=self.cfg.task, beam_size=1)[0])
            log.info("warmup complete")
        except Exception as exc:  # pragma: no cover
            log.warning("warmup failed: %s", exc)

    def translate(self, utt: Utterance) -> Translation | None:
        t0 = now()
        segments, info = self.model.transcribe(
            utt.audio,
            task=self.cfg.task,
            beam_size=self.cfg.beam_size,
            language=None,           # auto-detect per utterance
            vad_filter=False,        # our Segmenter already endpointed this
            without_timestamps=True,
            condition_on_previous_text=False,  # utterances are unrelated speakers
            temperature=0.0,
            no_speech_threshold=self.cfg.no_speech_threshold,
        )
        segments = list(segments)
        asr_ms = (now() - t0) * 1000
        self.last_reject_reason = None

        if not segments:
            self.last_reject_reason = "no-segments"
            log.debug("no segments (%.0fms)", asr_ms)
            return None

        text = " ".join(s.text.strip() for s in segments).strip()
        no_speech = min(getattr(s, "no_speech_prob", 0.0) for s in segments)
        compression = max(getattr(s, "compression_ratio", 0.0) for s in segments)

        reason = filters.should_reject(
            text,
            no_speech,
            compression,
            no_speech_threshold=self.cfg.no_speech_threshold,
            max_compression_ratio=self.cfg.max_compression_ratio,
        )
        if reason:
            self.last_reject_reason = reason.split("=")[0]
            log.debug("rejected (%s): %r", reason, text)
            return None

        lang = (info.language or "??").lower()
        lang_p = float(info.language_probability or 0.0)
        if lang_p < self.cfg.min_language_probability:
            self.last_reject_reason = "low-language-confidence"
            log.debug("rejected (language %s p=%.2f): %r", lang, lang_p, text)
            return None
        if lang in self.cfg.skip_languages:
            self.last_reject_reason = f"skipped-language:{lang}"
            log.info("[%s] (already this language, not spoken) %s", lang, text)
            return None

        return Translation(
            text=text,
            language=lang,
            language_probability=lang_p,
            utterance=utt,
            asr_ms=asr_ms,
        )
