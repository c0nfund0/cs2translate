"""Orchestration: capture -> VAD -> Whisper -> Piper -> playback.

Four threads connected by bounded queues. Every stage re-checks utterance age
and drops stale work, because a callout that arrives after the fight is over is
worse than silence.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import Counter
from dataclasses import dataclass, field

from .asr.whisper import Translation, WhisperTranslator
from .audio.backends import CaptureBackend
from .clock import now
from .config import AppConfig
from .vad.segmenter import Segmenter, Utterance

log = logging.getLogger(__name__)


@dataclass
class Stats:
    utterances: int = 0
    translated: int = 0
    spoken: int = 0
    dropped_stale: int = 0
    dropped_full: int = 0
    rejected: int = 0
    total_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str, amount: float = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def summary(self) -> str:
        avg = self.total_latency_ms / self.spoken if self.spoken else 0.0
        return (
            f"utterances={self.utterances} translated={self.translated} spoken={self.spoken} "
            f"rejected={self.rejected} stale={self.dropped_stale} overflow={self.dropped_full} "
            f"avg_latency={avg:.0f}ms"
        )


class Ducker:
    """Optionally lower every other app's volume while we speak, so the English
    is intelligible over an ongoing firefight."""

    def __init__(self, enabled: bool, level: float) -> None:
        self.level = level
        self._sessions: list = []
        self._saved: dict = {}
        self.enabled = False
        if not enabled:
            return
        import importlib.util

        if importlib.util.find_spec("pycaw") is None:
            log.warning("audio ducking needs pycaw; continuing without it")
        else:
            self.enabled = True

    def _own_pid(self) -> int:
        import os

        return os.getpid()

    def duck(self) -> None:
        if not self.enabled:
            return
        try:
            from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

            self._saved.clear()
            for session in AudioUtilities.GetAllSessions():
                if not session.Process or session.Process.pid == self._own_pid():
                    continue
                vol = session._ctl.QueryInterface(ISimpleAudioVolume)
                current = vol.GetMasterVolume()
                self._saved[session.Process.pid] = (vol, current)
                vol.SetMasterVolume(current * self.level, None)
        except Exception as exc:
            log.debug("duck failed: %s", exc)

    def restore(self) -> None:
        if not self.enabled:
            return
        for vol, level in self._saved.values():
            try:
                vol.SetMasterVolume(level, None)
            except Exception:
                pass
        self._saved.clear()


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        capture: CaptureBackend,
        segmenter: Segmenter,
        translator: WhisperTranslator,
        tts,
        playback,
        gate,
        on_line=None,
    ) -> None:
        self.cfg = cfg
        self.capture = capture
        self.segmenter = segmenter
        self.translator = translator
        self.tts = tts
        self.playback = playback
        self.gate = gate
        self.on_line = on_line
        self.stats = Stats()
        self.ducker = Ducker(cfg.pipeline.duck_game_audio, cfg.pipeline.duck_level)

        self._asr_q: queue.Queue[Utterance] = queue.Queue(cfg.pipeline.asr_queue_size)
        self._tts_q: queue.Queue[Translation] = queue.Queue(cfg.pipeline.tts_queue_size)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.paused = threading.Event()
        # Rolling tally so a silent app can say why, at INFO, without DEBUG.
        self._reasons: Counter[str] = Counter()
        self._last_report = now()

    @property
    def idle(self) -> bool:
        """No work in flight. Used by offline mode to know when to exit."""
        return self._asr_q.empty() and self._tts_q.empty()

    @property
    def max_age(self) -> float:
        return self.cfg.pipeline.max_utterance_age_ms / 1000.0

    def start(self) -> None:
        self.capture.start()
        for target, name in (
            (self._segment_loop, "segment"),
            (self._asr_loop, "asr"),
            (self._tts_loop, "tts"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("pipeline running")

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        for t in self._threads:
            t.join(timeout=3.0)
        self.playback.close()
        self.ducker.restore()
        log.info("stopped. %s", self.stats.summary())

    # -- stage 1: VAD endpointing -----------------------------------------
    def _segment_loop(self) -> None:
        try:
            for block in self.capture.blocks():
                if self._stop.is_set():
                    break
                if self.paused.is_set():
                    continue
                for utt in self.segmenter.push(block):
                    self.stats.bump("utterances")
                    log.debug(
                        "utterance %.2fs (peak p=%.2f)", utt.duration, utt.peak_probability
                    )
                    self._enqueue(self._asr_q, utt, "asr")
        except Exception:
            log.exception("segment loop crashed")

    def _enqueue(self, q: queue.Queue, item, name: str) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            # Prefer the newest speech: discard the oldest queued item and retry.
            try:
                stale = q.get_nowait()
                self.stats.bump("dropped_full")
                log.debug("%s queue full, discarded oldest (age %.1fs)", name, stale.age)
                q.put_nowait(item)
            except queue.Empty:  # pragma: no cover
                pass

    # -- stage 2: Whisper --------------------------------------------------
    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utt = self._asr_q.get(timeout=0.2)
            except queue.Empty:
                # Nothing waiting: a good moment to hand VRAM back to the game.
                unload = getattr(self.translator, "maybe_unload", None)
                if unload is not None:
                    unload()
                continue
            if utt.age > self.max_age:
                self.stats.bump("dropped_stale")
                log.debug("dropping stale utterance (%.1fs old) before ASR", utt.age)
                continue
            try:
                result = self.translator.translate(utt)
            except Exception:
                log.exception("translation failed")
                continue
            if result is None:
                self.stats.bump("rejected")
                reason = getattr(self.translator, "last_reject_reason", None) or "unknown"
                self._reasons[reason] += 1
                self._maybe_report()
                continue
            self._reasons.clear()
            self.stats.bump("translated")
            log.info(
                "[%s p=%.2f %.0fms] %s",
                result.language,
                result.language_probability,
                result.asr_ms,
                result.text,
            )
            if self.on_line:
                self.on_line(result)
            self._enqueue(self._tts_q, result, "tts")

    def _maybe_report(self, every: float = 20.0) -> None:
        """Periodically explain a stretch of nothing-happening.

        Without this the app looks identical whether it is hearing nothing,
        rejecting everything as non-speech, or skipping the language on
        purpose -- and the difference only showed up at DEBUG.
        """
        if now() - self._last_report < every or not self._reasons:
            return
        self._last_report = now()
        detail = ", ".join(f"{n}x {r}" for r, n in self._reasons.most_common(4))
        log.info("nothing spoken in the last %.0fs -- dropped: %s", every, detail)
        self._reasons.clear()

    # -- stage 3: Piper + playback ----------------------------------------
    def _tts_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._tts_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if result.age > self.max_age:
                self.stats.bump("dropped_stale")
                log.debug("dropping stale line (%.1fs old) before TTS", result.age)
                continue
            try:
                audio = self.tts.synthesize(result.text)
            except Exception:
                log.exception("synthesis failed")
                continue
            if len(audio) == 0:
                continue

            # Hold for a gap in incoming speech: the capture gate makes us deaf
            # while playing, so speaking over a live callout loses that callout
            # entirely. Give up after max_gap_wait_ms so we never stall.
            wait = self.cfg.tts.max_gap_wait_ms / 1000.0
            if wait > 0 and not self.segmenter.wait_for_gap(wait):
                log.debug("no speech gap after %.1fs; speaking anyway", wait)
            if result.age > self.max_age:
                self.stats.bump("dropped_stale")
                continue

            self.gate.begin_playback()
            self.ducker.duck()
            try:
                self.playback.play(audio, self.tts.sample_rate)
            except Exception:
                log.exception("playback failed")
            finally:
                self.ducker.restore()
                self.gate.end_playback()
                # Audio captured while gated was zeroed, so the segmenter's VAD
                # state is stale; reset it rather than let it drift.
                self.segmenter.reset()

            latency = (now() - result.utterance.captured_at) * 1000
            self.stats.bump("spoken")
            self.stats.bump("total_latency_ms", latency)
            log.debug("spoke after %.0fms end-to-end", latency)
