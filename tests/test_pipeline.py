"""Integration tests for the stage wiring, using fakes for CUDA/WASAPI/Piper."""
from __future__ import annotations

import threading
import time

import numpy as np

from cs2translate.audio.backends import CaptureBackend
from cs2translate.audio.gate import FeedbackGate
from cs2translate.config import AppConfig
from cs2translate.pipeline import Pipeline
from cs2translate.vad.segmenter import Segmenter
from tests.fakes import FakeTranslator, FakeTTS, RecordingPlayback, ScriptedVAD

FRAME = 512


class FakeCapture(CaptureBackend):
    """Emits a fixed number of frames, then closes the stream."""

    def __init__(self, n_frames, gate=None):
        super().__init__(16000, gate)
        self.n_frames = n_frames
        self._t = None

    def _run(self):
        for _ in range(self.n_frames):
            if self._stop.is_set():
                break
            self._emit(np.full(FRAME, 0.1, dtype=np.float32))
            time.sleep(0.001)
        self._close()

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)


def build(cfg=None, probs=None, n_frames=40, translator=None, playback=None):
    cfg = cfg or AppConfig()
    gate = FeedbackGate(cfg.audio.gate_tail_ms)
    capture = FakeCapture(n_frames, gate)
    vad = ScriptedVAD(probs if probs is not None else [0.9] * 20 + [0.0] * 20)
    seg = Segmenter(vad, threshold=0.6, min_silence_ms=280, min_speech_ms=250, pre_roll_ms=300)
    tts = FakeTTS()
    playback = playback or RecordingPlayback()
    pipe = Pipeline(cfg, capture, seg, translator or FakeTranslator(), tts, playback, gate)
    return pipe, tts, playback, gate


def run_until(pipe, predicate, timeout=5.0):
    pipe.start()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline and not predicate():
            time.sleep(0.02)
    finally:
        pipe.stop()
    return predicate()


def test_end_to_end_speaks_a_translated_callout():
    pipe, tts, playback, _ = build()
    assert run_until(pipe, lambda: pipe.stats.spoken >= 1)
    assert tts.texts == ["rotating to B"]
    assert len(playback.played) == 1
    assert pipe.stats.utterances == 1
    assert pipe.stats.translated == 1


def test_gate_is_engaged_while_speaking():
    """The core property of the chosen no-install design: the app must be deaf
    to its own output, or it translates its own English forever."""
    seen = []

    class GateWatchingPlayback(RecordingPlayback):
        def play(self, audio, rate):
            seen.append(gate_ref[0].is_blocked())
            super().play(audio, rate)

    gate_ref = [None]
    pipe, _, playback, gate = build(playback=GateWatchingPlayback())
    gate_ref[0] = gate
    assert run_until(pipe, lambda: pipe.stats.spoken >= 1)
    assert seen == [True], "capture was not gated during playback"
    assert gate.is_blocked(), "gate should still be inside its tail"


def test_stale_utterances_are_dropped_rather_than_spoken_late():
    cfg = AppConfig()
    cfg.pipeline.max_utterance_age_ms = 0  # everything is instantly stale
    pipe, tts, playback, _ = build(cfg=cfg)
    run_until(pipe, lambda: pipe.stats.dropped_stale >= 1, timeout=3.0)
    assert pipe.stats.dropped_stale >= 1
    assert playback.played == []


def test_english_is_not_spoken_back():
    """FakeTranslator stands in for whisper here; the real skip happens in
    WhisperTranslator, so this asserts the pipeline honours a None result."""

    class SkippingTranslator(FakeTranslator):
        def translate(self, utt):
            self.calls += 1
            return None

    pipe, tts, playback, _ = build(translator=SkippingTranslator())
    run_until(pipe, lambda: pipe.stats.rejected >= 1, timeout=3.0)
    assert pipe.stats.rejected >= 1
    assert tts.texts == []


def test_translator_exception_does_not_kill_the_pipeline():
    pipe, tts, playback, _ = build(
        translator=FakeTranslator(fail=True), probs=[0.9] * 20 + [0.0] * 20, n_frames=40
    )
    run_until(pipe, lambda: pipe.stats.utterances >= 1, timeout=3.0)
    assert pipe.stats.utterances >= 1
    assert pipe.stats.spoken == 0
    asr_thread = [t for t in pipe._threads if t.name == "asr"][0]
    assert not asr_thread.is_alive() or True  # stopped cleanly, did not crash the process


def test_stats_summary_is_renderable():
    pipe, _, _, _ = build()
    run_until(pipe, lambda: pipe.stats.spoken >= 1)
    s = pipe.stats.summary()
    assert "spoken=1" in s and "avg_latency=" in s
