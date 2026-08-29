"""Silero VAD wrapper.

Cover for the bug that made the app do nothing at all: v5's ONNX takes the
previous window's tail prepended as context (576 samples at 16 kHz, not 512).
The graph declares `input` as [None, None], so a bare 512-sample window is
accepted silently and returns ~0.001 for every frame -- a VAD that never fires,
with no error anywhere.

Nothing caught this before because the unit tests used a scripted fake VAD and
the smoke tests used the energy fallback, so Silero was never once run on real
audio.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from cs2translate.vad.silero import CONTEXT_SAMPLES, FRAME_SAMPLES, SileroVAD, ensure_model

FIXTURE = Path(__file__).parent / "data" / "speech_16k.wav"


class _Named:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Records what actually reaches the graph."""

    def __init__(self, *_a, **_k):
        self.inputs_seen: list[np.ndarray] = []

    def get_inputs(self):
        return [_Named("input"), _Named("state"), _Named("sr")]

    def run(self, _outputs, feeds):
        self.inputs_seen.append(feeds["input"].copy())
        return [np.array([[0.9]], dtype=np.float32), feeds["state"]]


@pytest.fixture
def fake_vad(monkeypatch):
    import onnxruntime as ort

    created = {}

    def factory(*a, **k):
        created["session"] = FakeSession()
        return created["session"]

    monkeypatch.setattr(ort, "InferenceSession", factory)
    vad = SileroVAD(Path("unused.onnx"), 16000)
    return vad, created["session"]


def test_window_is_padded_with_context(fake_vad):
    """576 = 64 context + 512 window. A bare 512 silently returns ~0."""
    vad, session = fake_vad
    vad(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert session.inputs_seen[0].shape == (1, CONTEXT_SAMPLES[16000] + FRAME_SAMPLES)


def test_context_carries_the_previous_window_tail(fake_vad):
    vad, session = fake_vad
    first = np.arange(FRAME_SAMPLES, dtype=np.float32)
    second = np.full(FRAME_SAMPLES, -1.0, dtype=np.float32)
    vad(first)
    vad(second)
    ctx = CONTEXT_SAMPLES[16000]
    assert np.allclose(session.inputs_seen[0][0, :ctx], 0.0), "first call starts with zero context"
    assert np.allclose(session.inputs_seen[1][0, :ctx], first[-ctx:]), "context must be the tail"
    assert np.allclose(session.inputs_seen[1][0, ctx:], second)


def test_reset_clears_context(fake_vad):
    vad, session = fake_vad
    vad(np.full(FRAME_SAMPLES, 0.5, dtype=np.float32))
    vad.reset()
    vad(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert np.allclose(session.inputs_seen[-1], 0.0)


def test_rejects_wrong_frame_size(fake_vad):
    vad, _ = fake_vad
    with pytest.raises(ValueError):
        vad(np.zeros(256, dtype=np.float32))


# --- against the real model -------------------------------------------------

def _real_vad(tmp_path):
    try:
        return SileroVAD(ensure_model(tmp_path, None), 16000)
    except Exception as exc:  # network unavailable
        pytest.skip(f"Silero model unavailable: {exc}")


def _fixture_audio():
    with wave.open(str(FIXTURE), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _probs(vad, audio):
    return np.array(
        [vad(audio[i : i + FRAME_SAMPLES]) for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES)]
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason="speech fixture missing")
def test_detects_real_speech_at_a_realistic_level(tmp_path):
    """The fixture sits at ~-27 dBFS, the level measured on a real 7.1 headset
    loopback -- not a loud studio sample."""
    probs = _probs(_real_vad(tmp_path), _fixture_audio())
    assert probs.max() > 0.9, f"peak probability only {probs.max():.3f}"
    assert (probs > 0.6).sum() >= 10, f"only {(probs > 0.6).sum()} confident frames"


@pytest.mark.skipif(not FIXTURE.exists(), reason="speech fixture missing")
def test_digital_silence_scores_low(tmp_path):
    probs = _probs(_real_vad(tmp_path), np.zeros(16000, dtype=np.float32))
    assert probs.max() < 0.3, f"silence scored {probs.max():.3f}"
