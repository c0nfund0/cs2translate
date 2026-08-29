"""Keeping out of the game's way.

The app shares a machine with CS2. large-v3 at fp16 holds ~3GB of VRAM, and a
card that is merely adequate for the game alone starts evicting textures when
another 3GB disappears -- which shows up as a sustained framerate collapse, not
as occasional hitches.
"""
from __future__ import annotations

import pytest

import os

from cs2translate.asr.whisper import WhisperTranslator, choose_cpu_threads
from cs2translate.config import AsrConfig
from cs2translate.priority import _CLASSES, limit_math_threads, set_process_priority


class FakeCT2:
    """Stands in for ctranslate2.models.Whisper."""

    def __init__(self):
        self.resident = True
        self.unloads = 0
        self.loads = 0

    def unload_model(self, to_cpu=False):
        self.resident = False
        self.unloads += 1

    def load_model(self):
        self.resident = True
        self.loads += 1


class FakeWhisper:
    def __init__(self):
        self.model = FakeCT2()


@pytest.fixture
def translator(monkeypatch):
    monkeypatch.setattr(WhisperTranslator, "_load", lambda self, ct: FakeWhisper())
    monkeypatch.setattr("cs2translate.asr.whisper.choose_compute_type", lambda cfg: "float16")
    cfg = AsrConfig(idle_unload_s=0.05)
    return WhisperTranslator(cfg)


def test_disabled_by_default():
    """Unloading costs latency on the next callout, so it is opt-in."""
    assert AsrConfig().idle_unload_s == 0.0


def test_does_not_unload_before_the_idle_window(translator):
    assert translator.maybe_unload() is False
    assert translator.model.model.resident


def test_unloads_after_the_idle_window(translator):
    import time

    time.sleep(0.06)
    assert translator.maybe_unload() is True
    assert not translator.model.model.resident
    assert translator.model.model.unloads == 1


def test_unload_is_not_repeated_while_already_unloaded(translator):
    import time

    time.sleep(0.06)
    translator.maybe_unload()
    assert translator.maybe_unload() is False
    assert translator.model.model.unloads == 1


def test_ensure_resident_brings_it_back(translator):
    import time

    time.sleep(0.06)
    translator.maybe_unload()
    translator.ensure_resident()
    assert translator.model.model.resident
    assert translator.model.model.loads == 1


def test_no_unload_when_the_backend_cannot_do_it(monkeypatch):
    """Older faster-whisper builds expose no unload; must degrade quietly."""

    class Bare:
        pass

    monkeypatch.setattr(WhisperTranslator, "_load", lambda self, ct: Bare())
    monkeypatch.setattr("cs2translate.asr.whisper.choose_compute_type", lambda cfg: "float16")
    t = WhisperTranslator(AsrConfig(idle_unload_s=0.0001))
    import time

    time.sleep(0.01)
    assert t.maybe_unload() is False


def test_cpu_thread_default_is_auto():
    assert AsrConfig().cpu_threads == 0


def test_cuda_keeps_the_thread_pool_small():
    """On CUDA these threads only feed the GPU; a pool per core would fight
    the game's render thread for no gain."""
    assert choose_cpu_threads(AsrConfig(device="cuda")) <= 4


def test_cpu_gets_most_of_the_machine():
    """On CPU these threads ARE the inference. Starving them at 2 took a
    medium model to ~7s per utterance -- past the staleness window, so
    nothing was ever spoken."""
    n = choose_cpu_threads(AsrConfig(device="cpu"))
    assert n >= 4
    assert n > choose_cpu_threads(AsrConfig(device="cuda"))
    assert n <= (os.cpu_count() or 4)


def test_explicit_thread_count_wins():
    assert choose_cpu_threads(AsrConfig(device="cpu", cpu_threads=3)) == 3


def test_thread_env_is_capped_before_native_imports():
    import os

    limit_math_threads(2)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"


def test_priority_levels_are_known():
    assert set(_CLASSES) == {"normal", "below_normal", "idle"}
    # "normal" and non-Windows are both no-ops rather than errors.
    assert set_process_priority("normal") is False
