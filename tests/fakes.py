"""Stand-ins so the pipeline can be exercised without CUDA, WASAPI or models."""
from __future__ import annotations

import numpy as np

from cs2translate.asr.whisper import Translation


class ScriptedVAD:
    """Returns a preset probability per frame index, so segmentation tests are
    deterministic instead of depending on Silero's behaviour."""

    frame_samples = 512

    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0

    def __call__(self, frame):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return float(p)

    def reset(self):
        pass


class FakeTranslator:
    def __init__(self, text="rotating to B", language="ru", fail=False):
        self.text = text
        self.language = language
        self.fail = fail
        self.calls = 0

    def translate(self, utt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return Translation(
            text=self.text,
            language=self.language,
            language_probability=0.99,
            utterance=utt,
            asr_ms=42.0,
        )


class FakeTTS:
    sample_rate = 22050

    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        return np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)


class RecordingPlayback:
    def __init__(self):
        self.played = []

    def play(self, audio, rate):
        self.played.append((len(audio), rate))

    def close(self):
        pass
