"""Turns a continuous 16k stream into endpointed utterances."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from ..clock import now

log = logging.getLogger(__name__)


@dataclass
class Utterance:
    audio: np.ndarray          # 16k mono float32
    captured_at: float         # monotonic time the utterance ENDED
    duration: float
    peak_probability: float

    @property
    def age(self) -> float:
        return now() - self.captured_at


class Segmenter:
    """VAD state machine with pre-roll and hangover.

    Also publishes `speech_active`, which the TTS scheduler reads so it can
    hold a translated line until there is a gap rather than talking over a
    live callout (the capture gate makes us deaf while speaking)."""

    def __init__(
        self,
        vad,
        sample_rate: int = 16000,
        threshold: float = 0.6,
        min_silence_ms: int = 280,
        min_speech_ms: int = 250,
        pre_roll_ms: int = 300,
        max_utterance_ms: int = 8000,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame = vad.frame_samples
        self.frame_ms = 1000.0 * self.frame / sample_rate

        self.min_silence_frames = max(1, int(min_silence_ms / self.frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / self.frame_ms))
        # The pre-roll must be at least min_speech_frames long: those are the
        # frames that proved this was speech, and they hold the first syllable.
        # A shorter pre-roll would silently clip the start of every callout.
        self.pre_roll_frames = max(
            self.min_speech_frames, int(pre_roll_ms / self.frame_ms)
        )
        self.max_frames = max(1, int(max_utterance_ms / self.frame_ms))

        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll: list[np.ndarray] = []
        self._active: list[np.ndarray] = []
        self._silence_run = 0
        self._speech_run = 0
        self._peak = 0.0
        self._in_speech = False

        self._speech_active = threading.Event()
        self._gap_event = threading.Event()
        self._gap_event.set()

    @property
    def speech_active(self) -> bool:
        return self._speech_active.is_set()

    def wait_for_gap(self, timeout: float) -> bool:
        """Block until no speech is in progress. False means we timed out and
        the caller should proceed anyway."""
        return self._gap_event.wait(timeout)

    def _set_active(self, active: bool) -> None:
        if active:
            self._speech_active.set()
            self._gap_event.clear()
        else:
            self._speech_active.clear()
            self._gap_event.set()

    def push(self, block: np.ndarray) -> list[Utterance]:
        """Feed a block of arbitrary length; get back any completed utterances."""
        out: list[Utterance] = []
        self._pending = np.concatenate([self._pending, block])
        while len(self._pending) >= self.frame:
            frame, self._pending = self._pending[: self.frame], self._pending[self.frame :]
            utt = self._push_frame(frame)
            if utt is not None:
                out.append(utt)
        return out

    def _push_frame(self, frame: np.ndarray) -> Utterance | None:
        prob = self.vad(frame)
        voiced = prob >= self.threshold

        if not self._in_speech:
            self._pre_roll.append(frame)
            if len(self._pre_roll) > self.pre_roll_frames:
                self._pre_roll.pop(0)
            if voiced:
                self._speech_run += 1
                if self._speech_run >= self.min_speech_frames:
                    self._in_speech = True
                    self._set_active(True)
                    self._active = list(self._pre_roll)
                    self._pre_roll = []
                    self._silence_run = 0
                    self._peak = prob
            else:
                self._speech_run = 0
            return None

        self._active.append(frame)
        self._peak = max(self._peak, prob)
        if voiced:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self.min_silence_frames:
                return self._flush(trim_silence=True)
        if len(self._active) >= self.max_frames:
            log.debug("max utterance length reached, force-flushing")
            return self._flush(trim_silence=False)
        return None

    def _flush(self, trim_silence: bool) -> Utterance | None:
        frames = self._active
        if trim_silence and self._silence_run > 1:
            # Keep one frame of trailing silence; Whisper likes a little room.
            frames = frames[: len(frames) - (self._silence_run - 1)]
        self._active = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_run = 0
        self._set_active(False)
        if not frames:
            return None
        audio = np.concatenate(frames)
        duration = len(audio) / self.sample_rate
        if duration * 1000 < self.min_speech_frames * self.frame_ms:
            return None
        return Utterance(
            audio=audio,
            captured_at=now(),
            duration=duration,
            peak_probability=self._peak,
        )

    def reset(self) -> None:
        self.vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll = []
        self._active = []
        self._silence_run = 0
        self._speech_run = 0
        self._in_speech = False
        self._set_active(False)
