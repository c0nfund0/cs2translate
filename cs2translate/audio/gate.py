"""The feedback gate.

We capture the loopback of the same device we play TTS into, so without a gate
the app hears its own English output and translates it again, forever. The gate
is a timestamp: capture drops every frame until it passes.
"""
from __future__ import annotations

import threading
import time


class FeedbackGate:
    def __init__(self, tail_ms: int = 300) -> None:
        self._tail = tail_ms / 1000.0
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        self._speaking = False
        self.dropped_frames = 0

    def begin_playback(self) -> None:
        with self._lock:
            self._speaking = True
            self._blocked_until = float("inf")

    def end_playback(self) -> None:
        """Playback finished; keep blocking for the tail so reverb and any
        buffered device audio do not leak back in."""
        with self._lock:
            self._speaking = False
            self._blocked_until = time.monotonic() + self._tail

    def is_blocked(self) -> bool:
        with self._lock:
            if self._speaking:
                return True
            return time.monotonic() < self._blocked_until

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking
