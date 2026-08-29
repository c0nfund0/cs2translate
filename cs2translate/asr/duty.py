"""Bounding how much of the GPU the pipeline is allowed to take.

In a match the VAD fires far more often than teammates actually speak: CS2's
own radio lines ("Enemy spotted", "Fire in the hole") are real speech, so they
legitimately trigger an utterance, cost a full inference, and are then thrown
away for being English. The result is near-continuous GPU work, which competes
with rendering and costs framerate.

This caps the fraction of wall-clock time the ASR stage may spend inferring.
It is a blunt instrument -- over budget, utterances are dropped rather than
queued, so real callouts can be lost too -- which is why it is opt-in.
"""
from __future__ import annotations

import threading
from collections import deque

from ..clock import now


class DutyLimiter:
    def __init__(self, max_duty: float, window: float = 6.0) -> None:
        self.max_duty = max_duty
        self.window = window
        self._events: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_duty > 0.0

    def _trim(self, t: float) -> None:
        cutoff = t - self.window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def busy_fraction(self) -> float:
        t = now()
        with self._lock:
            self._trim(t)
            busy = sum(d for _, d in self._events)
        return busy / self.window

    def allow(self) -> bool:
        """False when the recent inference load is already over budget."""
        if not self.enabled:
            return True
        return self.busy_fraction() < self.max_duty

    def record(self, seconds: float) -> None:
        t = now()
        with self._lock:
            self._events.append((t, seconds))
            self._trim(t)
