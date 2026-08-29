"""Guards the clock-resolution requirement.

These would have caught the Windows-only CI failure where an utterance was
created and age-checked inside a single 15.6ms GetTickCount64 tick, making its
age exactly 0.0 and silently defeating the staleness check.
"""
from __future__ import annotations

import time

from cs2translate.clock import now


def test_resolution_is_finer_than_a_millisecond():
    # time.monotonic() on Windows reports 0.015625 here and would fail.
    info = time.get_clock_info("perf_counter")
    assert info.resolution < 1e-3, f"clock too coarse: {info.resolution}s"
    assert info.monotonic, "staleness maths requires a monotonic clock"


def test_consecutive_reads_are_distinguishable():
    """A short interval must not quantise to zero."""
    samples = [now() for _ in range(200)]
    deltas = [b - a for a, b in zip(samples, samples[1:])]
    assert all(d >= 0 for d in deltas), "clock went backwards"
    assert any(d > 0 for d in deltas), "no measurable progress across 200 reads"


def test_a_tiny_elapsed_interval_is_strictly_positive():
    """The exact shape of the staleness check: `age > max_age` with max_age=0
    must be True for any real elapsed work."""
    t0 = now()
    sum(range(1000))
    assert now() - t0 > 0.0
