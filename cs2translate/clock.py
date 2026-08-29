"""The clock used for every latency and staleness measurement.

`time.monotonic()` is backed by GetTickCount64 on Windows, with a granularity
of ~15.6ms. That is unusable here for two reasons:

  * this pipeline budgets end-to-end latency at ~600-900ms and reports an
    average, so a 15.6ms quantum is a visible fraction of the measurement; and
  * short intervals collapse to exactly 0.0, which silently defeats
    strictly-greater-than comparisons such as the staleness check.

`time.perf_counter()` is monotonic on every platform and QPC-backed on Windows,
giving sub-microsecond resolution. Use `now()` everywhere rather than calling
time functions directly, so this decision stays in one place.
"""
from __future__ import annotations

import time

now = time.perf_counter
