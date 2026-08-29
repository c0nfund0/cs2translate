"""GPU duty-cycle limiter."""
from __future__ import annotations

import time

from cs2translate.asr.duty import DutyLimiter
from cs2translate.config import AsrConfig


def test_disabled_by_default():
    """It drops real callouts when over budget, so it must be opt-in."""
    assert AsrConfig().max_duty_cycle == 0.0
    assert not DutyLimiter(0.0).enabled
    assert DutyLimiter(0.0).allow()


def test_allows_until_the_budget_is_spent():
    d = DutyLimiter(0.25, window=1.0)
    assert d.allow()
    d.record(0.1)          # 10% of the window
    assert d.allow()
    d.record(0.2)          # now 30% > 25%
    assert not d.allow()


def test_budget_recovers_as_the_window_slides():
    d = DutyLimiter(0.5, window=0.2)
    d.record(0.15)
    assert not d.allow()
    time.sleep(0.25)       # old event falls out of the window
    assert d.allow()
    assert d.busy_fraction() == 0.0


def test_busy_fraction_reports_load():
    d = DutyLimiter(0.9, window=2.0)
    d.record(0.5)
    assert abs(d.busy_fraction() - 0.25) < 0.01


def test_unlimited_never_blocks_however_much_is_recorded():
    d = DutyLimiter(0.0, window=1.0)
    for _ in range(50):
        d.record(1.0)
    assert d.allow()
