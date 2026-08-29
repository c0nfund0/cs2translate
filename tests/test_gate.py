import time

from cs2translate.audio.gate import FeedbackGate


def test_blocks_during_playback_and_for_the_tail():
    gate = FeedbackGate(tail_ms=60)
    assert not gate.is_blocked()
    gate.begin_playback()
    assert gate.is_blocked() and gate.speaking
    gate.end_playback()
    assert not gate.speaking
    assert gate.is_blocked()          # still inside the tail
    time.sleep(0.09)
    assert not gate.is_blocked()
