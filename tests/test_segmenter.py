import numpy as np

from cs2translate.vad.segmenter import Segmenter
from tests.fakes import ScriptedVAD

FRAME = 512
RATE = 16000
FRAME_MS = 1000 * FRAME / RATE  # 32ms


def feed(seg, n_frames):
    """Push n frames of dummy audio one frame at a time."""
    out = []
    for _ in range(n_frames):
        out.extend(seg.push(np.full(FRAME, 0.1, dtype=np.float32)))
    return out


def test_endpoints_after_trailing_silence():
    # 20 voiced frames (640ms), then 15 silent (480ms) -> one utterance.
    vad = ScriptedVAD([0.9] * 20 + [0.0] * 15)
    seg = Segmenter(vad, min_silence_ms=280, min_speech_ms=250, pre_roll_ms=0)
    utts = feed(seg, 35)
    assert len(utts) == 1
    assert utts[0].duration > 0.5
    assert utts[0].peak_probability == 0.9


def test_short_blip_is_rejected():
    # A 2-frame (64ms) transient never reaches min_speech_ms.
    vad = ScriptedVAD([0.9] * 2 + [0.0] * 20)
    seg = Segmenter(vad, min_speech_ms=250, pre_roll_ms=0)
    assert feed(seg, 22) == []


def test_pre_roll_is_prepended():
    vad = ScriptedVAD([0.0] * 10 + [0.9] * 20 + [0.0] * 15)
    no_roll = Segmenter(ScriptedVAD([0.0] * 10 + [0.9] * 20 + [0.0] * 15), pre_roll_ms=0)
    with_roll = Segmenter(vad, pre_roll_ms=320)  # 10 frames
    a = feed(no_roll, 45)[0]
    b = feed(with_roll, 45)[0]
    assert len(b.audio) > len(a.audio)


def test_max_length_forces_flush():
    vad = ScriptedVAD([0.9] * 200)  # never stops talking
    seg = Segmenter(vad, max_utterance_ms=1000, pre_roll_ms=0)
    utts = feed(seg, 60)
    assert len(utts) >= 1
    assert utts[0].duration <= 1.1


def test_speech_active_tracks_state_for_the_tts_scheduler():
    vad = ScriptedVAD([0.9] * 20 + [0.0] * 15)
    seg = Segmenter(vad, min_speech_ms=250, pre_roll_ms=0)
    assert not seg.speech_active
    feed(seg, 12)
    assert seg.speech_active           # mid-callout: TTS should hold
    assert not seg.wait_for_gap(0.01)
    feed(seg, 23)
    assert not seg.speech_active       # gap: safe to speak
    assert seg.wait_for_gap(0.01)


def test_push_handles_arbitrary_block_sizes():
    vad = ScriptedVAD([0.9] * 20 + [0.0] * 15)
    seg = Segmenter(vad, min_speech_ms=250, pre_roll_ms=0)
    utts = []
    for _ in range(60):
        utts.extend(seg.push(np.full(300, 0.1, dtype=np.float32)))  # not a frame multiple
    assert len(utts) == 1
