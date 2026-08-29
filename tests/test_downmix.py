"""Multichannel downmix.

Regression cover for a bug found on a real 7.1 headset exposing an
8-channel loopback endpoint: averaging across all channels when only front L/R carry
signal attenuated everything by 12 dB before the VAD saw it.
"""
from __future__ import annotations

import numpy as np

from cs2translate.audio.resample import Downmixer, to_mono


def interleave(channel_data: list[np.ndarray]) -> np.ndarray:
    return np.stack(channel_data, axis=1).reshape(-1).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def stereo_on_surround(n=1024, channels=8):
    sig = (np.sin(np.linspace(0, 40, n)) * 0.3).astype(np.float32)
    chans = [np.zeros(n, np.float32) for _ in range(channels)]
    chans[0] = sig
    chans[1] = sig
    return interleave(chans), sig


def test_stereo_on_a_surround_endpoint_keeps_full_level():
    inter, sig = stereo_on_surround()
    d = Downmixer(8)
    for _ in range(20):
        out = d(inter)
    assert rms(out) == 0.0 or abs(rms(out) - rms(sig)) < 1e-4
    assert d.active_channels == 2


def test_flat_average_was_losing_12db():
    """Pins the magnitude of the original bug."""
    inter, sig = stereo_on_surround()
    d = Downmixer(8)
    for _ in range(20):
        good = d(inter)
    bad = to_mono(inter, 8)
    lost_db = 20 * np.log10(rms(good) / rms(bad))
    assert 11.5 < lost_db < 12.5, f"expected ~12 dB recovered, got {lost_db:.1f}"


def test_lfe_is_excluded():
    """LFE is rumble; it only muddies speech."""
    n = 1024
    speech = (np.sin(np.linspace(0, 40, n)) * 0.2).astype(np.float32)
    rumble = (np.sin(np.linspace(0, 2, n)) * 0.9).astype(np.float32)
    chans = [np.zeros(n, np.float32) for _ in range(8)]
    chans[0], chans[1] = speech, speech
    chans[3] = rumble  # LFE
    inter = interleave(chans)
    d = Downmixer(8)
    for _ in range(20):
        out = d(inter)
    # If LFE leaked in, the loud rumble would dominate the result.
    assert rms(out) < rms(speech) * 1.2


def test_genuine_surround_averages_active_channels():
    n = 1024
    rng = np.random.default_rng(0)
    chans = [(rng.standard_normal(n) * 0.1).astype(np.float32) for _ in range(8)]
    chans[3] = np.zeros(n, np.float32)  # silent LFE
    inter = interleave(chans)
    d = Downmixer(8)
    for _ in range(20):
        out = d(inter)
    assert d.active_channels == 7
    assert rms(out) > 0


def test_stereo_and_mono_are_untouched():
    n = 512
    left = np.full(n, 0.4, np.float32)
    right = np.full(n, 0.2, np.float32)
    out = Downmixer(2)(interleave([left, right]))
    assert np.allclose(out, 0.3)
    mono = np.full(n, 0.5, np.float32)
    assert np.allclose(Downmixer(1)(mono), mono)


def test_adapts_when_channels_go_silent():
    """Content can switch from 7.1 to stereo mid-stream; the mix must follow."""
    n = 1024
    rng = np.random.default_rng(1)
    full = interleave([(rng.standard_normal(n) * 0.1).astype(np.float32) for _ in range(8)])
    d = Downmixer(8)
    for _ in range(30):
        d(full)
    assert d.active_channels == 7
    stereo, _ = stereo_on_surround(n)
    for _ in range(80):
        d(stereo)
    assert d.active_channels == 2
