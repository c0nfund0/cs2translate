"""Downmix + resample from the device's native format to 16k mono float32."""
from __future__ import annotations

from math import gcd

import numpy as np

_BACKEND = "linear"
try:
    import soxr

    _BACKEND = "soxr"
except ImportError:  # pragma: no cover - depends on install
    try:
        from scipy.signal import resample_poly

        _BACKEND = "scipy"
    except ImportError:
        pass


def backend() -> str:
    return _BACKEND


def to_mono(pcm: np.ndarray, channels: int) -> np.ndarray:
    """Interleaved frames -> mono float32."""
    if channels == 1:
        return pcm.astype(np.float32, copy=False)
    usable = (len(pcm) // channels) * channels
    frames = pcm[:usable].reshape(-1, channels)
    return frames.mean(axis=1, dtype=np.float32)


class Resampler:
    """Stateful resampler.

    Stateful matters: resampling each block independently produces an audible
    click at every block boundary, and gives Whisper a stream full of
    transients that look like consonants.
    """

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.passthrough = in_rate == out_rate
        self._impl = None
        self._tail = np.zeros(0, dtype=np.float32)
        self._phase = 0.0
        if self.passthrough:
            return
        if _BACKEND == "soxr":
            self._impl = soxr.ResampleStream(
                in_rate, out_rate, 1, dtype="float32", quality="QQ"
            )
        elif _BACKEND == "scipy":
            g = gcd(in_rate, out_rate)
            self._up, self._down = out_rate // g, in_rate // g

    def process(self, mono: np.ndarray) -> np.ndarray:
        mono = np.asarray(mono, dtype=np.float32)
        if self.passthrough or len(mono) == 0:
            return mono
        if _BACKEND == "soxr":
            return np.asarray(self._impl.resample_chunk(mono), dtype=np.float32)
        if _BACKEND == "scipy":
            # Carry a short tail across blocks so the polyphase filter has
            # history and the seam stays inaudible.
            keep = 64
            buf = np.concatenate([self._tail, mono])
            out = np.asarray(resample_poly(buf, self._up, self._down), dtype=np.float32)
            skip = int(round(len(self._tail) * self._up / self._down))
            self._tail = buf[-keep:] if len(buf) >= keep else buf
            return out[skip:] if skip < len(out) else np.zeros(0, dtype=np.float32)
        return self._linear(mono)

    def _linear(self, mono: np.ndarray) -> np.ndarray:
        """Last-resort fallback. Aliases, but keeps the app usable if neither
        soxr nor scipy has a wheel for the running interpreter."""
        step = self.in_rate / self.out_rate
        buf = np.concatenate([self._tail, mono])
        n = int((len(buf) - 1 - self._phase) / step) + 1
        if n <= 0:
            self._tail = buf
            return np.zeros(0, dtype=np.float32)
        idx = self._phase + step * np.arange(n)
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        consumed = int(np.floor(idx[-1]))
        self._phase = idx[-1] + step - consumed
        self._tail = buf[consumed:]
        return out
