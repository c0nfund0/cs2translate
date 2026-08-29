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
    """Interleaved frames -> mono float32, flat average across all channels.

    Correct only for mono and stereo. Use `Downmixer` for anything wider -- see
    the explanation there.
    """
    if channels == 1:
        return pcm.astype(np.float32, copy=False)
    usable = (len(pcm) // channels) * channels
    frames = pcm[:usable].reshape(-1, channels)
    return frames.mean(axis=1, dtype=np.float32)


# Standard WAVE_FORMAT_EXTENSIBLE channel order puts LFE third for 5.1 and 7.1:
# FL, FR, FC, LFE, BL, BR, [SL, SR]. LFE is low-frequency rumble that only
# muddies speech, so it never contributes to the mono mix.
_LFE_INDEX = {6: 3, 8: 3}


class Downmixer:
    """Interleaved multichannel -> mono, normalised by *active* channels.

    A flat average is wrong for a surround endpoint. Windows renders stereo
    content into front L/R and leaves the remaining channels digitally silent,
    so averaging across an 8-channel 7.1 endpoint computes (L+R)/8 instead of
    (L+R)/2 and throws away 12 dB before the VAD ever sees the audio.

    Rather than assume a layout, track a smoothed per-channel RMS and divide by
    however many channels are actually carrying signal. That gives (L+R)/2 for
    stereo-on-a-7.1-endpoint and a sensible average for genuine surround, with
    no configuration.
    """

    def __init__(self, channels: int, smoothing: float = 0.9, floor_ratio: float = 0.02):
        self.channels = channels
        self._smoothing = smoothing
        self._floor_ratio = floor_ratio
        self._ema = np.zeros(channels, dtype=np.float32)
        self._lfe = _LFE_INDEX.get(channels)

    @property
    def channel_rms(self) -> np.ndarray:
        """Smoothed per-channel RMS. Drives the --monitor meter."""
        return self._ema.copy()

    @property
    def active_channels(self) -> int:
        return int(self._mask().sum())

    def _mask(self) -> np.ndarray:
        peak = float(self._ema.max())
        if peak <= 0.0:
            mask = np.ones(self.channels, dtype=bool)
        else:
            mask = self._ema > peak * self._floor_ratio
        if self._lfe is not None:
            mask[self._lfe] = False
        if not mask.any():
            mask = np.ones(self.channels, dtype=bool)
        return mask

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        if self.channels == 1:
            return pcm.astype(np.float32, copy=False)
        usable = (len(pcm) // self.channels) * self.channels
        frames = pcm[:usable].reshape(-1, self.channels).astype(np.float32, copy=False)
        if self.channels == 2:
            return frames.mean(axis=1, dtype=np.float32)
        rms = np.sqrt(np.mean(frames**2, axis=0))
        self._ema = self._smoothing * self._ema + (1.0 - self._smoothing) * rms
        mask = self._mask()
        return frames[:, mask].sum(axis=1, dtype=np.float32) / float(mask.sum())


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
