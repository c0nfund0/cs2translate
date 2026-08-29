"""Audio I/O backends.

Capture is abstracted so the whole pipeline can be exercised from a WAV file on
a machine with no WASAPI (i.e. for tests, and for tuning VAD/ASR settings
without launching CS2).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..clock import now
from .gate import FeedbackGate
from .resample import Resampler, to_mono

log = logging.getLogger(__name__)


class CaptureBackend(ABC):
    """Yields 16k mono float32 blocks. Blocks suppressed by the gate are
    replaced with silence rather than dropped, so downstream endpointing sees a
    continuous timeline and closes any open utterance."""

    def __init__(self, out_rate: int, gate: FeedbackGate | None) -> None:
        self.out_rate = out_rate
        self.gate = gate
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
        self._stop = threading.Event()

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @property
    def alive(self) -> bool:
        """False once the source is exhausted (only meaningful for finite
        sources such as WAV replay)."""
        return True

    def blocks(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def _emit(self, mono16k: np.ndarray) -> None:
        if self.gate is not None and self.gate.is_blocked():
            self.gate.dropped_frames += len(mono16k)
            mono16k = np.zeros_like(mono16k)
        try:
            self._q.put_nowait(mono16k)
        except queue.Full:
            # Better to lose a block than to stall the audio callback.
            log.debug("capture queue full, dropping %d frames", len(mono16k))

    def _close(self) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass


class WasapiLoopbackCapture(CaptureBackend):
    """WASAPI loopback of a render device: what CS2 is sending to your ears."""

    def __init__(
        self,
        out_rate: int = 16000,
        gate: FeedbackGate | None = None,
        device: str | None = None,
        block_frames: int = 1024,
    ) -> None:
        super().__init__(out_rate, gate)
        self.device_hint = device
        self.block_frames = block_frames
        self._pa = None
        self._stream = None
        self._resampler: Resampler | None = None
        self._channels = 2
        self.device_name = "?"

    def _resolve_device(self, pa) -> dict:
        import pyaudiowpatch as pyaudio

        if self.device_hint:
            for info in pa.get_loopback_device_info_generator():
                if self.device_hint.lower() in info["name"].lower():
                    return info
            raise RuntimeError(f"no loopback device matching {self.device_hint!r}")

        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if default_out.get("isLoopbackDevice"):
            return default_out
        # Every render device has a shadow loopback device with a matching name.
        for info in pa.get_loopback_device_info_generator():
            if default_out["name"] in info["name"]:
                return info
        raise RuntimeError(
            f"no loopback counterpart for default output {default_out['name']!r}"
        )

    def start(self) -> None:
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        dev = self._resolve_device(self._pa)
        self.device_name = dev["name"]
        in_rate = int(dev["defaultSampleRate"])
        self._channels = int(dev["maxInputChannels"])
        self._resampler = Resampler(in_rate, self.out_rate)
        log.info(
            "capturing loopback: %s (%d Hz, %d ch)", self.device_name, in_rate, self._channels
        )

        def callback(in_data, frame_count, time_info, status):
            if status:
                log.debug("capture status flags: %s", status)
            pcm = np.frombuffer(in_data, dtype=np.float32)
            mono = to_mono(pcm, self._channels)
            self._emit(self._resampler.process(mono))  # type: ignore[union-attr]
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._channels,
            rate=in_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=self.block_frames,
            stream_callback=callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # pragma: no cover
                pass
        if self._pa is not None:
            self._pa.terminate()
        self._close()


class WavFileCapture(CaptureBackend):
    """Replays a WAV through the pipeline. `realtime=True` paces it like a live
    stream so latency numbers mean something."""

    def __init__(
        self,
        path: str | Path,
        out_rate: int = 16000,
        gate: FeedbackGate | None = None,
        block_frames: int = 512,
        realtime: bool = True,
    ) -> None:
        super().__init__(out_rate, gate)
        self.path = Path(path)
        self.block_frames = block_frames
        self.realtime = realtime
        self._thread: threading.Thread | None = None
        self.device_name = str(self.path)

    def _run(self) -> None:
        with wave.open(str(self.path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            in_rate = wf.getframerate()
            resampler = Resampler(in_rate, self.out_rate)
            if width != 2:
                raise ValueError(f"{self.path}: expected 16-bit PCM, got {width * 8}-bit")
            period = self.block_frames / in_rate
            next_at = now()
            while not self._stop.is_set():
                raw = wf.readframes(self.block_frames)
                if not raw:
                    break
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                self._emit(resampler.process(to_mono(pcm, channels)))
                if self.realtime:
                    next_at += period
                    delay = next_at - now()
                    if delay > 0:
                        time.sleep(delay)
        self._close()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="wav-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class Playback:
    """Blocking playback of 22.05k-ish mono float32 (whatever Piper emits)."""

    def __init__(self, device: str | None = None, volume: float = 1.0) -> None:
        self.device_hint = device
        self.volume = volume
        self._pa = None
        self._stream = None
        self._rate = 0
        self._lock = threading.Lock()

    def _device_index(self, pa) -> int | None:
        if not self.device_hint:
            return None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0 and self.device_hint.lower() in info["name"].lower():
                return i
        raise RuntimeError(f"no output device matching {self.device_hint!r}")

    def _ensure(self, rate: int):
        import pyaudiowpatch as pyaudio

        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        if self._stream is not None and self._rate == rate:
            return self._stream
        if self._stream is not None:
            self._stream.close()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=rate,
            output=True,
            output_device_index=self._device_index(self._pa),
        )
        self._rate = rate
        return self._stream

    def play(self, audio: np.ndarray, rate: int) -> None:
        with self._lock:
            stream = self._ensure(rate)
            data = np.clip(audio * self.volume, -1.0, 1.0).astype(np.float32)
            stream.write(data.tobytes())

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None


class NullPlayback:
    """Used by offline/file mode so the pipeline runs without an audio device."""

    def __init__(self, sink: list | None = None) -> None:
        self.sink = sink if sink is not None else []

    def play(self, audio: np.ndarray, rate: int) -> None:
        self.sink.append((audio, rate))
        time.sleep(len(audio) / rate)  # keep gate timing realistic

    def close(self) -> None:
        pass
