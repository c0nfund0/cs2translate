"""Silero VAD v5 via onnxruntime.

Run directly against the ONNX graph rather than through the `silero-vad` pip
package, which drags in torch (~2.5GB with CUDA). faster-whisper already needs
onnxruntime, so this VAD is effectively free.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Pinned to a tag, not master: the upstream default branch can change the model
# and its input contract underneath us.
MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/"
    "src/silero_vad/data/silero_vad.onnx"
)
FRAME_SAMPLES = 512  # the analysis window at 16 kHz (32 ms)

# v5 does not analyse the window alone. It expects the tail of the PREVIOUS
# window prepended as context, so the tensor handed to the graph is 576 samples
# at 16 kHz (64 + 512), or 288 at 8 kHz (32 + 256). Feeding a bare 512 is
# accepted silently -- the input is declared [None, None] -- and returns ~0.001
# for every frame, i.e. a VAD that never fires.
CONTEXT_SAMPLES = {16000: 64, 8000: 32}


def ensure_model(cache_dir: Path, path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "silero_vad.onnx"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log.info("downloading Silero VAD -> %s", dest)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(dest)
    return dest


class SileroVAD:
    """Per-frame speech probability. Stateful across frames."""

    frame_samples = FRAME_SAMPLES

    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # One thread: this runs every 32ms on tiny tensors, and thread pool
        # churn costs more than the inference itself.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self._inputs = {i.name for i in self.session.get_inputs()}
        # v5 uses a single packed `state`; v4 used separate LSTM h/c.
        self._v5 = "state" in self._inputs
        self._context_samples = CONTEXT_SAMPLES.get(sample_rate, 64) if self._v5 else 0
        self.reset()

    def reset(self) -> None:
        self._context = np.zeros((1, self._context_samples), dtype=np.float32)
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if len(frame) != self.frame_samples:
            raise ValueError(f"expected {self.frame_samples} samples, got {len(frame)}")
        x = frame.astype(np.float32, copy=False).reshape(1, -1)
        sr = np.array(self.sample_rate, dtype=np.int64)
        if self._v5:
            # Prepend the previous window's tail; see CONTEXT_SAMPLES above.
            packed = np.concatenate([self._context, x], axis=1)
            out, self._state = self.session.run(
                None, {"input": packed, "state": self._state, "sr": sr}
            )
            self._context = x[:, -self._context_samples :]
        else:
            out, self._h, self._c = self.session.run(
                None, {"input": x, "sr": sr, "h": self._h, "c": self._c}
            )
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Fallback when the Silero download fails. Materially worse against
    gunfire, which is exactly the case Silero was chosen for -- this exists so
    the app still starts offline, not because it is good."""

    frame_samples = FRAME_SAMPLES

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._noise = 1e-3
        self.reset()

    def reset(self) -> None:
        self._noise = 1e-3

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9)
        # Spectral flatness separates broadband transients (shots) from voiced
        # speech reasonably well, though nowhere near as well as Silero.
        spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) + 1e-9
        flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        if rms < self._noise * 2:
            self._noise = 0.98 * self._noise + 0.02 * rms
        snr = rms / (self._noise + 1e-9)
        score = min(1.0, snr / 12.0) * (1.0 - min(1.0, flatness * 2.5))
        return float(max(0.0, score))


def load_vad(cache_dir: Path, model_path: Path | None, sample_rate: int = 16000):
    try:
        return SileroVAD(ensure_model(cache_dir, model_path), sample_rate)
    except Exception as exc:
        log.warning("Silero VAD unavailable (%s); falling back to energy VAD", exc)
        return EnergyVAD(sample_rate)
