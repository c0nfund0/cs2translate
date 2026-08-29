"""Configuration for cs2translate.

Defaults are tuned for the target setup: NVIDIA GPU with 8-10GB VRAM, Piper TTS
on CPU, and the "capture gate" feedback strategy (no virtual audio cable).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

try:  # py311+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py310 and older
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


DEFAULT_CACHE = Path.home() / ".cache" / "cs2translate"


@dataclass
class AudioConfig:
    # Capture. None = the system default output device's loopback.
    capture_device: str | None = None
    # Playback. None = system default output device. Same device as capture is
    # expected here; that is exactly why the feedback gate exists.
    playback_device: str | None = None
    # Internal working rate. Whisper and Silero both want 16k mono.
    sample_rate: int = 16000
    # PortAudio buffer. 1024 frames @ 48k is ~21ms; small enough not to matter
    # against the ~1s end-to-end budget, large enough to avoid dropouts.
    block_frames: int = 1024
    # Drop captured audio this long after TTS playback ends, to catch the tail
    # of speakers/room reverb making it back into the loopback mix.
    gate_tail_ms: int = 300


@dataclass
class VadConfig:
    # Silero v5 speech probability threshold. Game audio is noisy, so this sits
    # higher than the 0.5 the upstream repo suggests.
    threshold: float = 0.6
    # Trailing silence that ends an utterance. This is the dominant tunable in
    # the latency budget: lower is snappier but chops mid-sentence pauses.
    min_silence_ms: int = 280
    # Reject blips: gunfire transients occasionally poke above threshold.
    min_speech_ms: int = 250
    # Audio kept before the trigger point, so the first phoneme is not clipped.
    pre_roll_ms: int = 300
    # Hard cap. Someone holding the key open should not stall the pipeline.
    max_utterance_ms: int = 8000
    model_path: Path | None = None


@dataclass
class AsrConfig:
    model: str = "large-v3"
    device: str = "cuda"
    # Chosen at runtime by the VRAM guard when set to "auto".
    compute_type: str = "auto"
    # Below this many free MiB we drop from float16 to int8_float16.
    vram_guard_mib: int = 4608
    # Greedy decoding. Beam search buys little on short utterances and costs
    # latency we do not have.
    beam_size: int = 1
    task: str = "translate"
    # Utterances whose detected language is in this set never reach TTS.
    skip_languages: tuple[str, ...] = ("en",)
    # Discard when the language ID itself is a coin flip.
    min_language_probability: float = 0.5
    no_speech_threshold: float = 0.6
    # Whisper's classic failure mode on non-speech audio is looping n-grams.
    max_compression_ratio: float = 2.4
    download_root: Path | None = None
    # CTranslate2 worker threads. 0 = its default (one per core), which
    # competes with the game's render thread for no benefit on GPU.
    cpu_threads: int = 2
    # Cap the share of wall-clock time spent in inference. 0 disables. Set
    # this if the pipeline is costing framerate: in a match the VAD fires far
    # more often than teammates speak, because the game's own radio lines are
    # speech. Over budget, utterances are dropped -- real ones included.
    max_duty_cycle: float = 0.0
    # Move the model out of VRAM after this many seconds with nothing to
    # translate, and pull it back on demand. 0 disables. Frees ~3GB while
    # nobody is talking, at the cost of ~0.3-1s on the first callout after a
    # quiet stretch.
    idle_unload_s: float = 0.0


@dataclass
class TtsConfig:
    voice: str = "en_US-lessac-medium"
    voices_dir: Path | None = None
    # Piper on CPU deliberately: the GPU is fully committed to large-v3.
    use_cuda: bool = False
    length_scale: float = 0.95  # slightly faster than natural; these are callouts
    noise_scale: float = 0.667
    noise_w: float = 0.8
    volume: float = 1.0
    # How long the scheduler will hold a finished line waiting for a gap in
    # incoming speech before speaking over it anyway.
    max_gap_wait_ms: int = 1500


@dataclass
class PipelineConfig:
    # A callout older than this is worse than no callout.
    max_utterance_age_ms: int = 4000
    asr_queue_size: int = 8
    tts_queue_size: int = 4
    # Optional: duck other apps' volume while speaking (needs pycaw).
    duck_game_audio: bool = False
    duck_level: float = 0.35
    # Windows scheduling priority. The game should always win a contended
    # core; translation arriving 50ms later is invisible, a dropped frame
    # is not. "normal" | "below_normal" | "idle"
    process_priority: str = "below_normal"


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    cache_dir: Path = DEFAULT_CACHE
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            return cfg
        if tomllib is None:  # pragma: no cover
            raise RuntimeError(
                "TOML config needs Python 3.11+, or `pip install tomli` on 3.10"
            )
        with p.open("rb") as fh:
            data = tomllib.load(fh)
        _merge(cfg, data)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(target: Any, data: dict[str, Any]) -> None:
    """Overlay a parsed TOML mapping onto a nested dataclass instance."""
    valid = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in valid:
            raise ValueError(f"unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
            continue
        if value is None:
            setattr(target, key, None)
            continue
        # TOML has no Path type, so coerce based on the declared annotation
        # rather than on whatever the default happens to be (which is often None).
        if "Path" in str(valid[key].type):
            setattr(target, key, Path(value))
        elif isinstance(current, tuple):
            setattr(target, key, tuple(value))
        else:
            setattr(target, key, value)
