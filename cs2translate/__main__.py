from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from .asr.whisper import WhisperTranslator
from .audio.backends import NullPlayback, Playback, WasapiLoopbackCapture, WavFileCapture
from .audio.gate import FeedbackGate
from .config import AppConfig
from .logging_setup import setup as setup_logging
from .pipeline import Pipeline
from .vad.segmenter import Segmenter
from .vad.silero import load_vad

log = logging.getLogger("cs2translate")


def list_devices() -> int:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("pyaudiowpatch is not installed (Windows only).", file=sys.stderr)
        return 1
    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        print(f"default output: {default_out['name']}\n")
        print("loopback capture devices:")
        for info in pa.get_loopback_device_info_generator():
            mark = "*" if default_out["name"] in info["name"] else " "
            print(f" {mark} [{info['index']:3d}] {info['name']}  "
                  f"{int(info['defaultSampleRate'])} Hz  {info['maxInputChannels']} ch")
        print("\noutput devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0 and not info.get("isLoopbackDevice"):
                print(f"   [{i:3d}] {info['name']}")
    finally:
        pa.terminate()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cs2translate",
        description="Listen to CS2 voice comms, detect the language, and speak English locally.",
    )
    p.add_argument("--config", type=Path, help="path to a TOML config file")
    p.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    p.add_argument("--capture-device", help="substring of the loopback device name")
    p.add_argument("--playback-device", help="substring of the output device name")
    p.add_argument("--model", help="whisper model (default: large-v3)")
    p.add_argument("--compute-type", help="float16 | int8_float16 | int8 | auto")
    p.add_argument("--device", choices=["cuda", "cpu"], help="whisper device")
    p.add_argument("--voice", help="piper voice, e.g. en_US-lessac-medium")
    p.add_argument("--file", type=Path, help="offline mode: run the pipeline on a 16-bit WAV")
    p.add_argument("--no-speak", action="store_true", help="log translations without speaking")
    p.add_argument("--duck", action="store_true", help="lower other apps' volume while speaking")
    p.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING")
    return p


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    if args.capture_device:
        cfg.audio.capture_device = args.capture_device
    if args.playback_device:
        cfg.audio.playback_device = args.playback_device
    if args.model:
        cfg.asr.model = args.model
    if args.compute_type:
        cfg.asr.compute_type = args.compute_type
    if args.device:
        cfg.asr.device = args.device
    if args.voice:
        cfg.tts.voice = args.voice
    if args.duck:
        cfg.pipeline.duck_game_audio = True
    if args.log_level:
        cfg.log_level = args.log_level


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        return list_devices()

    cfg = AppConfig.load(args.config)
    apply_overrides(cfg, args)
    setup_logging(cfg.log_level)

    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    gate = FeedbackGate(cfg.audio.gate_tail_ms)

    if args.file:
        capture = WavFileCapture(
            args.file, cfg.audio.sample_rate, gate, block_frames=512, realtime=True
        )
    else:
        capture = WasapiLoopbackCapture(
            cfg.audio.sample_rate,
            gate,
            device=cfg.audio.capture_device,
            block_frames=cfg.audio.block_frames,
        )

    vad = load_vad(cache, cfg.vad.model_path, cfg.audio.sample_rate)
    segmenter = Segmenter(
        vad,
        sample_rate=cfg.audio.sample_rate,
        threshold=cfg.vad.threshold,
        min_silence_ms=cfg.vad.min_silence_ms,
        min_speech_ms=cfg.vad.min_speech_ms,
        pre_roll_ms=cfg.vad.pre_roll_ms,
        max_utterance_ms=cfg.vad.max_utterance_ms,
    )

    translator = WhisperTranslator(cfg.asr)
    translator.warmup()

    if args.no_speak:
        from .tts.piper import NullTTS

        tts, playback = NullTTS(), NullPlayback()
    else:
        from .tts.piper import PiperTTS

        tts = PiperTTS(cfg.tts, cache)
        playback = (
            NullPlayback()
            if args.file
            else Playback(cfg.audio.playback_device, cfg.tts.volume)
        )

    pipeline = Pipeline(cfg, capture, segmenter, translator, tts, playback, gate)

    done = threading.Event()

    def handle_signal(_sig, _frm):
        done.set()

    signal.signal(signal.SIGINT, handle_signal)
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except (AttributeError, ValueError):  # pragma: no cover - not on all platforms
        pass

    pipeline.start()
    log.info("listening. Ctrl-C to stop.")
    try:
        while not done.is_set():
            done.wait(0.5)
            # Offline mode: exit once the file has been fully replayed and the
            # queues have drained.
            if args.file and not capture.alive and pipeline.idle:
                break
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
