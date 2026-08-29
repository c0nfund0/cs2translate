"""Pre-download every model so the first match does not stall on a download.

    python scripts/download_models.py [--model large-v3] [--voice en_US-lessac-medium]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs2translate.config import AppConfig  # noqa: E402
from cs2translate.logging_setup import setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="whisper model id")
    ap.add_argument("--voice", default=None, help="piper voice id")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    if args.model:
        cfg.asr.model = args.model
    if args.voice:
        cfg.tts.voice = args.voice
    setup(cfg.log_level)
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    from cs2translate.vad.silero import ensure_model

    print("Silero VAD ->", ensure_model(cache, cfg.vad.model_path))

    from cs2translate.tts.piper import ensure_voice

    voices = Path(cfg.tts.voices_dir) if cfg.tts.voices_dir else cache / "piper"
    print("Piper voice ->", ensure_voice(cfg.tts.voice, voices))

    # Downloading whisper means instantiating it; do it on CPU/int8 so this
    # script works on a machine without CUDA.
    from faster_whisper import WhisperModel

    print(f"Whisper {cfg.asr.model} -> downloading (CPU int8 load)...")
    WhisperModel(cfg.asr.model, device="cpu", compute_type="int8")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
