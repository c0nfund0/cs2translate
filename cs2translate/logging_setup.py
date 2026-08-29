from __future__ import annotations

import logging
import sys


class _Fmt(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;110m",
        "WARNING": "\033[38;5;179m",
        "ERROR": "\033[38;5;167m",
        "CRITICAL": "\033[38;5;167m",
    }
    RESET = "\033[0m"

    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        if self.color:
            c = self.COLORS.get(record.levelname, "")
            return f"{c}{out}{self.RESET}"
        return out


def setup(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Fmt(color=sys.stderr.isatty()))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # These are chatty and we have our own status line.
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
