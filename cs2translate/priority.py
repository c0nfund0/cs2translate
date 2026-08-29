"""Process priority and thread limits.

The app shares a machine with a game that wants every core and every spare
GPU cycle. A callout arriving 50ms later than it could is invisible; a dropped
frame is not. So the pipeline deliberately runs at a lower scheduling priority
and keeps its CPU thread pools small.
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)

# From Windows' SetPriorityClass.
_CLASSES = {
    "normal": 0x00000020,
    "below_normal": 0x00004000,
    "idle": 0x00000040,
}


def set_process_priority(level: str) -> bool:
    """Lower this process's scheduling priority. No-op off Windows."""
    if level == "normal" or sys.platform != "win32":
        return False
    flag = _CLASSES.get(level)
    if flag is None:
        log.warning("unknown process_priority %r; leaving it at normal", level)
        return False
    try:
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        if k32.SetPriorityClass(k32.GetCurrentProcess(), flag):
            log.info("process priority set to %s so the game wins contended cores", level)
            return True
        log.debug("SetPriorityClass failed: %s", ctypes.get_last_error())
    except Exception as exc:
        log.debug("could not set process priority: %s", exc)
    return False


def limit_math_threads(threads: int = 2) -> None:
    """Cap the BLAS/OpenMP pools.

    Must run before numpy, onnxruntime or ctranslate2 are imported -- these
    libraries read the environment once, at import. Left unset they each spawn
    one worker per core, and Piper's onnxruntime session in particular will
    happily saturate every thread the machine has for a 100ms synthesis.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(threads))
