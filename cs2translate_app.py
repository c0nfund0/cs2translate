"""Frozen-app entry point.

PyInstaller needs a real script, not a `-m package` invocation. This also pins
multiprocessing behaviour, which the frozen bootloader requires on Windows.
"""
from __future__ import annotations

import multiprocessing
import os
import sys


def _prepare_frozen_env() -> None:
    """When frozen, CUDA/cuDNN DLLs live beside the exe in _internal. CTranslate2
    loads them via the OS loader, which does not search sys.path, so the
    directories have to be registered explicitly."""
    if not getattr(sys, "frozen", False):
        return
    base = os.path.dirname(sys.executable)
    internal = os.path.join(base, "_internal")
    candidates = [base, internal]
    for root in (internal, base):
        nvidia = os.path.join(root, "nvidia")
        if os.path.isdir(nvidia):
            for pkg in os.listdir(nvidia):
                binpath = os.path.join(nvidia, pkg, "bin")
                if os.path.isdir(binpath):
                    candidates.append(binpath)
    seen = set()
    for path in candidates:
        if not os.path.isdir(path) or path in seen:
            continue
        seen.add(path)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass
    os.environ["PATH"] = os.pathsep.join(seen) + os.pathsep + os.environ.get("PATH", "")


def main() -> int:
    multiprocessing.freeze_support()
    _prepare_frozen_env()
    from cs2translate.__main__ import main as real_main

    return real_main()


if __name__ == "__main__":
    sys.exit(main())
