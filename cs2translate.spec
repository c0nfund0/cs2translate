# PyInstaller spec for cs2translate.
#
# onedir, not onefile: the CUDA/cuDNN payload is well over a gigabyte, and
# onefile would re-extract all of it to %TEMP% on every launch.
#
# Models are NOT bundled -- whisper large-v3 alone is ~3GB. The app downloads
# them to %USERPROFILE%\.cache\cs2translate on first run.

import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files

WITH_CUDA = os.environ.get("CS2T_CUDA", "1") != "0"

datas, binaries, hiddenimports = [], [], []


def add(pkg, *, optional=True, data=True, bins=True, imports=True):
    try:
        d, b, h = collect_all(pkg)
    except Exception as exc:
        if not optional:
            raise
        print(f"[spec] skipping {pkg}: {exc}")
        return
    if data:
        datas.extend(d)
    if bins:
        binaries.extend(b)
    if imports:
        hiddenimports.extend(h)
    print(f"[spec] collected {pkg}: {len(d)} data, {len(b)} binaries")


# Core runtime. ctranslate2 and onnxruntime ship native libraries that
# PyInstaller's dependency scanner does not find on its own.
add("faster_whisper", optional=False)
add("ctranslate2", optional=False)
add("onnxruntime", optional=False)
add("tokenizers")
add("huggingface_hub")
add("av")

# TTS. piper pulls espeak-ng data through one of several helper packages
# depending on version; collect whichever is present.
add("piper")
add("piper_phonemize")
add("espeakng_loader")
add("espeak_phonemizer")

# Windows audio.
add("pyaudiowpatch", optional=False)
add("pycaw")
add("comtypes")

# Resampling / numerics.
add("soxr")
add("scipy", data=False)

if WITH_CUDA:
    # cuBLAS and cuDNN 9 are what CTranslate2 needs for float16 inference.
    # Bundling them is why the exe works on a machine with only a GPU driver
    # and no CUDA toolkit installed.
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
        try:
            libs = collect_dynamic_libs(pkg)
            binaries.extend(libs)
            print(f"[spec] cuda libs from {pkg}: {len(libs)}")
        except Exception as exc:
            print(f"[spec] no {pkg}: {exc}")
    try:
        datas.extend(collect_data_files("nvidia", include_py_files=False))
    except Exception:
        pass

hiddenimports += [
    "onnxruntime.capi._pybind_state",
    "cs2translate.asr.whisper",
    "cs2translate.tts.piper",
    "cs2translate.vad.silero",
    "cs2translate.audio.backends",
]

datas += [("config.example.toml", "."), ("README.md", ".")]

a = Analysis(
    ["cs2translate_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Trim things nothing in the pipeline uses; these pull in large trees.
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "IPython", "notebook", "pytest", "torch", "torchaudio"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cs2translate",
    console=True,
    debug=False,
    strip=False,
    upx=False,          # UPX corrupts some CUDA DLLs
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="cs2translate",
)
