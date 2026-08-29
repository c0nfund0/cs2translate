# cs2translate

Listens to CS2 voice comms on Windows, detects the spoken language, translates it
to English, and **speaks the English out loud** — entirely locally, no network
calls after the one-time model download.

```
CS2 audio ──► WASAPI loopback ──► Silero VAD ──► faster-whisper large-v3
                    ▲              (endpointing)   (language ID + translate)
                    │                                        │
              feedback gate ◄──────────────┐                 ▼
                    │                      │           utterance queue
                    ▼                      │            (drop-stale)
              your headphones ◄──── Piper TTS ◄───────────────┘
```

## Quick start

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_models.py     :: ~3GB whisper + 60MB voice, once
python -m cs2translate --list-devices :: check the right loopback is picked
python -m cs2translate
```

Set CS2 to **borderless windowed** or windowed. Exclusive fullscreen can take
the audio endpoint away from shared mode and break loopback capture.

## How it works

**Capture.** WASAPI loopback on your default output device — literally what CS2
is sending to your ears. No virtual audio cable, no driver install.

**The feedback gate.** We play TTS into the same device we capture from, so
without protection the app would hear its own English and translate it again,
forever. `FeedbackGate` blocks capture for the whole of playback plus a 300ms
tail. This is the one design cost of the no-install approach: **while speaking,
the app is deaf.** Two things soften it — the TTS scheduler holds a finished
line for up to 1.5s waiting for a gap in incoming speech rather than talking
over a live callout, and anything captured while gated is zeroed rather than
dropped, so the segmenter closes cleanly instead of splicing two speakers
together.

**VAD.** Silero v5 through onnxruntime directly, not the `silero-vad` package —
that one pulls in torch (~2.5GB with CUDA) for a 2MB model. Threshold defaults
to 0.6 rather than the usual 0.5, because CS2's mix is hostile: gunfire,
footsteps and radio stings all reach the loopback alongside voice.

**ASR + translation in one pass.** `faster-whisper` large-v3 with
`task="translate"` does language ID and speech→English together. `large-v3-turbo`
and `distil-large-v3` are much faster but were distilled on transcription only,
and their translate output is not usable — so large-v3 it is.

**Filtering.** Whisper hallucinates confidently on non-speech, and plenty of
non-speech survives VAD. Every segment is checked against `no_speech_prob`,
compression ratio, a known-hallucination phrase list ("Thanks for watching",
"Subtitles by the Amara.org community"), and a repeated-n-gram detector for
decoder loops. A spoken hallucination is far more disruptive than a dropped line.

**English is skipped.** Detected-English utterances are logged but never spoken —
no point repeating your own team back at you, and it roughly halves queue load
on mixed-language servers.

**Staleness.** Every stage re-checks utterance age and drops anything older than
4s. A callout that arrives after the fight is over is worse than silence.

## Latency

Measured from the moment a speaker stops talking to the first sample of English:

| stage | typical |
|---|---|
| VAD endpoint (`min_silence_ms`) | 280 ms |
| Whisper large-v3, fp16, short utterance | 150–300 ms |
| Piper synthesis (CPU) | 50–100 ms |
| buffering + scheduling | ~50 ms |
| **total** | **≈ 0.6–0.9 s** |

Plus however long the English itself takes to say. Two things worth being
honest about:

- **This cannot be made much lower.** Speech output has a floor that text does
  not: you cannot begin speaking a sentence until you know most of it, and you
  cannot un-speak a word, so partial hypotheses (which would help a subtitle
  overlay) are useless here.
- **In a fast exchange it will fall behind**, and the drop-stale policy means
  you will lose lines rather than hear them late. That is deliberate.

`python -m cs2translate --log-level DEBUG` prints per-utterance timings and
`avg_latency` in the shutdown summary.

## Diagnosing "it says listening but nothing happens"

```
cs2translate.exe --monitor
```

Loads no models, starts instantly, and prints a live meter of the audio path:

```
  mono  -28.3 dB   vad 0.94 SPEECH   ch 1: -22.1 2: -21.8 3:  --  4:  -- ...
```

On Ctrl-C it prints a verdict that separates the three failure modes: no audio
reaching the app at all (wrong capture device), audio arriving but the VAD never
firing (level or threshold), or both working (problem is downstream in ASR/TTS).

**Surround endpoints.** If the startup log shows more than 2 channels, e.g.

```
capturing loopback: Speakers (Your Headset) [Loopback] (48000 Hz, 8 ch)
```

that is a 7.1 endpoint. Windows puts ordinary stereo content in front L/R and
leaves the rest digitally silent, so the mono downmix normalises by the channels
actually carrying signal. `--monitor` shows the per-channel levels, so you can
see which ones are live.

## Running alongside the game

The app competes with CS2 for both VRAM and cores. Defaults now lean toward the
game: Windows scheduling priority is `below_normal`, and the CPU thread pools
are capped at 2 (Piper's onnxruntime session would otherwise saturate every
core for a 100ms synthesis).

VRAM is the bigger lever. large-v3 at fp16 holds ~3GB, and once the game plus
the model exceed the card the driver starts evicting — which shows up as a
**sustained** framerate collapse rather than occasional hitches.

| Setting | VRAM | Cost |
|---|---|---|
| `--compute-type float16` (default) | ~3.0 GB | — |
| `--compute-type int8_float16` | ~1.6 GB | slight accuracy loss |
| `--model medium --compute-type int8_float16` | ~0.8 GB | noticeably weaker translation |
| `--idle-unload 30` | 0 GB while idle | ~0.3-1s on the first callout after a quiet spell |

`--idle-unload` hands the weights back to system RAM after a quiet stretch and
pulls them in again on demand, so the game keeps the VRAM during the long
stretches when nobody is talking. It is off by default because it trades
latency for headroom.

**If VRAM is not the whole story**, the pipeline is probably inferring far more
often than teammates actually speak. CS2's own radio lines ("Enemy spotted",
"Fire in the hole") are real speech: they trigger the VAD legitimately, cost a
full inference, and are then discarded for being English. Two answers:

```
cs2translate.exe --max-gpu-duty 0.3     # cap inference at 30% of wall time
cs2translate.exe --device cpu --model medium --compute-type int8
```

The first bounds the framerate cost but drops some real callouts along with the
noise. The second removes the GPU from the pipeline entirely -- latency goes to
~1-2s, framerate cost goes to zero. The periodic log line reports the measured
`inference load` so you can see which regime you are in.

To tell VRAM pressure from compute contention: `--compute-type int8_float16`
halves memory with near-identical compute. If the framerate recovers, it was
memory. If only `--model medium` helps, it was compute. `nvidia-smi` while both
are running shows which.

## Tuning

Copy `config.example.toml` and pass `--config`.

| symptom | change |
|---|---|
| gunfire triggers phantom utterances | raise `vad.threshold` toward 0.7 |
| first word of callouts is clipped | raise `vad.pre_roll_ms` |
| sentences get chopped in half | raise `vad.min_silence_ms` |
| feels sluggish | lower `vad.min_silence_ms` to ~200 |
| English is hard to hear over the fight | `--duck` (needs `pycaw`) |
| speech is too slow | lower `tts.length_scale` to ~0.9 |
| VRAM pressure with CS2 running | `--compute-type int8_float16` |

**Precision guard.** `compute_type = "auto"` reads free VRAM at startup and picks
`float16` (~3GB) if there is headroom, `int8_float16` (~1.6GB) if not. On an
8–10GB card float16 fits, but CS2's own allocation moves around, so a load-time
OOM also falls back rather than failing.

## Offline mode

Run the whole pipeline against a WAV instead of live audio — useful for tuning
VAD and checking translation quality without launching a match:

```bash
python -m cs2translate --file sample.wav --no-speak --log-level DEBUG
```

`--no-speak` swaps in null TTS/playback, so this works on a machine with no
audio device at all.

## Known limits

- **Voice and game audio are mixed.** There is no clean voice-only tap from CS2 —
  Steam's separate voice output device setting does not apply to in-game voice.
  VAD and the filters cope, but accuracy degrades during loud fights.
- **Deaf while speaking**, per the feedback gate above. Migrating to a VB-Cable
  split would remove this; the capture layer is abstracted behind
  `CaptureBackend` to make that a contained change.
- **No speaker identification.** Overlapping speakers become one utterance.
- **Exclusive-fullscreen CS2** may break loopback capture.

## Building a Windows .exe

The build must run **on Windows** — PyInstaller cannot cross-compile, and the
payload here is native (CTranslate2, onnxruntime, PortAudio, optionally cuDNN).

### Option A — GitHub Actions (nothing to install)

CI **does not build on every push**. A full build pulls ~1.5GB of CUDA
libraries, takes ~9 minutes, and Windows runners bill at 2x, so it runs only
when a version is cut:

```bash
git tag -a v0.1.1 -m "what changed"
git push origin v0.1.1
```

That builds on a `windows-latest` runner, runs the test suite, smoke-tests the
frozen exe, and publishes a **Release** with `cs2translate-windows-x64.zip`
attached. Download, unzip on the gaming PC, run `cs2translate.exe`.

To build without cutting a version, use Actions → **build-windows-exe** →
*Run workflow*. That uploads a 30-day artifact and publishes no release.

Note the difference: artifacts live under the Actions run and expire after 30
days; releases are permanent and appear on the repo's Releases page.

### Option B — build on the Windows machine

Needs Python 3.11 from python.org with *Add python.exe to PATH* ticked.

```bat
build.bat            :: bundles cuBLAS + cuDNN, ~2GB output, works anywhere
build.bat nocuda     :: ~400MB, target needs its own CUDA 12 + cuDNN 9
```

Output is `dist\cs2translate\` — a self-contained folder. Copy the whole folder
to any Windows machine; it does not need Python installed. Build once, copy to
as many machines as you like.

### What is and isn't in the exe

**Bundled:** the Python runtime, CTranslate2, onnxruntime, PortAudio (via
PyAudioWPatch), Piper's synthesis code, and — unless you pass `nocuda` — the
cuBLAS and cuDNN 9 DLLs that CTranslate2 needs for fp16. Bundling those is what
lets the exe do GPU inference on a machine that has only an NVIDIA driver and no
CUDA toolkit.

**Not bundled:** the models. Whisper large-v3 alone is ~3GB, which would make the
download absurd and pin you to one model. On first run the app fetches them into
`%USERPROFILE%\.cache\cs2translate`. To pre-seed on a machine with no internet,
run `python scripts\download_models.py` elsewhere and copy that folder across.

It is **onedir, not onefile**, deliberately: onefile would re-extract well over a
gigabyte of CUDA DLLs to `%TEMP%` on every single launch.

### Why there is no exe in this repo already

I developed this on Linux. I did try to cross-build under Wine — Windows Python
3.10.11 and every Windows wheel (ctranslate2, onnxruntime, PyAudioWPatch,
piper-tts, PyInstaller) installed fine, but the available Wine is 6.0.3, whose
CRT is missing functions numpy needs; both numpy 1.26 and 2.2 abort on import,
so PyInstaller's hooks cannot run. Even had it produced a binary, nothing about
CUDA or WASAPI would have been testable from Linux. Use Option A or B above.

## Tests

```bash
python -m pytest tests -q
```

34 tests, no GPU or Windows audio required — fakes stand in for CUDA, WASAPI and
Piper, so segmentation, filtering, the feedback gate and the stage wiring are all
covered on any platform, along with the clock-resolution guarantee the
staleness policy depends on.
