"""Rejecting Whisper output that is not actually a translation.

Whisper hallucinates confidently on non-speech audio, and CS2's mix is full of
non-speech that survives VAD -- gunfire tails, radio stings, footsteps. These
filters run on every segment before anything reaches TTS, because a spoken
hallucination is far more disruptive than a dropped line.
"""
from __future__ import annotations

import re
import unicodedata

# Whisper's training data was heavily YouTube-derived, so on noise it falls back
# to subtitle boilerplate. These are the phrases seen most often in practice.
HALLUCINATION_PHRASES = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "thanks for watching!",
    "please subscribe",
    "subscribe to my channel",
    "subtitles by the amara.org community",
    "subtitles by",
    "amara.org",
    "www.mooji.org",
    "bye",
    "bye bye",
    "you",
    "the end",
    "to be continued",
    "transcription by castingwords",
    "i'm sorry",
    "okay",
    "oh",
    "hmm",
    "mm",
    "yeah",
    "so",
    ".",
    "...",
    "[music]",
    "(music)",
    "[applause]",
    "(applause)",
    "[silence]",
    "music",
    "applause",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = _PUNCT.sub("", text)
    return _WS.sub(" ", text).strip()


def is_hallucination(text: str) -> bool:
    norm = normalize(text)
    if not norm:
        return True
    if norm in {normalize(p) for p in HALLUCINATION_PHRASES}:
        return True
    words = norm.split()
    # A single repeated token ("go go go go go go") is a decoder loop, not a
    # callout -- but allow short genuine repeats like "go go go".
    if len(words) >= 5 and len(set(words)) == 1:
        return True
    return False


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are duplicates. Whisper loops score near 1.0."""
    words = normalize(text).split()
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def should_reject(
    text: str,
    no_speech_prob: float,
    compression_ratio: float,
    *,
    no_speech_threshold: float,
    max_compression_ratio: float,
) -> str | None:
    """Returns a reason string when the segment should be dropped, else None."""
    if not text or not text.strip():
        return "empty"
    if no_speech_prob >= no_speech_threshold:
        return f"no_speech_prob={no_speech_prob:.2f}"
    if compression_ratio > max_compression_ratio:
        return f"compression_ratio={compression_ratio:.2f}"
    if is_hallucination(text):
        return "hallucination-phrase"
    if repeated_ngram_ratio(text) > 0.5:
        return "repeated-ngrams"
    return None
