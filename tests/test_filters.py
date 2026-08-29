import pytest

from cs2translate.asr import filters


@pytest.mark.parametrize(
    "text",
    ["Thanks for watching!", "Subtitles by the Amara.org community", "", "   ",
     "[Music]", "you", "go go go go go go go"],
)
def test_flags_known_hallucinations(text):
    assert filters.is_hallucination(text)


@pytest.mark.parametrize("text", ["they are rotating to B", "one low mid", "go go go"])
def test_keeps_real_callouts(text):
    assert not filters.is_hallucination(text)


def test_repeated_ngrams_detects_decoder_loop():
    looped = "he is coming " * 8
    assert filters.repeated_ngram_ratio(looped) > 0.5
    assert filters.repeated_ngram_ratio("they are pushing banana with a full stack") < 0.5


def test_should_reject_uses_no_speech_prob():
    reason = filters.should_reject(
        "two at ramp", no_speech_prob=0.9, compression_ratio=1.0,
        no_speech_threshold=0.6, max_compression_ratio=2.4,
    )
    assert reason and "no_speech_prob" in reason


def test_should_reject_uses_compression_ratio():
    reason = filters.should_reject(
        "two at ramp", no_speech_prob=0.1, compression_ratio=3.5,
        no_speech_threshold=0.6, max_compression_ratio=2.4,
    )
    assert reason and "compression_ratio" in reason


def test_should_accept_clean_callout():
    assert filters.should_reject(
        "they are rotating to B", no_speech_prob=0.05, compression_ratio=1.2,
        no_speech_threshold=0.6, max_compression_ratio=2.4,
    ) is None
