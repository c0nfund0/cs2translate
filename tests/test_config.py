import sys

import pytest

from cs2translate.config import AppConfig, tomllib

HAVE_TOML = tomllib is not None


def test_defaults_match_the_target_setup():
    cfg = AppConfig()
    assert cfg.asr.model == "large-v3"
    assert cfg.asr.compute_type == "auto"
    assert cfg.asr.task == "translate"
    assert "en" in cfg.asr.skip_languages
    assert cfg.tts.use_cuda is False   # GPU stays reserved for whisper


@pytest.mark.skipif(not HAVE_TOML, reason="needs tomllib or tomli")
def test_toml_overrides_merge_into_nested_sections(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[vad]\nthreshold = 0.75\n\n[asr]\nmodel = "medium"\nskip_languages = ["en", "fi"]\n',
        encoding="utf-8",
    )
    cfg = AppConfig.load(p)
    assert cfg.vad.threshold == 0.75
    assert cfg.asr.model == "medium"
    assert cfg.asr.skip_languages == ("en", "fi")
    assert cfg.vad.min_silence_ms == 280  # untouched default survives


@pytest.mark.skipif(not HAVE_TOML, reason="needs tomllib or tomli")
def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[asr]\nmodl = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        AppConfig.load(p)


def test_missing_file_falls_back_to_defaults(tmp_path):
    assert AppConfig.load(tmp_path / "nope.toml").asr.model == "large-v3"
