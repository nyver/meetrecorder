"""Юнит-тесты для config.py."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_recorder.config import AppConfig, RecordingConfig, TranscriptionConfig, LLMConfig, load_config


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.output_dir == "C:/Meetings"
        assert cfg.recording.fps == 20
        assert cfg.recording.screen_grabber == "gdigrab"
        assert cfg.transcription.model == "large-v3"
        assert cfg.transcription.language == "ru"
        assert cfg.transcription.device == "cuda"
        assert cfg.llm.backend == "local"
        assert cfg.llm.base_url == "http://127.0.0.1:8080/v1"
        assert cfg.llm.model == "qwen2.5-14b-instruct"
        assert cfg.llm.temperature == 0.3


class TestRecordingConfig:
    def test_fps_validation(self):
        with pytest.raises(Exception):
            RecordingConfig(fps=0)
        with pytest.raises(Exception):
            RecordingConfig(fps=61)

    def test_screen_grabber_literal(self):
        RecordingConfig(screen_grabber="ddagrab")
        RecordingConfig(screen_grabber="gdigrab")
        with pytest.raises(Exception):
            RecordingConfig(screen_grabber="invalid")


class TestTranscriptionConfig:
    def test_hf_token_from_env(self):
        os.environ["HF_TOKEN"] = "test_hf_token_123"
        try:
            raw = {"model": "medium", "language": "en"}
            cfg = TranscriptionConfig.model_validate(raw)
            assert cfg.hf_token.get_secret_value() == "test_hf_token_123"
        finally:
            os.environ.pop("HF_TOKEN", None)

    def test_hf_token_not_logged(self):
        """SecretStr не должен раскрываться в repr."""
        from pydantic import SecretStr
        cfg = TranscriptionConfig(hf_token=SecretStr("super-secret"))
        assert "super-secret" not in repr(cfg)
        assert "**********" in repr(cfg)


class TestLLMConfig:
    def test_api_key_from_env(self):
        os.environ["LLM_API_KEY"] = "sk-or-test-key"
        try:
            raw = {"backend": "openrouter", "base_url": "https://openrouter.ai/api/v1"}
            cfg = LLMConfig.model_validate(raw)
            assert cfg.api_key.get_secret_value() == "sk-or-test-key"
        finally:
            os.environ.pop("LLM_API_KEY", None)

    def test_api_key_not_logged(self):
        """SecretStr не должен раскрываться в repr."""
        from pydantic import SecretStr
        cfg = LLMConfig(api_key=SecretStr("sk-or-secret"))
        assert "sk-or-secret" not in repr(cfg)
        assert "**********" in repr(cfg)


class TestLoadConfig:
    def test_load_from_yaml(self):
        yaml_content = """
output_dir: /tmp/test_meetings
recording:
  fps: 30
  screen_grabber: gdigrab
transcription:
  model: medium
  language: en
llm:
  backend: openrouter
  model: meta-llama/llama-3.1-70b-instruct
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)

        assert cfg.output_dir == "/tmp/test_meetings"
        assert cfg.recording.fps == 30
        assert cfg.recording.screen_grabber == "gdigrab"
        assert cfg.transcription.model == "medium"
        assert cfg.transcription.language == "en"
        assert cfg.llm.backend == "openrouter"
        assert cfg.llm.model == "meta-llama/llama-3.1-70b-instruct"

    def test_load_missing_file_returns_defaults(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, AppConfig)
