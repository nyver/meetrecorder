"""Юнит-тесты для config.py."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_recorder.config import (
    AppConfig,
    LLMConfig,
    RecordingConfig,
    TranscriptionConfig,
    load_config,
)


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


# ---------------------------------------------------------------------------
# save_config / update_config_file: секреты из env не попадают на диск
# ---------------------------------------------------------------------------

import yaml  # noqa: E402

from meeting_recorder.config import (  # noqa: E402
    resolve_config_path,
    save_config,
    update_config_file,
)


class TestSaveConfigSecrets:
    def test_env_secrets_not_written(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        monkeypatch.setenv("LLM_API_KEY", "sk_from_env")
        cfg = AppConfig()
        out = tmp_path / "c.yaml"
        save_config(cfg, out)
        text = out.read_text(encoding="utf-8")
        assert "hf_from_env" not in text
        assert "sk_from_env" not in text

    def test_explicit_secrets_written(self, tmp_path):
        cfg = AppConfig.model_validate({"llm": {"api_key": "sk_explicit"}})
        out = tmp_path / "c.yaml"
        save_config(cfg, out)
        assert (
            yaml.safe_load(out.read_text(encoding="utf-8"))["llm"]["api_key"]
            == "sk_explicit"
        )

    def test_roundtrip(self, tmp_path):
        cfg = AppConfig.model_validate(
            {"recording": {"fps": 25}, "llm": {"context_tokens": 65536}}
        )
        out = tmp_path / "sub" / "c.yaml"
        save_config(cfg, out)
        loaded = load_config(out)
        assert loaded.recording.fps == 25
        assert loaded.llm.context_tokens == 65536

    def test_no_temp_files_left(self, tmp_path):
        save_config(AppConfig(), tmp_path / "c.yaml")
        assert [p.name for p in tmp_path.iterdir()] == ["c.yaml"]


class TestUpdateConfigFile:
    def test_merges_and_preserves_other_keys(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "output_dir: D:/M\nllm:\n  api_key: sk-keep\nrecording:\n  fps: 15\n",
            encoding="utf-8",
        )
        update_config_file({"recording": {"mic_device": "USB Mic"}}, path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["output_dir"] == "D:/M"
        assert data["llm"]["api_key"] == "sk-keep"
        assert data["recording"] == {"fps": 15, "mic_device": "USB Mic"}

    def test_creates_missing_file(self, tmp_path):
        path = tmp_path / "new" / "c.yaml"
        update_config_file({"recording": {"mic_device": "X"}}, path)
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
            "recording": {"mic_device": "X"}
        }

    def test_invalid_update_rejected(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("recording:\n  fps: 15\n", encoding="utf-8")
        with pytest.raises(Exception):
            update_config_file({"recording": {"fps": 1000}}, path)
        assert "fps: 15" in path.read_text(encoding="utf-8")

    def test_env_secret_not_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk_env")
        path = tmp_path / "c.yaml"
        update_config_file({"recording": {"mic_device": "X"}}, path)
        assert "sk_env" not in path.read_text(encoding="utf-8")


class TestConfigHelpers:
    def test_resolve_config_path_explicit(self, tmp_path):
        assert resolve_config_path(tmp_path / "x.yaml") == tmp_path / "x.yaml"

    def test_resolve_config_path_default(self):
        assert resolve_config_path(None).name == "config.yaml"

    def test_input_token_budget(self):
        llm = LLMConfig(context_tokens=32768, max_tokens=4096)
        assert llm.input_token_budget() == 32768 - 4096 - 1024

    def test_input_token_budget_large_max_tokens(self):
        """max_tokens больше половины контекста — под ответ резервируется половина."""
        llm = LLMConfig(context_tokens=8192, max_tokens=16384)
        assert llm.input_token_budget() == 8192 - 4096 - 1024
        assert llm.output_token_budget() == 4096

    def test_minimum_context_leaves_room_for_request_overhead(self):
        llm = LLMConfig(context_tokens=2048)
        assert 0 < llm.input_token_budget()
        assert llm.input_token_budget() + llm.output_token_budget() < llm.context_tokens

    def test_new_defaults(self):
        cfg = AppConfig()
        assert cfg.recording.auto_mux is True
        assert cfg.transcription.keep_models_loaded is False
        assert cfg.llm.max_retries == 2
