"""Загрузка и валидация config.yaml через pydantic."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator

# ---------------------------------------------------------------------------
# Pydantic-модели
# ---------------------------------------------------------------------------


class RecordingConfig(BaseModel):
    """Параметры захвата экрана и звука."""

    fps: int = Field(default=20, ge=1, le=60)
    video_codec: str = "libx264"
    audio_sample_rate: int = Field(default=48000, ge=8000, le=192000)
    mic_device: str = "Настольный микрофон (Microsoft® LifeCam HD-3000)"
    system_audio_device: str = "virtual-audio-capturer"
    system_audio_grabber: Literal["dshow", "wasapi", "soundcard"] = "soundcard"
    record_system_audio: bool = True
    screen_grabber: Literal["ddagrab", "gdigrab"] = "gdigrab"


class TranscriptionConfig(BaseModel):
    """Параметры транскрипции и диаризации."""

    model: str = "large-v3"
    language: str = "ru"
    diarization: bool = True
    device: Literal["cuda", "cpu"] = "cuda"
    # beam_size=1 (greedy) в ~3-4x быстрее при минимальной потере точности
    beam_size: int = Field(default=5, ge=1, le=10)
    # batch_size=0 → авто: 16 на cuda, без батчинга на cpu
    # BatchedInferencePipeline обрабатывает чанки параллельно (3-5x быстрее на GPU)
    batch_size: int = Field(default=0, ge=0)
    hf_token: SecretStr = Field(default=SecretStr(""))
    speaker_names: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_hf_token(self) -> "TranscriptionConfig":
        if self.hf_token.get_secret_value():
            return self
        env = os.environ.get("HF_TOKEN", "").strip()
        if env:
            self.hf_token = SecretStr(env)
        return self


class LLMConfig(BaseModel):
    """Параметры LLM-клиента."""

    backend: Literal["local", "openrouter"] = "local"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = "qwen2.5-14b-instruct"
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=16384, gt=0)
    clean_protocol: bool = False
    timeout: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "LLMConfig":
        if self.api_key.get_secret_value():
            return self
        env = os.environ.get("LLM_API_KEY", "").strip()
        if env:
            self.api_key = SecretStr(env)
        return self


class AppConfig(BaseModel):
    """Корневая конфигурация приложения."""

    output_dir: str = "C:/Meetings"
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


# ---------------------------------------------------------------------------
# Функции загрузки
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    # Editable/dev install: config.yaml рядом с корнем проекта — используем его
    legacy = Path(__file__).resolve().parent.parent / "config.yaml"
    if legacy.exists():
        return legacy
    # Стандартное расположение для установленного пакета
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "MeetingRecorder" / "config.yaml"


_CONFIG_FILE = _default_config_path()


def load_config(path: Path | str | None = None) -> AppConfig:
    """Загрузить и валидировать конфиг из YAML-файла."""
    config_path = Path(path) if path else _CONFIG_FILE
    if not config_path.exists():
        # Если файла нет — вернуть дефолты
        return AppConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


def _reveal_secrets(d: dict) -> dict:
    """Рекурсивно преобразовать SecretStr в строки для сериализации в YAML."""
    result = {}
    for k, v in d.items():
        if isinstance(v, SecretStr):
            val = v.get_secret_value()
            if val:
                result[k] = val
        elif isinstance(v, dict):
            filtered = _reveal_secrets(v)
            if filtered:
                result[k] = filtered
        elif v not in ("", [], None):
            result[k] = v
    return result


def save_config(cfg: AppConfig, path: Path | str | None = None) -> None:
    """Сохранить конфиг в YAML-файл."""
    config_path = Path(path) if path else _CONFIG_FILE
    raw = cfg.model_dump(exclude_none=True)
    data = _reveal_secrets(raw)
    config_path.write_text(
        yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
