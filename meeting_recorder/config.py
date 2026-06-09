"""Загрузка и валидация config.yaml через pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

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
    hf_token: str = Field(default="")
    speaker_names: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_hf_token(self) -> "TranscriptionConfig":
        if self.hf_token:
            return self
        env = os.environ.get("HF_TOKEN", "").strip()
        if env:
            self.hf_token = env
        return self


class LLMConfig(BaseModel):
    """Параметры LLM-клиента."""

    backend: Literal["local", "openrouter"] = "local"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = Field(default="")
    model: str = "qwen2.5-14b-instruct"
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=16384, gt=0)
    clean_protocol: bool = False

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "LLMConfig":
        if self.api_key:
            return self
        env = os.environ.get("LLM_API_KEY", "").strip()
        if env:
            self.api_key = env
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

_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path | str | None = None) -> AppConfig:
    """Загрузить и валидировать конфиг из YAML-файла."""
    config_path = Path(path) if path else _CONFIG_FILE
    if not config_path.exists():
        # Если файла нет — вернуть дефолты
        return AppConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


def save_config(cfg: AppConfig, path: Path | str | None = None) -> None:
    """Сохранить конфиг в YAML-файл."""
    config_path = Path(path) if path else _CONFIG_FILE
    # Используем exclude_none=True и дополнительно исключаем пустые строки
    raw = cfg.model_dump(exclude_none=True)
    # Убираем пустые строки и пустые словари
    data = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            filtered = {k: v for k, v in values.items()
                        if v not in ("", {}, [], None)}
            if filtered:
                data[section] = filtered
        else:
            data[section] = values
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
