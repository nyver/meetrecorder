"""Загрузка и валидация config.yaml через pydantic."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, SecretStr, model_validator

# ---------------------------------------------------------------------------
# Pydantic-модели
# ---------------------------------------------------------------------------


class RecordingConfig(BaseModel):
    """Параметры захвата экрана и звука."""

    fps: int = Field(default=20, ge=1, le=60)
    video_codec: str = "libx264"
    audio_sample_rate: int = Field(default=48000, ge=8000, le=192000)
    record_mic: bool = True
    mic_device: str = ""  # "" — автодетект первого dshow capture-устройства
    system_audio_device: str = "virtual-audio-capturer"
    # Устройство вывода для grabber="soundcard": id или имя; "" — по умолчанию
    system_audio_speaker: str = ""
    system_audio_grabber: Literal["dshow", "wasapi", "soundcard"] = "soundcard"
    record_system_audio: bool = True
    screen_grabber: Literal["ddagrab", "gdigrab"] = "gdigrab"
    # Сводить видео + mix-аудио в *_final.mp4 при генерации отчёта
    auto_mux: bool = True


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
    # False — выгружать модели после обработки (освобождает VRAM в режиме трея)
    keep_models_loaded: bool = False

    # Секрет взят из переменной окружения — не сохраняем его в YAML
    _secret_from_env: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _resolve_hf_token(self) -> TranscriptionConfig:
        if self.hf_token.get_secret_value():
            return self
        env = os.environ.get("HF_TOKEN", "").strip()
        if env:
            self.hf_token = SecretStr(env)
            self._secret_from_env = True
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
    # Размер контекстного окна модели (вход + выход), токенов
    context_tokens: int = Field(default=32768, ge=2048)
    # Повторы при 429 / 5xx / сетевых сбоях
    max_retries: int = Field(default=2, ge=0, le=10)

    _secret_from_env: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _resolve_api_key(self) -> LLMConfig:
        if self.api_key.get_secret_value():
            return self
        env = os.environ.get("LLM_API_KEY", "").strip()
        if env:
            self.api_key = SecretStr(env)
            self._secret_from_env = True
        return self

    def output_token_budget(self) -> int:
        """Лимит ответа, согласованный с резервом входного контекста."""
        return min(self.max_tokens, self.context_tokens // 2)

    def input_token_budget(self) -> int:
        """Сколько токенов можно отдать под промпт, оставив место под ответ."""
        return max(self.context_tokens - self.output_token_budget() - 1024, 1)


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


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Вернуть фактический путь к конфигу: явный или путь по умолчанию."""
    return Path(path) if path else _CONFIG_FILE


def load_config(path: Path | str | None = None) -> AppConfig:
    """Загрузить и валидировать конфиг из YAML-файла."""
    config_path = resolve_config_path(path)
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


def _dump_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
        sort_keys=False,
    )


def _atomic_write(path: Path, text: str) -> None:
    from .session_utils import atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text)


def save_config(cfg: AppConfig, path: Path | str | None = None) -> None:
    """Сохранить конфиг в YAML-файл целиком.

    Секреты, подставленные из переменных окружения (HF_TOKEN, LLM_API_KEY),
    в файл не записываются. Запись атомарная.
    """
    config_path = resolve_config_path(path)
    raw = cfg.model_dump(exclude_none=True)
    if cfg.transcription._secret_from_env:
        raw["transcription"].pop("hf_token", None)
    if cfg.llm._secret_from_env:
        raw["llm"].pop("api_key", None)
    _atomic_write(config_path, _dump_yaml(_reveal_secrets(raw)))


def _deep_merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def update_config_file(updates: dict[str, Any], path: Path | str | None = None) -> None:
    """Точечно обновить ключи в YAML-конфиге, не трогая остальное содержимое.

    В отличие от save_config не переписывает весь конфиг из модели: значения,
    которых нет в updates (в т.ч. секреты), остаются как в файле. Результат
    валидируется перед записью.

    Пример: ``update_config_file({"recording": {"mic_device": "USB Mic"}})``.
    """
    config_path = resolve_config_path(path)
    raw: dict = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(raw, updates)
    AppConfig.model_validate(merged)  # не записываем невалидный конфиг
    _atomic_write(config_path, _dump_yaml(merged))
