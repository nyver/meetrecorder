"""Обёртка над WhisperX: аудио → сегменты с диаризацией."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import AppConfig
from .naming import SessionPaths

logger = logging.getLogger(__name__)

# Кэш для моделей (singleton)
_model_cache: dict[str, Any] = {}


def _get_transcribe_model(device: str, dtype: str | None = None):
    """Загрузить Whisper-модель с кэшированием."""
    key = f"{device}_{dtype}"
    if key in _model_cache:
        return _model_cache[key]

    from faster_whisper import WhisperModel

    logger.info("Загрузка модели Whisper (device=%s, dtype=%s)…", device, dtype)
    model = WhisperModel(
        "large-v3",  # всегда large-v3 для лучшего качества
        device=device,
        compute_type=dtype or ("float16" if device == "cuda" else "int8"),
        download_root=Path.home() / ".cache" / "whisper",
    )
    _model_cache[key] = model
    logger.info("Модель Whisper загружена")
    return model


def _get_diarization_model(cfg: AppConfig):
    """Загрузить pyannote.audio диаризацию с кэшированием."""
    key = "diarization"
    if key in _model_cache:
        return _model_cache[key]

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.warning("pyannote.audio не установлен — диаризация отключена")
        return None

    hf_token = cfg.transcription.hf_token
    if not hf_token:
        logger.warning(
            "HF_TOKEN не указан — диаризация будет отключена. "
            "Установите токен на huggingface.co/settings/tokens"
        )
        return None

    logger.info("Загрузка pyannote.pipeline (device=%s)…", cfg.transcription.device)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to_device()
    _model_cache[key] = pipeline
    logger.info("pyannote.pipeline загружена")
    return pipeline


class Segment:
    """Один сегмент транскрипта."""

    def __init__(self, start: float, end: float, speaker: str, text: str):
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": self.speaker,
            "text": self.text,
        }


def transcribe(
    audio_path: Path | str,
    cfg: AppConfig,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Транскрибировать аудио через WhisperX с диаризацией.

    Args:
        audio_path: Путь к WAV-файлу (микс или отдельная дорожка).
        cfg: Полная конфигурация приложения.
        output_path: Куда сохранить JSON (автоматически, если None).

    Returns:
        Словарь с ключами: session_id, language, duration_sec, segments.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

    tcfg = cfg.transcription
    device = tcfg.device
    language = tcfg.language

    # Проверка GPU
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "cuda запрошен, но GPU не найден — переключение на cpu"
                )
                device = "cpu"
        except ImportError:
            device = "cpu"

    # Загрузка модели Whisper
    model = _get_transcribe_model(device)

    logger.info(
        "Начинаю транскрипцию: %s (language=%s, device=%s)",
        audio_path.name, language, device,
    )

    # WhisperX: transcribe с word-level timing
    segments_raw, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=language if language else None,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        initial_prompt=(
            "Это запись деловой встречи на русском языке."
            if language == "ru"
            else ""
        ),
    )

    # Собираем сегменты
    segments: list[Segment] = []
    for seg in segments_raw:
        segments.append(Segment(
            start=seg.start,
            end=seg.end,
            speaker="UNKNOWN",  # будет заполнено диаризацией
            text=seg.text.strip(),
        ))

    logger.info("Whisper: %d сегментов (язык: %s)", len(segments), info.language)

    # Диаризация
    if tcfg.diarization:
        pipeline = _get_diarization_model(cfg)
        if pipeline is not None:
            segments = _apply_diarization(pipeline, audio_path, segments, cfg)
        else:
            logger.warning("Диаризация пропущена")
    else:
        logger.info("Диаризация отключена в конфиге")

    # Сопоставление имён
    speaker_names = tcfg.speaker_names
    if speaker_names:
        segments = _apply_speaker_names(segments, speaker_names)

    duration = info.duration if hasattr(info, "duration") else 0.0
    detected_lang = info.language if hasattr(info, "language") else language

    result = {
        "session_id": "",  # будет заполнено вызывающим
        "language": detected_lang,
        "duration_sec": round(duration, 2),
        "segments": [s.to_dict() for s in segments],
    }

    # Сохранение JSON
    if output_path is None:
        output_path = audio_path.with_name(
            audio_path.name.replace(".wav", "_transcript.json")
        )
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Транскрипт сохранён: %s (%d сегментов)", output_path, len(segments))

    return result


def _apply_diarization(
    pipeline,
    audio_path: Path,
    segments: list[Segment],
    cfg: AppConfig,
) -> list[Segment]:
    """Применить pyannote.audio диаризацию к сегментам Whisper."""
    from pyannote.audio import Audio

    logger.info("Выполняю диаризацию на %s…", audio_path.name)

    # Загружаем аудио для pipeline
    audio = Audio(sample_rate=16000, mono=True)
    snippet = audio.crop(audio_path, duration=None)

    # Запускаем пайплайн диаризации
    diarization = pipeline(snippet)

    # Создаем карту: segment_index → set of speakers
    speaker_map: dict[int, set[str]] = {i: set() for i in range(len(segments))}

    # Пробегаем по активным диапазонам диаризации
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        # Находим сегменты, перекрывающиеся с этим интервалом
        for i, seg in enumerate(segments):
            if seg.start < end and seg.end > start:
                speaker_map[i].add(speaker)

    # Заменяем speaker на первый найденный (или UNKNOWN)
    for i, seg in enumerate(segments):
        speakers = speaker_map.get(i, set())
        seg.speaker = speakers.pop() if speakers else "UNKNOWN"

    # Нормализуем имена говорящих (SPEAKER_00, SPEAKER_01, …)
    unique_speakers = sorted({s.speaker for s in segments if s.speaker != "UNKNOWN"})
    speaker_rename: dict[str, str] = {}
    for idx, old in enumerate(unique_speakers):
        speaker_rename[old] = f"SPEAKER_{idx:02d}"

    for seg in segments:
        if seg.speaker in speaker_rename:
            seg.speaker = speaker_rename[seg.speaker]

    logger.info("Диаризация: %d уникальных говорящих", len(speaker_rename))
    return segments


def _apply_speaker_names(
    segments: list[Segment],
    speaker_names: dict[str, str],
) -> list[Segment]:
    """Заменить технические метки (SPEAKER_00) на имена из конфига."""
    for seg in segments:
        if seg.speaker in speaker_names:
            seg.speaker = speaker_names[seg.speaker]
    return segments


def load_transcript(path: Path | str) -> dict[str, Any]:
    """Загрузить транскрипт из JSON-файла."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Транскрипт не найден: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data
