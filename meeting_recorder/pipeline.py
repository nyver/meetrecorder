"""Оркестрация этапов пайплайна.

Recording → finalize_recording (mix) → Transcription → Protocol → Summary →
Highlights → [mux] → HTML Protocol.

CLI и трей вызывают только функции этого модуля — логика этапов не дублируется.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from . import state
from .config import AppConfig
from .html_report import generate_html_protocol
from .naming import SessionPaths, resolve_session
from .recorder import mix_audio_files, mux_video, no_window_kwargs
from .report import generate_highlights, generate_protocol, generate_summary
from .session_utils import (
    load_transcript_data,
    replace_with_retry,
    save_transcript_data,
)
from .transcriber import transcribe, unload_models

logger = logging.getLogger(__name__)
T = TypeVar("T")


class PipelineError(Exception):
    """Ошибка пайплайна."""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        **no_window_kwargs(),
    )
    if result.returncode != 0:
        return 0.0
    info = json.loads(result.stdout)
    return float(info.get("format", {}).get("duration", 0))


def _run_optional_step(description: str, func: Callable[[], T]) -> T | None:
    try:
        return func()
    except Exception as exc:
        logger.warning("%s: %s", description, exc)
        return None


def _load_existing_transcript(paths: SessionPaths) -> dict[str, Any]:
    if not paths.transcript.exists():
        raise PipelineError(
            f"Транскрипт не найден: {paths.transcript}. Сначала выполните транскрипцию."
        )
    try:
        return load_transcript_data(paths.transcript)
    except Exception as exc:
        raise PipelineError(f"Не удалось прочитать транскрипт: {exc}") from exc


def _check_video_length(paths: SessionPaths) -> None:
    """Предупредить, если видео сильно короче аудио (gdigrab упал в середине записи)."""
    if not paths.video.exists() or not paths.mic_audio.exists():
        return
    try:
        video_dur = _ffprobe_duration(paths.video)
        audio_dur = _ffprobe_duration(paths.mic_audio)

        if audio_dur > 0 and video_dur < audio_dur * 0.5:
            logger.warning(
                "ВНИМАНИЕ: видео (%.0f с) значительно короче аудио (%.0f с). "
                "gdigrab завершился досрочно — скорее всего, Windows заблокировал "
                "захват экрана (UAC, защищённый рабочий стол, DRM). "
                "Для более надёжного захвата смените screen_grabber: ddagrab "
                "в config.yaml.",
                video_dur,
                audio_dur,
            )
    except Exception as exc:
        logger.debug("_check_video_length: %s", exc)


def _rename_with_retry(
    src: Path, dst: Path, retries: int = 10, delay: float = 0.5
) -> None:
    """Переименовать файл с повторами (файл может быть временно заблокирован)."""
    replace_with_retry(src, dst, retries=retries, delay=delay)


# ---------------------------------------------------------------------------
# Этапы
# ---------------------------------------------------------------------------


def _system_audio_offset(paths: SessionPaths) -> float:
    if not paths.capture_timing.exists():
        return 0.0
    try:
        timing = json.loads(paths.capture_timing.read_text(encoding="utf-8"))
        offset = float(timing["system_audio_offset_sec"])
        if not math.isfinite(offset):
            raise ValueError("Смещение должно быть конечным числом")
        return offset
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Синхронизация аудио недоступна, выравниваю по началу: %s", exc)
        return 0.0


def finalize_recording(cfg: AppConfig, paths: SessionPaths) -> Path:
    """Подготовить аудио после остановки записи: свести mic + system в mix.

    Если записана только одна дорожка — она становится mix-аудио.

    Raises:
        PipelineError: аудиофайлы не созданы или сведение не удалось.
    """
    _check_video_length(paths)
    try:
        if paths.mic_audio.exists() and paths.system_audio.exists():
            mix_audio_files(
                paths.mic_audio,
                paths.system_audio,
                paths.mix_audio,
                cfg.recording.audio_sample_rate,
                system_offset_sec=_system_audio_offset(paths),
            )
            logger.info("Аудио сведено: %s", paths.mix_audio)
        elif paths.mic_audio.exists():
            _rename_with_retry(paths.mic_audio, paths.mix_audio)
            logger.warning("Системный звук отсутствует — используется только микрофон")
        elif paths.system_audio.exists():
            _rename_with_retry(paths.system_audio, paths.mix_audio)
            logger.warning("Микрофон отсутствует — используется только системный звук")
        else:
            raise PipelineError(
                "Аудиофайлы не найдены — ffmpeg не записал данные. "
                "Проверьте mic_device / system_audio_* в config.yaml."
            )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Не удалось свести аудио: {exc}") from exc
    return paths.mix_audio


def run_transcribe(cfg: AppConfig, paths: SessionPaths) -> dict[str, Any]:
    """Транскрибировать сессию."""
    audio_path = paths.mix_audio
    if not audio_path.exists():
        active = state.load_state()
        if (
            active
            and active.get("session_id") == paths.session_id
            and state.is_recording_active()
        ):
            raise PipelineError("Сессия ещё записывается — сначала остановите запись")
        finalize_recording(cfg, paths)

    result = transcribe(
        audio_path,
        cfg,
        output_path=paths.transcript,
        session_id=paths.session_id,
    )
    if result.get("session_id") != paths.session_id:
        result["session_id"] = paths.session_id
        save_transcript_data(paths.transcript, result)
    return result


def run_mux(cfg: AppConfig, paths: SessionPaths, *, force: bool = False) -> Path | None:
    """Свести видео + mix-аудио в *_final.mp4.

    Returns:
        Путь к финальному видео или None, если нечего сводить
        (нет видео/аудио) или файл уже существует (при force=False).
    """
    if not paths.video.exists() or not paths.mix_audio.exists():
        return None
    if paths.final_video.exists() and not force:
        return paths.final_video
    logger.info("Свожу видео + аудио → %s", paths.final_video.name)
    return mux_video(paths.video, paths.mix_audio, paths.final_video)


def run_report(cfg: AppConfig, paths: SessionPaths) -> tuple[Path, Path]:
    """Сгенерировать протокол, summary и необязательные артефакты для сессии."""
    data = _load_existing_transcript(paths)

    protocol_path = generate_protocol(data, paths, cfg)
    summary_path = generate_summary(data, paths, cfg)

    # Необязательные шаги — ошибки не блокируют pipeline
    _run_optional_step(
        "Не удалось сгенерировать ключевые моменты",
        lambda: generate_highlights(data, paths, cfg),
    )
    if cfg.recording.auto_mux:
        _run_optional_step(
            "Не удалось свести видео + аудио",
            lambda: run_mux(cfg, paths),
        )
    _run_optional_step(
        "Не удалось сгенерировать HTML-протокол",
        lambda: generate_html_protocol(data, paths, cfg),
    )

    return protocol_path, summary_path


def release_models(cfg: AppConfig) -> None:
    """Выгрузить модели транскрипции (если не keep_models_loaded)."""
    if not cfg.transcription.keep_models_loaded:
        unload_models()


def process_session(
    cfg: AppConfig,
    paths: SessionPaths,
) -> tuple[dict[str, Any], Path, Path]:
    """Транскрипция + отчёт для сессии (общая точка для CLI и трея)."""
    t0 = time.monotonic()
    try:
        transcript = run_transcribe(cfg, paths)
    finally:
        release_models(cfg)
    logger.info(
        "Транскрипция завершена за %.1f сек (%d сегментов)",
        time.monotonic() - t0,
        len(transcript.get("segments", [])),
    )
    t1 = time.monotonic()
    protocol_path, summary_path = run_report(cfg, paths)
    logger.info("Отчёт сгенерирован за %.1f сек", time.monotonic() - t1)
    return transcript, protocol_path, summary_path


# ---------------------------------------------------------------------------
# Обёртки по session_id (для CLI)
# ---------------------------------------------------------------------------


def run_html(cfg: AppConfig, session_id: str) -> Path:
    """Сгенерировать HTML-протокол для существующей сессии."""
    paths = resolve_session(cfg.output_dir, session_id)
    data = _load_existing_transcript(paths)
    return generate_html_protocol(data, paths, cfg)


def run_transcribe_only(cfg: AppConfig, session_id: str) -> dict[str, Any]:
    """Запустить только транскрипцию над существующей сессией."""
    paths = resolve_session(cfg.output_dir, session_id)
    if paths.transcript.exists():
        logger.info("Транскрипт уже существует: %s — перезаписываю", paths.transcript)
    try:
        return run_transcribe(cfg, paths)
    finally:
        release_models(cfg)


def run_report_only(cfg: AppConfig, session_id: str) -> tuple[Path, Path]:
    """Запустить только генерацию отчёта над существующей сессией."""
    paths = resolve_session(cfg.output_dir, session_id)
    return run_report(cfg, paths)


def run_highlights_only(cfg: AppConfig, session_id: str) -> Path:
    """Сгенерировать ключевые моменты для существующей сессии."""
    paths = resolve_session(cfg.output_dir, session_id)
    data = _load_existing_transcript(paths)
    return generate_highlights(data, paths, cfg)


def run_process(
    cfg: AppConfig,
    session_id: str,
) -> tuple[dict[str, Any], Path, Path]:
    """Запустить транскрипцию + отчёт для существующей сессии."""
    paths = resolve_session(cfg.output_dir, session_id)
    if paths.transcript.exists():
        logger.info("Транскрипт уже существует: %s — перезаписываю", paths.transcript)
    return process_session(cfg, paths)
