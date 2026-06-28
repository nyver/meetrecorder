"""Оркестрация этапов пайплайна: запись → транскрипция → протокол → summary."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import AppConfig
from .naming import SessionPaths, create_session, resolve_session
from .recorder import MeetingRecorder, mix_audio_files
from .html_report import generate_html_protocol
from .report import generate_highlights, generate_protocol, generate_summary
from .session_utils import load_transcript_data, save_transcript_data
from .transcriber import transcribe

logger = logging.getLogger(__name__)
T = TypeVar("T")


class PipelineError(Exception):
    """Ошибка пайплайна."""


def _ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
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


def _check_video_length(paths: "SessionPaths") -> None:
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
                "Для более надёжного захвата смените screen_grabber: ddagrab в config.yaml.",
                video_dur, audio_dur,
            )
    except Exception as exc:
        logger.debug("_check_video_length: %s", exc)


def run_record(cfg: AppConfig) -> SessionPaths:
    """Запустить запись и вернуть SessionPaths."""
    paths = create_session(cfg.output_dir)
    recorder = MeetingRecorder(cfg, paths)

    try:
        recorder.start()
        logger.info("Запись запущена. Нажми stop для завершения.")

        # Ждём, пока запись не будет остановлена снаружи
        # В CLI режиме это реализуется через отдельный поток/сигнал
        while recorder.is_recording:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Получен KeyboardInterrupt — останавливаю запись")
    finally:
        recorder.stop()

    # Проверяем, что видео не обрезалось аварийно
    _check_video_length(paths)

    # Сведение аудио
    logger.info("Свожу аудио: mic + system → mix")
    if paths.mic_audio.exists() and paths.system_audio.exists():
        mix_audio_files(
            paths.mic_audio,
            paths.system_audio,
            paths.mix_audio,
            cfg.recording.audio_sample_rate,
        )
    elif paths.mic_audio.exists():
        # Если нет системного звука — используем микрофон
        try:
            paths.mic_audio.rename(paths.mix_audio)
        except OSError as exc:
            raise PipelineError(f"Не удалось переименовать аудиофайл: {exc}") from exc
        logger.warning("Системный звук отсутствует, используем микрофон для микса")
    else:
        raise PipelineError(
            "Не удалось создать аудиофайлы. Проверьте настройку устройств."
        )

    return paths


def run_transcribe(
    cfg: AppConfig,
    paths: SessionPaths,
) -> dict[str, Any]:
    """Транскрибировать сессию."""
    audio_path = paths.mix_audio
    if not audio_path.exists():
        raise PipelineError(
            f"Аудиофайл не найден: {audio_path}. Сначала выполните запись."
        )

    result = transcribe(
        audio_path,
        cfg,
        output_path=paths.transcript,
    )
    result["session_id"] = paths.session_id

    # Обновляем JSON с session_id
    save_transcript_data(paths.transcript, result)
    return result


def run_report(
    cfg: AppConfig,
    paths: SessionPaths,
) -> tuple[Path, Path]:
    """Сгенерировать протокол и summary для сессии."""
    data = _load_existing_transcript(paths)

    protocol_path = generate_protocol(data, paths, cfg)
    summary_path = generate_summary(data, paths, cfg)

    # Ключевые моменты — необязательный шаг, ошибки не блокируют pipeline
    _run_optional_step(
        "Не удалось сгенерировать ключевые моменты",
        lambda: generate_highlights(data, paths, cfg),
    )

    # HTML-протокол генерируется как необязательный шаг — ошибки не блокируют pipeline
    _run_optional_step(
        "Не удалось сгенерировать HTML-протокол",
        lambda: generate_html_protocol(data, paths, cfg),
    )

    return protocol_path, summary_path


def run_html(
    cfg: AppConfig,
    session_id: str,
) -> Path:
    """Сгенерировать HTML-протокол для существующей сессии."""
    paths = resolve_session(cfg.output_dir, session_id)

    data = _load_existing_transcript(paths)
    return generate_html_protocol(data, paths, cfg)


def run_pipeline(
    cfg: AppConfig,
    stop_callback=None,
) -> SessionPaths:
    """Запустить полный пайплайн: запись → транскрипция → протокол → summary.

    Args:
        cfg: конфигурация приложения.
        stop_callback: callable, вызываемый когда нужно остановить запись (для CLI).
    """
    logger.info("=" * 60)
    logger.info("Начинаю полный пайплайн: %s", cfg.llm.backend)
    logger.info("=" * 60)

    t0 = time.monotonic()

    # Шаг 1: Запись
    logger.info("=== ШАГ 1/3: Запись ===")
    t_start = time.monotonic()
    try:
        paths = run_record(cfg)
    except Exception as e:
        raise PipelineError(f"Ошибка записи: {e}") from e
    logger.info("Шаг 1 завершён за %.1f сек", time.monotonic() - t_start)

    # Шаг 2: Транскрипция
    logger.info("=== ШАГ 2/3: Транскрипция ===")
    t_start = time.monotonic()
    try:
        run_transcribe(cfg, paths)
    except Exception as e:
        raise PipelineError(f"Ошибка транскрипции: {e}") from e
    logger.info("Шаг 2 завершён за %.1f сек", time.monotonic() - t_start)

    # Шаг 3: Протокол + Summary
    logger.info("=== ШАГ 3/3: Протокол + Summary ===")
    t_start = time.monotonic()
    try:
        run_report(cfg, paths)
    except Exception as e:
        raise PipelineError(f"Ошибка генерации отчёта: {e}") from e
    logger.info("Шаг 3 завершён за %.1f сек", time.monotonic() - t_start)

    total = time.monotonic() - t0
    logger.info("=" * 60)
    logger.info(
        "Пайплайн завершён за %.1f сек. Артефакты в: %s",
        total, paths.dir,
    )
    logger.info("  Протокол: %s", paths.protocol)
    logger.info("  Summary:  %s", paths.summary)
    logger.info("=" * 60)

    return paths


def run_transcribe_only(
    cfg: AppConfig,
    session_id: str,
) -> dict[str, Any]:
    """Запустить только транскрипцию над существующей сессией."""
    paths = resolve_session(cfg.output_dir, session_id)

    if paths.transcript.exists():
        logger.info("Транскрипт уже существует: %s — перезаписываю", paths.transcript)

    return run_transcribe(cfg, paths)


def run_report_only(
    cfg: AppConfig,
    session_id: str,
) -> tuple[Path, Path]:
    """Запустить только генерацию отчёта над существующей сессией."""
    paths = resolve_session(cfg.output_dir, session_id)

    protocol_path, summary_path = run_report(cfg, paths)
    return protocol_path, summary_path


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
    transcript = run_transcribe_only(cfg, session_id)
    protocol_path, summary_path = run_report_only(cfg, session_id)
    return transcript, protocol_path, summary_path


def transcriber_load_transcript(path: Path | str) -> dict[str, Any]:
    """Импортировать без цикла зависимостей."""
    from .transcriber import load_transcript
    return load_transcript(path)
