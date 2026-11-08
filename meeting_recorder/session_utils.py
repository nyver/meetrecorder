"""Shared helpers for session artifacts."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .naming import SessionPaths

logger = logging.getLogger(__name__)

TranscriptSource = dict[str, Any] | Path | str


_REPLACE_RETRIES = 10
_REPLACE_DELAY = 0.3  # секунд между попытками


def replace_with_retry(
    src: Path | str,
    dst: Path | str,
    retries: int = _REPLACE_RETRIES,
    delay: float = _REPLACE_DELAY,
) -> None:
    """os.replace с повторами при PermissionError.

    На Windows замена падает, пока целевой файл открыт другим процессом без
    FILE_SHARE_DELETE (дашборд отдаёт файл, антивирус, индексатор). Обычно
    блокировка снимается за доли секунды.
    """
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt >= retries - 1:
                raise
            time.sleep(delay)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Записать файл атомарно: во временный файл рядом + os.replace.

    При падении процесса во время записи исходный файл остаётся целым.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
        replace_with_retry(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_transcript_data(transcript: TranscriptSource) -> dict[str, Any]:
    """Return transcript data from an in-memory dict or a JSON file path."""
    if isinstance(transcript, dict):
        return transcript

    path = Path(transcript)
    if not path.exists():
        raise FileNotFoundError(f"Транскрипт не найден: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_transcript_data(path: Path, data: dict[str, Any]) -> None:
    """Write transcript JSON with the project's standard formatting."""
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def parse_seconds(value: Any) -> float:
    """Привести таймкод к секундам: 12.5, "12.5", "05:30", "01:02:03" → float.

    Некорректные значения дают 0.0 — LLM иногда возвращает таймкоды строками.
    """
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    if isinstance(value, str):
        text = value.strip()
        try:
            if ":" in text:
                total = 0.0
                for part in text.split(":"):
                    total = total * 60 + float(part)
                return max(total, 0.0)
            return max(float(text), 0.0)
        except ValueError:
            return 0.0
    return 0.0


def normalize_highlights(raw: Any) -> list[dict[str, Any]]:
    """Отфильтровать и нормализовать список highlights из ответа LLM / файла.

    Каждый элемент приводится к виду {title: str, description: str, start: float};
    элементы не-словари отбрасываются.
    """
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "title": str(item.get("title", "") or ""),
                "description": str(item.get("description", "") or ""),
                "start": parse_seconds(item.get("start", 0)),
            }
        )
    return result


def load_highlights(path: Path) -> list[dict[str, Any]]:
    """Прочитать *_highlights.json; при отсутствии или ошибке — пустой список."""
    if not path.exists():
        return []
    try:
        return normalize_highlights(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Не удалось загрузить highlights %s: %s", path, exc)
        return []


def normalize_segments(
    segments: Iterable[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Copy transcript segments and apply configured speaker aliases.

    Таймкоды приводятся к float, текст и метки — к str: данные из JSON
    подставляются в HTML/JS (onclick="seek(...)"), поэтому типы проверяются.
    """
    aliases = speaker_names or {}
    normalized: list[dict[str, Any]] = []
    for seg in segments:
        speaker = str(seg.get("speaker", "UNKNOWN"))
        normalized.append(
            {
                "start": parse_seconds(seg["start"]),
                "end": parse_seconds(seg["end"]),
                "speaker": str(aliases.get(speaker, speaker)),
                "text": str(seg.get("text", "") or "").strip(),
            }
        )
    return normalized


def transcript_lines(
    segments: Iterable[dict[str, Any]],
    *,
    speaker_names: dict[str, str] | None = None,
    timestamp_formatter,
) -> list[str]:
    """Format transcript segments as timestamped plain-text lines."""
    return [
        f"[{timestamp_formatter(seg['start'])}] {seg['speaker']}: {seg['text']}"
        for seg in normalize_segments(segments, speaker_names)
    ]


def unique_speakers(
    segments: Iterable[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> list[str]:
    """Return sorted speaker names after aliasing."""
    return sorted(
        {seg["speaker"] for seg in normalize_segments(segments, speaker_names)}
    )


def pick_media(paths: SessionPaths, *, mime: bool = True) -> tuple[str, str] | None:
    """Choose the best available media file for a session."""
    if paths.final_video.exists():
        return paths.final_video.name, "video/mp4" if mime else "video"
    if paths.mix_audio.exists():
        return paths.mix_audio.name, "audio/wav" if mime else "audio"
    if paths.video.exists():
        return paths.video.name, "video/mp4" if mime else "video"
    return None


def strip_code_fences(text: str) -> str:
    """Remove wrapping Markdown code fences commonly returned by LLMs."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            line for line in stripped.splitlines() if not line.startswith("```")
        ).strip()
    return stripped
