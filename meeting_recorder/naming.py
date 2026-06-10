"""Генерация session_id, имён файлов и путей артефактов."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path


def _generate_session_id(suffix: int = 0) -> str:
    now = dt.datetime.now()
    base = f"meeting_{now:%Y-%m-%d_%H-%M-%S}"
    if suffix == 0:
        return base
    return f"{base}_{suffix}"


def _ensure_unique_session_id(output_dir: str) -> str:
    """Генерировать session_id, проверяя коллизии в output_dir.

    При коллизии суффиксы начинаются с _2, _3, … (FR-10).
    """
    base_id = _generate_session_id(0)
    if not (Path(output_dir) / base_id).exists():
        return base_id

    # Коллизия — начинаем с суффикса 2
    idx = 2
    while True:
        sid = f"{base_id}_{idx}"
        if not (Path(output_dir) / sid).exists():
            return sid
        idx += 1


# ---------------------------------------------------------------------------
# Структура путей сессии
# ---------------------------------------------------------------------------


class SessionPaths:
    """Полный набор путей для одной сессии."""

    def __init__(self, output_dir: str, session_id: str):
        self.output_dir = output_dir
        self.session_id = session_id
        self.dir = Path(output_dir) / session_id
        self.video = self.dir / f"{session_id}.mp4"
        self.mic_audio = self.dir / f"{session_id}_mic.wav"
        self.system_audio = self.dir / f"{session_id}_system.wav"
        self.mix_audio = self.dir / f"{session_id}_mix.wav"
        self.ffmpeg_log = self.dir / f"{session_id}_ffmpeg.log"
        self.final_video = self.dir / f"{session_id}_final.mp4"
        self.transcript = self.dir / f"{session_id}_transcript.json"
        self.protocol = self.dir / f"{session_id}_protocol.md"
        self.summary = self.dir / f"{session_id}_summary.md"

    def ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)


def create_session(output_dir: str) -> SessionPaths:
    """Создать новую сессию с уникальным session_id и вернуть SessionPaths."""
    session_id = _ensure_unique_session_id(output_dir)
    paths = SessionPaths(output_dir, session_id)
    paths.ensure_dir()
    return paths


def list_sessions(output_dir: str) -> list[SessionPaths]:
    """Вернуть список всех сессий в output_dir, отсортированный по session_id."""
    root = Path(output_dir)
    result: list[SessionPaths] = []
    if not root.is_dir():
        return result
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("meeting_"):
            result.append(SessionPaths(output_dir, subdir.name))
    return result


def resolve_session(output_dir: str, session_id: str) -> SessionPaths:
    """Найти существующую сессию или поднять ошибку."""
    session_dir = Path(output_dir) / session_id
    if not session_dir.is_dir():
        raise FileNotFoundError(
            f"Сессия '{session_id}' не найдена в '{output_dir}'"
        )
    return SessionPaths(output_dir, session_id)
