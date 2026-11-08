"""Генерация session_id, имён файлов и путей артефактов."""

from __future__ import annotations

import datetime as dt
import re
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
        self.html_protocol = self.dir / f"{session_id}_protocol.html"
        self.highlights = self.dir / f"{session_id}_highlights.json"

    def ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)


def create_session(output_dir: str) -> SessionPaths:
    """Создать новую сессию с уникальным session_id и вернуть SessionPaths.

    Каталог создаётся через mkdir(exist_ok=False): проверка уникальности и
    создание — одна атомарная операция ФС, поэтому два одновременных старта
    (CLI и трей в одну секунду) не получат один и тот же каталог.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_id = _generate_session_id(0)
    idx = 1
    while True:
        session_id = base_id if idx == 1 else f"{base_id}_{idx}"
        try:
            (root / session_id).mkdir()
        except FileExistsError:
            idx += 1
            continue
        return SessionPaths(output_dir, session_id)


def list_sessions(output_dir: str) -> list[SessionPaths]:
    """Вернуть список всех сессий в output_dir, отсортированный по session_id."""
    root = Path(output_dir)
    result: list[SessionPaths] = []
    if not root.is_dir():
        return result
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and is_valid_session_id(subdir.name):
            result.append(SessionPaths(output_dir, subdir.name))
    return result


_SESSION_ID_RE = re.compile(r"^meeting_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_\d+)?$")


def is_valid_session_id(session_id: str) -> bool:
    """True, если session_id имеет формат meeting_YYYY-MM-DD_HH-MM-SS[_N].

    Строгая проверка формата исключает path traversal (``..``, ``\\``, ``/``)
    при использовании session_id, пришедшего извне (CLI, HTTP).
    """
    return bool(_SESSION_ID_RE.fullmatch(session_id))


def resolve_session(output_dir: str, session_id: str) -> SessionPaths:
    """Найти существующую сессию или поднять ошибку.

    Raises:
        FileNotFoundError: session_id некорректен или каталог сессии не найден.
    """
    if not is_valid_session_id(session_id):
        raise FileNotFoundError(f"Некорректный идентификатор сессии: '{session_id}'")
    root = Path(output_dir).resolve()
    session_dir = Path(output_dir) / session_id
    if session_dir.resolve().parent != root:
        raise FileNotFoundError(f"Сессия '{session_id}' вне каталога '{output_dir}'")
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Сессия '{session_id}' не найдена в '{output_dir}'")
    return SessionPaths(output_dir, session_id)
