"""Межпроцессное состояние записи: каталог состояния, state-файл и lock записи.

Используется и CLI (`mrec start` / `mrec stop`), и треем: запись в один момент
времени может вести только один процесс.

Пока процесс записи жив, он удерживает эксклюзивную блокировку на LOCK_FILE.
ОС снимает блокировку при завершении процесса (в т.ч. аварийном), поэтому
состояние «запись идёт» нельзя спутать с устаревшим state-файлом.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import IO

OWNER_CLI = "cli"
OWNER_TRAY = "tray"


def app_data_dir() -> Path:
    """Каталог данных приложения (%APPDATA%\\MeetingRecorder / XDG state)."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return (
            Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        ) / "MeetingRecorder"
    xdg = os.environ.get("XDG_STATE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "state") / "MeetingRecorder"


def default_state_dir() -> Path:
    """Вернуть user-writable директорию для файлов состояния."""
    override = os.environ.get("MEETING_RECORDER_STATE_DIR")
    if override:
        return Path(override)

    candidate = app_data_dir() / ".state"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError:
        return Path.cwd() / ".meeting_recorder_state"


STATE_DIR = default_state_dir()
STATE_FILE = STATE_DIR / "active_session.json"
STOP_FILE = STATE_DIR / "stop_requested"
DONE_FILE = STATE_DIR / "recording_done"
LOCK_FILE = STATE_DIR / "recording.lock"


# ---------------------------------------------------------------------------
# State-файл активной сессии
# ---------------------------------------------------------------------------


def save_state(
    session_id: str,
    video_path: str,
    ffmpeg_pid: int | None = None,
    owner: str = OWNER_CLI,
    path: Path | None = None,
) -> None:
    path = path or STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "video_path": video_path,
                "ffmpeg_pid": ffmpeg_pid,
                "owner": owner,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_state(path: Path | None = None) -> dict | None:
    path = path or STATE_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def remove_state(path: Path | None = None) -> None:
    try:
        (path or STATE_FILE).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Lock записи
# ---------------------------------------------------------------------------


def try_lock(fh: IO) -> bool:
    try:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def unlock(fh: IO) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def acquire_recording_lock(path: Path | None = None) -> IO | None:
    """Захватить lock записи. Возвращает дескриптор или None, если lock занят."""
    path = path or LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    if try_lock(fh):
        return fh
    fh.close()
    return None


def release_recording_lock(fh: IO | None) -> None:
    if fh is not None:
        unlock(fh)
        fh.close()


def is_recording_active(path: Path | None = None) -> bool:
    """True, если какой-либо процесс сейчас ведёт запись (держит lock)."""
    fh = acquire_recording_lock(path)
    if fh is None:
        return True
    release_recording_lock(fh)
    return False
