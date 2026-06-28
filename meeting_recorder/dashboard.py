"""Локальный веб-дашборд Meeting Recorder."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AppConfig
from .naming import SessionPaths

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_SPEAKER_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#9333ea",
    "#ea580c", "#0891b2", "#db2777", "#65a30d",
    "#b45309", "#0f766e",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _parse_dt(session_id: str) -> datetime | None:
    try:
        parts = session_id.split("_")
        return datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None


def _fmt_ts(seconds: float) -> str:
    """Секунды → MM:SS для Jinja2-фильтра."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _session_meta(paths: SessionPaths, cfg: AppConfig) -> dict[str, Any]:
    """Собрать метаданные сессии для шаблона."""
    dt = _parse_dt(paths.session_id)
    meta: dict[str, Any] = {
        "session_id": paths.session_id,
        "date": dt.strftime("%Y-%m-%d") if dt else "",
        "time": dt.strftime("%H:%M") if dt else "",
        "duration_min": None,
        "speakers": [],
        "has_video": paths.final_video.exists() or paths.video.exists(),
        "has_audio": paths.mix_audio.exists(),
        "has_transcript": paths.transcript.exists(),
        "has_protocol": paths.protocol.exists(),
        "has_summary": paths.summary.exists(),
        "size_mb": 0.0,
    }

    if paths.transcript.exists():
        try:
            data = json.loads(paths.transcript.read_text(encoding="utf-8"))
            meta["duration_min"] = round(data.get("duration_sec", 0) / 60, 1)
            names = cfg.transcription.speaker_names
            unique = sorted({seg.get("speaker", "?") for seg in data.get("segments", [])})
            meta["speakers"] = [names.get(s, s) for s in unique]
        except Exception:
            pass

    try:
        total = sum(f.stat().st_size for f in paths.dir.iterdir() if f.is_file())
        meta["size_mb"] = round(total / 1024 / 1024, 1)
    except Exception:
        pass

    if meta["has_summary"]:
        meta["status"] = "Готово"
        meta["status_cls"] = "s-ready"
    elif meta["has_transcript"]:
        meta["status"] = "Транскрибировано"
        meta["status_cls"] = "s-partial"
    elif meta["has_audio"] or meta["has_video"]:
        meta["status"] = "Записано"
        meta["status_cls"] = "s-recorded"
    else:
        meta["status"] = "Пусто"
        meta["status_cls"] = "s-empty"

    return meta


def _pick_media(paths: SessionPaths) -> tuple[str, str] | None:
    """Выбрать лучший доступный медиафайл. Возвращает (filename, media_type)."""
    if paths.final_video.exists():
        return paths.final_video.name, "video"
    if paths.mix_audio.exists():
        return paths.mix_audio.name, "audio"
    if paths.video.exists():
        return paths.video.name, "video"
    return None


# ---------------------------------------------------------------------------
# FastAPI-приложение
# ---------------------------------------------------------------------------


def create_app(cfg: AppConfig) -> Any:
    """Создать и вернуть FastAPI-приложение дашборда."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.templating import Jinja2Templates

    from .html_report import _md_to_html
    from .naming import list_sessions, resolve_session

    app = FastAPI(title="Meeting Recorder", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["ts"] = _fmt_ts

    # ------------------------------------------------------------------ #
    #  GET /  — список всех сессий                                         #
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        all_paths = list_sessions(cfg.output_dir)
        sessions = [_session_meta(p, cfg) for p in reversed(all_paths)]
        return templates.TemplateResponse(request, "index.html", {
            "sessions": sessions,
            "output_dir": cfg.output_dir,
        })

    # ------------------------------------------------------------------ #
    #  GET /session/{id}  — детали сессии                                  #
    # ------------------------------------------------------------------ #

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    async def session_view(request: Request, session_id: str) -> HTMLResponse:
        try:
            paths = resolve_session(cfg.output_dir, session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Сессия не найдена")

        meta = _session_meta(paths, cfg)

        summary_html = (
            _md_to_html(paths.summary.read_text(encoding="utf-8"))
            if paths.summary.exists() else ""
        )
        protocol_html = (
            _md_to_html(paths.protocol.read_text(encoding="utf-8"))
            if paths.protocol.exists() else ""
        )

        transcript: list[dict[str, Any]] = []
        if paths.transcript.exists():
            data = json.loads(paths.transcript.read_text(encoding="utf-8"))
            names = cfg.transcription.speaker_names
            unique_spk = sorted({seg.get("speaker", "?") for seg in data.get("segments", [])})
            colors = {
                sp: _SPEAKER_COLORS[i % len(_SPEAKER_COLORS)]
                for i, sp in enumerate(unique_spk)
            }
            for seg in data.get("segments", []):
                sp = seg.get("speaker", "?")
                transcript.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": names.get(sp, sp),
                    "color": colors.get(sp, "#333"),
                    "text": seg.get("text", "").strip(),
                })

        highlights: list[dict[str, Any]] = []
        if paths.highlights.exists():
            try:
                highlights = json.loads(paths.highlights.read_text(encoding="utf-8"))
            except Exception:
                pass

        media = _pick_media(paths)
        media_url = f"/media/{session_id}/{media[0]}" if media else None
        media_type = media[1] if media else None

        return templates.TemplateResponse(request, "session.html", {
            "meta": meta,
            "session_id": session_id,
            "summary_html": summary_html,
            "protocol_html": protocol_html,
            "transcript": transcript,
            "highlights": highlights,
            "media_url": media_url,
            "media_type": media_type,
        })

    # ------------------------------------------------------------------ #
    #  GET /media/{id}/{file}  — стриминг медиафайла                       #
    # ------------------------------------------------------------------ #

    @app.get("/media/{session_id}/{filename}")
    async def serve_media(session_id: str, filename: str) -> FileResponse:
        try:
            paths = resolve_session(cfg.output_dir, session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404)

        safe_root = paths.dir.resolve()
        file_path = (safe_root / filename).resolve()

        # Защита от path traversal
        if not file_path.is_relative_to(safe_root):
            raise HTTPException(status_code=403)
        if not file_path.exists():
            raise HTTPException(status_code=404)

        return FileResponse(str(file_path))

    return app
