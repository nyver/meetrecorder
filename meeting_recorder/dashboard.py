"""Локальный веб-дашборд Meeting Recorder.

Без `from __future__ import annotations`: FastAPI должен видеть реальные
аннотации параметров (Request импортируется внутри create_app).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AppConfig
from .naming import SessionPaths
from .report import _format_timestamp as _fmt_ts
from .report import _session_datetime as _parse_dt_impl
from .session_utils import load_highlights, normalize_segments, pick_media

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_SPEAKER_COLORS = [
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#b45309",
    "#0f766e",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _parse_dt(session_id: str) -> datetime | None:
    try:
        return _parse_dt_impl(session_id)
    except Exception:
        return None


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
            unique = sorted(
                {seg.get("speaker", "?") for seg in data.get("segments", [])}
            )
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
    return pick_media(paths, mime=False)


# ---------------------------------------------------------------------------
# FastAPI-приложение
# ---------------------------------------------------------------------------


_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def create_app(cfg: AppConfig, allowed_hosts: "list[str] | None" = None) -> Any:
    """Создать и вернуть FastAPI-приложение дашборда.

    Args:
        allowed_hosts: допустимые значения заголовка Host. Защищает от DNS
            rebinding (чужая веб-страница, резолвящая свой домен в 127.0.0.1,
            не сможет читать ответы дашборда). None — без проверки.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.templating import Jinja2Templates

    from .html_report import _md_to_html
    from .naming import list_sessions, resolve_session

    app = FastAPI(title="Meeting Recorder", docs_url=None, redoc_url=None)
    if allowed_hosts:
        from starlette.middleware.trustedhost import TrustedHostMiddleware

        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=sorted(set(allowed_hosts) | set(_LOOPBACK_HOSTS)),
        )
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["ts"] = _fmt_ts

    # ------------------------------------------------------------------ #
    #  GET /  — список всех сессий                                         #
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        all_paths = list_sessions(cfg.output_dir)
        sessions = [_session_meta(p, cfg) for p in reversed(all_paths)]
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "sessions": sessions,
                "output_dir": cfg.output_dir,
            },
        )

    # ------------------------------------------------------------------ #
    #  GET /session/{id}  — детали сессии                                  #
    # ------------------------------------------------------------------ #

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    def session_view(request: Request, session_id: str) -> HTMLResponse:
        try:
            paths = resolve_session(cfg.output_dir, session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Сессия не найдена") from None

        # Защита от path traversal (аналогично /media/)
        safe_root = Path(cfg.output_dir).resolve()
        if not paths.dir.resolve().is_relative_to(safe_root):
            raise HTTPException(status_code=403)

        meta = _session_meta(paths, cfg)

        summary_html = (
            _md_to_html(paths.summary.read_text(encoding="utf-8"))
            if paths.summary.exists()
            else ""
        )
        protocol_html = (
            _md_to_html(paths.protocol.read_text(encoding="utf-8"))
            if paths.protocol.exists()
            else ""
        )

        transcript: list[dict[str, Any]] = []
        if paths.transcript.exists():
            try:
                data = json.loads(paths.transcript.read_text(encoding="utf-8"))
                unique_spk = sorted(
                    {seg.get("speaker", "?") for seg in data.get("segments", [])}
                )
                colors = {
                    sp: _SPEAKER_COLORS[i % len(_SPEAKER_COLORS)]
                    for i, sp in enumerate(unique_spk)
                }
                normalized = normalize_segments(
                    data.get("segments", []), cfg.transcription.speaker_names
                )
                for raw_seg, seg in zip(
                    data.get("segments", []), normalized, strict=True
                ):
                    sp = raw_seg.get("speaker", "?")
                    transcript.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "speaker": seg["speaker"],
                            "color": colors.get(sp, "#333"),
                            "text": seg["text"],
                        }
                    )
            except Exception:
                logger.warning("Не удалось прочитать транскрипт: %s", paths.transcript)

        highlights = load_highlights(paths.highlights)

        media = _pick_media(paths)
        media_url = f"/media/{session_id}/{media[0]}" if media else None
        media_type = media[1] if media else None

        return templates.TemplateResponse(
            request,
            "session.html",
            {
                "meta": meta,
                "session_id": session_id,
                "summary_html": summary_html,
                "protocol_html": protocol_html,
                "transcript": transcript,
                "highlights": highlights,
                "media_url": media_url,
                "media_type": media_type,
            },
        )

    # ------------------------------------------------------------------ #
    #  GET /media/{id}/{file}  — стриминг медиафайла                       #
    # ------------------------------------------------------------------ #

    @app.get("/media/{session_id}/{filename}")
    def serve_media(session_id: str, filename: str) -> FileResponse:
        try:
            paths = resolve_session(cfg.output_dir, session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404) from None

        # Защита от path traversal: каталог сессии обязан лежать прямо в
        # output_dir, а запрошенный файл — прямо в каталоге сессии.
        output_root = Path(cfg.output_dir).resolve()
        safe_root = paths.dir.resolve()
        if safe_root.parent != output_root:
            raise HTTPException(status_code=403)
        file_path = (safe_root / filename).resolve()
        if file_path.parent != safe_root:
            raise HTTPException(status_code=403)
        if not file_path.is_file():
            raise HTTPException(status_code=404)

        return FileResponse(str(file_path))

    return app
