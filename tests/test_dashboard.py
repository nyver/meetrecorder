"""Тесты для dashboard.py — FastAPI веб-дашборд."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths, create_session

pytest.importorskip("fastapi", reason="fastapi not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path) -> AppConfig:
    c = AppConfig()
    c.output_dir = str(tmp_path)
    return c


@pytest.fixture
def app(cfg):
    from meeting_recorder.dashboard import create_app
    return create_app(cfg)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def _make_session(
    cfg: AppConfig,
    *,
    transcript: bool = False,
    summary: bool = False,
    protocol: bool = False,
    highlights: bool = False,
    video: bool = False,
    mix_audio: bool = False,
) -> SessionPaths:
    paths = create_session(cfg.output_dir)
    if transcript:
        data = {
            "session_id": paths.session_id,
            "language": "ru",
            "duration_sec": 120.0,
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Привет."},
                {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "Здравствуй."},
            ],
        }
        paths.transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    if summary:
        paths.summary.write_text(
            "# Summary\n\n## Краткое резюме\n\nКоманда обсудила план и утвердила бюджет.\n\n## Темы\n- Тема 1",
            encoding="utf-8",
        )
    if protocol:
        paths.protocol.write_text("# Протокол\n\n**[00:00]** SPEAKER_00: Привет.", encoding="utf-8")
    if highlights:
        hl = [
            {"title": "Ключевое решение", "description": "Утвердили бюджет Q3.", "start": 30.0},
        ]
        paths.highlights.write_text(json.dumps(hl, ensure_ascii=False), encoding="utf-8")
    if video:
        paths.final_video.write_bytes(b"fake_mp4_data_" * 10)
    if mix_audio:
        paths.mix_audio.write_bytes(b"fake_wav_data_" * 10)
    return paths


# ---------------------------------------------------------------------------
# GET /  — список сессий
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_empty_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_html_content_type(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_session_appears_in_list(self, cfg, client):
        s = _make_session(cfg)
        resp = client.get("/")
        assert resp.status_code == 200
        assert s.session_id in resp.text

    def test_session_with_transcript_shows_duration(self, cfg, client):
        _make_session(cfg, transcript=True)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_session_status_ready(self, cfg, client):
        _make_session(cfg, transcript=True, summary=True)
        resp = client.get("/")
        assert "Готово" in resp.text

    def test_session_status_partial(self, cfg, client):
        _make_session(cfg, transcript=True)
        resp = client.get("/")
        assert "Транскрибировано" in resp.text

    def test_session_status_recorded(self, cfg, client):
        _make_session(cfg, mix_audio=True)
        resp = client.get("/")
        assert "Записано" in resp.text

    def test_multiple_sessions(self, cfg, client):
        s1 = _make_session(cfg)
        s2 = _make_session(cfg, transcript=True, summary=True)
        resp = client.get("/")
        assert resp.status_code == 200
        assert s1.session_id in resp.text
        assert s2.session_id in resp.text


# ---------------------------------------------------------------------------
# GET /session/{session_id} — детали сессии
# ---------------------------------------------------------------------------

class TestSessionRoute:
    def test_not_found(self, client):
        resp = client.get("/session/meeting_2000-01-01_00-00-00")
        assert resp.status_code == 404

    def test_empty_session_returns_200(self, cfg, client):
        s = _make_session(cfg)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert s.session_id in resp.text

    def test_with_transcript(self, cfg, client):
        s = _make_session(cfg, transcript=True)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert "Привет" in resp.text
        assert "Здравствуй" in resp.text

    def test_with_summary(self, cfg, client):
        s = _make_session(cfg, transcript=True, summary=True)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert "Summary" in resp.text

    def test_with_protocol(self, cfg, client):
        s = _make_session(cfg, transcript=True, protocol=True)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert "Протокол" in resp.text

    def test_with_highlights(self, cfg, client):
        s = _make_session(cfg, transcript=True, highlights=True)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert "Ключевое решение" in resp.text

    def test_with_video_shows_player(self, cfg, client):
        s = _make_session(cfg, transcript=True, video=True)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert "player" in resp.text
        assert "video" in resp.text

    def test_with_mix_audio_shows_audio_player(self, cfg, client):
        s = _make_session(cfg, transcript=True, mix_audio=True)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert "audio" in resp.text

    def test_all_artifacts(self, cfg, client):
        s = _make_session(cfg, transcript=True, summary=True, protocol=True, highlights=True, video=True)
        resp = client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200

    def test_highlights_corrupt_json_graceful(self, cfg, client):
        s = _make_session(cfg, transcript=True)
        s.highlights.write_text("not json!!!", encoding="utf-8")
        resp = client.get(f"/session/{s.session_id}")
        # должно вернуть 200 даже при плохом highlights
        assert resp.status_code == 200

    def test_speaker_names_rendered(self, cfg, client):
        cfg.transcription.speaker_names = {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}
        # пересоздаём app и client с новым cfg
        from fastapi.testclient import TestClient
        from meeting_recorder.dashboard import create_app
        local_client = TestClient(create_app(cfg))
        s = _make_session(cfg, transcript=True)
        resp = local_client.get(f"/session/{s.session_id}")
        assert resp.status_code == 200
        assert "Иван" in resp.text


# ---------------------------------------------------------------------------
# GET /media/{session_id}/{filename} — стриминг файлов
# ---------------------------------------------------------------------------

class TestMediaRoute:
    def test_session_not_found(self, client):
        resp = client.get("/media/meeting_2000-01-01_00-00-00/file.mp4")
        assert resp.status_code == 404

    def test_file_not_found(self, cfg, client):
        s = _make_session(cfg)
        resp = client.get(f"/media/{s.session_id}/nonexistent.mp4")
        assert resp.status_code == 404

    def test_file_served(self, cfg, client):
        s = _make_session(cfg, video=True)
        resp = client.get(f"/media/{s.session_id}/{s.final_video.name}")
        assert resp.status_code == 200
        assert resp.content == b"fake_mp4_data_" * 10

    def test_audio_served(self, cfg, client):
        s = _make_session(cfg, mix_audio=True)
        resp = client.get(f"/media/{s.session_id}/{s.mix_audio.name}")
        assert resp.status_code == 200

    def test_path_traversal_blocked(self, cfg, client):
        s = _make_session(cfg)
        # Создаём файл вне директории сессии
        outside = Path(cfg.output_dir) / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        # Пытаемся получить файл выше по дереву
        resp = client.get(f"/media/{s.session_id}/../secret.txt")
        assert resp.status_code in (403, 404, 422)


# ---------------------------------------------------------------------------
# _session_meta — вспомогательные сценарии
# ---------------------------------------------------------------------------

class TestSessionMeta:
    def test_empty_session_meta(self, cfg, client):
        """Сессия без файлов — статус 'Пусто'."""
        s = _make_session(cfg)
        resp = client.get("/")
        assert "Пусто" in resp.text

    def test_size_mb_shown(self, cfg, client):
        s = _make_session(cfg, video=True)
        resp = client.get("/")
        assert "МБ" in resp.text

    def test_duration_shown(self, cfg, client):
        s = _make_session(cfg, transcript=True)
        resp = client.get("/")
        assert "мин" in resp.text

    def test_speakers_shown(self, cfg, client):
        s = _make_session(cfg, transcript=True)
        resp = client.get("/")
        assert "SPEAKER" in resp.text
