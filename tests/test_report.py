"""Юнит-тесты для report.py (протокол, без LLM)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_recorder.report import generate_protocol, _format_timestamp
from meeting_recorder.naming import SessionPaths
from meeting_recorder.config import AppConfig


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0) == "00:00"

    def test_one_minute(self):
        assert _format_timestamp(60) == "01:00"

    def test_one_second_past_minute(self):
        assert _format_timestamp(61) == "01:01"

    def test_ten_minutes(self):
        assert _format_timestamp(600) == "10:00"

    def test_forty_five_seconds(self):
        assert _format_timestamp(45) == "00:45"


class TestGenerateProtocol:
    def test_basic_protocol(self, tmp_path):
        cfg = AppConfig()
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-05_14-30-12")
        paths.ensure_dir()

        transcript = {
            "session_id": paths.session_id,
            "language": "ru",
            "duration_sec": 30.0,
            "segments": [
                {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Привет, всем привет."},
                {"start": 5.5, "end": 10.0, "speaker": "SPEAKER_01", "text": "Здравствуй!"},
                {"start": 10.5, "end": 15.0, "speaker": "SPEAKER_00", "text": "Давай обсудим план."},
            ],
        }

        protocol_path = generate_protocol(transcript, paths, cfg)
        assert protocol_path.exists()

        content = protocol_path.read_text(encoding="utf-8")
        assert "Привет, всем привет." in content
        assert "Здравствуй!" in content
        assert "Давай обсудим план." in content
        assert "00:01" in content
        assert "00:05" in content
        assert "00:10" in content

    def test_protocol_with_speaker_names(self, tmp_path):
        cfg = AppConfig()
        cfg.transcription.speaker_names = {
            "SPEAKER_00": "Иван",
            "SPEAKER_01": "Мария",
        }
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-05_14-30-12")
        paths.ensure_dir()

        transcript = {
            "language": "ru",
            "duration_sec": 10.0,
            "segments": [
                {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Привет, Мария."},
                {"start": 5.5, "end": 10.0, "speaker": "SPEAKER_01", "text": "Привет, Иван!"},
            ],
        }

        protocol_path = generate_protocol(transcript, paths, cfg)
        content = protocol_path.read_text(encoding="utf-8")
        assert "Иван:" in content
        assert "Мария:" in content

    def test_protocol_from_file(self, tmp_path):
        cfg = AppConfig()
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-05_14-30-12")
        paths.ensure_dir()

        transcript = {
            "language": "ru",
            "duration_sec": 10.0,
            "segments": [
                {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Тест."},
            ],
        }

        # Сначала сохраняем JSON
        json_path = paths.dir / "test_transcript.json"
        import json
        json_path.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        protocol_path = generate_protocol(str(json_path), paths, cfg)
        assert protocol_path.exists()
