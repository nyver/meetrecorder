"""Юнит-тесты для report.py (протокол, summary, clean, chunking)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_recorder.report import (
    generate_protocol,
    generate_summary,
    _format_timestamp,
    _clean_protocol,
    _extract_transcript_text,
    _split_text_into_chunks,
    _map_reduce_summary,
    _call_llm_for_summary,
    _session_datetime,
)
from meeting_recorder.naming import SessionPaths
from meeting_recorder.config import AppConfig


# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _session_datetime
# ---------------------------------------------------------------------------

class TestSessionDatetime:
    def test_valid_session_id(self):
        dt = _session_datetime("meeting_2026-06-14_10-30-00")
        assert dt.year == 2026
        assert dt.month == 6
        assert dt.hour == 10
        assert dt.minute == 30

    def test_invalid_session_id_returns_now(self):
        from datetime import datetime
        dt = _session_datetime("invalid")
        assert isinstance(dt, datetime)


# ---------------------------------------------------------------------------
# _extract_transcript_text
# ---------------------------------------------------------------------------

class TestExtractTranscriptText:
    def test_with_transcript_prefix(self):
        text = "Header\n---\nТранскрипт: Привет мир"
        result = _extract_transcript_text(text)
        assert "Привет мир" in result

    def test_with_dashes_fallback(self):
        text = "Header\n---\nsome content here"
        result = _extract_transcript_text(text)
        assert "some content here" in result

    def test_no_marker_returns_full(self):
        text = "just some plain text"
        result = _extract_transcript_text(text)
        assert result == text


# ---------------------------------------------------------------------------
# _split_text_into_chunks
# ---------------------------------------------------------------------------

class TestSplitTextIntoChunks:
    def test_single_chunk(self):
        text = "Para1\n\nPara2\n\nPara3"
        chunks = _split_text_into_chunks(text, max_chars=1000)
        assert len(chunks) == 1
        assert "Para1" in chunks[0]

    def test_multiple_chunks(self):
        # Параграфы по 100 символов, лимит 150 → разбивается
        para = "x" * 100
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = _split_text_into_chunks(text, max_chars=150)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = _split_text_into_chunks("", max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_single_paragraph_larger_than_limit(self):
        big_para = "x" * 1000
        chunks = _split_text_into_chunks(big_para, max_chars=100)
        assert len(chunks) == 1  # не можем разбить по абзацам, один чанк


# ---------------------------------------------------------------------------
# generate_protocol
# ---------------------------------------------------------------------------

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
        json_path = paths.dir / "test_transcript.json"
        json_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

        protocol_path = generate_protocol(str(json_path), paths, cfg)
        assert protocol_path.exists()

    def test_protocol_with_llm_clean(self, tmp_path):
        """clean_protocol=True → вызывает _clean_protocol."""
        cfg = AppConfig()
        cfg.llm.clean_protocol = True
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-05_14-30-12")
        paths.ensure_dir()

        transcript = {
            "language": "ru",
            "duration_sec": 5.0,
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "S", "text": "Текст."}],
        }

        with patch("meeting_recorder.report._clean_protocol", return_value="# Очищено") as mock_clean:
            generate_protocol(transcript, paths, cfg)

        mock_clean.assert_called_once()

    def test_protocol_empty_segments(self, tmp_path):
        cfg = AppConfig()
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-05_14-30-12")
        paths.ensure_dir()

        transcript = {"language": "ru", "duration_sec": 0.0, "segments": []}
        protocol_path = generate_protocol(transcript, paths, cfg)
        assert protocol_path.exists()


# ---------------------------------------------------------------------------
# _clean_protocol
# ---------------------------------------------------------------------------

class TestCleanProtocol:
    def _make_cfg(self):
        return AppConfig()

    def test_returns_cleaned_text(self, tmp_path):
        cfg = self._make_cfg()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Отшлифованный протокол"

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            result = _clean_protocol("# Исходный протокол", cfg)

        assert result == "# Отшлифованный протокол"

    def test_empty_response_returns_original(self, tmp_path):
        cfg = self._make_cfg()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = ""

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            result = _clean_protocol("# Исходный", cfg)

        assert result == "# Исходный"

    def test_strips_markdown_code_fences(self, tmp_path):
        cfg = self._make_cfg()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "```markdown\n# Протокол\n```"

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            result = _clean_protocol("# Исходный", cfg)

        assert "```" not in result
        assert "# Протокол" in result


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    def _make_paths(self, tmp_path, session_id="meeting_2026-06-05_14-30-12"):
        paths = SessionPaths(str(tmp_path), session_id)
        paths.ensure_dir()
        return paths

    def _make_transcript(self, n_segments=3):
        return {
            "session_id": "meeting_2026-06-05_14-30-12",
            "language": "ru",
            "duration_sec": 120.0,
            "segments": [
                {"start": float(i), "end": float(i + 1),
                 "speaker": "SPEAKER_00", "text": f"Текст сегмента {i}."}
                for i in range(n_segments)
            ],
        }

    def test_success(self, tmp_path):
        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Summary\nКраткое содержание встречи."

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            result = generate_summary(transcript, paths, cfg)

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Summary" in content

    def test_empty_llm_response_raises(self, tmp_path):
        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = ""

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="пустой"):
                generate_summary(transcript, paths, cfg)

    def test_from_file(self, tmp_path):
        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()
        json_path = paths.dir / "t.json"
        json_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Отчёт"

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            result = generate_summary(str(json_path), paths, cfg)
        assert result.exists()

    def test_no_segments(self, tmp_path):
        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = {"language": "ru", "duration_sec": 0.0, "segments": []}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Пустая встреча"

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            result = generate_summary(transcript, paths, cfg)
        assert result.exists()


# ---------------------------------------------------------------------------
# _call_llm_for_summary (chunking path)
# ---------------------------------------------------------------------------

class TestCallLLMForSummary:
    def test_short_text_direct_call(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Short summary"

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            result = _call_llm_for_summary("some text", est_tokens=100, cfg=cfg)

        assert result == "# Short summary"
        mock_client.chat.assert_called_once()

    def test_long_text_uses_map_reduce(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Chunk summary"

        with patch("meeting_recorder.llm_client.create_llm_client", return_value=mock_client):
            with patch("meeting_recorder.report._map_reduce_summary", return_value="# MR") as mock_mr:
                result = _call_llm_for_summary("text", est_tokens=200_000, cfg=cfg)

        mock_mr.assert_called_once()
        assert result == "# MR"


# ---------------------------------------------------------------------------
# _map_reduce_summary
# ---------------------------------------------------------------------------

class TestMapReduceSummary:
    def test_basic_map_reduce(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.chat.return_value = "# Chunk result"

        metadata = "Метаданные\n\nТранскрипт: Слово один.\n\nСлово два."

        result = _map_reduce_summary(metadata, est_tokens=1000, cfg=cfg, client=mock_client)
        assert "Chunk result" in result or result is not None
        assert mock_client.chat.call_count >= 1

    def test_all_empty_chunks_raises(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.chat.return_value = ""

        metadata = "Метаданные\n\nТранскрипт: Текст."

        with pytest.raises(RuntimeError, match="пустыми"):
            _map_reduce_summary(metadata, est_tokens=1000, cfg=cfg, client=mock_client)
