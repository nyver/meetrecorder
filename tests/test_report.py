"""Юнит-тесты для report.py (протокол, summary, clean, chunking)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths
from meeting_recorder.report import (
    _call_llm_for_summary,
    _clean_protocol,
    _format_timestamp,
    _map_reduce_summary,
    _session_datetime,
    _split_text_into_chunks,
    generate_protocol,
    generate_summary,
)

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

    def test_suffixed_session_id(self):
        dt = _session_datetime("meeting_2026-06-14_10-30-00_2")
        assert dt.year == 2026
        assert dt.month == 6
        assert dt.hour == 10
        assert dt.minute == 30

    def test_invalid_session_id_returns_now(self):
        from datetime import datetime

        dt = _session_datetime("invalid")
        assert isinstance(dt, datetime)


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
        assert len(chunks) == 10
        assert all(len(chunk) <= 100 for chunk in chunks)


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
                {
                    "start": 1.0,
                    "end": 5.0,
                    "speaker": "SPEAKER_00",
                    "text": "Привет, всем привет.",
                },
                {
                    "start": 5.5,
                    "end": 10.0,
                    "speaker": "SPEAKER_01",
                    "text": "Здравствуй!",
                },
                {
                    "start": 10.5,
                    "end": 15.0,
                    "speaker": "SPEAKER_00",
                    "text": "Давай обсудим план.",
                },
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
                {
                    "start": 1.0,
                    "end": 5.0,
                    "speaker": "SPEAKER_00",
                    "text": "Привет, Мария.",
                },
                {
                    "start": 5.5,
                    "end": 10.0,
                    "speaker": "SPEAKER_01",
                    "text": "Привет, Иван!",
                },
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
        json_path.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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

        with patch(
            "meeting_recorder.report._clean_protocol", return_value="# Очищено"
        ) as mock_clean:
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

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = _clean_protocol("# Исходный протокол", cfg)

        assert result == "# Отшлифованный протокол"

    def test_empty_response_returns_original(self, tmp_path):
        cfg = self._make_cfg()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = ""

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = _clean_protocol("# Исходный", cfg)

        assert result == "# Исходный"

    def test_strips_markdown_code_fences(self, tmp_path):
        cfg = self._make_cfg()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "```markdown\n# Протокол\n```"

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
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
                {
                    "start": float(i),
                    "end": float(i + 1),
                    "speaker": "SPEAKER_00",
                    "text": f"Текст сегмента {i}.",
                }
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

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
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

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            with pytest.raises(RuntimeError, match="пустой"):
                generate_summary(transcript, paths, cfg)

    def test_from_file(self, tmp_path):
        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()
        json_path = paths.dir / "t.json"
        json_path.write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Отчёт"

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
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

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = generate_summary(transcript, paths, cfg)
        assert result.exists()


# ---------------------------------------------------------------------------
# _call_llm_for_summary (chunking path)
# ---------------------------------------------------------------------------


class TestCallLLMForSummary:
    _TEMPLATE = (
        "Дата: {date} {time}\nДлит: {duration}\nУч: {speakers}\n\n{transcript_text}"
    )
    _META = {"date": "2026-06-05", "time": "14:30", "duration": "2", "speakers": "S0"}

    def test_short_text_direct_call(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Short summary"

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = _call_llm_for_summary(
                "some text", self._TEMPLATE, self._META, est_tokens=100, cfg=cfg
            )

        assert result == "# Short summary"
        mock_client.chat.assert_called_once()

    def test_long_text_uses_map_reduce(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.return_value = "# Chunk summary"

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            with patch(
                "meeting_recorder.report._map_reduce_summary", return_value="# MR"
            ) as mock_mr:
                result = _call_llm_for_summary(
                    "text", self._TEMPLATE, self._META, est_tokens=200_000, cfg=cfg
                )

        mock_mr.assert_called_once()
        assert result == "# MR"


# ---------------------------------------------------------------------------
# _map_reduce_summary
# ---------------------------------------------------------------------------


class TestMapReduceSummary:
    _TEMPLATE = (
        "Дата: {date} {time}\nДлит: {duration}\nУч: {speakers}\n\n{transcript_text}"
    )
    _META = {"date": "2026-06-05", "time": "14:30", "duration": "2", "speakers": "S0"}

    def test_basic_map_reduce(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.chat.return_value = "# Chunk result"

        result = _map_reduce_summary(
            "Слово один.\n\nСлово два.",
            self._TEMPLATE,
            self._META,
            cfg=cfg,
            client=mock_client,
        )
        assert "Chunk result" in result
        assert mock_client.chat.call_count >= 1

    def test_all_empty_chunks_raises(self, tmp_path):
        cfg = AppConfig()
        mock_client = MagicMock()
        mock_client.chat.return_value = ""

        with pytest.raises(RuntimeError, match="пустыми"):
            _map_reduce_summary(
                "Текст.",
                self._TEMPLATE,
                self._META,
                cfg=cfg,
                client=mock_client,
            )

    def test_chunks_receive_different_text(self, tmp_path):
        """Каждый чанк должен получать свой фрагмент транскрипта, не полный текст."""
        import meeting_recorder.report as report_mod

        cfg = AppConfig()
        received_prompts = []

        def capture(msgs):
            received_prompts.append(msgs[0]["content"])
            return "# ok"

        mock_client = MagicMock()
        mock_client.chat.side_effect = capture

        chunk_a = "A" * 100
        chunk_b = "B" * 100
        full_text = f"{chunk_a}\n\n{chunk_b}"

        # Уменьшаем размер чанка, чтобы текст разбился на 2 чанка.
        with patch.object(report_mod, "_chunk_chars", return_value=150):
            _map_reduce_summary(
                full_text, self._TEMPLATE, self._META, cfg=cfg, client=mock_client
            )

        # 2 map-промпта + 1 reduce = минимум 3 вызова
        assert mock_client.chat.call_count >= 3
        # Первый map-промпт содержит chunk_a, но не chunk_b
        assert "A" * 10 in received_prompts[0]
        assert "B" * 10 not in received_prompts[0]


# ---------------------------------------------------------------------------
# generate_highlights
# ---------------------------------------------------------------------------


class TestGenerateHighlights:
    def _make_paths(self, tmp_path, session_id="meeting_2026-06-28_10-00-00"):
        paths = SessionPaths(str(tmp_path), session_id)
        paths.ensure_dir()
        return paths

    def _make_transcript(self):
        return {
            "language": "ru",
            "duration_sec": 120.0,
            "segments": [
                {
                    "start": float(i * 10),
                    "end": float(i * 10 + 9),
                    "speaker": "SPEAKER_00",
                    "text": f"Текст сегмента {i}.",
                }
                for i in range(10)
            ],
        }

    def _mock_client(self, response_text: str):
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.chat.return_value = response_text
        return mock

    def test_success(self, tmp_path):
        from meeting_recorder.report import generate_highlights

        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()

        hl = [
            {"title": f"Момент {i}", "description": "Описание", "start": float(i * 20)}
            for i in range(5)
        ]
        mock_client = self._mock_client(json.dumps(hl, ensure_ascii=False))

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = generate_highlights(transcript, paths, cfg)

        assert result.exists()
        saved = json.loads(result.read_text(encoding="utf-8"))
        assert isinstance(saved, list)
        assert saved[0]["title"] == "Момент 0"

    def test_from_file(self, tmp_path):
        from meeting_recorder.report import generate_highlights

        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()
        json_path = tmp_path / "t.json"
        json_path.write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
        )

        hl = [{"title": "T", "description": "D", "start": 0.0}]
        mock_client = self._mock_client(json.dumps(hl))

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = generate_highlights(str(json_path), paths, cfg)
        assert result.exists()

    def test_strips_code_fences(self, tmp_path):
        from meeting_recorder.report import generate_highlights

        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()

        hl = [{"title": "T", "description": "D", "start": 0.0}]
        fenced = f"```json\n{json.dumps(hl)}\n```"
        mock_client = self._mock_client(fenced)

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = generate_highlights(transcript, paths, cfg)
        assert result.exists()
        saved = json.loads(result.read_text(encoding="utf-8"))
        assert isinstance(saved, list)

    def test_non_list_response_raises(self, tmp_path):
        from meeting_recorder.report import generate_highlights

        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()

        mock_client = self._mock_client('{"error": "not a list"}')

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            with pytest.raises(ValueError, match="массив"):
                generate_highlights(transcript, paths, cfg)

    def test_speaker_names_substituted(self, tmp_path):
        from meeting_recorder.report import generate_highlights

        cfg = AppConfig()
        cfg.transcription.speaker_names = {"SPEAKER_00": "Иван"}
        paths = self._make_paths(tmp_path)
        transcript = self._make_transcript()

        captured: list[str] = []

        def _chat(msgs):
            captured.append(msgs[0]["content"])
            return json.dumps([{"title": "T", "description": "D", "start": 0.0}])

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.side_effect = _chat

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            generate_highlights(transcript, paths, cfg)

        assert "Иван" in captured[0]

    def test_empty_segments(self, tmp_path):
        from meeting_recorder.report import generate_highlights

        cfg = AppConfig()
        paths = self._make_paths(tmp_path)
        transcript = {"language": "ru", "duration_sec": 0.0, "segments": []}

        hl = [{"title": "T", "description": "D", "start": 0.0}]
        mock_client = self._mock_client(json.dumps(hl))

        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=mock_client
        ):
            result = generate_highlights(transcript, paths, cfg)
        assert result.exists()


# ---------------------------------------------------------------------------
# Чанкинг по строкам, бюджет контекста, highlights map-reduce, чистка по частям
# ---------------------------------------------------------------------------

from meeting_recorder.llm_client import LLMTruncatedError  # noqa: E402
from meeting_recorder.report import (  # noqa: E402
    _chunk_chars,
    _parse_highlights_response,
    _select_top_highlights,
    estimate_tokens,
    generate_highlights,
)


def _ctx_client(chat):
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    if callable(chat):
        m.chat.side_effect = chat
    else:
        m.chat.return_value = chat
    return m


class TestChunkingByLines:
    def test_transcript_lines_not_cut_mid_line(self):
        """Транскрипт склеен через \\n — чанки должны резаться по границам реплик."""
        lines = [f"[00:{i:02d}] SPEAKER_00: реплика номер {i}" for i in range(50)]
        text = "\n".join(lines)
        chunks = _split_text_into_chunks(text, max_chars=200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200
            for line in chunk.split("\n"):
                assert line in lines

    def test_chunks_cover_all_text(self):
        lines = [f"line {i}" for i in range(100)]
        chunks = _split_text_into_chunks("\n".join(lines), max_chars=50)
        assert "\n".join(chunks).split("\n") == lines


class TestContextBudget:
    def test_estimate_tokens_cyrillic(self):
        assert estimate_tokens("а" * 300) >= 100

    def test_chunk_chars_respects_context(self):
        small = AppConfig.model_validate(
            {"llm": {"context_tokens": 8192, "max_tokens": 2048}}
        )
        big = AppConfig.model_validate(
            {"llm": {"context_tokens": 131072, "max_tokens": 2048}}
        )
        assert _chunk_chars(small) < _chunk_chars(big)

    def test_chunk_chars_output_bound(self):
        cfg = AppConfig.model_validate(
            {"llm": {"context_tokens": 131072, "max_tokens": 4096}}
        )
        assert _chunk_chars(cfg, output_bound=True) < _chunk_chars(cfg)

    def test_summary_long_transcript_uses_map_reduce(self, tmp_path):
        """При малом контексте summary считается по частям, а не одним запросом."""
        cfg = AppConfig.model_validate(
            {"llm": {"context_tokens": 4096, "max_tokens": 1024}}
        )
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-28_10-00-00")
        paths.ensure_dir()
        segments = [
            {
                "start": float(i),
                "end": float(i + 1),
                "speaker": "SPEAKER_00",
                "text": "очень длинная реплика участника встречи " * 3,
            }
            for i in range(400)
        ]
        client = _ctx_client("# Резюме")
        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=client
        ):
            generate_summary({"segments": segments, "duration_sec": 400}, paths, cfg)
        assert client.chat.call_count > 2
        for call_ in client.chat.call_args_list:
            prompt = call_.args[0][0]["content"]
            assert estimate_tokens(prompt) <= cfg.llm.input_token_budget() + 200


class TestHierarchicalReduce:
    def test_reduce_in_rounds_when_summaries_too_big(self, tmp_path):
        import meeting_recorder.report as report_mod

        cfg = AppConfig()
        client = MagicMock()
        client.chat.return_value = "S" * 100
        text = "\n\n".join(["x" * 90] * 6)
        fits_calls = iter([False, True])
        with patch.object(report_mod, "_chunk_chars", return_value=100):
            with patch.object(
                report_mod, "_fits", side_effect=lambda *a, **k: next(fits_calls)
            ):
                result = report_mod._map_reduce_summary(
                    text, "{transcript_text}", {}, cfg, client
                )
        assert result == "S" * 100
        # 6 map + ≥1 промежуточный reduce + финальный reduce
        assert client.chat.call_count >= 8


class TestTruncatedSummary:
    def test_partial_saved_with_note(self, tmp_path):
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-28_10-00-00")
        paths.ensure_dir()
        client = _ctx_client(None)
        client.chat.side_effect = LLMTruncatedError("cut", partial="# Частичное резюме")
        data = {"segments": [{"start": 0, "end": 1, "speaker": "S", "text": "t"}]}
        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=client
        ):
            generate_summary(data, paths, AppConfig())
        text = paths.summary.read_text(encoding="utf-8")
        assert "Частичное резюме" in text
        assert "обрезан" in text

    def test_truncated_without_content_raises(self, tmp_path):
        paths = SessionPaths(str(tmp_path), "meeting_2026-06-28_10-00-00")
        paths.ensure_dir()
        client = _ctx_client(None)
        client.chat.side_effect = LLMTruncatedError("cut", partial="")
        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=client
        ):
            with pytest.raises(LLMTruncatedError):
                generate_summary({"segments": []}, paths, AppConfig())


class TestCleanProtocolChunks:
    def test_truncated_chunk_keeps_original(self):
        """Обрезанный ответ не должен заменять протокол — исходный текст сохраняется."""
        import meeting_recorder.report as report_mod

        text = "\n".join(f"**[00:{i:02d}]** S: фраза {i}" for i in range(40))
        answers = iter(
            ["CLEANED-1", LLMTruncatedError("cut", partial="half"), "CLEANED-3"]
        )

        def _chat(msgs):
            a = next(answers, "CLEANED-N")
            if isinstance(a, Exception):
                raise a
            return a

        client = _ctx_client(_chat)
        with patch.object(report_mod, "_chunk_chars", return_value=300):
            with patch(
                "meeting_recorder.llm_client.create_llm_client", return_value=client
            ):
                result = _clean_protocol(text, AppConfig())
        assert "CLEANED-1" in result
        assert "half" not in result
        assert "фраза" in result  # оригинал части 2 на месте
        assert client.chat.call_count > 2

    def test_llm_error_keeps_original(self):
        from meeting_recorder.llm_client import LLMClientError

        client = _ctx_client(None)
        client.chat.side_effect = LLMClientError("down")
        with patch(
            "meeting_recorder.llm_client.create_llm_client", return_value=client
        ):
            assert _clean_protocol("оригинал", AppConfig()) == "оригинал"


class TestHighlightsParsing:
    def test_array_inside_prose(self):
        resp = 'Вот результат:\n[{"title": "T", "description": "D", "start": "00:30"}]\nГотово.'
        assert _parse_highlights_response(resp) == [
            {"title": "T", "description": "D", "start": 30.0}
        ]

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            _parse_highlights_response("нет json")

    def test_broken_json_in_brackets_raises(self):
        with pytest.raises(ValueError, match="некорректный"):
            _parse_highlights_response("[{oops}]")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="пустой"):
            _parse_highlights_response("")


class TestHighlightsMapReduce:
    def _paths(self, tmp_path):
        p = SessionPaths(str(tmp_path), "meeting_2026-06-28_10-00-00")
        p.ensure_dir()
        return p

    def test_long_transcript_processed_in_chunks(self, tmp_path):
        import meeting_recorder.report as report_mod

        segments = [
            {
                "start": float(i * 10),
                "end": float(i * 10 + 9),
                "speaker": "S",
                "text": "текст " * 20,
            }
            for i in range(30)
        ]
        calls = {"n": 0}

        def _chat(msgs):
            calls["n"] += 1
            prompt = msgs[0]["content"]
            if "кандидатов" in prompt:
                return json.dumps(
                    [
                        {"title": f"Top{i}", "description": "", "start": i}
                        for i in range(5)
                    ]
                )
            return json.dumps(
                [
                    {
                        "title": f"C{calls['n']}-{j}",
                        "description": "",
                        "start": calls["n"] * 10 + j,
                    }
                    for j in range(3)
                ]
            )

        client = _ctx_client(_chat)
        with patch.object(report_mod, "_fits", return_value=False):
            with patch.object(report_mod, "_chunk_chars", return_value=2000):
                with patch(
                    "meeting_recorder.llm_client.create_llm_client", return_value=client
                ):
                    path = generate_highlights(
                        {"segments": segments}, self._paths(tmp_path), AppConfig()
                    )
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert [h["title"] for h in saved] == [f"Top{i}" for i in range(5)]
        assert client.chat.call_count > 2

    def test_failed_chunk_skipped(self, tmp_path):
        import meeting_recorder.report as report_mod

        answers = iter(
            ["мусор", json.dumps([{"title": "Ok", "description": "", "start": 1}])]
        )
        client = _ctx_client(lambda msgs: next(answers))
        with patch.object(report_mod, "_fits", return_value=False):
            with patch.object(
                report_mod, "_split_text_into_chunks", return_value=["a", "b"]
            ):
                with patch(
                    "meeting_recorder.llm_client.create_llm_client", return_value=client
                ):
                    path = generate_highlights(
                        {"segments": []}, self._paths(tmp_path), AppConfig()
                    )
        assert json.loads(path.read_text(encoding="utf-8"))[0]["title"] == "Ok"

    def test_all_chunks_failed_raises(self, tmp_path):
        import meeting_recorder.report as report_mod

        client = _ctx_client("мусор")
        with patch.object(report_mod, "_fits", return_value=False):
            with patch.object(
                report_mod, "_split_text_into_chunks", return_value=["a", "b"]
            ):
                with patch(
                    "meeting_recorder.llm_client.create_llm_client", return_value=client
                ):
                    with pytest.raises(ValueError, match="ни одного"):
                        generate_highlights(
                            {"segments": []}, self._paths(tmp_path), AppConfig()
                        )

    def test_select_fallback_evenly_spaced(self):
        client = MagicMock()
        client.chat.return_value = "мусор"
        cands = [
            {"title": str(i), "description": "", "start": float(i)} for i in range(10)
        ]
        picked = _select_top_highlights(client, cands)
        assert len(picked) == 5
        assert [h["start"] for h in picked] == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_select_few_candidates_no_llm(self):
        client = MagicMock()
        cands = [
            {"title": "a", "description": "", "start": 5.0},
            {"title": "b", "description": "", "start": 1.0},
        ]
        assert [h["title"] for h in _select_top_highlights(client, cands)] == ["b", "a"]
        client.chat.assert_not_called()
