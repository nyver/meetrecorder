"""Юнит-тесты для html_report.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_recorder.html_report import (
    _color_map,
    _inline,
    _is_sep_row,
    _md_to_html,
    _pick_media,
    generate_html_protocol,
)
from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths

_SESSION_ID = "meeting_2026-06-14_10-00-00"


def _paths(tmp_path) -> SessionPaths:
    p = SessionPaths(str(tmp_path), _SESSION_ID)
    p.ensure_dir()
    return p


def _transcript(**kwargs) -> dict:
    base = {
        "session_id": _SESSION_ID,
        "language": "ru",
        "duration_sec": 120.0,
        "segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Привет."},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "Здравствуй!"},
        ],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# _color_map
# ---------------------------------------------------------------------------

class TestColorMap:
    def test_first_speaker_gets_first_color(self):
        result = _color_map(["A"])
        assert result["A"] == "#2563eb"

    def test_multiple_speakers_different_colors(self):
        result = _color_map(["A", "B", "C"])
        colors = list(result.values())
        assert len(set(colors)) == 3

    def test_wraps_around_after_10(self):
        speakers = [f"S{i}" for i in range(12)]
        result = _color_map(speakers)
        assert result["S0"] == result["S10"]
        assert result["S1"] == result["S11"]

    def test_empty_returns_empty(self):
        assert _color_map([]) == {}


# ---------------------------------------------------------------------------
# _pick_media
# ---------------------------------------------------------------------------

class TestPickMedia:
    def test_final_video_has_highest_priority(self, tmp_path):
        p = _paths(tmp_path)
        p.final_video.write_bytes(b"")
        p.mix_audio.write_bytes(b"")
        p.video.write_bytes(b"")
        name, mime = _pick_media(p)
        assert name == p.final_video.name
        assert mime == "video/mp4"

    def test_mix_audio_preferred_over_raw_video(self, tmp_path):
        p = _paths(tmp_path)
        p.mix_audio.write_bytes(b"")
        p.video.write_bytes(b"")
        name, mime = _pick_media(p)
        assert name == p.mix_audio.name
        assert mime == "audio/wav"

    def test_raw_video_as_last_resort(self, tmp_path):
        p = _paths(tmp_path)
        p.video.write_bytes(b"")
        name, mime = _pick_media(p)
        assert name == p.video.name
        assert mime == "video/mp4"

    def test_none_when_no_files(self, tmp_path):
        p = _paths(tmp_path)
        assert _pick_media(p) is None


# ---------------------------------------------------------------------------
# _is_sep_row
# ---------------------------------------------------------------------------

class TestIsSepRow:
    def test_simple_separator(self):
        assert _is_sep_row("|---|---|---|")

    def test_with_spaces(self):
        assert _is_sep_row("| --- | --- |")

    def test_left_align(self):
        assert _is_sep_row("|:---|:---|")

    def test_center_align(self):
        assert _is_sep_row("|:---:|:---:|")

    def test_right_align(self):
        assert _is_sep_row("|---:|---:|")

    def test_header_row_is_not_separator(self):
        assert not _is_sep_row("| Задача | Ответственный |")

    def test_data_row_is_not_separator(self):
        assert not _is_sep_row("| 1 | Текст |")

    def test_mixed_cell_is_not_separator(self):
        assert not _is_sep_row("| abc | --- |")


# ---------------------------------------------------------------------------
# _inline
# ---------------------------------------------------------------------------

class TestInline:
    def test_bold(self):
        assert _inline("**жирный**") == "<strong>жирный</strong>"

    def test_italic(self):
        assert _inline("*курсив*") == "<em>курсив</em>"

    def test_bold_and_italic(self):
        result = _inline("**Важно** и *возможно*")
        assert "<strong>Важно</strong>" in result
        assert "<em>возможно</em>" in result

    def test_plain_text_unchanged(self):
        assert _inline("обычный текст") == "обычный текст"

    def test_empty_string(self):
        assert _inline("") == ""


# ---------------------------------------------------------------------------
# _md_to_html
# ---------------------------------------------------------------------------

class TestMdToHtml:
    # --- Заголовки ---

    def test_h1_maps_to_h3(self):
        result = _md_to_html("# Заголовок")
        assert "<h3>Заголовок</h3>" in result

    def test_h2_maps_to_h4(self):
        result = _md_to_html("## Раздел")
        assert "<h4>Раздел</h4>" in result

    def test_h3_maps_to_h5(self):
        result = _md_to_html("### Подраздел")
        assert "<h5>Подраздел</h5>" in result

    def test_heading_with_inline(self):
        result = _md_to_html("## **Важный** раздел")
        assert "<strong>Важный</strong>" in result

    # --- Списки ---

    def test_bullet_dash(self):
        result = _md_to_html("- Пункт 1\n- Пункт 2")
        assert "<ul>" in result
        assert "<li>Пункт 1</li>" in result
        assert "<li>Пункт 2</li>" in result
        assert "</ul>" in result

    def test_bullet_star(self):
        result = _md_to_html("* Элемент")
        assert "<li>Элемент</li>" in result

    def test_list_closes_before_heading(self):
        result = _md_to_html("- Пункт\n# Заголовок")
        assert result.index("</ul>") < result.index("<h3>")

    def test_list_closes_before_empty_line(self):
        result = _md_to_html("- Пункт\n\nТекст")
        assert "</ul>" in result
        assert "<p>Текст</p>" in result

    # --- HR и параграфы ---

    def test_horizontal_rule(self):
        assert "<hr>" in _md_to_html("---")

    def test_paragraph(self):
        assert "<p>Простой текст.</p>" in _md_to_html("Простой текст.")

    def test_empty_string(self):
        _md_to_html("")  # не должна бросить

    # --- XSS ---

    def test_xss_escaped_in_paragraph(self):
        result = _md_to_html("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_xss_escaped_in_list(self):
        result = _md_to_html("- <b>жирный</b>")
        assert "<b>жирный</b>" not in result

    # --- Таблицы ---

    def test_table_full(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = _md_to_html(md)
        assert '<table class="md-table">' in result
        assert "<th>A</th>" in result
        assert "<th>B</th>" in result
        assert "</thead><tbody>" in result
        assert "<td>1</td>" in result
        assert "<td>2</td>" in result
        assert "</tbody></table>" in result

    def test_table_multiple_rows(self):
        md = "| # | Задача |\n|---|---|\n| 1 | Первая |\n| 2 | Вторая |"
        result = _md_to_html(md)
        assert result.count("<tr>") == 3  # 1 header + 2 data

    def test_table_without_separator_uses_thead(self):
        md = "| X | Y |\n| a | b |"
        result = _md_to_html(md)
        assert "<th>" in result
        assert "</thead></table>" in result

    def test_table_closed_before_paragraph(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n\nТекст"
        result = _md_to_html(md)
        assert result.index("</tbody></table>") < result.index("<p>Текст</p>")

    def test_table_at_end_of_text(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = _md_to_html(md)
        assert result.endswith("</tbody></table>")

    def test_table_xss_in_cell(self):
        md = "| <script> | safe |\n|---|---|\n| x | y |"
        result = _md_to_html(md)
        assert "<script>" not in result

    def test_list_closes_before_table(self):
        md = "- Пункт\n| A |\n|---|\n| 1 |"
        result = _md_to_html(md)
        assert "</ul>" in result
        assert '<table class="md-table">' in result
        assert result.index("</ul>") < result.index('<table class="md-table">')

    def test_separator_alignment_markers_recognised(self):
        md = "| A | B | C |\n|:---|:---:|---:|\n| x | y | z |"
        result = _md_to_html(md)
        assert "<td>x</td>" in result


# ---------------------------------------------------------------------------
# generate_html_protocol
# ---------------------------------------------------------------------------

class TestGenerateHtmlProtocol:
    def test_creates_html_file(self, tmp_path):
        p = _paths(tmp_path)
        result = generate_html_protocol(_transcript(), p, AppConfig())
        assert result == p.html_protocol
        assert p.html_protocol.exists()

    def test_html_contains_session_id(self, tmp_path):
        p = _paths(tmp_path)
        generate_html_protocol(_transcript(), p, AppConfig())
        assert _SESSION_ID in p.html_protocol.read_text(encoding="utf-8")

    def test_html_contains_segment_text(self, tmp_path):
        p = _paths(tmp_path)
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert "Привет." in content
        assert "Здравствуй!" in content

    def test_from_file_path(self, tmp_path):
        p = _paths(tmp_path)
        json_path = p.dir / "t.json"
        json_path.write_text(json.dumps(_transcript(), ensure_ascii=False), encoding="utf-8")
        result = generate_html_protocol(str(json_path), p, AppConfig())
        assert result.exists()

    def test_from_path_object(self, tmp_path):
        p = _paths(tmp_path)
        json_path = p.dir / "t.json"
        json_path.write_text(json.dumps(_transcript(), ensure_ascii=False), encoding="utf-8")
        result = generate_html_protocol(json_path, p, AppConfig())
        assert result.exists()

    def test_speaker_renaming(self, tmp_path):
        cfg = AppConfig()
        cfg.transcription.speaker_names = {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}
        p = _paths(tmp_path)
        generate_html_protocol(_transcript(), p, cfg)
        content = p.html_protocol.read_text(encoding="utf-8")
        assert "Иван" in content
        assert "Мария" in content
        assert "SPEAKER_00" not in content

    def test_with_summary(self, tmp_path):
        p = _paths(tmp_path)
        p.summary.write_text("# Итоги\n\nКраткое содержание.", encoding="utf-8")
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert "Краткое содержание." in content
        assert "<details" in content

    def test_without_summary_no_details(self, tmp_path):
        p = _paths(tmp_path)
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert "<details" not in content

    def test_with_final_video_uses_video_tag(self, tmp_path):
        p = _paths(tmp_path)
        p.final_video.write_bytes(b"")
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert p.final_video.name in content
        assert "<video" in content

    def test_with_mix_audio_uses_audio_tag(self, tmp_path):
        p = _paths(tmp_path)
        p.mix_audio.write_bytes(b"")
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert p.mix_audio.name in content
        assert "<audio" in content

    def test_no_media_shows_no_media_block(self, tmp_path):
        p = _paths(tmp_path)
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert "no-media" in content

    def test_mux_attempted_when_video_and_audio_exist(self, tmp_path):
        p = _paths(tmp_path)
        p.video.write_bytes(b"")
        p.mix_audio.write_bytes(b"")
        with patch("meeting_recorder.recorder.mux_video") as mock_mux:
            mock_mux.return_value = p.final_video
            generate_html_protocol(_transcript(), p, AppConfig())
        mock_mux.assert_called_once_with(p.video, p.mix_audio, p.final_video)

    def test_mux_not_attempted_when_final_exists(self, tmp_path):
        p = _paths(tmp_path)
        p.final_video.write_bytes(b"")
        p.video.write_bytes(b"")
        p.mix_audio.write_bytes(b"")
        with patch("meeting_recorder.recorder.mux_video") as mock_mux:
            generate_html_protocol(_transcript(), p, AppConfig())
        mock_mux.assert_not_called()

    def test_mux_not_attempted_without_both_files(self, tmp_path):
        p = _paths(tmp_path)
        p.video.write_bytes(b"")  # только видео, нет аудио
        with patch("meeting_recorder.recorder.mux_video") as mock_mux:
            generate_html_protocol(_transcript(), p, AppConfig())
        mock_mux.assert_not_called()

    def test_mux_failure_is_non_blocking(self, tmp_path):
        p = _paths(tmp_path)
        p.video.write_bytes(b"")
        p.mix_audio.write_bytes(b"")
        with patch("meeting_recorder.recorder.mux_video", side_effect=RuntimeError("ffmpeg failed")):
            result = generate_html_protocol(_transcript(), p, AppConfig())
        assert result.exists()

    def test_xss_in_speaker_name_escaped(self, tmp_path):
        p = _paths(tmp_path)
        tr = _transcript(segments=[
            {"start": 0.0, "end": 1.0, "speaker": "<script>xss</script>", "text": "text"},
        ])
        generate_html_protocol(tr, p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert "<script>xss</script>" not in content

    def test_xss_in_segment_text_escaped(self, tmp_path):
        p = _paths(tmp_path)
        tr = _transcript(segments=[
            {"start": 0.0, "end": 1.0, "speaker": "S", "text": '<img onerror="alert(1)">'},
        ])
        generate_html_protocol(tr, p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert '<img onerror="alert(1)">' not in content

    def test_empty_segments_produces_valid_html(self, tmp_path):
        p = _paths(tmp_path)
        tr = _transcript(segments=[])
        result = generate_html_protocol(tr, p, AppConfig())
        content = result.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_speaker_filter_buttons_present(self, tmp_path):
        p = _paths(tmp_path)
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert 'class="sp-btn"' in content

    def test_timecodes_in_output(self, tmp_path):
        p = _paths(tmp_path)
        generate_html_protocol(_transcript(), p, AppConfig())
        content = p.html_protocol.read_text(encoding="utf-8")
        assert "00:00" in content  # первый сегмент с t=0
