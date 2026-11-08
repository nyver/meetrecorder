"""Тесты для session_utils.py — общие хелперы артефактов сессии."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from meeting_recorder.naming import create_session
from meeting_recorder.session_utils import (
    atomic_write_text,
    load_highlights,
    load_transcript_data,
    normalize_highlights,
    normalize_segments,
    parse_seconds,
    pick_media,
    save_transcript_data,
    strip_code_fences,
    transcript_lines,
    unique_speakers,
)


class TestAtomicWrite:
    def test_writes_text(self, tmp_path):
        p = tmp_path / "a.txt"
        atomic_write_text(p, "привет")
        assert p.read_text(encoding="utf-8") == "привет"

    def test_overwrites(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("old", encoding="utf-8")
        atomic_write_text(p, "new")
        assert p.read_text(encoding="utf-8") == "new"

    def test_failure_keeps_original_and_cleans_tmp(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("old", encoding="utf-8")
        with patch(
            "meeting_recorder.session_utils.os.replace", side_effect=OSError("locked")
        ):
            with pytest.raises(OSError):
                atomic_write_text(p, "new")
        assert p.read_text(encoding="utf-8") == "old"
        assert [x.name for x in tmp_path.iterdir()] == ["a.txt"]


class TestParseSeconds:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (12.5, 12.5),
            (3, 3.0),
            ("12.5", 12.5),
            ("05:30", 330.0),
            ("01:02:03", 3723.0),
            ("abc", 0.0),
            (None, 0.0),
            (True, 0.0),
            (-5, 0.0),
            ("", 0.0),
            ([1], 0.0),
        ],
    )
    def test_values(self, value, expected):
        assert parse_seconds(value) == expected


class TestHighlights:
    def test_normalize_filters_and_coerces(self):
        raw = [
            {"title": "A", "description": "d", "start": "01:00"},
            "garbage",
            {"title": None, "start": 5},
        ]
        assert normalize_highlights(raw) == [
            {"title": "A", "description": "d", "start": 60.0},
            {"title": "", "description": "", "start": 5.0},
        ]

    def test_normalize_non_list(self):
        assert normalize_highlights({"a": 1}) == []

    def test_load_missing(self, tmp_path):
        assert load_highlights(tmp_path / "none.json") == []

    def test_load_corrupt(self, tmp_path):
        p = tmp_path / "h.json"
        p.write_text("{not json", encoding="utf-8")
        assert load_highlights(p) == []

    def test_load_valid(self, tmp_path):
        p = tmp_path / "h.json"
        p.write_text(
            json.dumps([{"title": "T", "description": "D", "start": 1}]),
            encoding="utf-8",
        )
        assert load_highlights(p)[0]["start"] == 1.0


class TestTranscriptHelpers:
    def test_load_dict_passthrough(self):
        d = {"segments": []}
        assert load_transcript_data(d) is d

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_transcript_data(tmp_path / "x.json")

    def test_save_and_load(self, tmp_path):
        p = tmp_path / "t.json"
        save_transcript_data(p, {"text": "привет"})
        assert load_transcript_data(p) == {"text": "привет"}
        assert "привет" in p.read_text(encoding="utf-8")

    def test_aliases_applied(self):
        segs = [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": " hi "}]
        out = normalize_segments(segs, {"SPEAKER_00": "Иван"})
        assert out == [{"start": 0, "end": 1, "speaker": "Иван", "text": "hi"}]

    def test_lines_and_speakers(self):
        segs = [
            {"start": 0, "end": 1, "speaker": "B", "text": "x"},
            {"start": 61, "end": 62, "speaker": "A", "text": "y"},
        ]
        lines = transcript_lines(segs, timestamp_formatter=lambda s: f"{int(s)}")
        assert lines == ["[0] B: x", "[61] A: y"]
        assert unique_speakers(segs) == ["A", "B"]


class TestPickMediaAndFences:
    def test_pick_media_priority(self, tmp_path):
        s = create_session(str(tmp_path))
        assert pick_media(s) is None
        s.video.write_bytes(b"v")
        assert pick_media(s) == (s.video.name, "video/mp4")
        s.mix_audio.write_bytes(b"a")
        assert pick_media(s, mime=False) == (s.mix_audio.name, "audio")
        s.final_video.write_bytes(b"f")
        assert pick_media(s) == (s.final_video.name, "video/mp4")

    def test_strip_code_fences(self):
        assert strip_code_fences("```json\n[1]\n```") == "[1]"
        assert strip_code_fences("  plain ") == "plain"
