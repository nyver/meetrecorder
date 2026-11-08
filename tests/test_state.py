"""Тесты для state.py — state-файл и межпроцессный lock записи."""

from __future__ import annotations

import subprocess
import sys

from meeting_recorder import state


class TestStateFile:
    def test_roundtrip_with_owner(self, tmp_path):
        path = tmp_path / "s.json"
        state.save_state("sid", "/v.mp4", 42, owner=state.OWNER_TRAY, path=path)
        data = state.load_state(path)
        assert data == {
            "session_id": "sid",
            "video_path": "/v.mp4",
            "ffmpeg_pid": 42,
            "owner": "tray",
        }

    def test_default_owner_cli(self, tmp_path):
        path = tmp_path / "s.json"
        state.save_state("sid", "/v.mp4", path=path)
        assert state.load_state(path)["owner"] == "cli"

    def test_missing_and_corrupt(self, tmp_path):
        path = tmp_path / "s.json"
        assert state.load_state(path) is None
        path.write_text("[1, 2]", encoding="utf-8")  # не словарь
        assert state.load_state(path) is None
        path.write_text("{oops", encoding="utf-8")
        assert state.load_state(path) is None

    def test_remove_missing_ok(self, tmp_path):
        state.remove_state(tmp_path / "nope.json")


class TestRecordingLock:
    def test_exclusive_within_process(self, tmp_path):
        lock_path = tmp_path / "rec.lock"
        fh = state.acquire_recording_lock(lock_path)
        assert fh is not None
        try:
            assert state.acquire_recording_lock(lock_path) is None
            assert state.is_recording_active(lock_path)
        finally:
            state.release_recording_lock(fh)
        assert not state.is_recording_active(lock_path)

    def test_lock_visible_to_other_process(self, tmp_path):
        """Lock, взятый треем, виден процессу `mrec start` (и наоборот)."""
        lock_path = tmp_path / "rec.lock"
        fh = state.acquire_recording_lock(lock_path)
        try:
            code = (
                "import sys; from pathlib import Path; "
                "from meeting_recorder import state; "
                f"sys.exit(0 if state.is_recording_active(Path(r'{lock_path}')) else 1)"
            )
            result = subprocess.run([sys.executable, "-c", code], timeout=60)
            assert result.returncode == 0
        finally:
            state.release_recording_lock(fh)

    def test_release_none_is_noop(self):
        state.release_recording_lock(None)


class TestDirs:
    def test_state_dir_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEETING_RECORDER_STATE_DIR", str(tmp_path / "x"))
        assert state.default_state_dir() == tmp_path / "x"

    def test_app_data_dir_name(self):
        assert state.app_data_dir().name == "MeetingRecorder"
