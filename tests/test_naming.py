"""Юнит-тесты для naming.py."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_recorder.naming import (
    SessionPaths,
    create_session,
    list_sessions,
    resolve_session,
    _ensure_unique_session_id,
)


class TestSessionPaths:
    def test_paths_are_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = SessionPaths(tmpdir, "meeting_2026-06-05_14-30-12")
            base = Path(tmpdir) / "meeting_2026-06-05_14-30-12"

            assert paths.video == base / "meeting_2026-06-05_14-30-12.mp4"
            assert paths.mic_audio == base / "meeting_2026-06-05_14-30-12_mic.wav"
            assert paths.system_audio == base / "meeting_2026-06-05_14-30-12_system.wav"
            assert paths.mix_audio == base / "meeting_2026-06-05_14-30-12_mix.wav"
            assert paths.transcript == base / "meeting_2026-06-05_14-30-12_transcript.json"
            assert paths.protocol == base / "meeting_2026-06-05_14-30-12_protocol.md"
            assert paths.summary == base / "meeting_2026-06-05_14-30-12_summary.md"

    def test_ensure_dir_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = SessionPaths(tmpdir, "meeting_2026-06-05_14-30-12")
            paths.ensure_dir()
            assert paths.dir.is_dir()


class TestCreateSession:
    def test_creates_unique_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = create_session(tmpdir)
            assert paths.session_id.startswith("meeting_")
            assert paths.dir.is_dir()

    def test_collision_adds_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Создаём директорию вручную с нужным именем
            sid = "meeting_2026-06-05_14-30-00"
            (Path(tmpdir) / sid).mkdir(exist_ok=True)

            # patch _generate_session_id чтобы вернуть фиксированное имя
            import meeting_recorder.naming as naming_mod
            orig = naming_mod._generate_session_id

            def fixed_gen(suffix=0):
                if suffix == 0:
                    return sid
                return f"{sid}_{suffix}"

            naming_mod._generate_session_id = fixed_gen
            try:
                paths = create_session(tmpdir)
                assert paths.session_id == f"{sid}_2"
            finally:
                naming_mod._generate_session_id = orig

    def test_collision_three_way(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_id = "meeting_2026-06-05_14-30-00"

            # Создаём все четыре коллизирующие директории
            for i in range(4):
                sid = base_id if i == 0 else f"{base_id}_{i}"
                (Path(tmpdir) / sid).mkdir(exist_ok=True)

            import meeting_recorder.naming as naming_mod
            orig = naming_mod._generate_session_id

            def fixed_gen(suffix=0):
                if suffix == 0:
                    return base_id
                return f"{base_id}_{suffix}"

            naming_mod._generate_session_id = fixed_gen
            try:
                paths = create_session(tmpdir)
                assert paths.session_id == f"{base_id}_4"
            finally:
                naming_mod._generate_session_id = orig


class TestListSessions:
    def test_lists_existing_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_session(tmpdir)
            create_session(tmpdir)
            sessions = list_sessions(tmpdir)
            assert len(sessions) == 2
            assert all(s.session_id.startswith("meeting_") for s in sessions)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = list_sessions(tmpdir)
            assert sessions == []


class TestResolveSession:
    def test_resolve_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = create_session(tmpdir)
            resolved = resolve_session(tmpdir, paths.session_id)
            assert resolved.session_id == paths.session_id

    def test_resolve_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                resolve_session(tmpdir, "meeting_2099-01-01_00-00-00")
                assert False, "Should have raised"
            except FileNotFoundError as e:
                assert "не найдена" in str(e)


class TestEnsureUniqueSessionId:
    def test_first_is_clean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = _ensure_unique_session_id(tmpdir)
            assert "_" not in sid.split("_", 2)[2]  # нет суффикса
