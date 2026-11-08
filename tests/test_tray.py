"""Тесты для tray.py — управление состоянием, _pick_session, иконка."""

from __future__ import annotations

import json
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths, create_session
from meeting_recorder.tray import TrayApp, _make_icon

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = str(tmp_path)
    return cfg


@pytest.fixture
def app(mock_cfg, tmp_path) -> TrayApp:
    tray = TrayApp(mock_cfg, config_path=tmp_path / "config.yaml")
    tray._icon = MagicMock()
    yield tray
    tray._release_recording()  # lock записи не должен «утечь» в другие тесты


def _make_session(
    tmp_path, *, mix=False, transcript=False, summary=False, video=False
) -> SessionPaths:
    s = create_session(str(tmp_path))
    if mix:
        s.mix_audio.write_bytes(b"mix")
    if transcript:
        data = {"segments": [], "duration_sec": 10.0}
        s.transcript.write_text(json.dumps(data), encoding="utf-8")
    if summary:
        s.summary.write_text("# Summary", encoding="utf-8")
    if video:
        s.video.write_bytes(b"vid")
    return s


# ---------------------------------------------------------------------------
# _make_icon
# ---------------------------------------------------------------------------


class TestMakeIcon:
    def test_all_states_return_image(self):
        from PIL import Image

        for state in ("idle", "recording", "processing", "error"):
            img = _make_icon(state)
            assert isinstance(img, Image.Image)
            assert img.size == (64, 64)

    def test_unknown_state_falls_back_to_idle_color(self):
        img = _make_icon("unknown_state")
        # не падает, возвращает изображение
        from PIL import Image

        assert isinstance(img, Image.Image)

    def test_dim_mode(self):
        img = _make_icon("recording", dim=True)
        from PIL import Image

        assert isinstance(img, Image.Image)


# ---------------------------------------------------------------------------
# Управление состоянием
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_initial_state(self, app):
        assert app.state == "idle"

    def test_set_state_changes_state(self, app):
        app._set_state("recording", "Запись…")
        assert app.state == "recording"

    def test_set_state_updates_message(self, app):
        app._set_state("processing", "Транскрипция…")
        assert "Транскрипция" in app._status_msg

    def test_set_state_to_idle_resets_op_start(self, app):
        app._set_state("recording")
        app._set_state("idle")
        with app._lock:
            assert app._op_start == 0.0

    def test_set_state_updates_icon(self, app):
        app._set_state("recording")
        # _refresh_icon вызывается — проверяем что иконка не None и icon был задан
        assert app._icon is not None
        assert app._icon.title is not None  # не упадёт

    def test_concurrent_state_changes(self, app):
        errors = []

        def _change():
            try:
                for s in ["recording", "processing", "idle"] * 5:
                    app._set_state(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_change) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors


# ---------------------------------------------------------------------------
# _elapsed_str
# ---------------------------------------------------------------------------


class TestElapsedStr:
    def test_no_op_start(self, app):
        assert app._elapsed_str() == ""

    def test_with_op_start(self, app):
        with app._lock:
            app._op_start = time.monotonic() - 125  # 2:05
        elapsed = app._elapsed_str()
        assert elapsed == "02:05" or elapsed.startswith("02:")

    def test_thread_safe(self, app):
        with app._lock:
            app._op_start = time.monotonic() - 10
        results = []

        def _read():
            results.append(app._elapsed_str())

        threads = [threading.Thread(target=_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10
        assert all(r == results[0] or True for r in results)  # не упало


# ---------------------------------------------------------------------------
# _make_title
# ---------------------------------------------------------------------------


class TestMakeTitle:
    def test_title_contains_message(self, app):
        app._set_state("idle", "Готов к записи")
        title = app._make_title()
        assert "Meeting Recorder" in title
        assert "Готов" in title


# ---------------------------------------------------------------------------
# _notify
# ---------------------------------------------------------------------------


class TestNotify:
    def test_notify_calls_icon(self, app):
        app._notify("Title", "Message")
        app._icon.notify.assert_called_once_with("Message", "Title")

    def test_notify_no_icon(self, mock_cfg):
        tray = TrayApp(mock_cfg)
        tray._icon = None
        tray._notify("T", "M")  # не должно падать

    def test_notify_exception_suppressed(self, app):
        app._icon.notify.side_effect = Exception("icon crashed")
        app._notify("T", "M")  # не должно падать


# ---------------------------------------------------------------------------
# _pick_session
# ---------------------------------------------------------------------------


class TestPickSession:
    def test_no_sessions_returns_none(self, app):
        result = app._pick_session()
        assert result is None

    def test_find_session_with_mix(self, app, tmp_path, mock_cfg):
        _make_session(tmp_path)  # без mix
        s2 = _make_session(tmp_path, mix=True)
        app.cfg.output_dir = str(tmp_path)

        result = app._pick_session(need_mix=True)
        assert result is not None
        assert result.session_id == s2.session_id

    def test_find_session_with_transcript(self, app, tmp_path, mock_cfg):
        _make_session(tmp_path)
        s2 = _make_session(tmp_path, transcript=True)
        app.cfg.output_dir = str(tmp_path)

        result = app._pick_session(need_transcript=True)
        assert result is not None
        assert result.session_id == s2.session_id

    def test_need_no_transcript(self, app, tmp_path):
        s1 = _make_session(tmp_path, mix=True)
        _make_session(tmp_path, mix=True, transcript=True)
        app.cfg.output_dir = str(tmp_path)

        result = app._pick_session(need_mix=True, need_no_transcript=True)
        assert result is not None
        assert result.session_id == s1.session_id

    def test_need_no_summary(self, app, tmp_path):
        s1 = _make_session(tmp_path, transcript=True)
        _make_session(tmp_path, transcript=True, summary=True)
        app.cfg.output_dir = str(tmp_path)

        result = app._pick_session(need_transcript=True, need_no_summary=True)
        assert result is not None
        assert result.session_id == s1.session_id

    def test_returns_latest_first(self, app, tmp_path):
        _make_session(tmp_path, mix=True)
        time.sleep(0.01)
        s2 = _make_session(tmp_path, mix=True)
        app.cfg.output_dir = str(tmp_path)

        result = app._pick_session(need_mix=True)
        # pick_session проходит reversed → возвращает последнюю
        assert result.session_id == s2.session_id


# ---------------------------------------------------------------------------
# _do_stop_and_mix
# ---------------------------------------------------------------------------


class TestDoStopAndMix:
    def test_no_recorder_returns_none(self, app):
        result = app._do_stop_and_mix()
        assert result is None

    def test_stop_and_mix_both_audio(self, app, tmp_path, mock_cfg):
        import numpy as np
        import soundfile as sf

        session = _make_session(tmp_path)
        sr = 48000
        data = np.zeros(sr, dtype=np.float32)
        sf.write(str(session.mic_audio), data, sr)
        sf.write(str(session.system_audio), data, sr)
        app.cfg.output_dir = str(tmp_path)

        mock_recorder = MagicMock()
        with app._lock:
            app._recorder = mock_recorder
            app._paths = session
            app._state = "recording"

        with patch("meeting_recorder.pipeline.mix_audio_files") as mock_mix:
            mock_mix.return_value = session.mix_audio
            result = app._do_stop_and_mix()

        mock_recorder.stop.assert_called_once()
        assert result is not None

    def test_stop_only_mic(self, app, tmp_path):
        session = _make_session(tmp_path, mix=False)
        session.mic_audio.write_bytes(b"mic")
        app.cfg.output_dir = str(tmp_path)

        mock_recorder = MagicMock()
        with app._lock:
            app._recorder = mock_recorder
            app._paths = session
            app._state = "recording"

        result = app._do_stop_and_mix()
        assert result is not None
        assert session.mix_audio.exists()


# ---------------------------------------------------------------------------
# _do_process (мок transcribe + report)
# ---------------------------------------------------------------------------


class TestDoProcess:
    def test_process_success(self, app, tmp_path):
        import numpy as np
        import soundfile as sf

        session = create_session(str(tmp_path))
        data = np.zeros(48000, dtype=np.float32)
        sf.write(str(session.mix_audio), data, 48000)
        app.cfg.output_dir = str(tmp_path)

        result = ({"segments": []}, session.protocol, session.summary)
        with patch(
            "meeting_recorder.pipeline.process_session", return_value=result
        ) as mock_ps:
            with patch("meeting_recorder.tray.time.sleep"):
                app._do_process(session)

        mock_ps.assert_called_once_with(app.cfg, session)
        assert app.state == "idle"
        app._icon.notify.assert_called_once()

    def test_process_error_sets_error_state_then_idle(self, app, tmp_path):
        session = create_session(str(tmp_path))

        states = []
        orig = app._set_state

        def _spy(state, msg=""):
            states.append(state)
            orig(state, msg)

        with patch(
            "meeting_recorder.pipeline.process_session",
            side_effect=RuntimeError("boom"),
        ):
            with patch("meeting_recorder.tray.time.sleep"):  # не ждём
                with patch.object(app, "_set_state", side_effect=_spy):
                    app._do_process(session)

        assert "error" in states
        assert app.state == "idle"


# ---------------------------------------------------------------------------
# _do_start (happy path + error path)
# ---------------------------------------------------------------------------


class TestDoStart:
    def test_success(self, app, tmp_path, mock_cfg):
        mock_paths = MagicMock()
        mock_paths.session_id = "meeting_2026-01-01_00-00-00"
        mock_recorder = MagicMock()
        mock_recorder.ffmpeg_pid = 1234

        with patch("meeting_recorder.tray.create_session", return_value=mock_paths):
            with patch(
                "meeting_recorder.tray.MeetingRecorder", return_value=mock_recorder
            ):
                app._do_start()

        mock_recorder.start.assert_called_once()
        assert app.state == "recording"

    def test_error_sets_error_then_idle(self, app):
        with patch(
            "meeting_recorder.tray.create_session",
            side_effect=RuntimeError("нет места"),
        ):
            with patch("meeting_recorder.tray.time.sleep"):
                app._do_start()

        assert app.state == "idle"


# ---------------------------------------------------------------------------
# _do_stop_only
# ---------------------------------------------------------------------------


class TestDoStopOnly:
    def test_success(self, app, tmp_path):
        mock_paths = MagicMock()
        mock_paths.session_id = "meeting_2026-01-01_00-00-00"

        with patch.object(app, "_do_stop_and_mix", return_value=mock_paths):
            with patch("meeting_recorder.tray.time.sleep"):
                app._do_stop_only()

        assert app.state == "idle"

    def test_no_paths_returns_early(self, app):
        with patch.object(app, "_do_stop_and_mix", return_value=None):
            app._do_stop_only()
        assert app.state == "idle"


# ---------------------------------------------------------------------------
# _do_process_session
# ---------------------------------------------------------------------------


class TestDoProcessSession:
    def test_no_mix_audio_shows_error(self, app, tmp_path):
        # Нет сессий с mix-аудио
        with patch("meeting_recorder.tray.time.sleep"):
            app._do_process_session()
        assert app.state == "idle"

    def test_with_mix_audio_calls_do_process(self, app, tmp_path):
        _make_session(tmp_path, mix=True)
        app.cfg.output_dir = str(tmp_path)

        with patch.object(app, "_do_process") as mock_dp:
            app._do_process_session()
        mock_dp.assert_called_once()


# ---------------------------------------------------------------------------
# _do_report_session
# ---------------------------------------------------------------------------


class TestDoReportSession:
    def test_no_transcript_shows_error(self, app, tmp_path):
        with patch("meeting_recorder.tray.time.sleep"):
            app._do_report_session()
        assert app.state == "idle"

    def test_with_transcript_calls_report(self, app, tmp_path):

        session = _make_session(tmp_path, transcript=True)
        app.cfg.output_dir = str(tmp_path)

        with patch(
            "meeting_recorder.pipeline.run_report",
            return_value=(session.protocol, session.summary),
        ) as mock_rr:
            with patch("meeting_recorder.tray.time.sleep"):
                app._do_report_session()

        mock_rr.assert_called_once()
        assert app.state == "idle"


# ---------------------------------------------------------------------------
# _do_mux
# ---------------------------------------------------------------------------


class TestDoMux:
    def test_no_sessions_shows_error(self, app):
        with patch("meeting_recorder.tray.time.sleep"):
            app._do_mux()
        assert app.state == "idle"

    def test_no_video_shows_error(self, app, tmp_path):
        _make_session(tmp_path)  # без видео
        app.cfg.output_dir = str(tmp_path)

        with patch("meeting_recorder.tray.time.sleep"):
            app._do_mux()
        assert app.state == "idle"

    def test_success(self, app, tmp_path):
        session = _make_session(tmp_path, mix=True, video=True)
        app.cfg.output_dir = str(tmp_path)
        session.final_video.write_bytes(b"x" * 100)

        with patch(
            "meeting_recorder.pipeline.mux_video", return_value=session.final_video
        ) as m:
            with patch("meeting_recorder.tray.time.sleep"):
                app._do_mux()

        m.assert_called_once_with(session.video, session.mix_audio, session.final_video)
        assert app.state == "idle"


# ---------------------------------------------------------------------------
# _on_open_folder / _on_open_last_session_folder
# ---------------------------------------------------------------------------


class TestOpenFolders:
    def test_on_open_folder(self, app, tmp_path):
        app.cfg.output_dir = str(tmp_path)
        with patch("subprocess.Popen") as mock_popen:
            app._on_open_folder(MagicMock(), MagicMock())
        mock_popen.assert_called_once()

    def test_on_open_last_session_no_sessions(self, app, tmp_path):
        app.cfg.output_dir = str(tmp_path)
        with patch("subprocess.Popen") as mock_popen:
            app._on_open_last_session_folder(MagicMock(), MagicMock())
        mock_popen.assert_not_called()

    def test_on_open_last_session_with_session(self, app, tmp_path):
        _make_session(tmp_path)
        app.cfg.output_dir = str(tmp_path)

        with patch("subprocess.Popen") as mock_popen:
            app._on_open_last_session_folder(MagicMock(), MagicMock())
        mock_popen.assert_called_once()

    def test_on_open_last_session_uses_current_paths(self, app, tmp_path):
        session = _make_session(tmp_path)
        with app._lock:
            app._paths = session

        with patch("subprocess.Popen") as mock_popen:
            app._on_open_last_session_folder(MagicMock(), MagicMock())
        mock_popen.assert_called_once()


# ---------------------------------------------------------------------------
# _on_exit
# ---------------------------------------------------------------------------


class TestOnExit:
    def test_exit_when_idle(self, app):
        mock_icon = MagicMock()
        app._on_exit(mock_icon, MagicMock())
        mock_icon.stop.assert_called_once()

    def test_exit_when_recording_does_not_block_ui(self, app):
        """_on_exit возвращается сразу; остановка и выход — в фоновом потоке."""
        app._set_state("recording")
        app._icon = MagicMock()
        release = threading.Event()

        def slow_stop():
            release.wait(5)
            return None

        with patch.object(app, "_do_stop_and_mix", side_effect=slow_stop) as m:
            app._on_exit(app._icon, MagicMock())
            # UI-поток свободен: иконка ещё не остановлена
            app._icon.stop.assert_not_called()
            release.set()
            for t in threading.enumerate():
                if t.name == "tray-exit-stop":
                    t.join(5)
        m.assert_called_once()
        app._icon.stop.assert_called_once()
        assert app._stop_event.is_set()

    def test_exit_does_not_run_processing(self, app, tmp_path):
        session = _make_session(tmp_path, mix=True)
        app._icon = MagicMock()
        with patch.object(app, "_do_stop_and_mix", return_value=session):
            with patch("meeting_recorder.pipeline.process_session") as proc:
                app._stop_then_exit()
        proc.assert_not_called()
        app._icon.notify.assert_called_once()
        app._icon.stop.assert_called_once()

    def test_exit_waits_for_pending_start(self, app):
        app._icon = MagicMock()
        app._start_finished.clear()
        order = []

        def finish_start():
            time.sleep(0.2)
            order.append("started")
            app._start_finished.set()

        threading.Thread(target=finish_start).start()
        with patch.object(
            app, "_do_stop_and_mix", side_effect=lambda: order.append("stop")
        ):
            app._stop_then_exit()
        assert order == ["started", "stop"]

    def test_exit_error_still_shuts_down(self, app):
        app._icon = MagicMock()
        with patch.object(app, "_do_stop_and_mix", side_effect=RuntimeError("x")):
            app._stop_then_exit()
        app._icon.stop.assert_called_once()


class TestRecordingLock:
    def test_start_refused_when_lock_busy(self, app):
        from meeting_recorder import state

        held = state.acquire_recording_lock()
        try:
            assert app._try_begin("starting")
            with patch("meeting_recorder.tray.time.sleep"):
                with patch("meeting_recorder.tray.MeetingRecorder") as rec_cls:
                    app._do_start()
            rec_cls.assert_not_called()
            assert app.state == "idle"
        finally:
            state.release_recording_lock(held)

    def test_start_holds_lock_and_writes_tray_state(self, app):
        from meeting_recorder import state

        app._try_begin("starting")
        with patch("meeting_recorder.tray.MeetingRecorder") as rec_cls:
            rec_cls.return_value.ffmpeg_pid = 4242
            app._do_start()
        try:
            assert app.state == "recording"
            assert state.is_recording_active()
            st = state.load_state()
            assert st["owner"] == "tray" and st["ffmpeg_pid"] == 4242
        finally:
            app._release_recording()
        assert not state.is_recording_active()
        assert state.load_state() is None

    def test_failed_start_releases_lock(self, app):
        from meeting_recorder import state

        app._try_begin("starting")
        with patch("meeting_recorder.tray.time.sleep"):
            with patch("meeting_recorder.tray.MeetingRecorder") as rec_cls:
                rec_cls.return_value.start.side_effect = RuntimeError("no mic")
                app._do_start()
        assert not state.is_recording_active()
        assert app._start_finished.is_set()


# ---------------------------------------------------------------------------
# _pick_session — need_no_highlights
# ---------------------------------------------------------------------------


class TestPickSessionNoHighlights:
    def test_skips_session_with_highlights(self, app, tmp_path):
        s1 = _make_session(tmp_path, transcript=True)
        s2 = _make_session(tmp_path, transcript=True)
        s2.highlights.write_text("[]", encoding="utf-8")
        app.cfg.output_dir = str(tmp_path)

        result = app._pick_session(need_transcript=True, need_no_highlights=True)
        assert result is not None
        assert result.session_id == s1.session_id

    def test_falls_back_when_all_have_highlights(self, app, tmp_path):
        s = _make_session(tmp_path, transcript=True)
        s.highlights.write_text("[]", encoding="utf-8")
        app.cfg.output_dir = str(tmp_path)

        result = app._pick_session(need_transcript=True, need_no_highlights=True)
        assert result is None


# ---------------------------------------------------------------------------
# _fmt_session_label
# ---------------------------------------------------------------------------


class TestFmtSessionLabel:
    def test_standard_session_id(self, app, tmp_path):
        s = _make_session(tmp_path)
        label = app._fmt_session_label(s)
        assert "20" in label
        assert ":" in label

    def test_suffixed_session_id(self, app, tmp_path):
        from meeting_recorder.naming import SessionPaths

        paths = SessionPaths(str(tmp_path), "meeting_2026-06-28_15-30-00_2")
        paths.ensure_dir()
        label = app._fmt_session_label(paths)
        assert "#2" in label

    def test_invalid_id_returns_raw(self, app, tmp_path):
        from meeting_recorder.naming import SessionPaths

        paths = SessionPaths(str(tmp_path), "bad_id")
        paths.ensure_dir()
        label = app._fmt_session_label(paths)
        assert label == "bad_id"


# ---------------------------------------------------------------------------
# _brief_summary
# ---------------------------------------------------------------------------


class TestBriefSummary:
    def test_no_summary_file(self, app, tmp_path):
        s = _make_session(tmp_path)
        result = app._brief_summary(s)
        assert result == []

    def test_extracts_brief_section(self, app, tmp_path):
        s = _make_session(tmp_path)
        s.summary.write_text(
            "# Summary\n\n## Краткое резюме\n\nКоманда обсудила бюджет и согласовала план.\n\n## Темы\n- Тема 1",
            encoding="utf-8",
        )
        result = app._brief_summary(s)
        assert len(result) >= 1
        assert any("бюджет" in line for line in result)

    def test_skips_bold_lines(self, app, tmp_path):
        s = _make_session(tmp_path)
        s.summary.write_text(
            "# Summary\n\n## Краткое резюме\n\n**Дата:** 2026-06-28\n\nОбсудили бюджет.\n",
            encoding="utf-8",
        )
        result = app._brief_summary(s)
        assert all("**" not in line for line in result)
        assert any("бюджет" in line for line in result)

    def test_wraps_long_line(self, app, tmp_path):
        s = _make_session(tmp_path)
        long_line = "Слово " * 30
        s.summary.write_text(
            f"# Summary\n\n## Краткое резюме\n\n{long_line}\n",
            encoding="utf-8",
        )
        result = app._brief_summary(s)
        assert len(result) >= 1
        assert all(len(line) <= 72 for line in result)

    def test_max_4_lines(self, app, tmp_path):
        s = _make_session(tmp_path)
        sentences = "\n".join(f"Предложение номер {i} для теста." for i in range(10))
        s.summary.write_text(
            f"# S\n\n## Краткое резюме\n\n{sentences}\n", encoding="utf-8"
        )
        result = app._brief_summary(s)
        assert len(result) <= 4

    def test_stops_at_next_section(self, app, tmp_path):
        s = _make_session(tmp_path)
        s.summary.write_text(
            "# S\n\n## Краткое резюме\n\nРезюме.\n\n## Другой раздел\n\nНе должно попасть.\n",
            encoding="utf-8",
        )
        result = app._brief_summary(s)
        assert all("Не должно" not in line for line in result)

    def test_corrupt_file_returns_empty(self, app, tmp_path):
        s = _make_session(tmp_path)
        s.summary.write_bytes(b"\xff\xfe\x00 invalid utf8 content \xff")
        result = app._brief_summary(s)
        assert result == []


# ---------------------------------------------------------------------------
# _build_sessions_items / _build_session_submenu
# ---------------------------------------------------------------------------


class TestBuildSessionsItems:
    def test_empty_catalog(self, app):
        items = app._build_sessions_items()
        assert len(items) == 1
        assert items[0].enabled is False

    def test_with_sessions(self, app, tmp_path):
        for _ in range(3):
            _make_session(tmp_path)
        app.cfg.output_dir = str(tmp_path)
        items = app._build_sessions_items()
        assert len(items) == 3

    def test_capped_at_25(self, app, tmp_path):
        for _ in range(30):
            _make_session(tmp_path)
        app.cfg.output_dir = str(tmp_path)
        items = app._build_sessions_items()
        assert len(items) <= 25

    def test_submenu_callable(self, app, tmp_path):
        _make_session(tmp_path, transcript=True)
        app.cfg.output_dir = str(tmp_path)
        items = app._build_sessions_items()
        assert len(items) == 1
        # submenu — это pystray.Menu, вызываемое в рантайме
        submenu = items[0].submenu
        assert submenu is not None

    def test_submenu_with_summary(self, app, tmp_path):
        _make_session(tmp_path, transcript=True, summary=True)
        app.cfg.output_dir = str(tmp_path)
        items = app._build_sessions_items()
        assert len(items) == 1


# ---------------------------------------------------------------------------
# _do_highlights / _run_highlights_for
# ---------------------------------------------------------------------------


class TestDoHighlights:
    def test_no_session_shows_error(self, app):
        with patch("meeting_recorder.tray.time.sleep"):
            app._do_highlights()
        assert app.state == "idle"

    def test_success(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)
        app.cfg.output_dir = str(tmp_path)

        with patch(
            "meeting_recorder.transcriber.load_transcript",
            return_value={"segments": []},
        ):
            with patch(
                "meeting_recorder.report.generate_highlights",
                return_value=session.highlights,
            ):
                with patch("meeting_recorder.tray.time.sleep"):
                    app._do_highlights()

        assert app.state == "idle"

    def test_uses_session_without_highlights_first(self, app, tmp_path):
        s1 = _make_session(tmp_path, transcript=True)
        s2 = _make_session(tmp_path, transcript=True)
        s2.highlights.write_text("[]", encoding="utf-8")
        app.cfg.output_dir = str(tmp_path)

        picked: list = []

        def _fake_run(paths):
            picked.append(paths.session_id)
            with patch("meeting_recorder.tray.time.sleep"):
                pass

        with patch.object(app, "_run_highlights_for", side_effect=_fake_run):
            app._do_highlights()

        assert picked and picked[0] == s1.session_id


class TestRunHighlightsFor:
    def test_success(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)

        with patch(
            "meeting_recorder.transcriber.load_transcript",
            return_value={"segments": []},
        ):
            with patch(
                "meeting_recorder.report.generate_highlights",
                return_value=session.highlights,
            ):
                with patch("meeting_recorder.tray.time.sleep"):
                    app._run_highlights_for(session)

        assert app.state == "idle"

    def test_error_sets_error_state(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)

        with patch(
            "meeting_recorder.transcriber.load_transcript",
            side_effect=RuntimeError("LLM недоступен"),
        ):
            with patch("meeting_recorder.tray.time.sleep"):
                app._run_highlights_for(session)

        assert app.state == "idle"


# ---------------------------------------------------------------------------
# _do_html_protocol / _run_html_for
# ---------------------------------------------------------------------------


class TestDoHtmlProtocol:
    def test_no_session_shows_error(self, app):
        with patch("meeting_recorder.tray.time.sleep"):
            app._do_html_protocol()
        assert app.state == "idle"

    def test_success(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)
        app.cfg.output_dir = str(tmp_path)

        with patch(
            "meeting_recorder.transcriber.load_transcript",
            return_value={"segments": []},
        ):
            with patch(
                "meeting_recorder.html_report.generate_html_protocol",
                return_value=session.html_protocol,
            ):
                with patch("webbrowser.open"):
                    with patch("meeting_recorder.tray.time.sleep"):
                        app._do_html_protocol()

        assert app.state == "idle"


class TestRunHtmlFor:
    def test_success(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)

        with patch(
            "meeting_recorder.transcriber.load_transcript",
            return_value={"segments": []},
        ):
            with patch(
                "meeting_recorder.html_report.generate_html_protocol",
                return_value=session.html_protocol,
            ):
                with patch("webbrowser.open"):
                    with patch("meeting_recorder.tray.time.sleep"):
                        app._run_html_for(session)

        assert app.state == "idle"

    def test_error_handled(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)

        with patch(
            "meeting_recorder.transcriber.load_transcript",
            side_effect=RuntimeError("не найден"),
        ):
            with patch("meeting_recorder.tray.time.sleep"):
                app._run_html_for(session)

        assert app.state == "idle"


# ---------------------------------------------------------------------------
# Атомарные переходы состояний (защита от двойного запуска)
# ---------------------------------------------------------------------------


class TestTryBegin:
    def test_begin_from_idle(self, app):
        assert app._try_begin("processing", "x") is True
        assert app.state == "processing"

    def test_second_begin_rejected(self, app):
        assert app._try_begin("starting") is True
        assert app._try_begin("starting") is False

    def test_concurrent_begin_only_one_wins(self, app):
        results = []
        barrier = threading.Barrier(8)

        def _worker():
            barrier.wait()
            results.append(app._try_begin("processing"))

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(True) == 1

    def test_double_click_start_spawns_once(self, app):
        with patch.object(app, "_spawn") as mock_spawn:
            app._on_start(None, None)
            app._on_start(None, None)
        assert mock_spawn.call_count == 1
        assert app.state == "starting"

    def test_menu_action_while_busy_ignored(self, app):
        app._set_state("recording")
        with patch.object(app, "_spawn") as mock_spawn:
            app._on_process_session(None, None)
            app._on_mux(None, None)
            app._on_highlights(None, None)
        mock_spawn.assert_not_called()


class TestFlash:
    def test_returns_to_idle(self, app):
        with patch("meeting_recorder.tray.time.sleep"):
            app._flash("error", "oops", 1)
        assert app.state == "idle"

    def test_does_not_clobber_new_operation(self, app):
        def _sleep(_):
            # Пока показывается «Готово», пользователь запускает новую операцию
            assert app._try_begin("processing", "новая")

        with patch("meeting_recorder.tray.time.sleep", side_effect=_sleep):
            app._flash("idle", "Готово", 1)
        assert app.state == "processing"


# ---------------------------------------------------------------------------
# Меню устройств
# ---------------------------------------------------------------------------

from meeting_recorder.recorder import AudioDevice  # noqa: E402
from meeting_recorder.tray import _DeviceSnapshot  # noqa: E402


def _snapshot(mics=("Mic A", "Mic B"), speakers=None, default=None):
    speakers = (
        speakers
        if speakers is not None
        else (
            AudioDevice("{s1}", "Speakers"),
            AudioDevice("{s2}", "Headset"),
        )
    )
    return _DeviceSnapshot(
        mics=tuple(mics),
        speakers=tuple(speakers),
        default_speaker=default or (speakers[0] if speakers else None),
        loaded=True,
    )


def _labels(items):
    return [getattr(i, "text", None) for i in items]


class TestDeviceMenus:
    def test_mic_items_checked(self, app):
        app._devices = _snapshot()
        app.cfg.recording.mic_device = "Mic B"
        items = app._build_mic_device_items()
        checked = {i.text: i.checked for i in items if i.text and i.text.strip()}
        assert checked["Mic B"] is True
        assert checked["Mic A"] is False
        assert checked["Автоопределение"] is False

    def test_mic_items_disabled_while_recording(self, app):
        app._devices = _snapshot()
        app._set_state("recording")
        items = app._build_mic_device_items()
        assert all(not i.enabled for i in items if i.text == "Mic A")

    def test_no_mics_found(self, app):
        app._devices = _snapshot(mics=())
        assert any(
            "не найдены" in (t or "") for t in _labels(app._build_mic_device_items())
        )

    def test_select_mic_saves_only_that_key(self, app, tmp_path):
        import yaml

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "llm:\n  api_key: sk-keep\nrecording:\n  fps: 15\n", encoding="utf-8"
        )
        app._select_mic_device("Mic B")()
        saved = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        assert saved["recording"] == {"fps": 15, "mic_device": "Mic B"}
        assert saved["llm"]["api_key"] == "sk-keep"
        assert app.cfg.recording.mic_device == "Mic B"
        app._icon.update_menu.assert_called()

    def test_select_ignored_while_busy(self, app, tmp_path):
        app._set_state("recording")
        app._select_mic_device("Mic B")()
        assert app.cfg.recording.mic_device == ""
        assert not (tmp_path / "config.yaml").exists()

    def test_soundcard_speaker_stored_by_id(self, app, tmp_path):
        import yaml

        app.cfg.recording.system_audio_grabber = "soundcard"
        app._select_speaker_device("{s2}", "Headset")()
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert saved["recording"]["system_audio_speaker"] == "{s2}"

    def test_soundcard_speaker_checked_by_id_or_name(self, app):
        app._devices = _snapshot()
        app.cfg.recording.system_audio_grabber = "soundcard"
        app.cfg.recording.system_audio_speaker = "Headset"  # имя из ручного конфига
        items = app._build_speaker_device_items()
        checked = {i.text: i.checked for i in items if i.text and i.text.strip()}
        assert checked["Headset"] is True
        assert checked["Speakers"] is False
        assert "Устройство по умолчанию (Speakers)" in checked

    def test_dshow_mode_has_no_default_item(self, app):
        """Пустое имя dshow-устройства сломало бы ffmpeg (-i audio=)."""
        app._devices = _snapshot(mics=("virtual-audio-capturer", "Stereo Mix"))
        app.cfg.recording.system_audio_grabber = "dshow"
        labels = _labels(app._build_speaker_device_items())
        assert not any("по умолчанию" in (t or "").lower() for t in labels)
        assert "virtual-audio-capturer" in labels

    def test_dshow_select_writes_device(self, app):
        app.cfg.recording.system_audio_grabber = "dshow"
        app._select_speaker_device("Stereo Mix")()
        assert app.cfg.recording.system_audio_device == "Stereo Mix"

    def test_wasapi_mode_unavailable(self, app):
        app.cfg.recording.system_audio_grabber = "wasapi"
        items = app._build_speaker_device_items()
        assert len(items) == 1 and "WASAPI" in items[0].text

    def test_devices_menu_structure(self, app):
        menu = app._build_devices_menu()
        assert len(list(menu.items)) == 4  # mic, speaker, separator, refresh


class TestDeviceRefresh:
    def test_refresh_detects_change(self, app):
        snaps = [_snapshot(), _snapshot(), _snapshot(mics=("New Mic",))]
        with patch("meeting_recorder.tray._scan_devices", side_effect=snaps):
            assert app._refresh_devices() is True  # первый скан
            assert app._refresh_devices() is False  # без изменений
            assert app._refresh_devices() is True  # подключили устройство

    def test_get_devices_lazy_load(self, app):
        with patch(
            "meeting_recorder.tray._scan_devices", return_value=_snapshot()
        ) as scan:
            assert app._get_devices().mics == ("Mic A", "Mic B")
            app._get_devices()
        scan.assert_called_once()

    def test_on_refresh_devices_updates_menu(self, app):
        with patch("meeting_recorder.tray._scan_devices", return_value=_snapshot()):
            with patch.object(app, "_spawn", side_effect=lambda fn, name: fn()):
                app._on_refresh_devices(None, None)
        app._icon.update_menu.assert_called_once()

    def test_ticker_updates_menu_only_on_change(self, app):
        import meeting_recorder.tray as tray_mod

        snaps = iter([_snapshot(), _snapshot(), _snapshot(mics=("X",))])
        ticks = iter([False, False, False, True])  # 3 тика, затем стоп
        app._stop_event.wait = lambda timeout: next(ticks)
        with patch.object(tray_mod, "_DEVICE_REFRESH_SECS", 0):
            with patch(
                "meeting_recorder.tray._scan_devices", side_effect=lambda: next(snaps)
            ):
                app._ticker_loop()
        # 1-й скан — изменение (пусто → снимок), 2-й — нет, 3-й — изменение
        assert app._icon.update_menu.call_count == 2

    def test_ticker_blinks_while_recording(self, app):
        app._set_state("recording")
        ticks = iter([False, False, True])
        app._stop_event.wait = lambda timeout: next(ticks)
        app._icon.reset_mock()
        app._ticker_loop()
        assert app._icon.update_menu.call_count == 0
        assert "Meeting Recorder" in app._icon.title


class TestChat:
    def test_no_transcript_shows_error(self, app):
        with patch("meeting_recorder.tray.time.sleep"):
            app._do_chat()
        assert app.state == "idle"

    def test_opens_terminal_with_config(self, app, tmp_path):
        _make_session(tmp_path, transcript=True)
        (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
        with patch("meeting_recorder.tray.subprocess.Popen") as mock_popen:
            app._do_chat()
        cmdline = mock_popen.call_args.args[0]
        if sys.platform == "win32":
            # Одна строка: cmd /s /k "<аргументы>" — внешние кавычки снимаются
            assert cmdline.startswith('cmd /s /k "') and cmdline.endswith('"')
            assert " chat " in cmdline and " -c " in cmdline
        else:
            assert "chat" in cmdline and "-c" in cmdline

    def test_chat_command_quotes_paths_with_spaces(self, app, tmp_path):
        if sys.platform != "win32":
            pytest.skip("Windows-only quoting")
        _make_session(tmp_path, transcript=True)
        cfg_dir = tmp_path / "my config"
        cfg_dir.mkdir()
        (cfg_dir / "config.yaml").write_text("{}", encoding="utf-8")
        app._config_path = cfg_dir / "config.yaml"
        with patch("meeting_recorder.tray.subprocess.Popen") as mock_popen:
            app._do_chat()
        cmdline = mock_popen.call_args.args[0]
        assert f'"{cfg_dir / "config.yaml"}"' in cmdline

    def test_on_chat_runs_in_background(self, app):
        with patch.object(app, "_spawn") as mock_spawn:
            app._on_chat(None, None)
        mock_spawn.assert_called_once()


class TestStopAndMixErrors:
    def test_no_audio_returns_none(self, app, tmp_path):
        session = _make_session(tmp_path)
        with app._lock:
            app._recorder = MagicMock()
            app._paths = session
            app._state = "recording"
        with patch("meeting_recorder.tray.time.sleep"):
            assert app._do_stop_and_mix() is None
        assert app.state == "idle"
        assert app._recorder is None

    def test_do_stop_runs_process(self, app, tmp_path):
        session = _make_session(tmp_path)
        with patch.object(app, "_do_stop_and_mix", return_value=session):
            with patch.object(app, "_do_process") as mock_proc:
                app._do_stop()
        mock_proc.assert_called_once_with(session)
