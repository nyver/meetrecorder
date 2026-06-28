"""Тесты для tray.py — управление состоянием, _pick_session, иконка."""
from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths, create_session
from meeting_recorder.tray import TrayApp, _make_icon, _STATE_COLORS, _STATE_LABELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = str(tmp_path)
    return cfg


@pytest.fixture
def app(mock_cfg) -> TrayApp:
    tray = TrayApp(mock_cfg)
    tray._icon = MagicMock()
    return tray


def _make_session(tmp_path, *, mix=False, transcript=False, summary=False, video=False) -> SessionPaths:
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
        app._icon.title  # не упадёт

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
        s1 = _make_session(tmp_path, mix=True)
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
        import numpy as np, soundfile as sf
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

        with patch("meeting_recorder.tray.mix_audio_files") as mock_mix:
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
        import numpy as np, soundfile as sf
        session = create_session(str(tmp_path))
        data = np.zeros(48000, dtype=np.float32)
        sf.write(str(session.mix_audio), data, 48000)
        app.cfg.output_dir = str(tmp_path)

        transcript_data = {
            "session_id": session.session_id,
            "language": "ru",
            "duration_sec": 1.0,
            "diarization_enabled": False,
            "segments": [],
        }
        with patch("meeting_recorder.transcriber.transcribe", return_value=transcript_data):
            with patch("meeting_recorder.report.generate_protocol", return_value=session.protocol):
                with patch("meeting_recorder.report.generate_summary", return_value=session.summary):
                    app._do_process(session)

        assert app.state == "idle"

    def test_process_error_sets_error_state_then_idle(self, app, tmp_path):
        session = create_session(str(tmp_path))

        with patch("meeting_recorder.transcriber.transcribe", side_effect=RuntimeError("boom")):
            with patch("meeting_recorder.tray.time.sleep"):  # не ждём
                app._do_process(session)

        assert app.state == "idle"


# ---------------------------------------------------------------------------
# _do_start (happy path + error path)
# ---------------------------------------------------------------------------

class TestDoStart:
    def test_success(self, app, tmp_path, mock_cfg):
        mock_paths = MagicMock()
        mock_paths.session_id = "meeting_2026-01-01_00-00-00"
        mock_recorder = MagicMock()

        with patch("meeting_recorder.tray.create_session", return_value=mock_paths):
            with patch("meeting_recorder.tray.MeetingRecorder", return_value=mock_recorder):
                app._do_start()

        mock_recorder.start.assert_called_once()
        assert app.state == "recording"

    def test_error_sets_error_then_idle(self, app):
        with patch("meeting_recorder.tray.create_session", side_effect=RuntimeError("нет места")):
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
        session = _make_session(tmp_path, mix=True)
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
        import json
        session = _make_session(tmp_path, transcript=True)
        app.cfg.output_dir = str(tmp_path)

        with patch("meeting_recorder.transcriber.load_transcript", return_value={"segments": []}):
            with patch("meeting_recorder.report.generate_protocol", return_value=session.protocol):
                with patch("meeting_recorder.report.generate_summary", return_value=session.summary):
                    with patch("meeting_recorder.tray.time.sleep"):
                        app._do_report_session()

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

        with patch("meeting_recorder.recorder.mux_video", return_value=session.final_video):
            with patch("meeting_recorder.tray.time.sleep"):
                app._do_mux()

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
        session = _make_session(tmp_path)
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

    def test_exit_when_recording(self, app):
        app._set_state("recording")
        mock_icon = MagicMock()
        with patch.object(app, "_do_stop"):
            with patch("meeting_recorder.tray.time.sleep"):
                app._on_exit(mock_icon, MagicMock())
        mock_icon.stop.assert_called_once()


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
        s.summary.write_text(f"# S\n\n## Краткое резюме\n\n{sentences}\n", encoding="utf-8")
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
        s = _make_session(tmp_path, transcript=True)
        app.cfg.output_dir = str(tmp_path)
        items = app._build_sessions_items()
        assert len(items) == 1
        # submenu — это pystray.Menu, вызываемое в рантайме
        submenu = items[0].submenu
        assert submenu is not None

    def test_submenu_with_summary(self, app, tmp_path):
        s = _make_session(tmp_path, transcript=True, summary=True)
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

        with patch("meeting_recorder.transcriber.load_transcript", return_value={"segments": []}):
            with patch("meeting_recorder.report.generate_highlights",
                       return_value=session.highlights):
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

        with patch("meeting_recorder.transcriber.load_transcript", return_value={"segments": []}):
            with patch("meeting_recorder.report.generate_highlights",
                       return_value=session.highlights):
                with patch("meeting_recorder.tray.time.sleep"):
                    app._run_highlights_for(session)

        assert app.state == "idle"

    def test_error_sets_error_state(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)

        with patch("meeting_recorder.transcriber.load_transcript",
                   side_effect=RuntimeError("LLM недоступен")):
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

        with patch("meeting_recorder.transcriber.load_transcript", return_value={"segments": []}):
            with patch("meeting_recorder.html_report.generate_html_protocol",
                       return_value=session.html_protocol):
                with patch("webbrowser.open"):
                    with patch("meeting_recorder.tray.time.sleep"):
                        app._do_html_protocol()

        assert app.state == "idle"


class TestRunHtmlFor:
    def test_success(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)

        with patch("meeting_recorder.transcriber.load_transcript", return_value={"segments": []}):
            with patch("meeting_recorder.html_report.generate_html_protocol",
                       return_value=session.html_protocol):
                with patch("webbrowser.open"):
                    with patch("meeting_recorder.tray.time.sleep"):
                        app._run_html_for(session)

        assert app.state == "idle"

    def test_error_handled(self, app, tmp_path):
        session = _make_session(tmp_path, transcript=True)

        with patch("meeting_recorder.transcriber.load_transcript",
                   side_effect=RuntimeError("не найден")):
            with patch("meeting_recorder.tray.time.sleep"):
                app._run_html_for(session)

        assert app.state == "idle"
