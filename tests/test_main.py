"""Тесты CLI-команд из __main__.py через click.testing.CliRunner."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import threading
import pytest
from click.testing import CliRunner

from meeting_recorder.__main__ import (
    cli,
    _save_state,
    _load_state,
    _remove_state,
    _rename_with_retry,
    _mix_session_audio,
    _load_active_session,
    _build_chat_system_prompt,
    setup_logging,
    _STATE_FILE,
    _STOP_FILE,
    _DONE_FILE,
)
from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths, create_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_state_files():
    """Убираем state-файлы до и после каждого теста."""
    for f in (_STATE_FILE, _STOP_FILE, _DONE_FILE):
        f.unlink(missing_ok=True)
    yield
    for f in (_STATE_FILE, _STOP_FILE, _DONE_FILE):
        f.unlink(missing_ok=True)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_session(tmp_path) -> SessionPaths:
    return create_session(str(tmp_path))


@pytest.fixture
def mock_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = str(tmp_path)
    return cfg


def _invoke(runner, args, cfg=None):
    if cfg is None:
        cfg = AppConfig()
    with patch("meeting_recorder.__main__.load_config", return_value=cfg):
        return runner.invoke(cli, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# Вспомогательные функции состояния
# ---------------------------------------------------------------------------

class TestStateHelpers:
    def test_save_and_load(self):
        _save_state("meeting_2026-01-01_10-00-00", "/tmp/v.mp4", 12345)
        state = _load_state()
        assert state["session_id"] == "meeting_2026-01-01_10-00-00"
        assert state["ffmpeg_pid"] == 12345

    def test_load_missing_returns_none(self):
        assert _load_state() is None

    def test_remove_state(self):
        _save_state("sid", "/tmp/v.mp4")
        _remove_state()
        assert _load_state() is None

    def test_load_corrupted_returns_none(self):
        _STATE_FILE.parent.mkdir(exist_ok=True)
        _STATE_FILE.write_text("not-json", encoding="utf-8")
        assert _load_state() is None


# ---------------------------------------------------------------------------
# _rename_with_retry
# ---------------------------------------------------------------------------

class TestRenameWithRetry:
    def test_success(self, tmp_path):
        src = tmp_path / "a.txt"
        dst = tmp_path / "b.txt"
        src.write_text("hello")
        _rename_with_retry(src, dst, retries=3, delay=0.01)
        assert dst.exists()
        assert not src.exists()

    def test_raises_after_retries(self, tmp_path):
        src = tmp_path / "a.txt"
        dst = tmp_path / "b.txt"
        src.write_text("hello")
        with patch("meeting_recorder.__main__.Path.rename", side_effect=PermissionError("locked")):
            with pytest.raises(PermissionError):
                _rename_with_retry(src, dst, retries=2, delay=0.01)


# ---------------------------------------------------------------------------
# _mix_session_audio
# ---------------------------------------------------------------------------

class TestMixSessionAudio:
    def test_mix_both_exist(self, tmp_session, mock_cfg):
        import numpy as np
        import soundfile as sf
        sr = 48000
        data = np.zeros(sr, dtype=np.float32)
        sf.write(str(tmp_session.mic_audio), data, sr)
        sf.write(str(tmp_session.system_audio), data, sr)

        with patch("meeting_recorder.recorder.mix_audio_files") as mock_mix:
            mock_mix.return_value = tmp_session.mix_audio
            result = _mix_session_audio(tmp_session, mock_cfg)

        assert result is True
        mock_mix.assert_called_once()

    def test_only_mic_exists_renames(self, tmp_session, mock_cfg, tmp_path):
        tmp_session.mic_audio.write_bytes(b"mic")
        result = _mix_session_audio(tmp_session, mock_cfg)
        assert result is True
        assert tmp_session.mix_audio.exists()

    def test_only_system_exists_renames(self, tmp_session, mock_cfg):
        tmp_session.system_audio.write_bytes(b"sys")
        result = _mix_session_audio(tmp_session, mock_cfg)
        assert result is True
        assert tmp_session.mix_audio.exists()

    def test_no_audio_returns_false(self, tmp_session, mock_cfg):
        result = _mix_session_audio(tmp_session, mock_cfg)
        assert result is False


# ---------------------------------------------------------------------------
# _load_active_session
# ---------------------------------------------------------------------------

class TestLoadActiveSession:
    def test_no_state_returns_none(self, mock_cfg):
        sid, pid, paths = _load_active_session(mock_cfg)
        assert sid is None

    def test_with_valid_state(self, tmp_session, mock_cfg):
        _save_state(tmp_session.session_id, str(tmp_session.video), 12345)
        sid, pid, paths = _load_active_session(mock_cfg)
        assert sid == tmp_session.session_id
        assert pid == 12345
        assert paths is not None

    def test_corrupted_session_id(self, mock_cfg):
        _STATE_FILE.parent.mkdir(exist_ok=True)
        _STATE_FILE.write_text(json.dumps({"session_id": None}), encoding="utf-8")
        sid, pid, paths = _load_active_session(mock_cfg)
        assert sid is None

    def test_missing_session_dir(self, mock_cfg):
        _save_state("meeting_1900-01-01_00-00-00", "/tmp/v.mp4", 99)
        sid, pid, paths = _load_active_session(mock_cfg)
        assert sid is None


# ---------------------------------------------------------------------------
# _build_chat_system_prompt
# ---------------------------------------------------------------------------

class TestBuildChatSystemPrompt:
    def test_with_transcript(self, tmp_session, mock_cfg):
        data = {
            "session_id": tmp_session.session_id,
            "duration_sec": 120.0,
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Привет"},
            ],
        }
        tmp_session.transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        prompt = _build_chat_system_prompt(tmp_session, mock_cfg)
        assert "Привет" in prompt
        assert tmp_session.session_id in prompt

    def test_with_summary(self, tmp_session, mock_cfg):
        tmp_session.summary.write_text("# Summary\nКраткое содержание", encoding="utf-8")
        prompt = _build_chat_system_prompt(tmp_session, mock_cfg)
        assert "Краткое содержание" in prompt

    def test_without_any_data(self, tmp_session, mock_cfg):
        prompt = _build_chat_system_prompt(tmp_session, mock_cfg)
        assert tmp_session.session_id in prompt


# ---------------------------------------------------------------------------
# CLI: version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# CLI: list
# ---------------------------------------------------------------------------

class TestListCmd:
    def test_no_sessions(self, runner, mock_cfg):
        result = _invoke(runner, ["list"], mock_cfg)
        assert result.exit_code == 0
        assert "не найдено" in result.output

    def test_with_session(self, runner, mock_cfg, tmp_session):
        result = _invoke(runner, ["list"], mock_cfg)
        assert result.exit_code == 0
        assert tmp_session.session_id in result.output


# ---------------------------------------------------------------------------
# CLI: process
# ---------------------------------------------------------------------------

class TestProcessCmd:
    def test_success(self, runner, mock_cfg, tmp_session):
        with patch("meeting_recorder.__main__.run_process") as mock_proc:
            mock_proc.return_value = (
                {"session_id": tmp_session.session_id, "segments": []},
                tmp_session.protocol,
                tmp_session.summary,
            )
            result = _invoke(runner, ["process", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 0
        assert "завершена" in result.output

    def test_pipeline_error(self, runner, mock_cfg, tmp_session):
        from meeting_recorder.pipeline import PipelineError
        with patch("meeting_recorder.__main__.run_process", side_effect=PipelineError("bad")):
            result = _invoke(runner, ["process", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 1
        assert "bad" in result.output


# ---------------------------------------------------------------------------
# CLI: report
# ---------------------------------------------------------------------------

class TestReportCmd:
    def test_success(self, runner, mock_cfg, tmp_session):
        with patch("meeting_recorder.__main__.run_report_only") as mock_r:
            mock_r.return_value = (tmp_session.protocol, tmp_session.summary)
            result = _invoke(runner, ["report", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 0
        assert "перегенерирован" in result.output

    def test_pipeline_error(self, runner, mock_cfg, tmp_session):
        from meeting_recorder.pipeline import PipelineError
        with patch("meeting_recorder.__main__.run_report_only", side_effect=PipelineError("no tr")):
            result = _invoke(runner, ["report", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI: generate-config
# ---------------------------------------------------------------------------

class TestGenerateConfigCmd:
    def test_generates_file(self, runner, tmp_path):
        out = str(tmp_path / "out.yaml")
        result = runner.invoke(cli, ["generate-config", "-o", out], catch_exceptions=False)
        assert result.exit_code == 0
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# CLI: mux
# ---------------------------------------------------------------------------

class TestMuxCmd:
    def test_no_sessions(self, runner, mock_cfg):
        result = _invoke(runner, ["mux"], mock_cfg)
        assert result.exit_code == 1
        assert "не найдено" in result.output

    def test_missing_video(self, runner, mock_cfg, tmp_session):
        result = _invoke(runner, ["mux", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 1
        assert "Видеофайл" in result.output

    def test_success(self, runner, mock_cfg, tmp_session):
        tmp_session.video.write_bytes(b"")
        tmp_session.mix_audio.write_bytes(b"")

        fake_out = tmp_session.final_video
        fake_out.write_bytes(b"x" * 1024)  # 1 KiB → 0.00 МБ

        with patch("meeting_recorder.recorder.mux_video", return_value=fake_out):
            result = _invoke(runner, ["mux", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 0
        assert "Готово" in result.output


# ---------------------------------------------------------------------------
# CLI: stop (без реальной записи — тестируем ветку "нет активной сессии")
# ---------------------------------------------------------------------------

class TestStopCmd:
    def test_no_active_session(self, runner, mock_cfg):
        result = _invoke(runner, ["stop"], mock_cfg)
        assert result.exit_code == 0
        # выходим без ошибки (нечего останавливать)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_verbose(self):
        setup_logging(verbose=True)  # не должно падать

    def test_normal(self):
        setup_logging(verbose=False)  # не должно падать


# ---------------------------------------------------------------------------
# _stop_ffmpeg_graceful
# ---------------------------------------------------------------------------

class TestStopFfmpegGraceful:
    def test_process_already_gone(self):
        from meeting_recorder.__main__ import _stop_ffmpeg_graceful
        with patch("os.kill", side_effect=ProcessLookupError):
            result = _stop_ffmpeg_graceful(99999)
        assert result is True

    def test_graceful_stop_detected(self):
        from meeting_recorder.__main__ import _stop_ffmpeg_graceful
        with patch("os.kill"):
            with patch("subprocess.run") as mock_run:
                # Первый вызов: pid отсутствует → завершился
                mock_run.return_value = MagicMock(returncode=0, stdout="No tasks running")
                result = _stop_ffmpeg_graceful(12345, timeout=1)
        assert result is True

    def test_forcekill_on_timeout(self):
        from meeting_recorder.__main__ import _stop_ffmpeg_graceful
        with patch("os.kill"):
            with patch("subprocess.run") as mock_run:
                # Процесс всегда виден → таймаут → force kill
                mock_run.return_value = MagicMock(returncode=0, stdout="12345 ffmpeg.exe")
                result = _stop_ffmpeg_graceful(12345, timeout=0)
        # После force kill — результат False (не дождались)
        assert result is False


# ---------------------------------------------------------------------------
# _signal_stop_and_wait
# ---------------------------------------------------------------------------

class TestSignalStopAndWait:
    def test_done_file_appears(self, tmp_path):
        from meeting_recorder.__main__ import _signal_stop_and_wait, _DONE_FILE, _STOP_FILE

        def _write_done(*a, **kw):
            import time; time.sleep(0.1)
            _DONE_FILE.parent.mkdir(exist_ok=True)
            _DONE_FILE.touch()

        t = threading.Thread(target=_write_done, daemon=True)
        t.start()
        _signal_stop_and_wait(None)
        t.join(timeout=2)
        # без ошибок

    def test_timeout_triggers_force_kill(self):
        from meeting_recorder.__main__ import _signal_stop_and_wait
        with patch("meeting_recorder.__main__._stop_ffmpeg_graceful") as mock_kill:
            with patch("meeting_recorder.__main__.time.sleep"):
                with patch("meeting_recorder.__main__._DONE_FILE") as mock_done:
                    mock_done.exists.return_value = False
                    # deadline already passed
                    with patch("meeting_recorder.__main__.time.monotonic", side_effect=[0, 31, 31]):
                        _signal_stop_and_wait(12345)
        mock_kill.assert_called_once_with(12345)


# ---------------------------------------------------------------------------
# CLI: stop-only command
# ---------------------------------------------------------------------------

class TestStopOnlyCmd:
    def test_no_active_session(self, runner, mock_cfg):
        result = _invoke(runner, ["stop-only"], mock_cfg)
        assert result.exit_code == 0

    def test_with_active_session_no_audio(self, runner, mock_cfg, tmp_session):
        _save_state(tmp_session.session_id, str(tmp_session.video), 12345)
        with patch("meeting_recorder.__main__._signal_stop_and_wait"):
            result = _invoke(runner, ["stop-only"], mock_cfg)
        assert "Аудиофайлы не созданы" in result.output


# ---------------------------------------------------------------------------
# CLI: chat command (мок LLM и пустой сессии)
# ---------------------------------------------------------------------------

class TestChatCmd:
    def test_no_sessions_exits(self, runner, mock_cfg):
        result = _invoke(runner, ["chat"], mock_cfg)
        assert result.exit_code == 1
        assert "не найдено" in result.output

    def test_session_no_data(self, runner, mock_cfg, tmp_session):
        result = _invoke(runner, ["chat", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 1
        assert "Нет данных" in result.output

    def test_llm_unavailable(self, runner, mock_cfg, tmp_session):
        tmp_session.summary.write_text("# Summary", encoding="utf-8")
        from meeting_recorder.llm_client import LLMClientError
        with patch("meeting_recorder.llm_client.create_llm_client",
                   side_effect=LLMClientError("down")):
            result = _invoke(runner, ["chat", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 1
        assert "недоступен" in result.output

    def test_nonexistent_session(self, runner, mock_cfg):
        result = _invoke(runner, ["chat", "meeting_1900-01-01_00-00-00"], mock_cfg)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI: tray (только проверка ImportError ветки)
# ---------------------------------------------------------------------------

class TestTrayCmd:
    def test_pystray_not_installed(self, runner, mock_cfg):
        with patch.dict("sys.modules", {"pystray": None}):
            result = _invoke(runner, ["tray"], mock_cfg)
        assert result.exit_code == 1
        assert "pystray" in result.output
