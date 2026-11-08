"""Тесты CLI-команд из __main__.py через click.testing.CliRunner."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from meeting_recorder.__main__ import (
    _DONE_FILE,
    _STATE_FILE,
    _STOP_FILE,
    _build_chat_system_prompt,
    _load_active_session,
    _load_state,
    _mix_session_audio,
    _remove_state,
    _rename_with_retry,
    _save_state,
    cli,
    setup_logging,
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
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
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
        with patch(
            "meeting_recorder.pipeline.Path.replace",
            side_effect=PermissionError("locked"),
        ):
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

        with patch("meeting_recorder.pipeline.mix_audio_files") as mock_mix:
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
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
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
        tmp_session.transcript.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        prompt = _build_chat_system_prompt(tmp_session, mock_cfg)
        assert "Привет" in prompt
        assert tmp_session.session_id in prompt

    def test_with_summary(self, tmp_session, mock_cfg):
        tmp_session.summary.write_text(
            "# Summary\nКраткое содержание", encoding="utf-8"
        )
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

        with patch(
            "meeting_recorder.__main__.run_process", side_effect=PipelineError("bad")
        ):
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

        with patch(
            "meeting_recorder.__main__.run_report_only",
            side_effect=PipelineError("no tr"),
        ):
            result = _invoke(runner, ["report", tmp_session.session_id], mock_cfg)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI: generate-config
# ---------------------------------------------------------------------------


class TestGenerateConfigCmd:
    def test_generates_file(self, runner, tmp_path):
        out = str(tmp_path / "out.yaml")
        result = runner.invoke(
            cli, ["generate-config", "-o", out], catch_exceptions=False
        )
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

        with patch("meeting_recorder.pipeline.mux_video", return_value=fake_out):
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
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="No tasks running"
                )
                result = _stop_ffmpeg_graceful(12345, timeout=1)
        assert result is True

    def test_forcekill_on_timeout(self):
        from meeting_recorder.__main__ import _stop_ffmpeg_graceful

        with patch("os.kill"):
            with patch("subprocess.run") as mock_run:
                # Процесс всегда виден → таймаут → force kill
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="12345 ffmpeg.exe"
                )
                result = _stop_ffmpeg_graceful(12345, timeout=0)
        # После force kill — результат False (не дождались)
        assert result is False


# ---------------------------------------------------------------------------
# _signal_stop_and_wait
# ---------------------------------------------------------------------------


class TestSignalStopAndWait:
    def test_done_file_appears(self, tmp_path):
        from meeting_recorder.__main__ import (
            _DONE_FILE,
            _signal_stop_and_wait,
        )

        def _write_done(*a, **kw):
            import time

            time.sleep(0.1)
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
                    with patch(
                        "meeting_recorder.__main__.time.monotonic",
                        side_effect=[0, 31, 31],
                    ):
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

        with patch(
            "meeting_recorder.llm_client.create_llm_client",
            side_effect=LLMClientError("down"),
        ):
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


# ---------------------------------------------------------------------------
# Lock-файл `mrec start` и устаревшее состояние
# ---------------------------------------------------------------------------

import meeting_recorder.__main__ as main_mod  # noqa: E402


@pytest.fixture
def held_lock():
    fh = main_mod._acquire_start_lock()
    assert fh is not None
    yield fh
    main_mod._release_start_lock(fh)


class TestStartLock:
    def test_free_lock_not_alive(self):
        assert main_mod._is_recorder_alive() is False

    def test_held_lock_alive(self, held_lock):
        assert main_mod._is_recorder_alive() is True
        assert main_mod._acquire_start_lock() is None

    def test_release_frees(self):
        fh = main_mod._acquire_start_lock()
        main_mod._release_start_lock(fh)
        assert main_mod._is_recorder_alive() is False


class TestStartCommand:
    def test_refuses_when_recording_alive(self, runner, mock_cfg, held_lock):
        _save_state("meeting_2026-01-01_10-00-00", "v.mp4", 1)
        with patch("meeting_recorder.recorder.MeetingRecorder") as mock_rec:
            result = _invoke(runner, ["start"], mock_cfg)
        assert "Уже идёт запись" in result.output
        mock_rec.assert_not_called()

    def test_stale_state_cleared_and_start_proceeds(self, runner, mock_cfg):
        _save_state("meeting_2026-01-01_10-00-00", "v.mp4", 1)
        rec = MagicMock()
        rec.start.side_effect = RuntimeError("no device")
        rec.is_recording = False
        with patch("meeting_recorder.recorder.MeetingRecorder", return_value=rec):
            result = _invoke(runner, ["start"], mock_cfg)
        assert "прервана аварийно" in result.output
        assert "Ошибка запуска записи" in result.output
        assert _load_state() is None
        assert main_mod._is_recorder_alive() is False  # lock освобождён

    def test_start_and_stop_signal(self, runner, mock_cfg):
        rec = MagicMock()
        rec.ffmpeg_pid = 4242
        rec.is_recording = True

        def _start():
            _STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STOP_FILE.touch()  # сразу просим остановиться

        rec.start.side_effect = _start
        with patch("meeting_recorder.recorder.MeetingRecorder", return_value=rec):
            result = _invoke(runner, ["start"], mock_cfg)
        assert "Запись начата" in result.output
        rec.stop.assert_called_once()
        assert _DONE_FILE.exists()
        assert _load_state() is None


class TestSignalStopAlive:
    def test_waits_for_done_when_alive(self, held_lock):
        def _write_done():
            import time as _t

            _t.sleep(0.2)
            _DONE_FILE.touch()

        t = threading.Thread(target=_write_done, daemon=True)
        t.start()
        with patch("meeting_recorder.__main__._stop_ffmpeg_graceful") as kill:
            main_mod._signal_stop_and_wait(1234)
        t.join(timeout=2)
        kill.assert_not_called()
        assert not _STOP_FILE.exists()

    def test_timeout_when_alive_kills(self, held_lock):
        with patch("meeting_recorder.__main__._stop_ffmpeg_graceful") as kill:
            with patch("meeting_recorder.__main__.time.sleep"):
                with patch(
                    "meeting_recorder.__main__.time.monotonic", side_effect=[0, 31]
                ):
                    main_mod._signal_stop_and_wait(1234)
        kill.assert_called_once_with(1234)

    def test_stale_without_pid_no_kill(self):
        with patch("meeting_recorder.__main__._stop_ffmpeg_graceful") as kill:
            main_mod._signal_stop_and_wait(None)
        kill.assert_not_called()


class TestFfmpegPidCheck:
    def test_is_ffmpeg_true(self):
        out = (
            '"ffmpeg.exe","4242","Console","1","100 K"'
            if sys.platform == "win32"
            else "ffmpeg"
        )
        with patch("subprocess.run", return_value=MagicMock(stdout=out)):
            assert main_mod._is_ffmpeg_process(4242) is True

    def test_is_ffmpeg_false_for_other_process(self):
        out = (
            '"chrome.exe","4242","Console","1","100 K"'
            if sys.platform == "win32"
            else "chrome"
        )
        with patch("subprocess.run", return_value=MagicMock(stdout=out)):
            assert main_mod._is_ffmpeg_process(4242) is False

    def test_is_ffmpeg_error(self):
        with patch("subprocess.run", side_effect=OSError("no tasklist")):
            assert main_mod._is_ffmpeg_process(4242) is False

    def test_reused_pid_not_killed(self):
        """PID из устаревшего state-файла занят другим процессом — не трогаем его."""
        with patch("meeting_recorder.__main__._is_ffmpeg_process", return_value=False):
            with patch("os.kill") as kill:
                assert main_mod._stop_ffmpeg_graceful(4242) is True
        kill.assert_not_called()


class TestStopCmdFlow:
    def test_stale_session_processed(self, runner, mock_cfg, tmp_session):
        _save_state(tmp_session.session_id, str(tmp_session.video), None)
        tmp_session.mic_audio.write_bytes(b"mic")
        with patch(
            "meeting_recorder.__main__.process_session",
            return_value=({}, tmp_session.protocol, tmp_session.summary),
        ) as ps:
            result = _invoke(runner, ["stop"], mock_cfg)
        ps.assert_called_once()
        assert "Готово" in result.output
        assert tmp_session.mix_audio.exists()

    def test_pipeline_error_reported(self, runner, mock_cfg, tmp_session):
        _save_state(tmp_session.session_id, str(tmp_session.video), None)
        tmp_session.mic_audio.write_bytes(b"mic")
        with patch(
            "meeting_recorder.__main__.process_session",
            side_effect=RuntimeError("LLM down"),
        ):
            result = _invoke(runner, ["stop"], mock_cfg)
        assert "LLM down" in result.output
        assert f"mrec process {tmp_session.session_id}" in result.output

    def test_no_audio_prints_ffmpeg_log(self, runner, mock_cfg, tmp_session):
        _save_state(tmp_session.session_id, str(tmp_session.video), None)
        tmp_session.ffmpeg_log.write_text(
            "Could not find audio device", encoding="utf-8"
        )
        result = _invoke(runner, ["stop"], mock_cfg)
        assert "Could not find audio device" in result.output


# ---------------------------------------------------------------------------
# Чат: бюджет контекста
# ---------------------------------------------------------------------------


class TestChatBudget:
    def test_long_transcript_truncated_with_note(self, tmp_session):
        cfg = AppConfig.model_validate(
            {"llm": {"context_tokens": 4096, "max_tokens": 1024}}
        )
        segments = [
            {
                "start": float(i),
                "end": float(i + 1),
                "speaker": "SPEAKER_00",
                "text": "слово " * 30,
            }
            for i in range(500)
        ]
        tmp_session.transcript.write_text(
            json.dumps({"segments": segments, "duration_sec": 500}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_session.summary.write_text("# Итог встречи", encoding="utf-8")
        prompt = _build_chat_system_prompt(tmp_session, cfg)
        from meeting_recorder.report import estimate_tokens

        assert estimate_tokens(prompt) <= cfg.llm.input_token_budget()
        assert "Транскрипт обрезан" in prompt
        assert "Итог встречи" in prompt

    def test_speaker_names_applied(self, tmp_session, mock_cfg):
        mock_cfg.transcription.speaker_names = {"SPEAKER_00": "Иван"}
        tmp_session.transcript.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 0,
                            "end": 1,
                            "speaker": "SPEAKER_00",
                            "text": "Привет",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        prompt = _build_chat_system_prompt(tmp_session, mock_cfg)
        assert "Иван: Привет" in prompt
        assert "Участники: Иван" in prompt

    def test_protocol_used_without_transcript(self, tmp_session, mock_cfg):
        tmp_session.protocol.write_text("ПРОТОКОЛ", encoding="utf-8")
        assert "ПРОТОКОЛ" in _build_chat_system_prompt(tmp_session, mock_cfg)

    def test_trim_history_drops_oldest(self):
        history = [{"role": "system", "content": "s"}]
        for i in range(10):
            history.append({"role": "user", "content": f"q{i} " + "x" * 300})
            history.append({"role": "assistant", "content": "a" * 300})
        main_mod._trim_history(history, budget_tokens=500)
        assert history[0]["content"] == "s"
        assert history[-1]["role"] == "assistant"
        assert len(history) < 21

    def test_trim_history_keeps_system_and_last(self):
        history = [
            {"role": "system", "content": "s" * 3000},
            {"role": "user", "content": "q"},
        ]
        main_mod._trim_history(history, budget_tokens=10)
        assert len(history) == 2


# ---------------------------------------------------------------------------
# Прочие команды
# ---------------------------------------------------------------------------


class TestMiscCommands:
    def test_generate_config_excludes_env_secrets(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_secret_env")
        out = tmp_path / "c.yaml"
        result = runner.invoke(
            cli, ["generate-config", "-o", str(out)], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "hf_secret_env" not in out.read_text(encoding="utf-8")

    def test_tray_gets_config_path(self, runner, mock_cfg, tmp_path):
        pytest.importorskip("pystray")
        cfg_path = tmp_path / "my.yaml"
        with patch("meeting_recorder.tray.TrayApp") as tray_cls:
            with patch("meeting_recorder.__main__.load_config", return_value=mock_cfg):
                runner.invoke(
                    cli, ["-c", str(cfg_path), "tray"], catch_exceptions=False
                )
        assert tray_cls.call_args.kwargs["config_path"] == cfg_path

    @pytest.mark.parametrize("cmd", ["process", "report", "highlights", "html"])
    def test_invalid_session_id_no_traceback(self, runner, mock_cfg, cmd):
        result = _invoke(runner, [cmd, "..\\.."], mock_cfg)
        assert result.exit_code == 1
        assert "Некорректный" in result.output

    def test_log_file_outside_cwd(self):
        assert main_mod._default_log_file().name == "meeting_recorder.log"
        assert main_mod._default_log_file().parent != Path.cwd()

    def test_serve_warns_on_public_host(self, runner, mock_cfg):
        pytest.importorskip("uvicorn")
        with patch("uvicorn.run") as run:
            with patch("meeting_recorder.dashboard.create_app") as create:
                result = _invoke(
                    runner, ["serve", "--host", "0.0.0.0", "--no-open"], mock_cfg
                )
        run.assert_called_once()
        assert "аутентификации нет" in result.output
        assert create.call_args.kwargs["allowed_hosts"] is None

    def test_serve_local_sets_allowed_hosts(self, runner, mock_cfg):
        pytest.importorskip("uvicorn")
        with patch("uvicorn.run"):
            with patch("meeting_recorder.dashboard.create_app") as create:
                _invoke(runner, ["serve", "--no-open"], mock_cfg)
        assert create.call_args.kwargs["allowed_hosts"] == ["127.0.0.1"]
