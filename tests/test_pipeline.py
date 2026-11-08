"""Тесты для pipeline.py — mock transcriber, report, subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths, create_session
from meeting_recorder.pipeline import (
    PipelineError,
    _check_video_length,
    run_highlights_only,
    run_html,
    run_process,
    run_report,
    run_report_only,
    run_transcribe,
    run_transcribe_only,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_session(tmp_path) -> SessionPaths:
    return create_session(str(tmp_path))


@pytest.fixture
def app_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = str(tmp_path)
    return cfg


def _make_wav(path: Path, sr: int = 48000, duration: float = 1.0) -> Path:
    data = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(path), data, sr)
    return path


def _make_transcript(path: Path, session_id: str = "test_session") -> dict:
    data = {
        "session_id": session_id,
        "language": "ru",
        "duration_sec": 60.0,
        "diarization_enabled": False,
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Привет"}
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# ---------------------------------------------------------------------------
# _check_video_length
# ---------------------------------------------------------------------------


class TestCheckVideoLength:
    def test_ffprobe_error_silent(self, tmp_session):
        with patch("subprocess.run", side_effect=Exception("ffprobe not found")):
            _check_video_length(tmp_session)  # не должна бросить исключение

    def test_missing_files_returns_early(self, tmp_session):
        # Если нет видео или аудио — выходим без вызова subprocess
        with patch("subprocess.run") as mock_run:
            _check_video_length(tmp_session)
        mock_run.assert_not_called()

    def test_video_much_shorter_logs_warning(self, tmp_session, caplog):
        import logging

        # Создаём файлы чтобы не вышли раньше времени
        tmp_session.video.write_bytes(b"")
        tmp_session.mic_audio.write_bytes(b"")

        video_info = json.dumps({"format": {"duration": "5.0"}})
        audio_info = json.dumps({"format": {"duration": "60.0"}})

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout=video_info),
                MagicMock(returncode=0, stdout=audio_info),
            ]
            with caplog.at_level(logging.WARNING, logger="meeting_recorder.pipeline"):
                _check_video_length(tmp_session)
        assert any(
            "короче" in r.message or "видео" in r.message.lower()
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# run_transcribe
# ---------------------------------------------------------------------------


class TestRunTranscribe:
    def test_success(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mix_audio)
        transcript_data = {
            "session_id": "",
            "language": "ru",
            "duration_sec": 1.0,
            "diarization_enabled": False,
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "S", "text": "hi"}],
        }
        with patch(
            "meeting_recorder.pipeline.transcribe", return_value=transcript_data
        ):
            result = run_transcribe(app_cfg, tmp_session)

        assert result["session_id"] == tmp_session.session_id
        assert tmp_session.transcript.exists()
        saved = json.loads(tmp_session.transcript.read_text())
        assert saved["session_id"] == tmp_session.session_id

    def test_missing_mix_audio_raises(self, app_cfg, tmp_session):
        with pytest.raises(PipelineError):
            run_transcribe(app_cfg, tmp_session)

    def test_transcribe_exception_wrapped(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mix_audio)
        with patch(
            "meeting_recorder.pipeline.transcribe",
            side_effect=RuntimeError("model error"),
        ):
            with pytest.raises(Exception):
                run_transcribe(app_cfg, tmp_session)


# ---------------------------------------------------------------------------
# run_report
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_success(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)

        with patch(
            "meeting_recorder.pipeline.generate_protocol",
            return_value=tmp_session.protocol,
        ) as m_proto:
            with patch(
                "meeting_recorder.pipeline.generate_summary",
                return_value=tmp_session.summary,
            ) as m_summ:
                proto, summ = run_report(app_cfg, tmp_session)

        assert proto == tmp_session.protocol
        assert summ == tmp_session.summary
        m_proto.assert_called_once()
        m_summ.assert_called_once()

    def test_missing_transcript_raises(self, app_cfg, tmp_session):
        with pytest.raises(PipelineError):
            run_report(app_cfg, tmp_session)

    def test_report_exception_propagates(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)
        with patch(
            "meeting_recorder.pipeline.generate_protocol",
            side_effect=RuntimeError("llm error"),
        ):
            with pytest.raises(Exception):
                run_report(app_cfg, tmp_session)


# ---------------------------------------------------------------------------
# run_transcribe_only
# ---------------------------------------------------------------------------


class TestRunTranscribeOnly:
    def test_creates_transcript(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mix_audio)
        data = {
            "session_id": "",
            "language": "ru",
            "duration_sec": 1.0,
            "diarization_enabled": False,
            "segments": [],
        }

        with patch("meeting_recorder.pipeline.transcribe", return_value=data):
            result = run_transcribe_only(app_cfg, tmp_session.session_id)

        assert result["session_id"] == tmp_session.session_id

    def test_nonexistent_session_raises(self, app_cfg):
        with pytest.raises(FileNotFoundError):
            run_transcribe_only(app_cfg, "meeting_2000-01-01_00-00-00")


# ---------------------------------------------------------------------------
# run_report_only
# ---------------------------------------------------------------------------


class TestRunReportOnly:
    def test_creates_reports(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)

        with patch(
            "meeting_recorder.pipeline.generate_protocol",
            return_value=tmp_session.protocol,
        ):
            with patch(
                "meeting_recorder.pipeline.generate_summary",
                return_value=tmp_session.summary,
            ):
                proto, summ = run_report_only(app_cfg, tmp_session.session_id)

        assert proto == tmp_session.protocol
        assert summ == tmp_session.summary

    def test_nonexistent_session_raises(self, app_cfg):
        with pytest.raises(FileNotFoundError):
            run_report_only(app_cfg, "meeting_2000-01-01_00-00-00")


# ---------------------------------------------------------------------------
# run_process
# ---------------------------------------------------------------------------


class TestRunReportHtmlNonBlocking:
    """HTML-генерация в run_report не должна блокировать pipeline при ошибке."""

    def test_html_failure_does_not_raise(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)
        with patch(
            "meeting_recorder.pipeline.generate_protocol",
            return_value=tmp_session.protocol,
        ):
            with patch(
                "meeting_recorder.pipeline.generate_summary",
                return_value=tmp_session.summary,
            ):
                with patch(
                    "meeting_recorder.pipeline.generate_html_protocol",
                    side_effect=RuntimeError("html error"),
                ):
                    proto, summ = run_report(app_cfg, tmp_session)
        assert proto == tmp_session.protocol
        assert summ == tmp_session.summary

    def test_html_called_after_report(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)
        with patch(
            "meeting_recorder.pipeline.generate_protocol",
            return_value=tmp_session.protocol,
        ):
            with patch(
                "meeting_recorder.pipeline.generate_summary",
                return_value=tmp_session.summary,
            ):
                with patch(
                    "meeting_recorder.pipeline.generate_html_protocol",
                    return_value=tmp_session.html_protocol,
                ) as mock_html:
                    run_report(app_cfg, tmp_session)
        mock_html.assert_called_once()


# ---------------------------------------------------------------------------
# run_html
# ---------------------------------------------------------------------------


class TestRunHtml:
    def test_success(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)
        with patch(
            "meeting_recorder.pipeline.generate_html_protocol",
            return_value=tmp_session.html_protocol,
        ) as mock_html:
            result = run_html(app_cfg, tmp_session.session_id)
        mock_html.assert_called_once()
        assert result == tmp_session.html_protocol

    def test_missing_transcript_raises(self, app_cfg, tmp_session):
        with pytest.raises(PipelineError, match="Транскрипт"):
            run_html(app_cfg, tmp_session.session_id)

    def test_nonexistent_session_raises(self, app_cfg):
        with pytest.raises(FileNotFoundError):
            run_html(app_cfg, "meeting_2000-01-01_00-00-00")


# ---------------------------------------------------------------------------
# run_highlights_only
# ---------------------------------------------------------------------------


class TestRunHighlightsOnly:
    def test_success(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)

        with patch(
            "meeting_recorder.pipeline.generate_highlights",
            return_value=tmp_session.highlights,
        ) as mock_hl:
            result = run_highlights_only(app_cfg, tmp_session.session_id)

        mock_hl.assert_called_once()
        assert result == tmp_session.highlights

    def test_missing_transcript_raises(self, app_cfg, tmp_session):
        with pytest.raises(PipelineError, match="Транскрипт"):
            run_highlights_only(app_cfg, tmp_session.session_id)

    def test_nonexistent_session_raises(self, app_cfg):
        with pytest.raises(FileNotFoundError):
            run_highlights_only(app_cfg, "meeting_2000-01-01_00-00-00")


class TestRunProcess:
    def test_runs_transcribe_and_report(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mix_audio)
        data = {
            "session_id": "",
            "language": "ru",
            "duration_sec": 1.0,
            "diarization_enabled": False,
            "segments": [],
        }

        with patch("meeting_recorder.pipeline.transcribe", return_value=data):
            with patch(
                "meeting_recorder.pipeline.generate_protocol",
                return_value=tmp_session.protocol,
            ):
                with patch(
                    "meeting_recorder.pipeline.generate_summary",
                    return_value=tmp_session.summary,
                ):
                    transcript, proto, summ = run_process(
                        app_cfg, tmp_session.session_id
                    )

        assert transcript["session_id"] == tmp_session.session_id
        assert proto == tmp_session.protocol
        assert summ == tmp_session.summary


# ---------------------------------------------------------------------------
# finalize_recording / run_mux / process_session
# ---------------------------------------------------------------------------

from meeting_recorder.pipeline import (  # noqa: E402
    finalize_recording,
    process_session,
    release_models,
    run_mux,
)


class TestFinalizeRecording:
    def test_mix_both(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mic_audio)
        _make_wav(tmp_session.system_audio)
        assert finalize_recording(app_cfg, tmp_session) == tmp_session.mix_audio
        assert tmp_session.mix_audio.exists()

    def test_only_mic(self, app_cfg, tmp_session):
        tmp_session.mic_audio.write_bytes(b"mic")
        finalize_recording(app_cfg, tmp_session)
        assert tmp_session.mix_audio.read_bytes() == b"mic"
        assert not tmp_session.mic_audio.exists()

    def test_only_system(self, app_cfg, tmp_session):
        tmp_session.system_audio.write_bytes(b"sys")
        finalize_recording(app_cfg, tmp_session)
        assert tmp_session.mix_audio.read_bytes() == b"sys"

    def test_mic_replaces_stale_mix(self, app_cfg, tmp_session):
        tmp_session.mix_audio.write_bytes(b"old")
        tmp_session.mic_audio.write_bytes(b"new")
        finalize_recording(app_cfg, tmp_session)
        assert tmp_session.mix_audio.read_bytes() == b"new"

    def test_no_audio_raises(self, app_cfg, tmp_session):
        with pytest.raises(PipelineError, match="не найдены"):
            finalize_recording(app_cfg, tmp_session)

    def test_mix_error_wrapped(self, app_cfg, tmp_session):
        tmp_session.mic_audio.write_bytes(b"bad")
        tmp_session.system_audio.write_bytes(b"bad")
        with pytest.raises(PipelineError, match="свести"):
            finalize_recording(app_cfg, tmp_session)


class TestRunMux:
    def test_nothing_to_mux(self, app_cfg, tmp_session):
        assert run_mux(app_cfg, tmp_session) is None

    def test_existing_final_kept(self, app_cfg, tmp_session):
        tmp_session.video.write_bytes(b"v")
        tmp_session.mix_audio.write_bytes(b"a")
        tmp_session.final_video.write_bytes(b"f")
        with patch("meeting_recorder.pipeline.mux_video") as m:
            assert run_mux(app_cfg, tmp_session) == tmp_session.final_video
        m.assert_not_called()

    def test_force(self, app_cfg, tmp_session):
        tmp_session.video.write_bytes(b"v")
        tmp_session.mix_audio.write_bytes(b"a")
        tmp_session.final_video.write_bytes(b"f")
        with patch(
            "meeting_recorder.pipeline.mux_video", return_value=tmp_session.final_video
        ) as m:
            run_mux(app_cfg, tmp_session, force=True)
        m.assert_called_once()


class TestRunReportMux:
    def _run(self, app_cfg, tmp_session):
        _make_transcript(tmp_session.transcript, tmp_session.session_id)
        tmp_session.video.write_bytes(b"v")
        tmp_session.mix_audio.write_bytes(b"a")
        with patch(
            "meeting_recorder.pipeline.generate_protocol",
            return_value=tmp_session.protocol,
        ):
            with patch(
                "meeting_recorder.pipeline.generate_summary",
                return_value=tmp_session.summary,
            ):
                with patch("meeting_recorder.pipeline.generate_highlights"):
                    with patch("meeting_recorder.pipeline.generate_html_protocol"):
                        with patch("meeting_recorder.pipeline.mux_video") as m:
                            run_report(app_cfg, tmp_session)
        return m

    def test_auto_mux_enabled(self, app_cfg, tmp_session):
        self._run(app_cfg, tmp_session).assert_called_once()

    def test_auto_mux_disabled(self, app_cfg, tmp_session):
        app_cfg.recording.auto_mux = False
        self._run(app_cfg, tmp_session).assert_not_called()


class TestProcessSession:
    def test_releases_models_by_default(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mix_audio)
        data = {"session_id": tmp_session.session_id, "segments": []}
        with patch("meeting_recorder.pipeline.transcribe", return_value=data):
            with patch("meeting_recorder.pipeline.run_report", return_value=(1, 2)):
                with patch("meeting_recorder.pipeline.unload_models") as unload:
                    assert process_session(app_cfg, tmp_session) == (data, 1, 2)
        unload.assert_called_once()

    def test_releases_models_on_error(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mix_audio)
        with patch(
            "meeting_recorder.pipeline.transcribe", side_effect=RuntimeError("oom")
        ):
            with patch("meeting_recorder.pipeline.unload_models") as unload:
                with pytest.raises(RuntimeError):
                    process_session(app_cfg, tmp_session)
        unload.assert_called_once()

    def test_keep_models_loaded(self, app_cfg):
        app_cfg.transcription.keep_models_loaded = True
        with patch("meeting_recorder.pipeline.unload_models") as unload:
            release_models(app_cfg)
        unload.assert_not_called()

    def test_session_id_passed_to_transcribe(self, app_cfg, tmp_session):
        _make_wav(tmp_session.mix_audio)
        data = {"session_id": tmp_session.session_id, "segments": []}
        with patch("meeting_recorder.pipeline.transcribe", return_value=data) as t:
            run_transcribe(app_cfg, tmp_session)
        assert t.call_args.kwargs["session_id"] == tmp_session.session_id
