"""Тесты для recorder.py — mock subprocess и soundfile."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest
import soundfile as sf

from meeting_recorder.config import AppConfig, RecordingConfig
from meeting_recorder.naming import SessionPaths
from meeting_recorder.recorder import (
    SystemAudioCapture,
    MeetingRecorder,
    mix_audio_files,
    mux_video,
    split_streams,
    _FFMPEG_GRACEFUL_STOP_TIMEOUT,
    _FFMPEG_FORCE_KILL_TIMEOUT,
    _SYS_AUDIO_JOIN_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_session(tmp_path) -> SessionPaths:
    sid = "meeting_2024-01-01_10-00-00"
    d = tmp_path / sid
    d.mkdir()
    return SessionPaths(sid, d)


@pytest.fixture
def app_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = str(tmp_path)
    return cfg


def _make_wav(path: Path, duration_sec: float = 1.0, sr: int = 48000) -> Path:
    """Создать тестовый WAV-файл."""
    frames = int(sr * duration_sec)
    data = np.zeros((frames, 1), dtype=np.float32)
    sf.write(str(path), data, sr)
    return path


# ---------------------------------------------------------------------------
# SystemAudioCapture
# ---------------------------------------------------------------------------

class TestSystemAudioCapture:
    def test_chunk_frames_from_sample_rate(self):
        cap = SystemAudioCapture(Path("/tmp/out.wav"), sample_rate=48000)
        assert cap._chunk_frames == 4800  # 100ms @ 48000

    def test_chunk_frames_16k(self):
        cap = SystemAudioCapture(Path("/tmp/out.wav"), sample_rate=16000)
        assert cap._chunk_frames == 1600  # 100ms @ 16000

    def test_signal_stop_sets_event(self):
        cap = SystemAudioCapture(Path("/tmp/out.wav"))
        assert not cap._stop_event.is_set()
        cap.signal_stop()
        assert cap._stop_event.is_set()

    def test_start_creates_thread(self):
        cap = SystemAudioCapture(Path("/tmp/out.wav"))
        started = threading.Event()
        finished = threading.Event()

        def _slow_record():
            started.set()
            finished.wait(timeout=2)

        with patch.object(cap, "_record", side_effect=_slow_record):
            cap.start()
            started.wait(timeout=1)
            assert cap._thread is not None
            assert cap._thread.is_alive()
            finished.set()
            cap.signal_stop()

    def test_wait_joins_thread(self):
        cap = SystemAudioCapture(Path("/tmp/out.wav"))
        finished = threading.Event()

        def _fake_record():
            time.sleep(0.05)
            finished.set()

        cap._thread = threading.Thread(target=_fake_record, daemon=True)
        cap._thread.start()
        cap.wait(timeout=2.0)
        assert finished.is_set()

    def test_stop_calls_signal_and_wait(self):
        cap = SystemAudioCapture(Path("/tmp/out.wav"))
        cap.signal_stop = MagicMock()
        cap.wait = MagicMock()
        cap.stop()
        cap.signal_stop.assert_called_once()
        cap.wait.assert_called_once()

    def test_record_error_stored(self, tmp_path):
        cap = SystemAudioCapture(tmp_path / "out.wav")
        with patch("meeting_recorder.recorder.sf.write", side_effect=IOError("disk full")):
            with patch("soundcard.default_speaker", side_effect=RuntimeError("no device")):
                cap._record()
        assert cap.error is not None
        assert "no device" in str(cap.error)


# ---------------------------------------------------------------------------
# mix_audio_files
# ---------------------------------------------------------------------------

class TestMixAudioFiles:
    def test_equal_length(self, tmp_path):
        mic = _make_wav(tmp_path / "mic.wav", 1.0)
        sys_ = _make_wav(tmp_path / "sys.wav", 1.0)
        out = tmp_path / "mix.wav"
        result = mix_audio_files(mic, sys_, out)
        assert result == out
        assert out.exists()
        data, sr = sf.read(str(out))
        assert sr == 48000
        assert len(data) > 0

    def test_system_longer_trimmed(self, tmp_path):
        """system_audio длиннее → обрезать с начала system."""
        mic = _make_wav(tmp_path / "mic.wav", 1.0)
        sys_ = _make_wav(tmp_path / "sys.wav", 2.0)  # в 2 раза длиннее
        out = tmp_path / "mix.wav"
        mix_audio_files(mic, sys_, out)
        data_out, _ = sf.read(str(out))
        data_mic, _ = sf.read(str(mic))
        assert len(data_out) == len(data_mic)

    def test_mic_longer_trimmed(self, tmp_path):
        """mic длиннее → обрезать с начала mic."""
        mic = _make_wav(tmp_path / "mic.wav", 2.0)
        sys_ = _make_wav(tmp_path / "sys.wav", 1.0)
        out = tmp_path / "mix.wav"
        mix_audio_files(mic, sys_, out)
        data_out, _ = sf.read(str(out))
        data_sys, _ = sf.read(str(sys_))
        assert len(data_out) == len(data_sys)

    def test_empty_file_raises(self, tmp_path):
        mic = tmp_path / "mic.wav"
        sys_ = tmp_path / "sys.wav"
        # Пустые файлы
        sf.write(str(mic), np.zeros((0, 1), dtype=np.float32), 48000)
        sf.write(str(sys_), np.zeros((0, 1), dtype=np.float32), 48000)
        with pytest.raises(ValueError, match="пустой"):
            mix_audio_files(mic, sys_, tmp_path / "mix.wav")


# ---------------------------------------------------------------------------
# MeetingRecorder
# ---------------------------------------------------------------------------

def _make_mock_process(poll_return=None, pid=12345):
    proc = MagicMock()
    proc.poll.return_value = poll_return
    proc.pid = pid
    proc.stdin = MagicMock()
    proc.returncode = poll_return if poll_return is not None else 0
    return proc


class TestMeetingRecorderProperties:
    def test_not_recording_initially(self, app_cfg, tmp_session):
        r = MeetingRecorder(app_cfg, tmp_session)
        assert not r.is_recording
        assert r.ffmpeg_pid is None
        assert r.duration == 0.0

    def test_get_status(self, app_cfg, tmp_session):
        r = MeetingRecorder(app_cfg, tmp_session)
        s = r.get_status()
        assert s["session_id"] == tmp_session.session_id
        assert s["recording"] is False


class TestMeetingRecorderStart:
    def _start_with_mock(self, recorder: MeetingRecorder, poll_return=None):
        proc = _make_mock_process(poll_return)
        with patch("meeting_recorder.recorder.subprocess.Popen", return_value=proc):
            with patch("meeting_recorder.recorder._FFMPEG_STARTUP_TIMEOUT", 0.0):
                recorder.start()
        return proc

    def test_start_success(self, app_cfg, tmp_session):
        app_cfg.recording.record_system_audio = False
        r = MeetingRecorder(app_cfg, tmp_session)
        proc = self._start_with_mock(r)
        assert r.is_recording
        assert r.ffmpeg_pid == proc.pid

    def test_start_twice_raises(self, app_cfg, tmp_session):
        app_cfg.recording.record_system_audio = False
        r = MeetingRecorder(app_cfg, tmp_session)
        self._start_with_mock(r)
        with pytest.raises(RuntimeError, match="уже идёт"):
            r.start()

    def test_duration_while_recording(self, app_cfg, tmp_session):
        app_cfg.recording.record_system_audio = False
        r = MeetingRecorder(app_cfg, tmp_session)
        self._start_with_mock(r)
        time.sleep(0.05)
        assert r.duration > 0.0

    def test_ffmpeg_fails_at_startup_raises(self, app_cfg, tmp_session):
        """Если ffmpeg сразу завершился — RuntimeError."""
        app_cfg.recording.record_system_audio = False
        r = MeetingRecorder(app_cfg, tmp_session)
        proc = _make_mock_process(poll_return=1)  # ffmpeg сразу умер
        with patch("meeting_recorder.recorder.subprocess.Popen", return_value=proc):
            with patch("meeting_recorder.recorder._FFMPEG_STARTUP_TIMEOUT", 0.3):
                with patch("meeting_recorder.recorder._FFMPEG_STARTUP_POLL", 0.1):
                    with pytest.raises(RuntimeError, match="ffmpeg завершился"):
                        r.start()
        assert not r.is_recording

    def test_soundcard_error_at_startup_raises(self, app_cfg, tmp_session):
        """Если soundcard падает при старте — RuntimeError."""
        app_cfg.recording.record_system_audio = True
        app_cfg.recording.system_audio_grabber = "soundcard"
        r = MeetingRecorder(app_cfg, tmp_session)

        mock_capture = MagicMock(spec=SystemAudioCapture)
        mock_capture.error = RuntimeError("no soundcard device")
        proc = _make_mock_process(poll_return=None)

        with patch("meeting_recorder.recorder.subprocess.Popen", return_value=proc):
            with patch("meeting_recorder.recorder.SystemAudioCapture", return_value=mock_capture):
                with patch("meeting_recorder.recorder._FFMPEG_STARTUP_TIMEOUT", 0.3):
                    with patch("meeting_recorder.recorder._FFMPEG_STARTUP_POLL", 0.1):
                        with pytest.raises(RuntimeError, match="системного аудио"):
                            r.start()
        assert not r.is_recording


class TestMeetingRecorderStop:
    def _started_recorder(self, app_cfg, tmp_session) -> tuple[MeetingRecorder, MagicMock]:
        app_cfg.recording.record_system_audio = False
        r = MeetingRecorder(app_cfg, tmp_session)
        proc = _make_mock_process(poll_return=None)
        with patch("meeting_recorder.recorder.subprocess.Popen", return_value=proc):
            with patch("meeting_recorder.recorder._FFMPEG_STARTUP_TIMEOUT", 0.0):
                r.start()
        return r, proc

    def test_stop_sends_q(self, app_cfg, tmp_session):
        r, proc = self._started_recorder(app_cfg, tmp_session)
        r.stop()
        proc.stdin.write.assert_called_with(b"q")
        assert not r.is_recording

    def test_stop_not_recording_is_noop(self, app_cfg, tmp_session):
        r = MeetingRecorder(app_cfg, tmp_session)
        r.stop()  # не должно кидать исключение

    def test_stop_timeout_kills(self, app_cfg, tmp_session):
        r, proc = self._started_recorder(app_cfg, tmp_session)
        proc.wait.side_effect = [
            __import__("subprocess").TimeoutExpired("ffmpeg", _FFMPEG_GRACEFUL_STOP_TIMEOUT),
            None,
        ]
        r.stop()
        proc.terminate.assert_called_once()
        assert not r.is_recording


# ---------------------------------------------------------------------------
# mux_video / split_streams
# ---------------------------------------------------------------------------

class TestMuxVideo:
    def test_mux_success(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.wav"
        out = tmp_path / "final.mp4"
        video.write_bytes(b"")
        audio.write_bytes(b"")

        with patch("meeting_recorder.recorder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = mux_video(video, audio, out)
            assert result == out

    def test_mux_failure_raises(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.wav"
        out = tmp_path / "final.mp4"

        with patch("meeting_recorder.recorder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error details")
            with pytest.raises(RuntimeError, match="ffmpeg завершился"):
                mux_video(video, audio, out)


class TestSplitStreams:
    def test_split_missing_file(self, tmp_path, tmp_session):
        with pytest.raises(FileNotFoundError):
            split_streams(tmp_path / "nonexistent.mp4", tmp_session)

    def test_split_success(self, tmp_path, tmp_session):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        with patch("meeting_recorder.recorder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            mic, sys_ = split_streams(video, tmp_session, sample_rate=16000)
            assert mic == tmp_session.mic_audio
            assert sys_ == tmp_session.system_audio
            # проверяем что sample_rate передался в команду
            calls_args = [str(a) for args in [c.args[0] for c in mock_run.call_args_list] for a in args]
            assert "16000" in calls_args
