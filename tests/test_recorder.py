"""Тесты для recorder.py — mock subprocess и soundfile."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from meeting_recorder.config import AppConfig
from meeting_recorder.naming import SessionPaths
from meeting_recorder.recorder import (
    _FFMPEG_GRACEFUL_STOP_TIMEOUT,
    AudioDevice,
    MeetingRecorder,
    SystemAudioCapture,
    default_soundcard_speaker,
    list_dshow_audio_devices,
    list_soundcard_speakers,
    mix_audio_files,
    mux_video,
    no_window_kwargs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_session(tmp_path) -> SessionPaths:
    paths = SessionPaths(str(tmp_path), "meeting_2024-01-01_10-00-00")
    paths.ensure_dir()
    return paths


@pytest.fixture
def app_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.output_dir = str(tmp_path)
    # Явное устройство: тесты не должны зависеть от реального ffmpeg и микрофонов
    cfg.recording.mic_device = "Test Mic"
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
        with patch(
            "meeting_recorder.recorder.sf.write", side_effect=OSError("disk full")
        ):
            with patch(
                "soundcard.default_speaker", side_effect=RuntimeError("no device")
            ):
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

    def test_streaming_blocks_match_full_mix(self, tmp_path):
        """Блочное сведение даёт тот же результат, что и сведение целиком."""
        rng = np.random.default_rng(0)
        a = (rng.standard_normal(10_000) * 0.1).astype(np.float32)
        b = (rng.standard_normal(12_345) * 0.1).astype(np.float32)
        sf.write(str(tmp_path / "mic.wav"), a, 48000, subtype="PCM_16")
        sf.write(str(tmp_path / "sys.wav"), b, 48000, subtype="PCM_16")
        out = tmp_path / "mix.wav"
        with patch("meeting_recorder.recorder._MIX_BLOCK_FRAMES", 1000):
            mix_audio_files(tmp_path / "mic.wav", tmp_path / "sys.wav", out)
        mixed, _ = sf.read(str(out), dtype="float32")
        a16, _ = sf.read(str(tmp_path / "mic.wav"), dtype="float32")
        b16, _ = sf.read(str(tmp_path / "sys.wav"), dtype="float32")
        expected = (a16 + b16[len(b16) - len(a16) :]) / 2
        assert len(mixed) == len(a16)
        assert np.allclose(mixed, expected, atol=2e-4)
        assert not list(tmp_path.glob(".*part*"))

    def test_stereo_input_uses_first_channel(self, tmp_path):
        stereo = np.zeros((4800, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        stereo[:, 1] = -0.5
        sf.write(str(tmp_path / "mic.wav"), stereo, 48000)
        _make_wav(tmp_path / "sys.wav", 0.1)
        out = tmp_path / "mix.wav"
        mix_audio_files(tmp_path / "mic.wav", tmp_path / "sys.wav", out)
        info = sf.info(str(out))
        assert info.channels == 1

    def test_sample_rate_mismatch_resamples_via_ffmpeg(self, tmp_path):
        mic = tmp_path / "mic.wav"
        sys_ = tmp_path / "sys.wav"
        sf.write(str(mic), np.zeros(4800, dtype=np.float32), 48000)
        sf.write(str(sys_), np.zeros(4410, dtype=np.float32), 44100)

        def fake_resample(src, sr):
            dst = tmp_path / ".sys.48000hz.wav"
            sf.write(str(dst), np.zeros(4800, dtype=np.float32), sr)
            return dst

        with patch(
            "meeting_recorder.recorder._resample_with_ffmpeg", side_effect=fake_resample
        ) as m:
            mix_audio_files(mic, sys_, tmp_path / "mix.wav")
        m.assert_called_once_with(sys_, 48000)
        assert sf.info(str(tmp_path / "mix.wav")).samplerate == 48000
        assert not (tmp_path / ".sys.48000hz.wav").exists()  # временный файл удалён

    def test_resample_with_ffmpeg_failure(self, tmp_path):
        from meeting_recorder.recorder import _resample_with_ffmpeg

        with patch("meeting_recorder.recorder.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr="bad")
            with pytest.raises(RuntimeError, match="пересэмплировать"):
                _resample_with_ffmpeg(tmp_path / "s.wav", 48000)

    def test_resample_with_ffmpeg_success(self, tmp_path):
        from meeting_recorder.recorder import _resample_with_ffmpeg

        with patch(
            "meeting_recorder.recorder.subprocess.run", side_effect=_fake_ffmpeg_ok
        ):
            dst = _resample_with_ffmpeg(tmp_path / "s.wav", 48000)
        assert dst.exists()


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
            with patch(
                "meeting_recorder.recorder.SystemAudioCapture",
                return_value=mock_capture,
            ):
                with patch("meeting_recorder.recorder._FFMPEG_STARTUP_TIMEOUT", 0.3):
                    with patch("meeting_recorder.recorder._FFMPEG_STARTUP_POLL", 0.1):
                        with pytest.raises(RuntimeError, match="системного аудио"):
                            r.start()
        assert not r.is_recording


class TestMeetingRecorderStop:
    def _started_recorder(
        self, app_cfg, tmp_session
    ) -> tuple[MeetingRecorder, MagicMock]:
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
            __import__("subprocess").TimeoutExpired(
                "ffmpeg", _FFMPEG_GRACEFUL_STOP_TIMEOUT
            ),
            None,
        ]
        r.stop()
        proc.terminate.assert_called_once()
        assert not r.is_recording


# ---------------------------------------------------------------------------
# mux_video
# ---------------------------------------------------------------------------


def _fake_ffmpeg_ok(cmd, **kwargs):
    """Имитация успешного ffmpeg: создаёт выходной файл (последний аргумент)."""
    Path(cmd[-1]).write_bytes(b"data")
    return MagicMock(returncode=0, stderr="")


class TestMuxVideo:
    def test_mux_success(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.wav"
        out = tmp_path / "final.mp4"
        video.write_bytes(b"")
        audio.write_bytes(b"")

        with patch(
            "meeting_recorder.recorder.subprocess.run", side_effect=_fake_ffmpeg_ok
        ) as mock_run:
            result = mux_video(video, audio, out)
        assert result == out
        assert out.read_bytes() == b"data"
        # ffmpeg пишет во временный файл, а не сразу в итоговый
        assert mock_run.call_args.args[0][-1] != str(out)
        assert not list(tmp_path.glob("*.part*"))

    def test_mux_failure_raises_and_leaves_no_files(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.wav"
        out = tmp_path / "final.mp4"

        def fail(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"partial")
            return MagicMock(returncode=1, stderr="error details")

        with patch("meeting_recorder.recorder.subprocess.run", side_effect=fail):
            with pytest.raises(RuntimeError, match="ffmpeg завершился"):
                mux_video(video, audio, out)
        assert not out.exists()
        assert not list(tmp_path.glob("*.part*"))

    def test_mux_timeout_leaves_no_files(self, tmp_path):
        import subprocess

        out = tmp_path / "final.mp4"

        def hang(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"partial")
            raise subprocess.TimeoutExpired(cmd, 1)

        with patch("meeting_recorder.recorder.subprocess.run", side_effect=hang):
            with pytest.raises(subprocess.TimeoutExpired):
                mux_video(tmp_path / "v.mp4", tmp_path / "a.wav", out)
        assert not out.exists()
        assert not list(tmp_path.glob("*.part*"))

    def test_mux_uses_utf8_quiet_and_no_window(self, tmp_path):
        with patch(
            "meeting_recorder.recorder.subprocess.run", side_effect=_fake_ffmpeg_ok
        ) as mock_run:
            mux_video(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4")
        kwargs = mock_run.call_args.kwargs
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert "-nostats" in mock_run.call_args.args[0]


# ---------------------------------------------------------------------------
# Старт / стоп: очистка ресурсов
# ---------------------------------------------------------------------------


class TestStartCleanup:
    def test_cmd_build_failure_does_not_start_sys_capture(self, app_cfg, tmp_session):
        """Если микрофон не найден, поток системного звука не должен остаться работать."""
        app_cfg.recording.mic_device = ""
        app_cfg.recording.system_audio_grabber = "soundcard"
        r = MeetingRecorder(app_cfg, tmp_session)
        with patch(
            "meeting_recorder.recorder.list_dshow_audio_devices", return_value=[]
        ):
            with patch("meeting_recorder.recorder.SystemAudioCapture") as mock_cap:
                with pytest.raises(RuntimeError, match="dshow"):
                    r.start()
        mock_cap.assert_not_called()
        assert not r.is_recording

    def test_popen_failure_stops_sys_capture(self, app_cfg, tmp_session):
        app_cfg.recording.system_audio_grabber = "soundcard"
        r = MeetingRecorder(app_cfg, tmp_session)
        cap = MagicMock()
        with patch("meeting_recorder.recorder.SystemAudioCapture", return_value=cap):
            with patch(
                "meeting_recorder.recorder.subprocess.Popen",
                side_effect=OSError("no ffmpeg"),
            ):
                with pytest.raises(OSError):
                    r.start()
        cap.start.assert_called_once()
        cap.stop.assert_called_once()
        assert r._stderr_file is None

    def test_autodetect_mic(self, app_cfg, tmp_session):
        app_cfg.recording.mic_device = ""
        r = MeetingRecorder(app_cfg, tmp_session)
        with patch(
            "meeting_recorder.recorder.list_dshow_audio_devices",
            return_value=["Mic A", "Mic B"],
        ):
            assert r._resolve_mic_device(app_cfg.recording) == "Mic A"

    def test_stop_kills_when_terminate_hangs(self, app_cfg, tmp_session):
        import subprocess

        app_cfg.recording.record_system_audio = False
        r = MeetingRecorder(app_cfg, tmp_session)
        proc = _make_mock_process()
        with patch("meeting_recorder.recorder.subprocess.Popen", return_value=proc):
            with patch("meeting_recorder.recorder._FFMPEG_STARTUP_TIMEOUT", 0.0):
                r.start()
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("ffmpeg", 1),
            subprocess.TimeoutExpired("ffmpeg", 1),
            None,
        ]
        r.stop()
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert not r.is_recording
        assert r._stderr_file is None

    def test_stop_resets_state_even_if_kill_fails(self, app_cfg, tmp_session):
        import subprocess

        app_cfg.recording.record_system_audio = False
        r = MeetingRecorder(app_cfg, tmp_session)
        proc = _make_mock_process()
        with patch("meeting_recorder.recorder.subprocess.Popen", return_value=proc):
            with patch("meeting_recorder.recorder._FFMPEG_STARTUP_TIMEOUT", 0.0):
                r.start()
        proc.wait.side_effect = subprocess.TimeoutExpired("ffmpeg", 1)
        with pytest.raises(subprocess.TimeoutExpired):
            r.stop()
        assert not r.is_recording


# ---------------------------------------------------------------------------
# Список устройств
# ---------------------------------------------------------------------------


def _fake_soundcard(speakers, default=None):
    sc = MagicMock()
    sc.all_speakers.return_value = speakers
    sc.default_speaker.return_value = default or (speakers[0] if speakers else None)
    return sc


def _speaker(id_, name):
    sp = MagicMock()
    sp.id = id_
    sp.name = name
    return sp


class TestDeviceListing:
    def test_list_dshow_parses_audio_devices(self):
        stderr = (
            '[dshow @ 0] "Integrated Camera" (video)\n'
            '[dshow @ 0] "Microphone (Realtek)" (audio)\n'
            '[dshow @ 0]   Alternative name "@device_cm_{X}"\n'
            '[dshow @ 0] "virtual-audio-capturer" (audio)\n'
        )
        with patch("meeting_recorder.recorder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stderr=stderr)
            assert list_dshow_audio_devices() == [
                "Microphone (Realtek)",
                "virtual-audio-capturer",
            ]

    def test_list_dshow_error_returns_empty(self):
        with patch(
            "meeting_recorder.recorder.subprocess.run",
            side_effect=FileNotFoundError("ffmpeg"),
        ):
            assert list_dshow_audio_devices() == []

    def test_list_soundcard_speakers(self):
        sc = _fake_soundcard(
            [_speaker("{id-1}", "Speakers"), _speaker("{id-2}", "Headset")]
        )
        with patch.dict(sys.modules, {"soundcard": sc}):
            devices = list_soundcard_speakers()
        assert devices == [
            AudioDevice("{id-1}", "Speakers"),
            AudioDevice("{id-2}", "Headset"),
        ]

    def test_list_soundcard_speakers_error(self):
        sc = MagicMock()
        sc.all_speakers.side_effect = RuntimeError("COM error")
        with patch.dict(sys.modules, {"soundcard": sc}):
            assert list_soundcard_speakers() == []

    def test_default_speaker(self):
        sc = _fake_soundcard([_speaker("{id-1}", "Speakers")])
        with patch.dict(sys.modules, {"soundcard": sc}):
            assert default_soundcard_speaker() == AudioDevice("{id-1}", "Speakers")

    def test_default_speaker_error(self):
        sc = MagicMock()
        sc.default_speaker.side_effect = RuntimeError("no device")
        with patch.dict(sys.modules, {"soundcard": sc}):
            assert default_soundcard_speaker() is None

    def test_no_window_kwargs(self):
        kwargs = no_window_kwargs()
        if sys.platform == "win32":
            assert "creationflags" in kwargs
        else:
            assert kwargs == {}


class TestResolveSpeaker:
    def test_by_id(self):
        a, b = _speaker("{a}", "Same Name"), _speaker("{b}", "Same Name")
        cap = SystemAudioCapture(Path("x.wav"), device="{b}")
        assert cap._resolve_speaker(_fake_soundcard([a, b])) is b

    def test_by_name(self):
        a, b = _speaker("{a}", "Speakers"), _speaker("{b}", "Headset")
        cap = SystemAudioCapture(Path("x.wav"), device="Headset")
        assert cap._resolve_speaker(_fake_soundcard([a, b])) is b

    def test_missing_falls_back_to_default(self):
        a = _speaker("{a}", "Speakers")
        cap = SystemAudioCapture(Path("x.wav"), device="Gone")
        assert cap._resolve_speaker(_fake_soundcard([a], default=a)) is a

    def test_empty_uses_default(self):
        a = _speaker("{a}", "Speakers")
        sc = _fake_soundcard([a], default=a)
        cap = SystemAudioCapture(Path("x.wav"))
        assert cap._resolve_speaker(sc) is a
        sc.all_speakers.assert_not_called()


class TestRecordGapPadding:
    def test_gap_is_padded_with_silence(self, tmp_path):
        """Разрыв захвата (поток не успел забрать данные) дополняется тишиной."""
        sr = 1000
        out = tmp_path / "sys.wav"
        cap = SystemAudioCapture(out, sample_rate=sr)
        chunk = np.ones((cap._chunk_frames, 1), dtype=np.float32) * 0.1

        calls = {"n": 0}

        def _record(numframes):
            calls["n"] += 1
            if calls["n"] >= 3:
                cap.signal_stop()
            return chunk

        rec = MagicMock()
        rec.record.side_effect = _record
        rec.__enter__ = MagicMock(return_value=rec)
        rec.__exit__ = MagicMock(return_value=False)
        mic = MagicMock()
        mic.recorder.return_value = rec
        sc = _fake_soundcard([_speaker("{a}", "Speakers")])
        sc.get_microphone.return_value = mic

        # t0 = 0.0 - 0.1; затем «прошло» 2.1 с при записанных 0.2 с → разрыв ~1.9 с
        times = iter([0.0, 0.1, 2.1, 2.2, 2.3, 2.4])
        with patch.dict(sys.modules, {"soundcard": sc}):
            with patch(
                "meeting_recorder.recorder.time.monotonic",
                side_effect=lambda: next(times),
            ):
                cap._record()

        assert cap.error is None
        data, _ = sf.read(str(out))
        assert len(data) >= int(2.0 * sr)
        assert np.count_nonzero(np.abs(data) < 1e-6) > sr  # вставлена тишина
