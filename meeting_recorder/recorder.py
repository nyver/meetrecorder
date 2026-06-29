"""Запись экрана и звука через ffmpeg (subprocess)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .config import RecordingConfig, AppConfig
from .naming import SessionPaths

logger = logging.getLogger(__name__)

# Таймауты для управления ffmpeg-процессом
_FFMPEG_GRACEFUL_STOP_TIMEOUT = 10  # секунд ждать graceful stop перед force-kill
_FFMPEG_FORCE_KILL_TIMEOUT = 5      # секунд ждать после terminate()
_FFMPEG_STARTUP_TIMEOUT = 3.0       # секунд ждать инициализацию устройств при старте
_FFMPEG_STARTUP_POLL = 0.2          # интервал опроса во время ожидания старта
_SYS_AUDIO_JOIN_TIMEOUT = 10        # секунд ждать завершения потока soundcard


# ---------------------------------------------------------------------------
# Захват системного аудио через WASAPI loopback (soundcard)
# ---------------------------------------------------------------------------


class SystemAudioCapture:
    """Запись системного аудио (WASAPI loopback) в отдельном потоке."""

    _CHUNK_MS = 100  # длина чанка в миллисекундах

    def __init__(self, output_path: Path, sample_rate: int = 48000):
        self._path = output_path
        self._sr = sample_rate
        self._chunk_frames = sample_rate * self._CHUNK_MS // 1000
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record, daemon=True, name="sys-audio")
        self._thread.start()

    def signal_stop(self) -> None:
        """Сигнализировать о необходимости остановки без ожидания потока."""
        self._stop_event.set()

    def wait(self, timeout: float = 10) -> None:
        """Дождаться завершения потока записи."""
        if self._thread:
            self._thread.join(timeout=timeout)

    def stop(self) -> None:
        self.signal_stop()
        self.wait()

    def _record(self) -> None:
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            logger.info("Захват системного аудио: %s", speaker.name)

            frames_written = 0
            with loopback.recorder(samplerate=self._sr, channels=1, blocksize=self._chunk_frames) as rec:
                with sf.SoundFile(
                    str(self._path),
                    mode="w",
                    samplerate=self._sr,
                    channels=1,
                    subtype="PCM_16",
                ) as out:
                    while not self._stop_event.is_set():
                        chunk = rec.record(numframes=self._chunk_frames)
                        out.write(chunk)
                        frames_written += len(chunk)

            if frames_written:
                logger.info("Системный звук сохранён: %s (%.1f сек)", self._path, frames_written / self._sr)
            else:
                logger.warning("Системный звук: нет данных")
        except Exception as exc:
            self.error = exc
            logger.error("Ошибка захвата системного аудио: %s", exc)


# ---------------------------------------------------------------------------
# Утилиты аудио
# ---------------------------------------------------------------------------


def mix_audio_files(
    mic_path: Path,
    system_path: Path,
    output_path: Path,
    sample_rate: int = 48000,
) -> Path:
    """Свести два WAV-файла в один (смешивание по среднему)."""
    data_mic, sr_mic = sf.read(mic_path, dtype="float32")
    data_sys, sr_sys = sf.read(system_path, dtype="float32")

    if len(data_mic) == 0 or len(data_sys) == 0:
        raise ValueError("Один из аудиофайлов пустой")

    # Привести к общей частоте
    if sr_mic != sr_sys:
        try:
            import resampy
            logger.info(
                "Ресемплинг системного аудио: %d → %d Гц", sr_sys, sr_mic,
            )
            data_sys = resampy.resample(data_sys, sr_sys, sr_mic)
        except ImportError:
            raise RuntimeError(
                f"Несовпадение частот дискретизации (mic={sr_mic} Гц, system={sr_sys} Гц). "
                f"Установите resampy для автоматического ресемплинга: pip install resampy"
            )

    # Выравниваем по концу записи, а не по началу.
    # soundcard стартует раньше ffmpeg (~2-5 с на инициализацию gdigrab/dshow),
    # поэтому _system.wav длиннее — лишние кадры в начале, а не в конце.
    # Обрезаем лишнее с начала более длинного файла.
    len_diff = len(data_sys) - len(data_mic)
    if len_diff > 0:
        data_sys = data_sys[len_diff:]
        logger.debug("Синхронизация аудио: обрезано %d кадров (%.2f с) с начала system", len_diff, len_diff / sr_mic)
    elif len_diff < 0:
        data_mic = data_mic[-len_diff:]
        logger.debug("Синхронизация аудио: обрезано %d кадров (%.2f с) с начала mic", -len_diff, -len_diff / sr_mic)

    # Привести оба потока к моно перед смешиванием
    if data_mic.ndim > 1:
        data_mic = data_mic[:, :1]
    else:
        data_mic = data_mic[:, np.newaxis]
    if data_sys.ndim > 1:
        data_sys = data_sys[:, :1]
    else:
        data_sys = data_sys[:, np.newaxis]

    # Свести по среднему
    mixed = (data_mic + data_sys) / 2

    sf.write(str(output_path), mixed, sr_mic, subtype="PCM_16")
    logger.info("Сведённое аудио сохранено: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Класс записи
# ---------------------------------------------------------------------------


class MeetingRecorder:
    """Управление ffmpeg-процессом записи экрана + двух аудиодорожек."""

    def __init__(self, config: AppConfig, paths: SessionPaths):
        self.config = config
        self.paths = paths
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._stderr_file = None
        self._recording = False
        self._start_time: Optional[float] = None
        self._sys_capture: Optional[SystemAudioCapture] = None

    # -- публичные методы --------------------------------------------------

    def start(self) -> None:
        """Начать запись экрана + микрофон + системный звук."""
        if self._recording:
            raise RuntimeError("Запись уже идёт")

        rc = self.config.recording

        # Запуск soundcard-захвата системного аудио (до ffmpeg, чтобы не пропустить начало)
        if rc.record_system_audio and rc.system_audio_grabber == "soundcard":
            self._sys_capture = SystemAudioCapture(self.paths.system_audio, rc.audio_sample_rate)
            self._sys_capture.start()

        cmd = self._build_ffmpeg_cmd()
        logger.info("Начинаю запись (ffmpeg cmd: %s ...)", " ".join(cmd[:4]))

        self._stderr_file = open(str(self.paths.ffmpeg_log), "wb")
        try:
            kwargs: dict = {}
            if sys.platform == "win32":
                # Отдельная process group нужна для CTRL_BREAK_EVENT при graceful stop
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
                **kwargs,
            )
        except Exception:
            self._stderr_file.close()
            self._stderr_file = None
            if self._sys_capture is not None:
                self._sys_capture.signal_stop()
                self._sys_capture = None
            raise

        # Проверяем успешность старта: ждём до _FFMPEG_STARTUP_TIMEOUT с, детектируем ранний
        # выход ffmpeg или ошибку soundcard-захвата и прерываем запуск с понятным сообщением.
        elapsed = 0.0
        while elapsed < _FFMPEG_STARTUP_TIMEOUT:
            time.sleep(_FFMPEG_STARTUP_POLL)
            elapsed += _FFMPEG_STARTUP_POLL
            if self._process.poll() is not None:
                self._fail_startup_ffmpeg()
            if self._sys_capture is not None and self._sys_capture.error is not None:
                self._fail_startup_sys_audio()

        self._recording = True
        self._start_time = time.monotonic()
        logger.info("Запись начата: %s", self.paths.session_id)

    def _fail_startup_ffmpeg(self) -> None:
        """Прервать запуск: ffmpeg завершился с ошибкой при инициализации устройств."""
        exit_code = self._process.returncode
        if self._sys_capture is not None:
            self._sys_capture.signal_stop()
            self._sys_capture = None
        self._process = None
        try:
            self._stderr_file.flush()
            self._stderr_file.close()
        except Exception:
            pass
        self._stderr_file = None
        try:
            log_tail = self.paths.ffmpeg_log.read_text(encoding="utf-8", errors="replace")[-600:]
        except Exception:
            log_tail = ""
        msg = f"ffmpeg завершился с кодом {exit_code} при старте — ошибка захвата видео или аудио"
        logger.error("%s\n%s", msg, log_tail)
        raise RuntimeError(msg)

    def _fail_startup_sys_audio(self) -> None:
        """Прервать запуск: ошибка захвата системного аудио (soundcard) при инициализации."""
        err = self._sys_capture.error
        self._sys_capture.signal_stop()
        self._sys_capture = None
        try:
            if self._process and self._process.stdin:
                self._process.stdin.write(b"q")
                self._process.stdin.flush()
        except Exception:
            pass
        try:
            if self._process:
                self._process.wait(timeout=_FFMPEG_GRACEFUL_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=_FFMPEG_FORCE_KILL_TIMEOUT)
        self._process = None
        try:
            self._stderr_file.close()
        except Exception:
            pass
        self._stderr_file = None
        msg = f"Ошибка захвата системного аудио: {err}"
        logger.error(msg)
        raise RuntimeError(msg)

    def stop(self) -> None:
        """Gracefully остановить запись (отправить 'q' во stdin)."""
        if not self._recording or self._process is None:
            logger.warning("Запись не идёт — stop вызван без start")
            return

        # Отправляем 'q' для graceful stop ffmpeg и одновременно сигнализируем
        # soundcard-захвату, чтобы оба потока завершились в одно время.
        # Важно: если сначала ждать ffmpeg.wait() и только потом останавливать
        # soundcard, system_audio.wav будет длиннее на время flush-а ffmpeg
        # (2-5 с), и mix_audio_files срежет эти секунды с начала system-трека,
        # сдвигая его вперёд относительно mic.
        try:
            if self._process.stdin:
                self._process.stdin.write(b"q")
                self._process.stdin.flush()
        except Exception:
            pass

        if self._sys_capture is not None:
            self._sys_capture.signal_stop()

        try:
            self._process.wait(timeout=_FFMPEG_GRACEFUL_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg не завершился за %ds, завершаю принудительно", _FFMPEG_GRACEFUL_STOP_TIMEOUT)
            self._process.terminate()
            self._process.wait(timeout=_FFMPEG_FORCE_KILL_TIMEOUT)

        self._recording = False
        if self._stderr_file:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

        if self._sys_capture is not None:
            self._sys_capture.wait(timeout=_SYS_AUDIO_JOIN_TIMEOUT)
            if self._sys_capture.error:
                logger.error("Системный звук не записан: %s", self._sys_capture.error)
            self._sys_capture = None

        duration = time.monotonic() - self._start_time if self._start_time else 0
        logger.info(
            "Запись завершена: %s (длительность: %0.1f сек)",
            self.paths.session_id, duration,
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def ffmpeg_pid(self) -> int | None:
        return self._process.pid if self._process else None

    @property
    def duration(self) -> float:
        if not self._recording or self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _detect_dshow_mic() -> str | None:
        """Вернуть dshow-имя первого доступного capture-устройства или None."""
        import re
        try:
            result = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True, text=True, timeout=5,
                encoding="utf-8", errors="replace",
            )
            # ffmpeg пишет список устройств в stderr; audio-раздел идёт после video
            audio_section = False
            for line in result.stderr.splitlines():
                if "(audio)" in line and not audio_section:
                    audio_section = True
                m = re.search(r'"([^"]+)"\s*\(audio\)', line)
                if m:
                    return m.group(1)
        except Exception:
            pass
        return None

    def _resolve_mic_device(self, rc) -> str:
        """Вернуть имя mic-устройства: настроенное или автодетект."""
        if rc.mic_device:
            return rc.mic_device
        detected = self._detect_dshow_mic()
        if detected:
            logger.info("Микрофон автодетект: %s", detected)
            return detected
        raise RuntimeError(
            "Не удалось найти ни одного dshow audio capture-устройства. "
            "Проверьте подключение микрофона или задайте mic_device в config.yaml явно."
        )

    def _system_audio_input(self, rc) -> list[str]:
        if rc.system_audio_grabber == "wasapi":
            # WASAPI loopback: захватывает системный аудиовыход без доп. драйверов
            return ["-f", "wasapi", "-loopback", "1", "-i", "default"]
        return ["-f", "dshow", "-i", f"audio={rc.system_audio_device}"]

    def _build_ffmpeg_cmd(self) -> list[str]:
        rc = self.config.recording

        # Screen capture input differs by grabber
        if rc.screen_grabber == "ddagrab":
            # ddagrab: hardware DDA capture, input is display index
            screen_input = ["-f", "ddagrab", "-framerate", str(rc.fps), "-i", "0"]
            # hwdownload needed to get CPU-accessible frames for libx264
            video_filters = ["-vf", "hwdownload,format=bgr0"]
        else:
            # gdigrab: GDI capture, supports draw_mouse
            screen_input = [
                "-f", "gdigrab",
                "-framerate", str(rc.fps),
                "-draw_mouse", "1",
                "-i", "desktop",
            ]
            video_filters = []

        cmd: list[str] = [
            "ffmpeg", "-y",
            # --- input 0: screen ---
            *screen_input,
        ]

        # input 1: микрофон (опционально)
        if rc.record_mic:
            mic_device = self._resolve_mic_device(rc)
            cmd += ["-f", "dshow", "-i", f"audio={mic_device}"]

        # input индекс для системного звука зависит от того, есть ли mic
        sys_audio_input_idx = 2 if rc.record_mic else 1

        if rc.record_system_audio and rc.system_audio_grabber != "soundcard":
            cmd += self._system_audio_input(rc)

        cmd += [
            # --- output 1: видео (без аудио) ---
            "-map", "0:v:0",
            *video_filters,
            "-c:v", rc.video_codec,
            "-pix_fmt", "yuv420p",
            "-an",
            # Fragmented MP4: moov не нужен в конце — файл валиден после force-kill
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            str(self.paths.video),
        ]

        if rc.record_mic:
            cmd += [
                # --- output 2: микрофон WAV ---
                "-map", "1:a:0",
                "-c:a", "pcm_s16le",
                "-ar", str(rc.audio_sample_rate),
                "-ac", "1",
                str(self.paths.mic_audio),
            ]

        if rc.record_system_audio and rc.system_audio_grabber != "soundcard":
            cmd += [
                # --- output 3: системный звук WAV ---
                "-map", f"{sys_audio_input_idx}:a:0",
                "-c:a", "pcm_s16le",
                "-ar", str(rc.audio_sample_rate),
                "-ac", "1",
                str(self.paths.system_audio),
            ]

        return cmd

    def get_status(self) -> dict:
        return {
            "session_id": self.paths.session_id,
            "recording": self._recording,
            "duration_sec": self.duration,
            "video": str(self.paths.video),
            "mic_audio": str(self.paths.mic_audio),
            "system_audio": str(self.paths.system_audio),
        }


# ---------------------------------------------------------------------------
# Post-processing: разделение потока ffmpeg на отдельные дорожки
# ---------------------------------------------------------------------------


def mux_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Объединить видео и аудио в один MP4-файл (без перекодирования видео)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg завершился с ошибкой:\n{result.stderr[-1000:]}"
        )
    logger.info("Финальное видео сохранено: %s", output_path)
    return output_path


def split_streams(video_path: Path, paths: SessionPaths, sample_rate: int = 48000) -> tuple[Path, Path]:
    """Извлечь аудио-дорожки из одного video-файла (если ffmpeg записал всё в один файл).

    Эта функция — запасной вариант, если раздельная запись не сработала.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    cmd_mic = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-c:a", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(paths.mic_audio),
    ]
    cmd_sys = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-map", "0:a:1",
        "-c:a", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(paths.system_audio),
    ]

    subprocess.run(cmd_mic, check=True, capture_output=True)
    subprocess.run(cmd_sys, check=True, capture_output=True)
    return paths.mic_audio, paths.system_audio
