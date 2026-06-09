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


# ---------------------------------------------------------------------------
# Захват системного аудио через WASAPI loopback (soundcard)
# ---------------------------------------------------------------------------


class SystemAudioCapture:
    """Запись системного аудио (WASAPI loopback) в отдельном потоке."""

    CHUNK_FRAMES = 4800  # 100 мс при 48 кГц

    def __init__(self, output_path: Path, sample_rate: int = 48000):
        self._path = output_path
        self._sr = sample_rate
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record, daemon=True, name="sys-audio")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _record(self) -> None:
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            logger.info("Захват системного аудио: %s", speaker.name)

            chunks: list[np.ndarray] = []
            with loopback.recorder(samplerate=self._sr, channels=1, blocksize=self.CHUNK_FRAMES) as rec:
                while not self._stop_event.is_set():
                    chunk = rec.record(numframes=self.CHUNK_FRAMES)
                    chunks.append(chunk)

            if chunks:
                audio = np.concatenate(chunks)
                sf.write(str(self._path), audio, self._sr, subtype="PCM_16")
                logger.info("Системный звук сохранён: %s (%.1f сек)", self._path, len(audio) / self._sr)
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

    # Привести к общей частоте
    if sr_mic != sr_sys:
        logger.warning(
            "Несоответствие частот: mic=%d, system=%d. Приведение system к %d",
            sr_mic, sr_sys, sr_mic,
        )
        try:
            import resampy
            data_sys = resampy.resample(data_sys, sr_sys, sr_mic)
        except ImportError:
            logger.warning("resampy не установлен — пропускаю ресемплинг")

    # Привести к одинаковой длине
    min_len = min(len(data_mic), len(data_sys))
    if min_len == 0:
        raise ValueError("Один из аудиофайлов пустой")

    data_mic = data_mic[:min_len]
    data_sys = data_sys[:min_len]

    # Если моно — сделать 2D
    if data_mic.ndim == 1:
        data_mic = data_mic[:, np.newaxis]
    if data_sys.ndim == 1:
        data_sys = data_sys[:, np.newaxis]

    # Свести по среднему
    mixed = (data_mic + data_sys) / 2

    sf.write(str(output_path), mixed, sample_rate, subtype="PCM_16")
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
        self._stop_callback: Optional[callable] = None
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
        self._recording = True
        self._start_time = time.monotonic()
        logger.info("Запись начата: %s", self.paths.session_id)

    def stop(self) -> None:
        """Gracefully остановить запись (отправить 'q' во stdin)."""
        if not self._recording or self._process is None:
            logger.warning("Запись не идёт — stop вызван без start")
            return

        # Отправляем 'q' для graceful stop
        try:
            if self._process.stdin:
                self._process.stdin.write(b"q")
                self._process.stdin.flush()
        except Exception:
            pass

        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg не завершился за 10с, завершаю принудительно")
            self._process.terminate()
            self._process.wait(timeout=5)

        self._recording = False
        if self._stderr_file:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

        # Остановить soundcard-захват (ждём завершения записи в файл)
        if self._sys_capture is not None:
            self._sys_capture.stop()
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
            # --- input 1: микрофон (dshow) ---
            "-f", "dshow", "-i", f"audio={rc.mic_device}",
        ]

        if rc.record_system_audio and rc.system_audio_grabber != "soundcard":
            # --- input 2: системный звук (dshow/wasapi) ---
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
                "-map", "2:a:0",
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


def split_streams(video_path: Path, paths: SessionPaths) -> tuple[Path, Path]:
    """Извлечь аудио-дорожки из одного video-файла (если ffmpeg записал всё в один файл).

    Эта функция — запасной вариант, если раздельная запись не сработала.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    # Попробуем извлечь дорожки из видеофайла
    # Дорожка 1 — микрофон, дорожка 2 — системный звук
    cmd_mic = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        str(paths.mic_audio),
    ]
    cmd_sys = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-map", "0:a:1",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        str(paths.system_audio),
    ]

    subprocess.run(cmd_mic, check=True, capture_output=True)
    subprocess.run(cmd_sys, check=True, capture_output=True)
    return paths.mic_audio, paths.system_audio
