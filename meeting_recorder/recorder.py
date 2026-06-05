"""Запись экрана и звука через ffmpeg (subprocess)."""

from __future__ import annotations

import logging
import subprocess
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
        self._recording = False
        self._start_time: Optional[float] = None
        self._stop_callback: Optional[callable] = None

    # -- публичные методы --------------------------------------------------

    def start(self) -> None:
        """Начать запись экрана + микрофон + системный звук."""
        if self._recording:
            raise RuntimeError("Запись уже идёт")

        cmd = self._build_ffmpeg_cmd()
        logger.info("Начинаю запись (ffmpeg cmd: %s ...)", " ".join(cmd[:4]))

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
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
        duration = time.monotonic() - self._start_time if self._start_time else 0
        logger.info(
            "Запись завершена: %s (длительность: %0.1f сек)",
            self.paths.session_id, duration,
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def duration(self) -> float:
        if not self._recording or self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    # -- internals ---------------------------------------------------------

    def _build_ffmpeg_cmd(self) -> list[str]:
        rc = self.config.recording
        cmd: list[str] = [
            "ffmpeg",
            "-y",
            # --- video: экран ---
            "-f", rc.screen_grabber,
            "-draw_mouse", "0",
            "-i", "desktop",
            # --- audio: микрофон (dshow) ---
            "-f", "dshow",
            "-i", f"audio={rc.mic_device}",
            # --- audio: системный звук (loopback / virtual cable) ---
            "-f", "dshow",
            "-i", f"audio={rc.system_audio_device}",
            # --- video codec ---
            "-c:v", rc.video_codec,
            "-r", str(rc.fps),
            "-pix_fmt", "yuv420p",
            # --- audio: перекодировать в WAV 16kHz PCM ---
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            # --- map streams ---
            "-map", "0:v:0",
            "-map", "1:a:0",   # mic
            "-map", "2:a:0",   # system
            # --- output ---
            str(self.paths.video),
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
