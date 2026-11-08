"""Запись экрана и звука через ffmpeg (subprocess)."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

from .config import AppConfig
from .naming import SessionPaths
from .session_utils import replace_with_retry

logger = logging.getLogger(__name__)

# Таймауты для управления ffmpeg-процессом
_FFMPEG_GRACEFUL_STOP_TIMEOUT = 10  # секунд ждать graceful stop перед force-kill
_FFMPEG_FORCE_KILL_TIMEOUT = 5  # секунд ждать после terminate()
_FFMPEG_STARTUP_TIMEOUT = 3.0  # секунд ждать инициализацию устройств при старте
_FFMPEG_STARTUP_POLL = 0.2  # интервал опроса во время ожидания старта
_SYS_AUDIO_JOIN_TIMEOUT = 10  # секунд ждать завершения потока soundcard
_SYS_AUDIO_GAP_TOLERANCE = 0.5  # секунд: разрыв больше — дополняем тишиной
_DEVICE_LIST_TIMEOUT = 5  # секунд на ffmpeg -list_devices
_MUX_TIMEOUT = 1800  # секунд на mux (AAC-кодирование длинной встречи)
_MIX_BLOCK_FRAMES = 1 << 18  # ~5.5 с при 48 кГц: блок потокового сведения


def no_window_kwargs() -> dict:
    """kwargs для subprocess: не показывать консольное окно на Windows.

    Без CREATE_NO_WINDOW каждый вызов ffmpeg/ffprobe из GUI-процесса (трей,
    запущенный через pythonw) мигает консольным окном.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


class AudioDevice(NamedTuple):
    """Аудиоустройство: стабильный id (для поиска) и имя (для отображения)."""

    id: str
    name: str


# ---------------------------------------------------------------------------
# Захват системного аудио через WASAPI loopback (soundcard)
# ---------------------------------------------------------------------------


class SystemAudioCapture:
    """Запись системного аудио (WASAPI loopback) в отдельном потоке."""

    _CHUNK_MS = 100  # длина чанка в миллисекундах

    def __init__(self, output_path: Path, sample_rate: int = 48000, device: str = ""):
        self._path = output_path
        self._sr = sample_rate
        self._device = device  # id или имя устройства вывода; "" — по умолчанию
        self._chunk_frames = sample_rate * self._CHUNK_MS // 1000
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._record, daemon=True, name="sys-audio"
        )
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

    def _resolve_speaker(self, sc):
        """Найти устройство вывода по id, затем по имени; иначе — по умолчанию."""
        if self._device:
            speakers = sc.all_speakers()
            for attr in ("id", "name"):
                match = next(
                    (
                        sp
                        for sp in speakers
                        if str(getattr(sp, attr, "")) == self._device
                    ),
                    None,
                )
                if match is not None:
                    return match
            logger.warning(
                "Устройство вывода '%s' не найдено, "
                "используется устройство по умолчанию",
                self._device,
            )
        return sc.default_speaker()

    def _record(self) -> None:
        try:
            import soundcard as sc

            speaker = self._resolve_speaker(sc)
            loopback = sc.get_microphone(id=str(speaker.id), include_loopback=True)
            logger.info("Захват системного аудио: %s", speaker.name)

            frames_written = 0
            frames_padded = 0
            gap_limit = int(self._sr * _SYS_AUDIO_GAP_TOLERANCE)
            t0: float | None = None
            with loopback.recorder(
                samplerate=self._sr, channels=1, blocksize=self._chunk_frames
            ) as rec:
                with sf.SoundFile(
                    str(self._path),
                    mode="w",
                    samplerate=self._sr,
                    channels=1,
                    subtype="PCM_16",
                ) as out:
                    while not self._stop_event.is_set():
                        chunk = rec.record(numframes=self._chunk_frames)
                        if t0 is None:
                            t0 = time.monotonic() - len(chunk) / self._sr
                        out.write(chunk)
                        frames_written += len(chunk)
                        # Компенсация разрывов (переполнение буфера WASAPI, когда
                        # поток не успел забрать данные): дорожка должна идти
                        # синхронно с настенными часами, иначе mix_audio_files,
                        # выравнивающий дорожки по концу, сдвинет system-трек.
                        gap = int((time.monotonic() - t0) * self._sr) - frames_written
                        if gap > gap_limit:
                            out.write(np.zeros((gap, 1), dtype=np.float32))
                            frames_written += gap
                            frames_padded += gap

            if frames_padded:
                logger.warning(
                    "Системный звук: компенсировано %.1f сек разрывов захвата",
                    frames_padded / self._sr,
                )
            if frames_written:
                logger.info(
                    "Системный звук сохранён: %s (%.1f сек)",
                    self._path,
                    frames_written / self._sr,
                )
            else:
                logger.warning("Системный звук: нет данных")
        except Exception as exc:
            self.error = exc
            logger.error("Ошибка захвата системного аудио: %s", exc)


# ---------------------------------------------------------------------------
# Список доступных аудиоустройств
# ---------------------------------------------------------------------------


_DSHOW_AUDIO_RE = re.compile(r'"([^"]+)"\s*\(audio\)')


def list_dshow_audio_devices() -> list[str]:
    """Вернуть имена dshow audio capture-устройств (микрофоны, virtual cable и т.п.)."""
    devices: list[str] = []
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            text=True,
            timeout=_DEVICE_LIST_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            **no_window_kwargs(),
        )
        for line in (result.stderr or "").splitlines():
            m = _DSHOW_AUDIO_RE.search(line)
            if m:
                devices.append(m.group(1))
    except Exception as exc:
        logger.warning("Не удалось получить список dshow-устройств: %s", exc)
    return devices


def list_soundcard_speakers() -> list[AudioDevice]:
    """Вернуть устройства вывода звука, видимые библиотеке soundcard."""
    try:
        import soundcard as sc

        return [AudioDevice(str(sp.id), str(sp.name)) for sp in sc.all_speakers()]
    except Exception as exc:
        logger.warning("Не удалось получить список устройств вывода: %s", exc)
        return []


def default_soundcard_speaker() -> AudioDevice | None:
    """Вернуть устройство вывода по умолчанию или None."""
    try:
        import soundcard as sc

        sp = sc.default_speaker()
        return AudioDevice(str(sp.id), str(sp.name))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Утилиты аудио
# ---------------------------------------------------------------------------


def _resample_with_ffmpeg(src: Path, sample_rate: int) -> Path:
    """Пересэмплировать WAV во временный моно-файл через ffmpeg (потоково)."""
    tmp = src.with_name(f".{src.stem}.{sample_rate}hz.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-nostats",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(tmp),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_MUX_TIMEOUT,
        **no_window_kwargs(),
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Не удалось пересэмплировать {src.name}: {result.stderr[-500:]}"
        )
    return tmp


def mix_audio_files(
    mic_path: Path,
    system_path: Path,
    output_path: Path,
    sample_rate: int = 48000,
) -> Path:
    """Свести два WAV-файла в один моно-файл (смешивание по среднему).

    Обработка потоковая, блоками по _MIX_BLOCK_FRAMES: память не зависит от
    длительности встречи. Результат пишется во временный файл и атомарно
    переименовывается — недописанный mix не останется на диске.
    """
    mic_path, system_path, output_path = (
        Path(mic_path),
        Path(system_path),
        Path(output_path),
    )
    info_mic = sf.info(str(mic_path))
    info_sys = sf.info(str(system_path))
    if info_mic.frames == 0 or info_sys.frames == 0:
        raise ValueError("Один из аудиофайлов пустой")

    sr = info_mic.samplerate
    resampled: Path | None = None
    tmp_out = output_path.with_name(f".{output_path.stem}.part.wav")
    try:
        if info_sys.samplerate != sr:
            logger.info(
                "Ресемплинг системного аудио: %d → %d Гц", info_sys.samplerate, sr
            )
            resampled = _resample_with_ffmpeg(system_path, sr)
            system_path = resampled
            info_sys = sf.info(str(system_path))

        # Выравниваем по концу записи, а не по началу.
        # soundcard стартует раньше ffmpeg (~2-5 с на инициализацию gdigrab/dshow),
        # поэтому _system.wav длиннее — лишние кадры в начале, а не в конце.
        # Пропускаем лишнее с начала более длинного файла.
        skip_mic = max(info_mic.frames - info_sys.frames, 0)
        skip_sys = max(info_sys.frames - info_mic.frames, 0)
        if skip_mic or skip_sys:
            logger.debug(
                "Синхронизация аудио: пропущено %.2f с с начала %s",
                max(skip_mic, skip_sys) / sr,
                "mic" if skip_mic else "system",
            )
        remaining = min(info_mic.frames, info_sys.frames)

        with (
            sf.SoundFile(str(mic_path)) as f_mic,
            sf.SoundFile(str(system_path)) as f_sys,
            sf.SoundFile(
                str(tmp_out),
                mode="w",
                samplerate=sr,
                channels=1,
                subtype="PCM_16",
                format="WAV",
            ) as out,
        ):
            f_mic.seek(skip_mic)
            f_sys.seek(skip_sys)
            while remaining > 0:
                n = min(_MIX_BLOCK_FRAMES, remaining)
                a = f_mic.read(n, dtype="float32", always_2d=True)[:, 0]
                b = f_sys.read(n, dtype="float32", always_2d=True)[:, 0]
                m = min(len(a), len(b))
                if m == 0:
                    break
                out.write((a[:m] + b[:m]) * 0.5)
                remaining -= m

        replace_with_retry(tmp_out, output_path)
    finally:
        tmp_out.unlink(missing_ok=True)
        if resampled is not None:
            resampled.unlink(missing_ok=True)

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
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file = None
        self._recording = False
        self._start_time: float | None = None
        self._sys_capture: SystemAudioCapture | None = None

    # -- публичные методы --------------------------------------------------

    def start(self) -> None:
        """Начать запись экрана + микрофон + системный звук."""
        if self._recording:
            raise RuntimeError("Запись уже идёт")

        rc = self.config.recording

        # Команду собираем ДО запуска захвата системного звука: она может упасть
        # (например, микрофон не найден), и тогда поток soundcard остался бы
        # писать файл бесконечно.
        cmd = self._build_ffmpeg_cmd()
        logger.info("Начинаю запись (ffmpeg cmd: %s ...)", " ".join(cmd[:4]))

        # Запуск soundcard-захвата системного аудио
        # (до ffmpeg, чтобы не пропустить начало)
        if rc.record_system_audio and rc.system_audio_grabber == "soundcard":
            self._sys_capture = SystemAudioCapture(
                self.paths.system_audio,
                rc.audio_sample_rate,
                device=rc.system_audio_speaker,
            )
            self._sys_capture.start()

        try:
            self._stderr_file = open(str(self.paths.ffmpeg_log), "wb")
            kwargs: dict = {}
            if sys.platform == "win32":
                # Отдельная process group нужна для CTRL_BREAK_EVENT в `mrec stop`
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
                **kwargs,
            )
        except Exception:
            self._close_stderr()
            self._abort_sys_capture()
            raise

        # Проверяем успешность старта: ждём до _FFMPEG_STARTUP_TIMEOUT с,
        # детектируем ранний выход ffmpeg или ошибку soundcard-захвата и
        # прерываем запуск с понятным сообщением.
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

    def _close_stderr(self) -> None:
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    def _abort_sys_capture(self) -> None:
        """Остановить захват системного звука и дождаться потока."""
        if self._sys_capture is not None:
            self._sys_capture.stop()
            self._sys_capture = None

    def _send_quit(self) -> None:
        try:
            if self._process and self._process.stdin:
                self._process.stdin.write(b"q")
                self._process.stdin.flush()
        except Exception:
            pass

    def _wait_or_kill(self) -> None:
        """Дождаться ffmpeg; при зависании — terminate, затем kill."""
        proc = self._process
        if proc is None:
            return
        try:
            proc.wait(timeout=_FFMPEG_GRACEFUL_STOP_TIMEOUT)
            return
        except subprocess.TimeoutExpired:
            logger.warning(
                "ffmpeg не завершился за %ds, завершаю принудительно",
                _FFMPEG_GRACEFUL_STOP_TIMEOUT,
            )
        proc.terminate()
        try:
            proc.wait(timeout=_FFMPEG_FORCE_KILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_FFMPEG_FORCE_KILL_TIMEOUT)

    def _fail_startup_ffmpeg(self) -> None:
        """Прервать запуск: ffmpeg завершился с ошибкой при инициализации устройств."""
        exit_code = self._process.returncode
        self._abort_sys_capture()
        self._process = None
        self._close_stderr()
        try:
            log_tail = self.paths.ffmpeg_log.read_text(
                encoding="utf-8", errors="replace"
            )[-600:]
        except Exception:
            log_tail = ""
        msg = (
            f"ffmpeg завершился с кодом {exit_code} при старте — "
            "ошибка захвата видео или аудио"
        )
        logger.error("%s\n%s", msg, log_tail)
        raise RuntimeError(msg)

    def _fail_startup_sys_audio(self) -> None:
        """Прервать запуск: ошибка захвата системного аудио (soundcard) при старте."""
        err = self._sys_capture.error
        self._abort_sys_capture()
        self._send_quit()
        try:
            self._wait_or_kill()
        finally:
            self._process = None
            self._close_stderr()
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
        self._send_quit()
        if self._sys_capture is not None:
            self._sys_capture.signal_stop()

        try:
            self._wait_or_kill()
        finally:
            self._recording = False
            self._close_stderr()
            if self._sys_capture is not None:
                self._sys_capture.wait(timeout=_SYS_AUDIO_JOIN_TIMEOUT)
                if self._sys_capture.error:
                    logger.error(
                        "Системный звук не записан: %s", self._sys_capture.error
                    )
                self._sys_capture = None

        duration = time.monotonic() - self._start_time if self._start_time else 0
        logger.info(
            "Запись завершена: %s (длительность: %0.1f сек)",
            self.paths.session_id,
            duration,
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
        devices = list_dshow_audio_devices()
        return devices[0] if devices else None

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
                "-f",
                "gdigrab",
                "-framerate",
                str(rc.fps),
                "-draw_mouse",
                "1",
                "-i",
                "desktop",
            ]
            video_filters = []

        cmd: list[str] = [
            "ffmpeg",
            "-y",
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
            "-map",
            "0:v:0",
            *video_filters,
            "-c:v",
            rc.video_codec,
            "-pix_fmt",
            "yuv420p",
            "-an",
            # Fragmented MP4: moov не нужен в конце — файл валиден после force-kill
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            str(self.paths.video),
        ]

        if rc.record_mic:
            cmd += [
                # --- output 2: микрофон WAV ---
                "-map",
                "1:a:0",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(rc.audio_sample_rate),
                "-ac",
                "1",
                str(self.paths.mic_audio),
            ]

        if rc.record_system_audio and rc.system_audio_grabber != "soundcard":
            cmd += [
                # --- output 3: системный звук WAV ---
                "-map",
                f"{sys_audio_input_idx}:a:0",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(rc.audio_sample_rate),
                "-ac",
                "1",
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
# Post-processing: сведение видео и аудио
# ---------------------------------------------------------------------------


def mux_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Объединить видео и аудио в один MP4-файл (без перекодирования видео).

    ffmpeg пишет во временный *.part.mp4, который переименовывается только
    после успешного завершения: недописанный файл не выдаёт себя за готовый.
    """
    output_path = Path(output_path)
    tmp = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-nostats",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(tmp),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_MUX_TIMEOUT,
            **no_window_kwargs(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg завершился с ошибкой:\n" + (result.stderr or "")[-1000:]
            )
        replace_with_retry(tmp, output_path)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("Финальное видео сохранено: %s", output_path)
    return output_path
