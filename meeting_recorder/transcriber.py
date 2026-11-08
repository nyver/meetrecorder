"""Транскрипция (faster-whisper) + диаризация (pyannote.audio): аудио → сегменты."""

from __future__ import annotations

import contextlib
import gc
import logging
import threading
from pathlib import Path
from typing import Any

from .config import AppConfig
from .session_utils import load_transcript_data, save_transcript_data

logger = logging.getLogger(__name__)

# Кэш для моделей (singleton) + lock для thread-safety
_model_cache: dict[str, Any] = {}
_model_cache_lock = threading.Lock()

_SAMPLE_RATE = 16000  # частота, на которой работают Whisper и pyannote
# Последовательность batch_size при нехватке видеопамяти (0 — без батчинга)
_OOM_MIN_BATCH = 4


def _get_transcribe_model(model_name: str, device: str, dtype: str | None = None):
    """Загрузить Whisper-модель с кэшированием (thread-safe).

    Лок удерживается на всё время загрузки, чтобы не допустить одновременной
    загрузки одной модели двумя потоками (риск OOM на GPU).
    """
    key = f"{model_name}_{device}_{dtype}"
    with _model_cache_lock:
        if key in _model_cache:
            return _model_cache[key]

        from faster_whisper import WhisperModel

        logger.info(
            "Загрузка модели Whisper (model=%s, device=%s, dtype=%s)…",
            model_name,
            device,
            dtype,
        )
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=dtype or ("float16" if device == "cuda" else "int8"),
            # транскрибируем в одном потоке — лишние воркеры только едят RAM
            num_workers=1,
            download_root=Path.home() / ".cache" / "whisper",
        )
        logger.info("Модель Whisper загружена")
        _model_cache[key] = model
        return model


def unload_models() -> None:
    """Выгрузить закэшированные модели и освободить память GPU.

    В долгоживущем процессе (трей) модели иначе держат несколько ГБ VRAM
    между встречами. Повторная загрузка займёт время при следующей обработке.
    """
    with _model_cache_lock:
        if not _model_cache:
            return
        _model_cache.clear()
    _free_gpu_memory()
    logger.info("Модели транскрипции выгружены из памяти")


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch может быть недоступен — не критично
        pass


def _patch_torchaudio_compat() -> None:
    """Совместимость pyannote.audio 3.x с torchaudio 2.6+.

    torchaudio 2.6+ убрал torchaudio.info / AudioMetaData / list_audio_backends.
    Pyannote использует их как публичный API — патчим через soundfile.
    """
    import torchaudio

    if hasattr(torchaudio, "AudioMetaData"):
        return  # уже есть — ничего не делаем

    from collections import namedtuple

    import soundfile as sf
    import torch

    AudioMetaData = namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
    )
    torchaudio.AudioMetaData = AudioMetaData

    def _info(path, backend=None, **kwargs):
        info = sf.info(str(path))
        return AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )

    def _load(path, frame_offset=0, num_frames=-1, normalize=True, **kwargs):

        start = frame_offset
        # soundfile использует -1 для «читать всё», None недопустим
        frames = -1 if (num_frames is None or num_frames < 0) else num_frames
        data, sr = sf.read(
            str(path), start=start, frames=frames, dtype="float32", always_2d=True
        )
        tensor = torch.from_numpy(data.T)
        return tensor, sr

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.info = _info
    torchaudio.load = _load
    torchaudio.list_audio_backends = _list_audio_backends


def _patch_hf_use_auth_token() -> None:
    """huggingface_hub 1.x убрал use_auth_token — патч для pyannote 3.x."""
    try:
        import inspect

        import huggingface_hub as hfh

        if "use_auth_token" in inspect.signature(hfh.hf_hub_download).parameters:
            return

        _orig = hfh.hf_hub_download

        def _patched(*args, use_auth_token=None, **kwargs):
            if use_auth_token is not None and "token" not in kwargs:
                kwargs["token"] = use_auth_token
            return _orig(*args, **kwargs)

        hfh.hf_hub_download = _patched
        try:
            import pyannote.audio.core.pipeline as _pa

            if hasattr(_pa, "hf_hub_download"):
                _pa.hf_hub_download = _patched
        except ImportError:
            pass
    except Exception as e:
        logger.debug("Патч hf_hub не применён: %s", e)


_torch_load_lock = threading.Lock()


@contextlib.contextmanager
def _trusted_torch_load():
    """PyTorch 2.6+: weights_only=True по умолчанию — чекпоинты pyannote не грузятся.

    На время блока torch.load получает weights_only=False (если вызывающий код
    не передал weights_only явно), после выхода исходная функция
    восстанавливается. pl_load в pyannote вызывает torch.load по атрибуту
    модуля, поэтому патчится именно torch.load.

    БЕЗОПАСНОСТЬ: weights_only=False допускает выполнение произвольного кода
    при загрузке недоверенных checkpoint-файлов. Окно ограничено загрузкой
    моделей pyannote из доверенного репозитория HuggingFace — остальной код
    процесса (в т.ч. долгоживущего трея) работает с безопасным умолчанием.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    with _torch_load_lock:
        original = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original(*args, **kwargs)

        torch.load = _patched_load
        try:
            yield
        finally:
            torch.load = original


def _patch_speechbrain_lazy_module() -> None:
    """speechbrain LazyModule.__getattr__ кидает ImportError для dunder-атрибутов.

    inspect.stack() → hasattr(module, '__file__') → ImportError
    (k2/flair/etc. не установлены).
    Патч: для дандер-атрибутов перехватываем ImportError и поднимаем AttributeError,
    чтобы hasattr() корректно вернул False.
    """
    try:
        from speechbrain.utils.importutils import LazyModule

        if getattr(LazyModule, "_patched_dunder", False):
            return

        _orig_getattr = LazyModule.__getattr__

        def _safe_getattr(self, attr):
            if attr.startswith("__") and attr.endswith("__"):
                try:
                    return _orig_getattr(self, attr)
                except (ImportError, Exception):
                    raise AttributeError(attr) from None
            return _orig_getattr(self, attr)

        LazyModule.__getattr__ = _safe_getattr
        LazyModule._patched_dunder = True
    except Exception as e:
        logger.debug("Патч speechbrain LazyModule не применён: %s", e)


def _get_diarization_model(cfg: AppConfig):
    """Загрузить pyannote.audio диаризацию с кэшированием (thread-safe).

    Лок удерживается на всё время загрузки — аналогично _get_transcribe_model.
    """
    key = "diarization"
    with _model_cache_lock:
        if key in _model_cache:
            return _model_cache[key]

        try:
            _patch_torchaudio_compat()
            _patch_hf_use_auth_token()
            _patch_speechbrain_lazy_module()
            from pyannote.audio import Pipeline
        except Exception as e:
            logger.warning("pyannote.audio недоступна — диаризация отключена: %s", e)
            return None

        hf_token = cfg.transcription.hf_token.get_secret_value()
        if not hf_token:
            logger.warning(
                "HF_TOKEN не указан — диаризация будет отключена. "
                "Установите токен на huggingface.co/settings/tokens"
            )
            return None

        logger.info("Загрузка pyannote.pipeline (device=%s)…", cfg.transcription.device)
        try:
            with _trusted_torch_load():
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
        except Exception as e:
            logger.warning("Не удалось загрузить pyannote pipeline: %s", e)
            _model_cache[key] = None  # кэшируем None, чтобы не повторять попытку
            return None
        try:
            import torch

            pipeline.to(torch.device(cfg.transcription.device))
        except Exception as e:
            logger.warning(
                "Не удалось перенести pyannote на %s: %s", cfg.transcription.device, e
            )
        _model_cache[key] = pipeline
        logger.info("pyannote.pipeline загружена")
        return pipeline


def _is_out_of_memory(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "alloc_failed" in msg


def _batch_fallbacks(batch_size: int) -> list[int]:
    """16 → [16, 8, 4, 0]: при OOM уменьшаем batch, в конце — без батчинга."""
    sizes: list[int] = []
    while batch_size >= _OOM_MIN_BATCH:
        sizes.append(batch_size)
        batch_size //= 2
    if batch_size > 0 and not sizes:
        sizes.append(batch_size)
    sizes.append(0)
    return sizes


def _whisper_pass(model, audio, batch_size: int, kwargs: dict):
    """Один проход Whisper. Сегменты собираются в список: faster-whisper
    возвращает генератор, и ошибки (в т.ч. OOM) возникают при итерации."""
    if batch_size > 0:
        # BatchedInferencePipeline: параллельная обработка чанков — 3-5x
        # быстрее на GPU без изменения качества (тот же beam search)
        from faster_whisper import BatchedInferencePipeline

        pipeline = BatchedInferencePipeline(model=model)
        logger.info("Используется BatchedInferencePipeline (batch_size=%d)", batch_size)
        segments_raw, info = pipeline.transcribe(audio, batch_size=batch_size, **kwargs)
    else:
        segments_raw, info = model.transcribe(audio, **kwargs)
    return list(segments_raw), info


def _transcribe_with_oom_fallback(model, audio, batch_size: int, kwargs: dict):
    sizes = _batch_fallbacks(batch_size) if batch_size > 0 else [0]
    for i, size in enumerate(sizes):
        try:
            return _whisper_pass(model, audio, size, kwargs)
        except Exception as exc:
            if not _is_out_of_memory(exc) or i == len(sizes) - 1:
                raise
            logger.warning(
                "Нехватка видеопамяти при batch_size=%d — повторяю с batch_size=%d",
                size,
                sizes[i + 1],
            )
            _free_gpu_memory()
    raise AssertionError("unreachable")  # pragma: no cover


def _decode_audio(audio_path: Path):
    """Декодировать аудио в моно float32 16 кГц (один раз для Whisper и pyannote).

    Возвращает None, если декодирование недоступно — тогда модели читают файл
    сами.
    """
    try:
        from faster_whisper import decode_audio

        return decode_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
    except Exception as exc:
        logger.debug("decode_audio недоступен (%s) — модели читают файл сами", exc)
        return None


class Segment:
    """Один сегмент транскрипта."""

    def __init__(self, start: float, end: float, speaker: str, text: str):
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": self.speaker,
            "text": self.text,
        }


def transcribe(
    audio_path: Path | str,
    cfg: AppConfig,
    output_path: Path | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Транскрибировать аудио (faster-whisper) с диаризацией (pyannote).

    Метки говорящих сохраняются «как есть» (SPEAKER_00, …): имена из
    ``transcription.speaker_names`` применяются при отображении, поэтому
    маппинг можно менять и перегенерировать отчёты без повторной транскрипции.

    Args:
        audio_path: Путь к WAV-файлу (микс или отдельная дорожка).
        cfg: Полная конфигурация приложения.
        output_path: Куда сохранить JSON (автоматически, если None).
        session_id: Идентификатор сессии для записи в JSON.

    Returns:
        Словарь с ключами: session_id, language, duration_sec, segments.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

    tcfg = cfg.transcription
    device = tcfg.device
    language = tcfg.language.strip().lower()

    # Проверка GPU
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("cuda запрошен, но GPU не найден — переключение на cpu")
                device = "cpu"
        except ImportError:
            device = "cpu"

    # Автоматический batch_size: 16 на CUDA, 0 (без батчинга) на CPU
    batch_size = tcfg.batch_size
    if batch_size == 0 and device == "cuda":
        batch_size = 16

    # Загрузка модели Whisper
    model = _get_transcribe_model(tcfg.model, device)

    logger.info(
        "Начинаю транскрипцию: %s (model=%s, language=%s, device=%s, "
        "beam_size=%d, batch_size=%d)",
        audio_path.name,
        tcfg.model,
        language,
        device,
        tcfg.beam_size,
        batch_size,
    )

    transcribe_kwargs: dict = dict(
        beam_size=tcfg.beam_size,
        language=None if language in ("", "auto") else language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=(
            "Это запись деловой встречи на русском языке." if language == "ru" else None
        ),
    )

    # Декодируем один раз: тот же массив (16 кГц моно, ~230 МБ на час) уходит
    # и в Whisper, и в pyannote — без повторного чтения 48 кГц файла.
    audio = _decode_audio(audio_path)
    audio_input = audio if audio is not None else str(audio_path)

    segments_raw, info = _transcribe_with_oom_fallback(
        model, audio_input, batch_size, transcribe_kwargs
    )

    # Собираем сегменты
    segments: list[Segment] = []
    for seg in segments_raw:
        segments.append(
            Segment(
                start=seg.start,
                end=seg.end,
                speaker="UNKNOWN",  # будет заполнено диаризацией
                text=seg.text.strip(),
            )
        )

    logger.info("Whisper: %d сегментов (язык: %s)", len(segments), info.language)

    # Диаризация
    diarization_applied = False
    if tcfg.diarization:
        pipeline = _get_diarization_model(cfg)
        if pipeline is not None:
            segments = _apply_diarization(
                pipeline, audio_path, segments, cfg, waveform=audio
            )
            diarization_applied = True
        else:
            logger.warning(
                "Диаризация пропущена — HF_TOKEN не задан или pyannote недоступна"
            )
    else:
        logger.info("Диаризация отключена в конфиге")

    duration = info.duration if hasattr(info, "duration") else 0.0
    detected_lang = info.language if hasattr(info, "language") else language

    result = {
        "session_id": session_id,
        "language": detected_lang,
        "duration_sec": round(duration, 2),
        "diarization_enabled": diarization_applied,
        "segments": [s.to_dict() for s in segments],
    }

    # Сохранение JSON
    if output_path is None:
        output_path = audio_path.with_name(
            audio_path.name.replace(".wav", "_transcript.json")
        )
    output_path = Path(output_path)
    save_transcript_data(output_path, result)
    logger.info("Транскрипт сохранён: %s (%d сегментов)", output_path, len(segments))

    return result


def _apply_diarization(
    pipeline,
    audio_path: Path,
    segments: list[Segment],
    cfg: AppConfig,
    waveform=None,
) -> list[Segment]:
    """Применить pyannote.audio диаризацию к сегментам Whisper.

    Args:
        waveform: уже декодированное аудио (моно float32, 16 кГц). Если задано,
            pyannote не перечитывает файл (экономия памяти и времени).
    """
    logger.info("Выполняю диаризацию на %s…", audio_path.name)

    if waveform is not None:
        import torch

        audio_in: Any = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": _SAMPLE_RATE,
        }
    else:
        audio_in = str(audio_path)
    diarization = pipeline(audio_in)

    turns = sorted(
        (
            (float(turn.start), float(turn.end), speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ),
        key=lambda t: t[0],
    )
    overlaps_by_seg = _speaker_overlaps(segments, turns)

    # Назначаем говорящего с максимальным перекрытием (детерминированно)
    for seg, overlaps in zip(segments, overlaps_by_seg, strict=True):
        seg.speaker = max(overlaps, key=overlaps.__getitem__) if overlaps else "UNKNOWN"

    # Нормализуем имена говорящих (SPEAKER_00, SPEAKER_01, …)
    unique_speakers = sorted({s.speaker for s in segments if s.speaker != "UNKNOWN"})
    speaker_rename: dict[str, str] = {}
    for idx, old in enumerate(unique_speakers):
        speaker_rename[old] = f"SPEAKER_{idx:02d}"

    for seg in segments:
        if seg.speaker in speaker_rename:
            seg.speaker = speaker_rename[seg.speaker]

    logger.info("Диаризация: %d уникальных говорящих", len(speaker_rename))
    return segments


def _speaker_overlaps(
    segments: list[Segment], turns: list[tuple[float, float, str]]
) -> list[dict[str, float]]:
    """Суммарное перекрытие каждого сегмента с репликами каждого говорящего.

    Линейный проход «скользящим окном» вместо O(сегменты × реплики): сегменты
    обходятся по возрастанию start; реплика попадает в окно, когда её start
    меньше конца сегмента, и выбывает, когда закончилась до начала сегмента
    (последующие сегменты начинаются не раньше — с ними она не пересечётся).
    """
    result: list[dict[str, float]] = [{} for _ in segments]
    order = sorted(range(len(segments)), key=lambda i: segments[i].start)
    active: list[tuple[float, float, str]] = []
    nxt = 0
    for i in order:
        seg = segments[i]
        while nxt < len(turns) and turns[nxt][0] < seg.end:
            active.append(turns[nxt])
            nxt += 1
        active = [t for t in active if t[1] > seg.start]
        acc = result[i]
        for t_start, t_end, speaker in active:
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > 0:
                acc[speaker] = acc.get(speaker, 0.0) + overlap
    return result


def load_transcript(path: Path | str) -> dict[str, Any]:
    """Загрузить транскрипт из JSON-файла."""
    return load_transcript_data(Path(path))
