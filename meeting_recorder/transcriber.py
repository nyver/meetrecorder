"""Обёртка над WhisperX: аудио → сегменты с диаризацией."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from .config import AppConfig
from .naming import SessionPaths

logger = logging.getLogger(__name__)

# Кэш для моделей (singleton) + lock для thread-safety
_model_cache: dict[str, Any] = {}
_model_cache_lock = threading.Lock()


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

        logger.info("Загрузка модели Whisper (model=%s, device=%s, dtype=%s)…", model_name, device, dtype)
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=dtype or ("float16" if device == "cuda" else "int8"),
            num_workers=4,
            download_root=Path.home() / ".cache" / "whisper",
        )
        logger.info("Модель Whisper загружена")
        _model_cache[key] = model
        return model


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
        import numpy as np
        start = frame_offset
        # soundfile использует -1 для «читать всё», None недопустим
        frames = -1 if (num_frames is None or num_frames < 0) else num_frames
        data, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)
        return tensor, sr

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.info = _info
    torchaudio.load = _load
    torchaudio.list_audio_backends = _list_audio_backends


def _patch_hf_use_auth_token() -> None:
    """huggingface_hub 1.x убрал use_auth_token — патчим для совместимости с pyannote 3.x."""
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


def _patch_torch_load_compat() -> None:
    """PyTorch 2.6+ сменил weights_only=True по умолчанию — pyannote чекпоинты не грузятся.

    Патчим torch.load глобально: устанавливаем weights_only=False по умолчанию.
    pl_load в pyannote — прямая ссылка, поэтому патчить cloud_io бесполезно.

    БЕЗОПАСНОСТЬ: weights_only=False допускает выполнение произвольного кода при загрузке
    недоверенных checkpoint-файлов. Допустимо только для моделей из доверенных источников
    (HuggingFace + официальные репозитории pyannote).
    """
    try:
        import torch

        if getattr(torch, "_patched_weights_only", False):
            return

        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load
        torch._patched_weights_only = True
    except Exception as e:
        logger.debug("Патч torch.load не применён: %s", e)


def _patch_speechbrain_lazy_module() -> None:
    """speechbrain LazyModule.__getattr__ кидает ImportError для дандер-атрибутов (__file__ и др.)

    inspect.stack() → hasattr(module, '__file__') → ImportError (k2/flair/etc. не установлены).
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
                    raise AttributeError(attr)
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
            _patch_torch_load_compat()
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
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        try:
            import torch
            pipeline.to(torch.device(cfg.transcription.device))
        except Exception as e:
            logger.warning("Не удалось перенести pyannote на %s: %s", cfg.transcription.device, e)
        _model_cache[key] = pipeline
        logger.info("pyannote.pipeline загружена")
        return pipeline


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
) -> dict[str, Any]:
    """Транскрибировать аудио через WhisperX с диаризацией.

    Args:
        audio_path: Путь к WAV-файлу (микс или отдельная дорожка).
        cfg: Полная конфигурация приложения.
        output_path: Куда сохранить JSON (автоматически, если None).

    Returns:
        Словарь с ключами: session_id, language, duration_sec, segments.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

    tcfg = cfg.transcription
    device = tcfg.device
    language = tcfg.language

    # Проверка GPU
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "cuda запрошен, но GPU не найден — переключение на cpu"
                )
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
        "Начинаю транскрипцию: %s (model=%s, language=%s, device=%s, beam_size=%d, batch_size=%d)",
        audio_path.name, tcfg.model, language, device, tcfg.beam_size, batch_size,
    )

    transcribe_kwargs: dict = dict(
        beam_size=tcfg.beam_size,
        language=language if language else None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=(
            "Это запись деловой встречи на русском языке."
            if language == "ru"
            else ""
        ),
    )

    if batch_size > 0:
        # BatchedInferencePipeline: параллельная обработка чанков — 3-5x быстрее на GPU
        # без изменения качества (тот же алгоритм, тот же beam search)
        from faster_whisper import BatchedInferencePipeline
        pipeline = BatchedInferencePipeline(model=model)
        logger.info("Используется BatchedInferencePipeline (batch_size=%d)", batch_size)
        segments_raw, info = pipeline.transcribe(
            str(audio_path),
            batch_size=batch_size,
            **transcribe_kwargs,
        )
    else:
        segments_raw, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    # Собираем сегменты
    segments: list[Segment] = []
    for seg in segments_raw:
        segments.append(Segment(
            start=seg.start,
            end=seg.end,
            speaker="UNKNOWN",  # будет заполнено диаризацией
            text=seg.text.strip(),
        ))

    logger.info("Whisper: %d сегментов (язык: %s)", len(segments), info.language)

    # Диаризация
    diarization_applied = False
    if tcfg.diarization:
        pipeline = _get_diarization_model(cfg)
        if pipeline is not None:
            segments = _apply_diarization(pipeline, audio_path, segments, cfg)
            diarization_applied = True
        else:
            logger.warning("Диаризация пропущена — HF_TOKEN не задан или pyannote недоступна")
    else:
        logger.info("Диаризация отключена в конфиге")

    # Сопоставление имён
    speaker_names = tcfg.speaker_names
    if speaker_names:
        segments = _apply_speaker_names(segments, speaker_names)

    duration = info.duration if hasattr(info, "duration") else 0.0
    detected_lang = info.language if hasattr(info, "language") else language

    result = {
        "session_id": "",  # будет заполнено вызывающим
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
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Транскрипт сохранён: %s (%d сегментов)", output_path, len(segments))

    return result


def _apply_diarization(
    pipeline,
    audio_path: Path,
    segments: list[Segment],
    cfg: AppConfig,
) -> list[Segment]:
    """Применить pyannote.audio диаризацию к сегментам Whisper."""
    logger.info("Выполняю диаризацию на %s…", audio_path.name)

    # pyannote 3.x: передаём путь напрямую
    diarization = pipeline(str(audio_path))

    # Карта: segment_index → {speaker: суммарное перекрытие в секундах}
    speaker_map: dict[int, dict[str, float]] = {i: {} for i in range(len(segments))}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        t_start, t_end = turn.start, turn.end
        for i, seg in enumerate(segments):
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > 0:
                speaker_map[i][speaker] = speaker_map[i].get(speaker, 0.0) + overlap

    # Назначаем говорящего с максимальным перекрытием (детерминированно)
    for i, seg in enumerate(segments):
        overlaps = speaker_map.get(i, {})
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


def _apply_speaker_names(
    segments: list[Segment],
    speaker_names: dict[str, str],
) -> list[Segment]:
    """Заменить технические метки (SPEAKER_00) на имена из конфига."""
    for seg in segments:
        if seg.speaker in speaker_names:
            seg.speaker = speaker_names[seg.speaker]
    return segments


def load_transcript(path: Path | str) -> dict[str, Any]:
    """Загрузить транскрипт из JSON-файла."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Транскрипт не найден: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data
