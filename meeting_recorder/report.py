"""Генерация протокола и summary-отчёта из транскрипта."""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import AppConfig
from .naming import SessionPaths
from .transcriber import load_transcript

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _session_datetime(session_id: str) -> datetime:
    """Разобрать дату/время из session_id формата meeting_YYYY-MM-DD_HH-MM-SS."""
    try:
        # "meeting_2026-06-05_14-30-12" → "2026-06-05_14-30-12"
        dt_str = session_id.split("_", 1)[1]
        return datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return datetime.now()


# ---------------------------------------------------------------------------
# Протокол
# ---------------------------------------------------------------------------


def generate_protocol(
    transcript: dict[str, Any] | Path | str,
    paths: SessionPaths,
    cfg: AppConfig,
) -> Path:
    """Сгенерировать протокол встречи в Markdown из транскрипта.

    Args:
        transcript: словарь транскрипта или путь к JSON.
        paths: SessionPaths текущей сессии.
        cfg: полная конфигурация.

    Returns:
        Путь к сохранённому *_protocol.md.
    """
    if isinstance(transcript, (Path, str)):
        data = load_transcript(transcript)
    else:
        data = transcript

    # Применяем имена говорящих
    speaker_names = cfg.transcription.speaker_names
    segments = []
    for seg in data.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        if speaker in speaker_names:
            speaker = speaker_names[speaker]
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"],
        })

    # Формируем Markdown
    date_str = _session_datetime(paths.session_id).strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"# Протокол встречи — {date_str}",
        "",
        f"**Сессия:** {paths.session_id}",
        f"**Язык:** {data.get('language', 'unknown')}",
        f"**Длительность:** {data.get('duration_sec', 0):.0f} сек",
        "",
        "---",
        "",
    ]

    for seg in segments:
        ts = _format_timestamp(seg["start"])
        lines.append(f"**[{ts}]** {seg['speaker']}: {seg['text']}")

    protocol_text = "\n".join(lines)
    paths.protocol.write_text(protocol_text, encoding="utf-8")
    logger.info("Протокол сохранён: %s", paths.protocol)

    # Опциональная чистка через LLM
    if cfg.llm.clean_protocol:
        protocol_text = _clean_protocol(protocol_text, cfg)
        paths.protocol.write_text(protocol_text, encoding="utf-8")
        logger.info("Протокол отшлифован через LLM: %s", paths.protocol)

    return paths.protocol


def _format_timestamp(seconds: float) -> str:
    """Преобразовать секунды в MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _clean_protocol(protocol_text: str, cfg: AppConfig) -> str:
    """Отшлифовать протокол через LLM."""
    from .llm_client import create_llm_client

    template = (_PROMPTS_DIR / "protocol_clean.md").read_text(encoding="utf-8")
    prompt = template.replace("{protocol_text}", protocol_text)

    with create_llm_client(cfg.llm) as client:
        cleaned = client.chat([
            {"role": "user", "content": prompt},
        ])

    # Убираем возможные маркеры markdown-кода, если LLM их добавил
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        cleaned = "\n".join(lines)

    return cleaned


# ---------------------------------------------------------------------------
# Summary-отчёт
# ---------------------------------------------------------------------------

_CONTEXT_TOKEN_LIMIT = 120_000  # примерный лимит для 70B-моделей
_WORDS_PER_TOKEN = 1.3


def generate_summary(
    transcript: dict[str, Any] | Path | str,
    paths: SessionPaths,
    cfg: AppConfig,
) -> Path:
    """Сгенерировать summary-отчёт из транскрипта через LLM.

    Реализует chunking + map-reduce для длинных встреч.
    """
    if isinstance(transcript, (Path, str)):
        data = load_transcript(transcript)
    else:
        data = transcript

    segments = data.get("segments", [])
    full_text = "\n".join(seg.get("text", "") for seg in segments)

    # Оцениваем, поместится ли текст в контекстное окно
    est_tokens = len(full_text.split()) * _WORDS_PER_TOKEN

    # Формируем метаданные
    meeting_dt = _session_datetime(paths.session_id)
    date_str = meeting_dt.strftime("%Y-%m-%d")
    time_str = meeting_dt.strftime("%H:%M")
    duration_min = data.get("duration_sec", 0) / 60.0

    unique_speakers = sorted({
        seg.get("speaker", "UNKNOWN") for seg in segments
    })
    speakers_str = ", ".join(unique_speakers) if unique_speakers else "неизвестно"

    template = (_PROMPTS_DIR / "summary.md").read_text(encoding="utf-8")
    metadata = template.format(
        date=date_str,
        time=time_str,
        duration=f"{duration_min:.0f}",
        speakers=speakers_str,
        transcript_text=full_text,
    )

    summary_text = _call_llm_for_summary(metadata, est_tokens, cfg)
    paths.summary.write_text(summary_text, encoding="utf-8")
    logger.info("Summary-отчёт сохранён: %s", paths.summary)
    return paths.summary


def _call_llm_for_summary(
    metadata: str,
    est_tokens: int,
    cfg: AppConfig,
) -> str:
    """Вызвать LLM для генерации summary, с chunking при необходимости."""
    from .llm_client import create_llm_client

    with create_llm_client(cfg.llm) as client:
        if est_tokens < _CONTEXT_TOKEN_LIMIT:
            # Простой вызов — текст помещается
            logger.info("Транскрипт помещается в контекстное окно (%d токенов)", est_tokens)
            return client.chat([
                {"role": "user", "content": metadata},
            ])
        else:
            # Chunking + map-reduce
            logger.info(
                "Транскрипт превышает контекстное окно (%d > %d) — применяю chunking",
                est_tokens, _CONTEXT_TOKEN_LIMIT,
            )
            return _map_reduce_summary(metadata, est_tokens, cfg, client)


def _map_reduce_summary(
    metadata: str,
    est_tokens: int,
    cfg: AppConfig,
    client: LLMClient,
) -> str:
    """Map-reduce: разбить на чанки → промежуточные резюме → агрегация."""
    # Определяем размер чанка
    chunk_size = int(_CONTEXT_TOKEN_LIMIT / _WORDS_PER_TOKEN)

    # Разбиваем транскрипт на чанки по тексту
    segments = []
    for line in metadata.split("\n"):
        if line.startswith("# Summary-отчёт") or line.startswith("---") or not line.strip():
            continue
        if "Транскрипт:" in line:
            continue
        if line.startswith("Транскрипт:"):
            continue
        segments.append(line)

    # Извлекаем текст транскрипта из метаданных
    transcript_text = _extract_transcript_text(metadata)
    chunks = _split_text_into_chunks(transcript_text, chunk_size)

    # Map: промежуточные резюме
    intermediate_summaries = []
    template = _get_summary_template_without_transcript(cfg)

    for i, chunk in enumerate(chunks):
        logger.info("Map chunk %d/%d", i + 1, len(chunks))
        chunk_metadata = metadata.replace(
            "{transcript_text}", chunk
        )
        chunk_summary = client.chat([
            {"role": "user", "content": chunk_metadata},
        ])
        intermediate_summaries.append(chunk_summary)
        logger.info("Chunk %d summary: %d chars", i + 1, len(chunk_summary))

    # Reduce: агрегация промежуточных резюме
    logger.info("Reduce: агрегация %d промежуточных резюме", len(intermediate_summaries))
    aggregated = "\n\n".join(intermediate_summaries)

    reduce_template = textwrap.dedent("""\
    Ты — аналитик деловых встреч. Агрегируй промежуточные резюме частей встречи в единый summary-отчёт.

    ## Метаданные
    {metadata}

    ## Промежуточные резюме
    {intermediate_summaries}

    ## Требования
    Сгенерируй единый markdown-отчёт той же структуры:
    - Краткое резюме (2-4 абзаца)
    - Ключевые обсуждённые темы
    - Принятые решения
    - Задачи (Action Items)
    - Открытые вопросы

    Возвращай ТОЛЬКО markdown, без комментариев.
    """)

    reduce_prompt = reduce_template.format(
        metadata=template,
        intermediate_summaries=aggregated,
    )
    return client.chat([
        {"role": "user", "content": reduce_prompt},
    ])


def _extract_transcript_text(metadata: str) -> str:
    """Извлечь текст транскрипта из метаданных."""
    # Ищем текст после строки "Транскрипт:" или "transcript_text:"
    for prefix in ("Транскрипт:", "transcript_text="):
        idx = metadata.find(prefix)
        if idx != -1:
            return metadata[idx + len(prefix):].strip()

    # fallback — берём всё, что после "---"
    idx = metadata.find("---")
    if idx != -1:
        return metadata[idx + 3:].strip()

    return metadata


def _split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """Разбить текст на чанки не больше max_chars, разрывая по абзацам."""
    chunks = []
    current = []
    current_len = 0

    for para in text.split("\n\n"):
        para_len = len(para)
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _get_summary_template_without_transcript(cfg: AppConfig) -> str:
    """Вернуть шаблон summary без вставки транскрипта (для reduce)."""
    template_path = _PROMPTS_DIR / "summary.md"
    text = template_path.read_text(encoding="utf-8")
    # Убираем блок с транскриптом
    text = text.replace("{transcript_text}", "")
    return text
