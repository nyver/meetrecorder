"""Генерация протокола, summary-отчёта и ключевых моментов из транскрипта."""

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import AppConfig
from .naming import SessionPaths
from .session_utils import (
    atomic_write_text,
    load_transcript_data,
    normalize_highlights,
    normalize_segments,
    strip_code_fences,
    transcript_lines,
)
from .session_utils import (
    unique_speakers as _unique_speakers,
)

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Консервативная оценка: для кириллицы типичные BPE-токенизаторы (Qwen, Llama)
# дают ~2.5–3.5 символа на токен, для латиницы ~4. Берём нижнюю границу
# (больше токенов на тот же текст), чтобы не переполнить контекст.
_CHARS_PER_TOKEN = 2.5
# Если сервер всё же ответил «контекст переполнен» — повторяем с чанками
# меньшего размера (доля от расчётного бюджета).
_OVERFLOW_SCALES = (1.0, 0.5, 0.25)
_MIN_CHUNK_CHARS = 2000
_HIGHLIGHTS_COUNT = 5
_MAX_REDUCE_ROUNDS = 5
_TRUNCATED_NOTE = (
    "\n\n> ⚠️ Ответ модели обрезан по лимиту `llm.max_tokens` — "
    "увеличьте лимит и перегенерируйте отчёт (`mrec report`)."
)


def _session_datetime(session_id: str) -> datetime:
    """Разобрать дату/время из session_id формата meeting_YYYY-MM-DD_HH-MM-SS[_N]."""
    try:
        # "meeting_2026-06-05_14-30-12" или "meeting_2026-06-05_14-30-12_2"
        parts = session_id.split("_")
        dt_str = f"{parts[1]}_{parts[2]}"
        return datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return datetime.now()


def _format_timestamp(seconds: float) -> str:
    """Преобразовать секунды в MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _read_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Бюджет контекста и чанкинг
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Грубая (завышенная) оценка числа токенов в тексте."""
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _input_budget(cfg: AppConfig, scale: float = 1.0) -> int:
    return int(cfg.llm.input_token_budget() * scale)


def _chunk_chars(
    cfg: AppConfig,
    template: str = "",
    *,
    output_bound: bool = False,
    scale: float = 1.0,
) -> int:
    """Максимальный размер чанка (в символах), помещающийся в контекст модели.

    Args:
        template: шаблон промпта — его размер вычитается из бюджета.
        output_bound: ответ модели сопоставим по размеру со входом (чистка
            протокола) — чанк ограничивается ещё и лимитом max_tokens.
        scale: доля бюджета (< 1 — повтор после переполнения контекста).
    """
    budget = _input_budget(cfg, scale) - estimate_tokens(template)
    if output_bound:
        budget = min(budget, int(cfg.llm.max_tokens * 0.8 * scale))
    return max(int(budget * _CHARS_PER_TOKEN), _MIN_CHUNK_CHARS)


def _fits(text: str, cfg: AppConfig, template: str = "", scale: float = 1.0) -> bool:
    return estimate_tokens(template) + estimate_tokens(text) <= _input_budget(
        cfg, scale
    )


def _with_overflow_fallback(what: str, func):
    """Вызвать func(scale), уменьшая чанки при переполнении контекста модели."""
    from .llm_client import LLMContextOverflowError

    for i, scale in enumerate(_OVERFLOW_SCALES):
        try:
            return func(scale)
        except LLMContextOverflowError as exc:
            if i == len(_OVERFLOW_SCALES) - 1:
                raise
            logger.warning(
                "%s: контекст модели переполнен (%s) — повторяю с частями "
                "меньшего размера (%.0f%% бюджета). Проверьте llm.context_tokens.",
                what,
                exc,
                _OVERFLOW_SCALES[i + 1] * 100,
            )
    raise AssertionError("unreachable")  # pragma: no cover


def _pack(pieces: list[str], sep: str, max_chars: int) -> list[str]:
    """Жадно склеить куски в чанки не длиннее max_chars."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        add_len = len(piece) + (len(sep) if current else 0)
        if current and current_len + add_len > max_chars:
            chunks.append(sep.join(current))
            current, current_len = [piece], len(piece)
        else:
            current.append(piece)
            current_len += add_len
    if current:
        chunks.append(sep.join(current))
    return chunks


def _split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """Разбить текст на чанки не больше max_chars.

    Границы выбираются по убыванию приоритета: абзацы (``\\n\\n``), затем строки
    (``\\n`` — реплики транскрипта), и только для сверхдлинной строки — по символам.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    pieces: list[str] = []
    for para in text.split("\n\n"):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        lines: list[str] = []
        for line in para.split("\n"):
            if len(line) <= max_chars:
                lines.append(line)
            else:
                lines.extend(
                    line[i : i + max_chars] for i in range(0, len(line), max_chars)
                )
        pieces.extend(_pack(lines, "\n", max_chars))
    return _pack(pieces, "\n\n", max_chars) or [""]


# ---------------------------------------------------------------------------
# Протокол
# ---------------------------------------------------------------------------


def generate_protocol(
    transcript: dict[str, Any] | Path | str,
    paths: SessionPaths,
    cfg: AppConfig,
) -> Path:
    """Сгенерировать протокол встречи в Markdown из транскрипта.

    Returns:
        Путь к сохранённому *_protocol.md.
    """
    data = load_transcript_data(transcript)
    segments = normalize_segments(
        data.get("segments", []), cfg.transcription.speaker_names
    )

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
        lines.append(
            f"**[{_format_timestamp(seg['start'])}]** {seg['speaker']}: {seg['text']}"
        )

    protocol_text = "\n".join(lines)
    atomic_write_text(paths.protocol, protocol_text)
    logger.info("Протокол сохранён: %s", paths.protocol)

    if cfg.llm.clean_protocol:
        protocol_text = _clean_protocol(protocol_text, cfg)
        atomic_write_text(paths.protocol, protocol_text)
        logger.info("Протокол отшлифован через LLM: %s", paths.protocol)

    return paths.protocol


def _clean_protocol(protocol_text: str, cfg: AppConfig) -> str:
    """Отшлифовать протокол через LLM по частям.

    Ответ модели сопоставим по объёму со входом, поэтому протокол режется на
    части, укладывающиеся и в контекст, и в max_tokens. Если ответ по части
    пустой или обрезан — эта часть остаётся в исходном виде (без потери данных).
    """
    from .llm_client import LLMClientError, LLMTruncatedError, create_llm_client

    template = _read_prompt("protocol_clean.md")
    chunks = _split_text_into_chunks(
        protocol_text, _chunk_chars(cfg, template, output_bound=True)
    )
    result: list[str] = []
    with create_llm_client(cfg.llm) as client:
        for i, chunk in enumerate(chunks, 1):
            prompt = template.replace("{protocol_text}", chunk)
            try:
                cleaned = client.chat([{"role": "user", "content": prompt}])
            except LLMTruncatedError:
                logger.warning(
                    "Чистка протокола: часть %d/%d обрезана — оставлен оригинал",
                    i,
                    len(chunks),
                )
                cleaned = ""
            except LLMClientError as exc:
                logger.warning(
                    "Чистка протокола: часть %d/%d — ошибка LLM (%s), "
                    "оставлен оригинал",
                    i,
                    len(chunks),
                    exc,
                )
                cleaned = ""
            if not cleaned or not cleaned.strip():
                if len(chunks) == 1:
                    logger.warning(
                        "LLM вернул пустой ответ при чистке протокола — "
                        "используется оригинал"
                    )
                result.append(chunk)
            else:
                result.append(strip_code_fences(cleaned))
    return "\n\n".join(result)


# ---------------------------------------------------------------------------
# Summary-отчёт
# ---------------------------------------------------------------------------


def _fill_template(template: str, meta: dict, transcript_text: str) -> str:
    """Подставить метаданные и текст транскрипта в шаблон промпта.

    Использует str.replace вместо str.format, чтобы не падать на фигурных
    скобках в тексте транскрипта (обсуждение JSON, кода, шаблонов и т.п.).
    """
    return (
        template.replace("{date}", meta.get("date", ""))
        .replace("{time}", meta.get("time", ""))
        .replace("{duration}", meta.get("duration", ""))
        .replace("{speakers}", meta.get("speakers", ""))
        .replace("{transcript_text}", transcript_text)
    )


def _chat_allow_truncated(client: LLMClient, prompt: str) -> str:
    """Вызов LLM, при обрезке ответа возвращающий частичный текст с пометкой."""
    from .llm_client import LLMTruncatedError

    try:
        return client.chat([{"role": "user", "content": prompt}])
    except LLMTruncatedError as exc:
        if not exc.partial.strip():
            raise
        logger.warning("Ответ LLM обрезан по max_tokens — сохраняю частичный результат")
        return exc.partial + _TRUNCATED_NOTE


def generate_summary(
    transcript: dict[str, Any] | Path | str,
    paths: SessionPaths,
    cfg: AppConfig,
) -> Path:
    """Сгенерировать summary-отчёт из транскрипта через LLM.

    Для длинных встреч применяется chunking + map-reduce.
    """
    data = load_transcript_data(transcript)

    segments = data.get("segments", [])
    lines = transcript_lines(
        segments,
        speaker_names=cfg.transcription.speaker_names,
        timestamp_formatter=_format_timestamp,
    )
    full_text = "\n".join(lines)
    est_tokens = estimate_tokens(full_text)

    meeting_dt = _session_datetime(paths.session_id)
    duration_min = data.get("duration_sec", 0) / 60.0
    unique_speakers = _unique_speakers(segments, cfg.transcription.speaker_names)

    meta = {
        "date": meeting_dt.strftime("%Y-%m-%d"),
        "time": meeting_dt.strftime("%H:%M"),
        "duration": f"{duration_min:.0f}",
        "speakers": ", ".join(unique_speakers) if unique_speakers else "неизвестно",
    }
    template = _read_prompt("summary.md")

    summary_text = _call_llm_for_summary(full_text, template, meta, est_tokens, cfg)
    if not summary_text or not summary_text.strip():
        raise RuntimeError(
            "LLM вернул пустой ответ при генерации summary — проверьте настройки модели"
        )
    atomic_write_text(paths.summary, summary_text)
    logger.info("Summary-отчёт сохранён: %s", paths.summary)
    return paths.summary


def _call_llm_for_summary(
    full_text: str,
    template: str,
    meta: dict,
    est_tokens: int,
    cfg: AppConfig,
) -> str:
    """Вызвать LLM для генерации summary, с chunking при необходимости."""
    from .llm_client import create_llm_client

    with create_llm_client(cfg.llm) as client:

        def attempt(scale: float) -> str:
            budget = _input_budget(cfg, scale)
            if est_tokens + estimate_tokens(template) <= budget:
                logger.info(
                    "Транскрипт помещается в контекстное окно (~%d из %d токенов)",
                    est_tokens,
                    budget,
                )
                return _chat_allow_truncated(
                    client, _fill_template(template, meta, full_text)
                )
            logger.info(
                "Транскрипт превышает контекстное окно (~%d > %d токенов) — "
                "применяю chunking",
                est_tokens,
                budget,
            )
            return _map_reduce_summary(
                full_text, template, meta, cfg, client, scale=scale
            )

        return _with_overflow_fallback("Summary", attempt)


_REDUCE_PROMPT = textwrap.dedent("""\
    Ты — аналитик деловых встреч. Агрегируй промежуточные резюме частей встречи
    в единый summary-отчёт.

    ## Метаданные
    REPLACE_META

    ## Промежуточные резюме
    REPLACE_SUMMARIES

    ## Требования
    Сгенерируй единый markdown-отчёт той же структуры:
    - Краткое резюме (2-4 абзаца)
    - Ключевые обсуждённые темы
    - Принятые решения
    - Задачи (Action Items)
    - Открытые вопросы

    Возвращай ТОЛЬКО markdown, без комментариев.
    """)


def _map_reduce_summary(
    full_text: str,
    template: str,
    meta: dict,
    cfg: AppConfig,
    client: LLMClient,
    scale: float = 1.0,
) -> str:
    """Map-reduce: разбить на чанки → промежуточные резюме → агрегация.

    Если промежуточные резюме сами не помещаются в контекст, reduce выполняется
    в несколько раундов (иерархически).
    """
    chunks = _split_text_into_chunks(
        full_text, _chunk_chars(cfg, template, scale=scale)
    )

    intermediate: list[str] = []
    for i, chunk in enumerate(chunks):
        logger.info("Map chunk %d/%d", i + 1, len(chunks))
        chunk_summary = _chat_allow_truncated(
            client, _fill_template(template, meta, chunk)
        )
        if chunk_summary and chunk_summary.strip():
            intermediate.append(chunk_summary)
            logger.info("Chunk %d summary: %d chars", i + 1, len(chunk_summary))
        else:
            logger.warning("Chunk %d: LLM вернул пустой ответ, пропускаю", i + 1)

    if not intermediate:
        raise RuntimeError(
            "Все промежуточные резюме оказались пустыми — LLM не вернул результат"
        )

    meta_str = "\n".join(f"{k}: {v}" for k, v in meta.items())
    base_prompt = _REDUCE_PROMPT.replace("REPLACE_META", meta_str)

    def _reduce(parts: list[str]) -> str:
        prompt = base_prompt.replace("REPLACE_SUMMARIES", "\n\n".join(parts))
        return _chat_allow_truncated(client, prompt)

    for round_no in range(1, _MAX_REDUCE_ROUNDS + 1):
        aggregated = "\n\n".join(intermediate)
        if len(intermediate) == 1 and round_no > 1:
            return intermediate[0]
        if _fits(aggregated, cfg, base_prompt, scale):
            logger.info("Reduce: агрегация %d промежуточных резюме", len(intermediate))
            return _reduce(intermediate)
        groups = _pack(
            intermediate, "\n\n", _chunk_chars(cfg, base_prompt, scale=scale)
        )
        logger.info(
            "Reduce раунд %d: %d резюме → %d групп",
            round_no,
            len(intermediate),
            len(groups),
        )
        intermediate = [r for r in (_reduce([g]) for g in groups) if r and r.strip()]
        if not intermediate:
            raise RuntimeError("LLM не вернул результат на этапе reduce")

    return _reduce(intermediate)


# ---------------------------------------------------------------------------
# Ключевые моменты (Highlights)
# ---------------------------------------------------------------------------

_HIGHLIGHTS_SELECT_PROMPT = textwrap.dedent("""\
    Ниже — JSON-массив кандидатов в ключевые моменты встречи, найденных
    в разных её частях.
    Выбери ровно {count} самых важных и верни их ТОЛЬКО как JSON-массив тех же объектов
    (поля "title", "description", "start" без изменений), упорядоченный по "start".
    Без markdown-обёрток и комментариев.

    {candidates}
    """)


def _parse_highlights_response(response: str) -> list[dict[str, Any]]:
    """Разобрать JSON-массив highlights из ответа LLM."""
    text = strip_code_fences(response or "")
    if not text:
        raise ValueError("LLM вернул пустой ответ для highlights")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Модель могла добавить пояснение вокруг массива — вырезаем [...]
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            raise ValueError("LLM вернул не массив для highlights") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM вернул некорректный JSON для highlights: {exc}"
            ) from exc
    if not isinstance(parsed, list):
        raise ValueError("LLM вернул не массив для highlights")
    return normalize_highlights(parsed)


def _ask_highlights(
    client: LLMClient, template: str, text: str
) -> list[dict[str, Any]]:
    response = client.chat(
        [{"role": "user", "content": template.replace("{transcript_text}", text)}]
    )
    return _parse_highlights_response(response)


def _select_top_highlights(
    client: LLMClient, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Отобрать _HIGHLIGHTS_COUNT лучших кандидатов (LLM + запасной вариант)."""
    from .llm_client import LLMClientError

    candidates = sorted(candidates, key=lambda h: h["start"])
    if len(candidates) <= _HIGHLIGHTS_COUNT:
        return candidates
    prompt = _HIGHLIGHTS_SELECT_PROMPT.format(
        count=_HIGHLIGHTS_COUNT,
        candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
    )
    try:
        selected = _parse_highlights_response(
            client.chat([{"role": "user", "content": prompt}])
        )
        if selected:
            return sorted(selected, key=lambda h: h["start"])[:_HIGHLIGHTS_COUNT]
    except (ValueError, LLMClientError) as exc:
        logger.warning("Не удалось отобрать highlights через LLM: %s", exc)
    # Запасной вариант: равномерно по времени встречи
    step = len(candidates) / _HIGHLIGHTS_COUNT
    return [candidates[int(i * step)] for i in range(_HIGHLIGHTS_COUNT)]


def _highlights_attempt(
    client: LLMClient,
    template: str,
    transcript_text: str,
    cfg: AppConfig,
    scale: float,
) -> list[dict[str, Any]]:
    """Одна попытка генерации highlights при заданной доле бюджета контекста."""
    from .llm_client import LLMClientError, LLMContextOverflowError

    if _fits(transcript_text, cfg, template, scale):
        return _ask_highlights(client, template, transcript_text)

    chunks = _split_text_into_chunks(
        transcript_text, _chunk_chars(cfg, template, scale=scale)
    )
    logger.info("Highlights: транскрипт длинный — обрабатываю %d частей", len(chunks))
    candidates: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, 1):
        try:
            candidates.extend(_ask_highlights(client, template, chunk))
        except LLMContextOverflowError:
            raise  # повтор с более мелкими частями — в _with_overflow_fallback
        except (ValueError, LLMClientError) as exc:
            logger.warning("Highlights: часть %d/%d пропущена: %s", i, len(chunks), exc)
    if not candidates:
        raise ValueError("LLM не вернул ни одного ключевого момента")
    return _select_top_highlights(client, candidates)


def generate_highlights(
    transcript: dict[str, Any] | Path | str,
    paths: SessionPaths,
    cfg: AppConfig,
) -> Path:
    """Сгенерировать ключевые моменты встречи через LLM.

    Сохраняет JSON-список [{title, description, start}] в *_highlights.json.
    Длинные транскрипты обрабатываются по частям с последующим отбором лучших.
    """
    from .llm_client import create_llm_client

    data = load_transcript_data(transcript)
    lines = transcript_lines(
        data.get("segments", []),
        speaker_names=cfg.transcription.speaker_names,
        timestamp_formatter=_format_timestamp,
    )
    transcript_text = "\n".join(lines)
    template = _read_prompt("highlights.md")

    with create_llm_client(cfg.llm) as client:
        highlights = _with_overflow_fallback(
            "Highlights",
            lambda scale: _highlights_attempt(
                client, template, transcript_text, cfg, scale
            ),
        )

    atomic_write_text(
        paths.highlights, json.dumps(highlights, ensure_ascii=False, indent=2)
    )
    logger.info("Ключевые моменты сохранены: %s", paths.highlights)
    return paths.highlights
