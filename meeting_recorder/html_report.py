"""Генерация интерактивного HTML-протокола встречи с медиаплеером."""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import AppConfig
from .naming import SessionPaths
from .report import _format_timestamp, _session_datetime
from .transcriber import load_transcript

logger = logging.getLogger(__name__)

# Палитра для говорящих (до 10 уникальных)
_SPEAKER_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#9333ea",
    "#ea580c", "#0891b2", "#db2777", "#65a30d",
    "#b45309", "#0f766e",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _color_map(unique_speakers: list[str]) -> dict[str, str]:
    return {sp: _SPEAKER_COLORS[i % len(_SPEAKER_COLORS)] for i, sp in enumerate(unique_speakers)}


def _pick_media(paths: SessionPaths) -> tuple[str, str] | None:
    """Выбрать лучший доступный медиафайл.

    Приоритет: финальное видео (audio+video) → mix-аудио (есть звук) → сырое видео (без звука).
    Сырое видео gdigrab не содержит аудио, поэтому mix_audio предпочтительнее.
    Возвращает (filename, mime_type) или None.
    """
    if paths.final_video.exists():
        return paths.final_video.name, "video/mp4"
    if paths.mix_audio.exists():
        return paths.mix_audio.name, "audio/wav"
    if paths.video.exists():
        return paths.video.name, "video/mp4"
    return None


def _is_sep_row(line: str) -> bool:
    """True если строка — разделитель заголовка таблицы Markdown (|---|:--:|---:|)."""
    inner = line.strip()[1:-1]
    return bool(inner) and all(
        re.match(r"^:?-+:?$", cell.strip()) for cell in inner.split("|")
    )


def _md_to_html(text: str) -> str:
    """Минимальный Markdown → HTML: заголовки, таблицы, списки, жирный/курсив."""
    lines = text.split("\n")
    out: list[str] = []
    in_ul = False
    in_table = False
    table_has_header = False   # True после строки-разделителя |---|

    for raw in lines:
        stripped = raw.strip()

        # --- ТАБЛИЦА ---
        if stripped.startswith("|") and stripped.endswith("|"):
            if in_ul:
                out.append("</ul>")
                in_ul = False

            if _is_sep_row(stripped):
                if in_table and not table_has_header:
                    out.append("</thead><tbody>")
                    table_has_header = True
                continue

            cells = [_inline(html.escape(c.strip())) for c in stripped[1:-1].split("|")]
            if not in_table:
                out.append('<table class="md-table"><thead>')
                in_table = True
                table_has_header = False

            if not table_has_header:
                out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue

        # Закрыть таблицу при первой не-табличной строке
        if in_table:
            out.append("</thead></table>" if not table_has_header else "</tbody></table>")
            in_table = False
            table_has_header = False

        esc = html.escape(raw)

        # Заголовки (в контексте summary понижаем уровень на 2)
        for prefix, tag in (("### ", "h5"), ("## ", "h4"), ("# ", "h3")):
            if esc.startswith(prefix):
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                out.append(f"<{tag}>{_inline(esc[len(prefix):])}</{tag}>")
                break
        else:
            if esc.startswith("- ") or esc.startswith("* "):
                if not in_ul:
                    out.append("<ul>")
                    in_ul = True
                out.append(f"<li>{_inline(esc[2:])}</li>")
            elif esc.startswith("---"):
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                out.append("<hr>")
            elif esc.strip() == "":
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                out.append("")
            else:
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                out.append(f"<p>{_inline(esc)}</p>")

    if in_ul:
        out.append("</ul>")
    if in_table:
        out.append("</thead></table>" if not table_has_header else "</tbody></table>")
    return "\n".join(out)


def _inline(text: str) -> str:
    """Inline-форматирование: **bold** и *italic*."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def generate_html_protocol(
    transcript: dict[str, Any] | Path | str,
    paths: SessionPaths,
    cfg: AppConfig,
) -> Path:
    """Сгенерировать интерактивный HTML-протокол.

    Содержит HTML5-плеер (видео или аудио), кликабельные таймкоды,
    авто-подсветку текущего сегмента, фильтр по говорящим и поиск.

    Args:
        transcript: словарь или путь к JSON-транскрипту.
        paths:      SessionPaths текущей сессии.
        cfg:        конфигурация приложения.

    Returns:
        Путь к сохранённому *_protocol.html.
    """
    if isinstance(transcript, (Path, str)):
        data = load_transcript(transcript)
    else:
        data = transcript

    raw_segs = data.get("segments", [])
    speaker_names = cfg.transcription.speaker_names

    # Применяем переименование говорящих
    segments: list[dict[str, Any]] = []
    for seg in raw_segs:
        spk = seg.get("speaker", "UNKNOWN")
        segments.append({
            "start": seg["start"],
            "end":   seg["end"],
            "speaker": speaker_names.get(spk, spk),
            "text":  seg.get("text", "").strip(),
        })

    unique_speakers = sorted({s["speaker"] for s in segments})
    colors = _color_map(unique_speakers)

    meeting_dt  = _session_datetime(paths.session_id)
    date_str    = meeting_dt.strftime("%Y-%m-%d %H:%M")
    duration_s  = data.get("duration_sec", 0.0)

    # Попытка создать финальное видео (видео + аудио), если его ещё нет
    if not paths.final_video.exists() and paths.video.exists() and paths.mix_audio.exists():
        try:
            from .recorder import mux_video
            logger.info("Создаю финальное видео для HTML-протокола…")
            mux_video(paths.video, paths.mix_audio, paths.final_video)
            logger.info("Финальное видео готово: %s", paths.final_video)
        except Exception as exc:
            logger.warning("Не удалось создать финальное видео: %s", exc)

    media = _pick_media(paths)

    summary_html = ""
    if paths.summary.exists():
        summary_html = _md_to_html(paths.summary.read_text(encoding="utf-8"))

    content = _render(
        session_id=paths.session_id,
        date_str=date_str,
        duration_s=duration_s,
        segments=segments,
        colors=colors,
        unique_speakers=unique_speakers,
        media=media,
        summary_html=summary_html,
    )

    paths.html_protocol.write_text(content, encoding="utf-8")
    logger.info("HTML-протокол сохранён: %s", paths.html_protocol)
    return paths.html_protocol


# ---------------------------------------------------------------------------
# Рендер HTML
# ---------------------------------------------------------------------------

def _render(
    session_id: str,
    date_str: str,
    duration_s: float,
    segments: list[dict[str, Any]],
    colors: dict[str, str],
    unique_speakers: list[str],
    media: tuple[str, str] | None,
    summary_html: str,
) -> str:

    # --- медиаплеер ---
    if media:
        fname, mime = media
        tag = "video" if mime.startswith("video") else "audio"
        player_html = (
            f'<{tag} id="player" controls preload="metadata"'
            f' style="width:100%;display:block;max-height:420px">'
            f'<source src="{html.escape(fname)}" type="{html.escape(mime)}">'
            f'</{tag}>'
        )
        seek_js = (
            "function seek(t){"
            "var p=document.getElementById('player');"
            "p.currentTime=t;p.play();}"
        )
        highlight_js = _HIGHLIGHT_JS
    else:
        player_html = (
            '<p class="no-media">'
            "Медиафайл не найден — выполните <code>mrec mux</code> "
            "для создания финального видео."
            "</p>"
        )
        seek_js = "function seek(t){}"
        highlight_js = ""

    # --- блок summary ---
    if summary_html:
        summary_block = (
            '<details open><summary>&#128196; Summary</summary>'
            f'<div class="sum-body">{summary_html}</div></details>'
        )
    else:
        summary_block = ""

    # --- кнопки фильтра говорящих ---
    sp_buttons = "\n".join(
        f'<button class="sp-btn" '
        f'style="border-color:{colors[sp]};color:{colors[sp]}" '
        f'data-sp="{html.escape(sp)}" '
        f'onclick="toggleSp(this)">'
        f'{html.escape(sp)}</button>'
        for sp in unique_speakers
    )

    # --- строки транскрипта ---
    rows = []
    for seg in segments:
        t_start = seg["start"]
        t_end   = seg["end"]
        sp      = seg["speaker"]
        color   = colors.get(sp, "#333")
        rows.append(
            f'<div class="seg" '
            f'data-start="{t_start:.2f}" data-end="{t_end:.2f}" '
            f'data-sp="{html.escape(sp)}" '
            f'onclick="seek({t_start:.2f})">'
            f'<span class="ts">{_format_timestamp(t_start)}</span>'
            f'<span class="sp" style="color:{color}">{html.escape(sp)}</span>'
            f'<span class="tx">{html.escape(seg["text"])}</span>'
            f'</div>'
        )
    transcript_rows = "\n".join(rows)

    # </> предотвращают преждевременное закрытие </script> в JS-блоке
    speakers_json = (
        json.dumps(unique_speakers)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    duration_fmt  = _format_timestamp(duration_s)

    return _HTML_TEMPLATE.format(
        title=html.escape(f"Встреча — {date_str}"),
        h1=html.escape(f"Протокол встречи — {date_str}"),
        session_id=html.escape(session_id),
        duration=html.escape(duration_fmt),
        player_html=player_html,
        summary_block=summary_block,
        sp_buttons=sp_buttons,
        transcript_rows=transcript_rows,
        speakers_json=speakers_json,
        seek_js=seek_js,
        highlight_js=highlight_js,
    )


# ---------------------------------------------------------------------------
# JavaScript-фрагмент подсветки активного сегмента по timeupdate
# ---------------------------------------------------------------------------

_HIGHLIGHT_JS = """
var _player = document.getElementById('player');
if (_player) {
  var _lastSeg = null;
  var _userScrolling = false;
  var _scrollPauseTimer = null;

  /* Помечаем ручной скрол (wheel/touch/клавиши).
     programmatic scrollIntoView эти события НЕ генерирует,
     поэтому флаг не «застревает» в true после авто-скрола. */
  function _pauseAutoScroll() {
    _userScrolling = true;
    clearTimeout(_scrollPauseTimer);
    _scrollPauseTimer = setTimeout(function () { _userScrolling = false; }, 3000);
  }
  window.addEventListener('wheel', _pauseAutoScroll, {passive: true});
  window.addEventListener('touchmove', _pauseAutoScroll, {passive: true});
  window.addEventListener('keydown', function (e) {
    if (['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' '].indexOf(e.key) !== -1) {
      _pauseAutoScroll();
    }
  });

  _player.addEventListener('timeupdate', function () {
    var t = _player.currentTime;
    var active = null;
    document.querySelectorAll('.seg').forEach(function (el) {
      var s = parseFloat(el.dataset.start), e = parseFloat(el.dataset.end);
      var on = t >= s && t < e;
      el.classList.toggle('active', on);
      if (on) active = el;
    });
    /* Скролим только при смене сегмента и только если пользователь не скролит вручную */
    if (active && active !== _lastSeg && !_userScrolling) {
      active.scrollIntoView({block: 'nearest', behavior: 'smooth'});
    }
    _lastSeg = active;
  });
}
"""

# ---------------------------------------------------------------------------
# HTML-шаблон
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      background:#f0f4f8;color:#1a202c;line-height:1.55}}
.wrap{{max-width:980px;margin:0 auto;padding:24px 16px 60px}}
h1{{font-size:1.45rem;font-weight:700;margin-bottom:4px}}
.meta{{color:#718096;font-size:.83rem;margin-bottom:18px}}
/* player */
.player-box{{background:#111;border-radius:12px;overflow:hidden;
             margin-bottom:20px;box-shadow:0 4px 20px rgba(0,0,0,.25)}}
.no-media{{background:#fff3cd;border:1px solid #ffc107;border-radius:8px;
           padding:12px 16px;font-size:.9rem;margin-bottom:20px}}
/* summary */
details{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
         padding:16px;margin-bottom:20px}}
summary{{cursor:pointer;user-select:none;font-size:.95rem;font-weight:600}}
.sum-body{{margin-top:12px;font-size:.88rem;line-height:1.7}}
.sum-body h3,.sum-body h4,.sum-body h5{{margin:14px 0 4px;font-weight:700}}
.sum-body p{{margin-bottom:8px}}
.sum-body ul{{margin:4px 0 8px 20px}}
.sum-body li{{margin-bottom:3px}}
.sum-body hr{{border:none;border-top:1px solid #e2e8f0;margin:12px 0}}
/* tables in summary */
.md-table{{border-collapse:collapse;width:100%;font-size:.82rem;margin:10px 0}}
.md-table th,.md-table td{{border:1px solid #e2e8f0;padding:5px 10px;text-align:left;vertical-align:top}}
.md-table th{{background:#edf2f7;font-weight:700}}
.md-table tr:nth-child(even) td{{background:#f7fafc}}
/* controls */
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}
.search{{flex:1;min-width:180px;padding:7px 12px;border:1px solid #cbd5e0;
         border-radius:8px;font-size:.9rem;background:#fff}}
.search:focus{{outline:none;border-color:#4299e1;box-shadow:0 0 0 3px rgba(66,153,225,.2)}}
/* speaker filter */
.sp-filter{{display:flex;flex-wrap:wrap;gap:6px}}
.sp-btn{{border:2px solid;border-radius:20px;padding:3px 12px;font-size:.78rem;
         font-weight:600;cursor:pointer;background:transparent;transition:opacity .15s}}
.sp-btn.off{{opacity:.25}}
/* transcript */
.transcript{{display:flex;flex-direction:column;gap:1px}}
.seg{{display:grid;grid-template-columns:52px 130px 1fr;gap:8px;
      padding:8px 10px;border-radius:7px;cursor:pointer;
      transition:background .12s;align-items:baseline}}
.seg:hover{{background:#ebf4ff}}
.seg.active{{background:#ebf8ff;box-shadow:inset 3px 0 0 #3182ce}}
.seg.hidden{{display:none}}
.ts{{color:#a0aec0;font-size:.76rem;font-variant-numeric:tabular-nums;
     white-space:nowrap;text-decoration:underline dotted;text-underline-offset:2px}}
.sp{{font-weight:700;font-size:.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.tx{{font-size:.87rem;color:#2d3748;line-height:1.5}}
</style>
</head>
<body>
<div class="wrap">
  <h1>{h1}</h1>
  <p class="meta">{session_id}&nbsp;&nbsp;·&nbsp;&nbsp;Длительность:&nbsp;{duration}</p>

  <div class="player-box">{player_html}</div>

  {summary_block}

  <div class="controls">
    <input class="search" type="search" placeholder="Поиск по тексту…"
           oninput="filterAll()">
    <div class="sp-filter" id="sp-filter">{sp_buttons}</div>
  </div>

  <div class="transcript" id="transcript">
{transcript_rows}
  </div>
</div>

<script>
{seek_js}
{highlight_js}

var _activeSp = new Set({speakers_json});

function toggleSp(btn) {{
  var sp = btn.dataset.sp;
  if (_activeSp.has(sp)) {{ _activeSp.delete(sp); btn.classList.add('off'); }}
  else                   {{ _activeSp.add(sp);    btn.classList.remove('off'); }}
  filterAll();
}}

function filterAll() {{
  var q = (document.querySelector('.search').value || '').toLowerCase();
  document.querySelectorAll('.seg').forEach(function (el) {{
    if (!_activeSp.has(el.dataset.sp)) {{ el.classList.add('hidden'); return; }}
    var tx = el.querySelector('.tx').textContent.toLowerCase();
    el.classList.toggle('hidden', q.length > 0 && !tx.includes(q));
  }});
}}
</script>
</body>
</html>"""
