"""CLI-точка входа для Meeting Recorder."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import IO

# Принудительно UTF-8 на Windows (иначе кириллица ломается в консоли)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import click  # noqa: E402

from meeting_recorder import __version__  # noqa: E402
from meeting_recorder import state as _state  # noqa: E402
from meeting_recorder.config import AppConfig, load_config, resolve_config_path  # noqa: E402
from meeting_recorder.naming import list_sessions, resolve_session  # noqa: E402
from meeting_recorder.pipeline import (  # noqa: E402
    PipelineError,
    _rename_with_retry,  # noqa: F401  (реэкспорт для обратной совместимости)
    finalize_recording,
    process_session,
    run_highlights_only,
    run_html,
    run_mux,
    run_process,
    run_report_only,
)

logger = logging.getLogger("meeting_recorder")

# Ошибки, которые CLI-команды показывают пользователю без traceback
_USER_ERRORS = (PipelineError, FileNotFoundError)

# ---------------------------------------------------------------------------
# Состояние записи (persistence на диск)
# ---------------------------------------------------------------------------


def _app_data_dir() -> Path:
    return _state.app_data_dir()


def _default_state_dir() -> Path:
    """Вернуть user-writable директорию для файлов состояния."""
    return _state.default_state_dir()


def _default_log_file() -> Path:
    """Путь к лог-файлу; не зависит от cwd (важно для автозапуска трея)."""
    override = os.environ.get("MEETING_RECORDER_LOG_FILE")
    if override:
        return Path(override)
    return _app_data_dir() / "logs" / "meeting_recorder.log"


_STATE_DIR = _state.STATE_DIR
_STATE_FILE = _state.STATE_FILE
_STOP_FILE = _state.STOP_FILE
_DONE_FILE = _state.DONE_FILE
_LOCK_FILE = _state.LOCK_FILE

# Ожидание завершения `mrec start` после stop-сигнала. С запасом больше худшего
# случая MeetingRecorder.stop(): graceful 10 + terminate 5 + kill 5 + поток
# системного звука 10 = 30 с.
_STOP_WAIT_SECS = 45


def _ensure_state_dir() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def _save_state(
    session_id: str, video_path: str, ffmpeg_pid: int | None = None
) -> None:
    _state.save_state(session_id, video_path, ffmpeg_pid, path=_STATE_FILE)


def _load_state() -> dict | None:
    return _state.load_state(_STATE_FILE)


def _remove_state() -> None:
    _state.remove_state(_STATE_FILE)


# ---------------------------------------------------------------------------
# Lock записи (общий с треем, см. meeting_recorder.state)
# ---------------------------------------------------------------------------

_try_lock = _state.try_lock
_unlock = _state.unlock


def _acquire_start_lock() -> IO | None:
    """Захватить lock записи. Возвращает дескриптор или None, если lock занят."""
    return _state.acquire_recording_lock(_LOCK_FILE)


def _release_start_lock(fh: IO | None) -> None:
    _state.release_recording_lock(fh)


def _is_recorder_alive() -> bool:
    """True, если процесс записи (`mrec start` или трей) сейчас работает."""
    return _state.is_recording_active(_LOCK_FILE)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Настроить логирование (console + ротируемый файл)."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers: list[logging.Handler] = []
    if sys.stdout is not None:  # None под pythonw / --noconsole
        handlers.append(logging.StreamHandler(sys.stdout))
    log_file = _default_log_file()
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        )
    except OSError:
        pass
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def _subprocess_kwargs() -> dict:
    from meeting_recorder.recorder import no_window_kwargs

    return no_window_kwargs()


def _is_ffmpeg_process(pid: int) -> bool:
    """Проверить, что PID принадлежит ffmpeg (защита от переиспользования PID)."""
    import subprocess as sp

    try:
        if sys.platform == "win32":
            result = sp.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                **_subprocess_kwargs(),
            )
        else:
            result = sp.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True,
                text=True,
                timeout=5,
            )
    except Exception as exc:
        logger.debug("Не удалось проверить процесс PID=%s: %s", pid, exc)
        return False
    out = (result.stdout or "").lower()
    return "ffmpeg" in out and (sys.platform != "win32" or str(pid) in out)


def _stop_ffmpeg_graceful(pid: int, timeout: int = 15) -> bool:
    """Graceful stop ffmpeg по PID через CTRL_BREAK_EVENT, затем ждём завершения.

    Перед отправкой сигнала проверяется, что PID принадлежит ffmpeg: state-файл
    мог устареть, а Windows переиспользует PID — без проверки можно убить
    посторонний процесс.

    Returns:
        True — процесс завершился (или его уже нет); False — пришлось убить.
    """
    import signal
    import subprocess as sp

    if not _is_ffmpeg_process(pid):
        logger.info("PID=%s не является ffmpeg (процесс завершён) — пропускаю", pid)
        return True

    try:
        if sys.platform == "win32":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGTERM)
        logger.debug("Сигнал остановки отправлен ffmpeg PID=%s", pid)
    except (ProcessLookupError, PermissionError):
        logger.info("ffmpeg PID=%s уже завершён", pid)
        return True
    except Exception as e:
        logger.debug("Ошибка отправки сигнала PID=%s: %s", pid, e)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_ffmpeg_process(pid):
            logger.info("ffmpeg PID=%s завершился штатно", pid)
            return True
        time.sleep(0.5)

    logger.warning(
        "ffmpeg PID=%s не завершился за %ds — принудительное завершение", pid, timeout
    )
    try:
        if sys.platform == "win32":
            sp.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
                **_subprocess_kwargs(),
            )
        else:
            os.kill(pid, signal.SIGKILL)
        time.sleep(1.5)  # Дать ОС время закрыть файловые хэндлы
    except Exception as e:
        logger.error("Принудительное завершение ffmpeg PID=%s не удалось: %s", pid, e)
    return False


# ---------------------------------------------------------------------------
# Команды CLI
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="mrec")
@click.option("-v", "--verbose", is_flag=True, help="Подробное логирование")
@click.option(
    "-c", "--config", "config_path", type=click.Path(), help="Путь к config.yaml"
)
@click.pass_context
def cli(ctx, verbose, config_path):
    """Meeting Recorder — запись экрана, транскрипция и генерация отчётов."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path

    setup_logging(verbose)

    _t0 = time.monotonic()

    def _log_elapsed() -> None:
        elapsed = time.monotonic() - _t0
        mins, secs = divmod(int(elapsed), 60)
        fmt = f"{mins} мин {secs} сек" if mins else f"{elapsed:.1f} сек"
        logger.info("Время выполнения команды: %s", fmt)

    ctx.call_on_close(_log_elapsed)

    try:
        cfg = load_config(config_path)
    except Exception as e:
        logger.error("Ошибка загрузки config.yaml: %s", e)
        sys.exit(1)

    ctx.obj["cfg"] = cfg


@cli.command()
@click.pass_context
def start(ctx):
    """Начать запись экрана и звука."""
    cfg = ctx.obj["cfg"]

    existing = _load_state()
    if existing:
        sid = existing.get("session_id", "?")
        if _is_recorder_alive():
            print(f"⚠️ Уже идёт запись сессии: {sid}")
            print("   Остановите её командой: mrec stop")
            return
        # Процесс записи не работает — state-файл остался от аварийного завершения
        logger.warning("Найдено устаревшее состояние записи (%s) — очищаю", sid)
        print(f"⚠️ Предыдущая запись ({sid}) была прервана аварийно.")
        print(f"   Обработать её можно командой: mrec process {sid}")
        _remove_state()

    from meeting_recorder.naming import create_session
    from meeting_recorder.recorder import MeetingRecorder

    lock = _acquire_start_lock()
    if lock is None:
        print("⚠️ Уже идёт запись (другой процесс mrec start или трей).")
        return

    recorder = None
    paths = None
    try:
        _STOP_FILE.unlink(missing_ok=True)
        _DONE_FILE.unlink(missing_ok=True)

        paths = create_session(cfg.output_dir)
        recorder = MeetingRecorder(cfg, paths)
        recorder.start()

        ffmpeg_pid = recorder.ffmpeg_pid
        _save_state(paths.session_id, str(paths.video), ffmpeg_pid)
        logger.info("Запись начата: %s (ffmpeg PID=%s)", paths.session_id, ffmpeg_pid)
        print(f"\n✅ Запись начата: {paths.session_id}")
        print(f"   Папка: {paths.dir}")
        print("   Остановить: mrec stop  (или Ctrl+C)\n")

        while not _STOP_FILE.exists():
            time.sleep(0.5)
        logger.info("Stop-сигнал получен — останавливаю запись")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — останавливаю запись")
    except Exception as exc:
        logger.error("Ошибка запуска записи: %s", exc)
        print(f"\n❌ Ошибка запуска записи: {exc}\n")
    finally:
        try:
            if recorder is not None and recorder.is_recording:
                recorder.stop()
                logger.info("Запись завершена: %s", paths.session_id)
        finally:
            _DONE_FILE.touch()
            _STOP_FILE.unlink(missing_ok=True)
            _remove_state()
            _release_start_lock(lock)


def _signal_stop_and_wait(ffmpeg_pid: int | None) -> None:
    """Послать stop-сигнал процессу mrec start и дождаться завершения записи.

    Если `mrec start` уже не работает (аварийное завершение), ждать нечего —
    останавливаем осиротевший ffmpeg напрямую (после проверки PID).
    """
    _ensure_state_dir()
    _DONE_FILE.unlink(missing_ok=True)

    if not _is_recorder_alive():
        logger.warning("Процесс mrec start не запущен — состояние записи устарело")
        if ffmpeg_pid:
            _stop_ffmpeg_graceful(int(ffmpeg_pid))
        _remove_state()
        return

    _STOP_FILE.touch()
    print("⏳ Ожидаю завершения записи…")
    deadline = time.monotonic() + _STOP_WAIT_SECS
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if _DONE_FILE.exists():
            logger.info("Запись завершена штатно")
            break
    else:
        logger.warning("Таймаут ожидания mrec start — принудительная остановка ffmpeg")
        if ffmpeg_pid:
            _stop_ffmpeg_graceful(int(ffmpeg_pid))
        _remove_state()

    _DONE_FILE.unlink(missing_ok=True)
    _STOP_FILE.unlink(missing_ok=True)


def _mix_session_audio(paths, cfg) -> bool:
    """Свести аудиодорожки сессии. Возвращает True при успехе."""
    print("🔊 Свожу аудио…")
    try:
        finalize_recording(cfg, paths)
    except PipelineError as e:
        logger.warning("%s", e)
        return False
    return True


def _print_ffmpeg_log_tail(paths, lines: int = 30) -> None:
    ffmpeg_log = paths.ffmpeg_log
    if ffmpeg_log.exists():
        print(f"   Лог ffmpeg ({ffmpeg_log.name}):")
        tail = ffmpeg_log.read_text(encoding="utf-8", errors="replace").splitlines()[
            -lines:
        ]
        for line in tail:
            print(f"     {line}")
    else:
        print("   Проверьте mic_device и system_audio_* в config.yaml")
    print(f"   Папка сессии: {paths.dir}")


def _load_active_session(cfg) -> tuple[str | None, int | None, object | None]:
    """Загрузить активную сессию. Возвращает (session_id, ffmpeg_pid, paths)."""
    state = _load_state()
    if state is None:
        print("⚠️ Запись не идёт. Нечего останавливать.")
        return None, None, None

    session_id = state.get("session_id")
    ffmpeg_pid = state.get("ffmpeg_pid")

    if state.get("owner") == _state.OWNER_TRAY and _is_recorder_alive():
        print(f"⚠️ Запись {session_id} ведётся из трея — остановите её в меню трея.")
        return None, None, None

    if not session_id:
        print("⚠️ Состояние записи повреждено. Удалите .state/active_session.json")
        _remove_state()
        return None, None, None

    print(f"\n⏹ Останавливаю запись сессии: {session_id}…")

    try:
        paths = resolve_session(cfg.output_dir, session_id)
    except FileNotFoundError as e:
        print(f"❌ Сессия не найдена: {e}")
        _remove_state()
        return None, None, None

    return session_id, ffmpeg_pid, paths


@cli.command("stop-only")
@click.pass_context
def stop_only_cmd(ctx):
    """Остановить запись и свести аудио — без транскрипции и отчёта."""
    cfg = ctx.obj["cfg"]

    session_id, ffmpeg_pid, paths = _load_active_session(cfg)
    if paths is None:
        return

    _signal_stop_and_wait(ffmpeg_pid)

    if not _mix_session_audio(paths, cfg):
        print("\n❌ Аудиофайлы не созданы.")
        _print_ffmpeg_log_tail(paths, lines=20)
        return

    print(f"\n✅ Запись остановлена: {paths.dir}")
    print(f"   Чтобы обработать позже: mrec process {session_id}")


@cli.command()
@click.pass_context
def stop(ctx):
    """Остановить запись и запустить полный пайплайн."""
    cfg = ctx.obj["cfg"]

    session_id, ffmpeg_pid, paths = _load_active_session(cfg)
    if paths is None:
        return

    _signal_stop_and_wait(ffmpeg_pid)

    if not _mix_session_audio(paths, cfg):
        print("\n❌ Аудиофайлы не созданы — ffmpeg не смог записать данные.")
        _print_ffmpeg_log_tail(paths)
        return

    print(
        "\n🚀 Запускаю пайплайн: транскрипция → протокол → summary → "
        "highlights → HTML…\n"
    )
    try:
        _, protocol_path, summary_path = process_session(cfg, paths)
    except Exception as e:
        logger.error("Ошибка пайплайна: %s", e, exc_info=True)
        print(f"\n❌ Ошибка: {e}")
        print(f"   Медиафайлы сохранены: {paths.dir}")
        print(f"   Попробуйте перегенерировать: mrec process {paths.session_id}")
        return

    print(f"\n✅ Готово! Артефакты в: {paths.dir}")
    print(f"   Протокол: {protocol_path}")
    print(f"   Summary:  {summary_path}")


@cli.command("process")
@click.argument("session_id")
@click.pass_context
def process_cmd(ctx, session_id: str):
    """Транскрипция + отчёт для существующей сессии."""
    cfg = ctx.obj["cfg"]

    try:
        transcript, protocol, summary = run_process(cfg, session_id)
    except _USER_ERRORS as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    print(f"\n✅ Обработка завершена для: {session_id}")
    print(f"   Протокол: {protocol}")
    print(f"   Summary:  {summary}")


@cli.command("report")
@click.argument("session_id")
@click.pass_context
def report_cmd(ctx, session_id: str):
    """Только перегенерация summary для существующей сессии."""
    cfg = ctx.obj["cfg"]

    try:
        protocol, summary = run_report_only(cfg, session_id)
    except _USER_ERRORS as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    print(f"\n✅ Summary перегенерирован для: {session_id}")
    print(f"   Протокол: {protocol}")
    print(f"   Summary:  {summary}")


# ---------------------------------------------------------------------------
# Чат
# ---------------------------------------------------------------------------

_CHAT_PROMPT_SHARE = (
    0.7  # доля входного бюджета под системный промпт; остальное — история
)


def _build_chat_system_prompt(paths, cfg) -> str:
    """Сформировать системный промпт для чата по данным встречи.

    Промпт укладывается в контекстное окно модели (llm.context_tokens): если
    транскрипт не помещается целиком, он обрезается с явной пометкой, а summary
    и ключевые моменты (компактные) включаются всегда.
    """
    from datetime import datetime

    from meeting_recorder.report import _CHARS_PER_TOKEN, _format_timestamp
    from meeting_recorder.session_utils import (
        load_highlights,
        transcript_lines,
        unique_speakers,
    )

    session_id = paths.session_id
    try:
        parts = session_id.split("_")
        meeting_dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y-%m-%d_%H-%M-%S")
        date_str = meeting_dt.strftime("%Y-%m-%d")
        time_str = meeting_dt.strftime("%H:%M")
    except Exception:
        date_str = time_str = "?"

    header = [
        "Ты — аналитик встречи. Отвечай на вопросы пользователя строго "
        "по данным встречи ниже.",
        "Если информации нет в материалах — скажи об этом прямо. "
        "Отвечай на том же языке, на котором задан вопрос.",
        "",
        "## Метаданные",
        f"Сессия: {session_id}",
        f"Дата: {date_str}  Время: {time_str}",
    ]
    tail: list[str] = []

    if paths.summary.exists():
        tail += ["", "## Summary", paths.summary.read_text(encoding="utf-8")]
    highlights = load_highlights(paths.highlights)
    if highlights:
        tail += ["", "## Ключевые моменты"]
        tail += [
            f"[{_format_timestamp(h['start'])}] {h['title']}: {h['description']}"
            for h in highlights
        ]

    body: list[str] = []
    if paths.transcript.exists():
        data = json.loads(paths.transcript.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        names = cfg.transcription.speaker_names
        header.append(f"Длительность: {data.get('duration_sec', 0) / 60:.0f} мин")
        header.append(f"Участники: {', '.join(unique_speakers(segments, names))}")
        lines = transcript_lines(
            segments, speaker_names=names, timestamp_formatter=_format_timestamp
        )

        budget_chars = int(
            cfg.llm.input_token_budget() * _CHAT_PROMPT_SHARE * _CHARS_PER_TOKEN
        ) - sum(len(x) + 1 for x in header + tail)
        kept: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) + 1 > budget_chars:
                break
            kept.append(line)
            used += len(line) + 1
        body += ["", "## Транскрипт"] + kept
        if len(kept) < len(lines):
            cut_at = kept[-1].split("]")[0].lstrip("[") if kept else "00:00"
            logger.warning(
                "Транскрипт не помещается в контекст модели: включено %d из %d реплик",
                len(kept),
                len(lines),
            )
            body.append(
                f"[Транскрипт обрезан после {cut_at} из-за размера контекстного окна; "
                "для более поздней части опирайся на summary и ключевые моменты.]"
            )
    else:
        header.append("Длительность: ?")
        if paths.protocol.exists():
            body += ["", "## Протокол", paths.protocol.read_text(encoding="utf-8")]

    return "\n".join(header + body + tail)


def _trim_history(history: list[dict[str, str]], budget_tokens: int) -> None:
    """Удалить самые старые реплики диалога, чтобы история влезла в бюджет.

    Системный промпт (history[0]) и последний вопрос пользователя сохраняются.
    """
    from meeting_recorder.report import estimate_tokens

    def total() -> int:
        return sum(estimate_tokens(m["content"]) for m in history)

    while len(history) > 2 and total() > budget_tokens:
        del history[1]


@cli.command("chat")
@click.argument("session_id", required=False, default=None)
@click.pass_context
def chat_cmd(ctx, session_id: str | None):
    """Чат с LLM по данным встречи (транскрипт, протокол, summary).

    SESSION_ID — идентификатор сессии (по умолчанию последняя).
    Введите вопрос и нажмите Enter. Для выхода — 'exit' или Ctrl+C.
    """
    from meeting_recorder.llm_client import LLMClientError, create_llm_client

    cfg = ctx.obj["cfg"]

    if session_id is None:
        sessions = list_sessions(cfg.output_dir)
        if not sessions:
            print(f"❌ Сессий не найдено в {cfg.output_dir}")
            sys.exit(1)
        session_id = sessions[-1].session_id
        print(f"ℹ Используется последняя сессия: {session_id}")

    try:
        paths = resolve_session(cfg.output_dir, session_id)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    has_transcript = paths.transcript.exists()
    has_summary = paths.summary.exists()
    if not has_transcript and not has_summary and not paths.protocol.exists():
        print(f"❌ Нет данных для сессии {session_id}.")
        print(f"   Сначала выполните транскрипцию: mrec process {session_id}")
        sys.exit(1)

    print(f"\n💬 Чат по встрече: {session_id}")
    ctx_parts = []
    if has_transcript:
        ctx_parts.append("транскрипт")
    if paths.protocol.exists():
        ctx_parts.append("протокол")
    if has_summary:
        ctx_parts.append("summary")
    print(f"   Контекст: {', '.join(ctx_parts)}")
    print(f"   Модель:   {cfg.llm.model}")
    print("   Для выхода введите 'exit' или нажмите Ctrl+C\n")

    system_prompt = _build_chat_system_prompt(paths, cfg)
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    budget = cfg.llm.input_token_budget()

    try:
        client = create_llm_client(cfg.llm, check_health=True)
    except LLMClientError as e:
        print(f"❌ LLM недоступен: {e}")
        sys.exit(1)

    try:
        while True:
            try:
                user_input = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Выход.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "выход", "q"}:
                print("👋 Выход.")
                break

            history.append({"role": "user", "content": user_input})
            _trim_history(history, budget)

            try:
                answer = client.chat(history)
            except LLMClientError as e:
                print(f"❌ Ошибка LLM: {e}\n")
                history.pop()
                continue

            history.append({"role": "assistant", "content": answer})
            print(f"\nАссистент: {answer}\n")
    finally:
        client.close()


@cli.command("mux")
@click.argument("session_id", required=False, default=None)
@click.pass_context
def mux_cmd(ctx, session_id: str | None):
    """Свести финальное аудио (mix) с видео в один MP4.

    SESSION_ID — идентификатор сессии (по умолчанию последняя).
    """
    cfg = ctx.obj["cfg"]

    if session_id is None:
        sessions = list_sessions(cfg.output_dir)
        if not sessions:
            print(f"❌ Сессий не найдено в {cfg.output_dir}")
            sys.exit(1)
        session_id = sessions[-1].session_id
        print(f"ℹ Используется последняя сессия: {session_id}")

    try:
        paths = resolve_session(cfg.output_dir, session_id)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not paths.video.exists():
        print(f"❌ Видеофайл не найден: {paths.video}")
        sys.exit(1)

    if not paths.mix_audio.exists():
        print(f"❌ Аудиофайл (mix) не найден: {paths.mix_audio}")
        sys.exit(1)

    print(f"🎬 Свожу видео + аудио → {paths.final_video.name}…")
    try:
        out = run_mux(cfg, paths, force=True)
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"✅ Готово: {out}  ({size_mb:.1f} МБ)")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


@cli.command("list")
@click.pass_context
def list_cmd(ctx):
    """Показать список сессий."""
    cfg = ctx.obj["cfg"]

    sessions = list_sessions(cfg.output_dir)
    if not sessions:
        print(f"Сессий не найдено в {cfg.output_dir}")
        return

    print(f"\n{'Session ID':<35} {'Дата':<20} {'Файлы':>5}")
    print("-" * 70)
    for s in sessions:
        file_count = (
            sum(1 for f in s.dir.iterdir() if f.is_file()) if s.dir.exists() else 0
        )
        date_part = (
            s.session_id.split("_", 1)[1] if "_" in s.session_id else s.session_id
        )
        print(f"{s.session_id:<35} {date_part:<20} {file_count:>5} файлов")
    print()


@cli.command("tray")
@click.pass_context
def tray_cmd(ctx):
    """Запустить Meeting Recorder в режиме иконки системного трея.

    Иконка меняет цвет в зависимости от состояния:
      серый — ожидание, красный — запись, оранжевый — обработка.
    Меню позволяет начать/остановить запись и открыть папку встреч.
    """
    try:
        import pystray  # noqa: F401
    except ImportError:
        print("❌ pystray не установлен.")
        print("   Установите: pip install pystray")
        sys.exit(1)

    cfg = ctx.obj["cfg"]
    from meeting_recorder.tray import TrayApp

    TrayApp(cfg, config_path=resolve_config_path(ctx.obj.get("config_path"))).run()


@cli.command("highlights")
@click.argument("session_id", required=False, default=None)
@click.pass_context
def highlights_cmd(ctx, session_id: str | None):
    """Сгенерировать 5 ключевых моментов встречи через LLM.

    SESSION_ID — идентификатор сессии (по умолчанию последняя с транскриптом).
    Результат сохраняется в *_highlights.json рядом с транскриптом.
    """
    cfg = ctx.obj["cfg"]

    if session_id is None:
        sessions = list_sessions(cfg.output_dir)
        candidates = [s for s in sessions if s.transcript.exists()]
        if not candidates:
            print(f"❌ Нет сессий с транскриптом в {cfg.output_dir}")
            sys.exit(1)
        session_id = candidates[-1].session_id
        print(f"ℹ Используется последняя сессия с транскриптом: {session_id}")

    print("⭐ Генерирую ключевые моменты…")
    try:
        highlights_path = run_highlights_only(cfg, session_id)
    except _USER_ERRORS as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    print(f"\n✅ Ключевые моменты сохранены: {highlights_path}")


@cli.command("html")
@click.argument("session_id", required=False, default=None)
@click.pass_context
def html_cmd(ctx, session_id: str | None):
    """Сгенерировать HTML-протокол с кликабельными таймкодами и открыть в браузере.

    SESSION_ID — идентификатор сессии (по умолчанию последняя с транскриптом).
    """
    import webbrowser

    cfg = ctx.obj["cfg"]

    if session_id is None:
        sessions = list_sessions(cfg.output_dir)
        candidates = [s for s in sessions if s.transcript.exists()]
        if not candidates:
            print(f"❌ Нет сессий с транскриптом в {cfg.output_dir}")
            sys.exit(1)
        session_id = candidates[-1].session_id
        print(f"ℹ Используется последняя сессия с транскриптом: {session_id}")

    try:
        html_path = run_html(cfg, session_id)
    except _USER_ERRORS as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    print(f"✅ HTML-протокол: {html_path}")
    webbrowser.open(html_path.as_uri())


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Хост сервера")
@click.option("--port", default=7070, show_default=True, type=int, help="Порт сервера")
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Открыть браузер автоматически",
)
@click.option(
    "--token",
    default=None,
    help="Токен доступа. Для не-локального --host генерируется автоматически.",
)
@click.pass_context
def serve_cmd(ctx, host: str, port: int, open_browser: bool, token: str | None):
    """Запустить локальный веб-дашборд для просмотра встреч.

    Открывает веб-интерфейс со списком сессий, медиаплеером,
    транскриптом, протоколом и summary-отчётом.

    При запуске на не-локальном адресе (--host 0.0.0.0 и т.п.) дашборд
    требует токен доступа: ссылка с токеном печатается при старте.
    """
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn не установлен.")
        print('   Установите: pip install "meeting-recorder[web]"')
        sys.exit(1)

    try:
        from fastapi.templating import Jinja2Templates  # noqa: F401
    except ImportError:
        print("❌ FastAPI или Jinja2 не установлены.")
        print('   Установите: pip install "meeting-recorder[web]"')
        sys.exit(1)

    cfg = ctx.obj["cfg"]
    import secrets

    from meeting_recorder.dashboard import create_app

    public = host not in ("127.0.0.1", "localhost", "::1")
    if public and not token:
        token = secrets.token_urlsafe(18)
    if public:
        print(
            "⚠️ Дашборд доступен из сети. Доступ — только по ссылке с токеном; "
            "передавайте её лишь доверенным людям (трафик не шифруется, HTTP)."
        )

    url = f"http://{host}:{port}"
    if token:
        url += f"/?token={token}"
    print(f"\n🌐 Дашборд: {url}")
    print(f"   Встречи:  {cfg.output_dir}")
    print("   Остановить: Ctrl+C\n")

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app = create_app(
        cfg,
        allowed_hosts=None if host in ("0.0.0.0", "::") else [host],
        access_token=token,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


@cli.command("generate-config")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.yaml",
    help="Путь для сохранения конфига",
)
@click.option(
    "--force", is_flag=True, help="Перезаписать существующий файл конфигурации"
)
def generate_config_cmd(output: str, force: bool):
    """Сгенерировать шаблон config.yaml.

    Существующий файл не перезаписывается без --force.
    Секреты из переменных окружения (HF_TOKEN, LLM_API_KEY) в файл не попадают.
    """
    from meeting_recorder.config import save_config

    if Path(output).exists() and not force:
        print(f"❌ Файл уже существует: {output}")
        print("   Укажите другой путь (-o) или добавьте --force для перезаписи.")
        sys.exit(1)

    save_config(AppConfig(), output)
    print(f"Шаблон config.yaml сохранён: {output}")
    print("Отредактируйте его перед использованием.")


if __name__ == "__main__":
    cli()
