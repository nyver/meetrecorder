"""CLI-точка входа для Meeting Recorder."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Принудительно UTF-8 на Windows (иначе кириллица ломается в консоли)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import click

from meeting_recorder.config import AppConfig, load_config
from meeting_recorder.naming import list_sessions, resolve_session
from meeting_recorder.pipeline import PipelineError, run_process, run_report_only, run_transcribe_only

logger = logging.getLogger("meeting_recorder")

# ---------------------------------------------------------------------------
# Состояние записи (persistence на диск)
# ---------------------------------------------------------------------------


def _default_state_dir() -> Path:
    """Вернуть user-writable директорию для файлов состояния."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "MeetingRecorder" / ".state"


_STATE_DIR = _default_state_dir()
_STATE_FILE = _STATE_DIR / "active_session.json"
_STOP_FILE = _STATE_DIR / "stop_requested"
_DONE_FILE = _STATE_DIR / "recording_done"


def _ensure_state_dir() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def _save_state(session_id: str, video_path: str, ffmpeg_pid: int | None = None) -> None:
    _ensure_state_dir()
    _STATE_FILE.write_text(
        json.dumps(
            {"session_id": session_id, "video_path": video_path, "ffmpeg_pid": ffmpeg_pid},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_state() -> dict | None:
    if not _STATE_FILE.exists():
        return None
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _remove_state() -> None:
    try:
        _STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Настроить логирование (console + file)."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "meeting_recorder.log",
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Команды CLI
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="mrec")
@click.option("-v", "--verbose", is_flag=True, help="Подробное логирование")
@click.option("-c", "--config", "config_path", type=click.Path(), help="Путь к config.yaml")
@click.pass_context
def cli(ctx, verbose, config_path):
    """Meeting Recorder — запись экрана, транскрипция и генерация отчётов."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path

    setup_logging(verbose)
    try:
        cfg = load_config(config_path)
    except Exception as e:
        logger.error("Ошибка загрузки config.yaml: %s", e)
        sys.exit(1)

    ctx.obj["cfg"] = cfg


def _rename_with_retry(src: Path, dst: Path, retries: int = 10, delay: float = 0.5) -> None:
    """Переименовать файл с повторными попытками (файл может быть временно заблокирован)."""
    for attempt in range(retries):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _stop_ffmpeg_graceful(pid: int, timeout: int = 15) -> bool:
    """Graceful stop ffmpeg по PID через CTRL_BREAK_EVENT, затем ждём завершения.

    CTRL_BREAK_EVENT позволяет ffmpeg финализировать контейнер и закрыть файлы.
    Работает только если ffmpeg запущен с CREATE_NEW_PROCESS_GROUP.
    Если процесс не завершился за timeout — форс-килл как запасной вариант.
    """
    import os
    import signal
    import subprocess as sp

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

    # Ждём завершения
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sp.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        if str(pid) not in result.stdout:
            logger.info("ffmpeg PID=%s завершился штатно", pid)
            return True
        time.sleep(0.5)

    # Форс-килл как запасной вариант
    logger.warning("ffmpeg PID=%s не завершился за %ds — принудительное завершение", pid, timeout)
    try:
        sp.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
        time.sleep(1.5)  # Дать Windows время закрыть файловые хэндлы
    except Exception as e:
        logger.error("Принудительное завершение ffmpeg PID=%s не удалось: %s", pid, e)
    return False


@cli.command()
@click.pass_context
def start(ctx):
    """Начать запись экрана и звука."""
    cfg = ctx.obj["cfg"]

    # Проверяем, нет ли уже идущей записи
    existing = _load_state()
    if existing:
        sid = existing.get("session_id", "?")
        print(f"⚠️ Уже идёт запись сессии: {sid}")
        print(f"   Остановите её командой: mrec stop")
        return

    from meeting_recorder.naming import create_session
    from meeting_recorder.recorder import MeetingRecorder

    paths = create_session(cfg.output_dir)
    recorder = MeetingRecorder(cfg, paths)

    _STOP_FILE.unlink(missing_ok=True)
    _DONE_FILE.unlink(missing_ok=True)

    try:
        recorder.start()
    except Exception as exc:
        logger.error("Ошибка запуска записи: %s", exc)
        print(f"\n❌ Ошибка запуска записи: {exc}\n")
        return

    ffmpeg_pid = recorder.ffmpeg_pid

    _save_state(paths.session_id, str(paths.video), ffmpeg_pid)
    logger.info("Запись начата: %s (ffmpeg PID=%s)", paths.session_id, ffmpeg_pid)
    print(f"\n✅ Запись начата: {paths.session_id}")
    print(f"   Папка: {paths.dir}")
    print(f"   Остановить: mrec stop  (или Ctrl+C)\n")

    # Блокируем до получения сигнала остановки
    try:
        while not _STOP_FILE.exists():
            time.sleep(0.5)
        logger.info("Stop-сигнал получен — останавливаю запись")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — останавливаю запись")
    finally:
        # Останавливаем ffmpeg И soundcard-захват, ждём сохранения файлов
        recorder.stop()
        _DONE_FILE.touch()
        _STOP_FILE.unlink(missing_ok=True)
        _remove_state()
        logger.info("Запись завершена: %s", paths.session_id)


def _signal_stop_and_wait(ffmpeg_pid: int | None) -> None:
    """Послать stop-сигнал процессу mrec start и дождаться завершения записи."""
    _ensure_state_dir()
    _DONE_FILE.unlink(missing_ok=True)
    _STOP_FILE.touch()

    print("⏳ Ожидаю завершения записи…")
    deadline = time.monotonic() + 30
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
    from meeting_recorder.recorder import mix_audio_files

    print("🔊 Свожу аудио…")
    try:
        if paths.mic_audio.exists() and paths.system_audio.exists():
            mix_audio_files(
                paths.mic_audio,
                paths.system_audio,
                paths.mix_audio,
                cfg.recording.audio_sample_rate,
            )
            logger.info("Аудио сведено: %s", paths.mix_audio)
        elif paths.mic_audio.exists():
            _rename_with_retry(paths.mic_audio, paths.mix_audio)
            logger.info("Аудио (только микрофон): %s", paths.mix_audio)
        elif paths.system_audio.exists():
            _rename_with_retry(paths.system_audio, paths.mix_audio)
            logger.info("Аудио (только системный звук): %s", paths.mix_audio)
        else:
            logger.warning("Аудиофайлы не найдены — ffmpeg не записал данные")
            return False
    except Exception as e:
        logger.warning("Не удалось свести аудио: %s", e)
        return False
    return True


def _load_active_session(cfg) -> tuple[str | None, int | None, object | None]:
    """Загрузить активную сессию. Возвращает (session_id, ffmpeg_pid, paths)."""
    from meeting_recorder.naming import resolve_session

    state = _load_state()
    if state is None:
        print("⚠️ Запись не идёт. Нечего останавливать.")
        return None, None, None

    session_id = state.get("session_id")
    ffmpeg_pid = state.get("ffmpeg_pid")

    if not session_id:
        print("⚠️ Состояние записи повреждено. Удалите .state/active_session.json")
        _remove_state()
        return None, None, None

    print(f"\n⏹ Останавливаю запись сессии: {session_id}…")

    try:
        paths = resolve_session(cfg.output_dir, session_id)
    except FileNotFoundError as e:
        print(f"❌ Сессия не найдена: {e}")
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
        ffmpeg_log = paths.ffmpeg_log
        print(f"\n❌ Аудиофайлы не созданы.")
        if ffmpeg_log.exists():
            for line in ffmpeg_log.read_text(errors="replace").splitlines()[-20:]:
                print(f"     {line}")
        print(f"   Папка сессии: {paths.dir}")
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
        ffmpeg_log = paths.ffmpeg_log
        print(f"\n❌ Аудиофайлы не созданы — ffmpeg не смог записать данные.")
        if ffmpeg_log.exists():
            tail = ffmpeg_log.read_text(errors="replace").splitlines()[-30:]
            print(f"   Лог ffmpeg ({ffmpeg_log.name}):")
            for line in tail:
                print(f"     {line}")
        else:
            print(f"   Проверьте mic_device и system_audio_device в config.yaml")
        print(f"   Папка сессии: {paths.dir}")
        return

    # Запускаем пайплайн
    print("\n🚀 Запускаю пайплайн: транскрипция → протокол → summary…\n")
    try:
        from meeting_recorder.transcriber import transcribe
        from meeting_recorder.report import generate_protocol, generate_summary

        # Транскрипция
        print("📝 Транскрипция…")
        result = transcribe(
            paths.mix_audio,
            cfg,
            output_path=paths.transcript,
        )
        result["session_id"] = paths.session_id
        paths.transcript.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Транскрипция завершена: %s (%d сегментов)", paths.transcript, len(result["segments"]))

        # Протокол
        print("📋 Протокол…")
        generate_protocol(result, paths, cfg)

        # Summary
        print("📄 Summary…")
        generate_summary(result, paths, cfg)

        print(f"\n✅ Готово! Артефакты в: {paths.dir}")
        print(f"   Протокол: {paths.protocol}")
        print(f"   Summary:  {paths.summary}")

    except Exception as e:
        logger.error("Ошибка пайплайна: %s", e, exc_info=True)
        print(f"\n❌ Ошибка: {e}")
        print(f"   Медиафайлы сохранены: {paths.dir}")
        print(f"   Попробуйте перегенерировать: mrec report {paths.session_id}")


@cli.command("process")
@click.argument("session_id")
@click.pass_context
def process_cmd(ctx, session_id: str):
    """Транскрипция + отчёт для существующей сессии."""
    cfg = ctx.obj["cfg"]

    try:
        transcript, protocol, summary = run_process(cfg, session_id)
    except PipelineError as e:
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
    except PipelineError as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    print(f"\n✅ Summary перегенерирован для: {session_id}")
    print(f"   Протокол: {protocol}")
    print(f"   Summary:  {summary}")


def _build_chat_system_prompt(paths, cfg) -> str:
    """Сформировать системный промпт для чата по данным встречи."""
    import json as _json
    from datetime import datetime

    session_id = paths.session_id
    try:
        # Поддержка суффиксных ID: meeting_YYYY-MM-DD_HH-MM-SS[_N]
        parts = session_id.split("_")
        meeting_dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y-%m-%d_%H-%M-%S")
        date_str = meeting_dt.strftime("%Y-%m-%d")
        time_str = meeting_dt.strftime("%H:%M")
    except Exception:
        date_str = time_str = "?"

    lines = [
        "Ты — аналитик встречи. Отвечай на вопросы пользователя строго по данным встречи ниже.",
        "Если информации нет в материалах — скажи об этом прямо. Отвечай на том же языке, на котором задан вопрос.",
        "",
        f"## Метаданные",
        f"Сессия: {session_id}",
        f"Дата: {date_str}  Время: {time_str}",
    ]

    # Транскрипт
    if paths.transcript.exists():
        data = _json.loads(paths.transcript.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        duration_min = data.get("duration_sec", 0) / 60
        lines.append(f"Длительность: {duration_min:.0f} мин")

        speaker_names = cfg.transcription.speaker_names
        unique_speakers = sorted({seg.get("speaker", "UNKNOWN") for seg in segments})
        named = [speaker_names.get(s, s) for s in unique_speakers]
        lines.append(f"Участники: {', '.join(named)}")
        lines += ["", "## Транскрипт"]

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            speaker = speaker_names.get(speaker, speaker)
            start = int(seg["start"])
            ts = f"{start // 60:02d}:{start % 60:02d}"
            lines.append(f"[{ts}] {speaker}: {seg['text'].strip()}")
    else:
        lines.append("Длительность: ?")

    # Протокол (только если нет транскрипта или как дополнение)
    if paths.protocol.exists() and not paths.transcript.exists():
        lines += ["", "## Протокол"]
        lines.append(paths.protocol.read_text(encoding="utf-8"))

    # Summary
    if paths.summary.exists():
        lines += ["", "## Summary"]
        lines.append(paths.summary.read_text(encoding="utf-8"))

    return "\n".join(lines)


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
    print(f"   Для выхода введите 'exit' или нажмите Ctrl+C\n")

    system_prompt = _build_chat_system_prompt(paths, cfg)
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    try:
        client = create_llm_client(cfg.llm)
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
    from meeting_recorder.recorder import mux_video

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
        out = mux_video(paths.video, paths.mix_audio, paths.final_video)
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

    print(f"\n{'Session ID':<35} {'Дата':<12} {'Файлы':>5}")
    print("-" * 60)
    for s in sessions:
        file_count = sum(1 for f in s.dir.iterdir() if f.is_file()) if s.dir.exists() else 0
        date_part = s.session_id.split("_", 1)[1] if "_" in s.session_id else s.session_id
        print(f"{s.session_id:<35} {date_part:<12} {file_count:>5} файлов")
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
    TrayApp(cfg).run()


@cli.command("generate-config")
@click.option("--output", "-o", type=click.Path(), default="config.yaml", help="Путь для сохранения конфига")
def generate_config_cmd(output: str):
    """Сгенерировать шаблон config.yaml."""
    from meeting_recorder.config import save_config
    cfg = AppConfig()
    save_config(cfg, output)
    print(f"Шаблон config.yaml сохранён: {output}")
    print("Отредактируйте его перед использованием.")


if __name__ == "__main__":
    cli()
