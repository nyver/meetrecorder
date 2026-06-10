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
# Состояние записи (пersistence на диск)
# ---------------------------------------------------------------------------

_STATE_DIR = Path(__file__).parent / ".state"
_STATE_FILE = _STATE_DIR / "active_session.json"
_STOP_FILE = _STATE_DIR / "stop_requested"
_DONE_FILE = _STATE_DIR / "recording_done"


def _ensure_state_dir() -> None:
    _STATE_DIR.mkdir(exist_ok=True)


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

    recorder.start()
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


@cli.command()
@click.pass_context
def stop(ctx):
    """Остановить запись и запустить полный пайплайн."""
    cfg = ctx.obj["cfg"]

    # Загружаем состояние из файла
    state = _load_state()
    if state is None:
        print("⚠️ Запись не идёт. Нечего останавливать.")
        return

    session_id = state.get("session_id")
    ffmpeg_pid = state.get("ffmpeg_pid")

    if not session_id:
        print("⚠️ Состояние записи повреждено. Удалите .state/active_session.json")
        _remove_state()
        return

    print(f"\n⏹ Останавливаю запись сессии: {session_id}…")

    _ensure_state_dir()
    _DONE_FILE.unlink(missing_ok=True)
    _STOP_FILE.touch()

    # Ждём пока mrec start завершит остановку ffmpeg + soundcard и сохранит файлы
    print("⏳ Ожидаю завершения записи…")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if _DONE_FILE.exists():
            logger.info("Запись завершена штатно")
            break
    else:
        # Таймаут: mrec start не ответил — принудительно останавливаем ffmpeg
        logger.warning("Таймаут ожидания mrec start — принудительная остановка ffmpeg")
        if ffmpeg_pid:
            _stop_ffmpeg_graceful(int(ffmpeg_pid))
        _remove_state()

    _DONE_FILE.unlink(missing_ok=True)
    _STOP_FILE.unlink(missing_ok=True)

    # Восстанавливаем пути
    from meeting_recorder.recorder import mix_audio_files
    from meeting_recorder.naming import resolve_session

    try:
        paths = resolve_session(cfg.output_dir, session_id)
    except FileNotFoundError as e:
        print(f"❌ Сессия не найдена: {e}")
        return

    # Сводим аудио
    print("🔊 Свожу аудио…")
    mixed = False
    try:
        if paths.mic_audio.exists() and paths.system_audio.exists():
            mix_audio_files(
                paths.mic_audio,
                paths.system_audio,
                paths.mix_audio,
                cfg.recording.audio_sample_rate,
            )
            mixed = True
            logger.info("Аудио сведено: %s", paths.mix_audio)
        elif paths.mic_audio.exists():
            _rename_with_retry(paths.mic_audio, paths.mix_audio)
            mixed = True
            logger.info("Аудио (только микрофон): %s", paths.mix_audio)
        elif paths.system_audio.exists():
            _rename_with_retry(paths.system_audio, paths.mix_audio)
            mixed = True
            logger.info("Аудио (только системный звук): %s", paths.mix_audio)
        else:
            logger.warning("Аудиофайлы не найдены — ffmpeg не записал данные")
    except Exception as e:
        logger.warning("Не удалось свести аудио: %s", e)

    if not mixed:
        print(f"\n❌ Аудиофайлы не созданы — ffmpeg не смог записать данные.")
        ffmpeg_log = paths.ffmpeg_log
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
