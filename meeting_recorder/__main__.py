"""CLI-точка входа для Meeting Recorder."""

from __future__ import annotations

import logging
import sys
import threading
import time

import click
import yaml

from meeting_recorder.config import AppConfig, load_config
from meeting_recorder.naming import list_sessions, resolve_session
from meeting_recorder.pipeline import PipelineError, run_process, run_report_only, run_transcribe_only

logger = logging.getLogger("meeting_recorder")

# ---------------------------------------------------------------------------
# Глобальный указатель на процесс записи (для stop)
# ---------------------------------------------------------------------------

_recorder_state: dict = {
    "process": None,
    "thread": None,
}


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


@cli.command()
@click.pass_context
def start(ctx):
    """Начать запись экрана и звука."""
    from meeting_recorder.pipeline import run_record

    cfg = ctx.obj["cfg"]

    # Создаём сессию и запускаем запись в фоновом потоке
    from meeting_recorder.naming import create_session
    from meeting_recorder.recorder import MeetingRecorder

    paths = create_session(cfg.output_dir)
    recorder = MeetingRecorder(cfg, paths)

    def _record_thread():
        try:
            recorder.start()
            logger.info(
                "Идёт запись: %s | длительность: %0.1f сек",
                paths.session_id, recorder.duration,
            )
        except Exception as e:
            logger.error("Ошибка записи: %s", e)

    _recorder_state["process"] = recorder
    _recorder_state["thread"] = threading.Thread(target=_record_thread, daemon=True)
    _recorder_state["thread"].start()

    logger.info("Запись начата: %s", paths.session_id)
    logger.info("Остановите командой: mrec stop")
    print(f"\n✅ Запись начата: {paths.session_id}")
    print(f"   Папка: {paths.dir}")
    print(f"   Команда stop: mrec stop\n")


@cli.command()
@click.pass_context
def stop(ctx):
    """Остановить запись и запустить полный пайплайн."""
    cfg = ctx.obj["cfg"]

    recorder = _recorder_state.get("process")
    if recorder is None or not recorder.is_recording:
        print("⚠️ Запись не идёт. Нечего останавливать.")
        return

    print("\n⏹ Останавливаю запись…")
    recorder.stop()
    _recorder_state["process"] = None

    # Сводим аудио
    from meeting_recorder.recorder import mix_audio_files
    from meeting_recorder.naming import resolve_session

    paths = resolve_session(cfg.output_dir, recorder.paths.session_id)

    print("🔊 Свожу аудио…")
    if paths.mic_audio.exists() and paths.system_audio.exists():
        mix_audio_files(
            paths.mic_audio,
            paths.system_audio,
            paths.mix_audio,
            cfg.recording.audio_sample_rate,
        )
    elif paths.mic_audio.exists():
        paths.mic_audio.rename(paths.mix_audio)

    logger.info("Аудио сведено: %s", paths.mix_audio)

    # Запускаем пайплайн
    print("\n🚀 Запускаю пайплайн: транскрипция → протокол → summary…\n")
    try:
        from meeting_recorder.transcriber import transcribe
        from meeting_recorder.report import generate_protocol, generate_summary
        import json

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
