"""Иконка в системном трее для Meeting Recorder."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from .config import AppConfig
from .naming import SessionPaths, create_session, list_sessions
from .recorder import MeetingRecorder, mix_audio_files

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Состояния и цвета иконки
# ---------------------------------------------------------------------------

_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle":       (90,  90,  90),   # серый
    "recording":  (210, 35,  35),   # красный
    "processing": (230, 150,  0),   # оранжевый
    "error":      (160,  0, 160),   # фиолетовый
}

_STATE_LABELS: dict[str, str] = {
    "idle":       "Готов к записи",
    "recording":  "Идёт запись…",
    "processing": "Обработка…",
    "error":      "Ошибка",
}

# Длительность показа статусных сообщений перед возвратом в idle
_STATUS_DISPLAY_SECS = 4   # успех / завершение операции
_ERROR_DISPLAY_SECS = 4    # обычная ошибка
_ERROR_LONG_SECS = 6       # ошибка с длинным описанием (обработка)
_ERROR_BRIEF_SECS = 3      # кратковременная ошибка ("нет сессий" и т.п.)
_EXIT_WAIT_SECS = 2        # ожидание остановки записи при выходе


def _make_icon(state: str, dim: bool = False) -> Image.Image:
    """Нарисовать круглую иконку 64×64 для заданного состояния."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = _STATE_COLORS.get(state, _STATE_COLORS["idle"])
    alpha = 100 if dim else 255
    draw.ellipse([4, 4, 60, 60], fill=(r, g, b, alpha))
    if state == "recording" and not dim:
        # белая точка — «на записи»
        draw.ellipse([25, 25, 39, 39], fill=(255, 255, 255, 220))
    elif state == "processing":
        # три маленьких белых точки
        for x in (18, 30, 42):
            draw.ellipse([x, 30, x + 8, 38], fill=(255, 255, 255, 200))
    return img


# ---------------------------------------------------------------------------
# Основной класс трей-приложения
# ---------------------------------------------------------------------------


class TrayApp:
    """Управляет иконкой трея и жизненным циклом записи."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._state = "idle"
        self._status_msg = _STATE_LABELS["idle"]
        self._lock = threading.Lock()
        self._recorder: Optional[MeetingRecorder] = None
        self._paths: Optional[SessionPaths] = None
        self._icon = None          # pystray.Icon
        self._op_start: float = 0.0   # время начала текущей операции

    # ------------------------------------------------------------------
    # Управление состоянием
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _set_state(self, state: str, msg: str = "") -> None:
        prev_state = self.state
        with self._lock:
            self._state = state
            self._status_msg = msg or _STATE_LABELS.get(state, state)
            if state != "idle" and prev_state == "idle":
                self._op_start = time.monotonic()
            elif state == "idle":
                self._op_start = 0.0
        self._refresh_icon(state)
        # Запускаем тикер при переходе из idle в активное состояние
        if prev_state == "idle" and state != "idle":
            self._start_ticker()

    def _refresh_icon(self, state: str) -> None:
        if self._icon is None:
            return
        self._icon.icon = _make_icon(state)
        self._icon.title = self._make_title()
        self._icon.menu = self._build_menu()

    def _make_title(self) -> str:
        elapsed = self._elapsed_str()
        suffix = f"  {elapsed}" if elapsed else ""
        return f"Meeting Recorder — {self._status_msg}{suffix}"

    def _elapsed_str(self) -> str:
        with self._lock:
            op_start = self._op_start
        if not op_start:
            return ""
        sec = int(time.monotonic() - op_start)
        return f"{sec // 60:02d}:{sec % 60:02d}"

    # ------------------------------------------------------------------
    # Меню
    # ------------------------------------------------------------------

    def _build_menu(self):
        import pystray

        state = self.state
        elapsed = self._elapsed_str()
        suffix = f"  {elapsed}" if elapsed else ""
        status_label = f"  {self._status_msg}{suffix}"

        return pystray.Menu(
            pystray.MenuItem(status_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "▶  Начать запись",
                self._on_start,
                enabled=(state == "idle"),
            ),
            pystray.MenuItem(
                "⏹  Остановить запись",
                self._on_stop,
                enabled=(state == "recording"),
            ),
            pystray.MenuItem(
                "⏹  Остановить без обработки",
                self._on_stop_only,
                enabled=(state == "recording"),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "💬  Чат по встрече (chat)",
                self._on_chat,
                enabled=(state == "idle"),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "⚙️  Транскрипция + отчёт (process)",
                self._on_process_session,
                enabled=(state == "idle"),
            ),
            pystray.MenuItem(
                "📝  Перегенерировать отчёт (report)",
                self._on_report_session,
                enabled=(state == "idle"),
            ),
            pystray.MenuItem(
                "🎬  Свести видео + аудио (mux)",
                self._on_mux,
                enabled=(state == "idle"),
            ),
            pystray.MenuItem(
                "📂  Открыть папку встреч",
                self._on_open_folder,
            ),
            pystray.MenuItem(
                "📁  Открыть папку последней встречи",
                self._on_open_last_session_folder,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", self._on_exit),
        )

    # ------------------------------------------------------------------
    # Обработчики меню
    # ------------------------------------------------------------------

    def _on_start(self, icon, item) -> None:
        if self.state != "idle":
            return
        threading.Thread(target=self._do_start, daemon=True, name="tray-start").start()

    def _do_start(self) -> None:
        try:
            paths = create_session(self.cfg.output_dir)
            recorder = MeetingRecorder(self.cfg, paths)
            recorder.start()
            with self._lock:
                self._recorder = recorder
                self._paths = paths
            self._set_state("recording", f"Запись: {paths.session_id}")
        except Exception as exc:
            logger.error("Ошибка старта записи: %s", exc)
            self._set_state("error", str(exc)[:60])
            time.sleep(_ERROR_DISPLAY_SECS)
            self._set_state("idle")

    def _start_ticker(self) -> None:
        """Единый тикер: мигает иконкой при записи, обновляет таймер для всех состояний."""
        def _tick():
            blink = False
            while self.state != "idle":
                state = self.state
                if self._icon:
                    if state == "recording":
                        self._icon.icon = _make_icon("recording", dim=blink)
                    self._icon.title = self._make_title()
                blink = not blink
                time.sleep(1.0)
        threading.Thread(target=_tick, daemon=True, name="tray-ticker").start()

    def _on_stop(self, icon, item) -> None:
        if self.state != "recording":
            return
        threading.Thread(target=self._do_stop, daemon=True, name="tray-stop").start()

    def _on_stop_only(self, icon, item) -> None:
        if self.state != "recording":
            return
        threading.Thread(target=self._do_stop_only, daemon=True, name="tray-stop-only").start()

    def _do_stop_and_mix(self) -> Optional[SessionPaths]:
        """Остановить запись и свести аудио. Возвращает paths или None при ошибке."""
        with self._lock:
            recorder = self._recorder
            paths = self._paths

        if recorder is None or paths is None:
            return None

        self._set_state("processing", "Останавливаю запись…")
        try:
            recorder.stop()
        except Exception as exc:
            logger.error("Ошибка остановки ffmpeg: %s", exc)
        with self._lock:
            self._recorder = None

        self._set_state("processing", "Свожу аудио…")
        try:
            if paths.mic_audio.exists() and paths.system_audio.exists():
                mix_audio_files(
                    paths.mic_audio,
                    paths.system_audio,
                    paths.mix_audio,
                    self.cfg.recording.audio_sample_rate,
                )
            elif paths.mic_audio.exists():
                paths.mic_audio.rename(paths.mix_audio)
            elif paths.system_audio.exists():
                paths.system_audio.rename(paths.mix_audio)
            else:
                logger.warning("Нет аудиофайлов для сведения")
                self._set_state("error", "Аудиофайлы не найдены")
                time.sleep(_ERROR_DISPLAY_SECS)
                self._set_state("idle")
                return None
        except Exception as exc:
            logger.error("Ошибка сведения аудио: %s", exc)

        return paths

    def _do_stop(self) -> None:
        paths = self._do_stop_and_mix()
        if paths is None:
            return
        threading.Thread(
            target=self._do_process,
            args=(paths,),
            daemon=True,
            name="tray-process",
        ).start()

    def _do_stop_only(self) -> None:
        paths = self._do_stop_and_mix()
        if paths is None:
            return
        self._notify("Meeting Recorder", f"Запись остановлена: {paths.session_id}")
        self._set_state("idle", "Запись остановлена")
        time.sleep(_STATUS_DISPLAY_SECS)
        self._set_state("idle")

    def _do_process(self, paths: SessionPaths) -> None:
        try:
            from .transcriber import transcribe
            from .report import generate_protocol, generate_summary

            self._set_state("processing", "Транскрипция…")
            result = transcribe(paths.mix_audio, self.cfg, output_path=paths.transcript)
            result["session_id"] = paths.session_id
            paths.transcript.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self._set_state("processing", "Генерирую отчёт…")
            generate_protocol(result, paths, self.cfg)
            generate_summary(result, paths, self.cfg)

            self._notify("Meeting Recorder", f"Готово: {paths.session_id}")
            self._set_state("idle", "Готово")
            time.sleep(_STATUS_DISPLAY_SECS)
            self._set_state("idle")
        except Exception as exc:
            logger.error("Ошибка обработки: %s", exc, exc_info=True)
            self._set_state("error", str(exc)[:60])
            time.sleep(_ERROR_LONG_SECS)
            self._set_state("idle")

    def _on_chat(self, icon, item) -> None:
        """Открыть новый терминал с mrec chat (последняя сессия)."""
        sessions = list_sessions(self.cfg.output_dir)
        # Ищем последнюю сессию с транскриптом — без него чат бессмысленен
        paths = None
        for s in reversed(sessions):
            if s.transcript.exists():
                paths = s
                break

        if paths is None:
            self._set_state("error", "Нет сессии с транскриптом для чата")
            time.sleep(_ERROR_BRIEF_SECS)
            self._set_state("idle")
            return

        mrec = Path(sys.executable).parent / "mrec.exe"
        if not mrec.exists():
            mrec = Path(sys.executable).parent / "mrec"

        chat_args = [str(mrec), "chat", paths.session_id]
        try:
            if sys.platform == "win32":
                import shutil
                wt = shutil.which("wt")
                if wt:
                    subprocess.Popen([wt, "cmd", "/k", *chat_args])
                else:
                    subprocess.Popen(
                        ["cmd", "/k", *chat_args],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
            else:
                subprocess.Popen(["x-terminal-emulator", "-e", *chat_args])
        except Exception as exc:
            logger.error("Не удалось открыть терминал: %s", exc)
            self._set_state("error", "Не удалось открыть терминал")
            time.sleep(_ERROR_BRIEF_SECS)
            self._set_state("idle")

    def _on_process_session(self, icon, item) -> None:
        if self.state != "idle":
            return
        threading.Thread(target=self._do_process_session, daemon=True, name="tray-process-ext").start()

    def _do_process_session(self) -> None:
        paths = self._pick_session(need_mix=True, need_no_transcript=True)
        if paths is None:
            # fallback: последняя с mix-аудио (перезапуск транскрипции)
            paths = self._pick_session(need_mix=True)
        if paths is None:
            self._set_state("error", "Нет сессии с аудио для обработки")
            time.sleep(_ERROR_BRIEF_SECS)
            self._set_state("idle")
            return
        self._do_process(paths)

    def _on_report_session(self, icon, item) -> None:
        if self.state != "idle":
            return
        threading.Thread(target=self._do_report_session, daemon=True, name="tray-report").start()

    def _do_report_session(self) -> None:
        paths = self._pick_session(need_transcript=True, need_no_summary=True)
        if paths is None:
            # fallback: последняя с транскриптом (перегенерация)
            paths = self._pick_session(need_transcript=True)
        if paths is None:
            self._set_state("error", "Нет сессии с транскриптом")
            time.sleep(_ERROR_BRIEF_SECS)
            self._set_state("idle")
            return

        self._set_state("processing", f"Отчёт: {paths.session_id}…")
        try:
            from .report import generate_protocol, generate_summary
            from .transcriber import load_transcript

            data = load_transcript(paths.transcript)
            generate_protocol(data, paths, self.cfg)
            generate_summary(data, paths, self.cfg)
            self._notify("Meeting Recorder", f"Отчёт готов: {paths.session_id}")
            self._set_state("idle", "Отчёт готов")
        except Exception as exc:
            logger.error("Ошибка генерации отчёта: %s", exc)
            self._set_state("error", str(exc)[:60])
        time.sleep(_STATUS_DISPLAY_SECS)
        self._set_state("idle")

    def _pick_session(
        self,
        need_mix: bool = False,
        need_transcript: bool = False,
        need_no_transcript: bool = False,
        need_no_summary: bool = False,
    ) -> Optional[SessionPaths]:
        """Найти последнюю сессию, удовлетворяющую условиям."""
        for s in reversed(list_sessions(self.cfg.output_dir)):
            if need_mix and not s.mix_audio.exists():
                continue
            if need_transcript and not s.transcript.exists():
                continue
            if need_no_transcript and s.transcript.exists():
                continue
            if need_no_summary and s.summary.exists():
                continue
            return s
        return None

    def _on_mux(self, icon, item) -> None:
        if self.state != "idle":
            return
        threading.Thread(target=self._do_mux, daemon=True, name="tray-mux").start()

    def _do_mux(self) -> None:
        from .recorder import mux_video
        from .naming import list_sessions

        sessions = list_sessions(self.cfg.output_dir)
        if not sessions:
            self._set_state("error", "Нет сессий для mux")
            time.sleep(_ERROR_BRIEF_SECS)
            self._set_state("idle")
            return

        # Ищем последнюю сессию с видео и mix-аудио
        paths = None
        for s in reversed(sessions):
            if s.video.exists() and s.mix_audio.exists() and not s.final_video.exists():
                paths = s
                break

        if paths is None:
            # Если все уже смикшированы — берём последнюю у которой есть видео+аудио
            for s in reversed(sessions):
                if s.video.exists() and s.mix_audio.exists():
                    paths = s
                    break

        if paths is None:
            self._set_state("error", "Нет сессии с видео и аудио")
            time.sleep(_ERROR_BRIEF_SECS)
            self._set_state("idle")
            return

        self._set_state("processing", f"Mux: {paths.session_id}…")
        try:
            out = mux_video(paths.video, paths.mix_audio, paths.final_video)
            size_mb = out.stat().st_size / 1024 / 1024
            self._notify("Meeting Recorder", f"Mux готов: {out.name} ({size_mb:.1f} МБ)")
            self._set_state("idle", "Mux завершён")
        except Exception as exc:
            logger.error("Ошибка mux: %s", exc)
            self._set_state("error", str(exc)[:60])
        time.sleep(_STATUS_DISPLAY_SECS)
        self._set_state("idle")

    def _on_open_last_session_folder(self, icon, item) -> None:
        with self._lock:
            paths = self._paths
        if paths is None:
            sessions = list_sessions(self.cfg.output_dir)
            if sessions:
                paths = sessions[-1]
        if paths is None or not paths.dir.exists():
            return
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(paths.dir)])
        else:
            subprocess.Popen(["xdg-open", str(paths.dir)])

    def _on_open_folder(self, icon, item) -> None:
        folder = Path(self.cfg.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def _on_exit(self, icon, item) -> None:
        if self.state == "recording":
            threading.Thread(target=self._do_stop, daemon=True).start()
            time.sleep(_EXIT_WAIT_SECS)
        icon.stop()

    # ------------------------------------------------------------------
    # Системное уведомление Windows
    # ------------------------------------------------------------------

    def _notify(self, title: str, message: str) -> None:
        """Показать balloon-уведомление через иконку трея."""
        try:
            if self._icon:
                self._icon.notify(message, title)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------

    def run(self) -> None:
        import pystray

        self._icon = pystray.Icon(
            name="meetrecorder",
            icon=_make_icon("idle"),
            title="Meeting Recorder",
            menu=self._build_menu(),
        )
        logger.info("Трей запущен")
        self._icon.run()
