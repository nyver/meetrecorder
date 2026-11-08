"""Иконка в системном трее для Meeting Recorder."""

from __future__ import annotations

import enum
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from .config import AppConfig, resolve_config_path, update_config_file
from .naming import SessionPaths, create_session, list_sessions
from .recorder import (
    AudioDevice,
    MeetingRecorder,
    default_soundcard_speaker,
    list_dshow_audio_devices,
    list_soundcard_speakers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Состояния и цвета иконки
# ---------------------------------------------------------------------------


class State(enum.StrEnum):
    """Состояние трей-приложения. StrEnum: сравнивается со строками ("idle")."""

    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    State.IDLE: (90, 90, 90),  # серый
    State.STARTING: (230, 150, 0),  # оранжевый
    State.RECORDING: (210, 35, 35),  # красный
    State.PROCESSING: (230, 150, 0),  # оранжевый
    State.ERROR: (160, 0, 160),  # фиолетовый
}

_STATE_LABELS: dict[str, str] = {
    State.IDLE: "Готов к записи",
    State.STARTING: "Запуск записи…",
    State.RECORDING: "Идёт запись…",
    State.PROCESSING: "Обработка…",
    State.ERROR: "Ошибка",
}

# Длительность показа статусных сообщений перед возвратом в idle
_STATUS_DISPLAY_SECS = 4  # успех / завершение операции
_ERROR_DISPLAY_SECS = 4  # обычная ошибка
_ERROR_LONG_SECS = 6  # ошибка с длинным описанием (обработка)
_ERROR_BRIEF_SECS = 3  # кратковременная ошибка ("нет сессий" и т.п.)
_TICK_SECS = 1.0  # период тикера (мигание иконки, таймер в подсказке)
_DEVICE_REFRESH_SECS = 10  # период опроса списка устройств в idle
_EXIT_PROCESS_TIMEOUT = 1800  # до 30 мин: остановка + транскрипция + отчёт при выходе
_MAX_SESSIONS_IN_MENU = 25


def _fmt_elapsed(seconds: float) -> str:
    """Форматировать секунды в читаемую строку: '1 мин 23 сек' или '4.7 сек'."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins} мин {secs} сек" if mins else f"{seconds:.1f} сек"


def _make_icon(state: str, dim: bool = False) -> Image.Image:
    """Нарисовать круглую иконку 64×64 для заданного состояния."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = _STATE_COLORS.get(state, _STATE_COLORS[State.IDLE])
    alpha = 100 if dim else 255
    draw.ellipse([4, 4, 60, 60], fill=(r, g, b, alpha))
    if state == State.RECORDING and not dim:
        # белая точка — «на записи»
        draw.ellipse([25, 25, 39, 39], fill=(255, 255, 255, 220))
    elif state in (State.PROCESSING, State.STARTING):
        # три маленьких белых точки
        for x in (18, 30, 42):
            draw.ellipse([x, 30, x + 8, 38], fill=(255, 255, 255, 200))
    return img


@dataclass(frozen=True)
class _DeviceSnapshot:
    """Снимок доступных аудиоустройств (для меню; сравнивается при опросе)."""

    mics: tuple[str, ...] = ()
    speakers: tuple[AudioDevice, ...] = ()
    default_speaker: AudioDevice | None = None
    loaded: bool = field(default=False, compare=False)


def _scan_devices() -> _DeviceSnapshot:
    return _DeviceSnapshot(
        mics=tuple(list_dshow_audio_devices()),
        speakers=tuple(list_soundcard_speakers()),
        default_speaker=default_soundcard_speaker(),
        loaded=True,
    )


# ---------------------------------------------------------------------------
# Основной класс трей-приложения
# ---------------------------------------------------------------------------


class TrayApp:
    """Управляет иконкой трея и жизненным циклом записи."""

    def __init__(self, cfg: AppConfig, config_path: Path | str | None = None) -> None:
        self.cfg = cfg
        self._config_path = resolve_config_path(config_path)
        self._state: State = State.IDLE
        self._status_msg = _STATE_LABELS[State.IDLE]
        self._lock = threading.Lock()
        self._recorder: MeetingRecorder | None = None
        self._paths: SessionPaths | None = None
        self._icon = None  # pystray.Icon
        self._op_start: float = 0.0  # время начала текущей операции
        self._devices = _DeviceSnapshot()
        self._devices_lock = threading.Lock()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Управление состоянием
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _set_state(self, state: str, msg: str = "") -> None:
        with self._lock:
            prev_state = self._state
            self._state = State(state)
            self._status_msg = msg or _STATE_LABELS.get(state, state)
            if state != State.IDLE and prev_state == State.IDLE:
                self._op_start = time.monotonic()
            elif state == State.IDLE:
                self._op_start = 0.0
        self._refresh_icon(state)

    def _try_begin(self, state: str, msg: str = "") -> bool:
        """Атомарно перейти из idle в state. False — приложение занято.

        Проверка и смена состояния выполняются под одним lock-ом, поэтому
        двойной клик по пункту меню не запустит операцию дважды.
        """
        with self._lock:
            if self._state != State.IDLE:
                return False
            self._state = State(state)
            self._status_msg = msg or _STATE_LABELS.get(state, state)
            self._op_start = time.monotonic()
        self._refresh_icon(state)
        return True

    def _flash(self, state: str, msg: str, secs: float) -> None:
        """Показать статус на secs секунд и вернуться в idle.

        Возврат выполняется, только если статус за это время не изменился:
        после «Готово» (idle) пользователь мог уже запустить новую операцию.
        """
        self._set_state(state, msg)
        time.sleep(secs)
        with self._lock:
            unchanged = self._state == state and self._status_msg == msg
        if unchanged:
            self._set_state(State.IDLE)

    def _refresh_icon(self, state: str) -> None:
        if self._icon is None:
            return
        self._icon.icon = _make_icon(state)
        self._icon.title = self._make_title()
        self._icon.menu = self._build_menu()

    def _make_title(self) -> str:
        elapsed = self._elapsed_str()
        suffix = f"  {elapsed}" if elapsed else ""
        # Windows ограничивает подсказку иконки трея 128 символами
        return f"Meeting Recorder — {self._status_msg}{suffix}"[:127]

    def _elapsed_str(self) -> str:
        with self._lock:
            op_start = self._op_start
        if not op_start:
            return ""
        sec = int(time.monotonic() - op_start)
        return f"{sec // 60:02d}:{sec % 60:02d}"

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        threading.Thread(target=target, daemon=True, name=name).start()

    def _begin_and_spawn(
        self, target: Callable[[], None], name: str, msg: str = ""
    ) -> None:
        """Занять приложение (processing) и выполнить target в фоне."""
        if self._try_begin(State.PROCESSING, msg):
            self._spawn(target, name)

    # ------------------------------------------------------------------
    # Тикер: мигание иконки, таймер, опрос устройств
    # ------------------------------------------------------------------

    def _ticker_loop(self) -> None:
        """Единственный фоновый тикер приложения.

        - В активных состояниях раз в секунду мигает иконкой и обновляет таймер.
        - В idle раз в _DEVICE_REFRESH_SECS опрашивает аудиоустройства и
          пересобирает меню ТОЛЬКО при изменении списка: пересборка нативного
          меню (DestroyMenu) закрыла бы меню, открытое пользователем.
        """
        blink = False
        last_scan = time.monotonic()
        while not self._stop_event.wait(_TICK_SECS):
            state = self.state
            if self._icon is None:
                continue
            if state != State.IDLE:
                if state == State.RECORDING:
                    self._icon.icon = _make_icon(State.RECORDING, dim=blink)
                self._icon.title = self._make_title()
                blink = not blink
                continue
            blink = False
            if time.monotonic() - last_scan >= _DEVICE_REFRESH_SECS:
                last_scan = time.monotonic()
                if self._refresh_devices() and self.state == State.IDLE:
                    self._update_menu()

    def _refresh_devices(self) -> bool:
        """Пересканировать устройства. True — список изменился."""
        snapshot = _scan_devices()
        with self._devices_lock:
            changed = snapshot != self._devices
            self._devices = snapshot
        if changed:
            logger.debug("Список аудиоустройств изменился")
        return changed

    def _get_devices(self) -> _DeviceSnapshot:
        with self._devices_lock:
            snapshot = self._devices
        if not snapshot.loaded:
            self._refresh_devices()
            with self._devices_lock:
                snapshot = self._devices
        return snapshot

    def _update_menu(self) -> None:
        if self._icon is not None:
            try:
                self._icon.update_menu()
            except Exception as exc:
                logger.debug("update_menu: %s", exc)

    # ------------------------------------------------------------------
    # Меню
    # ------------------------------------------------------------------

    def _build_menu(self):
        import pystray

        state = self.state
        idle = state == State.IDLE
        elapsed = self._elapsed_str()
        suffix = f"  {elapsed}" if elapsed else ""
        status_label = f"  {self._status_msg}{suffix}"

        return pystray.Menu(
            pystray.MenuItem(status_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("▶  Начать запись", self._on_start, enabled=idle),
            pystray.MenuItem(
                "⏹  Остановить запись",
                self._on_stop,
                enabled=(state == State.RECORDING),
            ),
            pystray.MenuItem(
                "⏹  Остановить без обработки",
                self._on_stop_only,
                enabled=(state == State.RECORDING),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("💬  Чат по встрече (chat)", self._on_chat, enabled=idle),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "⚙️  Транскрипция + отчёт (process)",
                self._on_process_session,
                enabled=idle,
            ),
            pystray.MenuItem(
                "📝  Перегенерировать отчёт (report)",
                self._on_report_session,
                enabled=idle,
            ),
            pystray.MenuItem(
                "🎬  Свести видео + аудио (mux)", self._on_mux, enabled=idle
            ),
            pystray.MenuItem("🌐  HTML-протокол", self._on_html_protocol, enabled=idle),
            pystray.MenuItem(
                "⭐  Ключевые моменты (highlights)", self._on_highlights, enabled=idle
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("🎚️  Устройства записи", self._build_devices_menu()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("📋  Встречи", pystray.Menu(self._build_sessions_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("📂  Открыть папку встреч", self._on_open_folder),
            pystray.MenuItem(
                "📁  Открыть папку последней встречи", self._on_open_last_session_folder
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", self._on_exit),
        )

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    def _on_start(self, icon, item) -> None:
        if self._try_begin(State.STARTING):
            self._spawn(self._do_start, "tray-start")

    def _do_start(self) -> None:
        try:
            paths = create_session(self.cfg.output_dir)
            recorder = MeetingRecorder(self.cfg, paths)
            recorder.start()
            with self._lock:
                self._recorder = recorder
                self._paths = paths
            self._set_state(State.RECORDING, f"Запись: {paths.session_id}")
        except Exception as exc:
            logger.error("Ошибка старта записи: %s", exc)
            self._flash(State.ERROR, str(exc)[:60], _ERROR_DISPLAY_SECS)

    def _on_stop(self, icon, item) -> None:
        if self.state == State.RECORDING:
            self._spawn(self._do_stop, "tray-stop")

    def _on_stop_only(self, icon, item) -> None:
        if self.state == State.RECORDING:
            self._spawn(self._do_stop_only, "tray-stop-only")

    def _do_stop_and_mix(self) -> SessionPaths | None:
        """Остановить запись и свести аудио. Возвращает paths или None при ошибке."""
        from .pipeline import PipelineError, finalize_recording

        with self._lock:
            recorder = self._recorder
            paths = self._paths
            self._recorder = None  # повторный stop не должен останавливать дважды

        if recorder is None or paths is None:
            return None

        self._set_state(State.PROCESSING, "Останавливаю запись…")
        try:
            recorder.stop()
        except Exception as exc:
            logger.error("Ошибка остановки ffmpeg: %s", exc)

        self._set_state(State.PROCESSING, "Свожу аудио…")
        try:
            finalize_recording(self.cfg, paths)
        except PipelineError as exc:
            logger.error("Ошибка сведения аудио: %s", exc)
            self._flash(
                State.ERROR, f"Сведение аудио: {str(exc)[:60]}", _ERROR_DISPLAY_SECS
            )
            return None
        return paths

    def _do_stop(self) -> None:
        paths = self._do_stop_and_mix()
        if paths is not None:
            self._do_process(paths)

    def _do_stop_blocking(self) -> None:
        """Остановить запись и дождаться полной обработки (для _on_exit)."""
        self._do_stop()

    def _do_stop_only(self) -> None:
        paths = self._do_stop_and_mix()
        if paths is None:
            return
        self._notify("Meeting Recorder", f"Запись остановлена: {paths.session_id}")
        self._flash(State.IDLE, "Запись остановлена", _STATUS_DISPLAY_SECS)

    # ------------------------------------------------------------------
    # Обработка
    # ------------------------------------------------------------------

    def _do_process(self, paths: SessionPaths) -> None:
        t0 = time.monotonic()
        try:
            from .pipeline import process_session

            self._set_state(State.PROCESSING, "Транскрипция и отчёт…")
            process_session(self.cfg, paths)
            logger.info(
                "Обработка завершена за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                paths.session_id,
            )
            self._notify("Meeting Recorder", f"Готово: {paths.session_id}")
            self._flash(State.IDLE, "Готово", _STATUS_DISPLAY_SECS)
        except Exception as exc:
            logger.error(
                "Ошибка обработки за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                exc,
                exc_info=True,
            )
            self._flash(State.ERROR, str(exc)[:60], _ERROR_LONG_SECS)

    def _on_chat(self, icon, item) -> None:
        # Не блокируем UI-поток pystray (показ ошибки содержит sleep)
        self._spawn(self._do_chat, "tray-chat")

    def _do_chat(self) -> None:
        """Открыть новый терминал с mrec chat (последняя сессия с транскриптом)."""
        paths = self._pick_session(need_transcript=True)
        if paths is None:
            self._flash(
                State.ERROR, "Нет сессии с транскриптом для чата", _ERROR_BRIEF_SECS
            )
            return

        mrec = Path(sys.executable).parent / "mrec.exe"
        if not mrec.exists():
            mrec = Path(sys.executable).parent / "mrec"

        chat_args = [str(mrec), "chat", paths.session_id]
        if self._config_path.exists():
            chat_args[1:1] = ["-c", str(self._config_path)]
        try:
            if sys.platform == "win32":
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
            self._flash(State.ERROR, "Не удалось открыть терминал", _ERROR_BRIEF_SECS)

    def _on_process_session(self, icon, item) -> None:
        self._begin_and_spawn(self._do_process_session, "tray-process-ext")

    def _do_process_session(self) -> None:
        paths = self._pick_session(need_mix=True, need_no_transcript=True)
        if paths is None:
            # fallback: последняя с mix-аудио (перезапуск транскрипции)
            paths = self._pick_session(need_mix=True)
        if paths is None:
            self._flash(
                State.ERROR, "Нет сессии с аудио для обработки", _ERROR_BRIEF_SECS
            )
            return
        self._do_process(paths)

    def _on_report_session(self, icon, item) -> None:
        self._begin_and_spawn(self._do_report_session, "tray-report")

    def _do_report_session(self) -> None:
        paths = self._pick_session(need_transcript=True, need_no_summary=True)
        if paths is None:
            paths = self._pick_session(need_transcript=True)
        if paths is None:
            self._flash(State.ERROR, "Нет сессии с транскриптом", _ERROR_BRIEF_SECS)
            return
        self._run_report_for(paths)

    def _run_report_for(self, paths: SessionPaths) -> None:
        self._set_state(State.PROCESSING, f"Отчёт: {paths.session_id}…")
        t0 = time.monotonic()
        try:
            from .pipeline import run_report

            run_report(self.cfg, paths)
            logger.info(
                "Отчёт сгенерирован за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                paths.session_id,
            )
            self._notify("Meeting Recorder", f"Отчёт готов: {paths.session_id}")
            self._flash(State.IDLE, "Отчёт готов", _STATUS_DISPLAY_SECS)
        except Exception as exc:
            logger.error(
                "Ошибка генерации отчёта за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                exc,
            )
            self._flash(State.ERROR, str(exc)[:60], _STATUS_DISPLAY_SECS)

    def _pick_session(
        self,
        need_mix: bool = False,
        need_transcript: bool = False,
        need_no_transcript: bool = False,
        need_no_summary: bool = False,
        need_no_highlights: bool = False,
        need_video: bool = False,
        need_no_final_video: bool = False,
    ) -> SessionPaths | None:
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
            if need_no_highlights and s.highlights.exists():
                continue
            if need_video and not s.video.exists():
                continue
            if need_no_final_video and s.final_video.exists():
                continue
            return s
        return None

    def _on_mux(self, icon, item) -> None:
        self._begin_and_spawn(self._do_mux, "tray-mux")

    def _do_mux(self) -> None:
        paths = self._pick_session(
            need_video=True, need_mix=True, need_no_final_video=True
        )
        if paths is None:
            paths = self._pick_session(need_video=True, need_mix=True)
        if paths is None:
            self._flash(State.ERROR, "Нет сессии с видео и аудио", _ERROR_BRIEF_SECS)
            return
        self._run_mux_for(paths)

    def _run_mux_for(self, paths: SessionPaths) -> None:
        from .pipeline import run_mux

        self._set_state(State.PROCESSING, f"Mux: {paths.session_id}…")
        t0 = time.monotonic()
        try:
            out = run_mux(self.cfg, paths, force=True)
            if out is None:
                raise RuntimeError("Нет видео или аудио для сведения")
            size_mb = out.stat().st_size / 1024 / 1024
            logger.info(
                "Mux завершён за %s: %s (%.1f МБ)",
                _fmt_elapsed(time.monotonic() - t0),
                out.name,
                size_mb,
            )
            self._notify(
                "Meeting Recorder", f"Mux готов: {out.name} ({size_mb:.1f} МБ)"
            )
            self._flash(State.IDLE, "Mux завершён", _STATUS_DISPLAY_SECS)
        except Exception as exc:
            logger.error(
                "Ошибка mux за %s: %s", _fmt_elapsed(time.monotonic() - t0), exc
            )
            self._flash(State.ERROR, str(exc)[:60], _STATUS_DISPLAY_SECS)

    def _on_html_protocol(self, icon, item) -> None:
        self._begin_and_spawn(self._do_html_protocol, "tray-html")

    def _do_html_protocol(self) -> None:
        paths = self._pick_session(need_transcript=True)
        if paths is None:
            self._flash(
                State.ERROR, "Нет сессии с транскриптом для HTML", _ERROR_BRIEF_SECS
            )
            return
        self._run_html_for(paths)

    def _run_html_for(self, paths: SessionPaths) -> None:
        import webbrowser

        from .html_report import generate_html_protocol
        from .transcriber import load_transcript

        self._set_state(State.PROCESSING, f"HTML-протокол: {paths.session_id}…")
        t0 = time.monotonic()
        try:
            data = load_transcript(paths.transcript)
            html_path = generate_html_protocol(data, paths, self.cfg)
            logger.info(
                "HTML-протокол готов за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                html_path.name,
            )
            self._notify("Meeting Recorder", f"HTML готов: {html_path.name}")
            webbrowser.open(html_path.as_uri())
            self._flash(State.IDLE, "HTML-протокол готов", _STATUS_DISPLAY_SECS)
        except Exception as exc:
            logger.error(
                "Ошибка HTML-протокола за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                exc,
            )
            self._flash(State.ERROR, str(exc)[:60], _STATUS_DISPLAY_SECS)

    def _on_highlights(self, icon, item) -> None:
        self._begin_and_spawn(self._do_highlights, "tray-highlights")

    def _do_highlights(self) -> None:
        paths = self._pick_session(need_transcript=True, need_no_highlights=True)
        if paths is None:
            paths = self._pick_session(need_transcript=True)
        if paths is None:
            self._flash(State.ERROR, "Нет сессии с транскриптом", _ERROR_BRIEF_SECS)
            return
        self._run_highlights_for(paths)

    def _run_highlights_for(self, paths: SessionPaths) -> None:
        self._set_state(State.PROCESSING, f"Ключевые моменты: {paths.session_id}…")
        t0 = time.monotonic()
        try:
            from .report import generate_highlights
            from .transcriber import load_transcript

            data = load_transcript(paths.transcript)
            highlights_path = generate_highlights(data, paths, self.cfg)
            logger.info(
                "Ключевые моменты сгенерированы за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                highlights_path.name,
            )
            self._notify("Meeting Recorder", f"Готово: {highlights_path.name}")
            self._flash(State.IDLE, "Ключевые моменты готовы", _STATUS_DISPLAY_SECS)
        except Exception as exc:
            logger.error(
                "Ошибка генерации ключевых моментов за %s: %s",
                _fmt_elapsed(time.monotonic() - t0),
                exc,
            )
            self._flash(State.ERROR, str(exc)[:60], _STATUS_DISPLAY_SECS)

    # ------------------------------------------------------------------
    # Меню "Встречи" — перегенерация артефактов для любой сессии
    # ------------------------------------------------------------------

    def _brief_summary(self, paths: SessionPaths) -> list[str]:
        """Извлечь 3–4 строки из раздела 'Краткое резюме' summary-файла."""
        if not paths.summary.exists():
            return []
        try:
            text = paths.summary.read_text(encoding="utf-8")
            in_section = False
            raw: list[str] = []
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("#") and "резюме" in s.lower():
                    in_section = True
                    continue
                if in_section:
                    if s.startswith("#") or s.startswith("---"):
                        break
                    if s and not s.startswith("**"):
                        raw.append(s)

            # Word-wrap каждой строки до 68 символов, не более 4 строк итого
            result: list[str] = []
            max_w = 68
            for sentence in raw:
                if len(result) >= 4:
                    break
                if len(sentence) <= max_w:
                    result.append(sentence)
                else:
                    current = ""
                    for word in sentence.split():
                        if len(result) >= 4:
                            break
                        if current and len(current) + 1 + len(word) > max_w:
                            result.append(current)
                            current = word
                        else:
                            current = (current + " " + word).strip()
                    if current and len(result) < 4:
                        result.append(
                            current[:max_w] + "…" if len(current) > max_w else current
                        )

            return result
        except Exception:
            return []

    def _fmt_session_label(self, paths: SessionPaths) -> str:
        try:
            parts = paths.session_id.split("_")
            date = parts[1]
            t = parts[2].replace("-", ":")
            suffix = f"  #{parts[3]}" if len(parts) > 3 else ""
            return f"{date}  {t}{suffix}"
        except Exception:
            return paths.session_id

    def _build_session_submenu(self, paths: SessionPaths):
        import pystray as _pt

        def _items():
            idle = self.state == State.IDLE
            has_mix = paths.mix_audio.exists()
            has_transcript = paths.transcript.exists()
            has_video_audio = paths.video.exists() and has_mix

            def _run(fn, tname):
                def cb(icon, item):
                    self._begin_and_spawn(fn, tname)

                return cb

            items: list = []

            # Краткое саммари встречи (если есть)
            for line in self._brief_summary(paths):
                items.append(_pt.MenuItem(f"  {line}", None, enabled=False))
            if items:
                items.append(_pt.Menu.SEPARATOR)

            items += [
                _pt.MenuItem(
                    "⚙️  Транскрипция + отчёт",
                    _run(lambda: self._do_process(paths), "ses-process"),
                    enabled=idle and has_mix,
                ),
                _pt.MenuItem(
                    "📝  Перегенерировать отчёт",
                    _run(lambda: self._run_report_for(paths), "ses-report"),
                    enabled=idle and has_transcript,
                ),
                _pt.MenuItem(
                    "⭐  Ключевые моменты",
                    _run(lambda: self._run_highlights_for(paths), "ses-hl"),
                    enabled=idle and has_transcript,
                ),
                _pt.MenuItem(
                    "🌐  HTML-протокол",
                    _run(lambda: self._run_html_for(paths), "ses-html"),
                    enabled=idle and has_transcript,
                ),
                _pt.MenuItem(
                    "🎬  Mux видео + аудио",
                    _run(lambda: self._run_mux_for(paths), "ses-mux"),
                    enabled=idle and has_video_audio,
                ),
            ]
            return items

        return _pt.Menu(_items)

    def _build_sessions_items(self):
        import pystray as _pt

        sessions = list(reversed(list_sessions(self.cfg.output_dir)))[
            :_MAX_SESSIONS_IN_MENU
        ]
        if not sessions:
            return [_pt.MenuItem("  (нет сессий)", None, enabled=False)]

        return [
            _pt.MenuItem(self._fmt_session_label(s), self._build_session_submenu(s))
            for s in sessions
        ]

    # ------------------------------------------------------------------
    # Меню "Устройства записи" — выбор микрофона и устройства вывода звука
    # ------------------------------------------------------------------

    def _build_devices_menu(self):
        import pystray as _pt

        return _pt.Menu(
            _pt.MenuItem("🎙️  Микрофон", _pt.Menu(self._build_mic_device_items)),
            _pt.MenuItem(
                "🔊  Устройство вывода (системный звук)",
                _pt.Menu(self._build_speaker_device_items),
            ),
            _pt.Menu.SEPARATOR,
            _pt.MenuItem("🔄  Обновить список устройств", self._on_refresh_devices),
        )

    def _on_refresh_devices(self, icon, item) -> None:
        def _job():
            self._refresh_devices()
            self._update_menu()

        self._spawn(_job, "tray-devices")

    def _save_recording_setting(self, key: str, value: str, label: str) -> None:
        """Сохранить параметр recording.<key> в память и в файл конфига."""
        if self.state != State.IDLE:
            return
        setattr(self.cfg.recording, key, value)
        try:
            # Точечное обновление: остальные ключи и секреты в файле не трогаем
            update_config_file({"recording": {key: value}}, self._config_path)
        except Exception as exc:
            logger.error("Не удалось сохранить конфиг %s: %s", self._config_path, exc)
        self._notify("Meeting Recorder", label)
        self._update_menu()

    def _device_item(self, label: str, is_current: Callable[[], bool], on_select):
        import pystray as _pt

        return _pt.MenuItem(
            label,
            lambda icon, item: on_select(),
            checked=lambda item: is_current(),
            radio=True,
            enabled=lambda item: self.state == State.IDLE,
        )

    def _select_mic_device(self, name: str):
        def cb(icon=None, item=None):
            self._save_recording_setting(
                "mic_device", name, f"Микрофон: {name or 'автоопределение'}"
            )

        return cb

    def _build_mic_device_items(self):
        import pystray as _pt

        rc = self.cfg.recording
        devices = self._get_devices().mics
        items = [
            self._device_item(
                "Автоопределение",
                lambda: not rc.mic_device,
                self._select_mic_device(""),
            ),
        ]
        if devices:
            items.append(_pt.Menu.SEPARATOR)
            for name in devices:
                items.append(
                    self._device_item(
                        name,
                        lambda n=name: rc.mic_device == n,
                        self._select_mic_device(name),
                    )
                )
        else:
            items.append(_pt.MenuItem("  (устройства не найдены)", None, enabled=False))
        return items

    def _select_speaker_device(self, value: str, label: str = ""):
        def cb(icon=None, item=None):
            key = (
                "system_audio_speaker"
                if self.cfg.recording.system_audio_grabber == "soundcard"
                else "system_audio_device"
            )
            self._save_recording_setting(
                key, value, f"Устройство вывода: {label or value or 'по умолчанию'}"
            )

        return cb

    def _build_speaker_device_items(self):
        import pystray as _pt

        rc = self.cfg.recording

        if rc.system_audio_grabber == "wasapi":
            return [
                _pt.MenuItem(
                    "  Недоступно в режиме WASAPI (всегда устройство по умолчанию)",
                    None,
                    enabled=False,
                )
            ]

        snapshot = self._get_devices()
        items: list = []

        if rc.system_audio_grabber == "soundcard":
            default = snapshot.default_speaker
            default_text = (
                f"Устройство по умолчанию ({default.name})"
                if default
                else "Устройство по умолчанию"
            )
            items.append(
                self._device_item(
                    default_text,
                    lambda: not rc.system_audio_speaker,
                    self._select_speaker_device(""),
                )
            )
            if snapshot.speakers:
                items.append(_pt.Menu.SEPARATOR)
                for dev in snapshot.speakers:
                    items.append(
                        self._device_item(
                            dev.name,
                            # В конфиге — id (уникален); имя поддерживаем для
                            # конфигов, заполненных вручную.
                            lambda d=dev: rc.system_audio_speaker in (d.id, d.name),
                            self._select_speaker_device(dev.id, dev.name),
                        )
                    )
            else:
                items.append(
                    _pt.MenuItem("  (устройства не найдены)", None, enabled=False)
                )
            return items

        # dshow: захват через capture-устройство (virtual-audio-capturer, Stereo Mix
        # и т.п.). «По умолчанию» здесь нет — пустое имя сломало бы ffmpeg.
        if snapshot.mics:
            for name in snapshot.mics:
                items.append(
                    self._device_item(
                        name,
                        lambda n=name: rc.system_audio_device == n,
                        self._select_speaker_device(name),
                    )
                )
        else:
            items.append(_pt.MenuItem("  (устройства не найдены)", None, enabled=False))
        return items

    # ------------------------------------------------------------------
    # Папки
    # ------------------------------------------------------------------

    @staticmethod
    def _open_in_explorer(folder: Path) -> None:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def _on_open_last_session_folder(self, icon, item) -> None:
        with self._lock:
            paths = self._paths
        if paths is None:
            sessions = list_sessions(self.cfg.output_dir)
            if sessions:
                paths = sessions[-1]
        if paths is None or not paths.dir.exists():
            return
        self._open_in_explorer(paths.dir)

    def _on_open_folder(self, icon, item) -> None:
        folder = Path(self.cfg.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        self._open_in_explorer(folder)

    def _on_exit(self, icon, item) -> None:
        if self.state == State.RECORDING:
            # Не-daemon тред: дожидаемся остановки записи И полной обработки до выхода.
            t = threading.Thread(
                target=self._do_stop_blocking, daemon=False, name="tray-exit-stop"
            )
            t.start()
            t.join(timeout=_EXIT_PROCESS_TIMEOUT)
        self._stop_event.set()
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

        self._refresh_devices()
        self._icon = pystray.Icon(
            name="meetrecorder",
            icon=_make_icon(State.IDLE),
            title="Meeting Recorder",
            menu=self._build_menu(),
        )
        self._spawn(self._ticker_loop, "tray-ticker")
        logger.info("Трей запущен (конфиг: %s)", self._config_path)
        try:
            self._icon.run()
        finally:
            self._stop_event.set()
