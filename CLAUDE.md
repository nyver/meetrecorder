# CLAUDE.md

## Role

Ты — Senior Python Developer с экспертизой в проектировании CLI/Web-архитектур и написании идиоматичного, тестируемого кода.

## Project

Meeting Recorder (`mrec`) — Windows CLI/tray-приложение для записи экрана и звука
с транскрипцией (faster-whisper + pyannote diarization), генерацией протокола/summary/
highlights через LLM и веб-дашбордом (FastAPI) для просмотра встреч.

- Entry point: `mrec` → `meeting_recorder.__main__:cli` (click)
- Python 3.11+, Windows 10/11 x64, ffmpeg в PATH

### Структура

```
meeting_recorder/
├── __main__.py     # CLI-команды (start, stop, process, report, highlights, html, serve, mux, chat, tray, list, generate-config)
├── config.py       # Pydantic-конфиг (YAML)
├── naming.py       # session_id и пути артефактов (SessionPaths)
├── recorder.py     # ffmpeg: захват видео/аудио, mix_audio_files, mux_video
├── transcriber.py  # faster-whisper: транскрипция + pyannote-диаризация
├── llm_client.py   # HTTP-клиент, совместимый с OpenAI (local / openrouter)
├── report.py       # Протокол + summary + highlights (map-reduce для длинных встреч)
├── html_report.py  # Статический HTML-протокол со встроенным плеером
├── dashboard.py    # FastAPI веб-дашборд
├── pipeline.py     # Оркестрация пайплайна (finalize_recording, process_session, run_*)
├── session_utils.py # Общие хелперы артефактов (атомарная запись, транскрипт, highlights)
├── state.py        # Состояние записи и межпроцессный lock (общий для CLI и трея)
├── tray.py         # Иконка в трее (pystray + Pillow)
├── templates/      # Jinja2-шаблоны для дашборда
└── prompts/        # Промпты для LLM (summary, highlights, protocol_clean)
```

Пайплайн: Recording → mix_audio_files → Transcription → Protocol → Summary → Highlights → [auto_mux] mux_video → HTML Protocol.

## Правила работы

- Работай только в каталоге проекта (`e:\work\python\meetrecorder`).
- Покрытие тестов должно быть > 70% (`pytest --cov`).
- После добавления нового функционала:
  - актуализируй `README.md` и `config.example.yaml`;
  - пересобери приложение в `.venv` (`pip install -e ".[tray,web]"`).
- Тесты — `pytest -v`, лежат в `tests/`, зеркалируют модули `meeting_recorder/*.py`.
- Линтер/форматирование — `ruff` (`target-version = py311`, `line-length = 88`).
