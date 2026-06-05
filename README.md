# Meeting Recorder

Запись экрана и звука, транскрипция с диаризацией, протокол и summary-отчёт.

## Требования

- **ОС:** Windows 10/11 x64
- **Python:** 3.11+
- **ffmpeg:** должен быть в `PATH`
- **GPU (опционально):** NVIDIA с CUDA для WhisperX и локальной LLM

## Установка

```bash
# 1. Создайте виртуальное окружение
python -m venv .venv
.venv\Scripts\activate

# 2. Установите зависимости
pip install -e ".[dev]"

# 3. Убедитесь, что ffmpeg в PATH
ffmpeg -version
```

## Настройка

```bash
# Сгенерировать шаблон config.yaml
mrec generate-config

# Или отредактировать вручную config.yaml
```

Переопределение секретов через переменные окружения:

```bash
# Для диаризации (pyannote.audio)
set HF_TOKEN=hf_xxxxxxxx

# Для LLM (если backend=openrouter)
set LLM_API_KEY=sk-or-vx-xxxx
```

## Использование

### CLI

```bash
# Начать запись
mrec start

# Остановить запись + автоматическая транскрипция и генерация отчёта
mrec stop

# Повторная обработка существующей сессии
mrec process <session_id>

# Только перегенерация summary
mrec report <session_id>

# Список сессий
mrec list
```

### Программный API

```python
from meeting_recorder.config import load_config
from meeting_recorder.pipeline import run_transcribe_only, run_report_only
from meeting_recorder.naming import resolve_session

cfg = load_config()

# Транскрипция существующей сессии
transcript = run_transcribe_only(cfg, "meeting_2026-06-05_14-30-12")

# Генерация отчёта
protocol, summary = run_report_only(cfg, "meeting_2026-06-05_14-30-12")
```

## Архитектура

```
meeting_recorder/
├── __main__.py          # CLI (click)
├── config.py            # pydantic-конфиг (yaml)
├── naming.py            # session_id, пути артефактов
├── recorder.py          # ffmpeg: видео + 2 аудио-дорожки
├── transcriber.py       # WhisperX: транскрипция + диаризация
├── llm_client.py        # OpenAI-совместимый клиент
├── report.py            # Протокол + summary
├── pipeline.py          # Оркестрация
├── prompts/             # Шаблоны для LLM
└── ui/                  # PySide6 GUI (будет)
```

## Пайплайн

```
Запись (ffmpeg) → Транскрипция (WhisperX) → Протокол (Markdown) → Summary (LLM)
```

Каждый этап можно запустить независимо по `session_id`.

## Бэкенды LLM

| Параметр | local (llama-server) | openrouter |
|----------|----------------------|------------|
| `base_url` | `http://127.0.0.1:8080/v1` | `https://openrouter.ai/api/v1` |
| `api_key` | (пустой) | ваш ключ OpenRouter |
| `model` | ваша модель | напр. `meta-llama/llama-3.1-70b-instruct` |

## Локальный LLM

Поднять llama-server:

```bash
llama-server -m your-model.gguf --host 127.0.0.1 --port 8080
```

## Конфиденциальность

- `local` — все данные остаются на машине.
- `openrouter` — транскрипт отправляется в облако OpenRouter. Приложение предупреждает об этом в логах.

## Тесты

```bash
pytest -v
```
