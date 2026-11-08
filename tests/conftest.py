"""Конфигурация pytest для Meeting Recorder."""

import os
import sys
import tempfile
from pathlib import Path

# Добавляем корень проекта в sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

# Изоляция от реального окружения пользователя: state-файлы записи и лог
# не должны попадать в %APPDATA% (тесты удаляют active_session.json — это
# сломало бы идущую запись), секреты из env не должны влиять на конфиг.
# Значения задаются до импорта meeting_recorder.__main__ (пути вычисляются при импорте).
_TEST_HOME = Path(tempfile.mkdtemp(prefix="mrec-tests-"))
os.environ["MEETING_RECORDER_STATE_DIR"] = str(_TEST_HOME / "state")
os.environ["MEETING_RECORDER_LOG_FILE"] = str(
    _TEST_HOME / "logs" / "meeting_recorder.log"
)
os.environ.pop("HF_TOKEN", None)
os.environ.pop("LLM_API_KEY", None)

import httpx  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network_no_retry_delay(monkeypatch):
    """Запрет реальных HTTP-запросов и нулевые задержки между повторами LLM.

    Тест, забывший замокать LLM, мгновенно получит ConnectError вместо
    обращения к локальному llama-server и ожидания backoff.
    """
    real_send = httpx.Client.send

    def _blocked_send(self, request, *args, **kwargs):
        # Starlette TestClient — наследник httpx.Client с ASGI-транспортом
        if request.url.host == "testserver":
            return real_send(self, request, *args, **kwargs)
        raise httpx.ConnectError("network access disabled in tests", request=request)

    monkeypatch.setattr(httpx.Client, "send", _blocked_send)
    monkeypatch.setattr("meeting_recorder.llm_client._sleep", lambda _s: None)
