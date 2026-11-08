"""Унифицированный OpenAI-совместимый LLM-клиент (local / openrouter)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from .config import LLMConfig

logger = logging.getLogger(__name__)

# HTTP-статусы, при которых имеет смысл повторить запрос
_RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRY_BASE_DELAY = 2.0  # секунд; задержка растёт экспоненциально
_RETRY_MAX_DELAY = 30.0  # верхняя граница задержки (в т.ч. для Retry-After)
_ERROR_BODY_LIMIT = 500  # символов тела ответа в тексте ошибки

# Точка подмены задержки в тестах (патчить именно её, а не time.sleep глобально)
_sleep = time.sleep


class LLMClientError(Exception):
    """Ошибка LLM-клиента."""


class LLMTruncatedError(LLMClientError):
    """Ответ обрезан по лимиту max_tokens (finish_reason == "length").

    Атрибут ``partial`` содержит полученный (неполный) текст.
    """

    def __init__(self, message: str, partial: str = ""):
        super().__init__(message)
        self.partial = partial


class LLMContextOverflowError(LLMClientError):
    """Промпт не помещается в контекстное окно модели (HTTP 400/413).

    Повтор того же запроса бессмыслен — вызывающий код должен уменьшить вход
    (например, разбить транскрипт на более мелкие части).
    """


# Тексты ошибок переполнения контекста: llama-server, OpenAI, OpenRouter, vLLM
_CONTEXT_OVERFLOW_RE = re.compile(
    r"context[ _](length|size|window)|maximum context|exceeds? the (available )?"
    r"context|exceed_context_size|too many tokens|prompt is too long",
    re.IGNORECASE,
)


class _RetryableError(Exception):
    """Внутренний маркер: попытку можно повторить."""

    def __init__(self, error: LLMClientError, retry_after: float | None = None):
        super().__init__(str(error))
        self.error = error
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after") if response.headers else None
    if not value:
        return None
    try:
        return min(float(value), _RETRY_MAX_DELAY)
    except (TypeError, ValueError):
        return None


class LLMClient:
    """Клиент для вызова LLM через OpenAI-совместимый /v1/chat/completions."""

    def __init__(self, config: LLMConfig):
        self.config = config
        headers = {"Content-Type": "application/json"}
        api_key = config.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout, connect=10.0),
            headers=headers,
        )
        self._base_url = config.base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """Вызвать LLM и вернуть текст ответа.

        Raises:
            LLMTruncatedError: ответ обрезан по max_tokens.
            LLMClientError: прочие ошибки (сеть, HTTP, формат ответа).
        """
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.config.max_tokens,
        }

        if stream:
            return self._chat_stream(url, payload)

        logger.debug(
            "LLM request: model=%s, messages=%d", self.config.model, len(messages)
        )
        data = self._with_retries(lambda: self._post_json(url, payload))
        return self._extract_content(data, payload["max_tokens"])

    def health_check(self) -> bool:
        """Проверить, доступен ли LLM-бэкенд."""
        try:
            resp = self._client.get(f"{self._base_url}/models", timeout=5.0)
            return resp.is_success
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _with_retries(self, func):
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return func()
            except _RetryableError as exc:
                if attempt >= attempts:
                    raise exc.error from None
                delay = exc.retry_after or min(
                    _RETRY_BASE_DELAY * 2 ** (attempt - 1), _RETRY_MAX_DELAY
                )
                logger.warning(
                    "LLM: %s — повтор %d/%d через %.0f с",
                    exc.error,
                    attempt,
                    attempts - 1,
                    delay,
                )
                _sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise self._status_error(e) from e
        except httpx.ConnectError as e:
            raise _RetryableError(
                LLMClientError(
                    f"Не удалось подключиться к LLM-серверу ({self.config.base_url}). "
                    "Убедитесь, что сервер запущен."
                )
            ) from e
        except (httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            raise _RetryableError(
                LLMClientError(f"Таймаут подключения к LLM: {e}")
            ) from e
        except httpx.TimeoutException as e:
            # ReadTimeout/WriteTimeout не повторяем: генерация длиннее
            # llm.timeout повторится так же, а локальный сервер получит тот
            # же тяжёлый запрос повторно.
            raise LLMClientError(
                f"Таймаут ответа LLM ({self.config.timeout:.0f} с): {e}. "
                "Увеличьте llm.timeout или уменьшите llm.max_tokens."
            ) from e
        except httpx.HTTPError as e:
            # ReadError, RemoteProtocolError и прочие транспортные сбои
            raise _RetryableError(LLMClientError(f"Сетевая ошибка LLM: {e}")) from e

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise LLMClientError(f"LLM вернул не-JSON ответ: {e}") from e
        if not isinstance(data, dict):
            raise LLMClientError(
                f"Неожиданный формат ответа LLM: {type(data).__name__}"
            )
        return data

    def _status_error(self, e: httpx.HTTPStatusError) -> Exception:
        status = e.response.status_code
        body = (e.response.text or "")[:_ERROR_BODY_LIMIT]
        if status in (401, 403):
            return LLMClientError(
                f"Ошибка авторизации: {status}. Проверьте api_key и backend настройки."
            )
        if status in (400, 413) and _CONTEXT_OVERFLOW_RE.search(body):
            return LLMContextOverflowError(
                f"Запрос не помещается в контекст модели (HTTP {status}): {body}"
            )
        error = LLMClientError(f"HTTP ошибка {status}: {body}")
        if status in _RETRYABLE_STATUSES:
            return _RetryableError(error, _retry_after_seconds(e.response))
        return error

    def _extract_content(self, data: dict[str, Any], max_tokens: int) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise LLMClientError(f"LLM вернул пустой список choices: {str(data)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or ""

        if finish_reason == "length":
            if not content and message.get("reasoning_content"):
                # Thinking-модели (DeepSeek-R1 и аналоги) пишут рассуждения в
                # reasoning_content: бюджет закончился во время thinking-фазы.
                raise LLMTruncatedError(
                    "Thinking-модель исчерпала бюджет токенов "
                    f"(max_tokens={max_tokens}) "
                    "во время reasoning-фазы — ответ не был сгенерирован. "
                    "Увеличьте max_tokens в config.yaml (рекомендуется >= 16384)."
                )
            raise LLMTruncatedError(
                f"Ответ LLM обрезан по лимиту max_tokens={max_tokens}. "
                "Увеличьте llm.max_tokens или уменьшите объём входных данных.",
                partial=content,
            )

        logger.info(
            "LLM response: %d chars (usage: %s)", len(content), data.get("usage", {})
        )
        return content

    def _chat_stream(self, url: str, payload: dict[str, Any]) -> str:
        """Потоковый вызов LLM (SSE)."""
        full_text: list[str] = []
        payload = {**payload, "stream": True}
        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue  # пустые строки и SSE-комментарии (": keep-alive")
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue  # служебные чанки (например, только usage)
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        full_text.append(delta["content"])
        except Exception as e:
            raise LLMClientError(f"Ошибка потокового запроса: {e}") from e
        return "".join(full_text)


def create_llm_client(cfg: LLMConfig, *, check_health: bool = False) -> LLMClient:
    """Создать LLM-клиент из LLMConfig.

    Args:
        check_health: Если True — проверить доступность сервера перед возвратом.
            По умолчанию False: проверка происходит lazily при первом вызове chat(),
            что экономит один HTTP-запрос на каждую операцию.
    """
    logger.info(
        "Создаю LLM-клиент: backend=%s, model=%s, base_url=%s",
        cfg.backend,
        cfg.model,
        cfg.base_url,
    )

    if cfg.backend == "openrouter":
        logger.warning(
            "Режим openrouter: транскрипт будет отправлен во внешний сервис "
            "(OpenRouter). "
            "Убедитесь, что данные не содержат конфиденциальной информации."
        )

    client = LLMClient(cfg)

    if check_health and not client.health_check():
        client.close()
        raise LLMClientError(
            f"LLM-бэкенд недоступен: {cfg.base_url}. "
            "Для локального режима запустите llama-server. "
            "Для openrouter проверьте api_key."
        )

    return client
