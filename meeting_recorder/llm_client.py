"""Унифицированный OpenAI-совместимый LLM-клиент (local / openrouter)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Ошибка LLM-клиента."""


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

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """Вызвать LLM и вернуть текст ответа."""
        url = f"{self._base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }

        if stream:
            return self._chat_stream(url, payload)

        logger.debug("LLM request: model=%s, messages=%d", self.config.model, len(messages))

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            if "401" in str(e) or "unauthorized" in error_body.lower():
                raise LLMClientError(
                    f"Ошибка авторизации: {e.response.status_code}. "
                    f"Проверьте api_key и backend настройки."
                ) from e
            raise LLMClientError(
                f"HTTP ошибка {e.response.status_code}: {error_body}"
            ) from e
        except httpx.ConnectError as e:
            raise LLMClientError(
                f"Не удалось подключиться к LLM-серверу ({self.config.base_url}). "
                f"Убедитесь, что сервер запущен."
            ) from e
        except httpx.TimeoutException as e:
            raise LLMClientError(f"Таймаут запроса к LLM: {e}") from e

        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or ""

        # Thinking-модели (DeepSeek-R1 и аналоги) пишут рассуждения в reasoning_content.
        # Если content пуст и finish_reason == "length" — бюджет токенов закончился
        # во время thinking-фазы. Нужно увеличить max_tokens в конфиге.
        if not content:
            reasoning = message.get("reasoning_content", "")
            finish_reason = choice.get("finish_reason", "")
            if reasoning and finish_reason == "length":
                raise LLMClientError(
                    f"Thinking-модель исчерпала бюджет токенов (max_tokens={self.config.max_tokens}) "
                    f"во время reasoning-фазы — ответ не был сгенерирован. "
                    f"Увеличьте max_tokens в config.yaml (рекомендуется >= 16384)."
                )

        logger.info(
            "LLM response: %d chars (usage: %s)",
            len(content),
            data.get("usage", {}),
        )
        return content

    def _chat_stream(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> str:
        """Потоковый вызов LLM."""
        full_text = []
        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        import json
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            full_text.append(delta["content"])
        except Exception as e:
            raise LLMClientError(f"Ошибка потокового запроса: {e}") from e
        return "".join(full_text)

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


def create_llm_client(cfg: LLMConfig, *, check_health: bool = False) -> LLMClient:
    """Создать LLM-клиент из LLMConfig.

    Args:
        check_health: Если True — проверить доступность сервера перед возвратом.
            По умолчанию False: проверка происходит lazily при первом вызове chat(),
            что экономит один HTTP-запрос на каждую операцию.
    """
    logger.info(
        "Создаю LLM-клиент: backend=%s, model=%s, base_url=%s",
        cfg.backend, cfg.model, cfg.base_url,
    )

    if cfg.backend == "openrouter":
        logger.warning(
            "Режим openrouter: транскрипт будет отправлен во внешний сервис (OpenRouter). "
            "Убедитесь, что данные не содержат конфиденциальной информации."
        )

    client = LLMClient(cfg)

    if check_health and not client.health_check():
        raise LLMClientError(
            f"LLM-бэкенд недоступен: {cfg.base_url}. "
            f"Для локального режима запустите llama-server. "
            f"Для openrouter проверьте api_key."
        )

    return client
