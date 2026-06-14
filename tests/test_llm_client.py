"""Тесты для llm_client.py — mock httpx."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import httpx
from pydantic import SecretStr

from meeting_recorder.config import LLMConfig
from meeting_recorder.llm_client import LLMClient, LLMClientError, create_llm_client


def _cfg(**kw) -> LLMConfig:
    base = dict(backend="local", base_url="http://localhost:8080/v1",
                model="test", timeout=5.0)
    base.update(kw)
    return LLMConfig(**base)


def _ok_response(content: str = "reply") -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 5},
    }
    return r


# ---------------------------------------------------------------------------
# LLMClient: инициализация
# ---------------------------------------------------------------------------

class TestLLMClientInit:
    def test_no_api_key(self):
        with patch("meeting_recorder.llm_client.httpx.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = LLMClient(_cfg())
            # Authorization не передаётся
            call_kwargs = mock_cls.call_args[1]
            assert "Authorization" not in call_kwargs.get("headers", {})

    def test_with_api_key(self):
        with patch("meeting_recorder.llm_client.httpx.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            LLMClient(_cfg(api_key=SecretStr("sk-test")))
            headers = mock_cls.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer sk-test"

    def test_context_manager(self):
        with patch("meeting_recorder.llm_client.httpx.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            with LLMClient(_cfg()) as c:
                assert c is not None

    def test_base_url_strip_slash(self):
        with patch("meeting_recorder.llm_client.httpx.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            c = LLMClient(_cfg(base_url="http://localhost:8080/v1/"))
            assert not c._base_url.endswith("/")


# ---------------------------------------------------------------------------
# LLMClient.chat
# ---------------------------------------------------------------------------

class TestLLMClientChat:
    def _client_with_mock_http(self, mock_http: MagicMock) -> LLMClient:
        with patch("meeting_recorder.llm_client.httpx.Client") as mock_cls:
            mock_cls.return_value = mock_http
            return LLMClient(_cfg())

    def test_chat_success(self):
        mock_http = MagicMock()
        mock_http.post.return_value = _ok_response("Hello!")
        client = self._client_with_mock_http(mock_http)
        assert client.chat([{"role": "user", "content": "Hi"}]) == "Hello!"

    def test_chat_custom_temperature(self):
        mock_http = MagicMock()
        mock_http.post.return_value = _ok_response("ok")
        client = self._client_with_mock_http(mock_http)
        client.chat([{"role": "user", "content": "x"}], temperature=0.9, max_tokens=512)
        payload = mock_http.post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 512

    def test_chat_401(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "unauthorized"
        err = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp)
        mock_http = MagicMock()
        mock_http.post.return_value = MagicMock(raise_for_status=MagicMock(side_effect=err))
        client = self._client_with_mock_http(mock_http)
        with pytest.raises(LLMClientError, match="авторизации"):
            client.chat([{"role": "user", "content": "x"}])

    def test_chat_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "server error"
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)
        mock_http = MagicMock()
        mock_http.post.return_value = MagicMock(raise_for_status=MagicMock(side_effect=err))
        client = self._client_with_mock_http(mock_http)
        with pytest.raises(LLMClientError, match="500"):
            client.chat([{"role": "user", "content": "x"}])

    def test_chat_connect_error(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.ConnectError("refused")
        client = self._client_with_mock_http(mock_http)
        with pytest.raises(LLMClientError, match="подключиться"):
            client.chat([{"role": "user", "content": "x"}])

    def test_chat_timeout(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.TimeoutException("timeout")
        client = self._client_with_mock_http(mock_http)
        with pytest.raises(LLMClientError, match="Таймаут"):
            client.chat([{"role": "user", "content": "x"}])

    def test_chat_thinking_model_token_exhausted(self):
        mock_http = MagicMock()
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {
            "choices": [{
                "message": {"content": "", "reasoning_content": "thinking..."},
                "finish_reason": "length",
            }],
            "usage": {},
        }
        mock_http.post.return_value = r
        client = self._client_with_mock_http(mock_http)
        with pytest.raises(LLMClientError, match="max_tokens"):
            client.chat([{"role": "user", "content": "x"}])

    def test_chat_empty_content_no_reasoning(self):
        """Пустой ответ без reasoning_content — возвращаем пустую строку."""
        mock_http = MagicMock()
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {},
        }
        mock_http.post.return_value = r
        client = self._client_with_mock_http(mock_http)
        result = client.chat([{"role": "user", "content": "x"}])
        assert result == ""

    def test_chat_stream(self):
        chunks = [
            'data: {"choices": [{"delta": {"content": "He"}}]}',
            'data: {"choices": [{"delta": {"content": "llo"}}]}',
            'data: [DONE]',
            '',
        ]
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines = MagicMock(return_value=iter(chunks))

        mock_http = MagicMock()
        mock_http.stream.return_value = mock_resp
        client = self._client_with_mock_http(mock_http)
        result = client.chat([{"role": "user", "content": "x"}], stream=True)
        assert result == "Hello"


# ---------------------------------------------------------------------------
# LLMClient.health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def _client(self) -> tuple[LLMClient, MagicMock]:
        mock_http = MagicMock()
        with patch("meeting_recorder.llm_client.httpx.Client") as mock_cls:
            mock_cls.return_value = mock_http
            return LLMClient(_cfg()), mock_http

    def test_health_ok(self):
        client, mock_http = self._client()
        mock_http.get.return_value = MagicMock(is_success=True)
        assert client.health_check() is True

    def test_health_not_ok(self):
        client, mock_http = self._client()
        mock_http.get.return_value = MagicMock(is_success=False)
        assert client.health_check() is False

    def test_health_exception(self):
        client, mock_http = self._client()
        mock_http.get.side_effect = Exception("err")
        assert client.health_check() is False


# ---------------------------------------------------------------------------
# create_llm_client
# ---------------------------------------------------------------------------

class TestCreateLLMClient:
    def test_success(self):
        cfg = _cfg()
        with patch("meeting_recorder.llm_client.LLMClient") as MockClient:
            inst = MagicMock()
            inst.health_check.return_value = True
            MockClient.return_value = inst
            result = create_llm_client(cfg)
            assert result is inst

    def test_unavailable_raises(self):
        cfg = _cfg()
        with patch("meeting_recorder.llm_client.LLMClient") as MockClient:
            inst = MagicMock()
            inst.health_check.return_value = False
            MockClient.return_value = inst
            with pytest.raises(LLMClientError, match="недоступен"):
                create_llm_client(cfg)

    def test_openrouter_logs_warning(self, caplog):
        import logging
        cfg = _cfg(backend="openrouter")
        with patch("meeting_recorder.llm_client.LLMClient") as MockClient:
            inst = MagicMock()
            inst.health_check.return_value = True
            MockClient.return_value = inst
            with caplog.at_level(logging.WARNING, logger="meeting_recorder.llm_client"):
                create_llm_client(cfg)
        assert any("openrouter" in r.message.lower() for r in caplog.records)
