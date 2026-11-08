"""Тесты для llm_client.py — mock httpx."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from meeting_recorder.config import LLMConfig
from meeting_recorder.llm_client import LLMClient, LLMClientError, create_llm_client


def _cfg(**kw) -> LLMConfig:
    base = dict(
        backend="local", base_url="http://localhost:8080/v1", model="test", timeout=5.0
    )
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
            LLMClient(_cfg())
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
    @pytest.mark.parametrize("stream", [False, True])
    def test_output_limit_fits_context(self, stream):
        cfg = _cfg(context_tokens=8192, max_tokens=16384)
        with LLMClient(cfg) as client:
            with (
                patch.object(
                    client,
                    "_post_json",
                    return_value={"choices": [{"message": {"content": "ok"}}]},
                ) as post,
                patch.object(client, "_chat_stream", return_value="ok") as streaming,
            ):
                client.chat(
                    [{"role": "user", "content": "x" * (cfg.input_token_budget() * 2)}],
                    stream=stream,
                )
                payload = (
                    streaming.call_args.args[1] if stream else post.call_args.args[1]
                )
        assert payload["max_tokens"] == 4096

    def test_large_prompt_reduces_output_budget(self):
        with LLMClient(_cfg(context_tokens=2048)) as client:
            with patch.object(
                client,
                "_post_json",
                return_value={"choices": [{"message": {"content": "ok"}}]},
            ) as post:
                client.chat([{"role": "user", "content": "x" * 3000}])
        assert 0 < post.call_args.args[1]["max_tokens"] < 1024

    def test_oversized_prompt_rejected_before_http(self):
        from meeting_recorder.llm_client import LLMContextOverflowError

        with LLMClient(_cfg(context_tokens=2048)) as client:
            with (
                patch.object(client, "_post_json") as post,
                pytest.raises(LLMContextOverflowError),
            ):
                client.chat([{"role": "user", "content": "x" * 10000}])
        post.assert_not_called()

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
        mock_http.post.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=err)
        )
        client = self._client_with_mock_http(mock_http)
        with pytest.raises(LLMClientError, match="авторизации"):
            client.chat([{"role": "user", "content": "x"}])

    def test_chat_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "server error"
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)
        mock_http = MagicMock()
        mock_http.post.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=err)
        )
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
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "thinking..."},
                    "finish_reason": "length",
                }
            ],
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
            "data: [DONE]",
            "",
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
            MockClient.return_value = inst
            result = create_llm_client(cfg)
            assert result is inst
            inst.health_check.assert_not_called()  # по умолчанию check_health=False

    def test_check_health_true_success(self):
        cfg = _cfg()
        with patch("meeting_recorder.llm_client.LLMClient") as MockClient:
            inst = MagicMock()
            inst.health_check.return_value = True
            MockClient.return_value = inst
            result = create_llm_client(cfg, check_health=True)
            assert result is inst
            inst.health_check.assert_called_once()

    def test_unavailable_raises(self):
        cfg = _cfg()
        with patch("meeting_recorder.llm_client.LLMClient") as MockClient:
            inst = MagicMock()
            inst.health_check.return_value = False
            MockClient.return_value = inst
            with pytest.raises(LLMClientError, match="недоступен"):
                create_llm_client(cfg, check_health=True)

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


# ---------------------------------------------------------------------------
# Повторы, обрезанные ответы, транспортные ошибки
# ---------------------------------------------------------------------------

from meeting_recorder.llm_client import LLMTruncatedError  # noqa: E402


def _status_error(status: int, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = f"status {status}"
    resp.headers = headers or {}
    err = httpx.HTTPStatusError(str(status), request=MagicMock(), response=resp)
    return MagicMock(raise_for_status=MagicMock(side_effect=err))


def _client(mock_http: MagicMock, **cfg_kw) -> LLMClient:
    with patch("meeting_recorder.llm_client.httpx.Client", return_value=mock_http):
        return LLMClient(_cfg(**cfg_kw))


class TestRetries:
    def test_retry_on_500_then_success(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = [_status_error(500), _ok_response("ok")]
        assert _client(mock_http).chat([{"role": "user", "content": "x"}]) == "ok"
        assert mock_http.post.call_count == 2

    def test_retry_exhausted(self):
        mock_http = MagicMock()
        mock_http.post.return_value = _status_error(503)
        with pytest.raises(LLMClientError, match="503"):
            _client(mock_http, max_retries=2).chat([{"role": "user", "content": "x"}])
        assert mock_http.post.call_count == 3

    def test_no_retry_on_400(self):
        mock_http = MagicMock()
        mock_http.post.return_value = _status_error(400)
        with pytest.raises(LLMClientError, match="400"):
            _client(mock_http).chat([{"role": "user", "content": "x"}])
        assert mock_http.post.call_count == 1

    def test_no_retry_on_auth_error(self):
        mock_http = MagicMock()
        mock_http.post.return_value = _status_error(403)
        with pytest.raises(LLMClientError, match="авторизации"):
            _client(mock_http).chat([{"role": "user", "content": "x"}])
        assert mock_http.post.call_count == 1

    def test_retry_after_header_respected(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = [
            _status_error(429, {"retry-after": "7"}),
            _ok_response(),
        ]
        with patch("meeting_recorder.llm_client._sleep") as mock_sleep:
            _client(mock_http).chat([{"role": "user", "content": "x"}])
        mock_sleep.assert_called_once_with(7.0)

    def test_zero_retries(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(LLMClientError):
            _client(mock_http, max_retries=0).chat([{"role": "user", "content": "x"}])
        assert mock_http.post.call_count == 1

    def test_read_error_wrapped_and_retried(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = [httpx.ReadError("reset"), _ok_response("ok")]
        assert _client(mock_http).chat([{"role": "user", "content": "x"}]) == "ok"

    def test_remote_protocol_error_wrapped(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.RemoteProtocolError("bad")
        with pytest.raises(LLMClientError, match="Сетевая ошибка"):
            _client(mock_http, max_retries=0).chat([{"role": "user", "content": "x"}])


class TestResponseParsing:
    def test_invalid_json_wrapped(self):
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.side_effect = ValueError("no json")
        mock_http = MagicMock()
        mock_http.post.return_value = r
        with pytest.raises(LLMClientError, match="не-JSON"):
            _client(mock_http).chat([{"role": "user", "content": "x"}])

    def test_empty_choices(self):
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"choices": []}
        mock_http = MagicMock()
        mock_http.post.return_value = r
        with pytest.raises(LLMClientError, match="choices"):
            _client(mock_http).chat([{"role": "user", "content": "x"}])

    def test_truncated_with_content_raises_with_partial(self):
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"content": "half"}, "finish_reason": "length"}],
        }
        mock_http = MagicMock()
        mock_http.post.return_value = r
        with pytest.raises(LLMTruncatedError) as ei:
            _client(mock_http).chat([{"role": "user", "content": "x"}])
        assert ei.value.partial == "half"

    def test_stream_skips_service_chunks(self):
        chunks = [
            ": OPENROUTER PROCESSING",
            'data: {"choices": [{"delta": {"content": "Hi"}}]}',
            'data: {"choices": [], "usage": {"total_tokens": 3}}',
            'data: {"choices": [{"delta": {}}]}',
            "data: [DONE]",
        ]
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.iter_lines = MagicMock(return_value=iter(chunks))
        mock_http = MagicMock()
        mock_http.stream.return_value = resp
        client = _client(mock_http)
        assert client.chat([{"role": "user", "content": "x"}], stream=True) == "Hi"
        assert mock_http.stream.call_args.kwargs["json"]["stream"] is True


# ---------------------------------------------------------------------------
# Таймауты и переполнение контекста
# ---------------------------------------------------------------------------


class TestTimeoutsAndOverflow:
    def _client(self, mock_http: MagicMock, **kw) -> LLMClient:
        with patch("meeting_recorder.llm_client.httpx.Client") as mock_cls:
            mock_cls.return_value = mock_http
            return LLMClient(_cfg(**kw))

    def test_read_timeout_not_retried(self):
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.ReadTimeout("slow")
        client = self._client(mock_http, max_retries=3)
        with pytest.raises(LLMClientError, match="Таймаут ответа"):
            client.chat([{"role": "user", "content": "x"}])
        assert mock_http.post.call_count == 1

    def test_connect_timeout_retried(self):
        mock_http = MagicMock()
        ok = MagicMock()
        ok.raise_for_status.return_value = None
        ok.json.return_value = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
        }
        mock_http.post.side_effect = [httpx.ConnectTimeout("no route"), ok]
        client = self._client(mock_http, max_retries=2)
        assert client.chat([{"role": "user", "content": "x"}]) == "hi"
        assert mock_http.post.call_count == 2

    @pytest.mark.parametrize(
        "body",
        [
            '{"error":{"message":"the request exceeds the available context size"}}',
            "This model's maximum context length is 32768 tokens",
            '{"error":{"code":"context_length_exceeded"}}',
        ],
    )
    def test_context_overflow_detected(self, body):
        from meeting_recorder.llm_client import LLMContextOverflowError

        request = httpx.Request("POST", "http://x/v1/chat/completions")
        response = httpx.Response(400, text=body, request=request)
        mock_http = MagicMock()
        mock_http.post.return_value = response
        client = self._client(mock_http, max_retries=2)
        with pytest.raises(LLMContextOverflowError):
            client.chat([{"role": "user", "content": "x"}])
        assert mock_http.post.call_count == 1  # не повторяется

    def test_plain_400_is_not_overflow(self):
        from meeting_recorder.llm_client import LLMContextOverflowError

        request = httpx.Request("POST", "http://x/v1/chat/completions")
        response = httpx.Response(400, text="invalid temperature", request=request)
        mock_http = MagicMock()
        mock_http.post.return_value = response
        client = self._client(mock_http)
        with pytest.raises(LLMClientError) as exc_info:
            client.chat([{"role": "user", "content": "x"}])
        assert not isinstance(exc_info.value, LLMContextOverflowError)
