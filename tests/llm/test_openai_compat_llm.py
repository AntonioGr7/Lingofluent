import base64
import json
import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from lingofluent.llm.llm_base import (
    ChatOptions,
    ImagePart,
    LLMAuthError,
    LLMInputError,
    LLMRateLimitError,
    LLMUnavailableError,
    Message,
    TextPart,
    Tool,
)
from lingofluent.llm.openai_compat_llm import OpenAICompatLLM


def _chat_resp(content="hello", tool_calls=None, finish_reason="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "test-model",
    }


def _sse_body(tokens, finish_reason="stop"):
    lines = []
    for tok in tokens:
        lines.append(f'data: {json.dumps({"choices": [{"delta": {"content": tok}, "finish_reason": None}]})}\n\n')
    lines.append(
        f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": finish_reason}], "usage": {"prompt_tokens": 3, "completion_tokens": len(tokens), "total_tokens": 3 + len(tokens)}})}\n\n'
    )
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class TestOpenAICompatLLM(AioHTTPTestCase):
    """All routes registered upfront; app dict carries mutable state."""

    async def get_application(self):
        app = web.Application()
        app["resp"] = _chat_resp("Buongiorno")
        app["status"] = 200
        app["captured"] = {}

        async def chat(request):
            body = await request.json()
            app["captured"]["body"] = body
            app["captured"]["auth"] = request.headers.get("Authorization")
            return web.json_response(app["resp"], status=app["status"])

        async def models(request):
            return web.json_response({"data": []})

        app.router.add_post("/v1/chat/completions", chat)
        app.router.add_get("/v1/models", models)
        return app

    def _url(self):
        return str(self.server.make_url(""))

    @unittest_run_loop
    async def test_health_ok(self):
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            assert await llm.health() is True

    @unittest_run_loop
    async def test_simple_chat(self):
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            result = await llm.chat([Message(role="user", content="hello")])
        assert result.text == "Buongiorno"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 8
        assert result.model == "test-model"
        assert result.backend == "openai-compat"

    @unittest_run_loop
    async def test_401_raises_auth_error(self):
        self.app["status"] = 401
        self.app["resp"] = {"error": {"message": "Unauthorized"}}
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            with pytest.raises(LLMAuthError):
                await llm.chat([Message(role="user", content="hi")])

    @unittest_run_loop
    async def test_403_raises_auth_error(self):
        self.app["status"] = 403
        self.app["resp"] = {"error": {"message": "Forbidden"}}
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            with pytest.raises(LLMAuthError):
                await llm.chat([Message(role="user", content="hi")])

    @unittest_run_loop
    async def test_429_raises_rate_limit(self):
        self.app["status"] = 429
        self.app["resp"] = {"error": {"message": "Rate limited"}}
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            with pytest.raises(LLMRateLimitError):
                await llm.chat([Message(role="user", content="hi")])

    @unittest_run_loop
    async def test_500_raises_unavailable(self):
        self.app["status"] = 500
        self.app["resp"] = {"error": {"message": "Internal error"}}
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            with pytest.raises(LLMUnavailableError):
                await llm.chat([Message(role="user", content="hi")])

    @unittest_run_loop
    async def test_400_raises_input_error(self):
        self.app["status"] = 400
        self.app["resp"] = {"error": {"message": "Bad request"}}
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            with pytest.raises(LLMInputError, match="Bad request"):
                await llm.chat([Message(role="user", content="hi")])

    @unittest_run_loop
    async def test_tool_call_parsed(self):
        self.app["resp"] = _chat_resp(
            content=None,
            tool_calls=[{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Rome"}'},
            }],
            finish_reason="tool_calls",
        )
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            result = await llm.chat([Message(role="user", content="weather?")])
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Rome"}

    @unittest_run_loop
    async def test_malformed_tool_args_kept_as_raw(self):
        self.app["resp"] = _chat_resp(
            content=None,
            tool_calls=[{
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "fn", "arguments": "{invalid"},
            }],
        )
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            result = await llm.chat([Message(role="user", content="hi")])
        assert result.tool_calls[0].arguments == {"_raw": "{invalid"}

    @unittest_run_loop
    async def test_vision_image_url_serialized(self):
        self.app["resp"] = _chat_resp("ok")
        self.app["status"] = 200
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            await llm.chat([Message(role="user", content=[
                TextPart(text="describe"),
                ImagePart(url="https://example.com/img.png"),
            ])])
        content = self.app["captured"]["body"]["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/img.png"

    @unittest_run_loop
    async def test_vision_image_data_base64(self):
        self.app["resp"] = _chat_resp("ok")
        self.app["status"] = 200
        raw = b"\xff\xd8\xff"
        async with OpenAICompatLLM(base_url=self._url()) as llm:
            await llm.chat([Message(role="user", content=[
                ImagePart(data=raw, mime="image/jpeg"),
            ])])
        expected = f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"
        content = self.app["captured"]["body"]["messages"][0]["content"]
        assert content[0]["image_url"]["url"] == expected

    @unittest_run_loop
    async def test_api_key_in_header(self):
        self.app["resp"] = _chat_resp("ok")
        self.app["status"] = 200
        async with OpenAICompatLLM(base_url=self._url(), api_key="sk-test") as llm:
            await llm.chat([Message(role="user", content="hi")])
        assert self.app["captured"]["auth"] == "Bearer sk-test"


class TestOpenAICompatStreaming(AioHTTPTestCase):

    async def get_application(self):
        app = web.Application()

        async def stream(request):
            return web.Response(
                body=_sse_body(["Hello", " world", "!"]),
                content_type="text/event-stream",
            )

        app.router.add_post("/v1/chat/completions", stream)
        app.router.add_get("/v1/models", lambda r: web.json_response({"data": []}))
        return app

    @unittest_run_loop
    async def test_streaming_yields_all_chunks(self):
        async with OpenAICompatLLM(base_url=str(self.server.make_url(""))) as llm:
            texts = []
            finish = None
            async for chunk in llm.chat_stream([Message(role="user", content="hi")]):
                if chunk.delta_text:
                    texts.append(chunk.delta_text)
                if chunk.finish_reason:
                    finish = chunk.finish_reason
        assert "".join(texts) == "Hello world!"
        assert finish == "stop"
