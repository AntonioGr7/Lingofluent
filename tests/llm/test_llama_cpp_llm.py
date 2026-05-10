import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from lingofluent.llm.llama_cpp_llm import LlamaCppLLM
from lingofluent.llm.llm_base import Message


def _chat_resp(content="ok"):
    return {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "model": "lfm2.5-vl",
    }


class TestLlamaCppLLM(AioHTTPTestCase):

    async def get_application(self):
        app = web.Application()
        app["healthy"] = True

        async def health(request):
            return web.Response(status=200 if app["healthy"] else 503)

        async def chat(request):
            return web.json_response(_chat_resp("Ciao"))

        app.router.add_get("/health", health)
        app.router.add_post("/v1/chat/completions", chat)
        return app

    def _url(self):
        return str(self.server.make_url(""))

    @unittest_run_loop
    async def test_health_uses_slash_health(self):
        async with LlamaCppLLM(base_url=self._url()) as llm:
            assert await llm.health() is True

    @unittest_run_loop
    async def test_health_false_when_down(self):
        self.app["healthy"] = False
        async with LlamaCppLLM(base_url=self._url()) as llm:
            assert await llm.health() is False

    @unittest_run_loop
    async def test_chat_works(self):
        async with LlamaCppLLM(base_url=self._url()) as llm:
            result = await llm.chat([Message(role="user", content="say hi")])
        assert result.text == "Ciao"
        assert result.backend == "llamacpp"

    def test_name_is_llamacpp(self):
        assert LlamaCppLLM(base_url="http://localhost:8080").name == "llamacpp"


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://custom:9999")
    assert LlamaCppLLM().base_url == "http://custom:9999"


def test_default_base_url_without_env(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert LlamaCppLLM().base_url == "http://localhost:8080"
