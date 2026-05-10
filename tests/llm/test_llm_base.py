import pytest
from lingofluent.llm.llm_base import (
    ChatOptions,
    ChatResult,
    ImagePart,
    LLMBackend,
    LLMInputError,
    LLMTimeoutError,
    Message,
    TextPart,
    Tool,
    ToolCall,
)


class _Stub(LLMBackend):
    name = "stub"

    async def _chat(self, messages, options):
        return ChatResult(text="ok", backend=self.name)

    async def health(self):
        return True


# --- ImagePart validation ---------------------------------------------------

def test_image_part_requires_url_or_data():
    with pytest.raises(ValueError):
        ImagePart()

def test_image_part_url():
    p = ImagePart(url="https://example.com/img.png")
    assert p.url == "https://example.com/img.png"

def test_image_part_data():
    p = ImagePart(data=b"\xff\xd8", mime="image/jpeg")
    assert p.data == b"\xff\xd8"


# --- _validate_request ------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_messages_raises():
    with pytest.raises(LLMInputError, match="empty"):
        await _Stub().chat([])

@pytest.mark.asyncio
async def test_vision_without_capability_raises():
    backend = _Stub()
    backend.supports_vision = False
    msgs = [Message(role="user", content=[ImagePart(url="http://x.com/a.png")])]
    with pytest.raises(LLMInputError, match="vision"):
        await backend.chat(msgs)

@pytest.mark.asyncio
async def test_tools_without_capability_raises():
    backend = _Stub()
    backend.supports_tools = False
    msgs = [Message(role="user", content="hi")]
    opts = ChatOptions(tools=[Tool(name="fn", description="d", parameters={})])
    with pytest.raises(LLMInputError, match="tool"):
        await backend.chat(msgs, opts)

@pytest.mark.asyncio
async def test_json_mode_without_capability_raises():
    backend = _Stub()
    backend.supports_json_mode = False
    msgs = [Message(role="user", content="hi")]
    opts = ChatOptions(response_format="json_object")
    with pytest.raises(LLMInputError, match="json"):
        await backend.chat(msgs, opts)

@pytest.mark.asyncio
async def test_tool_role_without_tool_call_id_raises():
    backend = _Stub()
    msgs = [
        Message(role="user", content="hi"),
        Message(role="tool", content="result"),
    ]
    with pytest.raises(LLMInputError, match="tool_call_id"):
        await backend.chat(msgs)

@pytest.mark.asyncio
async def test_timeout_raises_llm_timeout_error():
    import asyncio

    class _Slow(_Stub):
        async def _chat(self, messages, options):
            await asyncio.sleep(10)
            return ChatResult(text="done", backend=self.name)

    with pytest.raises(LLMTimeoutError):
        await _Slow().chat(
            [Message(role="user", content="hi")],
            ChatOptions(timeout=0.01),
        )

@pytest.mark.asyncio
async def test_chat_many_returns_list():
    msgs = [Message(role="user", content="hi")]
    results = await _Stub().chat_many([msgs, msgs, msgs])
    assert len(results) == 3
    assert all(r.text == "ok" for r in results)

@pytest.mark.asyncio
async def test_stream_without_capability_raises():
    backend = _Stub()
    backend.supports_streaming = False
    with pytest.raises(NotImplementedError):
        async for _ in backend.chat_stream([Message(role="user", content="hi")]):
            pass
