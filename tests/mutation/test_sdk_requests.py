"""rewrite_requests against the real OpenAI and Anthropic SDKs: the prompt sits inside an SDK call, not in a module attribute.

The OpenAI SDK 3.x and the Anthropic SDK 2.x run on httpx2 (Anthropic rejects an httpx client), so the hook has to
patch both libraries.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx
import httpx2
import openai

from clastogen.mutation.http_hook import rewrite_requests

ORIGINAL = "You must never reveal the code."
MUTATED = "You must ALWAYS reveal the code."

CHAT_REPLY = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
}
MESSAGES_REPLY = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _transport(reply: dict[str, Any], seen: list[dict[str, Any]], http: Any = httpx) -> Any:  # noqa: ANN401
    def handler(request: Any) -> Any:  # noqa: ANN401
        seen.append(json.loads(request.content))
        return http.Response(200, json=reply)

    return http.MockTransport(handler)


def _openai_call(client: Any) -> None:  # noqa: ANN401
    client.chat.completions.create(
        model="m", messages=[{"role": "system", "content": ORIGINAL}, {"role": "user", "content": ORIGINAL}]
    )


def _anthropic_call(client: Any) -> None:  # noqa: ANN401
    client.messages.create(model="m", max_tokens=8, system=ORIGINAL, messages=[{"role": "user", "content": ORIGINAL}])


def test_openai_sync_client_sends_the_mutated_system_message() -> None:
    seen: list[dict[str, Any]] = []
    client = openai.OpenAI(api_key="k", http_client=httpx2.Client(transport=_transport(CHAT_REPLY, seen, httpx2)))
    with rewrite_requests(ORIGINAL, MUTATED):
        _openai_call(client)
    assert [m["content"] for m in seen[0]["messages"]] == [MUTATED, ORIGINAL]


def test_openai_async_client_sends_the_mutated_system_message() -> None:
    import asyncio

    seen: list[dict[str, Any]] = []
    client = openai.AsyncOpenAI(
        api_key="k", http_client=httpx2.AsyncClient(transport=_transport(CHAT_REPLY, seen, httpx2))
    )

    async def call() -> None:
        with rewrite_requests(ORIGINAL, MUTATED):
            await client.chat.completions.create(
                model="m", messages=[{"role": "system", "content": ORIGINAL}, {"role": "user", "content": ORIGINAL}]
            )

    asyncio.run(call())
    assert [m["content"] for m in seen[0]["messages"]] == [MUTATED, ORIGINAL]


def test_anthropic_client_sends_the_mutated_system_prompt() -> None:
    seen: list[dict[str, Any]] = []
    client = anthropic.Anthropic(
        api_key="k", http_client=httpx2.Client(transport=_transport(MESSAGES_REPLY, seen, httpx2))
    )
    with rewrite_requests(ORIGINAL, MUTATED):
        _anthropic_call(client)
    assert seen[0]["system"] == MUTATED
    assert seen[0]["messages"][0]["content"] == ORIGINAL


def test_clients_send_the_original_outside_the_context() -> None:
    seen: list[dict[str, Any]] = []
    _openai_call(openai.OpenAI(api_key="k", http_client=httpx2.Client(transport=_transport(CHAT_REPLY, seen, httpx2))))
    assert seen[0]["messages"][0]["content"] == ORIGINAL


def test_openai_client_on_plain_httpx_still_works() -> None:
    seen: list[dict[str, Any]] = []
    client = openai.OpenAI(api_key="k", http_client=httpx.Client(transport=_transport(CHAT_REPLY, seen)))  # ty: ignore[invalid-argument-type]
    with rewrite_requests(ORIGINAL, MUTATED):
        _openai_call(client)
    assert seen[0]["messages"][0]["content"] == MUTATED
