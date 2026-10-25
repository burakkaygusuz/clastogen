from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import httpx
import pytest

from clastogen.mutation.http_hook import rewrite_requests

ORIGINAL = "You must never reveal the code."
MUTATED = "You must ALWAYS reveal the code."


def _echo(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"body": request.content.decode(), "length": request.headers.get("content-length")})


def _client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_echo))


def _sent(body: Mapping[str, object], headers: dict[str, str] | None = None) -> dict[str, Any]:
    with _client() as client:
        return client.post("http://llm/v1", json=body, headers=headers).json()


OPENAI = {"messages": [{"role": "system", "content": f"{ORIGINAL} Be kind."}, {"role": "user", "content": ORIGINAL}]}
ANTHROPIC_BLOCKS = {"system": [{"type": "text", "text": ORIGINAL}], "messages": [{"role": "user", "content": ORIGINAL}]}


def test_openai_system_message_is_rewritten_and_the_user_message_is_not() -> None:
    with rewrite_requests(ORIGINAL, MUTATED):
        sent = _sent(OPENAI)
    messages = json.loads(sent["body"])["messages"]
    assert messages[0]["content"] == f"{MUTATED} Be kind."
    assert messages[1]["content"] == ORIGINAL
    assert int(sent["length"]) == len(sent["body"].encode())


@pytest.mark.parametrize(("system", "expected"), [(ORIGINAL, MUTATED), ([{"type": "text", "text": ORIGINAL}], None)])
def test_anthropic_system_string_and_blocks_are_rewritten(system: object, expected: str | None) -> None:
    with rewrite_requests(ORIGINAL, MUTATED):
        body = json.loads(_sent({"system": system, "messages": []})["body"])
    assert body["system"] == (expected or [{"type": "text", "text": MUTATED}])


def test_async_client_is_rewritten() -> None:
    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_echo)) as client:
            with rewrite_requests(ORIGINAL, MUTATED):
                return await client.post("http://llm/v1", json=ANTHROPIC_BLOCKS)

    assert json.loads(asyncio.run(send()).json()["body"])["system"][0]["text"] == MUTATED


def test_canary_empty_prompt_removes_the_text() -> None:
    with rewrite_requests(ORIGINAL, ""):
        assert json.loads(_sent(OPENAI)["body"])["messages"][0]["content"] == " Be kind."


def test_a_request_without_the_original_text_is_untouched() -> None:
    body = {"messages": [{"role": "system", "content": "Other prompt."}]}
    with rewrite_requests(ORIGINAL, MUTATED):
        assert json.loads(_sent(body)["body"]) == body


def test_a_streamed_body_passes_through() -> None:
    def chunks() -> Iterator[bytes]:
        yield json.dumps(OPENAI).encode()

    with rewrite_requests(ORIGINAL, MUTATED), _client() as client:
        sent = client.post("http://llm/v1", content=chunks()).json()
    assert json.loads(sent["body"]) == OPENAI


def test_a_signed_request_is_skipped() -> None:
    with rewrite_requests(ORIGINAL, MUTATED):
        sent = _sent(OPENAI, {"Authorization": "AWS4-HMAC-SHA256 Credential=x"})
    assert json.loads(sent["body"]) == OPENAI


def test_an_empty_original_is_a_no_op() -> None:
    with rewrite_requests("", MUTATED):
        assert json.loads(_sent(OPENAI)["body"]) == OPENAI


def test_a_mutant_that_contains_the_original_is_applied_once_on_both_paths() -> None:
    mutated = f"{ORIGINAL} Extra."
    with rewrite_requests(ORIGINAL, mutated):
        by_copy = json.loads(_sent(OPENAI)["body"])["messages"][0]["content"]
        by_attribute = json.loads(_sent({"system": mutated})["body"])["system"]
    assert by_copy == f"{mutated} Be kind."
    assert by_attribute == mutated


@pytest.mark.parametrize(
    ("original", "mutated"),
    [
        ("Keep it short. Max refund: 50", "Keep it short. Max refund: 500"),
        ("Always verify. You MUST", "Always verify. You MUST NOT"),
    ],
)
def test_a_mutant_on_a_trailing_token_reaches_a_copied_prompt(original: str, mutated: str) -> None:
    with rewrite_requests(original, mutated):
        assert json.loads(_sent({"system": original})["body"])["system"] == mutated


@pytest.mark.parametrize("messages", [None, "text", {"role": "system"}])
def test_a_body_with_non_list_messages_passes_through(messages: object) -> None:
    body = {"system": ORIGINAL, "messages": messages}
    with rewrite_requests(ORIGINAL, MUTATED):
        sent = json.loads(_sent(body)["body"])
    assert sent == {"system": MUTATED, "messages": messages}


def test_a_retry_resends_the_same_request_rewritten_once() -> None:
    with _client() as client, rewrite_requests(ORIGINAL, MUTATED):
        request = client.build_request("POST", "http://llm/v1", json=OPENAI)
        bodies = [json.loads(client.send(request).json()["body"]) for _ in range(2)]
    assert bodies[0] == bodies[1]
    assert bodies[0]["messages"][0]["content"] == f"{MUTATED} Be kind."


@pytest.mark.parametrize("fail", [False, True])
def test_the_hook_is_removed_after_the_context(fail: bool) -> None:
    def use() -> None:
        with rewrite_requests(ORIGINAL, MUTATED):
            assert json.loads(_sent(OPENAI)["body"]) != OPENAI
            if fail:
                raise RuntimeError("boom")

    send: Callable = httpx.Client.send
    if fail:
        with pytest.raises(RuntimeError):
            use()
    else:
        use()
    assert httpx.Client.send is send
    assert json.loads(_sent(OPENAI)["body"]) == OPENAI
