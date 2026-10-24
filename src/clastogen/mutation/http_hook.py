from __future__ import annotations

import copy
import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    import httpx

_SYSTEM_ROLES = ("system", "developer")
_SIGV4 = "AWS4-HMAC-SHA256"


def _swap(text: str, original: str, mutated: str) -> str:
    """Replaces original by mutated, unless text already holds the mutant (the attribute path got there first)."""
    return text if original in mutated and mutated in text else text.replace(original, mutated)


def _replace_text(content: object, original: str, mutated: str) -> object:
    """Replaces original in a message content: a string, or a list of blocks that have a "text" string."""
    if isinstance(content, str):
        return _swap(content, original, mutated)
    if isinstance(content, list):
        return [
            {**block, "text": _swap(block["text"], original, mutated)}
            if isinstance(block, dict) and isinstance(block.get("text"), str)
            else block
            for block in content
        ]
    return content


def _rewritten(request: httpx.Request, original: str, mutated: str) -> httpx.Request:
    """Returns request with original replaced in its system positions, or request itself when nothing changes."""
    if request.headers.get("authorization", "").startswith(_SIGV4):
        return request  # the signature covers the body, so a rewritten body would be rejected
    try:
        body = json.loads(request.content)  # RequestNotRead for a streamed body
    except (ValueError, RuntimeError):  # bad JSON, or RequestNotRead (a RuntimeError)
        return request
    if not isinstance(body, dict):
        return request
    new = copy.deepcopy(body)
    if "system" in new:  # Anthropic
        new["system"] = _replace_text(new["system"], original, mutated)
    for message in messages if isinstance(messages := new.get("messages"), list) else []:  # OpenAI
        if isinstance(message, dict) and message.get("role") in _SYSTEM_ROLES:
            message["content"] = _replace_text(message.get("content"), original, mutated)
    if new == body:
        return request
    headers = [(k, v) for k, v in request.headers.multi_items() if k.lower() != "content-length"]
    content = json.dumps(new, ensure_ascii=False).encode()
    return type(request)(request.method, request.url, headers=headers, content=content, extensions=request.extensions)


@contextmanager
def rewrite_requests(original: object, mutated: str) -> Generator[None, None, None]:
    """Rewrites the original prompt text in the system positions of every httpx request sent inside the context.

    This reaches prompts that a module attribute cannot: copies made at import time and prompts held inside an SDK
    object. It is a no-op when httpx is not installed, and when there is no original text.
    """
    if not isinstance(original, str) or original == "":
        yield
        return
    try:
        import httpx
    except ImportError:
        yield
        return

    send, async_send = httpx.Client.send, httpx.AsyncClient.send

    def patched_send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        return send(self, _rewritten(request, original, mutated), **kwargs)

    async def patched_async_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        return await async_send(self, _rewritten(request, original, mutated), **kwargs)

    with patch.object(httpx.Client, "send", patched_send), patch.object(httpx.AsyncClient, "send", patched_async_send):
        yield
