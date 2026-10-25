from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    import httpx

# httpx2 is the library of the Anthropic SDK 2.x; a client of one library is rejected by an SDK built on the other.
_HTTP_LIBRARIES = ("httpx", "httpx2")
_SYSTEM_ROLES = ("system", "developer")
_SYSTEM_FIELDS = ("system", "instructions", "preamble")  # Anthropic and Bedrock, OpenAI Responses, Cohere v1
_MESSAGE_LISTS = ("messages", "input")  # Chat Completions and Anthropic, Responses
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


def _rewritten_body(body: dict[str, Any], original: str, mutated: str) -> dict[str, Any]:
    """Returns a copy of a request body with original replaced in its system positions."""
    new = copy.deepcopy(body)
    for field in _SYSTEM_FIELDS:
        if field in new:
            new[field] = _replace_text(new[field], original, mutated)
    if isinstance(gemini := new.get("systemInstruction"), dict) and "parts" in gemini:
        gemini["parts"] = _replace_text(gemini["parts"], original, mutated)
    for key in _MESSAGE_LISTS:
        for message in new[key] if isinstance(new.get(key), list) else []:
            if isinstance(message, dict) and message.get("role") in _SYSTEM_ROLES:
                message["content"] = _replace_text(message.get("content"), original, mutated)
    return new


def _rewritten(request: httpx.Request, original: str, mutated: str) -> httpx.Request:
    """Returns request with original replaced in its system positions, or request itself when nothing changes."""
    if request.headers.get("authorization", "").startswith(_SIGV4):
        return request  # the signature covers the body, so a rewritten body would be rejected
    try:
        body = json.loads(request.content)
    except (ValueError, RuntimeError):  # bad JSON, or RequestNotRead (a RuntimeError)
        return request
    if not isinstance(body, dict):
        return request
    new = _rewritten_body(body, original, mutated)
    if new == body:
        return request
    headers = [(k, v) for k, v in request.headers.multi_items() if k.lower() != "content-length"]
    content = json.dumps(new, ensure_ascii=False).encode()
    return type(request)(request.method, request.url, headers=headers, content=content, extensions=request.extensions)


@contextmanager
def _patch_send(http: ModuleType, original: str, mutated: str) -> Generator[None, None, None]:
    send, async_send = http.Client.send, http.AsyncClient.send

    def patched_send(self: httpx.Client, request: httpx.Request, **kwargs: object) -> httpx.Response:
        return send(self, _rewritten(request, original, mutated), **kwargs)

    async def patched_async_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: object) -> httpx.Response:
        return await async_send(self, _rewritten(request, original, mutated), **kwargs)

    with patch.object(http.Client, "send", patched_send), patch.object(http.AsyncClient, "send", patched_async_send):
        yield


@contextmanager
def rewrite_requests(original: object, mutated: str) -> Generator[None, None, None]:
    """Rewrites the original prompt text in the system positions of every httpx request sent inside the context.

    This reaches prompts that a module attribute cannot: copies made at import time and prompts held inside an SDK
    object. It patches every HTTP library of _HTTP_LIBRARIES that is installed, because an SDK accepts only the
    client class of its own library. It is a no-op when none is installed, and when there is no original text.
    """
    if not isinstance(original, str) or original == "":
        yield
        return
    with ExitStack() as stack:
        for name in _HTTP_LIBRARIES:
            try:
                http = importlib.import_module(name)
            except ImportError:
                continue
            stack.enter_context(_patch_send(http, original, mutated))
        yield
