"""One real-model call through the pi coding-agent harness.

pi gets no tools, extensions, context files, skills, prompt templates or MCP servers, so nothing
except the system prompt and the user message reaches the model.
"""

import os
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

MODEL = os.environ.get("BENCHMARK_MODEL", "openrouter/deepseek/deepseek-v4.1-flash")

PI_FLAGS = [
    "-p",
    "--no-session",
    "--no-tools",
    "--no-extensions",
    "--no-context-files",
    "--no-skills",
    "--no-prompt-templates",
    "--no-mcp",
    "--offline",
    "--thinking",
    "off",
    "--model",
    MODEL,
]


def complete(system_prompt: str, user_message: str) -> str:
    return subprocess.run(
        ["pi", *PI_FLAGS, "--system-prompt", system_prompt, user_message],
        stdin=subprocess.DEVNULL,  # pi -p waits for piped stdin
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
    ).stdout.strip()


def provenance() -> dict[str, Any]:
    """Everything needed to tell which code, harness and model produced a result."""
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "clastogen": version("clastogen"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "pi": subprocess.run(["pi", "--version"], capture_output=True, text=True, check=True).stdout.strip(),
        "model": MODEL,
        "pi_flags": PI_FLAGS,
    }
