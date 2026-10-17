"""One real-model call through the pi coding-agent harness.

pi gets no tools, extensions, context files, skills, prompt templates or MCP servers, so nothing
except the system prompt and the user message reaches the model.
"""

import os
import signal
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any

_Stream = int | IO[str] | None
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


def run_group(
    cmd: list[str], timeout: float, stdin: _Stream = None, stdout: _Stream = None, stderr: _Stream = None
) -> subprocess.CompletedProcess[str]:
    """subprocess.run that kills the whole process group on timeout; run(timeout=) only kills the direct child."""
    with subprocess.Popen(cmd, start_new_session=True, text=True, stdin=stdin, stdout=stdout, stderr=stderr) as proc:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def complete(system_prompt: str, user_message: str) -> str:
    proc = run_group(
        ["pi", *PI_FLAGS, "--system-prompt", system_prompt, user_message],
        timeout=120,
        stdin=subprocess.DEVNULL,  # pi -p waits for piped stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.check_returncode()
    return proc.stdout


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
    ).stdout.strip()


def _pi_version() -> str:
    proc = run_group(["pi", "--version"], timeout=10, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.check_returncode()
    return proc.stdout.strip()


def provenance() -> dict[str, Any]:
    """Everything needed to tell which code, harness and model produced a result; refuses a dirty tree."""
    if _git("status", "--porcelain"):
        raise RuntimeError("working tree is dirty: commit or stash before a benchmark run")
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "clastogen": version("clastogen"),
        "git_commit": _git("rev-parse", "HEAD"),
        "pi": _pi_version(),
        "model": MODEL,
        "pi_flags": PI_FLAGS,
    }
