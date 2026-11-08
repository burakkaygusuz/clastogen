from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*)$")
_XML_TAG = re.compile(r"<([A-Za-z][\w-]*)(?:\s[^<>]*)?>|</([A-Za-z][\w-]*)>")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_LINE = re.compile(r"[^\n]+\n?|\n")
_EXAMPLE = re.compile(r"\bexamples?\b", re.IGNORECASE)


@dataclass(frozen=True)
class Block:
    """A span prompt[start:end] that deleting leaves the rest of the prompt well formed."""

    kind: Literal["section", "example"]
    start: int
    end: int
    label: str


def _xml_blocks(prompt: str) -> list[Block]:
    """Pairs XML tags with a stack: a close tag matches the nearest open tag of its name and drops the tags opened after it."""
    blocks: list[Block] = []
    open_tags: list[tuple[str, int, int]] = []
    for m in _XML_TAG.finditer(prompt):
        if m[1]:
            open_tags.append((m[1], m.start(), m.end()))
            continue
        i = next((i for i in range(len(open_tags) - 1, -1, -1) if open_tags[i][0] == m[2]), None)
        if i is None:
            continue
        name, start, content_start = open_tags[i]
        del open_tags[i:]
        is_example = name.lower().startswith("example")
        if not (is_example and "<example" in prompt[content_start : m.start()].lower()):
            blocks.append(Block("example" if is_example else "section", start, m.end(), f"<{name}>"))
    return blocks


def find_blocks(prompt: str) -> list[Block]:
    """Finds Markdown heading sections and XML tag blocks, ordered by position; fenced code is not scanned for headings."""
    blocks: list[Block] = []
    headings: list[tuple[int, int, str]] = []
    fence: str = ""
    offset = 0
    for line in _LINE.findall(prompt):
        if m := _FENCE.match(line):
            run = m[1]
            if not fence:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not line.lstrip().strip(run[0]).strip():
                fence = ""
        elif not fence and (m := _HEADING.match(line.rstrip())):
            headings.append((len(m[1]), offset, m[2]))
        offset += len(line)
    for i, (level, start, title) in enumerate(headings):
        end = next((s for lv, s, _ in headings[i + 1 :] if lv <= level), len(prompt))
        blocks.append(Block("example" if _EXAMPLE.search(title) else "section", start, end, title))
    blocks += _xml_blocks(prompt)
    return sorted(blocks, key=lambda b: (b.start, -b.end))
