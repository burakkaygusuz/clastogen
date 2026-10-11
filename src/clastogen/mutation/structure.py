from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_HEADING = re.compile(
    r"^(#{1,6})[ \t]+(\S.*)$"
)  # \S after the spaces: the two parts cannot overlap, so no backtracking
_XML_BLOCK = re.compile(r"<([A-Za-z][\w-]*)(?:\s[^>]*)?>(.*?)</\1>", re.DOTALL)
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_EXAMPLE = re.compile(r"\bexamples?\b", re.IGNORECASE)


@dataclass(frozen=True)
class Block:
    """A span prompt[start:end] that deleting leaves the rest of the prompt well formed."""

    kind: Literal["section", "example"]
    start: int
    end: int
    label: str


def _xml_blocks(prompt: str) -> list[Block]:
    blocks: list[Block] = []
    pending = [(0, len(prompt))]  # explicit worklist: nesting depth is untrusted, so no recursion
    while pending:
        pos, endpos = pending.pop()
        for m in _XML_BLOCK.finditer(prompt, pos, endpos):
            name = m[1]
            is_example = name.lower().startswith("example")
            # ponytail: nested same-name tags pair first-close; use a stack if bug reports appear
            if not (
                is_example and "<example" in m[2].lower()
            ):  # a wrapper such as <examples> is kept: its children are the blocks
                blocks.append(Block("example" if is_example else "section", m.start(), m.end(), f"<{name}>"))
            pending.append((m.start(2), m.end(2)))
    return blocks


def find_blocks(prompt: str) -> list[Block]:
    """Finds Markdown heading sections and XML tag blocks, ordered by position; fenced code is not scanned for headings."""
    blocks: list[Block] = []
    headings: list[tuple[int, int, str]] = []  # (level, start, title)
    fence: str = ""  # the opening run of ` or ~ while inside a fenced block
    offset = 0
    for line in prompt.splitlines(keepends=True):
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
