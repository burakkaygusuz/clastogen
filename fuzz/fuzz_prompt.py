"""Fuzzes prompt mutation and Markdown escaping of untrusted prompt text.

Atheris ships Linux x86_64 wheels for CPython 3.12+ only:
uv run --with atheris==3.1.0 python fuzz/fuzz_prompt.py -max_total_time=60
"""

import re
import sys

import atheris  # pyrefly: ignore[missing-source-for-stubs]

with atheris.instrument_imports():
    from clastogen.mutation.mutator import PromptMutator
    from clastogen.reporting.render import _code

MUTATOR = PromptMutator()


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    max_mutants = fdp.ConsumeIntInRange(1, 8)
    prompt = fdp.ConsumeUnicodeNoSurrogates(fdp.remaining_bytes())

    mutants = MUTATOR.generate_mutants(prompt, max_mutants)
    assert len(mutants) <= max_mutants
    assert all(m.mutated_prompt != prompt for m in mutants)

    cell = _code(prompt)
    fence = re.match("`+", cell)
    assert fence
    assert "\n" not in cell
    assert cell.endswith(f" {fence[0]}")
    assert all(len(run) < len(fence[0]) for run in re.findall("`+", cell[len(fence[0]) : -len(fence[0])]))


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
