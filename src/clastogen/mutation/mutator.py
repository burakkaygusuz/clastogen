from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from decimal import Decimal, localcontext
from functools import partial
from itertools import zip_longest
from typing import NamedTuple

from clastogen.models import CUSTOM_OPERATOR, CustomMutant, Mutant, snippet_covers
from clastogen.mutation.structure import Block, find_blocks

_LIST_MARKER = r"(?:[-*]|\d+[.)])\s+"


class Rule(NamedTuple):
    """A candidate rule and where it sits in the prompt, so edits hit that occurrence only."""

    start: int
    text: str

    @property
    def end(self) -> int:
        return self.start + len(self.text)


Operator = Callable[[str, Rule], Mutant | None]


def _clip(text: str, limit: int) -> str:
    """Shortens text to at most limit characters at a word boundary and marks the cut with an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit] if text[limit].isspace() else text[:limit].rsplit(maxsplit=1)[0]
    return head.rstrip(" ,.;:") + "…"


def _bullet_start(text: str) -> int | None:
    """Returns where a list marker (-, *, 1. or 1)) ending text starts its line, or None; no regex, so it stays linear."""
    if text.endswith(("-", "*")):
        start = len(text) - 1
    else:
        digits = text[:-1].rstrip("0123456789")
        if not text.endswith((".", ")")) or len(digits) == len(text) - 1:
            return None
        start = len(digits)
    head = text[:start].rstrip(" \t")
    return start if not head or head.endswith("\n") else None


def _word_table(*pairs: tuple[str, str]) -> tuple[tuple[re.Pattern[str], str], ...]:
    """Compiles (regex fragment, replacement) pairs as case-insensitive whole-word patterns; fragments are not escaped."""
    return tuple((re.compile(rf"\b{word}\b", re.IGNORECASE), new) for word, new in pairs)


class PromptMutator:
    """Extracts load-bearing imperative constraints and generates targeted prompt mutants."""

    RULE_PATTERN = re.compile(
        r"\b(?:never|do not|don't|must not|mustn't|should not|shouldn't|cannot|can't|always|must|shall not|may not|avoid|only|requires?|required"
        r"|if|unless|except|at most|at least|no more than|no fewer than|fewer than|within|without"
        r"|prohibited|not allowed|not permitted|escalate|refuse|reject|decline|approve)\b|\d",
        re.IGNORECASE,
    )

    ROLE_PATTERN = re.compile(r"you are (?:an?|the)\b", re.IGNORECASE)

    SENTENCE_SPLIT_PATTERN = re.compile(r"\n+|(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.?!]\s+")

    NEGATION_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = _word_table(
        (r"do not", "do"),
        (r"don't", "do"),
        (r"must not", "must"),
        (r"mustn't", "must"),
        (r"never", "always"),
        (r"cannot", "can"),
        (r"can't", "can"),
        (r"should not", "should"),
        (r"shouldn't", "should"),
        (r"shall not", "shall"),
        (r"may not", "may"),
        (r"always", "never"),
        (r"must", "must not"),
        (r"avoid", "prefer"),
    )

    WEAKENING_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = _word_table(
        (r"must not", "should not"),
        (r"mustn't", "shouldn't"),
        (r"never", "rarely"),
        (r"always", "usually"),
        (r"must", "should"),
        (r"shall not", "should not"),
        (r"cannot", "should not"),
        (r"can't", "shouldn't"),
        (r"required", "recommended"),
        (r"requires", "recommends"),
        (r"only", "preferably"),
    )

    COMPARATOR_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = _word_table(
        (r"no more than", "at least"),
        (r"no fewer than", "at most"),
        (r"at most", "at least"),
        (r"at least", "at most"),
        (r"fewer than", "more than"),
        (r"more than", "fewer than"),
        (r"maximum", "minimum"),
        (r"minimum", "maximum"),
    )

    SCOPE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = _word_table(
        (r"(?:all|every|each)", "some"),
        (r"both", "either"),
        (r"any", "every"),
    )

    EXCEPTION_PATTERN = re.compile(r"(?<=\S),?\s+(?:unless|except)\b[^.;\n,]*", re.IGNORECASE)

    NUMBER_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")

    def __init__(self, target_symbol: str = "SYSTEM_PROMPT", custom: Sequence[CustomMutant] = ()) -> None:
        self.target_symbol = target_symbol
        self.custom = [c for c in custom if c.target == target_symbol]

    def _sentences(self, prompt: str) -> list[Rule]:
        """Splits the prompt into sentences and list items longer than 10 characters, minus role statements."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        sentences: list[Rule] = []
        begin = 0
        for sep in [*self.SENTENCE_SPLIT_PATTERN.finditer(prompt), None]:
            line = prompt[begin : sep.start() if sep else len(prompt)]
            stripped = line.strip()
            marker = re.match(_LIST_MARKER, stripped)
            skipped = len(line) - len(line.lstrip()) + (marker.end() if marker else 0)
            cleaned = stripped[marker.end() if marker else 0 :]
            if len(cleaned) > 10 and not self.ROLE_PATTERN.match(cleaned):
                sentences.append(Rule(begin + skipped, cleaned))
            begin = sep.end() if sep else begin
        return sentences

    def _rule_spans(self, prompt: str) -> list[Rule]:
        """Finds candidate load-bearing sentences with their positions: those with an obligation, condition or action word or a number."""
        return [s for s in self._sentences(prompt) if self.RULE_PATTERN.search(s.text)]

    def _outside_examples(self, prompt: str) -> list[Rule]:
        """The sentences that are not inside an example block."""
        examples = [b for b in find_blocks(prompt) if b.kind == "example"]
        return [s for s in self._sentences(prompt) if not any(b.start <= s.start < b.end for b in examples)]

    def _custom_covered(self, prompt: str) -> set[str]:
        """The sentences outside example blocks that a custom mutant's original snippet equals or sits inside."""
        return {
            s.text
            for s in self._outside_examples(prompt)
            if any(snippet_covers(c.original, s.text) for c in self.custom)
        }

    def extract_candidate_rules(self, prompt: str) -> list[str]:
        """Extracts the text of the candidate rules found by _rule_spans, plus the sentences a custom mutant covers."""
        covered = self._custom_covered(prompt)
        return [s.text for s in self._sentences(prompt) if self.RULE_PATTERN.search(s.text) or s.text in covered]

    def extract_unmatched_sentences(self, prompt: str) -> list[str]:
        """Extracts the sentences that rule discovery skips, so no rule-level operator makes a mutant for them; example blocks are left out."""
        covered = self._custom_covered(prompt)
        return [
            s.text
            for s in self._outside_examples(prompt)
            if not self.RULE_PATTERN.search(s.text) and s.text not in covered
        ]

    def _delete_mutant(self, prompt: str, rule: Rule) -> Mutant | None:
        """Removes rule with its bullet and trailing punctuation; None when the prompt is unchanged."""
        before = prompt[: rule.start]
        trimmed = before.rstrip()
        bullet = _bullet_start(trimmed)
        head = (trimmed[:bullet] if bullet is not None and trimmed != before else before).rstrip(" \t")
        after = prompt[rule.end :]
        rest = re.sub(r"^[.?!]?[ \t]*\n?", "", after)
        if not head or head.endswith("\n"):
            joiner = ""
        elif after[: len(after) - len(rest)].endswith("\n"):
            joiner = "\n"
        else:
            joiner = " " if rest else ""
        mutated = (head + joiner + rest).strip()
        mutated = re.sub(r"\n\s*\.\s*\n", "\n", mutated)
        mutated = re.sub(r"\n{3,}", "\n\n", mutated)
        if mutated == prompt:
            return None
        return Mutant.create(
            target_symbol=self.target_symbol,
            operator_name="delete_constraint",
            original_snippet=rule.text,
            mutated_snippet="[DELETED]",
            mutated_prompt=mutated,
            description=f"Deleted constraint: '{_clip(rule.text, 60)}'",
        )

    def _replace_mutant(
        self,
        prompt: str,
        rule: Rule,
        replacements: tuple[tuple[re.Pattern[str], str], ...],
        operator_name: str,
        verb: str,
    ) -> Mutant | None:
        """Applies the first matching replacement in rule; None when none applies."""
        for pattern, replacement in replacements:
            if pattern.search(rule.text):
                changed = pattern.sub(replacement.upper(), rule.text, count=1)
                mutated = prompt[: rule.start] + changed + prompt[rule.end :]
                if mutated != prompt:
                    return Mutant.create(
                        target_symbol=self.target_symbol,
                        operator_name=operator_name,
                        original_snippet=rule.text,
                        mutated_snippet=changed,
                        mutated_prompt=mutated,
                        description=f"{verb} constraint: '{_clip(rule.text, 45)}' -> '{_clip(changed, 45)}'",
                    )
        return None

    def _threshold_mutant(self, prompt: str, rule: Rule) -> Mutant | None:
        """Multiplies the first number in rule by 10 (a $50 limit becomes $500); None when rule has no number."""
        match = self.NUMBER_PATTERN.search(rule.text)
        if match is None:
            return None
        raw = match.group()
        with localcontext(prec=len(raw) + 2):
            scaled = Decimal(raw.replace(",", "")) * 10
        number = format(scaled, ",f" if "," in raw else "f")
        changed = rule.text[: match.start()] + number + rule.text[match.end() :]
        mutated = prompt[: rule.start] + changed + prompt[rule.end :]
        if mutated == prompt:
            return None
        return Mutant.create(
            target_symbol=self.target_symbol,
            operator_name="change_threshold",
            original_snippet=rule.text,
            mutated_snippet=changed,
            mutated_prompt=mutated,
            description=f"Changed threshold: '{raw}' -> '{number}' in '{_clip(rule.text, 50)}'",
        )

    def _block_mutant(self, prompt: str, block: Block) -> Mutant | None:
        """Removes the whole section or example with its heading or tags; None when the prompt is unchanged."""
        mutated = re.sub(r"\n{3,}", "\n\n", prompt[: block.start] + prompt[block.end :]).strip()
        if mutated == prompt:
            return None
        return Mutant.create(
            target_symbol=self.target_symbol,
            operator_name=f"delete_{block.kind}",
            original_snippet=prompt[block.start : block.end],
            mutated_snippet="[DELETED]",
            mutated_prompt=mutated,
            description=f"Deleted {block.kind}: '{_clip(block.label, 60)}'",
        )

    def _eligible_blocks(self, prompt: str, rules: Sequence[Rule]) -> list[Block]:
        """Keeps examples and sections that hold a rule, and drops blocks over half of the prompt."""
        seen: set[str] = set()
        eligible: list[Block] = []
        for b in find_blocks(prompt):
            text = prompt[b.start : b.end]
            if (
                len(text) * 2 <= len(prompt)
                and (b.kind == "example" or any(b.start <= rule.start and rule.end <= b.end for rule in rules))
                and text not in seen
            ):
                seen.add(text)
                eligible.append(b)
        return eligible

    def custom_mutants(self, prompt: str) -> list[Mutant]:
        """Builds the custom mutants of this target; an original that is unchanged or not in the prompt exactly once raises ValueError."""
        mutants: list[Mutant] = []
        for c in self.custom:
            if c.mutated == c.original:
                raise ValueError(
                    f"Custom mutant for '{c.target}' changes nothing: original == mutated ({c.original!r})"
                )
            found = prompt.count(c.original)
            if found != 1:
                hint = " (check Unicode normalization (NFC) and line endings)" if not found else ""
                raise ValueError(
                    f"Custom mutant original {c.original!r} must occur exactly once in the prompt of '{c.target}', found {found}{hint}"
                )
            mutants.append(
                Mutant.create(
                    target_symbol=c.target,
                    operator_name=CUSTOM_OPERATOR,
                    original_snippet=c.original,
                    mutated_snippet=c.mutated,
                    mutated_prompt=prompt.replace(c.original, c.mutated),
                    description=c.description,
                )
            )
        return mutants

    def generate_mutants(self, prompt: str, max_mutants: int) -> Sequence[Mutant]:
        """Custom mutants first, then up to max_mutants built-in ones; a mutant whose prompt an earlier one already makes is dropped."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        if max_mutants < 1:
            raise ValueError(f"max_mutants must be >= 1, got {max_mutants}")
        unique: dict[str, Mutant] = {}
        for m in [*self.custom_mutants(prompt), *self._builtin_mutants(prompt, max_mutants)]:
            unique.setdefault(m.mutated_prompt, m)
        if len({m.id for m in unique.values()}) != len(unique):
            raise ValueError("Two different mutants of one prompt share a mutant ID")
        return list(unique.values())

    def _builtin_mutants(self, prompt: str, max_mutants: int) -> Sequence[Mutant]:
        """Generates a bounded sequence of mutants (deletions, negation inversions, modal weakenings, threshold changes, dropped exceptions, flipped comparators, narrowed scopes, deleted sections and examples)."""

        unique: dict[str, Rule] = {}
        for rule in self._rule_spans(prompt):
            unique.setdefault(rule.text, rule)
        candidates = list(unique.values())
        blocks = self._eligible_blocks(prompt, candidates)
        lanes: list[Rule | Block] = sorted([*candidates, *blocks], key=lambda lane: lane.start)
        if not lanes:
            return []

        if len(lanes) > max_mutants:
            step = len(lanes) / max_mutants
            lanes = [lanes[int(i * step)] for i in range(max_mutants)]

        def replace(table: tuple[tuple[re.Pattern[str], str], ...], operator_name: str, verb: str) -> Operator:
            return partial(self._replace_mutant, replacements=table, operator_name=operator_name, verb=verb)

        specific = (
            self._threshold_mutant,
            replace(((self.EXCEPTION_PATTERN, ""),), "drop_exception", "Dropped"),
            replace(self.COMPARATOR_REPLACEMENTS, "flip_comparator", "Flipped"),
            replace(self.SCOPE_REPLACEMENTS, "scope_swap", "Narrowed"),
        )
        rotating = (
            self._delete_mutant,
            replace(self.NEGATION_REPLACEMENTS, "invert_negation", "Inverted"),
            replace(self.WEAKENING_REPLACEMENTS, "weaken_modal", "Weakened"),
        )
        per_lane = []
        for i, lane in enumerate(lanes):
            if isinstance(lane, Block):
                per_lane.append([m] if (m := self._block_mutant(prompt, lane)) else [])
                continue
            offset = i % len(rotating)
            operators = (*specific, *rotating[offset:], *rotating[:offset])
            per_lane.append([m for op in operators if (m := op(prompt, lane))])
        mutants = [m for group in zip_longest(*per_lane) for m in group if m]
        return mutants[:max_mutants]
