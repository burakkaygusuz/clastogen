from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, localcontext
from itertools import zip_longest

from clastogen.models import Mutant
from clastogen.mutation.structure import Block, find_blocks

_LIST_MARKER = r"(?:[-*]|\d+[.)])\s+"


def _clip(text: str, limit: int) -> str:
    """Shortens text to at most limit characters at a word boundary and marks the cut with an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit] if text[limit].isspace() else text[:limit].rsplit(maxsplit=1)[0]
    return head.rstrip(" ,.;:") + "…"


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

    NEGATION_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bdo not\b", re.IGNORECASE), "do"),
        (re.compile(r"\bdon't\b", re.IGNORECASE), "do"),
        (re.compile(r"\bmust not\b", re.IGNORECASE), "must"),
        (re.compile(r"\bmustn't\b", re.IGNORECASE), "must"),
        (re.compile(r"\bnever\b", re.IGNORECASE), "always"),
        (re.compile(r"\bcannot\b", re.IGNORECASE), "can"),
        (re.compile(r"\bcan't\b", re.IGNORECASE), "can"),
        (re.compile(r"\bshould not\b", re.IGNORECASE), "should"),
        (re.compile(r"\bshouldn't\b", re.IGNORECASE), "should"),
        (re.compile(r"\bshall not\b", re.IGNORECASE), "shall"),
        (re.compile(r"\bmay not\b", re.IGNORECASE), "may"),
        (re.compile(r"\balways\b", re.IGNORECASE), "never"),
        (re.compile(r"\bmust\b", re.IGNORECASE), "must not"),
        (re.compile(r"\bavoid\b", re.IGNORECASE), "prefer"),
    )

    WEAKENING_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bmust not\b", re.IGNORECASE), "should not"),
        (re.compile(r"\bmustn't\b", re.IGNORECASE), "shouldn't"),
        (re.compile(r"\bnever\b", re.IGNORECASE), "rarely"),
        (re.compile(r"\balways\b", re.IGNORECASE), "usually"),
        (re.compile(r"\bmust\b", re.IGNORECASE), "should"),
        (re.compile(r"\bshall not\b", re.IGNORECASE), "should not"),
        (re.compile(r"\bcannot\b", re.IGNORECASE), "should not"),
        (re.compile(r"\bcan't\b", re.IGNORECASE), "shouldn't"),
        (re.compile(r"\brequired\b", re.IGNORECASE), "recommended"),
        (re.compile(r"\brequires\b", re.IGNORECASE), "recommends"),
        (re.compile(r"\bonly\b", re.IGNORECASE), "preferably"),
    )

    COMPARATOR_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bno more than\b", re.IGNORECASE), "at least"),
        (re.compile(r"\bno fewer than\b", re.IGNORECASE), "at most"),
        (re.compile(r"\bat most\b", re.IGNORECASE), "at least"),
        (re.compile(r"\bat least\b", re.IGNORECASE), "at most"),
        (re.compile(r"\bfewer than\b", re.IGNORECASE), "more than"),
        (re.compile(r"\bmore than\b", re.IGNORECASE), "fewer than"),
        (re.compile(r"\bmaximum\b", re.IGNORECASE), "minimum"),
        (re.compile(r"\bminimum\b", re.IGNORECASE), "maximum"),
    )

    SCOPE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\b(?:all|every|each)\b", re.IGNORECASE), "some"),
        (re.compile(r"\bboth\b", re.IGNORECASE), "either"),
        (re.compile(r"\bany\b", re.IGNORECASE), "every"),
    )

    EXCEPTION_PATTERN = re.compile(r"(?<=\S),?\s+(?:unless|except)\b[^.;\n,]*", re.IGNORECASE)

    NUMBER_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")

    def __init__(self, target_symbol: str = "SYSTEM_PROMPT") -> None:
        self.target_symbol = target_symbol

    def extract_candidate_rules(self, prompt: str) -> list[str]:
        """Extracts candidate load-bearing sentences: those with an obligation, condition or action word or a number, minus role statements."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        return [
            cleaned
            for line in self.SENTENCE_SPLIT_PATTERN.split(prompt)
            if (cleaned := re.sub(rf"^{_LIST_MARKER}", "", line.strip()))
            and len(cleaned) > 10
            and not self.ROLE_PATTERN.match(cleaned)
            and self.RULE_PATTERN.search(cleaned)
        ]

    def _delete_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Removes rule with its bullet and trailing punctuation; None when the prompt is unchanged."""
        clean_pat = re.compile(rf"(?<![ \t])[ \t]*(?:{_LIST_MARKER})?{re.escape(rule)}[.?!]?([ \t]*)(\n?)")

        def join(m: re.Match[str]) -> str:
            if m.start() == 0 or prompt[m.start() - 1] == "\n":
                return ""
            return m[2] or ("" if m.end() == len(prompt) else " ")

        mutated = clean_pat.sub(join, prompt).strip()
        mutated = re.sub(r"\n\s*\.\s*\n", "\n", mutated)
        mutated = re.sub(r"\n{3,}", "\n\n", mutated)
        if mutated == prompt:
            return None
        return Mutant.create(
            target_symbol=self.target_symbol,
            operator_name="delete_constraint",
            original_snippet=rule,
            mutated_snippet="[DELETED]",
            mutated_prompt=mutated,
            description=f"Deleted constraint: '{_clip(rule, 60)}'",
        )

    def _replace_mutant(
        self,
        prompt: str,
        rule: str,
        replacements: tuple[tuple[re.Pattern[str], str], ...],
        operator_name: str,
        verb: str,
    ) -> Mutant | None:
        """Applies the first matching replacement in rule; None when none applies."""
        for pattern, replacement in replacements:
            if pattern.search(rule):
                changed = pattern.sub(replacement.upper(), rule, count=1)
                mutated = prompt.replace(rule, changed)
                if mutated != prompt:
                    return Mutant.create(
                        target_symbol=self.target_symbol,
                        operator_name=operator_name,
                        original_snippet=rule,
                        mutated_snippet=changed,
                        mutated_prompt=mutated,
                        description=f"{verb} constraint: '{_clip(rule, 45)}' -> '{_clip(changed, 45)}'",
                    )
        return None

    def _invert_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Flips the first matching negation or obligation keyword in rule; None when none applies."""
        return self._replace_mutant(prompt, rule, self.NEGATION_REPLACEMENTS, "invert_negation", "Inverted")

    def _weaken_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Softens the first matching hard keyword in rule into a suggestion; None when none applies."""
        return self._replace_mutant(prompt, rule, self.WEAKENING_REPLACEMENTS, "weaken_modal", "Weakened")

    def _flip_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Reverses the first comparator or ordering word in rule (at most becomes AT LEAST); None when none applies."""
        return self._replace_mutant(prompt, rule, self.COMPARATOR_REPLACEMENTS, "flip_comparator", "Flipped")

    def _scope_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Narrows the first universal quantifier in rule (all becomes SOME); None when none applies."""
        return self._replace_mutant(prompt, rule, self.SCOPE_REPLACEMENTS, "scope_swap", "Narrowed")

    def _exception_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Drops the first unless or except clause of rule, up to the next comma or semicolon; None when there is none."""
        return self._replace_mutant(prompt, rule, ((self.EXCEPTION_PATTERN, ""),), "drop_exception", "Dropped")

    def _threshold_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Multiplies the first number in rule by 10 (a $50 limit becomes $500); None when rule has no number."""
        match = self.NUMBER_PATTERN.search(rule)
        if match is None:
            return None
        raw = match.group()
        with localcontext(prec=len(raw) + 2):  # the default 28 digits would round a long number
            scaled = Decimal(raw.replace(",", "")) * 10
        number = format(scaled, ",f" if "," in raw else "f")
        changed = rule[: match.start()] + number + rule[match.end() :]
        mutated = prompt.replace(rule, changed)
        if mutated == prompt:
            return None
        return Mutant.create(
            target_symbol=self.target_symbol,
            operator_name="change_threshold",
            original_snippet=rule,
            mutated_snippet=changed,
            mutated_prompt=mutated,
            description=f"Changed threshold: '{raw}' -> '{number}' in '{_clip(rule, 50)}'",
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

    def _eligible_blocks(self, prompt: str, rules: Sequence[str]) -> list[Block]:
        """Keeps examples and sections that hold a rule, and drops blocks over half of the prompt."""
        seen: set[str] = set()
        eligible: list[Block] = []
        for b in find_blocks(prompt):
            text = prompt[b.start : b.end]
            # Identical blocks share a mutant ID (it hashes the text, not the position), so only the first is kept.
            if (
                len(text) * 2 <= len(prompt)
                and (b.kind == "example" or any(rule in text for rule in rules))
                and text not in seen
            ):
                seen.add(text)
                eligible.append(b)
        return eligible

    def generate_mutants(self, prompt: str, max_mutants: int) -> Sequence[Mutant]:
        """Generates a bounded sequence of mutants (deletions, negation inversions, modal weakenings, threshold changes, dropped exceptions, flipped comparators, narrowed scopes, deleted sections and examples)."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        if max_mutants < 1:
            raise ValueError(f"max_mutants must be >= 1, got {max_mutants}")

        candidates = list(dict.fromkeys(self.extract_candidate_rules(prompt)))
        blocks = self._eligible_blocks(prompt, candidates)
        # Rules and blocks share one lane list in prompt order so every kind stays reachable at a small budget.
        lanes: list[str | Block] = sorted(
            [*candidates, *blocks], key=lambda lane: lane.start if isinstance(lane, Block) else prompt.find(lane)
        )
        if not lanes:
            return []

        # Breadth first: every selected lane gets a mutant before any lane gets a second one.
        if len(lanes) > max_mutants:
            step = len(lanes) / max_mutants
            lanes = [lanes[int(i * step)] for i in range(max_mutants)]

        # Specific operators first: each returns None when its pattern is absent, so the generic rotation fills in.
        specific = (self._threshold_mutant, self._exception_mutant, self._flip_mutant, self._scope_mutant)
        rotating = (self._delete_mutant, self._invert_mutant, self._weaken_mutant)
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
