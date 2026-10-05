from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal

from clastogen.models import Mutant


class PromptMutator:
    """Extracts load-bearing imperative constraints and generates targeted prompt mutants."""

    RULE_PATTERN = re.compile(
        r"\b(?:never|do not|don't|must not|mustn't|should not|shouldn't|cannot|can't|always|must|shall not|may not|avoid|only|requires?|required)\b",
        re.IGNORECASE,
    )

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

    NUMBER_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")

    def __init__(self, target_symbol: str = "SYSTEM_PROMPT") -> None:
        self.target_symbol = target_symbol

    def extract_candidate_rules(self, prompt: str) -> list[str]:
        """Extracts candidate load-bearing sentences using linear string splitting and keyword search."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        return [
            cleaned
            for line in self.SENTENCE_SPLIT_PATTERN.split(prompt)
            if (cleaned := line.strip().lstrip("*- ").strip())
            and len(cleaned) > 10
            and self.RULE_PATTERN.search(cleaned)
        ]

    def _delete_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Removes rule with its bullet and trailing punctuation; None when the prompt is unchanged."""
        clean_pat = re.compile(rf"[ \t]*(?:[-*]\s*)?{re.escape(rule)}[.?!]?[ \t]*\n?")
        mutated = clean_pat.sub("", prompt).strip()
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
            description=f"Deleted load-bearing constraint: '{rule[:50]}...'",
        )

    def _invert_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Flips the first matching negation or obligation keyword in rule; None when none applies."""
        for pattern, replacement in self.NEGATION_REPLACEMENTS:
            if pattern.search(rule):
                inverted = pattern.sub(replacement.upper(), rule, count=1)
                mutated = prompt.replace(rule, inverted)
                if mutated != prompt:
                    return Mutant.create(
                        target_symbol=self.target_symbol,
                        operator_name="invert_negation",
                        original_snippet=rule,
                        mutated_snippet=inverted,
                        mutated_prompt=mutated,
                        description=f"Inverted constraint: '{rule[:40]}' -> '{inverted[:40]}'",
                    )
        return None

    def _threshold_mutant(self, prompt: str, rule: str) -> Mutant | None:
        """Multiplies the first number in rule by 10 (a $50 limit becomes $500); None when rule has no number."""
        match = self.NUMBER_PATTERN.search(rule)
        if match is None:
            return None
        raw = match.group()
        scaled = Decimal(raw.replace(",", "")) * 10
        number = f"{scaled:,}" if "," in raw else str(scaled)
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
            description=f"Changed threshold: '{raw}' -> '{number}' in '{rule[:40]}'",
        )

    def generate_mutants(self, prompt: str, max_mutants: int) -> Sequence[Mutant]:
        """Generates a bounded sequence of mutants (constraint deletions, negation inversions, threshold changes)."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        if max_mutants < 1:
            raise ValueError(f"max_mutants must be >= 1, got {max_mutants}")

        candidates = list(dict.fromkeys(self.extract_candidate_rules(prompt)))
        if not candidates:
            return []

        needed_rules = max(1, (max_mutants + 1) // 2)
        if len(candidates) > needed_rules:
            step = len(candidates) / needed_rules
            selected_rules = [candidates[int(i * step)] for i in range(needed_rules)]
        else:
            selected_rules = candidates

        operators = (self._delete_mutant, self._invert_mutant, self._threshold_mutant)
        mutants = [m for rule in selected_rules for op in operators if (m := op(prompt, rule))]
        return mutants[:max_mutants]
