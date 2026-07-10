"""Zero-token task classifier.

Every task burns 0 Fireworks tokens to classify: pure regex/heuristics over
the prompt text. Categories mirror the 8 the hackathon grades on. When in
doubt we return "factual" (safest default: short local answer, cheap escalation).
"""
from __future__ import annotations

import re

CATEGORIES = [
    "code_debug",
    "code_gen",
    "math",
    "sentiment",
    "ner",
    "summarization",
    "logic",
    "factual",
]

_CODE_HINTS = re.compile(
    r"```|\bdef \w+\(|\bfunction\s+\w+\(|\bclass \w+|\bimport \w+|#include|"
    r"\bconsole\.log\b|\bprint\(|\breturn\b.*;|\bpublic static\b",
)

_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "code_debug",
        re.compile(
            r"\b(fix|debug|bug|error|broken|not work|doesn'?t work|incorrect output|"
            r"what is wrong|find the (bug|error|issue)|correct the (code|function))\b",
            re.I,
        ),
    ),
    (
        "code_gen",
        re.compile(
            r"\b(write|implement|create|generate|build)\b.{0,40}\b(function|program|"
            r"script|code|method|class|algorithm|regex|sql query)\b",
            re.I,
        ),
    ),
    (
        "sentiment",
        re.compile(
            r"\b(sentiment|positive,? negative,? or neutral|positive or negative|"
            r"emotional tone|classify (the |this )?(review|tweet|feedback|text)|"
            r"is (this|the) (review|comment|tweet) positive)\b",
            re.I,
        ),
    ),
    (
        "ner",
        re.compile(
            r"\b(named entit|extract (all |the )?(entit|names?|people|persons?|"
            r"organi[sz]ations?|locations?|places|dates)|identify (all |the )?"
            r"(entit|people|persons|organi[sz]ations|locations))",
            re.I,
        ),
    ),
    (
        "summarization",
        re.compile(
            r"\b(summari[sz]e|summary|tl;?dr|condense|shorten (this|the)|"
            r"main (points?|idea) of (the|this)|in (one|two|a few) sentences?)\b",
            re.I,
        ),
    ),
    (
        "math",
        re.compile(
            r"\b(calculate|compute|solve for|what is \d|how (much|many)|sum of|"
            r"product of|difference between \d|divided by|multiplied|percent|"
            r"average of|remainder|square root|area of|perimeter|probability)"
            r"|\d\s*[-+*/^×÷%]\s*(of\s+)?\d",
            re.I,
        ),
    ),
    (
        "logic",
        re.compile(
            r"\b(if all|all \w+ are|some \w+ are|no \w+ are|syllogism|deduce|"
            r"logical(ly)?|riddle|puzzle|premise|conclusion follows|"
            r"who (is|am i)|truth[- ]?teller|liar|older than|younger than|"
            r"left of|right of|true or false|which statement)\b",
            re.I,
        ),
    ),
]


def classify(prompt: str) -> str:
    """Return one of CATEGORIES for a task prompt."""
    has_code = bool(_CODE_HINTS.search(prompt))

    for category, pattern in _PATTERNS:
        if pattern.search(prompt):
            # Code block + debug-ish words beats everything else.
            if category in ("code_debug", "code_gen"):
                return category
            # A prompt containing code is almost never sentiment/NER/etc.
            if has_code:
                continue
            return category

    if has_code:
        # Code present but no explicit ask matched: assume debugging intent.
        return "code_debug"
    return "factual"
