"""Per-category answer strategy: classify, then one tuned Fireworks call.

Each category carries a terse system prompt, a token cap, and a model tier.
Prompts are deliberately short — input tokens count toward the score — and
push the model to answer directly without preamble.
"""

from __future__ import annotations

from classifier import Category, classify
from llm import complete, model_for

CHEAP, STRONG, CODE = "cheap", "strong", "code"

# Requests are plain OpenAI-compatible calls (no reasoning_effort), so
# reasoning models may spend completion tokens thinking before the answer.
# Caps leave room for that: a cap that truncates mid-thought yields an empty
# or cut-off answer, which costs far more score than the extra tokens.
# The "brief steps" instruction on math/logic is load-bearing — removing it
# measurably collapses accuracy there. Do not trim it.
_BASE = "Answer in English. Be concise and direct; no preamble, no restating the question."

_CONFIG: dict[Category, tuple[str, int, str]] = {
    Category.FACTUAL: (
        f"{_BASE} Give a correct, clear answer in under 120 words.",
        350, STRONG,
    ),
    Category.MATH: (
        f"{_BASE} Work through it in brief steps, then end with "
        f"'Answer: <value>' on its own line.",
        450, STRONG,
    ),
    Category.SENTIMENT: (
        f"{_BASE} State the sentiment as positive, negative, or neutral, "
        f"then one short reason.",
        100, CHEAP,
    ),
    Category.SUMMARIZATION: (
        f"{_BASE} Output only the summary and obey any length or format "
        f"constraint stated in the task.",
        250, CHEAP,
    ),
    Category.NER: (
        f"{_BASE} List each entity as 'label: value', one per line, using "
        f"the labels person, organization, location, date.",
        250, CHEAP,
    ),
    Category.CODE_DEBUG: (
        f"{_BASE} State the bug in one sentence, then give the corrected "
        f"code in a single fenced block.",
        900, CODE,
    ),
    Category.CODE_GEN: (
        f"{_BASE} Output only the code in a single fenced block — correct, "
        f"complete, and self-contained.",
        900, CODE,
    ),
    Category.LOGIC: (
        f"{_BASE} Reason in at most five brief numbered steps, one short "
        f"line each, then end with 'Answer: <value>' on its own line.",
        700, STRONG,
    ),
}


def solve(prompt: str) -> str:
    category = classify(prompt)
    system, max_tokens, tier = _CONFIG[category]
    primary = model_for(tier)
    # Blank/failed answers retry on the opposite general tier.
    fallback = model_for(STRONG if tier == CHEAP else CHEAP)
    return complete(prompt, system=system, max_tokens=max_tokens,
                    model=primary, fallback_model=fallback)
