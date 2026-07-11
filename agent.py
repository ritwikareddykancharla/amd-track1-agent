"""Per-category answer strategy: classify, answer free if possible, else API.

The regex classifier picks the category (zero cost, zero model time), then
every category except logic is attempted locally; only declined or
validation-failed tasks pay for a Fireworks call. Each API category carries
a terse system prompt, a token cap, and a model tier — prompts are
deliberately short because input tokens count toward the score.
"""

from __future__ import annotations

from classifier import Category, classify
from llm import complete, model_for
from local_model import LOCAL_CATEGORIES, LocalModel
from solvers import solve_math

_LOCAL = LocalModel()

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
    # Regex classification only. Gemma classification was tried (v8) and
    # timed out the grading VM: one extra model call per task doubled the
    # serialized local compute, and the regex routed every sample and
    # practice task identically anyway. Local compute is spent exclusively
    # on producing answers.
    category = classify(prompt)

    # Free tier: provable arithmetic never touches the API. solve_math()
    # declines (returns None) on anything it cannot fully parse, so a wrong
    # zero-token answer is structurally impossible — word problems and
    # unparseable phrasings fall through to the local/API paths below.
    if category is Category.MATH:
        exact = solve_math(prompt)
        if exact is not None:
            return f"Answer: {exact}"

    # Free tier 2: local Gemma for every category but logic. Sentiment/NER/
    # summarization ship only if their validators pass, code ships only if
    # it executes cleanly, factual and math word problems ship unvalidated
    # (no local oracle exists; the token ranking rewards the gamble). Any
    # decline returns None and the task pays for an API call instead.
    if category.value in LOCAL_CATEGORIES:
        local = _LOCAL.answer(category.value, prompt)
        if local is not None:
            return local

    system, max_tokens, tier = _CONFIG[category]
    primary = model_for(tier)
    # Blank/failed answers retry on the opposite general tier.
    fallback = model_for(STRONG if tier == CHEAP else CHEAP)
    return complete(prompt, system=system, max_tokens=max_tokens,
                    model=primary, fallback_model=fallback)
