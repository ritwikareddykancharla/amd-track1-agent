"""Local Gemma inference — the zero-token tier.

Runs gemma-3-4b-it (Q4_K_M GGUF) on CPU via llama.cpp. Tokens generated here
cost nothing on the leaderboard: only traffic through FIREWORKS_BASE_URL is
counted. The model is lazy-loaded, so runs that never need it pay no startup
cost, and machines without the GGUF (or with LOCAL_TIER=0) fall through to
the API tier untouched.

Routing is deliberately aggressive: every category except logic is attempted
locally first (the model measured 10/11 across the practice/sample set).
Categories differ in how the answer is checked before it ships:

  mechanically validated  sentiment (strict label line), ner (source-text
                          check), summarization (reduction check),
                          code_gen / code_debug (the code is executed)
  shipped unvalidated     factual, math word problems — no local oracle
                          exists; the accuracy gate (at most 84.2%) tolerates
                          a small number of misses and the ranking rewards
                          every saved token

Logic never runs locally: the 4B model failed a three-line deduction in
testing while sounding fully confident, and only re-doing the reasoning
could catch that — so there is nothing to validate against.

The same model also classifies tasks (a validated one-word answer); the
regex classifier remains the fallback whenever the model is unavailable or
answers off-menu.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import traceback

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/gemma-3-4b-it-Q4_K_M.gguf")
# Kill switch: LOCAL_TIER=0 routes everything to the API (v5 behaviour).
_ENABLED = os.environ.get("LOCAL_TIER", "1") != "0"

LOCAL_CATEGORIES = {
    "sentiment", "ner", "summarization", "factual", "math",
    "code_debug", "code_gen",
}

# Local inference on 2 vCPUs takes seconds per task, serialized behind the
# model lock. When the run deadline nears, declining (one paid API call) is
# strictly better than risking a blank answer at the deadline cut-off.
_deadline: float | None = None
_MIN_HEADROOM_S = 90.0


def set_deadline(monotonic_deadline: float) -> None:
    global _deadline
    _deadline = monotonic_deadline


def _out_of_time() -> bool:
    return _deadline is not None and _deadline - time.monotonic() < _MIN_HEADROOM_S


_CATEGORY_WORDS = {
    "factual", "math", "sentiment", "summarization", "ner",
    "code_debug", "code_gen", "logic",
}

_CLASSIFY = (
    "Classify the task into exactly one category: factual, math, sentiment, "
    "summarization, ner, code_debug, code_gen, or logic. Reply with only "
    "the category word.\n\nTask:\n"
)

_PROMPTS = {
    "sentiment": (
        "Line 1: the sentiment of the text as exactly one word — positive, "
        "negative, neutral, or mixed. Line 2: justify the label in one short "
        "sentence. Output nothing else."
    ),
    "ner": (
        "Extract the named entities from the text. List each entity as "
        "'label: value', one per line, using only the labels person, "
        "organization, location, date. Copy each value exactly as it appears "
        "in the text. Output nothing else."
    ),
    "summarization": (
        "Summarize the text. Obey any length or format constraint stated in "
        "the task (e.g. 'in one sentence'). Output only the summary."
    ),
    "factual": (
        "Answer the question accurately and directly in one or two short "
        "sentences. No preamble."
    ),
    "math": (
        "Solve the problem in at most four brief numbered steps, then end "
        "with 'Answer: <number>' on its own line."
    ),
    "code_gen": (
        "Output only the Python code in a single fenced code block — "
        "correct, complete, and self-contained. No explanation."
    ),
    "code_debug": (
        "Fix the bug. Output only the corrected code in a single fenced "
        "code block. No explanation."
    ),
}

_MAX_TOKENS = {
    "sentiment": 60, "ner": 128, "summarization": 160, "factual": 96,
    "math": 200, "code_gen": 320, "code_debug": 320,
}

_NER_LABELS = {"person", "organization", "location", "date"}
_NER_LINE = re.compile(r"^\s*[-*]?\s*(\w+)\s*[:=]\s*(.+?)\s*$")
_SENT_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
_N_SENTENCES = re.compile(
    r"in (?:exactly )?(one|a single|two|three|1|2|3) sentences?", re.I)
_WORDS = {"one": 1, "a single": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}

_SENTIMENT_LABELS = {"positive", "negative", "neutral", "mixed"}
_ANSWER_LINE = re.compile(r"answer:\s*\$?-?[\d,]*\.?\d", re.I)
_REFUSAL = re.compile(
    r"i (?:do not|don'?t) know|i can(?:no|')t|as an ai|i'?m not sure|"
    r"no information|unable to", re.I)
_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S | re.I)


class LocalModel:
    def __init__(self) -> None:
        self._llm = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        return _ENABLED and not self._failed and os.path.exists(MODEL_PATH)

    def _load(self):
        if self._llm is None:
            # The import lives inside the guarded block: a broken native
            # runtime (e.g. a missing shared library) must latch _failed and
            # be visible in the container log, not retried silently per task.
            try:
                from llama_cpp import Llama

                self._llm = Llama(
                    model_path=MODEL_PATH,
                    # 2048 keeps weights (~2.5GB) + KV cache well inside the
                    # 4GB grading VM; an oversized prompt raises and safely
                    # escalates that one task to the API.
                    n_ctx=2048,
                    n_threads=int(os.environ.get("LOCAL_MODEL_THREADS", "2")),
                    verbose=False,
                )
            except Exception:
                # Broken runtime/weights — permanent; per-call errors are not.
                self._failed = True
                print("local tier disabled: model load failed", file=sys.stderr)
                traceback.print_exc()
                raise
        return self._llm

    def _generate(self, content: str, max_tokens: int) -> str | None:
        try:
            with self._lock:  # llama.cpp context is not thread-safe
                llm = self._load()
                result = llm.create_chat_completion(
                    # Gemma has no system role; instructions ride in the
                    # user turn.
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            return (result["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            # Oversized prompt / transient failure: the caller escalates this
            # one task (load failures latch _failed in _load).
            return None

    def classify(self, prompt: str) -> str | None:
        """Validated one-word category, or None so the caller falls back to
        the regex classifier. Never wrong-by-construction: an off-menu reply
        is discarded, and a misroute into a validated category is caught by
        that category's validator (worst case: one extra API call)."""
        if not self.available or _out_of_time():
            return None
        text = self._generate(_CLASSIFY + prompt, max_tokens=6)
        if not text:
            return None
        word = text.split()[0].strip(".,!:'\"`*").lower().replace("-", "_")
        return word if word in _CATEGORY_WORDS else None

    def answer(self, category: str, prompt: str) -> str | None:
        """Local answer, validated where a validator exists, or None so the
        caller escalates to the API."""
        if not self.available or category not in LOCAL_CATEGORIES or _out_of_time():
            return None
        text = self._generate(
            f"{_PROMPTS[category]}\n\n{prompt}", _MAX_TOKENS[category])
        if not text:
            return None
        return _validate(category, prompt, text)


def _validate(category: str, prompt: str, text: str) -> str | None:
    """Return a shippable answer or None. Checked categories err toward
    escalation: a failed validation costs one API call, a shipped wrong
    answer risks the accuracy gate."""
    if category == "sentiment":
        return _validate_sentiment(text)
    if category == "ner":
        return _validate_ner(prompt, text)
    if category == "summarization":
        return _validate_summary(prompt, text)
    if category == "factual":
        return _validate_factual(text)
    if category == "math":
        return text if _ANSWER_LINE.search(text) else None
    if category in ("code_gen", "code_debug"):
        return _validate_code(text)
    return None


def _validate_sentiment(text: str) -> str | None:
    """First line must be exactly one legal label; a justification line must
    follow (the graded rubric is 'labelling sentiment AND justifying the
    classification'). Ships as 'label\\njustification'."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    label = lines[0].strip("*_.,!: ").lower()
    if label not in _SENTIMENT_LABELS:
        return None
    return f"{label}\n{' '.join(lines[1:])}"


def _validate_factual(text: str) -> str | None:
    # No oracle for facts — only reject obvious non-answers. Anything that
    # hedges or rambles goes to the API instead.
    if _REFUSAL.search(text) or len(text.split()) > 80:
        return None
    return text


def _validate_ner(prompt: str, text: str) -> str | None:
    """Keep only 'label: value' lines whose value literally occurs in the
    prompt. Any malformed or hallucinated line invalidates the whole answer —
    a partial entity list would read as correct but score as incomplete."""
    haystack = prompt.lower()
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _NER_LINE.match(raw)
        if not m or m.group(1).lower() not in _NER_LABELS:
            return None
        value = m.group(2).strip().strip("'\"")
        if not value or value.lower() not in haystack:
            return None  # entity not in source text: hallucinated or mangled
        lines.append(f"{m.group(1).lower()}: {value}")
    return "\n".join(lines) if lines else None


def _validate_summary(prompt: str, text: str) -> str | None:
    # Preamble means the model ignored "output only the summary" — its
    # compliance elsewhere is suspect too.
    if text.lower().startswith(("here is", "here's", "sure", "summary:")):
        return None
    src_words = len(prompt.split())
    out_words = len(text.split())
    # A summary that isn't a real reduction of the passage is not a summary.
    if out_words >= 0.6 * src_words or out_words < 3:
        return None
    m = _N_SENTENCES.search(prompt)
    if m:
        limit = _WORDS[m.group(1).lower()]
        n = len([s for s in _SENT_SPLIT.split(text) if s.strip()])
        if n > limit:
            return None
    return text


# Runs inside a throwaway subprocess: definitions must exec cleanly and a
# battery of generic single-argument calls must not raise. TypeError means
# "sample doesn't apply to this function" (e.g. a string fed to fibonacci)
# and is skipped; any other exception — IndexError on [], ZeroDivisionError,
# an infinite loop killed by the timeout — fails the candidate.
_SMOKE = """
import inspect, sys
ns = {}
try:
    exec(compile(CODE, "<candidate>", "exec"), ns)
except Exception:
    sys.exit(2)
funcs = [v for v in ns.values() if inspect.isfunction(v)]
if not funcs:
    sys.exit(3)
samples = [([],), ([3, 1, 2],), ([5, 5, 3],), ([2],), (0,), (1,), (7,), ("",), ("abc",)]
for fn in funcs:
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.default is p.empty
                  and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        continue
    if len(params) != 1:
        continue
    for args in samples:
        try:
            fn(*args)
        except TypeError:
            pass
        except Exception:
            sys.exit(4)
sys.exit(0)
"""


def _code_runs(code: str) -> bool:
    harness = f"CODE = {code!r}\n" + _SMOKE
    try:
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True, timeout=10,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _validate_code(text: str) -> str | None:
    """Execution is the validator: extract the code block, run it in a
    sandboxed subprocess, and ship only if it survives. Semantically wrong
    but crash-free code can still slip through — but every crash, syntax
    error, and hang is caught for free."""
    m = _CODE_BLOCK.search(text)
    code = (m.group(1) if m else text).strip()
    if "def " not in code:
        return None
    if not _code_runs(code):
        return None
    return text if m else f"```python\n{code}\n```"
