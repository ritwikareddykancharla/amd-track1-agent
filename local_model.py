"""Local Gemma inference — the zero-token tier for validatable categories.

Runs gemma-3-4b-it (Q4_K_M GGUF) on CPU via llama.cpp. Tokens generated here
cost nothing on the leaderboard: only traffic through FIREWORKS_BASE_URL is
counted. The model is lazy-loaded, so runs that never need it pay no startup
cost, and machines without the GGUF (or with LOCAL_TIER=0) fall through to
the API tier untouched.

The tier only handles categories whose output can be *checked* before it
ships — a wrong local answer risks the accuracy gate that every saved token
depends on:

  sentiment      answer must be exactly positive / negative / neutral
  ner            every extracted entity must literally occur in the source
  summarization  must be a real reduction and obey stated sentence limits

Factual stays on the API: a 4B model's factual recall cannot be validated
locally, and hallucinated facts are what sank the 21% graded run.
"""
from __future__ import annotations

import os
import re
import threading
import time

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/gemma-3-4b-it-Q4_K_M.gguf")
# Kill switch: LOCAL_TIER=0 routes everything to the API (v5 behaviour).
_ENABLED = os.environ.get("LOCAL_TIER", "1") != "0"

LOCAL_CATEGORIES = {"sentiment", "ner", "summarization"}

# Local inference on 2 vCPUs takes ~20-60s per task, serialized behind the
# model lock. When the run deadline nears, declining (one paid API call) is
# strictly better than risking a blank answer at the deadline cut-off.
_deadline: float | None = None
_MIN_HEADROOM_S = 90.0


def set_deadline(monotonic_deadline: float) -> None:
    global _deadline
    _deadline = monotonic_deadline


def _out_of_time() -> bool:
    return _deadline is not None and _deadline - time.monotonic() < _MIN_HEADROOM_S

_PROMPTS = {
    "sentiment": (
        "Classify the sentiment of the text. Reply with exactly one word: "
        "positive, negative, or neutral. No punctuation, no explanation."
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
}

_MAX_TOKENS = {"sentiment": 8, "ner": 128, "summarization": 160}

_NER_LABELS = {"person", "organization", "location", "date"}
_NER_LINE = re.compile(r"^\s*[-*]?\s*(\w+)\s*[:=]\s*(.+?)\s*$")
_SENT_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
_N_SENTENCES = re.compile(
    r"in (?:exactly )?(one|a single|two|three|1|2|3) sentences?", re.I)
_WORDS = {"one": 1, "a single": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}


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
            from llama_cpp import Llama

            try:
                self._llm = Llama(
                    model_path=MODEL_PATH,
                    # 4096 fits model weights (~2.5GB) + KV cache inside the
                    # 4GB grading VM and covers long summarization passages.
                    n_ctx=4096,
                    n_threads=int(os.environ.get("LOCAL_MODEL_THREADS", "2")),
                    verbose=False,
                )
            except Exception:
                # Broken runtime/weights — permanent; per-call errors are not.
                self._failed = True
                raise
        return self._llm

    def answer(self, category: str, prompt: str) -> str | None:
        """Validated local answer, or None so the caller escalates to the API."""
        if not self.available or category not in LOCAL_CATEGORIES or _out_of_time():
            return None
        try:
            with self._lock:  # llama.cpp context is not thread-safe
                llm = self._load()
                result = llm.create_chat_completion(
                    messages=[
                        # Gemma has no system role; fold instructions into the
                        # user turn.
                        {"role": "user",
                         "content": f"{_PROMPTS[category]}\n\n{prompt}"},
                    ],
                    max_tokens=_MAX_TOKENS[category],
                    temperature=0.0,
                )
            text = (result["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            # Oversized prompt / transient failure: escalate this task only
            # (load failures set _failed in _load and disable the tier).
            return None
        return _validate(category, prompt, text)


def _validate(category: str, prompt: str, text: str) -> str | None:
    """Return a shippable answer or None. Every check errs toward escalation:
    a failed validation costs one API call, a shipped wrong answer risks the
    accuracy gate."""
    if not text:
        return None
    if category == "sentiment":
        word = text.split()[0].strip(".,!:").lower()
        return word if word in ("positive", "negative", "neutral") else None
    if category == "ner":
        return _validate_ner(prompt, text)
    if category == "summarization":
        return _validate_summary(prompt, text)
    return None


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
