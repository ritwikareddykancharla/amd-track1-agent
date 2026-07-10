"""Local Gemma inference — the zero-Fireworks-token tier.

Runs gemma-3-4b-it (Q4_K_M GGUF) on CPU via llama.cpp. Tokens generated here
cost nothing on the leaderboard: only traffic through FIREWORKS_BASE_URL is
counted. The model is lazy-loaded so runs that never need it pay no startup
cost, and dev machines without the GGUF simply fall through to Fireworks.

Also serves as the fallback floor: when a Fireworks call fails, any category
can be answered here with strict=False — an imperfect local answer always
beats the guaranteed zero of an empty one.
"""
from __future__ import annotations

import os
import threading

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/gemma-3-4b-it-Q4_K_M.gguf")

# Categories the local model handles reliably when used as a primary tier;
# as a *fallback* (strict=False) every category is fair game.
LOCAL_CATEGORIES = {"sentiment", "ner", "summarization", "factual"}

_SYSTEM_PROMPTS = {
    "sentiment": (
        "Classify the sentiment of the text. Reply with exactly one word: "
        "positive, negative, or neutral. No punctuation, no explanation."
    ),
    "ner": (
        "Extract the named entities from the text. Reply with a comma-separated "
        "list of the entities only. No explanation."
    ),
    "summarization": (
        "Summarize the text in one or two short sentences. Reply with only the "
        "summary."
    ),
    "factual": (
        "Answer the question directly and briefly, in at most a short sentence. "
        "No explanation, no preamble."
    ),
    "math": (
        "Solve the problem. Reply with only the final numeric answer, nothing else."
    ),
    "code_debug": (
        "Identify and fix the bug. Reply with the corrected code only, plus at "
        "most one short sentence naming the bug."
    ),
    "code_gen": (
        "Write only the code requested. No explanation, no markdown fences "
        "unless asked."
    ),
    "logic": (
        "Solve the problem. Think silently; reply with only the final answer "
        "in one short sentence."
    ),
}

_MAX_TOKENS = {
    "sentiment": 4,
    "ner": 64,
    "summarization": 96,
    "factual": 48,
    "math": 16,
    "code_debug": 512,
    "code_gen": 512,
    "logic": 96,
}


class LocalModel:
    def __init__(self) -> None:
        self._llm = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        return not self._failed and os.path.exists(MODEL_PATH)

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama

            try:
                self._llm = Llama(
                    model_path=MODEL_PATH,
                    # 2048 keeps model + KV cache comfortably inside the 4GB
                    # grading VM with the 4B weights (~2.5GB) resident.
                    n_ctx=2048,
                    n_threads=int(os.environ.get("LOCAL_MODEL_THREADS", "2")),
                    verbose=False,
                )
            except Exception:
                # Broken runtime/weights — permanent; per-call errors are not.
                self._failed = True
                raise
        return self._llm

    def answer(self, category: str, prompt: str, strict: bool = True) -> str | None:
        """Answer locally, or None so the caller escalates to Fireworks.

        strict=True validates format (primary-tier use: a rambled answer is
        untrustworthy, escalate). strict=False returns the raw text (fallback
        use: any answer beats an empty one).
        """
        if not self.available:
            return None
        if strict and category not in LOCAL_CATEGORIES:
            return None
        instruction = _SYSTEM_PROMPTS.get(category, _SYSTEM_PROMPTS["factual"])
        try:
            with self._lock:  # llama.cpp context is not thread-safe
                llm = self._load()
                result = llm.create_chat_completion(
                    messages=[
                        # Gemma has no system role; fold instructions into user turn.
                        {
                            "role": "user",
                            "content": f"{instruction}\n\n{prompt}",
                        }
                    ],
                    max_tokens=_MAX_TOKENS.get(category, 96),
                    temperature=0.0,
                )
            text = result["choices"][0]["message"]["content"].strip()
        except Exception:
            # A single oversized/failed prompt shouldn't kill the tier for the
            # remaining tasks (load failures set _failed in _load).
            return None
        sanitized = _sanitize(category, text)
        if sanitized is not None:
            return sanitized
        return None if strict else (text or None)


def _sanitize(category: str, text: str) -> str | None:
    if not text:
        return None
    if category == "sentiment":
        word = text.split()[0].strip(".,!").lower()
        if word in ("positive", "negative", "neutral"):
            return word
        return None  # model rambled — not trustworthy, escalate
    if category == "math":
        token = text.split()[0].strip(".,")
        try:
            float(token.replace(",", ""))
            return token.replace(",", "")
        except ValueError:
            return None
    return text
