"""Local Gemma inference — the zero-Fireworks-token tier.

Runs gemma-2-2b-it (Q4_K_M GGUF) on CPU via llama.cpp. Tokens generated here
cost nothing on the leaderboard: only traffic through FIREWORKS_BASE_URL is
counted. The model is lazy-loaded so runs that never need it (all-deterministic
inputs) pay no startup cost, and dev machines without the GGUF simply fall
through to Fireworks.
"""
from __future__ import annotations

import os
import threading

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/gemma-2-2b-it-Q4_K_M.gguf")

# Categories the 2B model handles reliably; everything else escalates.
LOCAL_CATEGORIES = {"sentiment", "ner", "summarization", "factual", "math"}

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
}

_MAX_TOKENS = {
    "sentiment": 4,
    "ner": 64,
    "summarization": 96,
    "factual": 48,
    "math": 16,
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

            self._llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=int(os.environ.get("LOCAL_MODEL_THREADS", "2")),
                verbose=False,
            )
        return self._llm

    def answer(self, category: str, prompt: str) -> str | None:
        """Answer locally, or None so the caller escalates to Fireworks."""
        if not self.available or category not in LOCAL_CATEGORIES:
            return None
        try:
            with self._lock:  # llama.cpp context is not thread-safe
                llm = self._load()
                result = llm.create_chat_completion(
                    messages=[
                        # Gemma has no system role; fold instructions into user turn.
                        {
                            "role": "user",
                            "content": f"{_SYSTEM_PROMPTS[category]}\n\n{prompt}",
                        }
                    ],
                    max_tokens=_MAX_TOKENS[category],
                    temperature=0.0,
                )
            text = result["choices"][0]["message"]["content"].strip()
            return _sanitize(category, text)
        except Exception:
            self._failed = True  # broken runtime → stop trying, escalate everything
            return None


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
