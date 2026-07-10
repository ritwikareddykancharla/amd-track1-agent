"""Fireworks API tier — the only tokens that count against our score.

Everything here is about spending as few counted tokens as possible when we
do have to call out:
- pick the cheapest ALLOWED_MODELS entry that fits the task (Gemma preferred,
  code models only for code),
- terse per-category instructions and hard max_tokens caps,
- reasoning disabled where the API supports it (hidden reasoning tokens are
  billed!), and <think> blocks stripped if a model emits them anyway.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Cheap-first ranking by name fragment. Unknown models rank in the middle so a
# surprise ALLOWED_MODELS list still routes sanely.
_PRICE_RANK = [
    ("gemma", 0),
    ("llama-v3p2-3b", 1),
    ("llama-v3p2", 2),
    ("llama-v3p1-8b", 3),
    ("qwen", 4),
    ("mixtral", 5),
    ("minimax", 6),
    ("llama", 6),
    ("glm", 7),
    ("deepseek", 8),
    ("kimi", 9),
]
_CODE_HINT = re.compile(r"code|coder", re.I)

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_INSTRUCTIONS = {
    "code_gen": (
        "Write only the code requested. No explanation, no markdown fences "
        "unless asked."
    ),
    "code_debug": (
        "Identify and fix the bug. Reply with the corrected code only, plus at "
        "most one short sentence naming the bug."
    ),
    "logic": (
        "Solve the problem. Think silently; reply with only the final answer "
        "in one short sentence."
    ),
    "math": "Reply with only the final numeric answer.",
    "factual": "Answer directly in at most one short sentence.",
    "sentiment": "Reply with exactly one word: positive, negative, or neutral.",
    "ner": "Reply with a comma-separated list of the named entities only.",
    "summarization": "Reply with a one-to-two sentence summary only.",
}

_MAX_TOKENS = {
    "code_gen": 512,
    "code_debug": 512,
    "logic": 96,
    "math": 16,
    "factual": 48,
    "sentiment": 4,
    "ner": 64,
    "summarization": 96,
}


def _rank(model: str) -> int:
    name = model.lower()
    for fragment, rank in _PRICE_RANK:
        if fragment in name:
            return rank
    return 5


class FireworksClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = (
            os.environ.get("FIREWORKS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        )
        allowed = os.environ.get("ALLOWED_MODELS", "")
        self.allowed = [m.strip() for m in re.split(r"[,\n;]+", allowed) if m.strip()]
        self.tokens_spent = 0
        self._no_reasoning_param = True  # optimistic; drop on 400

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.allowed)

    def pick_model(self, category: str) -> str:
        code_task = category in ("code_gen", "code_debug")
        candidates = sorted(self.allowed, key=_rank)
        if code_task:
            for model in self.allowed:
                if _CODE_HINT.search(model):
                    return model
            return candidates[-1] if candidates else ""  # strongest general model
        # Non-code: cheapest capable; skip code-specialised models.
        general = [m for m in candidates if not _CODE_HINT.search(m)]
        pool = general or candidates
        if category == "logic" and len(pool) > 1:
            return pool[1]  # logic trips the very smallest models; one step up
        return pool[0] if pool else ""

    def answer(self, category: str, prompt: str, timeout: float = 60.0) -> str | None:
        if not self.available:
            return None
        model = self.pick_model(category)
        if not model:
            return None
        instruction = _INSTRUCTIONS.get(category, _INSTRUCTIONS["factual"])
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": _MAX_TOKENS.get(category, 96),
            "temperature": 0.0,
        }
        if self._no_reasoning_param:
            # Hidden reasoning tokens are billed by default on several hosted
            # models; this is the documented switch to turn them off.
            payload["reasoning_effort"] = "none"

        for attempt in (1, 2):
            try:
                data = self._post("/chat/completions", payload, timeout)
                usage = data.get("usage", {})
                self.tokens_spent += int(usage.get("total_tokens", 0))
                text = data["choices"][0]["message"]["content"] or ""
                text = _THINK_BLOCK.sub("", text).strip()
                return text or None
            except urllib.error.HTTPError as err:
                if err.code == 400 and "reasoning_effort" in payload:
                    # Model rejects the switch — retry once without it.
                    self._no_reasoning_param = False
                    payload.pop("reasoning_effort", None)
                    continue
                if err.code in (429, 500, 502, 503) and attempt == 1:
                    time.sleep(1.5)
                    continue
                return None
            except Exception:
                if attempt == 1:
                    time.sleep(1.0)
                    continue
                return None
        return None

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
