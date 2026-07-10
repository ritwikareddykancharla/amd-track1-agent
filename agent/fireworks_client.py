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
    # gpt-oss can't fully disable reasoning on Fireworks -> spends more tokens
    # per answer than "bigger" models with thinking disabled. Rank it last.
    ("gpt-oss", 10),
    ("kimi", 9),
]
_CODE_HINT = re.compile(r"code|coder", re.I)

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_INSTRUCTIONS = {
    "code_gen": (
        "Write only the code requested — complete and correct, but nothing "
        "beyond it. No explanation, no usage examples, no markdown fences "
        "unless asked."
    ),
    "code_debug": (
        "Identify and fix the bug. Reply with the corrected code only, plus at "
        "most one short sentence naming the bug. No other commentary."
    ),
    "logic": (
        "Solve the problem. Think silently; reply with only the final answer "
        "in one short sentence. Be precise — no restating the problem, no "
        "step-by-step working."
    ),
    "math": (
        "Reply with only the final numeric answer — no working, no units, "
        "no explanation."
    ),
    "factual": (
        "Answer with only the answer itself, in at most one short sentence. "
        "No preamble, no context, no explanation."
    ),
    "sentiment": (
        "Reply with exactly one word: positive, negative, or neutral. "
        "Nothing else."
    ),
    "ner": (
        "Reply with a comma-separated list of the named entities only. "
        "No labels, no explanation."
    ),
    "summarization": (
        "Reply with a one-to-two sentence summary only. No preamble like "
        "'Here is a summary'."
    ),
}

# Safety rails, not budgets: the prompt is what keeps answers terse, and a
# well-behaved completion ends at EOS far below these. Set high enough that a
# correct answer can never be truncated mid-sentence (a clipped answer reads
# as wrong to the judge); they only bite when a model ignores the instruction
# and rambles, where they bound the damage in billed tokens.
_MAX_TOKENS = {
    "code_gen": 1024,
    "code_debug": 1024,
    "logic": 256,
    "math": 64,
    "factual": 128,
    "sentiment": 8,
    "ner": 128,
    "summarization": 192,
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
        # Per-model reasoning-suppression mode, learned from 400s at runtime:
        # "thinking" -> {"thinking": {"type": "disabled"}}  (GLM/Kimi/DeepSeek)
        # "effort"   -> {"reasoning_effort": "low"}          (gpt-oss family)
        # "plain"    -> no suppression parameter accepted
        self._mode_by_model: dict[str, str] = {}

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
            # Strongest general model, but avoid gpt-oss when possible: it
            # can't fully disable reasoning, so it costs ~3x the tokens of a
            # thinking-disabled peer on the same code task.
            pool = [m for m in candidates if "gpt-oss" not in m.lower()]
            pool = pool or candidates
            return pool[-1] if pool else ""
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

        mode = self._mode_by_model.get(model, "thinking")
        for attempt in range(4):
            payload.pop("thinking", None)
            payload.pop("reasoning_effort", None)
            if mode == "thinking":
                # Hidden reasoning is billed by default on most hosted models;
                # this is the switch that turns it off (GLM/Kimi/DeepSeek).
                payload["thinking"] = {"type": "disabled"}
            elif mode == "effort":
                payload["reasoning_effort"] = "low"  # gpt-oss family
            try:
                data = self._post("/chat/completions", payload, timeout)
            except urllib.error.HTTPError as err:
                body = ""
                try:
                    body = err.read().decode("utf-8", "replace")
                except Exception:
                    pass
                if err.code == 400 and mode != "plain":
                    # Model rejects the suppression param — step down the ladder
                    # and remember what this model accepts.
                    mode = "effort" if mode == "thinking" else "plain"
                    self._mode_by_model[model] = mode
                    continue
                if err.code in (408, 429, 500, 502, 503) and attempt < 3:
                    time.sleep(1.5)
                    continue
                return None
            except Exception:
                if attempt < 3:
                    time.sleep(1.0)
                    continue
                return None

            self._mode_by_model[model] = mode
            usage = data.get("usage") or {}
            self.tokens_spent += int(usage.get("total_tokens", 0))
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = message.get("content") or ""
            if not text:
                # Reasoning model spent the whole budget thinking; salvage the
                # tail of the reasoning rather than answering nothing.
                text = (message.get("reasoning_content") or "")[-300:]
            text = _THINK_BLOCK.sub("", text).strip()
            return text or None
        return None

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Cloudflare in front of the API 403s (error 1010) the default
                # Python-urllib User-Agent; any real UA string passes.
                "User-Agent": "amd-track1-agent/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
