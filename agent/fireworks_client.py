"""Fireworks API tier — the only tokens that count against our score.

Calling convention deliberately mirrors a submission that verifiably passes
the grading harness (github.com/omerdduran/token-router, MIT), because our
previous hand-rolled urllib client produced zero answers inside the grading
VM while working perfectly against api.fireworks.ai directly:

- The official OpenAI SDK talks to the harness-injected FIREWORKS_BASE_URL
  (the SDK owns URL joining, retries, and connection handling).
- ``reasoning_effort="none"`` suppresses hidden thinking tokens (they are
  billed and scored). Models that reject the parameter get it dropped and
  are remembered. This replaces the vendor-specific ``"thinking"`` payload
  the proxy may not forward.
- Model tiers (cheap / strong / code) are inferred from whatever IDs arrive
  in ALLOWED_MODELS — MoE-aware, never hardcoded — so the agent adapts to
  the launch-day list (which is NOT the practice list).
- A blank answer retries once on the opposite tier: a blank scores zero, so
  the retry tokens are always worth it. (Known trap: gemma-4 models return
  empty content when reasoning is suppressed.)
- Math/logic prompts ask for brief VISIBLE steps ending in 'Answer: <value>'.
  With hidden reasoning off, visible chain-of-thought is the only reasoning
  a strong model gets — trimming it measurably collapses accuracy.
- Startup and every failure log the shape of the environment (never the key)
  so a failing grading run is debuggable from its log.
"""
from __future__ import annotations

import json
import os
import re
import threading

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

CHEAP, STRONG, CODE = "cheap", "strong", "code"

_BASE = (
    "Answer in English. Be concise and direct; no preamble, no restating "
    "the question."
)

# Per category: (system prompt, max_tokens cap, model tier). Caps are safety
# rails sized so a correct answer is never truncated mid-thought — a clipped
# answer reads as wrong to the LLM judge.
_CONFIG: dict[str, tuple[str, int, str]] = {
    "factual": (
        f"{_BASE} Give a correct, clear answer in under 120 words.",
        320, STRONG,
    ),
    "math": (
        f"{_BASE} Work through it in brief steps, then end with "
        f"'Answer: <value>' on its own line.",
        400, STRONG,
    ),
    "sentiment": (
        f"{_BASE} State the sentiment as positive, negative, or neutral, "
        f"then one short reason.",
        120, CHEAP,
    ),
    "summarization": (
        f"{_BASE} Output only the summary and obey any length or format "
        f"constraint stated in the task.",
        240, CHEAP,
    ),
    "ner": (
        f"{_BASE} List each entity as 'label: value', one per line, using "
        f"the labels person, organization, location, date.",
        260, CHEAP,
    ),
    "code_debug": (
        f"{_BASE} State the bug in one sentence, then give the corrected "
        f"code in a single fenced block.",
        520, CODE,
    ),
    "code_gen": (
        f"{_BASE} Output only the code in a single fenced block — correct, "
        f"complete, and self-contained.",
        520, CODE,
    ),
    "logic": (
        f"{_BASE} Reason in brief numbered steps, checking each constraint, "
        f"then end with 'Answer: <value>' on its own line.",
        460, STRONG,
    ),
}

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# --- Tier inference -----------------------------------------------------------
# strong = biggest general model, code = code-specialised (else strong),
# cheap  = fewest ACTIVE params (MoE-aware), preferring quantized on ties.
_MOE = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")  # 8x7b -> 56 total
_ACTIVE = re.compile(r"\ba(\d+)b\b")           # ...-a4b -> 4 active
_DENSE = re.compile(r"(\d+)\s*b\b")            # ...-8b -> 8
_CODE_HINT = re.compile(r"code|coder", re.I)
_QUANT = re.compile(r"nvfp4|fp4|fp8|int8|int4|awq|gptq|gguf", re.I)


def _total_params(model_id: str) -> int:
    mid = model_id.lower()
    moe = _MOE.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE.finditer(mid)]
    # No size in the name usually means a flagship — treat as big.
    return max(sizes) if sizes else 100


def _active_params(model_id: str) -> int:
    m = _ACTIVE.search(model_id.lower())
    return int(m.group(1)) if m else _total_params(model_id)


def _select_tiers(models: list[str]) -> dict[str, str]:
    coders = [m for m in models if _CODE_HINT.search(m)]
    general = [m for m in models if not _CODE_HINT.search(m)] or list(models)
    strong = max(general, key=lambda m: (_total_params(m), not _QUANT.search(m)))
    cheap = min(models, key=lambda m: (_active_params(m), not _QUANT.search(m)))
    code = max(coders, key=_total_params) if coders else strong
    return {CHEAP: cheap, STRONG: strong, CODE: code}


def _parse_allowed(raw: str) -> list[str]:
    """Contract says comma-separated; accept JSON-array/quoted shapes too."""
    raw = raw.strip()
    if not raw:
        return []
    if raw[0] in "[{":
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        data = value
                        break
            if isinstance(data, list):
                return [m for m in (str(x).strip() for x in data) if m]
        except (json.JSONDecodeError, TypeError):
            pass
    parts = re.split(r"[,\n;]+", raw)
    cleaned = [p.strip().strip("[]{}\"' \t") for p in parts]
    return [p for p in cleaned if p]


class FireworksClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = (
            os.environ.get("FIREWORKS_BASE_URL", "").strip() or DEFAULT_BASE_URL
        )
        raw_allowed = os.environ.get("ALLOWED_MODELS", "")
        self.allowed = _parse_allowed(raw_allowed)
        self.tiers = _select_tiers(self.allowed) if self.allowed else {}
        self.tokens_spent = 0
        self._lock = threading.Lock()
        # Models that rejected reasoning_effort; stop sending it to them.
        self._no_effort: set[str] = set()
        self._effort = os.environ.get("REASONING_EFFORT", "none")
        self._sdk = (
            OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=25.0,  # per-request harness rule is < 30s
                max_retries=2,
            )
            if self.available
            else None
        )
        # Environment shape to stdout so a failing grading run is debuggable
        # from its log. The key VALUE is never printed.
        print(
            "[fireworks] api_key="
            + (f"set(len={len(self.api_key)})" if self.api_key else "MISSING")
            + f" base_url={self.base_url!r}"
            + f" allowed_raw={raw_allowed!r}"
            + f" allowed_parsed={self.allowed}"
            + f" tiers={self.tiers}",
            flush=True,
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.allowed)

    def answer(self, category: str, prompt: str, timeout: float = 60.0) -> str | None:
        if not self.available:
            return None
        system, max_tokens, tier = _CONFIG.get(category, _CONFIG["factual"])
        primary = self.tiers[tier]
        # Blank/failed answers retry on the opposite general tier — a blank
        # scores zero, so the retry is always worth its tokens.
        fallback = self.tiers[STRONG if tier == CHEAP else CHEAP]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = ""
        try:
            text = self._chat(primary, messages, max_tokens)
        except Exception as err:
            print(
                f"[fireworks] {type(err).__name__} model={primary} "
                f"category={category}: {err}",
                flush=True,
            )
        if not text and fallback != primary:
            try:
                text = self._chat(fallback, messages, max_tokens)
            except Exception as err:
                print(
                    f"[fireworks] fallback {type(err).__name__} "
                    f"model={fallback} category={category}: {err}",
                    flush=True,
                )
        return text or None

    def _chat(self, model: str, messages: list[dict], max_tokens: int) -> str:
        kwargs: dict = {}
        if self._effort and model not in self._no_effort:
            # Hidden reasoning is billed and scored; "none" turns it off.
            kwargs["reasoning_effort"] = self._effort
        try:
            response = self._sdk.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                **kwargs,
            )
        except Exception as exc:
            if kwargs and "reasoning" in str(exc).lower():
                # This model rejects the parameter — drop it, remember, retry.
                self._no_effort.add(model)
                response = self._sdk.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                )
            else:
                raise
        usage = getattr(response, "usage", None)
        if usage is not None:
            with self._lock:
                self.tokens_spent += int(usage.total_tokens or 0)
        content = response.choices[0].message.content or ""
        return _THINK_BLOCK.sub("", content).strip()
