"""Fireworks access via the OpenAI-compatible SDK.

Everything comes from the environment the harness injects at evaluation
time: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS. Model tiers
are inferred from whatever model IDs arrive in ALLOWED_MODELS — never
hardcoded — so the agent adapts if the list changes on launch day.
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache

from openai import APIStatusError, OpenAI


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader for local runs; real env vars always win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=25.0,      # per-request rule is < 30s
        max_retries=2,
    )


@lru_cache(maxsize=1)
def _allowed() -> tuple[str, ...]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = tuple(m.strip() for m in raw.split(",") if m.strip())
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty")
    return models


# --- Tier inference ----------------------------------------------------------
# strong = biggest general model, code = code-specialised (else strong),
# cheap  = fewest active params (MoE-aware), preferring quantized on ties.

_MOE = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")   # 8x7b -> 56
_ACTIVE = re.compile(r"\ba(\d+)b\b")            # ...-a4b -> 4 active
_DENSE = re.compile(r"(\d+)\s*b\b")             # ...-8b -> 8
_CODE = re.compile(r"code|coder")
_QUANT = re.compile(r"nvfp4|fp4|fp8|int8|int4|awq|gptq|gguf")


def _total(mid: str) -> int:
    mid = mid.lower()
    moe = _MOE.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE.finditer(mid)]
    return max(sizes) if sizes else 100


def _active(mid: str) -> int:
    m = _ACTIVE.search(mid.lower())
    return int(m.group(1)) if m else _total(mid)


def _select_tiers(models: list[str]) -> dict[str, str]:
    """Map cheap/strong/code onto concrete IDs. Pure (no env) so tier choice
    is unit-testable against any model list without network access.

    The strong tier goes to the biggest general model even when it is a
    reasoning model: with reasoning_effort=none its hidden thinking stays
    off and the terse per-category prompts control its token cost. Some
    thinking models return empty content when suppressed, which the
    blank-answer fallback in complete() absorbs."""
    coders = [m for m in models if _CODE.search(m.lower())]
    general = [m for m in models if not _CODE.search(m.lower())] or models
    # gpt-oss is a measured token burner (its reasoning cannot be fully
    # suppressed: 354 vs 134 completion tokens on the identical code-gen
    # task), so strong/code prefer any alternative when one exists.
    non_oss = [m for m in general if "gpt-oss" not in m.lower()] or general
    strong = max(non_oss, key=lambda m: (_total(m), not _QUANT.search(m.lower())))
    cheap = min(models, key=lambda m: (_active(m), not _QUANT.search(m.lower())))
    code = max(coders, key=_total) if coders else strong
    return {"cheap": cheap, "strong": strong, "code": code}


@lru_cache(maxsize=1)
def tiers() -> dict[str, str]:
    return _select_tiers(list(_allowed()))


# PREFERRED_MODEL pins every tier to one allowed model chosen for token
# efficiency. Token counts are per-model (each has its own tokenizer), so the
# same text bills differently; measurement picked the leanest allowed model.
# Guarded: if no allowed model matches, tier inference is used unchanged, so
# this can never select an out-of-list model or break on an unexpected list.
_PREFERRED = os.environ.get("PREFERRED_MODEL", "").strip().lower()


def _preferred_model() -> str | None:
    if not _PREFERRED:
        return None
    for m in _allowed():
        if _PREFERRED in m.lower():
            return m
    return None


def model_for(tier: str) -> str:
    explicit = os.environ.get("MODEL") or os.environ.get(f"MODEL_{tier.upper()}")
    if explicit:
        return explicit
    return _preferred_model() or tiers()[tier]


def describe_tiers() -> str:
    return "  ".join(f"{t}={model_for(t)}" for t in ("cheap", "strong", "code"))


# --- Completions -------------------------------------------------------------

_LOCK = threading.Lock()
_USAGE = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
# Models that rejected a request carrying reasoning_effort; they get plain
# OpenAI-compatible calls from then on.
_NO_EFFORT: set[str] = set()
# 'none' suppresses hidden thinking: measured to roughly halve tokens and
# stop reasoning models burning their answer cap on visible step-by-step
# preambles. Best-effort only — any request error retries as a plain call.
_EFFORT = os.environ.get("REASONING_EFFORT", "none")

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def usage() -> dict[str, int]:
    with _LOCK:
        return dict(_USAGE)


def _record(u) -> None:
    if not u:
        return
    with _LOCK:
        _USAGE["prompt"] += u.prompt_tokens or 0
        _USAGE["completion"] += u.completion_tokens or 0
        _USAGE["total"] += u.total_tokens or 0
        _USAGE["calls"] += 1


def _chat(model: str, messages: list[dict], max_tokens: int) -> str:
    kwargs = {}
    if _EFFORT and model not in _NO_EFFORT:
        kwargs["reasoning_effort"] = _EFFORT
    try:
        resp = _client().chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=0, **kwargs,
        )
    except Exception as exc:
        if not kwargs:
            raise
        # Whatever went wrong, a plain standard-fields call is the shape
        # every OpenAI-compatible endpoint must accept — retry with that.
        # Only a request rejection means the endpoint dislikes the extra
        # field; transient failures (429/5xx/timeouts) keep suppression on.
        if isinstance(exc, APIStatusError) and exc.status_code in (400, 404, 422):
            _NO_EFFORT.add(model)
        resp = _client().chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=0,
        )
    _record(getattr(resp, "usage", None))
    return _THINK.sub("", resp.choices[0].message.content or "").strip()


def complete(prompt: str, system: str, max_tokens: int, model: str,
             fallback_model: str | None = None) -> str:
    """One completion. A blank answer or a hard failure retries once on the
    fallback model — a blank answer scores zero, so it's worth the tokens."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    use_fb = fallback_model and fallback_model != model
    try:
        answer = _chat(model, messages, max_tokens)
    except Exception:
        if not use_fb:
            raise
        answer = ""
    if not answer and use_fb:
        answer = _chat(fallback_model, messages, max_tokens)
    return answer
