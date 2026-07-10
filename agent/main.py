"""Entry point: /input/tasks.json -> /output/results.json, exit 0. Always.

Default (FIREWORKS_ONLY=1): every task goes to the Fireworks API — the tier
with the strongest models and the only one verified end to end. The bundled
local model (gemma-3-4b-it) is the fallback floor: any task the API fails on
is answered locally rather than shipped empty, because an empty answer is a
guaranteed zero at the accuracy gate.

A global deadline keeps us inside the 10-minute harness limit, the results
file is always written with every task_id, and we always exit 0.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .classifier import classify
from .fireworks_client import FireworksClient
from .local_model import LOCAL_CATEGORIES, LocalModel
from .solvers import try_deterministic

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
BUDGET_SECONDS = float(os.environ.get("RUNTIME_BUDGET_SECONDS", "530"))
LOCAL_RESERVE_SECONDS = 45.0  # bail out of local inference this early

# Accuracy-gate diagnostic: send every task to Fireworks, bypassing the
# deterministic and local tiers. The cheap tiers scored 21% on the hidden set
# (unvalidated local answers shipped wrong); this isolates the API path, which
# is the only tier we have verified end to end. Flip to "0" to restore tiering.
FIREWORKS_ONLY = os.environ.get("FIREWORKS_ONLY", "1") == "1"

_START = time.monotonic()


def _remaining() -> float:
    return BUDGET_SECONDS - (time.monotonic() - _START)


def _log(message: str) -> None:
    print(f"[{time.monotonic() - _START:6.1f}s] {message}", flush=True)


def run() -> None:
    with open(INPUT_PATH, encoding="utf-8") as fh:
        tasks = json.load(fh)

    answers: dict[str, str] = {}
    fireworks = FireworksClient()
    local = LocalModel()

    classified = []
    for task in tasks:
        task_id = str(task.get("task_id"))
        prompt = str(task.get("prompt", ""))
        category = classify(prompt)
        classified.append((task_id, prompt, category))
        _log(f"task {task_id}: {category}")

    # Tier 1 — deterministic (free, instant).
    pending = []
    for task_id, prompt, category in classified:
        answer = None if FIREWORKS_ONLY else try_deterministic(category, prompt)
        if answer is not None:
            answers[task_id] = answer
            _log(f"task {task_id}: solved deterministically")
        else:
            pending.append((task_id, prompt, category))

    local_queue = [
        t
        for t in pending
        if not FIREWORKS_ONLY and local.available and t[2] in LOCAL_CATEGORIES
    ]
    remote_queue = [t for t in pending if t not in local_queue]

    # Tier 3 first — network-bound, so it overlaps with local CPU work below.
    remote_futures = {}
    pool = ThreadPoolExecutor(max_workers=4)
    for task_id, prompt, category in remote_queue:
        remote_futures[pool.submit(fireworks.answer, category, prompt)] = task_id

    # Tier 2 — local Gemma, sequential (2 vCPU), deadline-aware.
    escalations = []
    for task_id, prompt, category in local_queue:
        if _remaining() < LOCAL_RESERVE_SECONDS:
            _log(f"task {task_id}: deadline pressure, escalating without local try")
            escalations.append((task_id, prompt, category))
            continue
        answer = local.answer(category, prompt)
        if answer is not None:
            answers[task_id] = answer
            _log(f"task {task_id}: answered locally (0 tokens)")
        else:
            escalations.append((task_id, prompt, category))

    for task_id, prompt, category in escalations:
        remote_futures[pool.submit(fireworks.answer, category, prompt)] = task_id

    try:
        for future in as_completed(remote_futures, timeout=max(_remaining(), 5.0)):
            task_id = remote_futures[future]
            try:
                answer = future.result()
            except Exception:
                answer = None
            if answer:
                answers[task_id] = answer
                _log(f"task {task_id}: answered via Fireworks")
    except TimeoutError:
        # Out of time waiting on the API — ship whatever we have.
        _log("deadline hit while waiting on Fireworks; writing partial results")
    pool.shutdown(wait=False, cancel_futures=True)

    # Fallback floor — an empty answer is a guaranteed zero, so any task the
    # API could not answer goes to the local model regardless of category
    # (strict=False: an imperfect local answer still beats an empty one).
    unanswered = [t for t in classified if t[0] not in answers]
    if unanswered:
        _log(
            f"{len(unanswered)} task(s) unanswered by Fireworks; "
            f"local fallback available={local.available}"
        )
    for task_id, prompt, category in unanswered:
        if _remaining() < 30.0:
            _log("deadline pressure; stopping local fallback")
            break
        answer = local.answer(category, prompt, strict=False)
        if answer:
            answers[task_id] = answer
            _log(f"task {task_id}: answered by local fallback (0 tokens)")
        else:
            _log(f"task {task_id}: local fallback failed too")

    # Never omit a task_id — an empty answer scores 0 for that task, but a
    # malformed/missing results file can zero the whole run.
    results = []
    for task in tasks:
        task_id = str(task.get("task_id"))
        results.append({"task_id": task_id, "answer": answers.get(task_id, "")})

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    _log(
        f"wrote {len(results)} answers; fireworks tokens spent: "
        f"{fireworks.tokens_spent}"
    )


def main() -> None:
    try:
        run()
    except Exception as err:  # noqa: BLE001 — harness contract: exit 0, file exists
        print(f"fatal: {err}", file=sys.stderr, flush=True)
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            if not os.path.exists(OUTPUT_PATH):
                with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
                    json.dump([], fh)
        except Exception:
            pass
    sys.exit(0)


if __name__ == "__main__":
    main()
