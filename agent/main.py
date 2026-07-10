"""Entry point: /input/tasks.json -> /output/results.json, exit 0. Always.

Routing ladder per task (cheapest tier that can answer wins):
  1. deterministic Python solver        — 0 tokens
  2. local Gemma-2-2B in the container  — 0 tokens
  3. Fireworks API                      — counted tokens, capped and terse

A global deadline keeps us inside the 10-minute harness limit: Fireworks calls
run first (network-bound, parallel), local CPU inference fills the remainder,
and if the clock runs low every unanswered task gets a best-effort escalation
or a placeholder — the results file is always written and we always exit 0.
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
        answer = try_deterministic(category, prompt)
        if answer is not None:
            answers[task_id] = answer
            _log(f"task {task_id}: solved deterministically")
        else:
            pending.append((task_id, prompt, category))

    local_queue = [t for t in pending if local.available and t[2] in LOCAL_CATEGORIES]
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
