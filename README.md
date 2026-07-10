# 🧭 Token-Miser Router Agent

**AMD Developer Hackathon ACT II — Track 1** · [![build](https://github.com/ritwikareddykancharla/amd-track1-agent/actions/workflows/build.yml/badge.svg)](https://github.com/ritwikareddykancharla/amd-track1-agent/actions)

## 💡 The idea

Track 1 scores like this: first your answers must pass an accuracy gate, and
**then submissions are ranked by who spent the fewest Fireworks API tokens**.
That flips the usual objective — the winning move isn't a smarter model, it's
*not calling the API at all* whenever a cheaper path can produce a correct
answer.

So this agent treats the Fireworks API as a last resort. Every task walks down
a ladder of increasingly expensive tiers, and stops at the first one that can
answer **provably or verifiably** correctly:

```
📥 /input/tasks.json
   │
   ▼
🔍 classify (regex, 0 tokens)
   │
   ├─ 1️⃣ deterministic Python ── provable math ───────────── 0 tokens 🆓
   ├─ 2️⃣ local Gemma-2-2B ────── sentiment/NER/summary/... ── 0 tokens 🆓
   └─ 3️⃣ Fireworks API ───────── code, logic, escalations ─── counted 💰
   │
   ▼
📤 /output/results.json
```

## 🚶 Follow one task through the pipeline

Take `"What is 15% of 240?"`:

1. **🔍 Classify (0 tokens).** A regex classifier (`agent/classifier.py`) tags
   it as `math` — it matches the `\d\s*%\s*of\s+\d` pattern. No model is ever
   asked "what kind of task is this"; that would cost tokens on every single
   task, which defeats the purpose.
2. **1️⃣ Try to solve it deterministically (0 tokens).** `agent/solvers.py`
   recognises the percentage pattern and computes `0.15 × 240 = 36` with plain
   Python. General arithmetic like `17 * 23 + 4` is parsed into a Python AST
   and evaluated with a whitelist of safe operators — real math, not `eval()`,
   and it cannot hallucinate. Answer found → done, **0 tokens spent**.
3. **2️⃣ If tier 1 can't prove it** (say, a word problem), the prompt goes to a
   **local Gemma-2-2B** model that ships *inside* the Docker image and runs on
   the 2 CPUs of the grading VM. Local inference doesn't touch the Fireworks
   API, so it's free by the scoring rules. Its output is **validated before
   shipping**: a sentiment answer must literally be `positive`, `negative`, or
   `neutral`; a math answer must parse as a number. Fail validation → the task
   *escalates* instead of submitting a guess, because one wrong answer risks
   the accuracy gate that all the token-saving depends on.
4. **3️⃣ Only what's left hits Fireworks** — code generation, code debugging,
   multi-step logic, and any tier-2 escapees. Even then, every trick below is
   applied to keep the counted tokens tiny.

## ✂️ How tier 3 spends as little as possible

- **🧠 Reasoning tokens are billed even though you never see them.** Hosted
  "thinking" models burn hidden chain-of-thought tokens before answering —
  asking GLM the capital of France cost ~30 tokens of invisible reasoning for
  a 3-token answer. The client sends `"thinking": {"type": "disabled"}` to
  turn that off. Models that reject the parameter get a graceful ladder,
  learned per model at runtime from 400 responses:
  `thinking: disabled` → `reasoning_effort: "low"` → plain request.
- **📉 Cheapest-capable model routing.** `ALLOWED_MODELS` is injected by the
  harness, so the client ranks whatever it's given by a price heuristic and
  picks the cheapest model that fits the category — small models for facts and
  sentiment, a strong model only for code. (gpt-oss is specifically avoided
  for code: it can't fully disable reasoning, and measured 290 tokens vs 101
  for a thinking-disabled peer on the same bug-fix task.)
- **🎯 Terse by contract.** Each category gets a one-line system prompt like
  *"Reply with only the final numeric answer."* plus a hard `max_tokens` cap —
  sentiment is capped at **4 tokens**, math at 16, code at 512. The model
  physically cannot ramble.
- **🧹 Defensive stripping.** If a model emits `<think>…</think>` anyway, it's
  stripped before the answer is written.

## 🛡️ Why it can't blow the harness rules

The grading harness gives 10 minutes, then kills the container. Producing *no*
results file is worse than producing partial answers, so `agent/main.py` is
built around a **530-second global budget**:

- Fireworks calls run in a thread pool *while* the local model chews through
  its queue on the CPU — network wait and CPU inference overlap instead of
  queueing.
- When the budget runs out, whatever answers exist are written; every
  `task_id` always gets an entry (blank if truly unanswered).
- The process **exits 0 no matter what fails** — a crash in any tier is
  caught, logged, and routed around.

## 📊 Measured on the bundled sample (9 tasks, all 8 categories)

Run without the local model (worst case — everything fuzzy escalates to the
API): **9/9 correct, 627 total Fireworks tokens, 5 seconds**. The two
arithmetic tasks cost 0. In the real container the local Gemma tier absorbs
sentiment/NER/summarization/factual too, so the counted spend drops further.

## 📦 Container contract

- 📥 Reads `/input/tasks.json` — `[{ "task_id": ..., "prompt": ... }]`
- 📤 Writes `/output/results.json` — `[{ "task_id": ..., "answer": ... }]`
- 🟢 Exits `0` always — every task_id gets an entry, no matter what fails.
- 🔑 Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from the
  environment at runtime. **Nothing is hardcoded or bundled.**

## 🚀 Run it (the same way the harness does)

```bash
docker pull ghcr.io/ritwikareddykancharla/amd-track1-agent:latest

docker run --rm \
  -v "$(pwd)/sample_input:/input:ro" \
  -v "$(pwd)/out:/output" \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS="accounts/fireworks/models/..." \
  ghcr.io/ritwikareddykancharla/amd-track1-agent:latest

cat out/results.json
```

## 🧪 Develop without Docker

The agent runs directly on Python 3.11+. Without a local GGUF, tier 2 is
skipped and those tasks escalate to Fireworks — same code path, just pricier:

```bash
INPUT_PATH=sample_input/tasks.json OUTPUT_PATH=out/results.json \
FIREWORKS_API_KEY=... ALLOWED_MODELS=... \
python -m agent.main
```

The `linux/amd64` image is rebuilt by GitHub Actions on every push to `main`
(`.github/workflows/build.yml`) and published to GHCR — the ~1.7GB Gemma GGUF
is baked in at build time, so the grading VM never downloads anything.

## 🗂️ Layout

```
agent/
  main.py              🧭 orchestrator: tiers, deadline budget, results contract
  classifier.py        🔍 zero-token regex classifier (8 categories)
  solvers.py           🧮 deterministic AST math solvers (tier 1)
  local_model.py       🦙 llama.cpp Gemma-2-2B wrapper + output validation (tier 2)
  fireworks_client.py  🎆 price-ranked model picker + terse capped API calls (tier 3)
Dockerfile             🐳 python:3.11-slim + llama.cpp CPU wheel + bundled GGUF
sample_input/          🧪 9 example tasks covering all 8 categories
```
