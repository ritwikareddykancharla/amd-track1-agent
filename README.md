# 🧭 Token-Miser Router Agent

**AMD Developer Hackathon ACT II — Track 1** · [![build](https://github.com/ritwikareddykancharla/amd-track1-agent/actions/workflows/build.yml/badge.svg)](https://github.com/ritwikareddykancharla/amd-track1-agent/actions)

An agent that answers all 8 benchmark task categories while spending as close to
**zero Fireworks tokens** as possible. Every task walks down a
cheapest-tier-first ladder — and only pays for what it truly can't get for free.

```
📥 /input/tasks.json ──▶ 🔍 classify ──▶ 🪜 route down the ladder ──▶ 📤 /output/results.json
```

## 🪜 The three-tier ladder

| Tier | Engine | 💸 Fireworks tokens | Handles |
|------|--------|---------------------|---------|
| 1️⃣ | **Deterministic Python** — AST-evaluated arithmetic, percentage patterns | **0** | provable math |
| 2️⃣ | **Local Gemma-2-2B-it** — Q4_K_M GGUF via llama.cpp, baked into the image, CPU-only | **0** | sentiment, NER, summarization, factual, word-problem math |
| 3️⃣ | **Fireworks API** — cheapest capable model from `ALLOWED_MODELS`, reasoning off, hard `max_tokens` caps | counted 💰 | code generation, code debugging, logic, escalations |

## ✨ Why it spends so little

- 🆓 **Zero-token classification** — routing is pure regex/heuristics, no model call ever.
- 🧠 **Hidden reasoning tokens are billed — so we switch them off.** Each model
  gets a suppression ladder learned at runtime:
  `"thinking": {"type": "disabled"}` → `reasoning_effort: "low"` → plain.
  Leftover `<think>` blocks are stripped defensively.
- ✅ **A wrong free answer escalates instead of shipping** — tier-1/2 outputs are
  validated (sentiment must literally be one of the three labels, math must
  parse as a number, …) and anything suspicious retries on Fireworks.
- 🎯 **Terse by contract** — per-category system prompts ("Reply with only the
  final numeric answer.") plus hard `max_tokens` caps (math: 16, sentiment: 4).
- ⏱️ **Deadline-aware** — a global budget (530s) guarantees the results file is
  written and the process exits 0 inside the 10-minute harness limit, even if
  the network hangs. Fireworks calls run in parallel threads while local CPU
  inference proceeds sequentially, so the clock is never wasted.

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

## 🛠️ Build

The `linux/amd64` image is built by GitHub Actions on every push to `main`
(see `.github/workflows/build.yml`) and published to GHCR. To build manually:

```bash
docker buildx build --platform linux/amd64 -t amd-track1-agent .
```

## 🧪 Develop without Docker

The agent runs directly on Python 3.11+ — no local GGUF needed (tier 2 is
skipped and those tasks escalate to Fireworks):

```bash
INPUT_PATH=sample_input/tasks.json OUTPUT_PATH=out/results.json \
FIREWORKS_API_KEY=... ALLOWED_MODELS=... \
python -m agent.main
```

## 🗂️ Layout

```
agent/
  main.py              🧭 orchestrator: tiers, deadline, results contract
  classifier.py        🔍 zero-token regex classifier (8 categories)
  solvers.py           🧮 deterministic math solvers (tier 1)
  local_model.py       🦙 llama.cpp Gemma-2-2B wrapper (tier 2)
  fireworks_client.py  🎆 model picker + terse capped API calls (tier 3)
Dockerfile             🐳 python:3.11-slim + llama.cpp + bundled GGUF
sample_input/          🧪 example tasks for a local smoke test
```
