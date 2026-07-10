# Token-Miser Router Agent

A routing agent for **AMD Developer Hackathon ACT II — Track 1** that answers tasks across eight categories while spending as close to zero Fireworks API tokens as possible.

[![build](https://github.com/ritwikareddykancharla/amd-track1-agent/actions/workflows/build.yml/badge.svg)](https://github.com/ritwikareddykancharla/amd-track1-agent/actions)

Image: `ghcr.io/ritwikareddykancharla/amd-track1-agent:latest` (public, linux/amd64, ~1.77 GB — a local Gemma-2-2B model is baked in)

---

## Why the agent is built this way

Track 1 scoring has two stages:

1. **Accuracy gate.** An LLM judge checks your answers. Fail the gate and nothing else matters.
2. **Token ranking.** Everyone who passes is ranked by total tokens recorded by the judging proxy — *fewest wins*. Only traffic through `FIREWORKS_BASE_URL` is counted; local inference inside the container is free.

This inverts the usual objective. The winning move is not a smarter model — it is **not calling the API at all** whenever a cheaper path can produce an answer that is provably or verifiably correct. Every design decision below follows from that one constraint, plus its counterweight: a wrong answer risks the accuracy gate that all the token-saving depends on, so cheaper tiers are only allowed to answer when their output can be validated.

## Architecture

Each task is classified for free, then walks down a ladder of increasingly expensive tiers. The first tier that can answer *and pass validation* wins; everything else escalates.

```
/input/tasks.json
       │
       ▼
 regex classifier ................................. 0 tokens
       │
       ├─ Tier 1  deterministic Python solvers ..... 0 tokens
       ├─ Tier 2  local Gemma-2-2B (in-container) .. 0 tokens
       └─ Tier 3  Fireworks API .................... counted
       │
       ▼
/output/results.json
```

| Stage | File | Handles | Cost |
|---|---|---|---|
| Classifier | `agent/classifier.py` | routes all 8 categories | 0 tokens |
| Tier 1 | `agent/solvers.py` | arithmetic, percentages | 0 tokens |
| Tier 2 | `agent/local_model.py` | sentiment, NER, summarization, factual | 0 tokens |
| Tier 3 | `agent/fireworks_client.py` | code gen, code debug, logic, escalations | counted |
| Orchestrator | `agent/main.py` | deadline budget, parallelism, output contract | — |

### Classification (0 tokens)

`agent/classifier.py` tags each prompt with one of the eight graded categories (facts, math, sentiment, summarization, NER, code debugging, logic, code generation) using regex heuristics — code-fence and keyword detection, arithmetic patterns, syllogism markers, and so on. Asking a model "what kind of task is this?" would cost tokens on *every* task, which defeats the whole design; regexes cost nothing and are wrong rarely enough that the fallback (`factual`, the cheapest category to answer) is safe.

### Tier 1 — deterministic solvers (0 tokens)

`agent/solvers.py` answers math it can *prove*. Expressions like `17 * 23 + 4` are parsed into a Python AST and evaluated with a whitelist of safe operators — real evaluation, not `eval()`, and structurally incapable of hallucinating. Percentage phrasings ("What is 15% of 240?") are recognized and computed directly. If the prompt doesn't parse cleanly, the solver declines and the task escalates — it never guesses.

### Tier 2 — local Gemma-2-2B (0 tokens)

A quantized Gemma-2-2B (Q4_K_M GGUF) ships **inside the Docker image** and runs on the grading VM's 2 vCPUs via `llama-cpp-python`. Because it never touches `FIREWORKS_BASE_URL`, its inference is free under the scoring rules. It handles the fuzzy-but-easy categories: sentiment, NER, summarization, and factual lookups.

The critical detail is **output validation before shipping** (`agent/local_model.py`): a sentiment answer must literally be `positive`, `negative`, or `neutral`; a math answer must parse as a number. Anything that fails validation escalates to Tier 3 instead of being submitted — a 2B model is allowed to save tokens, never to gamble the accuracy gate. The model is lazy-loaded, so runs whose tasks are all deterministic pay no startup cost, and dev machines without the GGUF simply fall through to Tier 3.

### Tier 3 — Fireworks API (the only counted tokens)

What remains — code generation, code debugging, multi-step logic, and any tier-2 escalations — goes to Fireworks through `agent/fireworks_client.py`, which is built entirely around minimizing counted tokens:

**Reasoning suppression.** Hosted "thinking" models bill their hidden chain-of-thought: in testing, asking GLM the capital of France burned ~30 invisible reasoning tokens for a 3-token answer. The client sends `"thinking": {"type": "disabled"}` to switch that off. Models that reject the parameter get a fallback ladder, learned per model at runtime from HTTP 400 responses and cached for the rest of the run:

```
"thinking": {"type": "disabled"}   → GLM, Kimi, DeepSeek
"reasoning_effort": "low"          → gpt-oss family
plain request                      → everything else
```

**Cheapest-capable routing.** `ALLOWED_MODELS` is injected by the harness at grading time, so the client cannot hardcode model names. Instead it ranks whatever list it receives with a price heuristic (name-fragment table: Gemma cheapest → Llama → Qwen → … → Kimi/gpt-oss most expensive, unknown names rank mid-table) and picks the cheapest model that fits the category — small models for facts and sentiment, one step up for logic, the strongest model only for code. gpt-oss is explicitly avoided for code tasks: it cannot fully disable reasoning, and it measured **290 tokens vs 101** for a thinking-disabled peer on the identical bug-fix task.

**Terse by contract.** Every category gets a one-line system prompt ("Reply with only the final numeric answer.") and a hard `max_tokens` cap, so the model physically cannot ramble:

| Category | max_tokens | Category | max_tokens |
|---|---|---|---|
| sentiment | 4 | ner | 64 |
| math | 16 | logic | 96 |
| factual | 48 | summarization | 96 |
| | | code gen / debug | 512 |

**Defensive handling.** `<think>…</think>` blocks are stripped if a model emits them anyway; if a reasoning model spends its whole budget thinking and returns empty content, the tail of the reasoning is salvaged rather than answering nothing. Transient errors (408/429/5xx) retry with backoff.

## Staying inside the harness rules

The grading harness gives the container 10 minutes on a 4 GB / 2 vCPU VM, then kills it. Producing *no* results file is worse than producing partial answers, so `agent/main.py` is built around a **530-second global budget**:

- Fireworks calls (network-bound) run in a thread pool **while** the local model works through its queue on the CPU — the two overlap instead of queueing.
- If the budget runs low, remaining local-tier tasks skip straight to the API; if it runs out entirely, whatever answers exist are written.
- Every `task_id` always gets an entry in `results.json` (empty string if truly unanswered — a blank answer loses one task, a missing file can zero the run).
- The process exits `0` unconditionally: any crash in any tier is caught, logged, and routed around.

## Measured results

On the bundled sample (`sample_input/tasks.json`, 9 tasks covering all 8 categories), run in the **worst case** — no local model, so every fuzzy task escalates to the API:

| Metric | Result |
|---|---|
| Accuracy | 9 / 9 correct |
| Fireworks tokens | 627 total |
| Wall time | ~5 seconds |
| Arithmetic tasks | 0 tokens (Tier 1) |

In the real grading container the local Gemma tier additionally absorbs sentiment, NER, summarization, and factual tasks, so counted spend drops further.

## Container contract

- Reads `/input/tasks.json` — `[{ "task_id": ..., "prompt": ... }]`
- Writes `/output/results.json` — `[{ "task_id": ..., "answer": ... }]`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from the environment at runtime — nothing is hardcoded or bundled
- Always exits `0`, always writes an entry for every `task_id`

## Running it

Exactly as the harness does:

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

### Development without Docker

The agent runs directly on Python 3.11+ with no dependencies beyond the standard library (Tier 2 needs `llama-cpp-python` and a GGUF, but without them those tasks escalate to Fireworks — same code path, just pricier):

```bash
INPUT_PATH=sample_input/tasks.json OUTPUT_PATH=out/results.json \
FIREWORKS_API_KEY=... ALLOWED_MODELS=... \
python -m agent.main
```

### Build pipeline

GitHub Actions (`.github/workflows/build.yml`) rebuilds the `linux/amd64` image on every push to `main` and publishes it to GHCR. The ~1.7 GB Gemma GGUF is downloaded at build time and baked into the image, so the grading VM never downloads anything at runtime.

## Repository layout

```
agent/
  main.py              orchestrator: tier routing, deadline budget, output contract
  classifier.py        zero-token regex classifier (8 categories)
  solvers.py           deterministic AST-based math solvers (Tier 1)
  local_model.py       llama.cpp Gemma-2-2B wrapper + output validation (Tier 2)
  fireworks_client.py  price-ranked model routing + reasoning suppression (Tier 3)
Dockerfile             python:3.11-slim + llama-cpp-python CPU wheel + bundled GGUF
sample_input/          9 example tasks covering all 8 categories
.github/workflows/     CI: build and push the image to GHCR
```
