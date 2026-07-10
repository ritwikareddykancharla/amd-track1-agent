# Hybrid Token-Efficient Routing Agent — AMD Developer Hackathon ACT II, Track 1

A containerized agent that answers the 8 benchmark task categories while
spending as few Fireworks tokens as possible. Every task goes down a
cheapest-tier-first ladder:

| Tier | Engine | Fireworks tokens | Handles |
|------|--------|------------------|---------|
| 1 | Deterministic Python (AST-evaluated arithmetic, percentage patterns) | **0** | provable math |
| 2 | Local **Gemma-2-2B-it** (Q4_K_M GGUF via llama.cpp, bundled in the image, CPU-only) | **0** | sentiment, NER, summarization, factual, word-problem math |
| 3 | Fireworks API (cheapest capable model from `ALLOWED_MODELS`, reasoning disabled, hard `max_tokens` caps) | counted | code generation, code debugging, logic, escalations |

Key properties:

- **Zero-token classification** — task routing is regex/heuristic, no model call.
- **A wrong free answer escalates instead of shipping**: tier-1/2 outputs are
  validated (e.g. sentiment must literally be one of the three labels) and
  anything suspicious is retried on Fireworks.
- **Deadline-aware**: a global budget (default 530s) guarantees the results
  file is written and the process exits 0 inside the 10-minute harness limit,
  even if the network hangs. Fireworks calls run in parallel threads while
  local CPU inference proceeds sequentially.
- **Hidden reasoning tokens are disabled** (`"thinking": {"type": "disabled"}`,
  stepping down to `reasoning_effort: "low"` and then plain requests for models
  that reject it — learned per model at runtime) and `<think>` blocks are
  stripped defensively.

## Container contract

- Reads `/input/tasks.json` — `[{ "task_id": ..., "prompt": ... }]`
- Writes `/output/results.json` — `[{ "task_id": ..., "answer": ... }]`
- Exits `0` always.
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from the
  environment at runtime. Nothing is hardcoded or bundled.

## Run it (the same way the harness does)

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

## Build

The image is built for `linux/amd64` by GitHub Actions on every push to
`main` (see `.github/workflows/build.yml`) and published to GHCR. To build
manually:

```bash
docker buildx build --platform linux/amd64 -t amd-track1-agent .
```

## Develop without Docker

The agent runs directly with Python 3.11+ (no local GGUF needed — tier 2 is
skipped and those tasks escalate to Fireworks):

```bash
INPUT_PATH=sample_input/tasks.json OUTPUT_PATH=out/results.json \
FIREWORKS_API_KEY=... ALLOWED_MODELS=... \
python -m agent.main
```

## Layout

```
agent/
  main.py              orchestrator: tiers, deadline, results contract
  classifier.py        zero-token regex classifier (8 categories)
  solvers.py           deterministic math solvers (tier 1)
  local_model.py       llama.cpp Gemma-2-2B wrapper (tier 2)
  fireworks_client.py  model picker + terse capped API calls (tier 3)
Dockerfile             python:3.11-slim + llama.cpp + bundled GGUF
sample_input/          example tasks for a local smoke test
```
