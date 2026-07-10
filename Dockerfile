# Track 1 — Hybrid Token-Efficient Routing Agent
# Target: linux/amd64, 4GB RAM / 2 vCPU grading VM, <=10GB compressed image.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 0.3.19 is the newest version with a prebuilt cp311 linux_x86_64 CPU wheel on
# this index; wheel-only install so a version bump can never silently trigger
# a source build (which fails on the slim image and would need a toolchain).
RUN pip install --no-cache-dir \
    --only-binary=llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.19 "openai>=1.30.0"

# Bundled local model: gemma-3-4b-it Q4_K_M (~2.5GB) — the fallback floor for
# tasks the Fireworks API fails on. Local inference costs zero counted tokens;
# only API traffic counts toward the score.
RUN mkdir -p /models && curl -fL --retry 3 -o /models/gemma-3-4b-it-Q4_K_M.gguf \
    "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf"

WORKDIR /app
COPY agent /app/agent

ENV LOCAL_MODEL_PATH=/models/gemma-3-4b-it-Q4_K_M.gguf \
    LOCAL_MODEL_THREADS=2 \
    PYTHONUNBUFFERED=1

# Harness contract: read /input/tasks.json, write /output/results.json, exit 0.
CMD ["python", "-m", "agent.main"]
