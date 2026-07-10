# Track 1 — Hybrid Token-Efficient Routing Agent
# Target: linux/amd64, 4GB RAM / 2 vCPU grading VM, <=10GB compressed image.
FROM python:3.11-slim

# Build tools only needed if the prebuilt CPU wheel for llama-cpp-python is
# unavailable for this platform; cheap insurance against a broken build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.9

# Bundled local model: gemma-2-2b-it Q4_K_M (~1.7GB). Local inference is the
# zero-Fireworks-token tier; only API traffic counts toward the score.
RUN mkdir -p /models && curl -fL --retry 3 -o /models/gemma-2-2b-it-Q4_K_M.gguf \
    "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"

WORKDIR /app
COPY agent /app/agent

ENV LOCAL_MODEL_PATH=/models/gemma-2-2b-it-Q4_K_M.gguf \
    LOCAL_MODEL_THREADS=2 \
    PYTHONUNBUFFERED=1

# Harness contract: read /input/tasks.json, write /output/results.json, exit 0.
CMD ["python", "-m", "agent.main"]
