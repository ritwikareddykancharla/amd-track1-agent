# Judging VM runs linux/amd64, 4GB RAM / 2 vCPU. CI builds and pushes that
# platform explicitly. python:3.11 because llama-cpp-python 0.3.19 is the
# newest version with a prebuilt cp311 linux_x86_64 CPU wheel on this index;
# wheel-only install so a version bump can never silently trigger a source
# build (which fails on the slim image without a toolchain).
FROM python:3.11-slim

# libgomp1: the prebuilt llama-cpp-python CPU wheel links libgomp.so.1
# (OpenMP), which slim images do not ship — without it the import fails and
# the local tier silently dies (every task escalates to the paid API).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --only-binary=llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.19 "openai>=1.30.0"

# Bundled local model: gemma-3-4b-it Q4_K_M (~2.5GB). Local inference costs
# zero counted tokens; only traffic through FIREWORKS_BASE_URL is scored.
RUN mkdir -p /models && curl -fL --retry 3 -o /models/gemma-3-4b-it-Q4_K_M.gguf \
    "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf"

WORKDIR /app

COPY main.py agent.py classifier.py llm.py solvers.py local_model.py ./

# PREFERRED_MODEL pins every tier to the measured leanest allowed model
# (deepseek-v4-pro: 415 tokens vs 601 for gpt-oss on the 4 API-bound sample
# tasks, zero blanks). Guarded in llm.py: if no allowed model matches the
# fragment on grading day, tier inference takes over unchanged.
ENV LOCAL_MODEL_PATH=/models/gemma-3-4b-it-Q4_K_M.gguf \
    LOCAL_MODEL_THREADS=2 \
    PREFERRED_MODEL=deepseek \
    PYTHONUNBUFFERED=1

# Harness mounts /input and /output and injects FIREWORKS_* at run time.
ENTRYPOINT ["python", "main.py"]
