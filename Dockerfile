# Judging VM runs linux/amd64, 4GB RAM / 2 vCPU. CI builds and pushes that
# platform explicitly. python:3.11 because llama-cpp-python 0.3.19 is the
# newest version with a prebuilt cp311 linux_x86_64 CPU wheel on this index;
# wheel-only install so a version bump can never silently trigger a source
# build (this image has no toolchain).
#
# Alpine, not Debian slim: the prebuilt wheel's libllama.so is linked against
# musl libc (needs libc.musl-x86_64.so.1), so on any glibc base the import
# fails at dlopen and the local tier silently dies — every task escalates to
# the paid API. That is exactly what the graded 5,437-token run was.
FROM python:3.11-alpine

# libstdc++/libgomp: C++ runtime and OpenMP for the wheel's native library.
RUN apk add --no-cache curl ca-certificates libstdc++ libgomp

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
