# Judging VM runs linux/amd64. CI builds and pushes that platform explicitly.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "openai>=1.30.0"

COPY main.py agent.py classifier.py llm.py ./

# Harness mounts /input and /output and injects FIREWORKS_* at run time.
ENTRYPOINT ["python", "main.py"]
