# syntax=docker/dockerfile:1
# Build on the Ubuntu x86_64 GPU host. CUDA user-space libraries come from uv.lock.
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv
FROM python:3.12-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv
# Triton JIT needs a C compiler; kernels may fetch code through git at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 jeff \
    && install -d -o jeff -g jeff /cache

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --extra cuda --no-editable

ENV JEFF_BACKEND=pytorch \
    JEFF_DEVICE=cuda \
    JEFF_CHECKPOINT=/models/jeff-2b \
    JEFF_HOST=0.0.0.0 \
    PORT=8765 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    CUDA_CACHE_PATH=/cache/cuda
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD ["/app/.venv/bin/python", "-c", "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5)); assert r['status']=='ready'"]
CMD ["/app/.venv/bin/jeff-serve"]
