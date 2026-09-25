FROM python:3.14.7-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.13 \
    /uv \
    /uvx \
    /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --locked \
    --no-dev \
    --no-install-project

COPY README.md ./
COPY configs ./configs
COPY src ./src

RUN uv sync \
    --locked \
    --no-dev \
    --no-editable


FROM python:3.14.7-slim AS runtime

RUN apt-get update \
    && apt-get upgrade --yes \
    && rm -rf /var/lib/apt/lists/*
    
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd \
    --create-home \
    --uid 10001 \
    appuser

RUN mkdir -p /app/runtime/mlflow \
    && chown -R appuser:appuser /app/runtime
    
WORKDIR /app

COPY --from=builder \
    --chown=appuser:appuser \
    /app/.venv \
    /app/.venv

COPY --from=builder \
    --chown=appuser:appuser \
    /app/pyproject.toml \
    /app/pyproject.toml

COPY --from=builder \
    --chown=appuser:appuser \
    /app/configs \
    /app/configs

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=3)"

CMD ["uvicorn", "mlops_sales_forecasting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]