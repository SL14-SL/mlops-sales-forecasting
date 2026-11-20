FROM python:3.12.16-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.13 \
    /uv \
    /uvx \
    /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
ARG INSTALL_AWS=false

COPY pyproject.toml uv.lock ./

RUN if [ "$INSTALL_AWS" = "true" ]; then \
        uv sync \
            --locked \
            --no-dev \
            --extra aws \
            --no-install-project; \
    else \
        uv sync \
            --locked \
            --no-dev \
            --no-install-project; \
    fi

COPY README.md ./
COPY configs ./configs
COPY examples ./examples
COPY src ./src

RUN if [ "$INSTALL_AWS" = "true" ]; then \
        uv sync \
            --locked \
            --no-dev \
            --extra aws \
            --no-editable; \
    else \
        uv sync \
            --locked \
            --no-dev \
            --no-editable; \
    fi


FROM python:3.12.16-slim AS runtime

RUN apt-get update \
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends libgomp1 \
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

COPY --from=builder \
    --chown=appuser:appuser \
    /app/examples \
    /app/examples
    
COPY --from=builder \
    --chown=appuser:appuser \
    /app/src \
    /app/src

USER appuser

EXPOSE 8000 8501

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=3)"

CMD ["uvicorn", "mlops_sales_forecasting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]