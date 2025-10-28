# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_NAME=lamtat-mcp-server \
    HOST=0.0.0.0 \
    PORT=6565

WORKDIR /app

COPY pyproject.toml README.md ./
COPY server.py ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 6565

CMD ["lamtat-mcp-server"]
