#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

IMAGE_TAG=${IMAGE_TAG:-lamtat-mcp-server:local}
PORT=${PORT:-6565}
ENV_FILE=${ENV_FILE:-.env}

if [ ! -f "$ENV_FILE" ]; then
  echo "Environment file '$ENV_FILE' not found. Create one or set ENV_FILE=/path/to/file." >&2
  exit 1
fi

echo "Building image $IMAGE_TAG..."
docker build -t "$IMAGE_TAG" .

echo "Running container on port $PORT using env file $ENV_FILE"
docker run --rm -p "$PORT":6565 --env-file "$ENV_FILE" "$IMAGE_TAG"
