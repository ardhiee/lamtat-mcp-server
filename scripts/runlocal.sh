#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -e .

export APP_NAME=${APP_NAME:-lamtat-mcp-server}
export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-6565}
export S3_BUCKET_NAME=${S3_BUCKET_NAME:-}
export AWS_REGION=${AWS_REGION:-$AWS_DEFAULT_REGION}
export BEDROCK_REGION=${BEDROCK_REGION:-$AWS_REGION}
export BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID:-arn:aws:bedrock:ap-southeast-1:243026918123:inference-profile/global.cohere.embed-v4:0}
export BEDROCK_INPUT_TYPE=${BEDROCK_INPUT_TYPE:-}
export BEDROCK_EMBEDDING_TYPES=${BEDROCK_EMBEDDING_TYPES:-}
export BEDROCK_OUTPUT_DIMENSION=${BEDROCK_OUTPUT_DIMENSION:-}
export BEDROCK_TRUNCATE=${BEDROCK_TRUNCATE:-}
export OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT:-}
export OPENSEARCH_INDEX=${OPENSEARCH_INDEX:-}
export CHUNK_SIZE=${CHUNK_SIZE:-1000}
export CHUNK_OVERLAP=${CHUNK_OVERLAP:-100}

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python server.py
