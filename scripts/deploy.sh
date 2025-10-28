#!/usr/bin/env bash
set -euo pipefail

ROOT_NAME=$(basename "$(pwd)")
APP_NAME="${ROOT_NAME}-app"
SERVICE_NAME="${ROOT_NAME}"
ENV_NAME="test"
PROFILE="default"
REGION="us-east-1"

if ! copilot app show --name "${APP_NAME}" >/dev/null 2>&1; then
  copilot app init --name "${APP_NAME}" --resource-tags "project=${ROOT_NAME}"
fi

if ! copilot env show --name "${ENV_NAME}" >/dev/null 2>&1; then
  copilot env init --app "${APP_NAME}" --name "${ENV_NAME}" --profile "${PROFILE}" --region "${REGION}"
fi

copilot env deploy --app "${APP_NAME}" --name "${ENV_NAME}"
copilot deploy --name "${SERVICE_NAME}" --env "${ENV_NAME}"
