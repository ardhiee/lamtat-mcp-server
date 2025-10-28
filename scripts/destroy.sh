#!/usr/bin/env bash
set -euo pipefail

ROOT_NAME=$(basename "$(pwd)")
APP_NAME="${ROOT_NAME}-app"
SERVICE_NAME="${ROOT_NAME}"
ENV_NAME="test"

if copilot svc show --name "${SERVICE_NAME}" --env "${ENV_NAME}" >/dev/null 2>&1; then
  copilot svc delete --name "${SERVICE_NAME}" --env "${ENV_NAME}" --yes
fi

if copilot env show --name "${ENV_NAME}" >/dev/null 2>&1; then
  copilot env delete --name "${ENV_NAME}" --yes
fi

if copilot app show --name "${APP_NAME}" >/dev/null 2>&1; then
  copilot app delete --yes
fi
