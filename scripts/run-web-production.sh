#!/usr/bin/env bash
set -euo pipefail

IMAGE="${GEMMACLIP_WEB_IMAGE:-gemmaclip-web:latest}"
PORT="${GEMMACLIP_WEB_PORT:-8000}"

docker build -f Dockerfile.web -t "$IMAGE" .
docker volume create gemmaclip-runs >/dev/null

env_args=()
if [[ -f .env ]]; then
  env_args+=(--env-file .env)
fi

docker run --rm \
  --name gemmaclip-web \
  -p "${PORT}:8000" \
  -e GEMMACLIP_WEB_HOST=0.0.0.0 \
  -e GEMMACLIP_WEB_PORT=8000 \
  -e GEMMACLIP_WEB_RUNS_DIR=/data/runs \
  "${env_args[@]}" \
  -v gemmaclip-runs:/data/runs \
  "$IMAGE"
