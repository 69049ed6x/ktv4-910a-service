#!/usr/bin/env bash
set -e
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$TASK_ROOT/deployment.env" ]]; then
  source "$TASK_ROOT/deployment.env"
fi
exec "$TASK_ROOT/env210/bin/python" -B -u "$TASK_ROOT/app/service.py" status
