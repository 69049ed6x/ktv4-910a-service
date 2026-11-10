#!/usr/bin/env bash
set -e
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$TASK_ROOT/deployment.env" ]]; then
  source "$TASK_ROOT/deployment.env"
fi
"$TASK_ROOT/env210/bin/python" -B -u "$TASK_ROOT/app/service.py" up
if [[ "${1:-}" == "--no-chat" ]]; then exit 0; fi
exec bash "$TASK_ROOT/chat.sh" "$@"
