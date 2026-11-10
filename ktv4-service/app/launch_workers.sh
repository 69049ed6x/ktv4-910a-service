#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$TASK_ROOT/env210/bin/python" -B -u -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=31220 \
  "$TASK_ROOT/app/worker.py"
