#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
TASK_CONTEXT="${DEEPSEEK_MAX_SEQ_LEN:-1048576}"
exec "$TASK_ROOT/env210/bin/python" -B -u -m vllm.entrypoints.openai.api_server \
  --model "${KTV4_MODEL_PATH:-/mnt/nvme1n1/bf16/w8a8-mtp}" \
  --served-model-name deepseek-v4-flash --tensor-parallel-size 4 \
  --distributed-executor-backend executor.CPUOffloadExecutor \
  --quantization ascend --max-model-len "$TASK_CONTEXT" --max-num-batched-tokens "$TASK_CONTEXT" \
  --max-num-seqs 1 --no-enable-chunked-prefill --no-enable-prefix-caching \
  --no-async-scheduling --enforce-eager --trust-remote-code --port 8000 --host 0.0.0.0
