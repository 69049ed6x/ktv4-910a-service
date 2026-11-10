#!/usr/bin/env bash
# The installed Python environment and system CANN/driver are kept in place.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$TASK_ROOT/deployment.env" ]]; then
  source "$TASK_ROOT/deployment.env"
fi
mkdir -p "$TASK_ROOT/run/tmp" "$TASK_ROOT/cache" "$TASK_ROOT/run/ascend-log"
export TMPDIR="$TASK_ROOT/run/tmp"
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$TASK_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$TASK_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$TASK_ROOT/cache/vllm"
export VLLM_CONFIG_ROOT="$TASK_ROOT/run/vllm-config"
export NUMBA_CACHE_DIR="$TASK_ROOT/cache/numba"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=false
export PATH="$TASK_ROOT/env210/bin:$PATH"
source "${KTV4_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
source "${KTV4_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}" --cxx_abi=1
export PYTHONPATH="$TASK_ROOT/app:$TASK_ROOT/vendor/vllm${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_PROCESS_LOG_PATH="$TASK_ROOT/run/ascend-log"
export ASCEND_CACHE_PATH="$TASK_ROOT/cache"
export ASCEND_WORK_PATH="$TASK_ROOT/cache"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=4
export ASCEND_EVENT_ENABLE=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=64
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=3600
export HCCL_NPU_SOCKET_PORT_RANGE=31000-31063
export HCCL_HOST_SOCKET_PORT_RANGE=31100-31163
export VLLM_VERSION=0.23.0
export VLLM_HOST_IP=127.0.0.1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_RPC_TIMEOUT=3600000
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset CANN_KNOWLEDGE_BANK_PROCESS_NUM
cd "$TASK_ROOT/run"
