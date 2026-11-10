#!/usr/bin/env bash
# Install this source delivery on a compatible Linux aarch64 Ascend host.
set -eo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON=python3.11
TASK_MODEL="${KTV4_MODEL_PATH:-/mnt/nvme1n1/bf16/w8a8-mtp}"
TASK_CANN="${KTV4_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
TASK_ATB="${KTV4_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
TASK_SOC="${SOC_VERSION:-ascend910b1}"
TASK_PLAN=0
TASK_SKIP_DEPS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) TASK_MODEL="${2:?Missing model path}"; shift 2 ;;
    --python) TASK_PYTHON="${2:?Missing Python path}"; shift 2 ;;
    --cann-env) TASK_CANN="${2:?Missing CANN environment script}"; shift 2 ;;
    --atb-env) TASK_ATB="${2:?Missing ATB environment script}"; shift 2 ;;
    --soc-version) TASK_SOC="${2:?Missing SoC version}"; shift 2 ;;
    --plan) TASK_PLAN=1; shift ;;
    --skip-deps) TASK_SKIP_DEPS=1; shift ;;
    -h|--help)
      echo 'Usage: bash setup.sh --model-path /data/w8a8-mtp [--python python3.11] [--cann-env FILE] [--atb-env FILE] [--soc-version ascend910b1] [--skip-deps] [--plan]'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || { echo 'Requires Linux aarch64.' >&2; exit 1; }
command -v "$TASK_PYTHON" >/dev/null || { echo 'Install Python 3.11 with venv and development headers first.' >&2; exit 1; }
"$TASK_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"'
for TASK_FILE in "$TASK_CANN" "$TASK_ATB" "$TASK_MODEL/config.json"; do
  [[ -f "$TASK_FILE" ]] || { echo "Required file missing: $TASK_FILE" >&2; exit 1; }
done
for TASK_TOOL in gcc g++ make git patch npu-smi; do
  command -v "$TASK_TOOL" >/dev/null || { echo "Required command missing: $TASK_TOOL" >&2; exit 1; }
done
TASK_ENV="$TASK_ROOT/env210"
printf 'Project: %s\nWeights: %s\nCANN: %s\nATB: %s\nBuild SoC: %s\n' "$TASK_ROOT" "$TASK_MODEL" "$TASK_CANN" "$TASK_ATB" "$TASK_SOC"
if [[ "$TASK_PLAN" == 1 ]]; then
  echo 'Plan only: create env210; install pinned build/runtime packages; build patched vLLM (empty backend) and Ascend native/custom operators; save deployment.env; run preflight.'
  echo 'No weights, drivers or CANN/ATB installers will be downloaded by setup.sh.'
  echo 'No environment, configuration or service was changed.'
  exit 0
fi
if [[ -f "$TASK_ENV/pyvenv.cfg" ]]; then
  : # Reuse only the project's existing venv.
elif [[ -e "$TASK_ENV" ]]; then
  echo "Refusing to overwrite an existing non-venv environment: $TASK_ENV" >&2
  exit 1
fi
mkdir -p "$TASK_ROOT/run/tmp" "$TASK_ROOT/cache/pip" "$TASK_ROOT/cache/build"
export TMPDIR="$TASK_ROOT/run/tmp" PIP_CACHE_DIR="$TASK_ROOT/cache/pip"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
[[ -f "$TASK_ENV/pyvenv.cfg" ]] || "$TASK_PYTHON" -m venv "$TASK_ENV"
TASK_BIN="$TASK_ENV/bin/python"
export PATH="$TASK_ENV/bin:$PATH"
source "$TASK_CANN"
source "$TASK_ATB" --cxx_abi=1
export MAX_JOBS="${MAX_JOBS:-16}"
export GIT_CONFIG_GLOBAL="$TASK_ROOT/cache/build/gitconfig"
export SOC_VERSION="$TASK_SOC"
export VLLM_VERSION=0.23.0
export VLLM_VERSION_OVERRIDE=0.23.1.dev0+g0fc695fc6.d20261109.empty
if [[ "$TASK_SKIP_DEPS" == 0 ]]; then
  "$TASK_BIN" -m pip install --no-deps -r "$TASK_ROOT/deploy/requirements.build.lock"
  "$TASK_BIN" -m pip install --no-deps --index-url "${KTV4_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
    torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu
  "$TASK_BIN" -m pip install --no-deps -r "$TASK_ROOT/deploy/requirements.runtime.lock"
fi
# Build from this delivery. Never resolve to an unpatched PyPI vLLM package.
VLLM_TARGET_DEVICE=empty "$TASK_BIN" -m pip install --no-build-isolation --no-deps -e "$TASK_ROOT/vendor/vllm"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0.post2.dev0+g1cdb8c4db.d20261109
export COMPILE_CUSTOM_KERNELS=1
"$TASK_BIN" -m pip install --no-build-isolation --no-deps "$TASK_ROOT/vendor/vllm-ascend"
unset SETUPTOOLS_SCM_PRETEND_VERSION VLLM_VERSION_OVERRIDE
"$TASK_BIN" -B "$TASK_ROOT/deploy/configure.py" --model-path "$TASK_MODEL" --cann-env "$TASK_CANN" --atb-env "$TASK_ATB"
export KTV4_MODEL_PATH="$TASK_MODEL" KTV4_CANN_ENV="$TASK_CANN" KTV4_ATB_ENV="$TASK_ATB"
source "$TASK_ROOT/app/env.sh"
"$TASK_BIN" -B "$TASK_ROOT/deploy/preflight.py" --model-path "$TASK_MODEL" --report "$TASK_ROOT/config/target_preflight.json"
echo "Installation and preflight completed. Start chat with: bash '$TASK_ROOT/start.sh'"
