#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'USAGE'
Run vLLM evaluation on local model and input paths.

Environment:
  TEMPERATURE (default: 0.6), TOP_P (default: 0.95), TOP_K (default: 20),
  N_RESPONSES (default: 1), MAX_TOKENS (default: 2048),
  MAX_MODEL_LEN (default: 4096), and TRUST_REMOTE_CODE (default: true).

The command is printed without execution unless --run is supplied.
USAGE
    local_common_usage
}

local_init
local_parse_common "$@"
local_scope_output eval
if ((LOCAL_HELP)); then
    usage
    exit 0
fi

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
N_RESPONSES="${N_RESPONSES:-1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

local_export_runtime
cd "$LOCAL_REPO_ROOT"
if [[ "$LOCAL_RUN" == 1 ]]; then
    local_validate_common 0 1 1
    local_prepare_output
fi

cmd=(
    "$LOCAL_PYTHON_BIN" -m evals.rollout_engine.vllm_rollout
    --model "$LOCAL_MODEL_PATH"
    --input "$LOCAL_VAL_FILE"
    --output-dir "$LOCAL_OUTPUT_DIR"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --n "$N_RESPONSES"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --data-parallel-size "$LOCAL_GPUS"
)
if [[ "$TRUST_REMOTE_CODE" == 1 || "$TRUST_REMOTE_CODE" == true || "$TRUST_REMOTE_CODE" == True ]]; then
    cmd+=(--trust-remote-code)
fi
if ((${#LOCAL_EXTRA_OVERRIDES[@]})); then
    cmd+=("${LOCAL_EXTRA_OVERRIDES[@]}")
fi
if ((${#LOCAL_REMAINING[@]})); then
    cmd+=("${LOCAL_REMAINING[@]}")
fi

local_run_command "${cmd[@]}"
