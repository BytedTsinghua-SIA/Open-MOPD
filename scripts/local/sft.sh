#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'USAGE'
Run supervised fine-tuning on local paths.

Environment:
  TRAIN_BATCH_SIZE (default: 1), MAX_LENGTH (default: 4096), TOTAL_EPOCHS
  (default: 1), RESUME_MODE (default: auto), and MODEL_PATH.

The command is printed without execution unless --run is supplied.
USAGE
    local_common_usage
}

local_init
local_parse_common "$@"
local_scope_output sft
if ((LOCAL_HELP)); then
    usage
    exit 0
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
RESUME_MODE="${RESUME_MODE:-auto}"
NUM_TRAINERS="${NUM_TRAINERS:-${LOCAL_GPUS}}"

local_export_runtime
cd "$LOCAL_TRAINING_DIR"
if [[ "$LOCAL_RUN" == 1 ]]; then
    local_validate_common 1 0 1
    command -v "$LOCAL_TORCHRUN_BIN" >/dev/null 2>&1 || local_die "torchrun executable not found: $LOCAL_TORCHRUN_BIN"
    local_prepare_output
fi

cmd=(
    "$LOCAL_TORCHRUN_BIN"
    --standalone
    "--nnodes=${LOCAL_NODES}"
    "--nproc-per-node=${NUM_TRAINERS}"
    -m verl.trainer.sft_trainer
    "data.train_files=${LOCAL_TRAIN_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_length=${MAX_LENGTH}"
    "data.truncation=error"
    "data.use_dynamic_bsz=True"
    "model.path=${LOCAL_MODEL_PATH}"
    "trainer.project_name=${PROJECT_NAME:-OpenOPD-local}"
    "trainer.experiment_name=${EXPERIMENT_NAME:-sft-local}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.default_local_dir=${LOCAL_CHECKPOINT_DIR}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.logger=['console']"
)
if [[ "$LOCAL_VAL_EXPLICIT" == 1 ]]; then
    cmd+=("data.val_files=${LOCAL_VAL_FILE}")
fi
if ((${#LOCAL_EXTRA_OVERRIDES[@]})); then
    cmd+=("${LOCAL_EXTRA_OVERRIDES[@]}")
fi
if ((${#LOCAL_REMAINING[@]})); then
    cmd+=("${LOCAL_REMAINING[@]}")
fi

local_run_command "${cmd[@]}"
