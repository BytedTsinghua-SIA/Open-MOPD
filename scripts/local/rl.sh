#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'USAGE'
Run GRPO/RL training on local paths.

Environment:
  TRAIN_BATCH_SIZE, MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, N_RESPONSES,
  TOTAL_EPOCHS, ADV_ESTIMATOR, REWARD_FUNCTION_PATH, REWARD_FUNCTION_NAME.
  Defaults are 1, 1024, 2048, 1, 1, grpo, and no custom reward function.

The command is printed without execution unless --run is supplied.
USAGE
    local_common_usage
}

local_init
local_parse_common "$@"
local_scope_output rl
if ((LOCAL_HELP)); then
    usage
    exit 0
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
N_RESPONSES="${N_RESPONSES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"

local_export_runtime
cd "$LOCAL_TRAINING_DIR"
if [[ "$LOCAL_RUN" == 1 ]]; then
    local_validate_common 1 1 1
    local_prepare_output
fi

cmd=(
    "$LOCAL_PYTHON_BIN" -m verl.trainer.main_ppo
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "data.train_files=${LOCAL_TRAIN_FILE}"
    "data.val_files=${LOCAL_VAL_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "actor_rollout_ref.model.path=${LOCAL_MODEL_PATH}"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=${N_RESPONSES}"
    "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    "reward_model.enable=False"
    "trainer.n_gpus_per_node=${LOCAL_GPUS}"
    "trainer.nnodes=${LOCAL_NODES}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.default_local_dir=${LOCAL_CHECKPOINT_DIR}"
    "trainer.project_name=${PROJECT_NAME:-OpenOPD-local}"
    "trainer.experiment_name=${EXPERIMENT_NAME:-rl-local}"
    "trainer.logger=['console']"
)

if [[ -n "${REWARD_FUNCTION_PATH:-}" ]]; then
    cmd+=(
        "custom_reward_function.path=${REWARD_FUNCTION_PATH}"
        "custom_reward_function.name=${REWARD_FUNCTION_NAME:-compute_score}"
    )
fi
if ((${#LOCAL_EXTRA_OVERRIDES[@]})); then
    cmd+=("${LOCAL_EXTRA_OVERRIDES[@]}")
fi
if ((${#LOCAL_REMAINING[@]})); then
    cmd+=("${LOCAL_REMAINING[@]}")
fi

local_run_command "${cmd[@]}"
