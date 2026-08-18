#!/usr/bin/env bash
# Shared helpers for the explicit, single-machine launchers in this directory.
#
# The launchers are intentionally dry-run by default. Pass --run (or set
# LOCAL_RUN=1) after checking the printed command and local paths.

set -euo pipefail

LOCAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${LOCAL_SCRIPT_DIR}/../.." && pwd)"

local_init() {
    LOCAL_TRAINING_DIR="${TRAINING_DIR:-${LOCAL_REPO_ROOT}/training}"
    LOCAL_VERL_DIR="${VERL_DIR:-${LOCAL_TRAINING_DIR}/verl}"
    LOCAL_PYTHON_BIN="${PYTHON_BIN:-python3}"
    LOCAL_TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
    LOCAL_RUN="${LOCAL_RUN:-0}"
    LOCAL_MODEL_PATH="${MODEL_PATH:-${LOCAL_REPO_ROOT}/model}"
    LOCAL_TRAIN_FILE="${TRAIN_FILE:-${LOCAL_REPO_ROOT}/data/train.parquet}"
    LOCAL_VAL_FILE="${VAL_FILE:-${LOCAL_REPO_ROOT}/data/val.parquet}"
    LOCAL_OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_REPO_ROOT}/runs/local}"
    LOCAL_CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOCAL_OUTPUT_DIR}/checkpoints}"
    LOCAL_OUTPUT_EXPLICIT=0
    LOCAL_CHECKPOINT_EXPLICIT=0
    LOCAL_VAL_EXPLICIT=0
    [[ -z "${OUTPUT_DIR:-}" ]] || LOCAL_OUTPUT_EXPLICIT=1
    [[ -z "${CHECKPOINT_DIR:-}" ]] || LOCAL_CHECKPOINT_EXPLICIT=1
    [[ -z "${VAL_FILE:-}" ]] || LOCAL_VAL_EXPLICIT=1
    LOCAL_GPUS="${GPUS:-1}"
    LOCAL_NODES="${NODES:-1}"
    LOCAL_DOMAINS="${TEACHER_DOMAINS:-math,code,if}"
    LOCAL_HELP=0
    LOCAL_EXTRA_OVERRIDES=()
    LOCAL_TEACHER_PATHS=()
    LOCAL_REMAINING=()
}

local_die() {
    echo "[local] error: $*" >&2
    return 2
}

local_common_usage() {
    cat <<'USAGE'
Common options:
  --run, --exec          execute the printed command (default is dry-run)
  --dry-run              print the command without executing it
  --model PATH           local model directory (MODEL_PATH)
  --train PATH           local training parquet (TRAIN_FILE)
  --val PATH             local validation/input parquet (VAL_FILE)
  --output PATH          local output directory (OUTPUT_DIR)
  --checkpoint PATH      local checkpoint directory (CHECKPOINT_DIR)
  --gpus N               processes/GPUs per node (GPUS)
  --nodes N              node count; must be 1 (NODES, defaults to 1)
  --python PATH          Python executable (PYTHON_BIN)
  --torchrun PATH        torchrun executable (TORCHRUN_BIN)
  --teacher PATH         append a teacher model path (TEACHER_PATH or
                         TEACHER_MODEL_PATHS can provide it via environment)
  --domains LIST         comma-separated teacher domains
  --extra VALUE          append a Hydra override
  --                    pass all remaining arguments to the underlying command
USAGE
}

local_parse_common() {
    while (($#)); do
        case "$1" in
            -h|--help)
                LOCAL_HELP=1
                shift
                ;;
            --run|--exec)
                LOCAL_RUN=1
                shift
                ;;
            --dry-run)
                LOCAL_RUN=0
                shift
                ;;
            --model|--model-path)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_MODEL_PATH="$2"
                shift 2
                ;;
            --train|--train-data)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_TRAIN_FILE="$2"
                shift 2
                ;;
            --val|--val-data|--input)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_VAL_FILE="$2"
                LOCAL_VAL_EXPLICIT=1
                shift 2
                ;;
            --output|--output-dir)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_OUTPUT_DIR="$2"
                LOCAL_OUTPUT_EXPLICIT=1
                if [[ "$LOCAL_CHECKPOINT_EXPLICIT" == 0 ]]; then
                    LOCAL_CHECKPOINT_DIR="${LOCAL_OUTPUT_DIR}/checkpoints"
                fi
                shift 2
                ;;
            --checkpoint|--checkpoint-dir)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_CHECKPOINT_DIR="$2"
                LOCAL_CHECKPOINT_EXPLICIT=1
                shift 2
                ;;
            --gpus)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_GPUS="$2"
                shift 2
                ;;
            --nodes)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_NODES="$2"
                shift 2
                ;;
            --python)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_PYTHON_BIN="$2"
                shift 2
                ;;
            --torchrun)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_TORCHRUN_BIN="$2"
                shift 2
                ;;
            --teacher|--teacher-path)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_TEACHER_PATHS+=("$2")
                shift 2
                ;;
            --domains)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_DOMAINS="$2"
                shift 2
                ;;
            --extra)
                (($# >= 2)) || local_die "$1 requires a value"
                LOCAL_EXTRA_OVERRIDES+=("$2")
                shift 2
                ;;
            --)
                shift
                LOCAL_REMAINING+=("$@")
                break
                ;;
            *)
                LOCAL_REMAINING+=("$1")
                shift
                ;;
        esac
    done
}

local_scope_output() {
    local kind="$1"
    if [[ "$LOCAL_OUTPUT_EXPLICIT" == 0 ]]; then
        LOCAL_OUTPUT_DIR="${LOCAL_REPO_ROOT}/runs/local/${kind}"
        if [[ "$LOCAL_CHECKPOINT_EXPLICIT" == 0 ]]; then
            LOCAL_CHECKPOINT_DIR="${LOCAL_OUTPUT_DIR}/checkpoints"
        fi
    fi
}

local_export_runtime() {
    [[ -d "$LOCAL_TRAINING_DIR" ]] || local_die "training directory not found: $LOCAL_TRAINING_DIR"
    if [[ ":${PYTHONPATH:-}:" != *":${LOCAL_VERL_DIR}:"* ]]; then
        export PYTHONPATH="${LOCAL_VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}

local_validate_number() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || local_die "$name must be a positive integer, got: $value"
}

local_require_path() {
    local label="$1"
    local path="$2"
    [[ -e "$path" ]] || local_die "$label does not exist: $path"
}

local_require_local_path() {
    local label="$1"
    local path="$2"
    [[ "$path" != *"://"* ]] || local_die "$label must be a local filesystem path: $path"
}

local_validate_common() {
    local require_train="${1:-1}"
    local require_val="${2:-1}"
    local require_model="${3:-1}"
    local_validate_number GPUS "$LOCAL_GPUS"
    local_validate_number NODES "$LOCAL_NODES"
    [[ "$LOCAL_NODES" == 1 ]] || local_die "NODES must be 1 for the single-machine launcher"
    command -v "$LOCAL_PYTHON_BIN" >/dev/null 2>&1 || local_die "Python executable not found: $LOCAL_PYTHON_BIN"
    local_require_local_path output "$LOCAL_OUTPUT_DIR"
    local_require_local_path checkpoint "$LOCAL_CHECKPOINT_DIR"
    if [[ "$require_model" == 1 ]]; then
        local_require_local_path model "$LOCAL_MODEL_PATH"
        local_require_path model "$LOCAL_MODEL_PATH"
    fi
    if [[ "$require_train" == 1 ]]; then
        local_require_local_path training-data "$LOCAL_TRAIN_FILE"
        local_require_path training-data "$LOCAL_TRAIN_FILE"
    fi
    if [[ "$require_val" == 1 ]]; then
        local_require_local_path validation-data "$LOCAL_VAL_FILE"
        local_require_path validation-data "$LOCAL_VAL_FILE"
    fi
}

local_print_command() {
    printf '[local]'
    printf ' %q' "$@"
    printf '\n'
}

local_run_command() {
    local_print_command "$@"
    if [[ "$LOCAL_RUN" != 1 ]]; then
        echo "[local] dry-run only; pass --run to execute"
        return 0
    fi
    "$@"
}

local_prepare_output() {
    mkdir -p "$LOCAL_OUTPUT_DIR" "$LOCAL_CHECKPOINT_DIR"
}
