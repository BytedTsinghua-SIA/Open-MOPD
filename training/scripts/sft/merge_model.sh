#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash training/scripts/sft/merge_model.sh CHECKPOINT_DIR [OUTPUT_DIR]

Merge a local VERL FSDP checkpoint into a local Hugging Face directory.
CHECKPOINT_DIR must contain fsdp_config.json and the checkpoint shards. If the
output is omitted, it is written next to the checkpoint under merged_hf/.

Environment:
  PYTHON_BIN          Python executable (default: python3)
  TRUST_REMOTE_CODE   Pass --trust-remote-code when set to 1/true (default: 1)
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

input_dir="${1:-}"
case "$input_dir" in
  ""|-h|--help)
    usage
    exit 0
    ;;
esac

[[ "$input_dir" != *"://"* ]] || die "remote URI is not supported: $input_dir"
[[ -d "$input_dir" ]] || die "checkpoint directory does not exist: $input_dir"
[[ -f "$input_dir/fsdp_config.json" ]] || die "missing fsdp_config.json under $input_dir"

input_dir="${input_dir%/}"
checkpoint_name="$(basename "$input_dir")"
output_dir="${2:-$(dirname "$input_dir")/merged_hf/$checkpoint_name}"
[[ "$output_dir" != *"://"* ]] || die "remote URI is not supported: $output_dir"

python_bin="${PYTHON_BIN:-python3}"
command -v "$python_bin" >/dev/null 2>&1 || die "python executable not found: $python_bin"
mkdir -p "$output_dir"

args=(
  -m verl.model_merger merge
  --backend fsdp
  --local_dir "$input_dir"
  --target_dir "$output_dir"
)
if [[ "${TRUST_REMOTE_CODE:-1}" == 1 || "${TRUST_REMOTE_CODE:-1}" == true ]]; then
  args+=(--trust-remote-code)
fi

echo "[merge_model] input:  $input_dir"
echo "[merge_model] output: $output_dir"
"$python_bin" "${args[@]}"
[[ -f "$output_dir/config.json" ]] || die "merged output missing config.json"
echo "[merge_model] done"
