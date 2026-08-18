#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/runs/local/sft_mix}"
LOG_PATH="${2:-${OUT_DIR}.log}"
MATH_SOURCE="${MATH_SOURCE:?set MATH_SOURCE to a local parquet path}"
CODE_SOURCE="${CODE_SOURCE:?set CODE_SOURCE to a local directory or parquet path}"
IF_SOURCE="${IF_SOURCE:?set IF_SOURCE to a local parquet path}"

mkdir -p "$(dirname "${LOG_PATH}")"
rm -rf "${OUT_DIR}"

python "${ROOT_DIR}/training/scripts/sft/build_weighted_mix.py" \
  --source math_oda math "${MATH_SOURCE}" \
  --source code_ocr code "${CODE_SOURCE}" \
  --source instruction_nemotron if "${IF_SOURCE}" \
  --weight math_oda 2 \
  --weight code_ocr 2 \
  --weight instruction_nemotron 1 \
  --target math_oda 459646 \
  --target code_ocr 459646 \
  --target instruction_nemotron 229823 \
  --output-dir "${OUT_DIR}" \
  --seed 20260512 \
  --shard-rows 50000 \
  --overwrite \
  >> "${LOG_PATH}" 2>&1
