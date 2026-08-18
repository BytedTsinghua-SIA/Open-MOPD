#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
PYTHONUSERBASE="${PYTHONUSERBASE:-${HOME}/.local}"
export PYTHONUSERBASE
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
PIP_BIN=("${PYTHON_BIN}" -m pip)

"${PIP_BIN[@]}" install --user swanlab math-verify matplotlib duckdb

pushd verl
"${PIP_BIN[@]}" install --user -e .
popd

"${PIP_BIN[@]}" install --user ray==2.55.1
"${PIP_BIN[@]}" install --user transformers==4.57.6
"${PIP_BIN[@]}" install --user protobuf==3.20.3

if [ -f requirements-if-rl.txt ]; then
  "${PIP_BIN[@]}" install --user -r requirements-if-rl.txt
fi

VERIFIABLE_INSTRUCTIONS_PATH="${VERIFIABLE_INSTRUCTIONS_PATH:-}"
if [ -z "${VERIFIABLE_INSTRUCTIONS_PATH}" ] && [ -d third_party/verifiable-instructions ]; then
  VERIFIABLE_INSTRUCTIONS_PATH="third_party/verifiable-instructions"
fi
if [ -n "${VERIFIABLE_INSTRUCTIONS_PATH}" ] && [ -d "${VERIFIABLE_INSTRUCTIONS_PATH}" ]; then
  "${PIP_BIN[@]}" install --user -e "${VERIFIABLE_INSTRUCTIONS_PATH}"
else
  echo "[install_requirements] verifiable-instructions not installed; set VERIFIABLE_INSTRUCTIONS_PATH or add third_party/verifiable-instructions" >&2
fi

"${PYTHON_BIN}" - <<'PY' || true
import nltk
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
PY
