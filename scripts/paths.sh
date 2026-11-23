#!/usr/bin/env bash
# Load paths written by setup_env.sh. Explicit environment values win.

CALLER_DIR="${CALLER_DIR:-$PWD}"
CONFIG="${TIR_CONFIG:-$REPO_ROOT/.tir.paths.env}"
if [[ ! -f "$CONFIG" && -z "${TIR_CONFIG:-}" ]]; then
  CONFIG="$REPO_ROOT/tir.local.env"
fi

if [[ -f "$CONFIG" ]]; then
  declare -A path_overrides=()
  for name in TIR_ENV_PREFIX TIR_CACHE_ROOT TIR_BASE_MODEL_PATH TIR_DATA_ROOT \
              TIR_OUTPUT_ROOT TIR_N_GPUS TIR_SEED TIR_PYTHON_BIN; do
    if [[ -v "$name" ]]; then path_overrides["$name"]="${!name}"; fi
  done
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG"
  set +a
  for name in "${!path_overrides[@]}"; do
    printf -v "$name" '%s' "${path_overrides[$name]}"
    export "$name"
  done
  unset path_overrides
elif [[ -n "${TIR_CONFIG:-}" ]]; then
  echo "Missing TIR_CONFIG file: $CONFIG" >&2
  exit 2
fi

if [[ -n "${TIR_ENV_PREFIX:-}" ]]; then
  export TIR_PYTHON_BIN="${TIR_PYTHON_BIN:-$TIR_ENV_PREFIX/bin/python}"
  export PATH="$TIR_ENV_PREFIX/bin:$PATH"
fi
if [[ -n "${TIR_CACHE_ROOT:-}" ]]; then
  export HF_HOME="$TIR_CACHE_ROOT/huggingface"
fi

absolute_path() {
  if [[ "$1" == /* ]]; then realpath -m "$1"; else realpath -m "$CALLER_DIR/$1"; fi
}

cd "$REPO_ROOT"
