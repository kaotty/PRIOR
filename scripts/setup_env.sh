#!/usr/bin/env bash
# Create the pinned x86_64 environment and keep all installation caches local.
set -euo pipefail

[[ $# == 0 || $# == 2 || $# == 5 ]] || {
  echo "Usage: bash scripts/setup_env.sh [ENV_PREFIX CACHE_ROOT [BASE_MODEL DATA_ROOT OUTPUT_ROOT]]" >&2
  exit 2
}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ $# == 0 ]]; then
  WORK_ROOT="${TIR_WORK_ROOT:-$REPO_ROOT/.tir}"
  [[ "$WORK_ROOT" == /* ]] || {
    echo "TIR_WORK_ROOT must be absolute" >&2; exit 2
  }
  ENV_PREFIX="$WORK_ROOT/env"
  CACHE_ROOT="$WORK_ROOT/cache"
  MODEL_PATH="$WORK_ROOT/models/Qwen2.5-VL-7B-Instruct"
  DATA_ROOT="$WORK_ROOT/data"
  OUTPUT_ROOT="$WORK_ROOT/outputs"
  # Re-running the documented command keeps an explicitly selected layout.
  if [[ -z "${TIR_WORK_ROOT:-}" && -f "$REPO_ROOT/.tir.paths.env" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.tir.paths.env"
    ENV_PREFIX="$TIR_ENV_PREFIX"
    CACHE_ROOT="$TIR_CACHE_ROOT"
    MODEL_PATH="$TIR_BASE_MODEL_PATH"
    DATA_ROOT="$TIR_DATA_ROOT"
    OUTPUT_ROOT="$TIR_OUTPUT_ROOT"
  fi
else
  ENV_PREFIX="$1"
  CACHE_ROOT="$2"
  if [[ $# == 5 ]]; then
    MODEL_PATH="$3"
    DATA_ROOT="$4"
    OUTPUT_ROOT="$5"
  fi
fi

[[ "$ENV_PREFIX" == /* && "$CACHE_ROOT" == /* ]] || {
  echo "Environment and cache paths must be absolute" >&2
  exit 2
}
if [[ $# == 0 || $# == 5 ]]; then
  for path in "$MODEL_PATH" "$DATA_ROOT" "$OUTPUT_ROOT"; do
    [[ "$path" == /* ]] || { echo "Model, data and output paths must be absolute" >&2; exit 2; }
  done
fi
[[ "$(uname -m)" == x86_64 ]] || {
  echo "The pinned FlashAttention wheel requires Linux x86_64" >&2
  exit 2
}
[[ "$(uname -s)" == Linux ]] || { echo "Linux is required" >&2; exit 2; }
command -v conda >/dev/null || { echo "Conda must be available on PATH" >&2; exit 2; }

mkdir -p "$CACHE_ROOT"/{conda_pkgs,pip,huggingface,xdg,tmp,torch,triton}
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda_pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset TRANSFORMERS_CACHE
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
source "$REPO_ROOT/scripts/runtime_env.sh"

if [[ ! -e "$ENV_PREFIX" ]]; then
  conda create -p "$ENV_PREFIX" python=3.12 -y
fi
[[ -x "$ENV_PREFIX/bin/python" ]] || {
  echo "Incomplete Conda environment: $ENV_PREFIX" >&2
  exit 2
}
"$ENV_PREFIX/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "The target environment must use Python 3.12"'

conda run --no-capture-output -p "$ENV_PREFIX" bash "$REPO_ROOT/scripts/install_env.sh"

TIR_EXPECTED_ENV_PREFIX="$ENV_PREFIX" "$ENV_PREFIX/bin/python" - <<'PY'
import importlib
import importlib.metadata as metadata
import os
import subprocess
import sys

assert sys.version_info[:2] == (3, 12), sys.version
expected = os.path.join(os.environ["TIR_EXPECTED_ENV_PREFIX"], "bin/python")
assert os.path.samefile(sys.executable, expected), (sys.executable, expected)
assert os.path.samefile(sys.prefix, os.environ["TIR_EXPECTED_ENV_PREFIX"])
for module, package in (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("ray", "ray"),
    ("sglang", "sglang"),
    ("flash_attn", "flash-attn"),
    ("flashinfer", "flashinfer-python"),
    ("verl", "verl"),
):
    importlib.import_module(module)
    print(f"{module}=={metadata.version(package)}")
check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
known = "torch 2.9.1 has requirement nvidia-cudnn-cu12==9.10.2.21"
problems = [line for line in check.stdout.splitlines() if line != "No broken requirements found."]
unexpected = [line for line in problems if not (line.startswith(known) and "9.16.0.29" in line)]
if unexpected or check.returncode not in (0, 1):
    raise RuntimeError("Dependency check failed: " + "\n".join(unexpected) + check.stderr)
if problems:
    print("Dependency check: the recorded cuDNN 9.16 override differs from torch's package metadata.")
print("CPU import check PASS:", sys.executable)
PY
"$ENV_PREFIX/bin/python" -m pip freeze > "$ENV_PREFIX/pip-freeze.txt"

# No-argument setup uses a repository-relative local runtime layout. Explicit
# paths remain supported for shared clusters with a separate scratch volume.
if [[ $# == 0 || $# == 5 ]]; then
  mkdir -p "$DATA_ROOT/inputs/sft" "$DATA_ROOT/inputs/rl" "$DATA_ROOT/eval_inputs" \
    "$OUTPUT_ROOT" "$(dirname "$MODEL_PATH")"
  "$ENV_PREFIX/bin/python" "$REPO_ROOT/scripts/download_model.py" \
    --output "$MODEL_PATH" --cache-dir "$HF_HUB_CACHE"
  config="$REPO_ROOT/.tir.paths.env"
  {
    printf 'TIR_ENV_PREFIX=%q\n' "$ENV_PREFIX"
    printf 'TIR_CACHE_ROOT=%q\n' "$CACHE_ROOT"
    printf 'TIR_BASE_MODEL_PATH=%q\n' "$MODEL_PATH"
    printf 'TIR_DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'TIR_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TIR_N_GPUS=2\n'
  } > "$config"
  chmod 600 "$config"
  echo "Local paths saved: $config"
  echo "Input root: $DATA_ROOT/inputs; base model: $MODEL_PATH"
fi
