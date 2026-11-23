#!/usr/bin/env bash
# Shared compute-node runtime paths; sourced by setup and all three stages.
runtime_cache_root="${CACHE_ROOT:-${TIR_CACHE_ROOT:?Set the local cache root}}"
export HF_HOME="${HF_HOME:-$runtime_cache_root/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$runtime_cache_root/xdg}"
export TMPDIR="${TMPDIR:-$runtime_cache_root/tmp}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT:-${TIR_OUTPUT_ROOT:-$runtime_cache_root}}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$runtime_cache_root/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$runtime_cache_root/wandb/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$runtime_cache_root/wandb/data}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-${TIR_WANDB_PROJECT:-tir_reproduction}}"
unset TRANSFORMERS_CACHE
export TORCH_HOME="${TORCH_HOME:-$runtime_cache_root/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$runtime_cache_root/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$runtime_cache_root/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$runtime_cache_root/cuda}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$runtime_cache_root/flashinfer}"
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-$runtime_cache_root/deep_gemm}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$SGLANG_DG_CACHE_DIR}"

runtime_cuda_home="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "$runtime_cuda_home" ]]; then
  if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    runtime_cuda_home=/usr/local/cuda-12.8
  elif command -v nvcc >/dev/null 2>&1; then
    runtime_cuda_home="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  fi
fi
if [[ -n "$runtime_cuda_home" && -x "$runtime_cuda_home/bin/nvcc" &&
      -f "$runtime_cuda_home/include/cuda_runtime.h" ]]; then
  export CUDA_HOME="$runtime_cuda_home" CUDA_PATH="$runtime_cuda_home"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
elif [[ "${TIR_DRY_RUN:-0}" != 1 ]]; then
  echo "CUDA toolkit not found. This recipe requires CUDA 12.8 with nvcc and development headers." >&2
  return 2
fi

# Ray appends a long session/socket suffix. Use a short symlink while keeping
# the actual cache files on the reader-selected compute-node storage.
runtime_ray_root="${RAY_TMPDIR:-$runtime_cache_root/ray}"
if (( ${#runtime_ray_root} > 24 )); then
  runtime_ray_target="$(realpath -m "$runtime_ray_root")"
  runtime_ray_alias="/tmp/tir-$(id -u)-$(printf '%s' "$runtime_ray_target" | sha256sum | cut -c1-8)"
  if [[ "${TIR_DRY_RUN:-0}" == 0 ]]; then
    mkdir -p "$runtime_ray_target"
    if [[ -L "$runtime_ray_alias" ]]; then
      [[ "$(readlink "$runtime_ray_alias")" == "$runtime_ray_target" ]] || return 2
    elif [[ -e "$runtime_ray_alias" ]]; then
      echo "Ray cache alias conflicts: $runtime_ray_alias" >&2; return 2
    else
      ln -s "$runtime_ray_target" "$runtime_ray_alias" ||
        [[ "$(readlink "$runtime_ray_alias")" == "$runtime_ray_target" ]] || return 2
    fi
  fi
  runtime_ray_root="$runtime_ray_alias"
fi
export RAY_TMPDIR="$runtime_ray_root"
if [[ "${TIR_DRY_RUN:-0}" == 0 ]]; then
  mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" \
    "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
    "$FLASHINFER_WORKSPACE_BASE" "$SGLANG_DG_CACHE_DIR" "$RAY_TMPDIR"
fi
