#!/usr/bin/env bash
# Prepare one SFT arm or train it with the frozen Qwen2.5-VL recipe.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/paths.sh"

PYTHON_BIN="${TIR_PYTHON_BIN:-python3}"
MODEL_PATH="${TIR_BASE_MODEL_PATH:?Run bash scripts/setup_env.sh first}"
DATA_ROOT="${TIR_DATA_ROOT:-}"
OUTPUT_ROOT="${TIR_OUTPUT_ROOT:-}"
DRY_RUN="${TIR_DRY_RUN:-0}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$OUTPUT_ROOT/cache/huggingface}/hub}"

# Data preparation is implemented in Python, including pinned downloads.
if [[ "${1:-}" == prepare ]]; then
  [[ -n "$DATA_ROOT" ]] || { echo "Set TIR_DATA_ROOT" >&2; exit 2; }
  shift
  ARM=original
  if [[ $# -gt 0 && "$1" != --* ]]; then ARM="$1"; shift; fi
  INPUT_ROOT="$DATA_ROOT/inputs/sft"
  while [[ $# -gt 0 ]]; do
    [[ "$1" == --input-root && $# -ge 2 ]] || { echo "Usage: bash sft/run.sh prepare [arm] [--input-root DIR]" >&2; exit 2; }
    INPUT_ROOT="$(absolute_path "$2")"
    shift 2
  done
  DRY=()
  [[ "$DRY_RUN" == 1 ]] && DRY=(--dry-run)
  exec "$PYTHON_BIN" -m sft.prepare \
    --arm "$ARM" --input-root "$INPUT_ROOT" --cache-dir "$HF_HUB_CACHE" \
    --model "$MODEL_PATH" --sft-root "$DATA_ROOT/sft" "${DRY[@]}"
fi

# Each arm changes only the training parquet and its matched batch size.
if [[ $# -eq 1 || "${2:-}" == --* ]]; then
  RUN_NAME="${1:?Select an SFT arm}"
  [[ -n "$DATA_ROOT" ]] || { echo "Set TIR_DATA_ROOT" >&2; exit 2; }
  case "$RUN_NAME" in
    original-low) DATASET=frequency_low; ARM_BATCH=40 ;;
    original-medium|no-tool-25) DATASET=no_tool_25__frequency_medium; ARM_BATCH=40 ;;
    original-high) DATASET=frequency_high; ARM_BATCH=40 ;;
    no-tool-0) DATASET=no_tool_0; ARM_BATCH=40 ;;
    no-tool-50) DATASET=no_tool_50; ARM_BATCH=40 ;;
    mix-25-75) DATASET=mixture_25_75; ARM_BATCH=40 ;;
    mix-50-50) DATASET=mixture_50_50; ARM_BATCH=40 ;;
    mix-75-25) DATASET=mixture_75_25; ARM_BATCH=40 ;;
    filter25-top) DATASET=high_precall_token_control_v1/top16_e4/train_runtime_50step.parquet; ARM_BATCH=26 ;;
    filter50-top) DATASET=high_precall_token_control_v1/top36_e2/train_runtime_50step.parquet; ARM_BATCH=29 ;;
    filter25-random) DATASET=high_precall_random_token_v1/random25_e4/train_runtime_50step.parquet; ARM_BATCH=26 ;;
    filter50-random) DATASET=high_precall_random_token_v1/random50_e2/train_runtime_50step.parquet; ARM_BATCH=29 ;;
    *) echo "Unknown SFT arm: $RUN_NAME" >&2; exit 2 ;;
  esac
  if [[ "$DATASET" != *.parquet ]]; then DATASET="core_clean_prompt_v1/$DATASET/train.parquet"; fi
  TRAIN_FILE="$DATA_ROOT/sft/$DATASET"
  shift
  while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
    case "$1" in
      --batch-size) SFT_BATCH_SIZE="$2" ;;
      --steps) SFT_TOTAL_STEPS="$2" ;;
      *) echo "Unknown SFT option: $1" >&2; exit 2 ;;
    esac
    shift 2
  done
else
  TRAIN_FILE="${1:?Usage: bash sft/run.sh ARM or TRAIN_PARQUET RUN_NAME}"
  RUN_NAME="${2:?Give the legacy training file a run name}"
fi

# Edit these values to change the SFT experiment.
TRAIN_FILE="$(absolute_path "$TRAIN_FILE")"
MODEL_PATH="$(absolute_path "$MODEL_PATH")"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:?Set TIR_OUTPUT_ROOT}")"
NUM_GPUS="${TIR_N_GPUS:-2}"
SEED="${TIR_SEED:-42}"
TRAIN_BATCH_SIZE="${SFT_BATCH_SIZE:-${ARM_BATCH:-40}}"
TOTAL_STEPS="${SFT_TOTAL_STEPS:-50}"
MAX_LENGTH="${SFT_MAX_LENGTH:-73728}"
SP_SIZE="${SFT_SP_SIZE:-$NUM_GPUS}"
TOKENS_PER_GPU="${SFT_TOKENS_PER_GPU:-$(( (MAX_LENGTH + SP_SIZE - 1) / SP_SIZE ))}"
RUN_DIR="$(absolute_path "${SFT_OUTPUT_DIR:-$OUTPUT_ROOT/sft/$RUN_NAME/seed_$SEED}")"
CHAT_TEMPLATE="$REPO_ROOT/sft/chat_template.jinja"

[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ &&
   "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$MAX_LENGTH" =~ ^[1-9][0-9]*$ &&
   "$TOTAL_STEPS" =~ ^[1-9][0-9]*$ && "$SP_SIZE" =~ ^[1-9][0-9]*$ &&
   "$TOKENS_PER_GPU" =~ ^[1-9][0-9]*$ && "$DRY_RUN" =~ ^[01]$ ]] || {
  echo "Invalid SFT batch, step, GPU, sequence or dry-run setting" >&2; exit 2;
}
(( NUM_GPUS % SP_SIZE == 0 && TOKENS_PER_GPU * SP_SIZE >= MAX_LENGTH )) || {
  echo "Sequence-parallel capacity cannot hold MAX_LENGTH" >&2; exit 2;
}
if [[ "$DRY_RUN" == 0 ]]; then
  [[ -f "$TRAIN_FILE" && -f "$MODEL_PATH/config.json" && -f "$CHAT_TEMPLATE" ]] || {
    echo "Missing train parquet, model config, or chat template" >&2; exit 2;
  }
  [[ ! -e "$RUN_DIR" ]] || { echo "Output already exists: $RUN_DIR" >&2; exit 2; }
fi

CACHE_ROOT="$(absolute_path "${TIR_CACHE_ROOT:-$OUTPUT_ROOT/cache}")"
source "$REPO_ROOT/scripts/runtime_env.sh"
export PYTHONPATH="$REPO_ROOT/scripts/compat:$REPO_ROOT"
export TIR_MROPE_LAYOUT_COMPAT="${TIR_MROPE_LAYOUT_COMPAT:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TIR_SFT_TOOL_CHAT_TEMPLATE="$(<"$CHAT_TEMPLATE")"
unset RAY_ADDRESS RAY_RUNTIME_ENV_HOOK ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

run_command() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN command:'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

cd "$REPO_ROOT"
run_command "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nnodes=1 "--nproc-per-node=$NUM_GPUS" \
  -m verl.trainer.sft_trainer \
  "data.train_files=$TRAIN_FILE" \
  data.val_files=null \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu=1 \
  "data.max_length=$MAX_LENGTH" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.use_dynamic_bsz=true \
  "data.max_token_len_per_gpu=$TOKENS_PER_GPU" \
  "data.min_num_micro_batch=$TRAIN_BATCH_SIZE" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.num_workers=0 \
  data.ignore_input_ids_mismatch=false \
  data.require_tool_call_rendering=true \
  "model.path=$MODEL_PATH" \
  'model.custom_chat_template=${oc.env:TIR_SFT_TOOL_CHAT_TEMPLATE}' \
  model.use_remove_padding=true \
  engine=fsdp \
  optim=fsdp \
  optim.lr=1e-5 \
  optim.lr_warmup_steps_ratio=0.03 \
  optim.weight_decay=0.05 \
  'optim.betas=[0.9,0.95]' \
  optim.clip_grad=1.0 \
  optim.min_lr_ratio=0.1 \
  optim.warmup_style=cosine \
  "engine.ulysses_sequence_parallel_size=$SP_SIZE" \
  engine.strategy=fsdp2 \
  engine.fsdp_size=-1 \
  "engine.seed=$SEED" \
  "trainer.seed=$SEED" \
  "trainer.project_name=$WANDB_PROJECT" \
  "trainer.experiment_name=tir_sft_${RUN_NAME}_seed${SEED}" \
  "trainer.n_gpus_per_node=$NUM_GPUS" \
  trainer.total_epochs=1 \
  "trainer.total_training_steps=$TOTAL_STEPS" \
  "trainer.default_local_dir=$RUN_DIR" \
  'trainer.logger=[console,wandb]' \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.max_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  "hydra.run.dir=$RUN_DIR/hydra" \
  'checkpoint.save_contents=[hf_model,extra]'

if [[ "$DRY_RUN" == 0 ]]; then
  [[ -f "$RUN_DIR/global_step_$TOTAL_STEPS/huggingface/config.json" ]] || {
    echo "SFT ended without the expected HF checkpoint" >&2; exit 2;
  }
  echo "SFT complete: $RUN_DIR/global_step_$TOTAL_STEPS/huggingface"
fi
