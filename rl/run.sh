#!/usr/bin/env bash
# Prepare frozen RL questions or run outcome-only GRPO from an SFT checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/paths.sh"
PYTHON_BIN="${TIR_PYTHON_BIN:-python3}"
DATA_ROOT="${TIR_DATA_ROOT:-}"
OUTPUT_ROOT="${TIR_OUTPUT_ROOT:-}"
MODEL_BASE="${TIR_BASE_MODEL_PATH:-}"
DRY_RUN="${TIR_DRY_RUN:-0}"
DATA_CACHE="${HF_DATASETS_CACHE:-${HF_HOME:-$OUTPUT_ROOT/cache/huggingface}/datasets}"

if [[ "${1:-}" == prepare ]]; then
  [[ -n "$DATA_ROOT" && -n "$MODEL_BASE" ]] || { echo "Run bash scripts/setup_env.sh first" >&2; exit 2; }
  shift
  if [[ "${1:-}" == --heldout-manifest ]]; then
    [[ $# -ge 2 ]] || { echo "Missing held-out manifest path" >&2; exit 2; }
    manifest="$(absolute_path "$2")"
    shift 2
    set -- --rebuild-selection --heldout-manifest "$manifest" "$@"
  fi
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  DRY=()
  [[ "$DRY_RUN" == 1 ]] && DRY=(--dry-run)
  exec "$PYTHON_BIN" -m rl.prepare \
    --data-root "$DATA_ROOT" --model "$MODEL_BASE" --cache-dir "$DATA_CACHE" \
    "$@" "${DRY[@]}"
fi

# The older explicit three-argument form already names its RL data directory.
if [[ -z "$DATA_ROOT" && $# -eq 3 ]]; then
  DATA_ROOT="$(dirname "$(dirname "$(absolute_path "$2")")")"
fi
[[ -n "$DATA_ROOT" && -n "$OUTPUT_ROOT" ]] || { echo "Run bash scripts/setup_env.sh first" >&2; exit 2; }
resolved="$("$PYTHON_BIN" -m rl.preflight resolve \
  --output-root "$OUTPUT_ROOT" --data-root "$DATA_ROOT" --caller-dir "$CALLER_DIR" "$@")"
mapfile -t selection <<< "$resolved"
[[ ${#selection[@]} -eq 5 ]] || { echo "Could not resolve the SFT checkpoint" >&2; exit 2; }
MODEL_PATH="${selection[0]}"
DATA_DIR="${selection[1]}"
RUN_NAME="${selection[2]}"
[[ "${selection[3]}" == - ]] || RL_TRAIN_BATCH_SIZE="${selection[3]}"
[[ "${selection[4]}" == - ]] || RL_TOTAL_STEPS_OVERRIDE="${selection[4]}"

ulimit -n 65535 2>/dev/null || true

MODEL_PATH="$(absolute_path "$MODEL_PATH")"
DATA_DIR="$(absolute_path "$DATA_DIR")"
if [[ -n "${RL_RESUME_FROM_PATH:-}" && "$RL_RESUME_FROM_PATH" != /* ]]; then
  export RL_RESUME_FROM_PATH="$CALLER_DIR/$RL_RESUME_FROM_PATH"
fi
cd "$REPO_ROOT"
OUTPUT_ROOT="${TIR_OUTPUT_ROOT:?Set TIR_OUTPUT_ROOT to node-local output storage}"
[[ "$OUTPUT_ROOT" == /* ]] || OUTPUT_ROOT="$CALLER_DIR/$OUTPUT_ROOT"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
PYTHON_BIN="${TIR_PYTHON_BIN:-python3}"
TIR_N_GPUS="${TIR_N_GPUS:-2}"
TIR_SEED="${TIR_SEED:-42}"
TIR_DRY_RUN="${TIR_DRY_RUN:-0}"
TIR_DATA_ROOT="${TIR_DATA_ROOT:-$(dirname "$(dirname "$DATA_DIR")")}"

run_command() {
  if [[ "$TIR_DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN command:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# Edit the values in this block to change the RL experiment.
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/loader_val.parquet"
CONFIG_DIR="$REPO_ROOT/rl/config"
TOOL_CONFIG="$CONFIG_DIR/tools/both.yaml"
REWARD_FILE="$REPO_ROOT/verl/utils/reward_score/frozen_rl_mixed.py"
RUN_ID="${TIR_RUN_ID:-tir_${RUN_NAME//-/_}_seed$TIR_SEED}"
RUN_DIR="$OUTPUT_ROOT/rl/$RUN_NAME/both/seed_$TIR_SEED/rl/$RUN_ID"
CKPT_DIR="$RUN_DIR/checkpoints"

TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-512}"
ROLLOUT_N="${RL_ROLLOUT_N:-5}"
TOTAL_STEPS="${RL_TOTAL_STEPS_OVERRIDE:-32}"
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
MAX_EFFECTIVE_TOOL_CALLS=8
MAX_ASSISTANT_TURNS=9
MAX_TOOL_RESPONSE_LENGTH=2048

# Resource controls may change without changing the examples or optimizer steps.
PPO_TOKENS_PER_GPU="${TIR_RL_TOKENS_PER_GPU:-10240}"
SEQUENCE_PARALLEL_SIZE="${TIR_RL_SP_SIZE:-1}"
ROLLOUT_GPU_UTIL="${TIR_ROLLOUT_GPU_UTIL:-0.5}"
ROLLOUT_MAX_NUM_SEQS="${TIR_ROLLOUT_MAX_NUM_SEQS:-128}"
CHUNKED_PREFILL_SIZE="${TIR_ROLLOUT_CHUNKED_PREFILL:-2048}"
SAVE_FREQ="${RL_SAVE_FREQ_OVERRIDE:-${TIR_RL_SAVE_FREQ:-8}}"

# Resource and checkpoint checks are in rl.preflight.
CHECKPOINT_CONTENTS='[model,optimizer,extra]'
if [[ "$SAVE_FREQ" != 8 ]]; then
  CHECKPOINT_CONTENTS='[model,extra]'
fi

RESUME_ARGS=(trainer.resume_mode=disable)
if [[ -n "${RL_RESUME_FROM_PATH:-}" ]]; then
  RL_RESUME_FROM_PATH="$(absolute_path "$RL_RESUME_FROM_PATH")"
  RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=$RL_RESUME_FROM_PATH")
fi

SP_ARGS=()
if (( SEQUENCE_PARALLEL_SIZE > 1 )); then
  SP_ARGS=(
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SEQUENCE_PARALLEL_SIZE"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SEQUENCE_PARALLEL_SIZE"
  )
fi

CKPT_KEEP_ARGS=()
if [[ "$TOTAL_STEPS" == 32 ]]; then
  CKPT_KEEP_ARGS=(trainer.max_actor_ckpt_to_keep=4)
fi

CACHE_ROOT="$(absolute_path "${TIR_CACHE_ROOT:-$OUTPUT_ROOT/cache}")"
source "$REPO_ROOT/scripts/runtime_env.sh"
export PYTHONPATH="$REPO_ROOT/scripts/compat:$REPO_ROOT"
export TIR_MROPE_LAYOUT_COMPAT="${TIR_MROPE_LAYOUT_COMPAT:-1}"
unset RAY_ADDRESS RAY_RUNTIME_ENV_HOOK ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export TIR_EFFECTIVE_TOOL_CALL_LIMIT="$MAX_EFFECTIVE_TOOL_CALLS"
export TIR_AGENT_HORIZON_COMPAT=1
export SGLANG_WATCHDOG_TIMEOUT=1800
export HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0
export PYTHONHASHSEED="$TIR_SEED"
export TRAJECTORY_LOG_DIR="$RUN_DIR/trajectories"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CODE_EXECUTION_TIMEOUT=15 CODE_FRAMEWORK_TIMEOUT=300
export WANDB_RESUME=never
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-pure_rl_4arm_seed${TIR_SEED}}"
if [[ -z "${WANDB_API_KEY:-}" && -n "${TIR_WANDB_API_KEY_FILE:-}" &&
      -r "$TIR_WANDB_API_KEY_FILE" ]]; then
  IFS= read -r WANDB_API_KEY < "$TIR_WANDB_API_KEY_FILE"
  export WANDB_API_KEY
fi

"$PYTHON_BIN" -m rl.preflight before \
  --model "$MODEL_PATH" --data "$DATA_DIR" --data-root "$TIR_DATA_ROOT" \
  --run-dir "$RUN_DIR" --checkpoint-dir "$CKPT_DIR" \
  --tool-config "$TOOL_CONFIG" --reward-file "$REWARD_FILE" \
  --gpus "$TIR_N_GPUS" --seed "$TIR_SEED" \
  --batch "$TRAIN_BATCH_SIZE" --rollouts "$ROLLOUT_N" --steps "$TOTAL_STEPS" \
  --prompt-length "$MAX_PROMPT_LENGTH" --response-length "$MAX_RESPONSE_LENGTH" \
  --tokens-per-gpu "$PPO_TOKENS_PER_GPU" --sequence-parallel "$SEQUENCE_PARALLEL_SIZE" \
  --max-sequences "$ROLLOUT_MAX_NUM_SEQS" --chunked-prefill "$CHUNKED_PREFILL_SIZE" \
  --save-freq "$SAVE_FREQ" --resume "${RL_RESUME_FROM_PATH:-}" --dry-run "$TIR_DRY_RUN"
if [[ "$TIR_DRY_RUN" == 0 ]]; then
  mkdir -p "$CKPT_DIR" "$RUN_DIR/hydra" "$TRAJECTORY_LOG_DIR"
  printf '%s\n' "$(realpath -m "$MODEL_PATH")" > "$RUN_DIR/sft_checkpoint.txt"
fi

printf 'RL run=%s model=%s data=%s steps=%s save_every=%s\n' \
  "$RUN_NAME" "$MODEL_PATH" "$DATA_DIR" "$TOTAL_STEPS" "$SAVE_FREQ"
cd "$REPO_ROOT"

# Mirrors the veRL example style: one visible Hydra command with fixed settings.
run_command "$PYTHON_BIN" -m verl.trainer.main_ppo \
  "--config-path=$CONFIG_DIR" --config-name=grpo \
  "hydra.run.dir=$RUN_DIR/hydra" \
  ray_kwargs.ray_init.num_cpus=32 \
  "custom_reward_function.path=$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  algorithm.adv_estimator=grpo \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
  "data.max_response_length=$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.dataloader_num_workers=0 \
  data.shuffle=False \
  "data.seed=$TIR_SEED" \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$VAL_FILE" \
  "data.tool_config_path=$TOOL_CONFIG" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_TOKENS_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  "actor_rollout_ref.actor.data_loader_seed=$TIR_SEED" \
  "actor_rollout_ref.actor.fsdp_config.seed=$TIR_SEED" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  "actor_rollout_ref.actor.checkpoint.save_contents=$CHECKPOINT_CONTENTS" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_UTIL" \
  "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
  "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_TOOL_RESPONSE_LENGTH" \
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=600 \
  "+actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=$TIR_SEED" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  "actor_rollout_ref.ref.fsdp_config.seed=$TIR_SEED" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  'trainer.logger=[console,wandb]' \
  "trainer.project_name=$WANDB_PROJECT" \
  "trainer.experiment_name=tir_both_both_seed${TIR_SEED}_${RUN_ID}" \
  "trainer.n_gpus_per_node=$TIR_N_GPUS" \
  trainer.nnodes=1 \
  "trainer.save_freq=$SAVE_FREQ" \
  trainer.test_freq=-1 \
  "trainer.default_local_dir=$CKPT_DIR" \
  "${RESUME_ARGS[@]}" \
  trainer.val_before_train=False \
  trainer.ray_wait_register_center_timeout=300 \
  "trainer.total_training_steps=$TOTAL_STEPS" \
  trainer.total_epochs=1 \
  "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS" \
  "+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=$CHUNKED_PREFILL_SIZE" \
  "${SP_ARGS[@]}" "${CKPT_KEEP_ARGS[@]}"

[[ "$TIR_DRY_RUN" == 0 ]] || exit 0
"$PYTHON_BIN" -m rl.preflight after \
  --checkpoint-dir "$CKPT_DIR" --gpus "$TIR_N_GPUS" \
  --steps "$TOTAL_STEPS" --save-freq "$SAVE_FREQ"
