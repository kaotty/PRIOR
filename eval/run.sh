#!/usr/bin/env bash
# Evaluate one checkpoint on the paired Tool-On and Tool-Off suite.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/paths.sh"
PYTHON_BIN="${TIR_PYTHON_BIN:-python3}"
DATA_ROOT="${TIR_DATA_ROOT:-}"
OUTPUT_ROOT="${TIR_OUTPUT_ROOT:-}"
DRY_RUN="${TIR_DRY_RUN:-0}"

if [[ "${1:-}" == prepare ]]; then
  [[ -n "$DATA_ROOT" ]] || { echo "Run bash scripts/setup_env.sh first" >&2; exit 2; }
  shift
  DRY=()
  [[ "$DRY_RUN" == 1 ]] && DRY=(--dry-run)
  exec "$PYTHON_BIN" -m eval.prepare \
    --data-root "$DATA_ROOT" --cache-dir "${HF_HOME:-$OUTPUT_ROOT/cache/huggingface}" \
    "$@" "${DRY[@]}"
fi

# eval.resolve locates the parent SFT model for a raw RL checkpoint.
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT:?Run bash scripts/setup_env.sh first}")"
resolved="$("$PYTHON_BIN" -m eval.resolve \
  --output-root "$OUTPUT_ROOT" --caller-dir "$CALLER_DIR" "$@")"
mapfile -t selection <<< "$resolved"
[[ ${#selection[@]} -eq 6 ]] || { echo "Could not resolve evaluation checkpoint" >&2; exit 2; }
if [[ "${selection[5]}" != - ]]; then export TIR_RL_CHECKPOINT="${selection[5]}"; fi
set -- "${selection[@]:0:5}"

ulimit -n 65535 2>/dev/null || true

KIND="${1:?Usage: bash eval/run.sh sft|rl SFT_MODEL_DIR RUN_NAME [benchmark|all] [tool-on|tool-off|both]}"
SFT_MODEL="${2:?Pass the SFT/HF checkpoint directory}"
SFT_MODEL="$(absolute_path "$SFT_MODEL")"
RUN_NAME="${3:?Give this evaluation a short name}"
BENCH_SELECTION="${4:-all}"
MODE_SELECTION="${5:-both}"
DATA_ROOT="${TIR_DATA_ROOT:?Set TIR_DATA_ROOT to the benchmark data root}"
OUTPUT_ROOT="${TIR_OUTPUT_ROOT:?Set TIR_OUTPUT_ROOT to node-local output storage}"
DATA_ROOT="$(absolute_path "$DATA_ROOT")"
OUTPUT_ROOT="$(absolute_path "$OUTPUT_ROOT")"
[[ -z "${TIR_RL_CHECKPOINT:-}" ]] || export TIR_RL_CHECKPOINT="$(absolute_path "$TIR_RL_CHECKPOINT")"
[[ -z "${TIR_EVAL_MODEL_PATH:-}" ]] || export TIR_EVAL_MODEL_PATH="$(absolute_path "$TIR_EVAL_MODEL_PATH")"
PYTHON_BIN="${TIR_PYTHON_BIN:-python3}"
NUM_GPUS="${TIR_N_GPUS:-2}"
SEED="${TIR_SEED:-42}"
DRY_RUN="${TIR_DRY_RUN:-0}"

[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ &&
   "$DRY_RUN" =~ ^[01]$ ]] || { echo "Invalid GPU count, seed, or dry-run flag" >&2; exit 2; }

require_file() {
  [[ -f "$1" ]] || { echo "Missing input file: $1" >&2; exit 2; }
}

run_command() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN command:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

eval_suite() {
  local benchmark="$1" condition="$2" category=core5_toolon_v1
  [[ "$condition" == tool_off ]] && category=core5_tooloff_v1
  case "$benchmark" in
    vstar|chartqa|mathvista) echo "$DATA_ROOT/eval/$category/$benchmark/$condition" ;;
    wemath) echo "$DATA_ROOT/eval/ocrbench_wemath_v1/wemath/full/$condition" ;;
    charxiv) echo "$DATA_ROOT/eval/charxiv_reasoning_v1/full/$condition" ;;
  esac
}

# The full suite and both conditions are the paper defaults.
if [[ "$BENCH_SELECTION" == all ]]; then
  BENCHMARKS=(vstar chartqa mathvista wemath charxiv)
else
  BENCHMARKS=("$BENCH_SELECTION")
fi
if [[ "$MODE_SELECTION" == both ]]; then
  MODES=(tool-on tool-off)
else
  MODES=("$MODE_SELECTION")
fi

# An RL run can use a raw FSDP checkpoint or a separately merged HF model.
CKPT=''
MODEL_PATH="$SFT_MODEL"
if [[ "$KIND" == rl ]]; then
  if [[ -n "${TIR_EVAL_MODEL_PATH:-}" ]]; then
    MODEL_PATH="$TIR_EVAL_MODEL_PATH"
  else
    MODEL_PATH=''
    CKPT="${TIR_RL_CHECKPOINT:?Set TIR_RL_CHECKPOINT to the raw RL checkpoint}"
  fi
elif [[ -n "${TIR_EVAL_MODEL_PATH:-}" ]]; then
  echo "TIR_EVAL_MODEL_PATH applies only to RL evaluation" >&2
  exit 2
fi

if [[ -n "$CKPT" ]]; then
  MODEL_OVERRIDE="actor_rollout_ref.model.path=$SFT_MODEL"
  RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CKPT")
else
  MODEL_OVERRIDE="actor_rollout_ref.model.path=$MODEL_PATH"
  RESUME_ARGS=(trainer.resume_mode=disable)
fi

if [[ "$DRY_RUN" == 0 ]]; then
  require_file "${MODEL_PATH:-$SFT_MODEL}/config.json"
  if [[ -n "$CKPT" ]]; then
    [[ -d "$CKPT/actor" && "$CKPT" == *global_step_* ]] || {
      echo "Expected an RL global_step_* checkpoint with actor/: $CKPT" >&2; exit 2;
    }
    for ((rank=0; rank<NUM_GPUS; rank++)); do
      require_file "$CKPT/actor/model_world_size_${NUM_GPUS}_rank_${rank}.pt"
    done
  fi
fi

CACHE_ROOT="$(absolute_path "${TIR_CACHE_ROOT:-$OUTPUT_ROOT/cache}")"
source "$REPO_ROOT/scripts/runtime_env.sh"
export PYTHONPATH="$REPO_ROOT/scripts/compat:$REPO_ROOT"
export TIR_MROPE_LAYOUT_COMPAT="${TIR_MROPE_LAYOUT_COMPAT:-1}"
unset RAY_ADDRESS RAY_RUNTIME_ENV_HOOK ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export TIR_BASE_MODEL_PATH="$SFT_MODEL"
export TIR_AGENT_HORIZON_COMPAT=1
export WANDB_JOB_TYPE=evaluation
export SGLANG_WATCHDOG_TIMEOUT=1800
export HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0
export PYTHONHASHSEED="$SEED"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CODE_EXECUTION_TIMEOUT=15 CODE_FRAMEWORK_TIMEOUT=300
export WANDB_RESUME=never
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gu_eval_seed${SEED}}"
if [[ -z "${WANDB_API_KEY:-}" && -n "${TIR_WANDB_API_KEY_FILE:-}" &&
      -r "$TIR_WANDB_API_KEY_FILE" ]]; then
  IFS= read -r WANDB_API_KEY < "$TIR_WANDB_API_KEY_FILE"
  export WANDB_API_KEY
fi

CONFIG_DIR="$REPO_ROOT/rl/config"
REWARD_FILE="$REPO_ROOT/verl/utils/reward_score/bench_w_tools.py"
require_file "$REWARD_FILE"
cd "$REPO_ROOT"

for bench in "${BENCHMARKS[@]}"; do
  for mode in "${MODES[@]}"; do
    condition="${mode//-/_}"
    suite="$(eval_suite "$bench" "$condition")"
    eval_file="$suite/shard_0/test.parquet"
    if [[ "$BENCH_SELECTION" != all ]]; then
      if [[ "$mode" == tool-on && -n "${TIR_EVAL_FILE_ON:-}" ]]; then
        eval_file="$TIR_EVAL_FILE_ON"
      elif [[ "$mode" == tool-off && -n "${TIR_EVAL_FILE_OFF:-}" ]]; then
        eval_file="$TIR_EVAL_FILE_OFF"
      fi
    fi
    run_dir="${TIR_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/eval/$RUN_NAME/$KIND/$bench/$condition/shard_0}"
    # Tool-On and Tool-Off limits are specified here.
    tool_config="$CONFIG_DIR/tools/both.yaml"
    agent_loop=tool_agent
    response_length=8192
    assistant_turns=9
    effective_calls=8
    if [[ "$mode" == tool-off ]]; then
      tool_config="$CONFIG_DIR/tools/none.yaml"
      agent_loop=single_turn_agent
      response_length=2048
      assistant_turns=1
      effective_calls=0
    fi
    require_file "$tool_config"

    if [[ "$DRY_RUN" == 0 ]]; then
      require_file "$eval_file"
      [[ ! -e "$run_dir" ]] || { echo "Output already exists: $run_dir" >&2; exit 2; }
      mkdir -p "$run_dir/trajectories" "$run_dir/hydra"
    fi

    # One row per question, deterministic decoding, with the same seed in
    # Tool-On and Tool-Off. The tool schema is the only interface difference.
    export TRAJECTORY_LOG_DIR="$run_dir/trajectories"
    export MAX_EFFECTIVE_TOOL_CALLS="$effective_calls"
    export TIR_EFFECTIVE_TOOL_CALL_LIMIT="$effective_calls"
    experiment="tir_${RUN_NAME}_${KIND}_${bench}_${condition}_$(basename "${CKPT:-$MODEL_PATH}")"
    printf 'Eval arm=%s checkpoint=%s benchmark=%s mode=%s\n' \
      "$RUN_NAME" "${CKPT:-$MODEL_PATH}" "$bench" "$mode"

    run_command "$PYTHON_BIN" -m verl.trainer.main_ppo \
      "--config-path=$CONFIG_DIR" --config-name=grpo \
      "hydra.run.dir=$run_dir/hydra" \
      ray_kwargs.ray_init.num_cpus=32 \
      "custom_reward_function.path=$REWARD_FILE" \
      custom_reward_function.name=compute_score \
      algorithm.adv_estimator=grpo \
      data.train_batch_size=32 \
      data.val_batch_size=32 \
      data.max_prompt_length=8192 \
      "data.max_response_length=$response_length" \
      data.filter_overlong_prompts=True \
      data.truncation=error \
      data.return_raw_chat=True \
      data.shuffle=False \
      data.validation_shuffle=False \
      "data.seed=$SEED" \
      "data.train_files=$eval_file" \
      "data.val_files=$eval_file" \
      "$MODEL_OVERRIDE" \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=16 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      'actor_rollout_ref.actor.checkpoint.load_contents=[model]' \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=sglang \
      actor_rollout_ref.rollout.enforce_eager=True \
      actor_rollout_ref.rollout.free_cache_engine=True \
      "actor_rollout_ref.rollout.gpu_memory_utilization=${TIR_ROLLOUT_GPU_UTIL:-0.5}" \
      actor_rollout_ref.rollout.max_num_seqs=32 \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$assistant_turns" \
      actor_rollout_ref.rollout.multi_turn.max_tokens_per_turn=2048 \
      +actor_rollout_ref.rollout.multi_turn.max_consecutive_identical_tool_calls=0 \
      +actor_rollout_ref.rollout.multi_turn.max_identical_tool_call_occurrences=0 \
      +actor_rollout_ref.rollout.multi_turn.max_consecutive_tool_failures=0 \
      actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
      actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
      "actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config" \
      "actor_rollout_ref.rollout.agent.default_agent_loop=$agent_loop" \
      actor_rollout_ref.rollout.agent.num_workers=1 \
      +actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=600 \
      "+actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=$SEED" \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.use_kl_in_reward=False \
      'trainer.logger=[console,wandb]' \
      "trainer.project_name=$WANDB_PROJECT" \
      "trainer.experiment_name=$experiment" \
      "trainer.n_gpus_per_node=$NUM_GPUS" trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      trainer.val_before_train=True \
      trainer.val_only=True \
      "${RESUME_ARGS[@]}" \
      trainer.ray_wait_register_center_timeout=300 \
      "trainer.validation_data_dir=$run_dir/generations" \
      "trainer.default_local_dir=$run_dir/scratch" \
      trainer.total_training_steps=1 \
      trainer.total_epochs=1

    [[ "$DRY_RUN" == 0 ]] || continue
    printf 'completed=%s\n' "$(date --iso-8601=seconds)" > "$run_dir/completed.txt"
    "$PYTHON_BIN" eval/validate_shard.py \
      --eval-file "$eval_file" --run-dir "$run_dir" \
      --tools "$([[ "$mode" == tool-on ]] && echo both || echo none)"
    "$PYTHON_BIN" -m eval.summarize \
      --eval-file "$eval_file" --run-dir "$run_dir" \
      --output "$(dirname "$run_dir")/summary.json"
    printf 'validated=%s\n' "$(date --iso-8601=seconds)" > "$run_dir/validated.txt"
  done
done
