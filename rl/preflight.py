"""Check RL inputs, resource limits, resume shards, and final checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def absolute(path: str, caller_dir: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else caller_dir / value).resolve(strict=False)


def resolve(args: argparse.Namespace) -> None:
    values = args.remaining
    root = absolute(args.output_root, args.caller_dir)
    data_root = absolute(args.data_root, args.caller_dir)
    if len(values) == 3:
        model = absolute(values[0], args.caller_dir)
        data = absolute(values[1], args.caller_dir)
        name = values[2]
    elif len(values) == 1:
        value = values[0]
        if "/" in value or (args.caller_dir / value).is_dir():
            model = absolute(value, args.caller_dir)
            try:
                name = model.relative_to(root / "sft").parts[0]
            except (ValueError, IndexError):
                name = "sft_" + hashlib.sha256(str(model).encode()).hexdigest()[:10]
        else:
            name = value
            seed = os.environ.get("TIR_SEED", "42")
            step = os.environ.get("SFT_TOTAL_STEPS", "50")
            model = root / "sft" / name / f"seed_{seed}" / f"global_step_{step}" / "huggingface"
        data = data_root / "formal/rl_16k_v2"
    else:
        raise ValueError("Pass an SFT checkpoint or SFT_MODEL DATA_DIR RUN_NAME")
    if args.sft_checkpoint:
        model = absolute(args.sft_checkpoint, args.caller_dir)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"invalid run name: {name}")
    result = (str(model), str(data), name, args.batch_size or "-",
              str(args.steps) if args.steps is not None else "-")
    if any("\n" in value for value in result):
        raise ValueError("Newlines in checkpoint paths or run names are unsupported")
    for value in result:
        print(value)


def check_resume(path: Path, checkpoint_dir: Path, gpus: int) -> None:
    if path.name not in {"global_step_8", "global_step_16", "global_step_24"}:
        raise ValueError(f"unsupported resume step: {path}")
    if path.parent != checkpoint_dir:
        raise ValueError("resume checkpoint must belong to this run")
    require(path / "data.pt")
    if (path / "data.pt").stat().st_size == 0:
        raise ValueError(f"empty resume checkpoint metadata: {path / 'data.pt'}")
    for rank in range(gpus):
        for kind in ("model", "optim", "extra_state"):
            require(path / "actor" / f"{kind}_world_size_{gpus}_rank_{rank}.pt")


def before(args: argparse.Namespace) -> None:
    for name in ("gpus", "batch", "rollouts", "steps", "prompt_length", "response_length",
                 "tokens_per_gpu", "sequence_parallel", "max_sequences", "chunked_prefill"):
        positive(name, getattr(args, name))
    if args.seed < 0 or args.dry_run not in (0, 1):
        raise ValueError("seed must be nonnegative and dry-run must be 0 or 1")
    if args.rollouts < 2:
        raise ValueError("GRPO needs at least two rollouts per question")
    if args.gpus % args.sequence_parallel:
        raise ValueError("GPU count must be divisible by sequence-parallel size")
    if args.tokens_per_gpu * args.sequence_parallel < args.prompt_length + args.response_length:
        raise ValueError("dynamic-batch capacity cannot hold the longest response")
    if args.save_freq not in (8, 32) and os.environ.get("TIR_ALLOW_FINAL_ONLY_CHECKPOINT") != "1":
        raise ValueError("save every eight steps, or explicitly allow final-only checkpointing")
    if args.resume and (args.save_freq != 8 or args.steps != 32):
        raise ValueError("resume requires a 32-step run saving every eight steps")
    if args.dry_run:
        return

    require(args.model / "config.json")
    require(args.data / "train.parquet")
    require(args.data / "loader_val.parquet")
    require(args.data / "dataset_manifest.json")
    require(args.tool_config)
    require(args.reward_file)
    if args.resume:
        check_resume(Path(args.resume).resolve(strict=True), args.checkpoint_dir, args.gpus)
    elif args.run_dir.exists():
        raise FileExistsError(f"output already exists: {args.run_dir}")
    report = os.environ.get("RL_VALIDATION_REPORT")
    command = [sys.executable, "-m", "rl.validate", str(args.data),
               "--data-root", str(args.data_root)]
    if report:
        command += ["--report-path", report]
    subprocess.run(command, check=True)


def after(args: argparse.Namespace) -> None:
    final = args.checkpoint_dir / f"global_step_{args.steps}"
    if args.save_freq == 8 and args.steps == 32:
        require(final / "data.pt")
        for rank in range(args.gpus):
            for kind in ("model", "optim", "extra_state"):
                require(final / "actor" / f"{kind}_world_size_{args.gpus}_rank_{rank}.pt")
        print("Final checkpoint verified; keeping intermediate recovery checkpoints.")
    if final.is_dir():
        print(f"RL checkpoint: {final}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("resolve", "before", "after"))
    parser.add_argument("--output-root")
    parser.add_argument("--caller-dir", type=Path)
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--batch-size")
    parser.add_argument("remaining", nargs="*")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--tool-config", type=Path)
    parser.add_argument("--reward-file", type=Path)
    parser.add_argument("--resume", default="")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--response-length", type=int)
    parser.add_argument("--tokens-per-gpu", type=int)
    parser.add_argument("--sequence-parallel", type=int)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--chunked-prefill", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--dry-run", type=int, default=0)
    args = parser.parse_intermixed_args()
    try:
        if args.phase == "resolve":
            if not args.output_root or not args.data_root or not args.caller_dir:
                raise ValueError("resolve needs --output-root, --data-root and --caller-dir")
            resolve(args)
        elif args.phase == "before":
            required = ("model", "data", "data_root", "run_dir", "checkpoint_dir",
                        "tool_config", "reward_file", "gpus", "batch", "rollouts", "steps",
                        "prompt_length", "response_length", "tokens_per_gpu",
                        "sequence_parallel", "max_sequences", "chunked_prefill", "save_freq")
            if any(getattr(args, name) is None for name in required):
                raise ValueError("before needs model, data, resources and checkpoint paths")
            before(args)
        else:
            if any(getattr(args, name) is None for name in ("checkpoint_dir", "gpus", "steps", "save_freq")):
                raise ValueError("after needs checkpoint directory, GPUs, steps and save frequency")
            after(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
