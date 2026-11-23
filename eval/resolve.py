"""Resolve a checkpoint into the SFT model, run name, benchmark, and tool mode."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


BENCHMARKS = ("all", "vstar", "chartqa", "mathvista", "wemath", "charxiv")
MODES = ("both", "tool-on", "tool-off")


def absolute(path: str, caller_dir: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else caller_dir / value).resolve(strict=False)


def run_name(checkpoint: Path, output_root: Path, kind: str) -> str:
    try:
        return checkpoint.relative_to(output_root / kind).parts[0]
    except (ValueError, IndexError):
        digest = hashlib.sha256(str(checkpoint).encode()).hexdigest()[:10]
        return f"{kind}_{digest}"


def resolve(args: argparse.Namespace) -> tuple[str, str, str, str, str, str]:
    root = absolute(args.output_root, args.caller_dir)
    remaining = args.remaining

    if args.checkpoint_or_kind in ("sft", "rl"):
        if not 2 <= len(remaining) <= 4:
            raise ValueError("Explicit form: sft|rl SFT_MODEL RUN_NAME [BENCHMARK] [MODE]")
        kind = args.checkpoint_or_kind
        sft_model = absolute(remaining[0], args.caller_dir)
        name = remaining[1]
        benchmark = remaining[2] if len(remaining) >= 3 else args.benchmark
        mode = remaining[3] if len(remaining) >= 4 else args.mode
        raw = args.rl_checkpoint or os.environ.get("TIR_RL_CHECKPOINT")
        rl_checkpoint = absolute(raw, args.caller_dir) if kind == "rl" and raw else None
    else:
        if remaining:
            raise ValueError("Pass one checkpoint path, or use the explicit sft|rl form")
        checkpoint = absolute(args.checkpoint_or_kind, args.caller_dir)
        is_sft = (checkpoint / "config.json").is_file() or checkpoint.is_relative_to(root / "sft")
        kind = "sft" if is_sft else "rl"
        if kind == "rl" and not (checkpoint / "actor").is_dir() and not checkpoint.is_relative_to(root / "rl"):
            raise ValueError(f"Cannot identify checkpoint type: {checkpoint}")
        name = run_name(checkpoint, root, kind)
        benchmark, mode = args.benchmark, args.mode
        rl_checkpoint = checkpoint if kind == "rl" else None
        if kind == "sft":
            sft_model = checkpoint
        else:
            receipt = checkpoint.parent.parent / "sft_checkpoint.txt"
            if receipt.is_file():
                sft_model = absolute(receipt.read_text().strip(), args.caller_dir)
            else:
                seed = os.environ.get("TIR_SEED", "42")
                step = os.environ.get("SFT_TOTAL_STEPS", "50")
                sft_model = root / "sft" / name / f"seed_{seed}" / f"global_step_{step}" / "huggingface"

    if args.sft_checkpoint:
        sft_model = absolute(args.sft_checkpoint, args.caller_dir)
    if kind == "rl" and rl_checkpoint is None and not os.environ.get("TIR_EVAL_MODEL_PATH"):
        raise ValueError("RL evaluation needs a raw checkpoint or TIR_EVAL_MODEL_PATH")
    if benchmark not in BENCHMARKS or mode not in MODES:
        raise ValueError(f"Unknown benchmark or tool mode: {benchmark}, {mode}")
    if benchmark == "all" and (os.environ.get("TIR_EVAL_FILE_ON") or os.environ.get("TIR_EVAL_FILE_OFF")):
        raise ValueError("Eval-file overrides require one named benchmark")
    if os.environ.get("TIR_EVAL_OUTPUT_DIR") and (benchmark == "all" or mode == "both"):
        raise ValueError("TIR_EVAL_OUTPUT_DIR requires one benchmark and one tool mode")

    result = (kind, str(sft_model), name, benchmark, mode, str(rl_checkpoint or "-"))
    if any("\n" in value for value in result):
        raise ValueError("Newlines in checkpoint paths or run names are unsupported")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--caller-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default=os.environ.get("TIR_EVAL_BENCHMARK", "all"))
    parser.add_argument("--mode", default=os.environ.get("TIR_EVAL_MODE", "both"))
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--rl-checkpoint")
    parser.add_argument("checkpoint_or_kind")
    parser.add_argument("remaining", nargs="*")
    args = parser.parse_intermixed_args()
    try:
        for field in resolve(args):
            print(field)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
