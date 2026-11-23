"""Download source releases and build the frozen SFT arms."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq
from sft.frozen import validate_pool


def run(module: str, *args: str, expected: Path, dry_run: bool) -> None:
    command = [sys.executable, "-m", module, *args]
    print(" ".join(command), flush=True)
    if dry_run:
        return
    if expected.exists():
        if pq.ParquetFile(expected).metadata.num_rows == 0:
            raise RuntimeError(f"empty existing source pool: {expected}")
        print(f"Reusing source pool: {expected}", flush=True)
        return
    subprocess.run(command, check=True)
    if not expected.is_file():
        raise RuntimeError(f"{module} did not produce {expected}")


def convert_sources(args: argparse.Namespace) -> None:
    raw = args.sft_root / "source_pools_v1"
    clean = args.sft_root / "source_pools_v2_clean"
    mini_o3 = args.mini_o3 / "data" if (args.mini_o3 / "data").is_dir() else args.mini_o3
    thyme = args.thyme / "data" if (args.thyme / "data").is_dir() else args.thyme
    converters = [
        ("cof_zoom", "sft.utils.cof_zoom",
         ("--source_dir", str(args.cof), "--output_dir", str(raw), "--allow_multi_zoom")),
        ("mini_o3_zoom", "sft.utils.mini_o3_zoom",
         ("--input-dir", str(mini_o3), "--output", str(raw / "mini_o3_zoom/train.parquet"))),
        ("deepeyes_compute", "sft.utils.deepeyes_code",
         ("--input-root", str(args.deepeyes_v2), "--tokenizer", str(args.model),
          "--output", str(raw / "deepeyes_compute/train.parquet"))),
        ("thyme_compute", "sft.utils.thyme_code",
         ("--input-dir", str(thyme), "--output", str(raw / "thyme_compute/train.parquet"))),
    ]
    if args.pixel_reasoner is not None:
        converters.append(("pixel_reasoner_zoom", "sft.utils.pixel_zoom",
                           ("--input-json", str(args.pixel_reasoner / "release_checked_toolpart.json"),
                            "--images-zip", str(args.pixel_reasoner / "images.zip"),
                            "--output", str(raw / "pixel_reasoner_zoom/train.parquet"))))
    for source, module, options in converters:
        if not args.dry_run and (clean / source / "train.parquet").is_file():
            validate_pool(source, clean / source / "train.parquet")
            print(f"Reusing clean source pool: {clean / source}", flush=True)
            continue
        run(module, *options, expected=raw / source / "train.parquet", dry_run=args.dry_run)
        run("sft.utils.clean_pool",
            "--input", str(raw / source / "train.parquet"),
            "--output", str(clean / source / "train.parquet"),
            expected=clean / source / "train.parquet", dry_run=args.dry_run)
        if not args.dry_run:
            validate_pool(source, clean / source / "train.parquet")


def prepare_arm(args: argparse.Namespace) -> None:
    from sft.download import download
    from sft.frozen import materialize

    arm = args.arm
    input_root = args.input_root or args.sft_root.parent / "inputs/sft"
    cache_dir = args.cache_dir or Path(os.environ.get("HF_HOME", str(args.sft_root.parent / "cache"))) / "hub"
    high_file = args.sft_root / "core_clean_prompt_v1/frequency_high/train.parquet"
    if arm.startswith("filter"):
        if not high_file.is_file():
            args.arm = "high"
            prepare_arm(args)
            args.arm = arm
        budget = 25 if arm.startswith("filter25") else 50
        selection = "top" if arm.endswith("top") else "random"
        command = [sys.executable, "-m", "sft.select_pre_call", "--source", str(high_file),
                   "--model", str(args.model), "--template", "sft/chat_template.jinja",
                   "--sft-root", str(args.sft_root), "--budget", str(budget),
                   "--selection", selection]
        if args.dry_run:
            command.append("--dry-run")
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        return

    args.cof = input_root / "cof"
    args.mini_o3 = input_root / "mini_o3"
    args.deepeyes_v2 = input_root / "deepeyes_v2"
    args.thyme = input_root / "thyme"
    needs_pixel = arm == "ablations" or arm.startswith("mix-")
    args.pixel_reasoner = input_root / "pixel_reasoner" if needs_pixel else None
    pools = {"cof": "cof_zoom", "mini_o3": "mini_o3_zoom",
             "deepeyes_v2": "deepeyes_compute", "thyme": "thyme_compute"}
    if needs_pixel:
        pools["pixel_reasoner"] = "pixel_reasoner_zoom"
    missing = [name for name, pool in pools.items()
               if not (args.sft_root / "source_pools_v2_clean" / pool / "train.parquet").is_file()]
    if missing:
        download(input_root, cache_dir, only=missing, dry_run=args.dry_run)
        convert_sources(args)
    materialize(args.sft_root, "all" if arm == "original" else arm, dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("original", "low", "medium", "high", "ablations",
                                          "no-tool-0", "no-tool-25", "no-tool-50",
                                          "mix-25-75", "mix-50-50", "mix-75-25",
                                          "filter25-top", "filter50-top", "filter25-random", "filter50-random"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cof", type=Path, help="Local CoF snapshot for raw conversion")
    parser.add_argument("--mini-o3", type=Path)
    parser.add_argument("--deepeyes-v2", type=Path)
    parser.add_argument("--thyme", type=Path)
    parser.add_argument("--pixel-reasoner", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.arm:
        prepare_arm(args)
    else:
        if any(getattr(args, name) is None for name in ("cof", "mini_o3", "deepeyes_v2", "thyme")):
            parser.error("raw conversion requires --cof, --mini-o3, --deepeyes-v2 and --thyme")
        convert_sources(args)


if __name__ == "__main__":
    main()
