"""Restore and validate the frozen 16,384-question RL dataset.

By default, download pinned source files and restore the bundled selection.
Historical decontamination can be recomputed with --rebuild-selection and the
original held-out parquet manifest; it is not silently substituted by new data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run(module: str, options: list[str], *, output: Path, marker: str, dry_run: bool) -> None:
    command = [sys.executable, "-m", module, *options]
    print(" ".join(command), flush=True)
    if dry_run:
        return
    if output.exists():
        if not (output / marker).is_file():
            raise RuntimeError(f"partial output requires inspection: {output}")
        print(f"Reusing completed stage: {output}", flush=True)
        return
    subprocess.run(command, check=True)
    if not (output / marker).is_file():
        raise RuntimeError(f"{module} did not produce {output / marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Tokenizer/processor for final parquet")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--rebuild-selection", action="store_true",
                        help="Recompute historical selection from the original held-out parquets")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate an existing final dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    formal = args.data_root / "formal"
    clean_v1 = formal / "clean_rl_16k_v1"
    clean_v2 = formal / "clean_rl_16k_v2"
    result = formal / "rl_16k_v2"
    if not args.check_only:
        if args.model is None or args.cache_dir is None:
            parser.error("--model and --cache-dir are required to build the dataset")
        download = ["--no-download"] if args.no_download else []
        if args.rebuild_selection:
            if args.heldout_manifest is None:
                parser.error("--rebuild-selection requires --heldout-manifest")
            run("rl.build",
                ["--output-dir", str(clean_v1), "--cache-dir", str(args.cache_dir),
                 "--heldout-manifest", str(args.heldout_manifest),
                 "--seed", "20260812", *download],
                output=clean_v1, marker="manifest.json", dry_run=args.dry_run)
            run("rl.repair",
                ["--input-dir", str(clean_v1), "--output-dir", str(clean_v2),
                 "--cache-dir", str(args.cache_dir),
                 "--heldout-manifest", str(args.heldout_manifest)],
                output=clean_v2, marker="manifest.json", dry_run=args.dry_run)
        else:
            if args.heldout_manifest is not None:
                parser.error("--heldout-manifest requires --rebuild-selection")
            restore = [sys.executable, "-m", "rl.restore", "--output-root", str(formal),
                       "--cache-dir", str(args.cache_dir), *download]
            print(" ".join(restore), flush=True)
            if not args.dry_run:
                subprocess.run(restore, check=True)
        run("rl.format",
            ["--input-dir", str(clean_v2), "--output-dir", str(result),
             "--model", str(args.model), "--system-prompt-profile", "canonical",
             "--batch-size", "512"],
            output=result, marker="dataset_manifest.json", dry_run=args.dry_run)
    command = [sys.executable, "-m", "rl.validate", str(result),
               "--data-root", str(args.data_root), "--check-only"]
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
