"""Select Original-High data by pre-call length or a matched random control.

``--budget`` names the nominal SFT token budget (25% or 50%), not the fraction
of unique demonstrations. The frozen top arms contain 16% and 36% of the
original rows and are repeated for four and two effective epochs, respectively.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ARMS = {25: ("top16_e4", "random25_e4"),
        50: ("top36_e2", "random50_e2")}


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
    parser.add_argument("--source", type=Path, required=True, help="Original High SFT parquet")
    parser.add_argument("--model", type=Path, required=True, help="Original tokenizer")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=tuple(ARMS), required=True)
    parser.add_argument("--selection", choices=("top", "random"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    top_name, random_name = ARMS[args.budget]
    ranked = args.sft_root / "high_precall_filtered_random_v2"
    matched = args.sft_root / "high_precall_token_control_v1"
    run("sft.filter_top",
        ["--stage", "rank", "--source", str(args.source), "--tokenizer", str(args.model),
         "--template", str(args.template), "--output", str(ranked)],
        output=ranked, marker="filtered/selection.json", dry_run=args.dry_run)
    run("sft.filter_top",
        ["--stage", "budget", "--source", str(args.source),
         "--top50-selection", str(ranked / "filtered/selection.json"),
         "--output", str(matched)],
        output=matched, marker=f"{top_name}/manifest.json", dry_run=args.dry_run)
    reference = matched / top_name
    if args.selection == "top":
        print(f"Selected {reference / 'train_runtime_50step.parquet'}", flush=True)
        return
    audit = reference / "render_audit_runtime.json"
    audit_command = [sys.executable, "-m", "sft.render_check",
                     "--parquet", str(reference / "train_runtime_50step.parquet"),
                     "--model", str(args.model), "--template", str(args.template),
                     "--output", str(audit)]
    print(" ".join(audit_command), flush=True)
    if not args.dry_run and not audit.is_file():
        subprocess.run(audit_command, check=True)
    output = args.sft_root / "high_precall_random_token_v1" / random_name
    run("sft.filter_random",
        ["--source", str(args.source), "--reference", str(reference),
         "--model", str(args.model), "--template", str(args.template),
         "--arm", random_name, "--output", str(output)],
        output=output, marker="manifest.json", dry_run=args.dry_run)
    print(f"Selected {output / 'train_runtime_50step.parquet'}", flush=True)


if __name__ == "__main__":
    main()
