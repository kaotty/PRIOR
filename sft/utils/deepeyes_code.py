#!/usr/bin/env python3
"""Rebuild the frozen DeepEyesV2 computation-only source pool on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .records import convert_trajectory
from .source_validation import (
    Candidate,
    _canonical_row,
    _scan_code,
)
from .records import _output_schema
from .parse_traces import parse_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-root",
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(
    json_paths: list[Path],
    images_root: Path,
    candidates: list[Candidate],
    output: Path,
    config_hash: str,
) -> tuple[int, Counter[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    wanted: dict[tuple[str, int], Candidate] = {}
    sample_ids: set[str] = set()
    for candidate in candidates:
        key = (Path(candidate.source_path).name, candidate.source_row_index)
        if key in wanted:
            raise ValueError(f"duplicate source location: {key}")
        if candidate.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {candidate.sample_id}")
        wanted[key] = candidate
        sample_ids.add(candidate.sample_id)

    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary output: {temporary}")
    writer = pq.ParquetWriter(temporary, _output_schema(), compression="zstd")
    histogram: Counter[str] = Counter()
    written = 0
    buffer: list[dict[str, Any]] = []
    try:
        for trajectory in parse_files(json_paths):
            candidate = wanted.get((trajectory.source_file, trajectory.raw_index))
            if candidate is None:
                continue
            converted = convert_trajectory(
                trajectory,
                images_root=images_root,
                target="diverse",
                data_source="deepeyesv2.compute_only",
            )
            if converted is None:
                raise RuntimeError(f"accepted trajectory failed conversion: {candidate.sample_id}")
            row = _canonical_row(converted, candidate, "", 42, config_hash)
            # Reproduce the historical pool, including its source-code errors.
            # sft.prepare checks every normalized payload against frozen hashes;
            # adding a new syntax filter here would change the baseline data.
            row["extra_info"]["domain_group"] = "unlabeled"
            row["extra_info"]["pool_role"] = "call_density_candidate_core"
            buffer.append(row)
            histogram[candidate.density] += 1
            if len(buffer) == 16:
                writer.write_table(pa.Table.from_pylist(buffer, schema=_output_schema()))
                written += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=_output_schema()))
            written += len(buffer)
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    if written != len(candidates):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"materialized {written} of {len(candidates)} accepted trajectories")
    temporary.replace(output)
    return written, histogram


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root).resolve()
    json_paths = sorted((input_root / "json").glob("sft_part_*.json"))
    images_root = input_root / "images"
    output = Path(args.output).resolve()
    if not json_paths or not images_root.is_dir():
        raise FileNotFoundError(f"missing DeepEyesV2 JSON or images under {input_root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    from verl.utils import hf_tokenizer

    tokenizer_path = Path(args.tokenizer).resolve()
    tokenizer = hf_tokenizer(tokenizer_path, local_files_only=True)
    candidates, rejected = _scan_code(json_paths, images_root, tokenizer)
    from sft.frozen import source_selection

    frozen = source_selection("deepeyes_compute")
    expected_ids = [row["sample_id"] for row in frozen]
    expected_set = set(expected_ids)
    selected = {row.sample_id: row for row in candidates if row.sample_id in expected_set}
    if set(selected) != expected_set:
        raise ValueError("DeepEyesV2 raw records do not cover the frozen source selection")
    candidates = [selected[sample_id] for sample_id in expected_ids]
    config = {
        "version": "deepeyesv2_compute_only_v1",
        "input_root": str(input_root),
        "json_paths": [str(path) for path in json_paths],
        "tokenizer": str(tokenizer_path),
        "filter": "one original image; computation-only Code; text output; no errors or repeated calls",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    written, histogram = _write_rows(json_paths, images_root, candidates, output, config_hash)
    report = {
        "config": config,
        "config_sha256": config_hash,
        "accepted_rows": len(candidates),
        "converted_rows": written,
        "converted_density_counts": dict(sorted(histogram.items())),
        "rejected": dict(sorted(rejected.items())),
        "source_sha256": {str(path): _sha256(path) for path in json_paths},
        "output": str(output),
        "output_sha256": _sha256(output),
        "formal_label_status": {
            "difficulty": "not_measured_requires_rollout",
            "oracle_utility": "not_measured_requires_rollout",
            "effective_call_count": "not_claimed_observed_call_count_only",
            "tool_execution": "source_output_not_replayed_yet",
        },
    }
    report_path = output.with_name("build_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
