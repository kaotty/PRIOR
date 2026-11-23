#!/usr/bin/env python3
"""Clean one converted source pool into the strict formal SFT format on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .records import _output_schema
from .format import CanonicalFormatError, normalize_assistant_messages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Converted canonical source parquet.")
    parser.add_argument("--output", required=True, help="New clean parquet; never overwritten.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    row["messages"] = normalize_assistant_messages(row.get("messages") or [])
    return row


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = _parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    report_path = output.with_suffix(".clean_report.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite output or report: {output}")

    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow != _output_schema():
        raise ValueError(f"{source}: input is not in the canonical Arrow schema")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output.stem}-", dir=output.parent))
    temporary = temporary_dir / output.name
    writer = pq.ParquetWriter(temporary, _output_schema(), compression="zstd")
    source_rows = 0
    written_rows = 0
    rejected: Counter[str] = Counter()
    rejected_examples: dict[str, str] = {}
    try:
        for batch in parquet.iter_batches(batch_size=32):
            clean_rows = []
            for row in batch.to_pylist():
                source_rows += 1
                try:
                    clean_rows.append(_clean_row(row))
                    written_rows += 1
                except CanonicalFormatError as exc:
                    rejected[exc.reason] += 1
                    rejected_examples.setdefault(exc.reason, str((row.get("extra_info") or {}).get("sample_id")))
            if clean_rows:
                writer.write_table(pa.Table.from_pylist(clean_rows, schema=_output_schema()))
        writer.close()
        writer = None
        if pq.ParquetFile(temporary).schema_arrow != _output_schema():
            raise RuntimeError("clean output Arrow schema mismatch")
        temporary.rename(output)
    finally:
        if writer is not None:
            writer.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)

    report = {
        "input": str(source),
        "input_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "source_rows": source_rows,
        "written_rows": written_rows,
        "rejected_rows": source_rows - written_rows,
        "rejected_by_reason": dict(sorted(rejected.items())),
        "rejected_example_sample_id": dict(sorted(rejected_examples.items())),
        "assistant_contract": {
            "tool_turn": "<think>reasoning</think> + structured tool_calls",
            "final_turn": "<think>reasoning</think>\\n\\boxed{answer}",
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
