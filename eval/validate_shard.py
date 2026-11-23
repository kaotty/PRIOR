#!/usr/bin/env python3
"""Fail closed unless one formal-eval shard has complete, valid trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


SUPPORTED_TOOLS = {"image_zoom_in_tool", "code_interpreter"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tools", choices=("none", "both"), required=True)
    args = parser.parse_args()

    if not (args.run_dir / "completed.txt").is_file():
        raise SystemExit(f"missing completion marker: {args.run_dir}")
    expected = {
        str(row["item_id"])
        for row in pq.read_table(args.eval_file, columns=["extra_info.item_id"]).to_pylist()
    }
    records: dict[str, dict] = {}
    for path in sorted((args.run_dir / "trajectories").glob("*.jsonl")):
        for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = str(record.get("item_id") or record.get("row_id") or "")
            if not item_id or item_id in records:
                raise SystemExit(f"missing or duplicate item ID at {path}:{line_number}")
            if record.get("replicate_id") != 0:
                raise SystemExit(f"unexpected replicate at {path}:{line_number}")
            if record.get("trajectory_schema_version") != "tir-eval-trajectory-v2":
                raise SystemExit(f"unexpected trajectory schema at {path}:{line_number}")
            parsed = int(record.get("parsed_tool_calls", -1))
            parsed_supported = int(record.get("parsed_supported_calls", -1))
            attempted_supported = int(record.get("attempted_supported_calls", -1))
            successful_supported = int(record.get("successful_supported_calls", -1))
            failed_supported = int(record.get("failed_supported_calls", -1))
            blocked_supported = int(record.get("blocked_supported_calls", -1))
            unknown = int(record.get("unknown_tool_calls", -1))
            textual = int(record.get("unparsed_textual_tool_calls", -1))
            if min(parsed, parsed_supported, attempted_supported, successful_supported,
                   failed_supported, blocked_supported, unknown, textual) < 0:
                raise SystemExit(f"missing v2 call counters at {path}:{line_number}")
            if (
                parsed_supported != attempted_supported + blocked_supported
                or attempted_supported != successful_supported + failed_supported
                or parsed != parsed_supported + unknown + textual
            ):
                raise SystemExit(f"inconsistent v2 call counters at {path}:{line_number}")
            if int(record.get("num_tool_calls", -1)) != parsed:
                raise SystemExit(f"legacy/v2 parsed-call counters disagree at {path}:{line_number}")
            if int(record.get("valid_tool_calls", -1)) != parsed_supported:
                raise SystemExit(f"legacy/v2 supported-call counters disagree at {path}:{line_number}")
            if int(record.get("successful_tool_calls", -1)) != successful_supported:
                raise SystemExit(f"legacy/v2 successful-call counters disagree at {path}:{line_number}")
            for turn in record.get("turns") or []:
                name = (turn.get("tool_call") or {}).get("name")
                if name and args.tools == "none":
                    raise SystemExit(f"None branch parsed a tool call at {path}:{line_number}: {name}")
                if name and name not in SUPPORTED_TOOLS and turn.get("executed_ok"):
                    raise SystemExit(f"unsupported tool executed at {path}:{line_number}: {name}")
            if args.tools == "none" and attempted_supported:
                raise SystemExit(f"None branch attempted an exposed tool at {path}:{line_number}")
            if args.tools == "both" and attempted_supported > 8:
                raise SystemExit(f"Both branch exceeded eight attempted tool calls at {path}:{line_number}")
            if record.get("duplicate_call_terminated") or record.get("termination_reason") == "tool_failure_limit":
                raise SystemExit(f"behavior-dependent early guard fired at {path}:{line_number}")
            records[item_id] = record
    if set(records) != expected:
        raise SystemExit(
            f"trajectory coverage mismatch: expected={len(expected)} actual={len(records)} run={args.run_dir}"
        )
    print(json.dumps({"status": "PASS", "tools": args.tools, "rows": len(records), "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    main()
