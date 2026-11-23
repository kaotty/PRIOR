#!/usr/bin/env python3
"""Summarize one completed Tool-On or Tool-Off benchmark shard."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


def load_records(root: Path):
    records = {}
    # Requeued runs can leave archived partial shards beside the live shard.
    for shard in root.iterdir():
        if not shard.is_dir() or re.fullmatch(r"shard_\d+", shard.name) is None:
            continue
        for path in (shard / "trajectories").glob("*.jsonl"):
            for line in path.open():
                row = json.loads(line)
                item_id = str(row.get("item_id") or "")
                assert item_id and item_id not in records, (item_id, path)
                records[item_id] = row
    return records


def one(records, expected):
    correct = calls = started = unparsed = unknown = 0
    categories = collections.Counter()
    category_n = collections.Counter()
    terminations = collections.Counter()
    for item_id, row in expected.items():
        trajectory = records[item_id]
        is_correct = bool(trajectory.get("correct"))
        correct += is_correct
        attempted = int(trajectory.get("attempted_supported_calls") or 0)
        calls += attempted
        started += attempted > 0
        category = str(row["extra_info"].get("category") or "")
        categories[category] += is_correct
        category_n[category] += 1
        terminations[str(trajectory.get("termination_reason") or "final_answer")] += 1
        unparsed += int(trajectory.get("unparsed_textual_tool_calls") or 0)
        unknown += int(trajectory.get("unknown_tool_calls") or 0)
    return {
        "rows": len(expected),
        "accuracy_pct": 100 * correct / len(expected),
        "correct": correct,
        "tool_start_rate_pct": 100 * started / len(expected),
        "conditional_depth": calls / started if started else 0,
        "attempted_calls": calls,
        "unparsed_textual_tool_calls": unparsed,
        "unknown_tool_calls": unknown,
        "termination_reasons": dict(terminations),
        "by_category": {
            category: {
                "n": category_n[category],
                "accuracy_pct": 100 * categories[category] / category_n[category],
            }
            for category in sorted(category_n)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        str(row["extra_info"]["item_id"]): row
        for row in pq.read_table(args.eval_file, columns=["extra_info"]).to_pylist()
    }
    records = load_records(args.run_dir.parent)
    if set(records) != set(expected):
        raise ValueError(f"Expected {len(expected)} item IDs, found {len(records)}")
    result = {"status": "PASS", **one(records, expected)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("rows", "accuracy_pct", "tool_start_rate_pct", "conditional_depth")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
