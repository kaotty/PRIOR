"""Download pinned sources and restore the exact frozen RL question selection.

The bundled index contains identifiers, hashes and selection metadata, never
images or answers. It preserves the historical decontamination decision; it
does not claim to rerun the unavailable historical held-out parquet transforms.
Use ``rl.prepare --rebuild-selection`` to rerun that audit from supplied files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rl import build


INDEX_PATH = Path(__file__).with_name("frozen_selection.json.gz")
INDEX_SHA256 = "c55623aed201c4ae4698ab22a7225c1bc6a2203a15f29ce540bec22c686467fe"
CLEAN_FIELDS = (
    "item_id", "source", "source_repo", "source_revision", "source_split",
    "source_id", "question", "answer", "image", "expected_tool_need",
    "task_family", "operation", "image_width", "image_height", "image_sha256",
    "image_phash", "question_sha256", "selection_score", "zoom_score", "code_score",
)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_selection() -> dict:
    raw = INDEX_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INDEX_SHA256:
        raise ValueError("frozen RL selection index checksum mismatch")
    index = json.loads(gzip.decompress(raw))
    if index["schema_version"] != "tir-rl-frozen-selection-v1" or index["sources"] != build.SOURCES:
        raise ValueError("frozen RL selection source/version mismatch")
    for stage, order in index["orders"].items():
        if len(order) != build.EXPECTED_TOTAL or len(set(order)) != len(order):
            raise ValueError(f"invalid frozen order for {stage}")
        if text_hash("\n".join(order)) != index["datasets"][stage]["ordered_item_id_sha256"]:
            raise ValueError(f"frozen order checksum mismatch: {stage}")
    return index


def materialize_row(entry: dict, source_id: str, question: str, answer: str, image: dict) -> dict:
    """Apply the one frozen normalization, checking all content before writing."""
    if str(source_id) != entry["source_id"]:
        raise ValueError(f"source ID mismatch: {entry['item_id']}")
    transform = entry.get("question_transform")
    if transform == "remove_calculator_prefix":
        question = re.sub(r"^Use a calculator to find\b", "Find", question, flags=re.I)
    elif transform is not None:
        raise ValueError(f"unknown frozen question transform: {transform}")
    raw = build.image_bytes(image)
    actual = {
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "question_text_sha256": text_hash(question),
        "question_sha256": text_hash(build.normalized_question(question)),
        "answer_sha256": text_hash(answer),
    }
    for key, value in actual.items():
        if value != entry[key]:
            raise ValueError(f"{entry['item_id']}: {key} differs from the frozen selection")
    spec = build.SOURCES[entry["source"]]
    fingerprint = text_hash(
        f"{spec['revision']}\0{source_id}\0{build.normalized_question(question)}\0{actual['image_sha256']}"
    )
    if entry["item_id"] != f"{entry['source']}:{fingerprint[:24]}":
        raise ValueError(f"content ID mismatch: {entry['item_id']}")
    values = {
        **entry, "source_repo": spec["repo"], "source_revision": spec["revision"],
        "source_split": "train", "question": question, "answer": answer,
        "image": {"bytes": raw, "path": entry["image_path"]},
    }
    return {key: values[key] for key in CLEAN_FIELDS}


def verify_clean_file(path: Path, stage: str, index: dict) -> None:
    """Verify content and order of either newly restored or existing clean rows."""
    expected = index["orders"][stage]
    position = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if position >= len(expected) or row["item_id"] != expected[position]:
                raise ValueError(f"{stage}: row order differs at {position}")
            entry = index["rows"][row["item_id"]]
            restored = materialize_row(entry, row["source_id"], row["question"], row["answer"], row["image"])
            if restored != row:
                raise ValueError(f"{stage}: frozen metadata differs at {position}")
            position += 1
    if position != len(expected):
        raise ValueError(f"{stage}: expected {len(expected)} rows, got {position}")


def restore(output_root: Path, cache_dir: Path, no_download: bool = False) -> None:
    index = load_selection()
    missing = []
    for stage in index["orders"]:
        target = output_root / stage
        if target.exists():
            if not (target / "manifest.json").is_file() or not (target / "train.parquet").is_file():
                raise RuntimeError(f"partial output needs inspection: {target}")
            verify_clean_file(target / "train.parquet", stage, index)
            print(f"Verified existing frozen selection: {target}", flush=True)
        else:
            missing.append(stage)
    if not missing:
        return

    wanted = defaultdict(list)
    for entry in index["rows"].values():
        wanted[(entry["source"], entry["source_id"])].append(entry)
    materialized = {}
    for source, spec in build.SOURCES.items():
        snapshot = build.hub_snapshot(spec, cache_dir, no_download)
        paths = []
        for record in index["source_files"][source]:
            path = snapshot / record["path"]
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise ValueError(f"missing or wrong-sized pinned source: {source}/{record['path']}")
            if build.sha256_file(path) != record["sha256"]:
                raise ValueError(f"pinned source checksum mismatch: {source}/{record['path']}")
            paths.append(path)
        for row_index, raw_row in build.source_rows(source, paths):
            try:
                source_id, question, answer, image = build.clean_source_row(source, row_index, raw_row)
            except ValueError:
                continue
            entries = wanted.get((source, source_id), ())
            if not entries:
                continue
            image_sha256 = hashlib.sha256(build.image_bytes(image)).hexdigest()
            for entry in entries:
                if image_sha256 != entry["image_sha256"]:
                    continue
                row = materialize_row(entry, source_id, question, answer, image)
                previous = materialized.get(entry["item_id"])
                if previous is not None and previous != row:
                    raise ValueError(f"ambiguous pinned source row: {entry['item_id']}")
                materialized[entry["item_id"]] = row
        print(f"Read pinned {source}; restored {len(materialized)}/{len(index['rows'])} unique rows", flush=True)
    absent = set(index["rows"]) - materialized.keys()
    if absent:
        raise ValueError(f"{len(absent)} frozen rows were not found in pinned sources: {sorted(absent)[:5]}")

    for stage in missing:
        rows = [materialized[item] for item in index["orders"][stage]]
        build.validate_output(rows)
        target = output_root / stage
        temporary = target.with_name(f".{target.name}.restore")
        if temporary.exists():
            raise FileExistsError(f"unfinished restore needs inspection: {temporary}")
        temporary.mkdir(parents=True)
        parquet = temporary / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows), parquet, compression="zstd", compression_level=3, row_group_size=128)
        verify_clean_file(parquet, stage, index)
        manifest = {
            **index["datasets"][stage], "dataset_name": stage, "prompt_free": True,
            "selection_mode": "restore_frozen_selection", "selection_index_sha256": INDEX_SHA256,
            "historical_overlap_recomputed": False,
            "historical_overlap_note": index["historical_overlap_note"],
            "historical_overlap_files": index["historical_overlap_files"],
            "source_revisions": {name: spec["revision"] for name, spec in build.SOURCES.items()},
            "train_parquet": {"path": "train.parquet", "size_bytes": parquet.stat().st_size,
                              "sha256": build.sha256_file(parquet)},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(target)
        print(f"Restored exact frozen selection: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    restore(args.output_root, args.cache_dir, args.no_download)


if __name__ == "__main__":
    main()
