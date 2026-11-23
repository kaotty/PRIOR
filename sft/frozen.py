"""Materialize the published SFT selections without rerunning model sampling.

The compact bundle stores record IDs/order, token metadata, generated direct
answers, and hashes of the original training payload. Images and tool traces
are recovered from their official source pools and must match those hashes.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from sft.utils.format import normalize_prompt_messages, validate_assistant_messages, validate_unified_prompt
from sft.utils.records import _normalize_messages, _output_schema
from sft.utils.tool_schema import FORMAL_TWO_TOOL_SCHEMAS


BUNDLE = Path(__file__).with_name("frozen_selections.json.gz")
ARMS = {
    "low": "frequency_low", "medium": "no_tool_25__frequency_medium", "high": "frequency_high",
    "no-tool-0": "no_tool_0", "no-tool-25": "no_tool_25__frequency_medium", "no-tool-50": "no_tool_50",
    "mix-25-75": "mixture_25_75", "mix-50-50": "mixture_50_50", "mix-75-25": "mixture_75_25",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def without_nulls(value):
    if isinstance(value, dict):
        return {key: without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [without_nulls(item) for item in value]
    return value


def payload_hash(row: dict) -> str:
    """Hash every model-visible field, ignoring Arrow's optional null keys."""
    value = {key: without_nulls(row[key]) for key in ("messages", "tools", "data_source", "ability")}
    value["images"] = [
        {"path": str(image.get("path") or ""),
         "sha256": hashlib.sha256(bytes(image.get("bytes") or b"")).hexdigest()}
        for image in row["images"]
    ]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def source_selection(pool: str) -> list[dict]:
    with gzip.open(BUNDLE, "rt", encoding="utf-8") as handle:
        return json.load(handle)["source_pools"][pool]


def validate_pool(pool: str, path: Path) -> None:
    expected = source_selection(pool)
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if index >= len(expected):
                raise ValueError(f"{pool}: unexpected extra source records")
            record = expected[index]
            if (row["extra_info"]["sample_id"] != record["sample_id"] or
                    payload_hash(row) != record["payload_sha256"]):
                raise ValueError(f"{pool}: frozen source payload mismatch at row {index}: {record['sample_id']}")
            index += 1
    if index != len(expected):
        raise ValueError(f"{pool}: {index} rows, expected {len(expected)}")
    print(f"Frozen source pool PASS: {pool}, {index} rows", flush=True)


def materialize(sft_root: Path, arm: str, output_root: Path | None = None, dry_run: bool = False) -> None:
    with gzip.open(BUNDLE, "rt", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if arm == "all":
        names = [ARMS[key] for key in ("low", "medium", "high")]
    elif arm == "ablations":
        names = [ARMS[key] for key in ("no-tool-0", "no-tool-25", "no-tool-50",
                                       "mix-25-75", "mix-50-50", "mix-75-25")]
    else:
        names = [ARMS[arm]]
    output_root = output_root or sft_root / "core_clean_prompt_v1"
    bundle_hash = sha256(BUNDLE)
    pending = []
    for name in names:
        target = output_root / name
        if dry_run:
            print(f"Frozen selection: {name}, 2000 rows -> {target / 'train.parquet'}")
            continue
        if target.exists():
            manifest_path = target / "prompt_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"partial output requires inspection: {target}")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("frozen_selection_sha256") != bundle_hash:
                raise RuntimeError(f"existing output was not built from this frozen selection: {target}")
            if sha256(target / "train.parquet") != manifest["parquet_sha256"]:
                raise RuntimeError(f"existing training parquet changed: {target}")
            print(f"Validated existing frozen arm: {name}", flush=True)
        else:
            pending.append(name)
    if dry_run or not pending:
        return

    needed: dict[str, set[str]] = {}
    for name in pending:
        for record in bundle["arms"][name]["rows"]:
            needed.setdefault(record["source_pool"], set()).add(record["source_sample_id"])
    sources = {}
    for pool, sample_ids in needed.items():
        path = sft_root / "source_pools_v2_clean" / pool / "train.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing source pool: {path}; run SFT preparation first")
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for index, row in enumerate(batch.to_pylist()):
                sample_id = str(row["extra_info"]["sample_id"])
                if sample_id in sample_ids:
                    key = (pool, sample_id)
                    if key in sources:
                        raise ValueError(f"duplicate source ID: {key}")
                    sources[key] = (path, offset + index, row)
            offset += batch.num_rows
        missing = sample_ids - {key[1] for key in sources if key[0] == pool}
        if missing:
            raise ValueError(f"{pool}: {len(missing)} frozen IDs missing, e.g. {sorted(missing)[:3]}")

    output_root.mkdir(parents=True, exist_ok=True)
    for name in pending:
        specification = bundle["arms"][name]
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=output_root))
        path = temporary / "train.parquet"
        try:
            with pq.ParquetWriter(path, _output_schema(), compression="zstd") as writer:
                buffered = []
                for record in specification["rows"]:
                    source_path, source_index, source = sources[(record["source_pool"], record["source_sample_id"])]
                    messages = _normalize_messages(source["messages"])
                    images = source["images"]
                    metadata = copy.deepcopy(record["extra_info"])
                    if metadata["tool_type"] == "none":
                        messages = [messages[0], next(message for message in messages if message["role"] == "user"),
                                    {"role": "assistant", "content": bundle["direct_responses"][metadata["sample_id"]],
                                     "tool_calls": None}]
                        images = images[:1]
                    messages, _ = normalize_prompt_messages(messages)
                    validate_unified_prompt(messages)
                    validate_assistant_messages(messages)
                    metadata.update(source_parquet=str(source_path), source_row_index=source_index,
                                    spec_sha256=bundle_hash)
                    row = {"messages": messages, "images": images, "tools": copy.deepcopy(FORMAL_TWO_TOOL_SCHEMAS),
                           "data_source": record["data_source"], "ability": record["ability"], "extra_info": metadata}
                    if payload_hash(row) != record["payload_sha256"]:
                        raise ValueError(f"source payload differs from frozen experiment: {name}/{metadata['sample_id']}")
                    buffered.append(row)
                    if len(buffered) == 16:
                        writer.write_table(pa.Table.from_pylist(buffered, schema=_output_schema()))
                        buffered.clear()
                if buffered:
                    writer.write_table(pa.Table.from_pylist(buffered, schema=_output_schema()))
            manifest = {
                "status": "PASS", "arm": name, "rows": len(specification["rows"]),
                "frozen_selection_sha256": bundle_hash, "parquet_sha256": sha256(path),
                "assistant_loss_tokens": specification["assistant_loss_tokens"],
                "tool_type_counts": specification["tool_type_counts"],
                "payload_validation": "all messages, tools, image bytes/paths, data_source and ability match frozen hashes",
            }
            (temporary / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.rename(output_root / name)
            print(f"Frozen arm PASS: {name}, {manifest['rows']} rows", flush=True)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "all", "ablations"), default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    materialize(args.sft_root, args.arm, args.output_root, args.dry_run)


if __name__ == "__main__":
    main()
