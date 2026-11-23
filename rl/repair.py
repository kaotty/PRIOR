#!/usr/bin/env python3
"""Replace computation-tool-hinted rows while preserving every formal RL stratum and batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rl import build as v1


# Exclude instructions or descriptions that explicitly make a calculator/computer
# available as a computational aid. Ordinary task content such as pseudocode,
# computer screens, or a physical calculator object remains valid.
COMPUTATION_AID_PATTERN = re.compile(
    r"\b(?:use|using|used)\s+(?:a\s+|the\s+)?(?:scientific\s+)?calculator\b|"
    r"\bcalculator(?:\s+or\s+computer)?\s+(?:can\s+be|is|was)\s+used\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_replacements(
    selected: list[dict[str, Any]],
    source_paths: dict[str, list[Path]],
) -> dict[str, dict[str, Any]]:
    wanted = {(row["source"], row["source_index"]): row for row in selected}
    output: dict[str, dict[str, Any]] = {}
    for source, paths in source_paths.items():
        for index, source_row in v1.source_rows(source, paths):
            metadata = wanted.get((source, index))
            if metadata is None:
                continue
            source_id, question, answer, encoded_image = v1.clean_source_row(source, index, source_row)
            raw = v1.image_bytes(encoded_image)
            if hashlib.sha256(raw).hexdigest() != metadata["image_sha256"]:
                raise RuntimeError(f"image changed while materializing {source}:{source_id}")
            item_id = f"{source}:{metadata['fingerprint'][:24]}"
            output[item_id] = {
                "item_id": item_id,
                "source": source,
                "source_repo": v1.SOURCES[source]["repo"],
                "source_revision": v1.SOURCES[source]["revision"],
                "source_split": "train",
                "source_id": source_id,
                "question": question,
                "answer": answer,
                "image": {
                    "bytes": raw,
                    "path": f"{item_id}{Path(str(encoded_image.get('path') or '.jpg')).suffix}",
                },
                "expected_tool_need": metadata["expected_tool_need"],
                "task_family": metadata["task_family"],
                "operation": metadata["operation"],
                "image_width": metadata["image_width"],
                "image_height": metadata["image_height"],
                "image_sha256": metadata["image_sha256"],
                "image_phash": metadata["image_phash"],
                "question_sha256": metadata["question_sha256"],
                "selection_score": float(metadata["selection_score"]),
                "zoom_score": metadata["zoom_score"],
                "code_score": metadata["code_score"],
            }
    if len(output) != len(selected):
        raise RuntimeError(f"materialized {len(output)} of {len(selected)} replacements")
    return output


def main() -> int:
    args = parse_args()
    input_path = args.input_dir / "train.parquet"
    input_manifest_path = args.input_dir / "manifest.json"
    if not input_path.is_file() or not input_manifest_path.is_file():
        raise FileNotFoundError(f"missing v1 input under {args.input_dir}")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    old_rows = pq.read_table(input_path).to_pylist()
    if len(old_rows) != v1.EXPECTED_TOTAL:
        raise ValueError(f"v1 row count is {len(old_rows)}")
    bad_indices = [index for index, row in enumerate(old_rows) if COMPUTATION_AID_PATTERN.search(row["question"])]
    if not bad_indices:
        raise ValueError("no computation-aid prompts found; repair would be vacuous")
    bad_rows = [old_rows[index] for index in bad_indices]
    normalization_indices = [
        index
        for index in bad_indices
        if old_rows[index]["source"] == "geometry3k"
        and old_rows[index]["expected_tool_need"] == "direct"
        and old_rows[index]["task_family"] == "angle"
    ]
    if len(normalization_indices) != 1:
        raise RuntimeError(f"expected one saturated Geometry3K direct/angle row, found {normalization_indices}")
    replacement_bad_indices = [index for index in bad_indices if index not in set(normalization_indices)]
    desired_strata = Counter(
        (row["source"], row["expected_tool_need"], row["task_family"]) for row in bad_rows
    )

    rejects: Counter[str] = Counter()
    heldout_files = v1.load_heldout_manifest(args.heldout_manifest)
    evaluation, eval_files = v1.build_eval_index(args.cache_dir, True, rejects, heldout_files)
    if dict(evaluation.counts) != v1.EXPECTED_EVAL_INDEX_COUNTS:
        raise RuntimeError("held-out evaluation index differs from the frozen RL build")
    source_paths: dict[str, list[Path]] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for source, spec in v1.SOURCES.items():
        snapshot = v1.hub_snapshot(spec, args.cache_dir, True)
        paths = sorted(path for pattern in spec["patterns"] for path in snapshot.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no cached files for {source}")
        source_paths[source] = paths
        candidates[source] = v1.collect_candidates(source, paths, evaluation, v1.SEED, rejects)

    candidate_by_fingerprint = {
        row["fingerprint"]: row for source_rows in candidates.values() for row in source_rows
    }
    candidate_by_prefix = {fingerprint[:24]: row for fingerprint, row in candidate_by_fingerprint.items()}
    if len(candidate_by_prefix) != len(candidate_by_fingerprint):
        raise RuntimeError("candidate fingerprint prefix collision")

    # Start from every clean v1 assignment. The 13 hinted slots are left empty.
    # A standard augmenting-path matcher then fills them. It may move another
    # already-selected clean candidate and refill that candidate's old slot,
    # which preserves the exact stratum attached to every row position.
    bad_index_set = set(replacement_bad_indices)
    original_hint_index_set = set(bad_indices)
    assignment: dict[int, str] = {}
    owner: dict[str, int] = {}
    for index, row in enumerate(old_rows):
        if index in bad_index_set:
            continue
        prefix = row["item_id"].split(":", 1)[1]
        candidate = candidate_by_prefix.get(prefix)
        if candidate is None:
            raise RuntimeError(f"v1 row is absent from reconstructed candidates: {row['item_id']}")
        fingerprint = candidate["fingerprint"]
        if fingerprint in owner:
            raise RuntimeError(f"duplicate reconstructed candidate: {fingerprint}")
        assignment[index] = fingerprint
        owner[fingerprint] = index

    def slot_candidates(index: int) -> list[dict[str, Any]]:
        slot = old_rows[index]
        source = slot["source"]
        need = slot["expected_tool_need"]
        family = slot["task_family"]
        options = [
            row
            for row in candidates[source]
            if row["task_family"] == family
            and v1.eligible(row, need)
            and not COMPUTATION_AID_PATTERN.search(row["question"])
        ]
        options.sort(
            key=lambda row: (
                row["fingerprint"] in owner,
                -v1.category_score(row, need),
                v1.stable_rank(v1.SEED, f"v2-replacement:{source}:{need}:{family}", row["fingerprint"]),
            )
        )
        return options

    def set_assignment(index: int, fingerprint: str) -> None:
        old_fingerprint = assignment.get(index)
        if old_fingerprint is not None:
            owner.pop(old_fingerprint)
        assignment[index] = fingerprint
        owner[fingerprint] = index

    def fill_slot(index: int, seen_slots: set[int], seen_candidates: set[str]) -> bool:
        if index in seen_slots:
            return False
        seen_slots.add(index)
        for candidate in slot_candidates(index):
            fingerprint = candidate["fingerprint"]
            if fingerprint in seen_candidates:
                continue
            seen_candidates.add(fingerprint)
            current_owner = owner.get(fingerprint)
            if current_owner is None:
                set_assignment(index, fingerprint)
                return True
            if current_owner != index and fill_slot(current_owner, seen_slots, seen_candidates):
                set_assignment(index, fingerprint)
                return True
        return False

    for index in replacement_bad_indices:
        if not fill_slot(index, set(), set()):
            slot = old_rows[index]
            raise RuntimeError(
                f"no clean augmenting assignment for {slot['source']}/{slot['expected_tool_need']}/{slot['task_family']}"
            )
    if len(assignment) != v1.EXPECTED_TOTAL or len(owner) != v1.EXPECTED_TOTAL:
        raise RuntimeError(f"assignment is incomplete: slots={len(assignment)} candidates={len(owner)}")

    changed_indices = [
        index
        for index, row in enumerate(old_rows)
        if assignment[index][:24] != row["item_id"].split(":", 1)[1]
    ]
    replacement_metadata: dict[int, dict[str, Any]] = {}
    for index in changed_indices:
        slot = old_rows[index]
        metadata = dict(candidate_by_fingerprint[assignment[index]])
        metadata["expected_tool_need"] = slot["expected_tool_need"]
        metadata["selection_score"] = v1.category_score(metadata, slot["expected_tool_need"])
        replacement_metadata[index] = metadata

    materialized = materialize_replacements(list(replacement_metadata.values()), source_paths)
    rows = list(old_rows)
    replacement_audit: list[dict[str, Any]] = []
    for index, metadata in sorted(replacement_metadata.items()):
        old = rows[index]
        key = (old["source"], old["expected_tool_need"], old["task_family"])
        new_item_id = f"{metadata['source']}:{metadata['fingerprint'][:24]}"
        new = materialized[new_item_id]
        rows[index] = new
        replacement_audit.append(
            {
                "row_index": index,
                "source": key[0],
                "expected_tool_need": key[1],
                "task_family": key[2],
                "old_item_id": old["item_id"],
                "new_item_id": new["item_id"],
                "old_question": old["question"],
                "new_question": new["question"],
                "reason": "explicit_computation_aid" if index in original_hint_index_set else "stratum_preserving_swap",
            }
        )

    # The Geometry3K direct/angle candidate pool has no unused clean candidate
    # and no feasible stratum-preserving augmenting path. Remove only the
    # external-aid instruction; preserve the actual mathematical request,
    # image, answer, position, and stratum, and derive a fresh content ID.
    for index in normalization_indices:
        old = rows[index]
        new = dict(old)
        new_question = re.sub(r"^Use a calculator to find\b", "Find", old["question"], flags=re.I)
        if new_question == old["question"] or COMPUTATION_AID_PATTERN.search(new_question):
            raise RuntimeError("Geometry3K prompt normalization failed")
        normalized = v1.normalized_question(new_question)
        fingerprint = hashlib.sha256(
            f"{new['source_revision']}\0{new['source_id']}\0{normalized}\0{new['image_sha256']}".encode()
        ).hexdigest()
        new["question"] = new_question
        new["question_sha256"] = hashlib.sha256(normalized.encode()).hexdigest()
        new["item_id"] = f"{new['source']}:{fingerprint[:24]}"
        rows[index] = new
        replacement_audit.append(
            {
                "row_index": index,
                "source": new["source"],
                "expected_tool_need": new["expected_tool_need"],
                "task_family": new["task_family"],
                "old_item_id": old["item_id"],
                "new_item_id": new["item_id"],
                "old_question": old["question"],
                "new_question": new_question,
                "reason": "remove_external_calculator_instruction_from_saturated_stratum",
            }
        )

    v1.validate_output(rows)
    if any(COMPUTATION_AID_PATTERN.search(row["question"]) for row in rows):
        raise RuntimeError("computation-aid prompt survived replacement")
    old_strata = Counter((row["source"], row["expected_tool_need"], row["task_family"]) for row in old_rows)
    new_strata = Counter((row["source"], row["expected_tool_need"], row["task_family"]) for row in rows)
    if old_strata != new_strata:
        raise RuntimeError("source/need/task-family histogram changed")
    for batch in range(32):
        start = batch * 512
        stop = start + 512
        old_batch = Counter((row["source"], row["expected_tool_need"]) for row in old_rows[start:stop])
        new_batch = Counter((row["source"], row["expected_tool_need"]) for row in rows[start:stop])
        if old_batch != new_batch:
            raise RuntimeError(f"batch {batch} source/need histogram changed")

    temp_dir = args.output_dir.with_name(f".{args.output_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    output_path = temp_dir / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows),
        output_path,
        compression="zstd",
        compression_level=3,
        row_group_size=128,
    )
    old_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        **old_manifest,
        "dataset_name": "clean_rl_16k_v2",
        "created_for": "frozen two-tool RL comparison with canonical neutral prompts",
        "parent": {
            "train_parquet": str(input_path),
            "train_sha256": sha256(input_path),
            "manifest": str(input_manifest_path),
            "manifest_sha256": sha256(input_manifest_path),
        },
        "prompt_free": True,
        "computation_aid_prompt_count": 0,
        "replacement_count": len(replacement_audit),
        "source_prompt_normalization_count": len(normalization_indices),
        "replacement_strata": {"/".join(key): value for key, value in sorted(desired_strata.items())},
        "repair_rule": COMPUTATION_AID_PATTERN.pattern,
        "replacement_audit": replacement_audit,
        "train_parquet": {
            "path": "train.parquet",
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "ordered_item_id_sha256": hashlib.sha256("\n".join(row["item_id"] for row in rows).encode()).hexdigest(),
    }
    (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (temp_dir / "README.md").write_text(
        "# clean_rl_16k_v2\n\n"
        "The formal 16,384-row RL source pool. It preserves every v1 source/need/task-family stratum and every "
        "512-row batch composition while replacing source questions that explicitly offer a calculator/computer "
        "as a computation aid. No system prompt or tool schema is stored here.\n",
        encoding="utf-8",
    )
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    temp_dir.rename(args.output_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output_dir),
                "rows": len(rows),
                "replacements": len(replacement_audit),
                "sha256": manifest["train_parquet"]["sha256"],
                "evaluation_files": len(eval_files),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
