#!/usr/bin/env python3
"""Validate the canonical-prompt, computation-aid-free formal RL 16K v2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from jinja2 import Environment
from omegaconf import OmegaConf

from rl.format import CANONICAL_SYSTEM_PROMPT, RULE_SUFFIX, VQA_SUFFIX
from rl.repair import COMPUTATION_AID_PATTERN
from rl.restore import load_selection


EXPECTED_ROWS = 16_384


EXPECTED_BATCH_SIZE = 512


EXPECTED_STEPS = 32


EXPECTED_SOURCES = {"vstar": 6_400, "mmk12": 6_656, "chartqa": 1_792, "geometry3k": 1_536}


EXPECTED_NEEDS = {"zoom_favored": 6_144, "code_favored": 4_480, "both_favored": 3_072, "direct": 2_688}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(data_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = data_dir / "dataset_manifest.json"
    train_path = data_dir / "train.parquet"
    loader_val_path = data_dir / "loader_val.parquet"
    for path in (manifest_path, train_path, loader_val_path):
        if not path.is_file():
            errors.append(f"missing {path}")
    if errors:
        return errors

    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "formal_eligible": True,
        "shared_by_all_arms": True,
        "row_count": EXPECTED_ROWS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "steps_per_epoch": EXPECTED_STEPS,
        "max_prompt_length": 2048,
        "max_visual_tokens_per_image": 1024,
        "reward_positive_control_failures": 0,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            errors.append(f"manifest.{key}: expected {expected!r}, got {manifest.get(key)!r}")
    if (manifest.get("prompt_length_both_tools") or {}).get("max", 10**9) > 2048:
        errors.append("Both-arm prompt length exceeds 2048")

    columns = ["data_source", "prompt", "images", "reward_model", "extra_info"]
    parquet = pq.ParquetFile(train_path)
    if parquet.metadata.num_rows != EXPECTED_ROWS:
        errors.append(f"train rows: expected {EXPECTED_ROWS}, got {parquet.metadata.num_rows}")
        return errors

    item_ids: set[str] = set()
    ordered_item_ids: list[str] = []
    frozen = load_selection()
    image_hashes: set[str] = set()
    source_counts: Counter[str] = Counter()
    need_counts: Counter[str] = Counter()
    batch_counts: list[Counter[str]] = [Counter() for _ in range(EXPECTED_STEPS)]
    row_number = 0
    source_alias = {
        "vstar": "vstar",
        "mmk12": "mmk12",
        "HuggingFaceM4/ChartQA": "chartqa",
        "hiyouga/geometry3k": "geometry3k",
    }
    for batch in parquet.iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            extra = row.get("extra_info") or {}
            item_id = str(extra.get("item_id") or "")
            image_hash = str(extra.get("image_sha256") or "")
            source = source_alias.get(str(row.get("data_source")), "unknown")
            need = str(extra.get("expected_tool_need") or "")
            if not item_id or item_id in item_ids:
                errors.append(f"row {row_number}: empty or duplicate item_id")
            if not image_hash or image_hash in image_hashes:
                errors.append(f"row {row_number}: empty or duplicate image hash")
            item_ids.add(item_id)
            ordered_item_ids.append(item_id)
            image_hashes.add(image_hash)
            source_counts[source] += 1
            need_counts[need] += 1
            batch_counts[row_number // EXPECTED_BATCH_SIZE][f"{source}/{need}"] += 1
            if len(row.get("images") or []) != 1 or not row.get("prompt"):
                errors.append(f"row {row_number}: missing prompt or image")
            if not str((row.get("reward_model") or {}).get("ground_truth") or "").strip():
                errors.append(f"row {row_number}: missing ground truth")
            if set((extra.get("tools_kwargs") or {}).keys()) != {"image_zoom_in_tool", "code_interpreter"}:
                errors.append(f"row {row_number}: invalid tools_kwargs")
            if int(extra.get("prompt_tokens_both_tools", 10**9)) > 2048:
                errors.append(f"row {row_number}: prompt over 2048")
            selected = frozen["rows"].get(item_id)
            if selected is None:
                errors.append(f"row {row_number}: item is absent from the frozen selection")
            else:
                question = str(extra.get("question") or "")
                answer = str((row.get("reward_model") or {}).get("ground_truth") or "")
                if hashlib.sha256(question.encode()).hexdigest() != selected["question_text_sha256"]:
                    errors.append(f"row {row_number}: frozen question differs")
                if hashlib.sha256(answer.encode()).hexdigest() != selected["answer_sha256"]:
                    errors.append(f"row {row_number}: frozen answer differs")
                images = row.get("images") or []
                raw = images[0].get("bytes") if len(images) == 1 else None
                if raw is None or hashlib.sha256(raw).hexdigest() != selected["image_sha256"]:
                    errors.append(f"row {row_number}: frozen image differs")
                if source != selected["source"] or need != selected["expected_tool_need"]:
                    errors.append(f"row {row_number}: frozen source/need metadata differs")
                suffix = VQA_SUFFIX if source == "vstar" else RULE_SUFFIX
                expected_prompt = [
                    {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>\n{question}{suffix}"},
                ]
                if row.get("prompt") != expected_prompt:
                    errors.append(f"row {row_number}: canonical prompt differs")
            row_number += 1

    if dict(source_counts) != EXPECTED_SOURCES:
        errors.append(f"source counts: expected {EXPECTED_SOURCES}, got {dict(source_counts)}")
    if dict(need_counts) != EXPECTED_NEEDS:
        errors.append(f"need counts: expected {EXPECTED_NEEDS}, got {dict(need_counts)}")
    ordered_digest = hashlib.sha256("\n".join(ordered_item_ids).encode()).hexdigest()
    if ordered_digest != frozen["formatted_ordered_item_id_sha256"]:
        errors.append("formatted item order differs from the frozen 32-step dataset")
    expected_batch = Counter(manifest.get("rows_per_batch") or {})
    for index, counts in enumerate(batch_counts):
        if counts != expected_batch:
            errors.append(f"batch {index}: expected {dict(expected_batch)}, got {dict(counts)}")

    if pq.ParquetFile(loader_val_path).metadata.num_rows != EXPECTED_BATCH_SIZE:
        errors.append("loader_val.parquet must contain exactly one 512-row batch")
    for name, path in (("train.parquet", train_path), ("loader_val.parquet", loader_val_path)):
        expected = (manifest.get("outputs", {}).get(name) or {}).get("sha256")
        actual = sha256_file(path)
        if expected != actual:
            errors.append(f"{name} SHA-256 mismatch")
    return errors


USER_TOOL_HINT = re.compile(
    r"image_zoom_in_tool|code_interpreter|expected_tool_need|zoom_favored|code_favored|both_favored|"
    r"\b(?:use|call|invoke)\b.{0,40}\b(?:zoom|crop|python|code interpreter|tool)\b",
    re.I,
)
LABEL_LEAK = re.compile(r"expected_tool_need|utility_stratum|zoom_favored|code_favored|both_favored", re.I)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=Path,
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root containing formal/ and sft/; does not change the frozen data contract.",
    )
    parser.add_argument(
        "--report-path", type=Path,
        help="Write the audit outside the frozen dataset (default: data_dir/semantic_audit.json).",
    )
    parser.add_argument("--check-only", action="store_true", help="Print the report without writing any files.")
    args = parser.parse_args()
    errors = validate(args.data_dir)
    train_path = args.data_dir / "train.parquet"
    clean_v1 = args.data_root / "formal/clean_rl_16k_v1/train.parquet"
    clean_v2 = args.data_root / "formal/clean_rl_16k_v2/train.parquet"
    sft_neutral = args.data_root / "sft/core_clean_prompt_v1/no_tool_25__frequency_medium/train.parquet"
    for path in (train_path, clean_v1, clean_v2, sft_neutral):
        if not path.is_file():
            errors.append(f"missing semantic-audit input: {path}")
    if errors:
        raise SystemExit("RL 16K v2 validation failed:\n  - " + "\n  - ".join(errors))

    sft_system = pq.read_table(sft_neutral, columns=["messages"]).slice(0, 1).to_pylist()[0]["messages"][0]["content"]
    if sft_system != CANONICAL_SYSTEM_PROMPT:
        errors.append("canonical RL system prompt does not exactly match clean SFT")

    project = Path(__file__).resolve().parents[1]
    sft_template_path = project / "sft/chat_template.jinja"
    rl_config_path = project / "rl/config/grpo.yaml"
    rl_tools_path = project / "rl/config/tools/both.yaml"
    sft_template = sft_template_path.read_text(encoding="utf-8")
    rl_template = str(OmegaConf.load(rl_config_path).actor_rollout_ref.model.custom_chat_template)
    sft_tools = pq.read_table(sft_neutral, columns=["tools"]).slice(0, 1).to_pylist()[0]["tools"]
    rl_tools = [
        row["tool_schema"]
        for row in OmegaConf.to_container(OmegaConf.load(rl_tools_path), resolve=True)["tools"]
    ]

    def drop_none(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: drop_none(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [drop_none(item) for item in value]
        return value

    tool_schema_semantic_match = drop_none(sft_tools) == rl_tools
    if not tool_schema_semantic_match:
        errors.append("RL executable tool schemas do not match clean SFT schemas after Arrow-null removal")
    synthetic_messages = [
        {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
        {"role": "user", "content": "<image>\nSAME USER"},
        {
            "role": "assistant",
            "content": "I will calculate.",
            "tool_calls": [
                {
                    "function": {
                        "name": "code_interpreter",
                        "arguments": '{"code":"print(6*7)"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "42"},
        {"role": "assistant", "content": "\\boxed{42}", "tool_calls": []},
    ]
    environment = Environment()
    sft_render = environment.from_string(sft_template).render(
        messages=synthetic_messages,
        tools=sft_tools,
        add_generation_prompt=False,
        add_vision_id=False,
    )
    rl_render = environment.from_string(rl_template).render(
        messages=synthetic_messages,
        tools=rl_tools,
        add_generation_prompt=False,
        add_vision_id=False,
    )
    runtime_render_exact_sft_match = sft_render == rl_render
    if not runtime_render_exact_sft_match:
        errors.append("RL multi-turn runtime rendering does not exactly match clean SFT rendering")

    source_counts: Counter[str] = Counter()
    need_counts: Counter[str] = Counter()
    systems: Counter[str] = Counter()
    user_hint_rows: list[dict[str, str]] = []
    label_leak_rows: list[str] = []
    rows = 0
    parquet = pq.ParquetFile(train_path)
    for batch in parquet.iter_batches(batch_size=256, columns=["prompt", "extra_info"]):
        for row in batch.to_pylist():
            prompt = row["prompt"]
            extra = row["extra_info"]
            system_messages = [message["content"] for message in prompt if message["role"] == "system"]
            user_messages = [message["content"] for message in prompt if message["role"] == "user"]
            if len(system_messages) != 1 or len(user_messages) != 1 or len(prompt) != 2:
                errors.append(f"{extra.get('item_id')}: prompt roles are not exactly system+user")
                continue
            system = system_messages[0]
            user = user_messages[0]
            systems[system] += 1
            source_counts[str(extra["source"])] += 1
            need_counts[str(extra["expected_tool_need"])] += 1
            if COMPUTATION_AID_PATTERN.search(user) or USER_TOOL_HINT.search(user):
                user_hint_rows.append(
                    {
                        "item_id": str(extra["item_id"]),
                        "source": str(extra["source"]),
                        "need": str(extra["expected_tool_need"]),
                        "user": user,
                    }
                )
            if LABEL_LEAK.search(user):
                label_leak_rows.append(str(extra["item_id"]))
            rows += 1

    if rows != EXPECTED_ROWS:
        errors.append(f"semantic scan rows: expected={EXPECTED_ROWS}, actual={rows}")
    if systems != {CANONICAL_SYSTEM_PROMPT: EXPECTED_ROWS}:
        errors.append(f"system prompts are not exactly one canonical prompt: {dict(systems)}")
    if dict(source_counts) != EXPECTED_SOURCES:
        errors.append(f"semantic source counts mismatch: {dict(source_counts)}")
    if dict(need_counts) != EXPECTED_NEEDS:
        errors.append(f"semantic need counts mismatch: {dict(need_counts)}")
    if user_hint_rows:
        errors.append(f"{len(user_hint_rows)} user prompts contain a tool/computation-aid hint")
    if label_leak_rows:
        errors.append(f"{len(label_leak_rows)} user prompts leak expected-tool labels")

    def clean_histogram(path: Path) -> Counter[tuple[str, str, str]]:
        histogram: Counter[tuple[str, str, str]] = Counter()
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=256,
            columns=["source", "expected_tool_need", "task_family"],
        ):
            for row in batch.to_pylist():
                histogram[(row["source"], row["expected_tool_need"], row["task_family"])] += 1
        return histogram

    if clean_histogram(clean_v1) != clean_histogram(clean_v2):
        errors.append("clean v2 changed the source/need/task-family histogram")

    manifest = json.loads((args.data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "tir-rl-16k-v2-semantic-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "rows": rows,
        "source_counts": dict(source_counts),
        "expected_tool_need_counts": dict(need_counts),
        "system_prompt_exact_sft_match": sft_system == CANONICAL_SYSTEM_PROMPT,
        "tool_schema_semantic_match": tool_schema_semantic_match,
        "runtime_render_exact_sft_match": runtime_render_exact_sft_match,
        "unique_system_prompts": len(systems),
        "user_tool_hint_rows": user_hint_rows,
        "expected_tool_label_leak_rows": label_leak_rows,
        "prompt_length_both_tools": manifest.get("prompt_length_both_tools"),
        "source_need_task_family_histogram_preserved": clean_histogram(clean_v1) == clean_histogram(clean_v2),
        "errors": errors,
    }
    if not args.check_only:
        report_path = args.report_path or args.data_dir / "semantic_audit.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("RL 16K v2 validation failed:\n  - " + "\n  - ".join(errors[:50]))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
