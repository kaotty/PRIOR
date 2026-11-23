#!/usr/bin/env python3
"""Convert the frozen clean RL selection into the canonical parquet for all arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from verl.utils import hf_processor
from verl.utils.dataset.vision_utils import process_image


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "rl/config"

LEGACY_SYSTEM_PROMPT = (
    "You are a careful multimodal reasoner. Solve the problem from the image and question. "
    "If tools are available, use them only when they are genuinely helpful. The code tool cannot inspect the "
    "original image; it can only compute from quantities and relations you provide."
)
CANONICAL_SYSTEM_PROMPT = (
    "You are a careful multimodal reasoner. Solve the problem from the image and question. "
    "If tools are available, use them only when they are genuinely helpful. Use image_zoom_in_tool only for "
    "closer visual inspection. Use code_interpreter only for genuine numerical or symbolic computation over "
    "quantities and relations you have already extracted; it cannot inspect the original image."
)
VQA_SUFFIX = "\nAnswer briefly. Put only the final answer inside <answer> </answer>."
RULE_SUFFIX = "\nPut the final answer inside \\boxed{}."
DEFAULT_MAX_IMAGE_PIXELS = 1024 * 28 * 28

SOURCE_CONFIG = {
    "vstar": {"data_source": "vstar", "ability": "perception", "answer_format": "answer"},
    "mmk12": {"data_source": "mmk12", "ability": "math", "answer_format": "boxed"},
    "chartqa": {"data_source": "HuggingFaceM4/ChartQA", "ability": "chart", "answer_format": "boxed"},
    "geometry3k": {"data_source": "hiyouga/geometry3k", "ability": "math", "answer_format": "boxed"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-image-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--system-prompt-profile",
        choices=("legacy", "canonical"),
        default="legacy",
        help="Use canonical to match the clean SFT and evaluation system prompt exactly.",
    )
    parser.add_argument("--skip-prompt-length-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_order(rows: list[dict[str, Any]], batch_size: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(rows) % batch_size:
        raise ValueError(f"{len(rows)} rows are not divisible by batch_size={batch_size}")
    steps = len(rows) // batch_size
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["expected_tool_need"])].append(row)

    per_batch: dict[tuple[str, str], int] = {}
    for key, group in groups.items():
        if len(group) % steps:
            raise ValueError(f"group {key} has {len(group)} rows, not divisible by {steps} steps")
        per_batch[key] = len(group) // steps

    ordered: list[dict[str, Any]] = []
    keys = sorted(groups)
    for step in range(steps):
        for key in keys:
            count = per_batch[key]
            ordered.extend(groups[key][step * count : (step + 1) * count])
    return ordered, {f"{source}/{need}": count for (source, need), count in sorted(per_batch.items())}


def canonical_row(row: dict[str, Any], max_image_pixels: int, system_prompt: str) -> dict[str, Any]:
    source = str(row["source"])
    config = SOURCE_CONFIG[source]
    tool_image = dict(row["image"])
    model_image = dict(tool_image)
    # The policy sees at most 1,024 visual tokens, while Zoom retains the
    # original bytes. Small images are unchanged; only oversized images shrink.
    model_image["max_pixels"] = max_image_pixels
    suffix = VQA_SUFFIX if config["answer_format"] == "answer" else RULE_SUFFIX
    item_id = str(row["item_id"])
    return {
        "data_source": config["data_source"],
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<image>\n{row['question']}{suffix}"},
        ],
        "images": [model_image],
        "ability": config["ability"],
        "reward_model": {"style": "rule", "ground_truth": str(row["answer"])},
        "extra_info": {
            "split": "train",
            "index": item_id,
            "item_id": item_id,
            "row_id": item_id,
            "replicate_id": 0,
            "source": source,
            "source_repo": row["source_repo"],
            "source_revision": row["source_revision"],
            "source_id": row["source_id"],
            "question": row["question"],
            "answer": str(row["answer"]),
            "image_sha256": row["image_sha256"],
            "expected_tool_need": row["expected_tool_need"],
            "utility_stratum": row["expected_tool_need"],
            "utility_label_basis": "static_structure_heuristic_v1",
            "cf_eligible_tools": [],
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "image_zoom_in_tool": {"create_kwargs": {"image": tool_image}},
                "code_interpreter": {"create_kwargs": {"image": None}},
            },
        },
    }


def messages_and_images(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    images = row["images"]
    messages = copy.deepcopy(row["prompt"])
    processed_images: list[Any] = []
    image_offset = 0
    for message in messages:
        parts: list[dict[str, Any]] = []
        for segment in [part for part in re.split("(<image>)", message["content"]) if part]:
            if segment == "<image>":
                image = process_image(copy.deepcopy(images[image_offset]), image_patch_size=14)
                image_offset += 1
                processed_images.append(image)
                parts.append({"type": "image", "image": image})
            else:
                parts.append({"type": "text", "text": segment})
        message["content"] = parts
    if image_offset != len(images):
        raise ValueError("image placeholder count does not match images")
    return messages, processed_images


def prompt_lengths(rows: list[dict[str, Any]], model: Path) -> list[int]:
    config = OmegaConf.load(CONFIG_DIR / "grpo.yaml")
    tools = OmegaConf.to_container(
        OmegaConf.load(CONFIG_DIR / "tools/both.yaml"), resolve=True
    )["tools"]
    schemas = [tool["tool_schema"] for tool in tools]
    processor = hf_processor(str(model), trust_remote_code=False, use_fast=True)
    processor.chat_template = config.actor_rollout_ref.model.custom_chat_template
    lengths: list[int] = []
    for index, row in enumerate(rows, start=1):
        messages, images = messages_and_images(row)
        prompt = processor.apply_chat_template(messages, tools=schemas, add_generation_prompt=True, tokenize=False)
        encoded = processor(text=[prompt], images=images, return_attention_mask=False)
        lengths.append(len(encoded["input_ids"][0]))
        if index % 1024 == 0:
            print(f"      prompt lengths checked: {index}/{len(rows)}", flush=True)
    return lengths


def positive_reward_check(rows: list[dict[str, Any]]) -> None:
    from verl.utils.reward_score.frozen_rl_mixed import compute_score

    for row in rows:
        answer = row["reward_model"]["ground_truth"]
        if row["data_source"] == "vstar":
            solution = f"<answer>{answer}</answer>"
        else:
            solution = f"\\boxed{{{answer}}}"
        score = compute_score(
            solution,
            answer,
            data_source=row["data_source"],
            extra_info=row["extra_info"],
        )
        if float(score["score"]) != 1.0:
            raise ValueError(f"positive reward check failed for {row['extra_info']['item_id']}")


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, int(fraction * len(values)))]


def main() -> None:
    args = parse_args()
    train_input = args.input_dir / "train.parquet"
    clean_manifest = args.input_dir / "manifest.json"
    if not train_input.is_file() or not clean_manifest.is_file():
        raise FileNotFoundError(f"missing clean dataset under {args.input_dir}")
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output_dir}; pass --overwrite explicitly")

    print("[1/4] Reading clean rows and creating balanced batches", flush=True)
    clean_rows = pq.read_table(train_input).to_pylist()
    ordered, rows_per_batch = balanced_order(clean_rows, args.batch_size)
    system_prompt = CANONICAL_SYSTEM_PROMPT if args.system_prompt_profile == "canonical" else LEGACY_SYSTEM_PROMPT
    rows = [canonical_row(row, args.max_image_pixels, system_prompt) for row in ordered]
    positive_reward_check(rows)

    lengths: list[int] = []
    if not args.skip_prompt_length_check:
        print("[2/4] Checking exact prompt lengths with both tool schemas", flush=True)
        lengths = prompt_lengths(rows, args.model)
        over = [length for length in lengths if length > args.max_prompt_length]
        if over:
            raise ValueError(
                f"{len(over)} prompts exceed max_prompt_length={args.max_prompt_length}; longest={max(over)}"
            )
        for row, length in zip(rows, lengths, strict=True):
            row["extra_info"]["prompt_tokens_both_tools"] = length
    else:
        print("[2/4] Prompt-length check skipped by request")

    print("[3/4] Writing canonical parquet")
    temp_dir = args.output_dir.with_name(f".{args.output_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    train_output = temp_dir / "train.parquet"
    loader_val_output = temp_dir / "loader_val.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train_output, compression="zstd", compression_level=3, row_group_size=128)
    # PPO constructs a validation loader even when validation is disabled. This
    # one-batch file exists only for loader initialization and must not be cited.
    pq.write_table(pa.Table.from_pylist(rows[: args.batch_size]), loader_val_output, compression="zstd", compression_level=3)

    source_counts = Counter(row["extra_info"]["source"] for row in rows)
    need_counts = Counter(row["extra_info"]["expected_tool_need"] for row in rows)
    manifest = {
        "name": args.output_dir.name,
        "formal_eligible": not args.skip_prompt_length_check,
        "shared_by_all_arms": True,
        "row_count": len(rows),
        "batch_size": args.batch_size,
        "steps_per_epoch": len(rows) // args.batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_image_pixels": args.max_image_pixels,
        "max_visual_tokens_per_image": args.max_image_pixels // (28 * 28),
        "source_counts": dict(sorted(source_counts.items())),
        "expected_tool_need_counts": dict(sorted(need_counts.items())),
        "rows_per_batch": rows_per_batch,
        "system_prompt_profile": args.system_prompt_profile,
        "system_prompt": system_prompt,
        "prompt_length_both_tools": (
            {"min": min(lengths), "median": percentile(lengths, 0.5), "p99": percentile(lengths, 0.99), "max": max(lengths)}
            if lengths
            else None
        ),
        "reward_positive_control_failures": 0,
        "loader_val_note": "training duplicate used only because PPO requires a val loader; validation stays disabled",
        "input": {
            "train.parquet": {"path": str(train_input), "sha256": sha256_file(train_input)},
            "manifest.json": {"path": str(clean_manifest), "sha256": sha256_file(clean_manifest)},
        },
        "outputs": {
            "train.parquet": {"sha256": sha256_file(train_output)},
            "loader_val.parquet": {"sha256": sha256_file(loader_val_output)},
        },
    }
    (temp_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    temp_dir.rename(args.output_dir)
    print("[4/4] Prepared RL dataset")
    print(json.dumps({key: manifest[key] for key in ("row_count", "steps_per_epoch", "source_counts", "prompt_length_both_tools")}, indent=2))


if __name__ == "__main__":
    main()
