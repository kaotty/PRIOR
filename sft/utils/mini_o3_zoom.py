#!/usr/bin/env python3
"""Convert Mini-o3 cold-start crop trajectories to the formal Zoom schema.

Mini-o3 uses normalized boxes that may refer to an earlier crop.  The formal
``image_zoom_in_tool`` always crops the original image with pixel coordinates,
so this converter composes nested boxes back into the original frame and
re-bakes every observation with the runtime crop implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .records import _normalize_messages, _output_schema
from .zoom_crop import bake_zoom, zoom_response_text
from .tool_schema import FORMAL_TWO_TOOL_SCHEMAS
from .parse_traces import normalize_final_answer


GROUNDING_RE = re.compile(r"<grounding>\s*(.*?)\s*</grounding>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*.+?\s*</answer>", re.DOTALL | re.IGNORECASE)
BOX_RE = re.compile(
    r"\[\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\]"
)
OBSERVATION_RE = re.compile(r"`?observation_\d+`?", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    source_path: Path
    row_index: int
    sample_index: str
    rollout_index: str
    call_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-dir",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    return parser.parse_args()


def _density(call_count: int) -> str:
    if call_count == 1:
        return "low"
    if call_count <= 3:
        return "medium"
    return "high"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_conversations(value: str) -> list[dict[str, str]]:
    conversations = json.loads(value)
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("empty_conversations")
    return conversations


def _calls(conversations: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in conversations:
        if message.get("from") != "gpt":
            continue
        matches = GROUNDING_RE.findall(message.get("value") or "")
        if len(matches) > 1:
            raise ValueError("multiple_calls_in_one_turn")
        if matches:
            call = json.loads(matches[0])
            if not isinstance(call, dict):
                raise ValueError("invalid_call_json")
            calls.append(call)
    return calls


def _valid_metadata(row: dict[str, Any]) -> tuple[int, str]:
    try:
        conversations = _parse_conversations(row["conversations"])
        calls = _calls(conversations)
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        return 0, str(exc)
    if not calls:
        return 0, "no_zoom_call"
    if len(row.get("image_names") or []) != len(calls) + 1:
        return 0, "image_call_mismatch"
    if not any(
        ANSWER_RE.search(message.get("value") or "")
        for message in conversations
        if message.get("from") == "gpt"
    ):
        return 0, "missing_final_answer"
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for call_index, call in enumerate(calls, start=1):
        bbox = call.get("bbox_2d")
        source = str(call.get("source") or "")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int | float) and 0 <= float(value) <= 1 for value in bbox)
            or not (float(bbox[0]) < float(bbox[2]) and float(bbox[1]) < float(bbox[3]))
        ):
            return 0, "invalid_normalized_bbox"
        valid_sources = {"original_image"} | {f"observation_{index}" for index in range(1, call_index)}
        if source not in valid_sources:
            return 0, "unknown_or_forward_source"
        key = (source, tuple(round(float(value), 8) for value in bbox))
        if key in seen:
            return 0, "duplicate_exact_call"
        seen.add(key)
    return len(calls), "ok"


def _rewrite_reasoning(content: str, normalized_bbox: Sequence[float], pixel_bbox: Sequence[int]) -> str:
    content = GROUNDING_RE.sub("", content)

    def replace_box(match: re.Match[str]) -> str:
        values = [float(match.group(index)) for index in range(1, 5)]
        if all(abs(left - right) <= 1e-6 for left, right in zip(values, normalized_bbox, strict=True)):
            return str(list(pixel_bbox))
        return match.group(0)

    content = BOX_RE.sub(replace_box, content)
    content = OBSERVATION_RE.sub("the previous zoom view", content)
    return normalize_final_answer(content).strip()


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _question_fingerprint(question: str, image_bytes: bytes) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L").resize((17, 16))
        pixels = gray.tobytes()
    bits = [pixels[y * 17 + x] > pixels[y * 17 + x + 1] for y in range(16) for x in range(16)]
    difference_hash = bytes(
        sum((1 << (7 - offset)) if bits[index + offset] else 0 for offset in range(8))
        for index in range(0, 256, 8)
    )
    return hashlib.sha256(normalized.encode() + b"\0" + difference_hash).hexdigest()


def _extra_info(candidate: Candidate, fingerprint: str) -> dict[str, Any]:
    source = {
        "dataset": "Mini-o3/Mini-o3-Coldstart-Dataset",
        "license": "CC-BY-NC-4.0",
        "sample_index": candidate.sample_index,
        "rollout_index": candidate.rollout_index,
        "question_image_fingerprint": fingerprint,
        "nested_boxes_composed_to_original": True,
    }
    return {
        "sample_id": f"mini_o3/{candidate.sample_index}",
        "source_parquet": str(candidate.source_path),
        "source_row_index": candidate.row_index,
        "source_group": "mini_o3.zoom",
        "domain_group": "unlabeled",
        "pool_role": "call_density_candidate_core",
        "shared_anchor_type": "",
        "tool_type": "zoom",
        "difficulty": "",
        "density": _density(candidate.call_count),
        "no_tool_correct_count": None,
        "oracle_tool_correct_count": None,
        "no_tool_success_rate": None,
        "oracle_tool_success_rate": None,
        "oracle_tool_gain": None,
        "actual_tool_calls": candidate.call_count,
        "effective_tool_calls": None,
        "assistant_loss_tokens": None,
        "experimental_arm": "",
        "selection_seed": 42,
        "spec_sha256": "",
        "labels_sha256": "",
        "source_extra_json": json.dumps(source, sort_keys=True, ensure_ascii=False),
    }


def _convert(row: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    conversations = _parse_conversations(row["conversations"])
    original_record = (row.get("images") or [None])[0]
    if not isinstance(original_record, dict) or original_record.get("bytes") is None:
        raise ValueError("missing_original_image")
    original_bytes = bytes(original_record["bytes"])
    with Image.open(io.BytesIO(original_bytes)) as loaded:
        original = loaded.convert("RGB")
    width, height = original.size

    humans = [message for message in conversations if message.get("from") == "human"]
    if not humans or (humans[0].get("value") or "").count("<image>") != 1:
        raise ValueError("invalid_user_image_prompt")
    question = (humans[0].get("value") or "").replace("<image>", "").strip()
    if not question:
        raise ValueError("empty_question")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"<image>\n{question}"},
    ]
    images = [{"bytes": original_bytes, "path": ""}]
    regions: dict[str, list[float]] = {"original_image": [0.0, 0.0, float(width), float(height)]}
    seen_pixel_boxes: set[tuple[int, int, int, int]] = set()
    emitted_calls = 0
    terminal_seen = False

    for message in conversations:
        if message.get("from") != "gpt":
            continue
        content = message.get("value") or ""
        matches = GROUNDING_RE.findall(content)
        if not matches:
            if not ANSWER_RE.search(content):
                raise ValueError("nonterminal_assistant_without_call")
            if terminal_seen:
                raise ValueError("multiple_terminal_answers")
            terminal_seen = True
            messages.append({"role": "assistant", "content": normalize_final_answer(content).strip()})
            continue
        if terminal_seen or len(matches) != 1:
            raise ValueError("invalid_call_order")
        call = json.loads(matches[0])
        normalized_bbox = [float(value) for value in call["bbox_2d"]]
        source_name = str(call["source"])
        if source_name not in regions:
            raise ValueError("unresolved_source_observation")
        source = regions[source_name]
        source_width = source[2] - source[0]
        source_height = source[3] - source[1]
        absolute = [
            source[0] + normalized_bbox[0] * source_width,
            source[1] + normalized_bbox[1] * source_height,
            source[0] + normalized_bbox[2] * source_width,
            source[1] + normalized_bbox[3] * source_height,
        ]
        baked = bake_zoom(original, absolute)
        if baked is None:
            raise ValueError("runtime_bbox_rejected")
        crop, pixel_bbox = baked
        pixel_key = tuple(pixel_bbox)
        if pixel_key in seen_pixel_boxes:
            raise ValueError("duplicate_composed_bbox")
        seen_pixel_boxes.add(pixel_key)
        emitted_calls += 1
        regions[f"observation_{emitted_calls}"] = [float(value) for value in pixel_bbox]

        assistant = {
            "role": "assistant",
            "content": _rewrite_reasoning(content, normalized_bbox, pixel_bbox),
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "image_zoom_in_tool",
                        "arguments": json.dumps({"bbox_2d": pixel_bbox}, ensure_ascii=False),
                    },
                }
            ],
        }
        messages.append(assistant)
        messages.append({"role": "tool", "content": f"<image>\n{zoom_response_text(pixel_bbox)}"})
        images.append({"bytes": _jpeg_bytes(crop), "path": ""})

    if not terminal_seen or emitted_calls != candidate.call_count:
        raise ValueError("incomplete_trajectory")
    if sum(message["content"].count("<image>") for message in messages) != len(images):
        raise ValueError("image_placeholder_mismatch")
    fingerprint = _question_fingerprint(question, original_bytes)
    return {
        "messages": _normalize_messages(messages),
        "images": images,
        "tools": copy.deepcopy(FORMAL_TWO_TOOL_SCHEMAS),
        "data_source": "mini_o3.zoom",
        "ability": "perception",
        "extra_info": _extra_info(candidate, fingerprint),
    }


def _inventory(paths: Sequence[Path]) -> tuple[list[Candidate], dict[str, Any]]:
    from sft.frozen import source_selection

    by_name = {path.name: path for path in paths}
    selected = []
    for record in source_selection("mini_o3_zoom"):
        selected.append(Candidate(
            source_path=by_name[record["source_file"]], row_index=record["source_row_index"],
            sample_index=record["sample_index"], rollout_index=record["rollout_index"],
            call_count=record["call_count"],
        ))
    return selected, {"unique_source_samples": len(selected),
                      "selection_rule": "frozen raw record locators; independent of local filesystem path"}


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output = Path(args.output).resolve()
    paths = sorted(input_dir.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Mini-o3 parquet shards under {input_dir}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    selected, report = _inventory(paths)
    wanted = {(candidate.source_path, candidate.row_index): candidate for candidate in selected}
    output.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = pq.ParquetWriter(output, _output_schema(), compression="zstd")
    converted_hist: Counter[str] = Counter()
    conversion_rejected: Counter[str] = Counter()
    fingerprints: set[str] = set()
    written = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            offset = 0
            for batch in parquet.iter_batches(batch_size=4):
                output_rows = []
                for local_index, row in enumerate(batch.to_pylist()):
                    candidate = wanted.get((path, offset + local_index))
                    if candidate is None:
                        continue
                    try:
                        converted = _convert(row, candidate)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
                        conversion_rejected[str(exc)] += 1
                        continue
                    fingerprint = json.loads(converted["extra_info"]["source_extra_json"])[
                        "question_image_fingerprint"
                    ]
                    if fingerprint in fingerprints:
                        conversion_rejected["duplicate_question_and_image"] += 1
                        continue
                    fingerprints.add(fingerprint)
                    converted_hist[_density(candidate.call_count)] += 1
                    output_rows.append(converted)
                if output_rows:
                    writer.write_table(pa.Table.from_pylist(output_rows, schema=_output_schema()))
                    written += len(output_rows)
                offset += batch.num_rows
    finally:
        writer.close()

    report.update(
        {
            "converted_rows": written,
            "converted_density_counts": dict(converted_hist),
            "conversion_rejected": dict(conversion_rejected),
            "output": str(output),
            "output_sha256": _sha256(output),
        }
    )
    report_path = output.with_name("build_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
