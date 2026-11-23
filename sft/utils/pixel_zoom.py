#!/usr/bin/env python3
"""Convert PixelReasoner's image-only crop trajectories to formal Zoom SFT rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .records import _normalize_messages, _output_schema
from .zoom_crop import bake_zoom, zoom_response_text
from .tool_schema import FORMAL_TWO_TOOL_SCHEMAS
from .parse_traces import normalize_final_answer


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    row_index: int
    qid: str
    call_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-json",
        required=True,
    )
    parser.add_argument(
        "--images-zip",
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


def _density(call_count: int) -> str:
    if call_count == 1:
        return "low"
    if call_count <= 3:
        return "medium"
    return "high"


def _content_text(message: dict[str, Any]) -> str:
    return "\n".join(str(item.get("text") or "") for item in message.get("content") or [] if item.get("text"))


def _content_media(message: dict[str, Any], key: str) -> list[str]:
    return [str(item[key]) for item in message.get("content") or [] if item.get(key)]


def _assistant_call(message: dict[str, Any]) -> dict[str, Any] | None:
    matches = TOOL_CALL_RE.findall(_content_text(message))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("multiple_calls_in_one_turn")
    try:
        call = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_call_json") from exc
    if not isinstance(call, dict):
        raise ValueError("invalid_call_object")
    return call


def _clean_question(text: str) -> str:
    question = re.split(r"\n\s*Guidelines\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not question:
        raise ValueError("empty_question")
    return question


def _validate_candidate(row: dict[str, Any], row_index: int) -> Candidate:
    messages = list(row.get("message_list") or [])
    if not messages:
        raise ValueError("empty_messages")
    if any(_content_media(message, "video") for message in messages):
        raise ValueError("video_trajectory")
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages or len(_content_media(user_messages[0], "image")) != 1:
        raise ValueError("requires_one_original_image")
    _clean_question(_content_text(user_messages[0]))

    calls: list[dict[str, Any]] = []
    terminal_answers = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        call = _assistant_call(message)
        if call is None:
            if "\\boxed{" in _content_text(message):
                terminal_answers += 1
            continue
        if call.get("name") != "crop_image_normalized":
            raise ValueError("unsupported_tool")
        arguments = call.get("arguments") or {}
        bbox = arguments.get("bbox_2d")
        target = arguments.get("target_image")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int | float) and 0 <= float(value) <= 1 for value in bbox)
            or not (float(bbox[0]) < float(bbox[2]) and float(bbox[1]) < float(bbox[3]))
        ):
            raise ValueError("invalid_normalized_bbox")
        if not isinstance(target, int) or not 1 <= target <= len(calls) + 1:
            raise ValueError("unknown_or_forward_target")
        calls.append(call)
    if not calls:
        raise ValueError("no_zoom_call")
    if terminal_answers != 1:
        raise ValueError("missing_or_multiple_final_answer")
    observation_images = sum(len(_content_media(message, "image")) for message in user_messages[1:])
    if observation_images != len(calls):
        raise ValueError("call_observation_mismatch")
    return Candidate(row_index=row_index, qid=str(row.get("qid") or row_index), call_count=len(calls))


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _question_fingerprint(question: str, image_bytes: bytes) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    with Image.open(io.BytesIO(image_bytes)) as image:
        pixels = image.convert("L").resize((17, 16)).tobytes()
    bits = [pixels[y * 17 + x] > pixels[y * 17 + x + 1] for y in range(16) for x in range(16)]
    difference_hash = bytes(
        sum((1 << (7 - offset)) if bits[index + offset] else 0 for offset in range(8))
        for index in range(0, 256, 8)
    )
    return hashlib.sha256(normalized.encode() + b"\0" + difference_hash).hexdigest()


def _extra_info(candidate: Candidate, fingerprint: str) -> dict[str, Any]:
    source = {
        "dataset": "TIGER-Lab/PixelReasoner-SFT-Data",
        "revision": "23a10b7a944ee81d9e4ec91580b2cdf8d0665660",
        "license": "Apache-2.0",
        "qid": candidate.qid,
        "question_image_fingerprint": fingerprint,
        "nested_boxes_composed_to_original": True,
    }
    return {
        "sample_id": f"pixel_reasoner/{candidate.qid}",
        "source_parquet": "PixelReasoner/release_checked_toolpart.json",
        "source_row_index": candidate.row_index,
        "source_group": "pixel_reasoner.zoom",
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


def _convert(row: dict[str, Any], candidate: Candidate, archive: zipfile.ZipFile) -> dict[str, Any]:
    source_messages = list(row["message_list"])
    first_user = next(message for message in source_messages if message.get("role") == "user")
    question = _clean_question(_content_text(first_user))
    original_path = _content_media(first_user, "image")[0]
    original_bytes = archive.read(original_path)
    with Image.open(io.BytesIO(original_bytes)) as loaded:
        original = loaded.convert("RGB")
    width, height = original.size

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"<image>\n{question}"},
    ]
    images = [{"bytes": original_bytes, "path": ""}]
    regions: list[list[float]] = [[0.0, 0.0, float(width), float(height)]]
    seen_boxes: set[tuple[int, int, int, int]] = set()
    emitted = 0

    for source_message in source_messages:
        if source_message.get("role") != "assistant":
            continue
        text = _content_text(source_message)
        call = _assistant_call(source_message)
        if call is None:
            if "\\boxed{" in text:
                messages.append({"role": "assistant", "content": normalize_final_answer(text).strip()})
            continue
        arguments = call["arguments"]
        normalized_bbox = [float(value) for value in arguments["bbox_2d"]]
        target_index = int(arguments["target_image"]) - 1
        if not 0 <= target_index < len(regions):
            raise ValueError("unresolved_target_image")
        source = regions[target_index]
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
        if tuple(pixel_bbox) in seen_boxes:
            raise ValueError("duplicate_composed_bbox")
        seen_boxes.add(tuple(pixel_bbox))
        regions.append([float(value) for value in pixel_bbox])
        reasoning = TOOL_CALL_RE.sub("", text).strip()
        messages.append(
            {
                "role": "assistant",
                "content": reasoning,
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
        )
        messages.append({"role": "tool", "content": f"<image>\n{zoom_response_text(pixel_bbox)}"})
        images.append({"bytes": _jpeg_bytes(crop), "path": ""})
        emitted += 1

    if emitted != candidate.call_count or not messages[-1]["content"]:
        raise ValueError("incomplete_trajectory")
    if sum(message["content"].count("<image>") for message in messages) != len(images):
        raise ValueError("image_placeholder_mismatch")
    fingerprint = _question_fingerprint(question, original_bytes)
    return {
        "messages": _normalize_messages(messages),
        "images": images,
        "tools": copy.deepcopy(FORMAL_TWO_TOOL_SCHEMAS),
        "data_source": "pixel_reasoner.zoom",
        "ability": "perception",
        "extra_info": _extra_info(candidate, fingerprint),
    }


def main() -> None:
    args = _parse_args()
    input_json = Path(args.input_json).resolve()
    images_zip = Path(args.images_zip).resolve()
    output = Path(args.output).resolve()
    if not input_json.is_file() or not images_zip.is_file():
        raise FileNotFoundError("PixelReasoner JSON and images.zip are both required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    rows = json.loads(input_json.read_text(encoding="utf-8"))
    candidates: dict[int, Candidate] = {}
    rejected: Counter[str] = Counter()
    raw_histogram: Counter[str] = Counter()
    qids: set[str] = set()
    for row_index, row in enumerate(rows):
        try:
            candidate = _validate_candidate(row, row_index)
        except (KeyError, TypeError, ValueError) as exc:
            rejected[str(exc)] += 1
            continue
        if candidate.qid in qids:
            rejected["duplicate_qid"] += 1
            continue
        qids.add(candidate.qid)
        candidates[row_index] = candidate
        raw_histogram[_density(candidate.call_count)] += 1

    import pyarrow as pa
    import pyarrow.parquet as pq

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary output: {temporary}")
    writer = pq.ParquetWriter(temporary, _output_schema(), compression="zstd")
    converted_histogram: Counter[str] = Counter()
    conversion_rejected: Counter[str] = Counter()
    fingerprints: set[str] = set()
    written = 0
    output_rows: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(images_zip) as archive:
            for row_index, candidate in candidates.items():
                try:
                    converted = _convert(rows[row_index], candidate, archive)
                    fingerprint = json.loads(converted["extra_info"]["source_extra_json"])[
                        "question_image_fingerprint"
                    ]
                    if fingerprint in fingerprints:
                        raise ValueError("duplicate_question_and_image")
                except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
                    conversion_rejected[str(exc)] += 1
                    continue
                fingerprints.add(fingerprint)
                output_rows.append(converted)
                converted_histogram[_density(candidate.call_count)] += 1
                if len(output_rows) == 16:
                    writer.write_table(pa.Table.from_pylist(output_rows, schema=_output_schema()))
                    written += len(output_rows)
                    output_rows.clear()
            if output_rows:
                writer.write_table(pa.Table.from_pylist(output_rows, schema=_output_schema()))
                written += len(output_rows)
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    temporary.replace(output)

    report = {
        "source": {
            "dataset": "TIGER-Lab/PixelReasoner-SFT-Data",
            "revision": "23a10b7a944ee81d9e4ec91580b2cdf8d0665660",
            "license": "Apache-2.0",
            "input_json": str(input_json),
            "images_zip": str(images_zip),
        },
        "metadata_valid_rows": len(candidates),
        "metadata_density_counts": dict(sorted(raw_histogram.items())),
        "metadata_rejected": dict(sorted(rejected.items())),
        "converted_rows": written,
        "converted_density_counts": dict(sorted(converted_histogram.items())),
        "conversion_rejected": dict(sorted(conversion_rejected.items())),
        "output": str(output),
        "output_sha256": _sha256(output),
        "formal_label_status": {
            "difficulty": "not_measured_requires_rollout",
            "oracle_utility": "not_measured_requires_rollout",
            "effective_call_count": "not_claimed_observed_call_count_only",
        },
    }
    output.with_name("build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
