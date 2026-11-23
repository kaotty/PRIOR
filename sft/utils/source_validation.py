"""Validate source tool traces and inventory DeepEyes computation examples.

These are the frozen source filters; experiment selection lives in sft.frozen.
"""


from __future__ import annotations
import copy
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from PIL import Image
from .records import _make_user_message, _truncate
from .records import _assistant_loss_tokens, _normalize_messages, _tool_calls
from .parse_traces import RawTrajectory, parse_files
from .tool_schema import FORMAL_TWO_TOOL_SCHEMAS
from .parse_traces import normalize_final_answer


FAMILY = {
    "code_crop_zoom": "perception_access", "code_annotate": "perception_access",
    "code_display": "perception_access", "code_ocr": "perception_access",
    "code_chart": "visualization",
    "code_compute": "computation", "code_plain": "computation", "code_table": "computation",
    "text_search": "external_retrieval", "image_search": "external_retrieval",
    "other_toolcall": "other",
}


def classify_code(body, returns_image):
    b = body.lower()
    has = lambda kws: any(k in b for k in kws)  # noqa: E731
    if ".crop(" in b or "cropped" in b or "[y1:y2" in b or "[ymin:" in b or "crop_img" in b:
        return "code_crop_zoom"
    if has((".add_patch(", ".rectangle(", ".circle(", ".puttext(", "imagedraw", ".line(",
            "draw.", ".polygon(", ".ellipse(")):
        return "code_annotate"
    if has(("pytesseract", "easyocr", "paddleocr", "image_to_string")):
        return "code_ocr"
    if has((".plot(", ".scatter(", ".bar(", ".hist(", ".pie(", "axvline", "axhline")) \
            and not has(("imshow", "crop")):
        return "code_chart"
    if returns_image or has(("imshow", "cv2", "image.open", ".convert(", ".resize(", "cvtcolor",
                             "import pil", "from pil")):
        return "code_display"
    if has(("pandas", "pd.", "read_csv", "dataframe")):
        return "code_table"
    if has(("sympy", "solve(", "symbols(", "eq(", ".subs(", "simplify", "numpy", "np.",
            "import math", "math.", "scipy", "fractions", "statistics", "sqrt(", "factorial",
            "comb(", "cmath")):
        return "code_compute"
    return "code_plain"


EXECUTION_ERROR = re.compile(
    r"(?:traceback \(most recent call last\)|syntaxerror|nameerror|"
    r"filenotfounderror|modulenotfounderror|timed?\s*out)",
    re.IGNORECASE,
)


IMAGE_ACCESS = re.compile(
    r"(?:\binput_image\b|\bimage_[0-9]+\b|\.(?:png|jpe?g|webp|gif)\b|"
    r"/mnt/data|base64|\b(?:http|https)://)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    tool_type: str
    density: str
    call_count: int
    assistant_loss_tokens: int
    source_path: str
    source_row_index: int
    source_group: str
    data_source: str
    dedup_key: str


def _density(call_count: int) -> str:
    if call_count == 1:
        return "low"
    if 2 <= call_count <= 3:
        return "medium"
    if call_count >= 4:
        return "high"
    raise ValueError(f"call_count must be positive, got {call_count}")


def _placeholder_count(messages: Sequence[dict[str, Any]]) -> int:
    return sum(str(message.get("content") or "").count("<image>") for message in messages)


def _question_image_key(messages: Sequence[dict[str, Any]], image_bytes: bytes) -> str:
    user = next((str(message.get("content") or "") for message in messages if message.get("role") == "user"), "")
    user = user.replace("<image>", " ")
    user = re.sub(r"^\s*Question:\s*", "", user, flags=re.IGNORECASE)
    user = re.split(
        r"\n(?:Answer with|Think in the mind|Format strictly)", user, maxsplit=1, flags=re.IGNORECASE
    )[0]
    normalized = re.sub(r"[^a-z0-9]+", " ", user.lower()).strip()
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            pixels = image.convert("L").resize((17, 16)).tobytes()
    except Exception as exc:
        raise ValueError("invalid_original_image") from exc
    bits = [pixels[y * 17 + x] > pixels[y * 17 + x + 1] for y in range(16) for x in range(16)]
    difference_hash = bytes(
        sum((1 << (7 - offset)) if bits[index + offset] else 0 for offset in range(8))
        for index in range(0, 256, 8)
    )
    return hashlib.sha256(normalized.encode() + b"\0" + difference_hash).hexdigest()


def _valid_zoom_row(row: dict[str, Any]) -> tuple[bool, str, int]:
    messages = list(row.get("messages") or [])
    images = list(row.get("images") or [])
    calls = _tool_calls(messages)
    if not calls or {name for name, _ in calls} != {"image_zoom_in_tool"}:
        return False, "not_zoom_only", len(calls)
    if not images or images[0].get("bytes") is None:
        return False, "missing_original_image", len(calls)
    if _placeholder_count(messages) != len(images):
        return False, "image_placeholder_mismatch", len(calls)
    try:
        with Image.open(io.BytesIO(bytes(images[0]["bytes"]))) as image:
            width, height = image.size
    except Exception:
        return False, "invalid_original_image", len(calls)
    seen_boxes: set[tuple[float, float, float, float]] = set()
    for _, arguments in calls:
        bbox = arguments.get("bbox_2d")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int | float) and math.isfinite(value) for value in bbox)
        ):
            return False, "invalid_bbox", len(calls)
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return False, "bbox_outside_original", len(calls)
        key = tuple(round(value, 6) for value in (x1, y1, x2, y2))
        if key in seen_boxes:
            return False, "duplicate_exact_call", len(calls)
        seen_boxes.add(key)
    return True, "ok", len(calls)


def _code_call_arguments(traj: RawTrajectory) -> Iterable[tuple[str, str]]:
    for turn in traj.turns:
        if not turn.assistant.tool_calls:
            continue
        call = turn.assistant.tool_calls[0]
        arguments = json.loads(call["function"]["arguments"])
        yield str(arguments.get("code") or "").strip(), turn.response_text.strip()


def _valid_code_trajectory(traj: RawTrajectory) -> tuple[bool, str]:
    if not (
        traj.n_code_calls > 0
        and traj.n_zoom_calls == 0
        and traj.n_search_calls == 0
        and traj.n_image_search_calls == 0
    ):
        return False, "not_code_only"
    seen_code: set[str] = set()
    seen_response: set[str] = set()
    try:
        pairs = list(_code_call_arguments(traj))
    except (KeyError, TypeError, json.JSONDecodeError):
        return False, "invalid_code_arguments"
    if len(pairs) != traj.n_code_calls:
        return False, "call_count_mismatch"
    call_index = 0
    for turn in traj.turns:
        if not turn.assistant.tool_calls:
            continue
        if turn.assistant.call_categories != ["code"] or turn.response_image_indices:
            return False, "code_reads_or_returns_image"
        code, response = pairs[call_index]
        call_index += 1
        if not code or not response:
            return False, "empty_code_or_response"
        if _PERCEPTION_CODE.search(code) or IMAGE_ACCESS.search(code):
            return False, "perception_or_external_access"
        if FAMILY.get(classify_code(code, returns_image=False)) != "computation":
            return False, "not_genuine_computation"
        if EXECUTION_ERROR.search(response):
            return False, "execution_error"
        normalized_code = re.sub(r"\s+", " ", code).strip()
        normalized_response = re.sub(r"\s+", " ", response).strip()
        if normalized_code in seen_code or normalized_response in seen_response:
            return False, "duplicate_call_or_response"
        seen_code.add(normalized_code)
        seen_response.add(normalized_response)
    return True, "ok"


def _code_messages(traj: RawTrajectory) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [_make_user_message(traj.question, len(traj.user_image_indices))]
    for turn in traj.turns:
        content = normalize_final_answer(turn.assistant.text_content or "").replace("<image>", "").rstrip()
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if turn.assistant.tool_calls:
            assistant["tool_calls"] = [copy.deepcopy(turn.assistant.tool_calls[0])]
            messages.append(assistant)
            messages.append({"role": "tool", "content": _truncate(turn.response_text)})
        else:
            messages.append(assistant)
    return _normalize_messages(messages)


def _scan_code(json_paths: Sequence[Path], images_root: Path, tokenizer: Any) -> tuple[list[Candidate], Counter[str]]:
    candidates: list[Candidate] = []
    rejected: Counter[str] = Counter()
    for traj in parse_files(json_paths):
        valid, reason = _valid_code_trajectory(traj)
        if not valid:
            rejected[reason] += 1
            continue
        if any(not (images_root / traj.image_paths[index]).exists() for index in traj.user_image_indices):
            rejected["missing_user_image"] += 1
            continue
        messages = _code_messages(traj)
        try:
            original_path = images_root / traj.image_paths[traj.user_image_indices[0]]
            dedup_key = _question_image_key(messages, original_path.read_bytes())
        except (IndexError, OSError, ValueError) as exc:
            rejected[f"dedup_key:{exc}"] += 1
            continue
        candidates.append(
            Candidate(
                sample_id=f"deepeyesv2/{traj.sample_id}",
                tool_type="code",
                density=_density(traj.n_code_calls),
                call_count=traj.n_code_calls,
                assistant_loss_tokens=_assistant_loss_tokens(messages, tokenizer),
                source_path=str(Path(traj.source_file)),
                source_row_index=traj.raw_index,
                source_group="deepeyesv2.compute_only",
                data_source="deepeyesv2.compute_only",
                dedup_key=dedup_key,
            )
        )
    return candidates, rejected


def _canonical_row(
    row: dict[str, Any], candidate: Candidate, arm_name: str, seed: int, config_hash: str
) -> dict[str, Any]:
    source_extra = dict(row.get("extra_info") or {})
    return {
        "messages": _normalize_messages(row["messages"]),
        "images": [
            {
                "bytes": bytes(image["bytes"]) if image.get("bytes") is not None else None,
                "path": str(image.get("path") or ""),
            }
            for image in row.get("images") or []
        ],
        "tools": copy.deepcopy(FORMAL_TWO_TOOL_SCHEMAS),
        "data_source": candidate.data_source,
        "ability": "perception" if candidate.tool_type == "zoom" else "math_code",
        "extra_info": {
            "sample_id": candidate.sample_id,
            "source_parquet": candidate.source_path,
            "source_row_index": candidate.source_row_index,
            "source_group": candidate.source_group,
            "domain_group": "unlabeled",
            "pool_role": "call_density_candidate_core",
            "shared_anchor_type": "",
            "tool_type": candidate.tool_type,
            "difficulty": "",
            "density": candidate.density,
            "no_tool_correct_count": None,
            "oracle_tool_correct_count": None,
            "no_tool_success_rate": None,
            "oracle_tool_success_rate": None,
            "oracle_tool_gain": None,
            "actual_tool_calls": candidate.call_count,
            # A later quality audit must decide whether every call added useful
            # evidence.  Do not silently equate observed and effective calls.
            "effective_tool_calls": None,
            "assistant_loss_tokens": candidate.assistant_loss_tokens,
            "experimental_arm": arm_name,
            "selection_seed": seed,
            "spec_sha256": config_hash,
            "labels_sha256": "",
            "source_extra_json": json.dumps(source_extra, sort_keys=True, ensure_ascii=False),
        },
    }
