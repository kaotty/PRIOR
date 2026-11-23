"""Shared SFT record conversion, token accounting, and Parquet schema."""


from __future__ import annotations
import copy
import io
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Optional, Sequence
import pyarrow as pa
from PIL import Image
from . import tool_schema as sft_schema
from .parse_traces import RawTrajectory, normalize_final_answer
from .zoom_crop import bake_zoom, zoom_response_text
from .tool_schema import FORMAL_TWO_TOOL_SYSTEM_PROMPT


logger = logging.getLogger("tir.sft.records")


def _load_image_bytes(path: Path) -> bytes:
    """Re-encode any PIL-loadable file to JPEG bytes for parquet storage.

    JPEG keeps parquets compact while preserving enough fidelity for VL
    training. If a future ablation needs the original PNGs/WEBPs, swap to
    ``open(path, 'rb').read()`` here.
    """

    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


def _bake_zoom_bytes(
    original_image_path: Path, bbox_2d: list[int]
) -> Optional[tuple[bytes, list[int]]]:
    try:
        with Image.open(original_image_path) as im:
            im = im.convert("RGB")
            baked = bake_zoom(im, bbox_2d)
            if baked is None:
                return None
            cropped, resized_bbox = baked
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=92)
            return buf.getvalue(), resized_bbox
    except Exception as exc:  # pragma: no cover — defensive only
        logger.warning("Failed to bake zoom for %s: %s", original_image_path, exc)
        return None


def _make_user_message(question: str, n_user_images: int) -> dict[str, Any]:
    placeholders = "\n".join(["<image>"] * n_user_images)
    if question:
        return {"role": "user", "content": f"{placeholders}\n{question}".strip()}
    return {"role": "user", "content": placeholders}


def _truncate(s: str, max_chars: int = 1500) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n... (truncated)"


def convert_trajectory(
    traj: RawTrajectory,
    images_root: Path,
    target: str,
    data_source: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Apply tool-call conversion + image baking to produce one parquet row.

    Returns ``None`` if the trajectory cannot be converted (e.g. non-static
    crop bbox, missing image file).

    ``data_source`` overrides the ``data_source`` column (defaults to
    ``deepeyesv2_sft.<target>``). Other sources (CoF, synthesized C, …) reuse
    this exact emit + zoom-baking path so every mother-pool row shares one
    schema; they just pass their own tag, e.g. ``"cof.zoom"``.
    """

    is_zoom_only = target == "zoom_only"
    if is_zoom_only and not traj.is_zoom_only:
        return None

    # Diverse mode also requires that EVERY tool call is something we support
    # (zoom / code / search). All ``__bad__`` cases were filtered upstream.

    image_paths = [images_root / p for p in traj.image_paths]
    for p in image_paths:
        if not p.exists():
            return None

    out_messages: list[dict[str, Any]] = [
        {"role": "system", "content": sft_schema.system_prompt_for(target)},
    ]
    out_images: list[bytes] = []

    # First user turn: original image(s) + question.
    for idx in traj.user_image_indices:
        out_images.append(_load_image_bytes(image_paths[idx]))
    out_messages.append(_make_user_message(traj.question, len(traj.user_image_indices)))

    for turn in traj.turns:
        # Assistant turn body. We strip residual <image> tags (they would have
        # come from the source data's tool responses, never from the assistant
        # itself, but be defensive). Replace <answer>X</answer> with \boxed{X}.
        text = normalize_final_answer(turn.assistant.text_content or "")
        text = text.replace("<image>", "").rstrip()

        asst_msg: dict[str, Any] = {"role": "assistant", "content": text}

        if turn.assistant.tool_calls:
            call = turn.assistant.tool_calls[0]
            cat = turn.assistant.call_categories[0]

            # Pure-zoom branch: re-bake the crop with our exact runtime logic
            # so the SFT image matches what RL will produce later.
            if cat == "zoom":
                _zoom_args = json.loads(call["function"]["arguments"])
                bbox_2d = _zoom_args.get("bbox_2d")
                _label = _zoom_args.get("label")
                # Source images are at user_image_indices[0] for now; multi-image
                # zoom trajectories are filtered upstream (we only allow single user image).
                if len(traj.user_image_indices) != 1:
                    return None
                user_img_path = image_paths[traj.user_image_indices[0]]
                baked = _bake_zoom_bytes(user_img_path, bbox_2d)
                if baked is None:
                    return None
                cropped_bytes, resized_bbox = baked
                # Update the call to use the (possibly resized) bbox so the
                # response_text matches what RL will narrate. Preserve the
                # optional label if the source provided one (CoF does).
                _new_args: dict[str, Any] = {"bbox_2d": resized_bbox}
                if _label:
                    _new_args["label"] = _label
                call["function"]["arguments"] = json.dumps(_new_args, ensure_ascii=False)
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                out_messages.append(
                    {
                        "role": "tool",
                        "content": "<image>\n" + zoom_response_text(resized_bbox, _label),
                    }
                )
                out_images.append(cropped_bytes)
                continue

            # Code-interpreter branch: pass through the source images and text.
            if cat == "code":
                if is_zoom_only:
                    return None
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                tool_content = _truncate(turn.response_text)
                if turn.response_image_indices:
                    placeholders = "\n".join(["<image>"] * len(turn.response_image_indices))
                    tool_content = f"{placeholders}\n{tool_content}".strip()
                    for idx in turn.response_image_indices:
                        out_images.append(_load_image_bytes(image_paths[idx]))
                out_messages.append({"role": "tool", "content": tool_content})
                continue

            # Web-search branch: text-only response.
            if cat == "search":
                if is_zoom_only:
                    return None
                if turn.response_image_indices:
                    return None  # text-search responses must be image-free
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                out_messages.append(
                    {"role": "tool", "content": _truncate(turn.response_text)}
                )
                continue

            # Reverse-image-search branch: response carries N thumbnails plus
            # textual context (e.g. "## Web Results\n1. <image>\n[Title]\n...").
            if cat == "image_search":
                if is_zoom_only:
                    return None
                if not turn.response_image_indices:
                    return None  # image_search must return at least one thumbnail
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                placeholders = "\n".join(["<image>"] * len(turn.response_image_indices))
                tool_content = _truncate(turn.response_text)
                tool_content = f"{placeholders}\n{tool_content}".strip()
                for idx in turn.response_image_indices:
                    out_images.append(_load_image_bytes(image_paths[idx]))
                out_messages.append({"role": "tool", "content": tool_content})
                continue

            return None  # unknown category

        # Terminal turn — no tool call. Just emit the assistant message.
        out_messages.append(asst_msg)

    # Sanity check: <image> placeholders == len(out_images)
    placeholder_count = 0
    for m in out_messages:
        c = m["content"]
        if isinstance(c, str):
            placeholder_count += c.count("<image>")
    if placeholder_count != len(out_images):
        return None

    return {
        "messages": out_messages,
        "images": [{"bytes": b, "path": None} for b in out_images],
        "tools": sft_schema.tool_schemas_for(target),
        "data_source": data_source or f"deepeyesv2_sft.{target}",
        "ability": _infer_ability(traj),
        "extra_info": {
            "source_file": traj.source_file,
            "raw_index": traj.raw_index,
            "sample_id": traj.sample_id,
            "n_zoom_calls": traj.n_zoom_calls,
            "n_code_calls": traj.n_code_calls,
            "n_search_calls": traj.n_search_calls,
            "n_image_search_calls": traj.n_image_search_calls,
        },
    }


def _infer_ability(traj: RawTrajectory) -> str:
    """Cheap heuristic ability tag for downstream filtering / weighting."""

    if traj.is_zoom_using:
        return "perception"
    if traj.n_image_search_calls > 0:
        return "image_search"
    if traj.n_search_calls > 0:
        return "search"
    if traj.n_code_calls > 0:
        return "math_code"
    return "general"


def _stratified_split(
    items: list[Any], test_frac: float, rng: random.Random
) -> tuple[list[Any], list[Any]]:
    """Shuffle ``items`` and slice off ``test_frac`` for the test split.

    Empty / single-item pools degrade gracefully:
      * empty pool      → ``([], [])``
      * single-item pool → goes to train, test stays empty
      * otherwise        → at least 1 item in test, the rest in train
    """

    items = list(items)
    if not items:
        return [], []
    rng.shuffle(items)
    if len(items) == 1:
        return items, []
    n_test = max(1, int(len(items) * test_frac))
    return items[n_test:], items[:n_test]


_PERCEPTION_CODE = re.compile(
    r"(?:\b(?:PIL|cv2|opencv|pytesseract|easyocr|imageio|skimage)\b|"
    r"matplotlib\s*\.\s*image|\b(?:imread|crop|resize)\s*\(|"
    r"\b(?:requests|urllib|socket|pathlib|glob)\b|\bopen\s*\()",
    re.IGNORECASE,
)


def _tool_calls(messages: Sequence[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"__invalid_json__": arguments}
            calls.append((name, dict(arguments) if isinstance(arguments, dict) else {"__invalid_args__": arguments}))
    return calls


def _assistant_loss_tokens(messages: Sequence[dict[str, Any]], tokenizer: Any) -> int:
    """Mirror MultiTurnSFTDataset's per-message assistant loss mask."""

    from verl.utils.chat_template import apply_chat_template, extract_system_prompt_and_generation
    from verl.utils.tokenizer import normalize_token_ids

    system_prompt, generation_prompt = extract_system_prompt_and_generation(tokenizer)
    total = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        input_ids = apply_chat_template(
            tokenizer,
            messages=[message],
            tools=None,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        input_ids = normalize_token_ids(input_ids)
        if index != 0:
            input_ids = input_ids[len(system_prompt) :]
        count = len(input_ids) - len(generation_prompt)
        if count <= 0:
            raise ValueError(f"assistant turn {index} produced a non-positive loss-token count")
        total += count
    if total <= 0:
        raise ValueError("trajectory has no trainable assistant tokens")
    return total


def _normalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(list(messages))
    system = {"role": "system", "content": FORMAL_TWO_TOOL_SYSTEM_PROMPT, "tool_calls": None}
    if normalized and normalized[0].get("role") == "system":
        normalized[0] = system
    else:
        normalized.insert(0, system)
    output: list[dict[str, Any]] = []
    for message in normalized:
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            tool_calls.append(
                {
                    "type": str(call.get("type") or "function"),
                    "function": {"name": str(function.get("name") or ""), "arguments": arguments},
                }
            )
        output.append(
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
                "tool_calls": tool_calls or None,
            }
        )
    return output


def _output_schema():
    import pyarrow as pa

    message_tool_call = pa.struct(
        [
            pa.field("type", pa.string()),
            pa.field("function", pa.struct([pa.field("name", pa.string()), pa.field("arguments", pa.string())])),
        ]
    )
    bbox = pa.struct(
        [
            pa.field("type", pa.string()),
            pa.field("items", pa.struct([pa.field("type", pa.string())])),
            pa.field("minItems", pa.int64()),
            pa.field("maxItems", pa.int64()),
            pa.field("description", pa.string()),
        ]
    )
    simple_property = pa.struct([pa.field("type", pa.string()), pa.field("description", pa.string())])
    tool = pa.struct(
        [
            pa.field("type", pa.string()),
            pa.field(
                "function",
                pa.struct(
                    [
                        pa.field("name", pa.string()),
                        pa.field("description", pa.string()),
                        pa.field(
                            "parameters",
                            pa.struct(
                                [
                                    pa.field("type", pa.string()),
                                    pa.field(
                                        "properties",
                                        pa.struct(
                                            [
                                                pa.field("bbox_2d", bbox),
                                                pa.field("label", simple_property),
                                                pa.field("code", simple_property),
                                            ]
                                        ),
                                    ),
                                    pa.field("required", pa.list_(pa.string())),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    extra = pa.struct(
        [
            pa.field("sample_id", pa.string()),
            pa.field("source_parquet", pa.string()),
            pa.field("source_row_index", pa.int64()),
            pa.field("source_group", pa.string()),
            pa.field("domain_group", pa.string()),
            pa.field("pool_role", pa.string()),
            pa.field("shared_anchor_type", pa.string()),
            pa.field("tool_type", pa.string()),
            pa.field("difficulty", pa.string()),
            pa.field("density", pa.string()),
            pa.field("no_tool_correct_count", pa.int64()),
            pa.field("oracle_tool_correct_count", pa.int64()),
            pa.field("no_tool_success_rate", pa.float64()),
            pa.field("oracle_tool_success_rate", pa.float64()),
            pa.field("oracle_tool_gain", pa.float64()),
            pa.field("actual_tool_calls", pa.int64()),
            pa.field("effective_tool_calls", pa.int64()),
            pa.field("assistant_loss_tokens", pa.int64()),
            pa.field("experimental_arm", pa.string()),
            pa.field("selection_seed", pa.int64()),
            pa.field("spec_sha256", pa.string()),
            pa.field("labels_sha256", pa.string()),
            pa.field("source_extra_json", pa.large_string()),
        ]
    )
    return pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                            pa.field("tool_calls", pa.list_(message_tool_call)),
                        ]
                    )
                ),
            ),
            pa.field("images", pa.list_(pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]))),
            pa.field("tools", pa.list_(tool)),
            pa.field("data_source", pa.string()),
            pa.field("ability", pa.string()),
            pa.field("extra_info", extra),
        ]
    )
