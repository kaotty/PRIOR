"""Convert a supplied CoF snapshot into canonical Zoom SFT parquets.

The public SFT preparation shell downloads the pinned source before conversion.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator, Optional

from .records import _stratified_split, _normalize_messages, _output_schema, convert_trajectory
from .source_validation import _density, _valid_zoom_row
from .tool_schema import FORMAL_TWO_TOOL_SCHEMAS
from .parse_traces import ParsedAssistantTurn, RawTrajectory, RawTurn


logger = logging.getLogger("agentic_sft.cof_zoom")

_JSON_NAMES = ("cof_sft_data.json",)
_ZIP_NAMES = ("images.zip",)
DATA_SOURCE = "cof.zoom"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Read the pinned CoF snapshot (cof_sft_data.json + images/) from here.",
    )
    p.add_argument("--test_frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow_multi_zoom",
        action="store_true",
        help=(
            "Also emit trajectories with >1 zoom call. OFF by default: multi-zoom "
            "bboxes may be expressed in a previous crop's frame, but convert_trajectory "
            "always bakes from the original image. Keep single-zoom until audited."
        ),
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Don't write parquet; print counts + a worked example and exit.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Debug: cap raw samples read.")
    args = p.parse_args()
    return args


def _locate_snapshot(args: argparse.Namespace) -> Path:
    return Path(args.source_dir).expanduser().resolve()


def _find_first(root: Path, names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        hit = root / name
        if hit.exists():
            return hit
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _resolve_images_root(snapshot: Path) -> Path:
    """Return the dir D such that ``D / "0/0.jpg"`` (CoF's relative paths) resolves.

    Extracts ``images.zip`` once next to it if the images aren't already unpacked.
    """

    # Already-unpacked layouts we accept.
    for cand in (snapshot / "images", snapshot):
        if (cand / "0").is_dir():
            return cand

    zip_path = _find_first(snapshot, _ZIP_NAMES)
    if zip_path is None:
        raise SystemExit(
            f"Could not find images.zip or an unpacked images dir under {snapshot}. "
            "Pass --source_dir to a snapshot that contains the CoF images."
        )
    dest = zip_path.parent / "images_unzipped"
    if not dest.exists():
        logger.info("Extracting %s → %s (one-time)…", zip_path, dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    for cand in (dest, dest / "images"):
        if (cand / "0").is_dir():
            return cand
    # Fall back to the extraction root; convert_trajectory will drop rows whose
    # image files don't resolve, so a wrong guess fails loudly (0 rows) not silently.
    return dest


def _gather(json_path: Path, args: argparse.Namespace) -> list[RawTrajectory]:
    """Parse CoF and keep the zoom pool (single user image; single-zoom by default)."""

    kept: list[RawTrajectory] = []
    n_total = n_zoom = n_multi = n_multi_img = 0
    for traj in parse_cof_json(json_path, max_samples=args.max_samples):
        n_total += 1
        if not traj.is_zoom_using:
            continue  # text-only reasoning rows are not part of the Z pool
        n_zoom += 1
        if len(traj.user_image_indices) != 1:
            n_multi_img += 1
            continue
        if traj.n_zoom_calls > 1:
            n_multi += 1
            if not args.allow_multi_zoom:
                continue
        kept.append(traj)
    logger.info(
        "CoF: parsed=%d, zoom-using=%d, multi-user-image(dropped)=%d, "
        "multi-zoom=%d (kept=%s), Z-pool kept=%d",
        n_total, n_zoom, n_multi_img, n_multi,
        "yes" if args.allow_multi_zoom else "no", len(kept),
    )
    return kept


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    )
    rng = random.Random(args.seed)

    snapshot = _locate_snapshot(args)
    json_path = _find_first(snapshot, _JSON_NAMES)
    if json_path is None:
        raise SystemExit(
            f"Could not find cof_sft_data.json under {snapshot}. Pass --source_dir explicitly."
        )
    images_root = _resolve_images_root(snapshot)
    logger.info("json=%s  images_root=%s", json_path, images_root)

    trajs = _gather(json_path, args)
    if not trajs:
        raise SystemExit("No CoF zoom trajectories survived parsing — check --source_dir / format.")

    if args.inspect:
        _inspect(trajs, images_root)
        return

    train, test = _stratified_split(trajs, args.test_frac, rng)
    train_rows = _materialise(train, images_root)
    test_rows = _materialise(test, images_root)
    logger.info(
        "Converted: train %d/%d, test %d/%d (drops = unrecoverable bbox / missing image)",
        len(train_rows), len(train), len(test_rows), len(test),
    )
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    out = Path(args.output_dir).expanduser() / "cof_zoom"  # no .resolve() — honor the literal path (symlinks kept)
    _write_formal_pool(train_rows, out / "train.parquet")
    if test_rows:
        _write_formal_pool(test_rows, out / "test.parquet")
    else:
        logger.warning("Empty test split (test_frac too small for pool size); skipped test.parquet.")


def _materialise(trajs: list[RawTrajectory], images_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for traj in trajs:
        row = convert_trajectory(traj, images_root, target="zoom_only", data_source=DATA_SOURCE)
        if row is not None:
            rows.append(row)
    return rows


def _write_formal_pool(rows: list[dict[str, Any]], output: Path) -> None:
    """Apply the original CoF canonicalization stage after its shuffled split."""
    import copy
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = []
    for index, row in enumerate(rows):
        valid, _, calls = _valid_zoom_row(row)
        if not valid:
            continue
        original = dict(row.get("extra_info") or {})
        metadata = {
            "sample_id": original["sample_id"], "source_parquet": str(output),
            "source_row_index": index, "source_group": "cof.zoom", "domain_group": "unlabeled",
            "pool_role": "call_density_candidate_core", "shared_anchor_type": "", "tool_type": "zoom",
            "difficulty": "", "density": _density(calls), "no_tool_correct_count": None,
            "oracle_tool_correct_count": None, "no_tool_success_rate": None, "oracle_tool_success_rate": None,
            "oracle_tool_gain": None, "actual_tool_calls": calls, "effective_tool_calls": None,
            "assistant_loss_tokens": None, "experimental_arm": "", "selection_seed": 42,
            "spec_sha256": "", "labels_sha256": "", "source_extra_json": json.dumps(original, sort_keys=True),
        }
        accepted.append({"messages": _normalize_messages(row["messages"]),
                         "images": [{"bytes": image["bytes"], "path": str(image.get("path") or "")}
                                    for image in row["images"]],
                         "tools": copy.deepcopy(FORMAL_TWO_TOOL_SCHEMAS), "data_source": "cof.zoom",
                         "ability": "perception", "extra_info": metadata})
    if not accepted:
        raise RuntimeError("No canonical CoF trajectories survived")
    pq.write_table(pa.Table.from_pylist(accepted, schema=_output_schema()), output, compression="zstd")
    logger.info("Canonical CoF pool: %d/%d rows -> %s", len(accepted), len(rows), output)


def _inspect(trajs: list[RawTrajectory], images_root: Path) -> None:
    print("=== CoF zoom pool inspect ===")
    print(f"  Z-pool trajectories: {len(trajs)}")
    calls = [t.n_zoom_calls for t in trajs]
    if calls:
        print(f"  zoom calls / trajectory: min={min(calls)} max={max(calls)} "
              f"mean={sum(calls)/len(calls):.2f}")
    ok = _materialise(trajs[:200], images_root)
    print(f"  convert_trajectory success on first {min(200, len(trajs))}: {len(ok)}")
    traj = trajs[0]
    print("\nFirst trajectory:")
    print("  question:", traj.question[:200])
    for i, turn in enumerate(traj.turns):
        print(f"  turn {i}: calls={turn.assistant.call_categories} "
              f"text={turn.assistant.text_content[:100]!r} "
              f"resp_imgs={turn.response_image_indices}")
    if ok:
        row = ok[0]
        print("\n  emitted row: data_source=", row["data_source"],
              " n_images=", len(row["images"]),
              " roles=", [m["role"] for m in row["messages"]])


_IMAGE_PLACEHOLDER = re.compile(r"<image>")


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


_BOXED_RE = re.compile(r"\\boxed\{")


def _consume_images(content: str, cursor: int, total: int) -> tuple[list[int], int]:
    """Pull image indices out of a content string in ``<image>`` order.

    Returns ``(indices, new_cursor)``; ``new_cursor == -1`` signals the source
    referenced more images than the ``images`` list provides (drop the sample).
    """

    indices: list[int] = []
    for _ in _IMAGE_PLACEHOLDER.findall(content):
        if cursor >= total:
            return indices, -1
        indices.append(cursor)
        cursor += 1
    return indices, cursor


def _build_zoom_call(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Native CoF ``image_zoom_in_tool`` payload → OpenAI-style call dict.

    Preserves the optional ``label``. Returns ``None`` for a malformed bbox so
    the caller can drop the trajectory rather than emit a broken zoom.
    """

    args = payload.get("arguments", {}) or {}
    bbox = args.get("bbox_2d")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        bbox = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    out: dict[str, Any] = {"bbox_2d": bbox}
    label = args.get("label")
    if isinstance(label, str) and label.strip():
        out["label"] = label.strip()
    return {
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "arguments": json.dumps(out, ensure_ascii=False),
        },
    }


def _parse_cof_assistant(content: str) -> Optional[ParsedAssistantTurn]:
    """Split one CoF assistant turn into (text, [zoom call]).

    CoF assistant turns invoke at most one ``image_zoom_in_tool`` per turn. A
    turn with no ``<tool_call>`` is a terminal answer turn. Returns ``None`` if a
    ``<tool_call>`` is present but is not a valid ``image_zoom_in_tool`` call
    (unexpected tool / malformed bbox → drop the trajectory).
    """

    tool_calls: list[dict[str, Any]] = []
    categories: list[str] = []

    matches = _TOOL_CALL_RE.findall(content)
    if len(matches) > 1:
        return None  # >1 tool call per turn is out of scope (matches DeepEyesV2 rule)
    for body in matches:
        try:
            payload = json.loads(body.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("name") != "image_zoom_in_tool":
            return None  # CoF is a zoom-only source; any other tool is unexpected
        call = _build_zoom_call(payload)
        if call is None:
            return None
        tool_calls.append(call)
        categories.append("zoom")

    text = _TOOL_CALL_RE.sub("", content).strip()

    final_answer = None
    m = _ANSWER_RE.search(text)
    if m is not None:
        final_answer = m.group(1).strip()

    return ParsedAssistantTurn(
        text_content=text,
        tool_calls=tool_calls,
        call_categories=categories,
        final_answer=final_answer,
    )


def parse_one(sample: dict, raw_index: int, source_file: str) -> Optional[RawTrajectory]:
    """Parse a single CoF sample. Return ``None`` when malformed / out of scope."""

    messages = sample.get("messages", [])
    image_paths = list(sample.get("images", []))
    if not messages or not image_paths:
        return None

    if messages and messages[0].get("role") == "system":
        messages = messages[1:]
    if len(messages) < 2 or messages[0].get("role") != "user":
        return None

    cursor = 0
    user_content = messages[0].get("content", "") or ""
    user_image_indices, cursor = _consume_images(user_content, cursor, len(image_paths))
    if cursor < 0 or not user_image_indices:
        return None
    question = _IMAGE_PLACEHOLDER.sub("", user_content).strip()

    turns: list[RawTurn] = []
    categories: set[str] = set()
    sample_id = str(sample.get("id") or f"{Path(source_file).stem}:{raw_index}")

    i = 1
    while i < len(messages):
        asst = messages[i]
        if asst.get("role") != "assistant":
            return None
        parsed = _parse_cof_assistant(asst.get("content") or "")
        if parsed is None:
            return None

        cat = parsed.call_categories[0] if parsed.tool_calls else None
        if cat is not None:
            categories.add(cat)

        response_text = ""
        response_image_indices: list[int] = []
        if cat is not None:
            # A zoom call is answered by the next turn carrying exactly one crop
            # image. CoF uses a ``user`` follower; accept ``tool`` too, defensively.
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is None or nxt.get("role") not in ("user", "tool"):
                return None
            resp_content = nxt.get("content") or ""
            response_image_indices, cursor = _consume_images(
                resp_content, cursor, len(image_paths)
            )
            if cursor < 0:
                return None
            if len(response_image_indices) != 1:
                return None  # zoom → exactly one cropped image
            response_text = _IMAGE_PLACEHOLDER.sub("", resp_content).strip()
            i += 2
        else:
            i += 1

        turns.append(
            RawTurn(
                assistant=parsed,
                response_text=response_text,
                response_image_indices=response_image_indices,
                is_terminal=(cat is None),
            )
        )

    if not turns or not turns[-1].is_terminal:
        return None
    terminal_text = (turns[-1].assistant.text_content or "").strip()
    if not terminal_text:
        return None
    # The final turn must actually carry an answer we can grade against
    # (``<answer>`` gets normalized to ``\boxed{}`` at emit time).
    if turns[-1].assistant.final_answer is None and not _BOXED_RE.search(terminal_text):
        return None

    return RawTrajectory(
        question=question,
        user_image_indices=user_image_indices,
        image_paths=image_paths,
        turns=turns,
        categories=categories,
        sample_id=sample_id,
        raw_index=raw_index,
        source_file=Path(source_file).name,
    )


def parse_cof_json(path: str | Path, max_samples: Optional[int] = None) -> Iterator[RawTrajectory]:
    """Stream-parse the CoF ``cof_sft_data.json`` and yield valid trajectories."""

    path = Path(path)
    logger.info("Parsing CoF %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level list in {path}, got {type(data).__name__}")
    seen = 0
    for idx, sample in enumerate(data):
        if max_samples is not None and seen >= max_samples:
            return
        traj = parse_one(sample, raw_index=idx, source_file=path.name)
        if traj is None:
            continue
        seen += 1
        yield traj


if __name__ == "__main__":
    sys.exit(main())
