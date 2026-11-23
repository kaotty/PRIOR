"""Shared schedule for factorial tool-availability rollouts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FACTORIAL_CONDITIONS = ("none", "zoom", "code", "both")
ZOOM_TOOL_NAME = "image_zoom_in_tool"
CODE_TOOL_NAME = "code_interpreter"

_CONDITION_TO_TOOL_NAMES = {
    "none": frozenset(),
    "zoom": frozenset({ZOOM_TOOL_NAME}),
    "code": frozenset({CODE_TOOL_NAME}),
    "both": frozenset({ZOOM_TOOL_NAME, CODE_TOOL_NAME}),
}


def validate_factorial_schedule(rollout_n: int, rollouts_per_condition: int) -> None:
    """Require a balanced four-cell rollout group."""

    if rollouts_per_condition <= 0:
        raise ValueError("factorial rollouts_per_condition must be positive")
    expected = len(FACTORIAL_CONDITIONS) * rollouts_per_condition
    if rollout_n != expected:
        raise ValueError(
            f"factorial tool availability requires rollout.n={expected} "
            f"({rollouts_per_condition} per condition), got {rollout_n}"
        )


def condition_for_rollout(rollout_index: int, rollouts_per_condition: int) -> str:
    """Map an interleaved rollout index to none/zoom/code/both."""

    if rollout_index < 0:
        raise ValueError("rollout_index must be non-negative")
    total = len(FACTORIAL_CONDITIONS) * rollouts_per_condition
    if rollout_index >= total:
        raise ValueError(f"rollout_index {rollout_index} is outside the balanced group of size {total}")
    return FACTORIAL_CONDITIONS[rollout_index // rollouts_per_condition]


def allowed_tool_names(condition: str) -> frozenset[str]:
    """Return the exact tool allowlist for one factorial condition."""

    try:
        return _CONDITION_TO_TOOL_NAMES[condition]
    except KeyError as exc:
        raise ValueError(f"unknown factorial tool condition: {condition!r}") from exc


def find_delimited_token_spans(
    token_ids: list[int], start_token_id: int, end_token_id: int
) -> list[tuple[int, int]]:
    """Return complete non-overlapping ``[start, end)`` spans."""

    spans = []
    cursor = 0
    while cursor < len(token_ids):
        try:
            start = token_ids.index(start_token_id, cursor)
            end = token_ids.index(end_token_id, start + 1)
        except ValueError:
            break
        spans.append((start, end + 1))
        cursor = end + 1
    return spans


def factorial_metadata_for_uids(
    uids: Sequence[Any], rollouts_per_condition: int
) -> tuple[list[str], list[int]]:
    """Build and validate the condition schedule before worker sharding."""

    counts: dict[str, int] = {}
    conditions = []
    rollout_indices = []
    for raw_uid in uids:
        uid = str(raw_uid)
        rollout_index = counts.get(uid, 0)
        conditions.append(condition_for_rollout(rollout_index, rollouts_per_condition))
        rollout_indices.append(rollout_index)
        counts[uid] = rollout_index + 1

    expected = len(FACTORIAL_CONDITIONS) * rollouts_per_condition
    bad_counts = {uid: count for uid, count in counts.items() if count != expected}
    if bad_counts:
        preview = dict(list(bad_counts.items())[:5])
        raise ValueError(f"factorial rollout groups must each contain {expected} items; bad uid counts: {preview}")
    return conditions, rollout_indices
