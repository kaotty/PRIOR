"""Factorial Zoom/Code credit routed only to structured tool-call tokens.

Each prompt has a balanced rollout group under four tool-availability cells:
None, Zoom, Code, and Both. Final-answer reward remains the ordinary GRPO
signal. This module adds a small, per-tool factorial credit to the policy tokens
that formed a successfully executed call to that tool.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from verl.experimental.agent_loop.factorial_tool_availability import (
    CODE_TOOL_NAME,
    FACTORIAL_CONDITIONS,
    ZOOM_TOOL_NAME,
)


@dataclass(frozen=True)
class FactorialCredits:
    """Per-trajectory raw credits and per-prompt cell diagnostics."""

    zoom: torch.Tensor
    code: torch.Tensor
    interaction: torch.Tensor
    cell_means: dict[str, torch.Tensor]


def _to_python_list(values: Sequence[Any] | np.ndarray | torch.Tensor) -> list[Any]:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def compute_factorial_credits(
    task_scores: torch.Tensor,
    uids: Sequence[Any] | np.ndarray,
    conditions: Sequence[Any] | np.ndarray,
    *,
    rollouts_per_condition: int = 2,
) -> FactorialCredits:
    """Compute rollout-specific Zoom and Code credits.

    Zoom-only and Code-only rollouts use their own outcome relative to the None
    cell mean. Both rollouts use the rollout-specific Shapley split. Averaging
    the two Both credits recovers the standard two-factor Shapley effect.
    """

    if task_scores.ndim != 1:
        raise ValueError("task_scores must be a rank-1 tensor")
    if not torch.isfinite(task_scores).all():
        raise ValueError("factorial task_scores must be finite")
    if torch.any((task_scores < 0) | (task_scores > 1)):
        raise ValueError("factorial task_scores must be accuracy rewards in [0, 1]")
    if rollouts_per_condition <= 0:
        raise ValueError("rollouts_per_condition must be positive")

    uid_list = [str(value) for value in _to_python_list(uids)]
    condition_list = [str(value) for value in _to_python_list(conditions)]
    if len(uid_list) != len(task_scores) or len(condition_list) != len(task_scores):
        raise ValueError("task_scores, uids, and conditions must have the same length")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(uid_list):
        grouped[uid].append(index)

    zoom = torch.zeros_like(task_scores, dtype=torch.float32)
    code = torch.zeros_like(task_scores, dtype=torch.float32)
    interaction = torch.zeros_like(task_scores, dtype=torch.float32)
    cell_means = {
        condition: torch.zeros_like(task_scores, dtype=torch.float32) for condition in FACTORIAL_CONDITIONS
    }

    for uid, indices in grouped.items():
        by_condition: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            condition = condition_list[index]
            if condition not in FACTORIAL_CONDITIONS:
                raise ValueError(f"uid={uid} has unknown factorial condition {condition!r}")
            by_condition[condition].append(index)

        counts = {condition: len(by_condition[condition]) for condition in FACTORIAL_CONDITIONS}
        expected = {condition: rollouts_per_condition for condition in FACTORIAL_CONDITIONS}
        if counts != expected:
            raise ValueError(f"uid={uid} has unbalanced factorial cells: expected {expected}, got {counts}")

        means = {
            condition: task_scores[by_condition[condition]].to(torch.float32).mean()
            for condition in FACTORIAL_CONDITIONS
        }
        group_interaction = means["both"] - means["zoom"] - means["code"] + means["none"]

        for index in indices:
            condition = condition_list[index]
            score = task_scores[index].to(torch.float32)
            interaction[index] = group_interaction
            for cell in FACTORIAL_CONDITIONS:
                cell_means[cell][index] = means[cell]

            if condition == "zoom":
                zoom[index] = score - means["none"]
            elif condition == "code":
                code[index] = score - means["none"]
            elif condition == "both":
                zoom[index] = 0.5 * ((means["zoom"] - means["none"]) + (score - means["code"]))
                code[index] = 0.5 * ((means["code"] - means["none"]) + (score - means["zoom"]))

    return FactorialCredits(zoom=zoom, code=code, interaction=interaction, cell_means=cell_means)


def normalized_tool_call_masks(
    tool_call_token_spans: Sequence[Any] | np.ndarray,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one unit-sum mask per tool and trajectory.

    Only successfully executed spans are included. A trajectory's total weight
    for a tool is one regardless of the number or serialized length of calls.
    """

    if response_mask.ndim != 2:
        raise ValueError("response_mask must be rank 2")
    span_rows = _to_python_list(tool_call_token_spans)
    if len(span_rows) != response_mask.shape[0]:
        raise ValueError("tool_call_token_spans must have one row per trajectory")

    zoom_mask = torch.zeros_like(response_mask, dtype=torch.float32)
    code_mask = torch.zeros_like(response_mask, dtype=torch.float32)
    masks = {ZOOM_TOOL_NAME: zoom_mask, CODE_TOOL_NAME: code_mask}
    response_length = response_mask.shape[1]

    for row_index, spans in enumerate(span_rows):
        if spans is None:
            continue
        if not isinstance(spans, Iterable) or isinstance(spans, (str, bytes, Mapping)):
            raise ValueError(f"tool_call_token_spans[{row_index}] must be a list of records")
        for span in spans:
            if not isinstance(span, Mapping):
                raise ValueError(f"tool_call_token_spans[{row_index}] contains a non-record value")
            tool_name = span.get("name")
            if tool_name not in masks or span.get("executed_ok") is not True:
                continue
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            if start < 0 or end <= start or end > response_length:
                raise ValueError(
                    f"invalid tool-call token span at row {row_index}: start={start}, end={end}, "
                    f"response_length={response_length}"
                )
            masks[tool_name][row_index, start:end] = response_mask[row_index, start:end].to(torch.float32)

    for mask in masks.values():
        denominator = mask.sum(dim=1, keepdim=True)
        mask.div_(denominator.clamp_min(1.0))
    return zoom_mask, code_mask


def apply_factorial_tool_credit(data, config) -> tuple[Any, dict[str, float]]:
    """Add the optional tool-credit channel to an already computed GRPO advantage."""

    enabled = bool(config.get("enable", False))
    if not enabled:
        return data, {}
    credit_coef = float(config.get("credit_coef", 0.2))
    rollouts_per_condition = int(config.get("rollouts_per_condition", 2))
    if credit_coef < 0:
        raise ValueError("factorial tool credit_coef must be non-negative")

    required_non_tensor = {"uid", "factorial_tool_condition", "tool_call_token_spans"}
    missing = required_non_tensor - set(data.non_tensor_batch)
    if missing:
        raise ValueError(f"factorial tool credit is missing rollout metadata: {sorted(missing)}")
    if "advantages" not in data.batch or "token_level_scores" not in data.batch:
        raise ValueError("factorial tool credit requires computed advantages and token_level_scores")

    task_scores = data.batch["token_level_scores"].sum(dim=-1).to(torch.float32)
    credits = compute_factorial_credits(
        task_scores,
        data.non_tensor_batch["uid"],
        data.non_tensor_batch["factorial_tool_condition"],
        rollouts_per_condition=rollouts_per_condition,
    )
    zoom_mask, code_mask = normalized_tool_call_masks(
        data.non_tensor_batch["tool_call_token_spans"], data.batch["response_mask"]
    )

    zoom_has_call = zoom_mask.sum(dim=1) > 0
    code_has_call = code_mask.sum(dim=1) > 0
    effective_zoom = credits.zoom.to(zoom_mask.device) * zoom_has_call
    effective_code = credits.code.to(code_mask.device) * code_has_call
    advantage_delta = credit_coef * (
        effective_zoom[:, None] * zoom_mask + effective_code[:, None] * code_mask
    )
    data.batch["advantages"] = data.batch["advantages"] + advantage_delta.to(data.batch["advantages"].dtype)

    data.non_tensor_batch["factorial_zoom_credit"] = effective_zoom.detach().cpu().numpy()
    data.non_tensor_batch["factorial_code_credit"] = effective_code.detach().cpu().numpy()
    data.non_tensor_batch["factorial_interaction"] = credits.interaction.detach().cpu().numpy()

    conditions = np.asarray(data.non_tensor_batch["factorial_tool_condition"], dtype=object)
    zoom_available = torch.as_tensor(
        np.isin(conditions, ["zoom", "both"]), dtype=torch.bool, device=zoom_has_call.device
    )
    code_available = torch.as_tensor(
        np.isin(conditions, ["code", "both"]), dtype=torch.bool, device=code_has_call.device
    )
    zoom_credit_on_calls = effective_zoom[zoom_has_call]
    code_credit_on_calls = effective_code[code_has_call]
    metrics: dict[str, float] = {
        "factorial/zoom_credit_mean": float(
            zoom_credit_on_calls.mean().item() if zoom_credit_on_calls.numel() else 0.0
        ),
        "factorial/code_credit_mean": float(
            code_credit_on_calls.mean().item() if code_credit_on_calls.numel() else 0.0
        ),
        "factorial/zoom_credit_mean_all": float(effective_zoom.mean().item()),
        "factorial/code_credit_mean_all": float(effective_code.mean().item()),
        "factorial/interaction_mean": float(credits.interaction.mean().item()),
        "factorial/zoom_executed_rate": float(zoom_has_call[zoom_available].to(torch.float32).mean().item()),
        "factorial/code_executed_rate": float(code_has_call[code_available].to(torch.float32).mean().item()),
        "factorial/tool_advantage_abs_mean": float(advantage_delta.abs().sum(dim=1).mean().item()),
    }
    for condition in FACTORIAL_CONDITIONS:
        selected = torch.as_tensor(conditions == condition, device=task_scores.device)
        metrics[f"factorial/reward_{condition}"] = float(task_scores[selected].mean().item())
    return data, metrics
