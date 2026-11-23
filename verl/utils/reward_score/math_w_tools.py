"""Boxed-answer math reward with optional format diagnostics and trajectory logging.

Accuracy is the only nonzero reward component by default. The optional
REWARD_W_BASE_FMT, REWARD_W_TOOL_FMT and REWARD_W_PENALTY environment variables
control extra shaping terms; the frozen experiment recipes leave them at zero.
"""

import json
import os
import re

from mathruler.grader import extract_boxed_content, grade_answer

from verl.utils import trajectory_logger

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_BASE_FMT_RE = re.compile(r"<think>.*?</think>.*?\\boxed\{.*?\}", re.DOTALL)


def acc_reward(solution_str: str, ground_truth: str) -> float:
    """1.0 iff the LAST \\boxed{} in the FULL response matches ground truth.

    Full string (no ``[-300:]``) so long tool-heavy trajectories are not cut off,
    and boxed extraction (not an ``Answer:`` regex) so tool stdout cannot spoof
    the answer.
    """
    return 1.0 if grade_answer(extract_boxed_content(solution_str), ground_truth) else 0.0


def base_format_reward(solution_str: str) -> float:
    """1.0 if the response contains a <think> block followed by a \\boxed{}."""
    return 1.0 if _BASE_FMT_RE.search(solution_str) else 0.0


def tool_stats(solution_str: str):
    """Count <tool_call> blocks and how many carry parseable JSON.

    Returned as diagnostics regardless of whether they are rewarded.
    """
    calls = _TOOL_CALL_RE.findall(solution_str)
    num_calls = len(calls)
    valid = 0
    for call_str in calls:
        try:
            if isinstance(json.loads(call_str.strip()), dict):
                valid += 1
        except json.JSONDecodeError:
            pass
    tool_format_score = float(valid) / float(num_calls) if num_calls else 0.0
    # Optional shaping counts only calls beyond the second.
    penalty = float(num_calls - 2) / float(num_calls) if num_calls > 2 else 0.0
    return tool_format_score, penalty, num_calls, valid


def compute_score(solution_str: str, ground_truth: str, **kwargs):
    w_acc = float(os.environ.get("REWARD_W_ACC", "1.0"))
    w_base_fmt = float(os.environ.get("REWARD_W_BASE_FMT", "0.0"))
    w_tool_fmt = float(os.environ.get("REWARD_W_TOOL_FMT", "0.0"))
    w_penalty = float(os.environ.get("REWARD_W_PENALTY", "0.0"))

    r_acc = acc_reward(solution_str, ground_truth)
    r_base_fmt = base_format_reward(solution_str)
    r_tool_fmt, r_penalty, num_calls, valid_calls = tool_stats(solution_str)

    final_score = (
        w_acc * r_acc
        + w_base_fmt * r_base_fmt
        + w_tool_fmt * r_tool_fmt
        - w_penalty * r_penalty
    )

    boxed = extract_boxed_content(solution_str)
    # Diagnostic for the grader-truncation bias we are avoiding: how far from the
    # end of the response does the boxed answer sit? math_dapo's [-300:] window
    # would have MISSED anything beyond 300 chars, which is exactly the bias that
    # penalises long tool-using trajectories. Logging it lets us quantify it.
    box_idx = solution_str.rfind("\\boxed{")
    chars_after_box = (len(solution_str) - box_idx) if box_idx >= 0 else -1

    # Env-gated (TRAJECTORY_LOG_DIR). No-op otherwise; never raises.
    if trajectory_logger.is_enabled():
        extra_info = kwargs.get("extra_info") or {}
        turns = extra_info.get("trajectory", []) or []
        per_tool_counts = {}
        for t in turns:
            tc = t.get("tool_call") if isinstance(t, dict) else None
            name = tc.get("name") if isinstance(tc, dict) else None
            if name:
                per_tool_counts[name] = per_tool_counts.get(name, 0) + 1
        trajectory_logger.log_record(
            {
                "index": extra_info.get("index"),
                "data_source": kwargs.get("data_source"),
                "question": extra_info.get("question"),
                "num_turns": extra_info.get("num_turns"),
                "turns": turns,
                "final_answer": boxed,
                "ground_truth": ground_truth,
                "correct": bool(r_acc >= 1.0),
                "reward": float(final_score),
                "r_acc": float(r_acc),
                "r_base_fmt": float(r_base_fmt),
                "r_tool_fmt": float(r_tool_fmt),
                "r_penalty": float(r_penalty),
                "num_tool_calls": int(num_calls),
                "valid_tool_calls": int(valid_calls),
                "per_tool_counts": per_tool_counts,
                "chars_after_box": int(chars_after_box),
                "solution_str": solution_str,
            }
        )

    return {
        "score": float(final_score),
        "acc": float(r_acc),          # val core metric picks this up
        "r_acc": float(r_acc),
        "r_base_fmt": float(r_base_fmt),
        "r_tool_fmt": float(r_tool_fmt),
        "r_penalty": float(r_penalty),
        "num_tool_calls": float(num_calls),
        "valid_tool_calls": float(valid_calls),
        "chars_after_box": float(chars_after_box),
    }
