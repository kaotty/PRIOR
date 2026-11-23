"""Outcome-only reward for the legacy and two-source frozen RL pools.

CoF rows use the deterministic VQA proxy; ChartQA rows use relaxed numeric
accuracy. Tool calls are logged but never rewarded.
"""

from __future__ import annotations

from mathruler.grader import extract_boxed_content

from verl.utils import trajectory_logger
from verl.utils.reward_score import bench_w_tools, math_w_tools, vqa_w_tools


def _tool_statistics(turns: list[dict]) -> tuple[int, int, dict[str, int]]:
    calls = 0
    successful = 0
    per_tool: dict[str, int] = {}
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        tool_call = turn.get("tool_call")
        if not isinstance(tool_call, dict):
            continue
        calls += 1
        name = tool_call.get("name")
        if isinstance(name, str) and name:
            per_tool[name] = per_tool.get(name, 0) + 1
        if turn.get("executed_ok") is True:
            successful += 1
    return calls, successful, per_tool


def compute_score(solution_str: str, ground_truth: str, **kwargs):
    """Return accuracy only, dispatching the grader by frozen-pool source."""

    data_source = str(kwargs.get("data_source") or "")
    if data_source in {"vstar", "xintongzhang/CoF-RL-Data"}:
        accuracy = vqa_w_tools.acc_reward(solution_str, ground_truth)
        final_answer = vqa_w_tools._extract_answer(solution_str)
    elif data_source in {"mmk12", "hiyouga/geometry3k"}:
        accuracy = math_w_tools.acc_reward(solution_str, ground_truth)
        final_answer = extract_boxed_content(solution_str)
    elif data_source == "HuggingFaceM4/ChartQA":
        final_answer = bench_w_tools._extract(solution_str)
        accuracy = float(bench_w_tools._relaxed(final_answer, ground_truth))
    else:
        raise ValueError(f"unsupported frozen RL data_source: {data_source!r}")

    extra_info = kwargs.get("extra_info") or {}
    turns = extra_info.get("trajectory", []) or []
    calls, successful_calls, per_tool = _tool_statistics(turns)

    if trajectory_logger.is_enabled():
        record = {
            "index": extra_info.get("index"),
            "item_id": extra_info.get("item_id"),
            "row_id": extra_info.get("row_id"),
            "replicate_id": extra_info.get("replicate_id"),
            "replay_id": extra_info.get("replay_id"),
            "branch": extra_info.get("branch"),
            "eval_condition": extra_info.get("eval_condition"),
            "utility_stratum": extra_info.get("utility_stratum"),
            "factorial_tool_condition": extra_info.get("factorial_tool_condition"),
            "factorial_rollout_n": extra_info.get("factorial_rollout_n"),
            "tool_call_token_spans": extra_info.get("tool_call_token_spans"),
            "prefix_hash": extra_info.get("prefix_hash"),
            "tool_call_hash": extra_info.get("tool_call_hash"),
            "observation_hash": extra_info.get("observation_hash"),
            "data_source": data_source,
            "question": extra_info.get("question"),
            "num_turns": extra_info.get("num_turns"),
            "turns": turns,
            "final_answer": final_answer,
            "ground_truth": ground_truth,
            "correct": bool(accuracy >= 1.0),
            "reward": float(accuracy),
            "r_acc": float(accuracy),
            "num_tool_calls": calls,
            # A trajectory entry exists only after the parser accepted a
            # structured call. Execution success is a separate diagnostic.
            "valid_tool_calls": calls,
            "successful_tool_calls": successful_calls,
            "per_tool_counts": per_tool,
            "solution_str": solution_str,
        }
        # Formal counterfactual eligibility comes from a separately frozen
        # oracle-label/replay manifest. Do not emit an empty declaration here,
        # because that would falsely conflict with a non-empty replay manifest.
        if "cf_eligible_tools" in extra_info and extra_info["cf_eligible_tools"]:
            record["cf_eligible_tools"] = extra_info["cf_eligible_tools"]
        trajectory_logger.log_record(record)

    return {
        "score": float(accuracy),
        "acc": float(accuracy),
        "r_acc": float(accuracy),
        "num_tool_calls": float(calls),
        "valid_tool_calls": float(calls),
        "successful_tool_calls": float(successful_calls),
    }
