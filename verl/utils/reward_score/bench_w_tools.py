"""Benchmark-aware reward for the eval arm (MathVista / MathVision / ChartQA / MMMU), with logging.

⚠️ This APPROXIMATES the official scorers (which live in VLMEvalKit). It is for comparing OUR
arms, NOT for citing leaderboard numbers. Grades per data_source:
  *_mcq   -> extract an option LETTER from the response; correct if it matches the gold letter,
             OR (fallback) the response equals/contains the gold option TEXT (extra_info.gt_text).
  chartqa -> relaxed accuracy: numeric within 5% relative (VLMEvalKit default), else exact match.
  *_free / mmmu_open -> mathruler grade_answer (math-equivalence), plus a numeric-tolerant fallback.
Outcome-only; tool use is measured (num_tool_calls logged) not rewarded.
"""

import os
import re

from mathruler.grader import extract_boxed_content, grade_answer

from verl.utils import trajectory_logger

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_BOX_RE = re.compile(r"\\boxed\{")
_SUPPORTED_TOOLS = {"image_zoom_in_tool", "code_interpreter"}
_BLOCKED_STATUSES = {
    "response_length_limit",
    "assistant_turn_limit",
    "user_turn_limit",
    "duplicate_call_limit",
}


def _extract(solution_str: str) -> str:
    """Prefer the last \\boxed{...}; then the last <answer>...; then the last non-empty line."""
    s = solution_str or ""
    if _BOX_RE.search(s):
        b = extract_boxed_content(s)
        if b:
            return b.strip()
    m = _ANSWER_RE.findall(s)
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _option_letter(text: str):
    """Extract a single option letter A-H from a short model answer, else None."""
    t = (text or "").strip()
    if len(t) == 1 and t.upper() in "ABCDEFGH":
        return t.upper()
    # patterns like "(C)", "C.", "C)", "answer: C", "option B"
    m = re.search(r"\b([A-H])\b", t.upper())
    return m.group(1) if m else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.\-/%]", " ", (s or "").lower())).strip()


def _to_float(s: str):
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _relaxed(pred: str, gt: str) -> bool:
    """VLMEvalKit-style relaxed accuracy: numeric within 5% relative, else normalized exact."""
    p, g = _to_float(pred), _to_float(gt)
    if p is not None and g is not None:
        return abs(p - g) <= 0.05 * abs(g) if g != 0 else abs(p) <= 1e-6
    np_, ng = _norm(pred), _norm(gt)
    return bool(np_) and np_ == ng


def _mcq_correct(pred_boxed: str, gt_letter: str, gt_text) -> bool:
    gl = (gt_letter or "").strip().upper()
    if _option_letter(pred_boxed) == gl and gl:
        return True
    # fallback: model boxed the option TEXT rather than the letter
    if gt_text:
        nt = _norm(gt_text)
        if nt and _norm(pred_boxed) == nt:
            return True
    return False


def acc_reward(solution_str: str, ground_truth: str, data_source: str, extra_info: dict) -> float:
    pred = _extract(solution_str)
    ds = data_source or ""
    if ds.endswith("_mcq"):
        ok = _mcq_correct(pred, ground_truth, (extra_info or {}).get("gt_text"))
    elif ds == "chartqa":
        ok = _relaxed(pred, ground_truth)
    else:  # *_free, mmmu_open, or unknown -> math grader + numeric fallback
        ok = bool(grade_answer(pred, ground_truth)) or _relaxed(pred, ground_truth)
    return 1.0 if ok else 0.0


def _num_tool_calls(solution_str: str) -> int:
    return len(_TOOL_CALL_RE.findall(solution_str or ""))


def _tool_statistics(turns: list[dict], solution_str: str) -> dict:
    parsed = 0
    parsed_supported = 0
    attempted_supported = 0
    successful_supported = 0
    failed_supported = 0
    blocked_supported = 0
    unknown = 0
    attempted_by_tool: dict[str, int] = {}
    successful_by_tool: dict[str, int] = {}
    for turn in turns:
        call = turn.get("tool_call") if isinstance(turn, dict) else None
        name = call.get("name") if isinstance(call, dict) else None
        if not name:
            continue
        parsed += 1
        if name not in _SUPPORTED_TOOLS:
            unknown += 1
            continue
        parsed_supported += 1
        if turn.get("status") in _BLOCKED_STATUSES:
            blocked_supported += 1
            continue
        attempted_supported += 1
        attempted_by_tool[name] = attempted_by_tool.get(name, 0) + 1
        if turn.get("executed_ok") is True:
            successful_supported += 1
            successful_by_tool[name] = successful_by_tool.get(name, 0) + 1
        else:
            failed_supported += 1
    # Count malformed textual calls even when the same trajectory also contains
    # ordinary parsed turns.  The raw response retains every <tool_call> marker;
    # subtract calls already represented in the structured trajectory.
    textual_unparsed = max(0, _num_tool_calls(solution_str) - parsed)
    parsed += textual_unparsed
    return {
        "parsed": parsed,
        "parsed_supported": parsed_supported,
        "attempted_supported": attempted_supported,
        "successful_supported": successful_supported,
        "failed_supported": failed_supported,
        "blocked_supported": blocked_supported,
        "unknown": unknown,
        "textual_unparsed": textual_unparsed,
        "attempted_by_tool": attempted_by_tool,
        "successful_by_tool": successful_by_tool,
    }


def _termination_reason(extra_info: dict, turns: list[dict]) -> tuple[str | None, bool]:
    reason = extra_info.get("termination_reason")
    has_final = any(
        turn.get("tool_call") is None and turn.get("status") == "final_answer"
        for turn in turns
        if isinstance(turn, dict)
    )
    inferred = False
    if not reason and turns and not has_final:
        last = turns[-1]
        if isinstance(last.get("tool_call"), dict) and last.get("executed_ok") is True:
            # ToolAgentLoop currently terminates before appending an observation
            # that would overflow response_length. Preserve that distinct event
            # instead of silently folding it into ordinary tool use.
            reason = "tool_response_length_limit_inferred"
            inferred = True
    return reason, inferred


def compute_score(solution_str: str, ground_truth: str, **kwargs):
    data_source = kwargs.get("data_source")
    extra_info = kwargs.get("extra_info") or {}
    r_acc = acc_reward(solution_str, ground_truth, data_source, extra_info)
    turns = extra_info.get("trajectory", []) or []
    tool_stats = _tool_statistics(turns, solution_str)
    termination_reason, inferred_termination = _termination_reason(extra_info, turns)

    if trajectory_logger.is_enabled():
        trajectory_logger.log_record({
            "index": extra_info.get("index"),
            "item_id": extra_info.get("item_id"),
            "row_id": extra_info.get("row_id"),
            "replicate_id": extra_info.get("replicate_id"),
            "replay_id": extra_info.get("replay_id"),
            "branch": extra_info.get("branch"),
            "tool": extra_info.get("tool"),
            "prefix_hash": extra_info.get("prefix_hash"),
            "tool_call_hash": extra_info.get("tool_call_hash"),
            "observation_hash": extra_info.get("observation_hash"),
            "observation_kind": extra_info.get("observation_kind"),
            "trajectory_schema_version": "tir-eval-trajectory-v2",
            "termination_reason": termination_reason,
            "termination_reason_inferred": inferred_termination,
            "duplicate_call_terminated": bool(extra_info.get("duplicate_call_terminated", False)),
            "duplicate_call_guard": extra_info.get("duplicate_call_guard"),
            "consecutive_tool_failures": extra_info.get("consecutive_tool_failures"),
            "data_source": data_source,
            "question": extra_info.get("question"),
            "num_turns": extra_info.get("num_turns"),
            "turns": turns,
            "final_answer": _extract(solution_str),
            "ground_truth": ground_truth,
            "correct": bool(r_acc >= 1.0),
            "reward": float(r_acc),
            "r_acc": float(r_acc),
            # Legacy fields are retained, with explicit v2 fields defining the
            # semantics used by all new summaries.
            "num_tool_calls": int(tool_stats["parsed"]),
            "valid_tool_calls": int(tool_stats["parsed_supported"]),
            "successful_tool_calls": int(tool_stats["successful_supported"]),
            "parsed_tool_calls": int(tool_stats["parsed"]),
            "parsed_supported_calls": int(tool_stats["parsed_supported"]),
            "attempted_supported_calls": int(tool_stats["attempted_supported"]),
            "successful_supported_calls": int(tool_stats["successful_supported"]),
            "failed_supported_calls": int(tool_stats["failed_supported"]),
            "blocked_supported_calls": int(tool_stats["blocked_supported"]),
            "unknown_tool_calls": int(tool_stats["unknown"]),
            "unparsed_textual_tool_calls": int(tool_stats["textual_unparsed"]),
            "per_tool_counts": tool_stats["attempted_by_tool"],
            "successful_per_tool_counts": tool_stats["successful_by_tool"],
            "solution_str": solution_str,
        })

    return {
        "score": float(r_acc),
        "acc": float(r_acc),
        "r_acc": float(r_acc),
        "num_tool_calls": float(tool_stats["parsed"]),
        "valid_tool_calls": float(tool_stats["parsed_supported"]),
        "successful_tool_calls": float(tool_stats["successful_supported"]),
        "attempted_supported_calls": float(tool_stats["attempted_supported"]),
        "blocked_supported_calls": float(tool_stats["blocked_supported"]),
        "unknown_tool_calls": float(tool_stats["unknown"]),
    }
