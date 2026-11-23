"""Rule-based VQA reward proxy for the DeepEyes zoom RL baseline, with trajectory logging.

WHY A SEPARATE REWARD (not math_w_tools.py)
-------------------------------------------
DeepEyes-47k answers are free-form VQA sentences (reward_model.style="model"), e.g.
"No, the car is not on the left side of the person." The upstream paper grades these with a
72B LLM-as-judge. We cannot serve a 72B judge alongside a 7B policy on 4 GPUs, and a plain
string/mathruler match fails on NEGATION ("not on the left" == "on the right" but shares no
surface tokens with a "right" answer). So this is a deliberately-scoped RULE PROXY tuned to the
vstar spatial-relation subset (the canonical zoom task):

  1. If the ground truth carries an explicit yes/no  -> grade on yes/no POLARITY match.
     (Most vstar questions are "Is A on the left of B?" -> gt "No, ..." -> correct answer is "No".)
  2. Else if a spatial RELATION word is present (left/right/above/below/front/behind/top/under)
     -> grade on the canonical relation, flipping it under negation.
  3. Else -> normalized token-F1 >= REWARD_VQA_F1 (default 0.5).

This is an APPROXIMATION of DeepEyes' LLM judge, not an equal. It is faithful on the spatial
subset and degrades gracefully elsewhere; for the full 47k mix (chart/reasoning answers) an LLM
judge would be needed. Outcome-only by default (tool use is measured, not rewarded), matching
the invocation-vs-integration design and the field consensus (rewarding the *act* of calling gets
hacked). Env overrides: REWARD_VQA_F1 (fallback threshold).
"""

import os
import re

from verl.utils import trajectory_logger

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# spatial relation -> its opposite (for negation flipping)
_OPP = {
    "left": "right", "right": "left",
    "above": "below", "below": "above",
    "top": "bottom", "bottom": "top",
    "front": "behind", "behind": "front",
    "under": "over", "over": "under",
}
_ARTICLES = {"the", "a", "an", "is", "are", "of", "to", "on", "in", "side", "it", "s"}


def _extract_answer(solution_str: str) -> str:
    """Last <answer>...</answer>; fallback to the last non-empty line."""
    m = _ANSWER_RE.findall(solution_str or "")
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in (solution_str or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _norm(s: str):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def _has_neg(toks) -> bool:
    return any(t in ("no", "not", "nt", "isnt", "arent", "cannot", "never") for t in toks)


def _yesno(toks):
    """Return 'yes' / 'no' / None from the polarity of an answer."""
    if not toks:
        return None
    if toks[0] in ("yes", "yeah", "yep") or ("yes" in toks and not _has_neg(toks)):
        return "yes"
    if toks[0] in ("no", "nope") or _has_neg(toks):
        return "no"
    return None


def _canon_relations(toks):
    """Set of spatial relations asserted, flipped under a negation anywhere in the answer."""
    neg = _has_neg(toks)
    rels = set()
    for t in toks:
        if t in _OPP:
            rels.add(_OPP[t] if neg else t)
    return rels


def _content_tokens(toks):
    return set(t for t in toks if t not in _ARTICLES and t not in _OPP
               and t not in ("yes", "no", "not", "nt"))


def acc_reward(solution_str: str, ground_truth: str) -> float:
    """1.0 iff the model's <answer> matches the ground truth under the VQA proxy rules."""
    pred = _norm(_extract_answer(solution_str))
    gt = _norm(ground_truth)
    if not pred or not gt:
        return 0.0

    # exact normalized match
    if pred == gt:
        return 1.0

    # rule 1: yes/no polarity (dominant case for vstar)
    gt_yn, pred_yn = _yesno(gt), _yesno(pred)
    if gt_yn is not None:
        return 1.0 if pred_yn == gt_yn else 0.0

    # rule 2: canonical spatial relation
    gt_rel = _canon_relations(gt)
    if gt_rel:
        pred_rel = _canon_relations(pred)
        if not pred_rel:
            return 0.0
        # correct if they assert the same relation AND share the objects
        rel_ok = bool(gt_rel & pred_rel) and not (pred_rel - gt_rel)
        objs = _content_tokens(gt)
        obj_ok = (len(objs & _content_tokens(pred)) >= max(1, len(objs) - 1)) if objs else True
        return 1.0 if (rel_ok and obj_ok) else 0.0

    # rule 3: token-F1 fallback (non-spatial answers)
    thr = float(os.environ.get("REWARD_VQA_F1", "0.5"))
    ps, gs = set(pred), set(gt)
    inter = len(ps & gs)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(ps), inter / len(gs)
    f1 = 2 * prec * rec / (prec + rec)
    return 1.0 if f1 >= thr else 0.0


def _tool_stats(solution_str: str):
    calls = _TOOL_CALL_RE.findall(solution_str or "")
    return len(calls)


def compute_score(solution_str: str, ground_truth: str, **kwargs):
    r_acc = acc_reward(solution_str, ground_truth)
    num_calls = _tool_stats(solution_str)
    final_answer = _extract_answer(solution_str)

    if trajectory_logger.is_enabled():
        extra_info = kwargs.get("extra_info") or {}
        turns = extra_info.get("trajectory", []) or []
        per_tool = {}
        for t in turns:
            tc = t.get("tool_call") if isinstance(t, dict) else None
            name = tc.get("name") if isinstance(tc, dict) else None
            if name:
                per_tool[name] = per_tool.get(name, 0) + 1
        trajectory_logger.log_record(
            {
                "index": extra_info.get("index"),
                "data_source": kwargs.get("data_source"),
                "question": extra_info.get("question"),
                "num_turns": extra_info.get("num_turns"),
                "turns": turns,
                "final_answer": final_answer,
                "ground_truth": ground_truth,
                "correct": bool(r_acc >= 1.0),
                "reward": float(r_acc),
                "r_acc": float(r_acc),
                "num_tool_calls": int(num_calls),
                "valid_tool_calls": int(num_calls),
                "per_tool_counts": per_tool,
                "solution_str": solution_str,
            }
        )

    return {
        "score": float(r_acc),
        "acc": float(r_acc),
        "r_acc": float(r_acc),
        "num_tool_calls": float(num_calls),
    }
