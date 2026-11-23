"""Opt-in agent-loop safeguards for the TIR eight-tool-call contract.

The upstream loop uses an assistant-turn ceiling. TIR sets that ceiling to
effective_tool_calls + 1, so eight calls may execute and the ninth assistant
generation is the final answer or a blocked ninth call. This module also keeps
trailing tool images synchronized with response truncation and validates the
Qwen image-token/feature count before FSDP sees a micro-batch.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys

TARGETS = {
    "verl.experimental.agent_loop.tool_agent_loop",
    "verl.experimental.agent_loop.agent_loop",
}
logger = logging.getLogger(__name__)


def effective_tool_limit() -> int:
    value = os.environ.get("TIR_EFFECTIVE_TOOL_CALL_LIMIT")
    if value is None or not value.isdigit():
        raise RuntimeError("TIR_EFFECTIVE_TOOL_CALL_LIMIT must be a nonnegative integer")
    return int(value)


def count_image_blocks(token_ids, image_token_id: int) -> int:
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if ids and isinstance(ids[0], list):
        ids = [item for row in ids for item in row]
    count = 0
    previous = False
    for token in ids:
        current = int(token) == int(image_token_id)
        count += int(current and not previous)
        previous = current
    return count


def terminate_at_unbacked_generated_image(output, image_token_id: int) -> int:
    """Terminate a rollout before an image token sampled by the language model.

    Image tokens supplied by the prompt or a Zoom observation have backing image
    features.  In the response, those observation tokens have response_mask=0;
    tokens sampled by the model have response_mask=1.  A sampled image token has
    no corresponding image features and cannot be scored by Qwen-VL.  Preserve
    the valid on-policy prefix and mark the malformed action as a terminal
    rollout instead of letting one sample abort the whole GRPO batch.
    """
    response_ids = list(output.response_ids)
    response_mask = list(output.response_mask)
    if len(response_ids) != len(response_mask):
        raise RuntimeError(
            f"TIR image contract: response ids/mask length mismatch "
            f"({len(response_ids)} != {len(response_mask)})"
        )
    invalid = [
        index
        for index, (token, generated) in enumerate(zip(response_ids, response_mask, strict=True))
        if int(token) == int(image_token_id) and int(generated) == 1
    ]
    if not invalid:
        return 0

    cut = invalid[0]
    removed = len(response_ids) - cut
    # Generated spans are separated by tool-observation spans (mask=0).  Keep
    # diagnostics for completed earlier turns and replace the malformed turn
    # with one explicit terminal record; calls produced after the cut must not
    # be credited as attempted or successful.
    generated_turn = 0
    was_generated = False
    for generated in response_mask[: cut + 1]:
        is_generated = int(generated) == 1
        if is_generated and not was_generated:
            generated_turn += 1
        was_generated = is_generated
    output.response_ids = response_ids[:cut]
    output.response_mask = response_mask[:cut]
    if output.response_logprobs is not None:
        output.response_logprobs = list(output.response_logprobs)[:cut]
    spans = output.extra_fields.get("tool_call_token_spans")
    if isinstance(spans, list):
        output.extra_fields["tool_call_token_spans"] = [
            span for span in spans if int(span.get("end", cut + 1)) <= cut
        ]
    trajectory = list(output.extra_fields.get("trajectory") or [])
    completed = trajectory[: max(0, generated_turn - 1)]
    current = trajectory[generated_turn - 1] if 0 < generated_turn <= len(trajectory) else {}
    completed.append(
        {
            "turn": generated_turn,
            "reasoning": current.get("reasoning"),
            "tool_call": None,
            "tool_output_text": None,
            "num_output_images": 0,
            "tool_reward": None,
            "executed_ok": None,
            "status": "invalid_generated_image_token",
        }
    )
    output.extra_fields["trajectory"] = completed
    output.extra_fields["termination_reason"] = "invalid_generated_image_token"
    output.extra_fields["tir_unbacked_generated_image_tokens"] = len(invalid)
    output.extra_fields["tir_unbacked_generated_image_cut"] = cut
    logger.warning(
        "TIR terminated malformed rollout at generated image token %d; dropped %d response tokens",
        cut,
        removed,
    )
    return removed


def trim_unrepresented_trailing_images(output, image_token_id: int) -> int:
    data = output.multi_modal_data or {}
    images = data.get("images")
    if images is None:
        return 0
    images = list(images) if isinstance(images, (list, tuple)) else [images]
    represented = count_image_blocks([*output.prompt_ids, *output.response_ids], image_token_id)
    if represented > len(images):
        raise RuntimeError(f"TIR image contract: {represented} token blocks but only {len(images)} images")
    removed = len(images) - represented
    if removed:
        updated = dict(data)
        updated["images"] = images[:represented]
        output.multi_modal_data = updated
        output.extra_fields["tir_truncated_trailing_images"] = removed
        logger.warning("TIR removed %d trailing images not represented after response truncation", removed)
    return removed


def relabel_final_answer_at_budget(output) -> bool:
    if output.extra_fields.get("termination_reason") != "assistant_turn_limit":
        return False
    trajectory = output.extra_fields.get("trajectory") or []
    if not trajectory or trajectory[-1].get("tool_call") is not None:
        return False
    if trajectory[-1].get("status") != "assistant_turn_limit":
        return False
    trajectory[-1]["status"] = "final_answer"
    output.extra_fields.pop("termination_reason", None)
    output.extra_fields["tir_final_answer_at_tool_budget"] = True
    return True


def qwen_image_feature_count(processor, multi_modal_inputs) -> int | None:
    grid = multi_modal_inputs.get("image_grid_thw")
    if grid is None:
        return None
    merge_size = int(getattr(processor.image_processor, "merge_size", 1))
    return int((grid.prod(dim=-1) // (merge_size * merge_size)).sum().item())


def patch_tool_agent(module) -> None:
    cls = module.ToolAgentLoop
    if getattr(cls, "_tir_horizon_compat", False):
        return
    original_run = cls.run

    async def run(self, sampling_params, **kwargs):
        limit = effective_tool_limit()
        if int(self.max_assistant_turns or 0) != limit + 1:
            raise RuntimeError(
                f"TIR tool budget mismatch: max_assistant_turns={self.max_assistant_turns}, expected {limit + 1}"
            )
        output = await original_run(self, sampling_params, **kwargs)
        if self.processor is not None and hasattr(self.processor, "image_token_id"):
            terminate_at_unbacked_generated_image(output, self.processor.image_token_id)
            trim_unrepresented_trailing_images(output, self.processor.image_token_id)
        relabel_final_answer_at_budget(output)
        return output

    cls.run = run
    cls._tir_horizon_compat = True
    print("[tir-agent-compat] eight-tool-call horizon safeguards enabled", flush=True)


def patch_agent_worker(module) -> None:
    cls = module.AgentLoopWorker
    if getattr(cls, "_tir_image_contract_compat", False):
        return
    original = cls._compute_multi_modal_inputs

    def compute_multi_modal_inputs(self, output, input_ids):
        result = original(self, output, input_ids)
        if self.processor is None or not hasattr(self.processor, "image_token_id"):
            return result
        actual = int((input_ids == self.processor.image_token_id).sum().item())
        expected = qwen_image_feature_count(self.processor, result)
        if expected is None or actual == expected:
            return result
        data = output.multi_modal_data or {}
        images = data.get("images")
        videos = data.get("videos")
        if videos or not isinstance(images, (list, tuple)) or len(images) <= 1 or actual > expected:
            raise RuntimeError(
                f"TIR image contract cannot repair token/features mismatch: tokens={actual}, features={expected}"
            )
        # Response truncation can retain the token prefix while the output still
        # holds trailing Zoom images. Reprocess successively shorter prefixes;
        # accept only an exact token/feature match and record the intervention.
        for keep in range(len(images) - 1, 0, -1):
            trial = output.model_copy(deep=False)
            trial.multi_modal_data = dict(data)
            trial.multi_modal_data["images"] = list(images[:keep])
            candidate = original(self, trial, input_ids)
            if qwen_image_feature_count(self.processor, candidate) == actual:
                removed = len(images) - keep
                output.multi_modal_data = trial.multi_modal_data
                output.extra_fields["tir_repaired_trailing_images"] = removed
                logger.warning(
                    "TIR repaired image-token contract by dropping %d trailing truncated images", removed
                )
                return candidate
        raise RuntimeError(
            f"TIR image contract found non-prefix mismatch: tokens={actual}, features={expected}, images={len(images)}"
        )

    cls._compute_multi_modal_inputs = compute_multi_modal_inputs
    cls._tir_image_contract_compat = True
    print("[tir-agent-compat] per-sample image-token contract enabled", flush=True)


def patch_module(module) -> None:
    if module.__name__.endswith("tool_agent_loop"):
        patch_tool_agent(module)
    elif module.__name__.endswith("agent_loop"):
        patch_agent_worker(module)


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        patch_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    if os.environ.get("TIR_AGENT_HORIZON_COMPAT") != "1":
        return
    for name in TARGETS:
        if name in sys.modules:
            patch_module(sys.modules[name])
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
