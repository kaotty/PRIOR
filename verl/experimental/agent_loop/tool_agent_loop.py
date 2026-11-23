# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.factorial_tool_availability import (
    CODE_TOOL_NAME,
    ZOOM_TOOL_NAME,
    allowed_tool_names,
    find_delimited_token_spans,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils import trajectory_logger
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        allowed_tools: Optional[frozenset[str]] = None,
        factorial_condition: Optional[str] = None,
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.allowed_tools = allowed_tools
        self.factorial_condition = factorial_condition
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.active_tool_call_span_indices: list[int] = []
        self.tool_call_token_spans: list[dict[str, Any]] = []
        self.last_tool_call_key: Optional[tuple[str, str]] = None
        self.consecutive_identical_tool_calls = 0
        self.tool_call_key_occurrences: dict[tuple[str, str], int] = {}
        self.consecutive_tool_failures = 0

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_tokens_per_turn = self.rollout_config.multi_turn.max_tokens_per_turn
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        self.max_consecutive_identical_tool_calls = int(
            self.rollout_config.multi_turn.get("max_consecutive_identical_tool_calls", 0)
        )
        self.max_identical_tool_call_occurrences = int(
            self.rollout_config.multi_turn.get("max_identical_tool_call_occurrences", 0)
        )
        self.max_consecutive_tool_failures = int(
            self.rollout_config.multi_turn.get("max_consecutive_tool_failures", 0)
        )
        if self.max_consecutive_identical_tool_calls < 0:
            raise ValueError("multi_turn.max_consecutive_identical_tool_calls must be non-negative")
        if self.max_identical_tool_call_occurrences < 0:
            raise ValueError("multi_turn.max_identical_tool_call_occurrences must be non-negative")
        if self.max_consecutive_tool_failures < 0:
            raise ValueError("multi_turn.max_consecutive_tool_failures must be non-negative")
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schema_by_name = {
            tool.name: tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
        }
        self.tool_schemas = list(self.tool_schema_by_name.values())
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format
        self.factorial_tool_availability = bool(
            self.rollout_config.multi_turn.get("factorial_tool_availability", False)
        )
        if self.factorial_tool_availability:
            expected_tools = {ZOOM_TOOL_NAME, CODE_TOOL_NAME}
            if set(self.tools) != expected_tools:
                raise ValueError(
                    "factorial tool availability requires exactly image_zoom_in_tool and code_interpreter; "
                    f"loaded {sorted(self.tools)}"
                )
            if self.tool_parser_name != "hermes":
                raise ValueError("factorial tool-call token masks currently require the hermes parser")
            start_ids = self.tokenizer.encode("<tool_call>", add_special_tokens=False)
            end_ids = self.tokenizer.encode("</tool_call>", add_special_tokens=False)
            if len(start_ids) != 1 or len(end_ids) != 1:
                raise ValueError("factorial tool-call masks require single-token Hermes delimiters")
            self.tool_call_start_token_id = start_ids[0]
            self.tool_call_end_token_id = end_ids[0]

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        factorial_condition = kwargs.get("factorial_tool_condition")
        if self.factorial_tool_availability:
            if factorial_condition is None:
                raise ValueError("factorial tool availability is enabled but the rollout condition is missing")
            rollout_allowed_tools = allowed_tool_names(str(factorial_condition))
        else:
            rollout_allowed_tools = None

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            allowed_tools=rollout_allowed_tools,
            factorial_condition=str(factorial_condition) if factorial_condition is not None else None,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        if self.factorial_tool_availability:
            output.extra_fields["tool_call_token_spans"] = agent_data.tool_call_token_spans
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        tool_schemas = self.tool_schemas
        if agent_data.allowed_tools is not None:
            tool_schemas = [
                schema for name, schema in self.tool_schema_by_name.items() if name in agent_data.allowed_tools
            ]
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []
        turn_response_start = len(agent_data.response_mask)

        with simple_timer("generate_sequences", agent_data.metrics):
            turn_sampling_params = dict(sampling_params)
            if self.max_tokens_per_turn is not None:
                if self.max_tokens_per_turn <= 0:
                    raise ValueError("multi_turn.max_tokens_per_turn must be positive")
                turn_sampling_params["max_new_tokens"] = int(self.max_tokens_per_turn)
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        response_limit_reached = not ignore_termination and len(agent_data.response_mask) >= self.response_length
        assistant_turn_limit_reached = (
            self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
        )
        user_turn_limit_reached = self.max_user_turns and agent_data.user_turns >= self.max_user_turns

        # Extract tool calls
        tools = [
            tool.tool_schema
            for name, tool in self.tools.items()
            if agent_data.allowed_tools is None or name in agent_data.allowed_tools
        ]
        if self.factorial_tool_availability:
            agent_data.tool_calls = []
            agent_data.active_tool_call_span_indices = []
            local_spans = find_delimited_token_spans(
                agent_data.response_ids,
                self.tool_call_start_token_id,
                self.tool_call_end_token_id,
            )
            calls_with_spans: list[tuple[FunctionCall, tuple[int, int]]] = []
            for local_span in local_spans:
                _, parsed_calls = await self.tool_parser.extract_tool_calls(
                    agent_data.response_ids[local_span[0] : local_span[1]], tools
                )
                if len(parsed_calls) != 1:
                    continue
                tool_call = parsed_calls[0]
                if tool_call.name in agent_data.allowed_tools:
                    calls_with_spans.append((tool_call, local_span))

            for tool_call, (local_start, local_end) in calls_with_spans[: self.max_parallel_calls]:
                agent_data.tool_calls.append(tool_call)
                agent_data.tool_call_token_spans.append(
                    {
                        "name": tool_call.name,
                        "start": turn_response_start + local_start,
                        "end": turn_response_start + local_end,
                        "executed_ok": False,
                        "status": "pending",
                    }
                )
                agent_data.active_tool_call_span_indices.append(len(agent_data.tool_call_token_spans) - 1)
        else:
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        # Capture this turn's assistant text for trajectory logging (env-gated).
        if trajectory_logger.is_enabled():
            try:
                agent_data.last_assistant_text = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                )
            except Exception:
                agent_data.last_assistant_text = None

        termination_reason = None
        if response_limit_reached:
            termination_reason = "response_length_limit"
        elif assistant_turn_limit_reached:
            termination_reason = "assistant_turn_limit"
        elif user_turn_limit_reached:
            termination_reason = "user_turn_limit"
        if termination_reason is not None:
            self._record_limit_termination(agent_data, termination_reason)
            return AgentState.TERMINATED

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            if self._duplicate_call_limit_reached(agent_data):
                return AgentState.TERMINATED
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            # Terminal turn: model produced a final answer with no tool call.
            if trajectory_logger.is_enabled():
                agent_data.extra_fields.setdefault("trajectory", []).append(
                    {
                        "turn": agent_data.assistant_turns,
                        "reasoning": getattr(agent_data, "last_assistant_text", None),
                        "tool_call": None,
                        "tool_output_text": None,
                        "num_output_images": 0,
                        "tool_reward": None,
                        "executed_ok": None,
                        "status": "final_answer",
                    }
                )
            return AgentState.TERMINATED

    def _record_limit_termination(self, agent_data: AgentData, status: str) -> None:
        """Record the last generated action even when a rollout limit prevents execution."""

        agent_data.extra_fields["termination_reason"] = status
        if agent_data.active_tool_call_span_indices:
            for span_index in agent_data.active_tool_call_span_indices:
                agent_data.tool_call_token_spans[span_index]["executed_ok"] = False
                agent_data.tool_call_token_spans[span_index]["status"] = status
        if not trajectory_logger.is_enabled():
            return
        if agent_data.tool_calls:
            for index, tool_call in enumerate(agent_data.tool_calls[: self.max_parallel_calls]):
                agent_data.extra_fields.setdefault("trajectory", []).append(
                    {
                        "turn": agent_data.assistant_turns,
                        "reasoning": getattr(agent_data, "last_assistant_text", None) if index == 0 else None,
                        "tool_call": {"name": tool_call.name, "arguments": tool_call.arguments},
                        "tool_output_text": "Tool call was not executed because the rollout limit was reached.",
                        "num_output_images": 0,
                        "tool_reward": None,
                        "executed_ok": False,
                        "status": status,
                        "tool_metrics": {"executed_ok": False, "status": status},
                    }
                )
        else:
            agent_data.extra_fields.setdefault("trajectory", []).append(
                {
                    "turn": agent_data.assistant_turns,
                    "reasoning": getattr(agent_data, "last_assistant_text", None),
                    "tool_call": None,
                    "tool_output_text": None,
                    "num_output_images": 0,
                    "tool_reward": None,
                    "executed_ok": None,
                    "status": status,
                }
            )

    @staticmethod
    def _normalized_tool_call_key(tool_call: FunctionCall) -> tuple[str, str]:
        try:
            arguments = json.dumps(json.loads(tool_call.arguments), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = tool_call.arguments
        return tool_call.name, arguments

    def _duplicate_call_limit_reached(self, agent_data: AgentData) -> bool:
        """Stop a deterministic exact-call loop without pretending it produced a valid answer."""

        if not agent_data.tool_calls:
            return False
        key = self._normalized_tool_call_key(agent_data.tool_calls[0])
        if key == agent_data.last_tool_call_key:
            agent_data.consecutive_identical_tool_calls += 1
        else:
            agent_data.last_tool_call_key = key
            agent_data.consecutive_identical_tool_calls = 1

        occurrence_count = agent_data.tool_call_key_occurrences.get(key, 0) + 1
        agent_data.tool_call_key_occurrences[key] = occurrence_count

        consecutive_limit = self.max_consecutive_identical_tool_calls
        occurrence_limit = getattr(self, "max_identical_tool_call_occurrences", 0)
        consecutive_reached = (
            consecutive_limit > 0 and agent_data.consecutive_identical_tool_calls > consecutive_limit
        )
        occurrence_reached = occurrence_limit > 0 and occurrence_count > occurrence_limit
        if not consecutive_reached and not occurrence_reached:
            return False

        if agent_data.active_tool_call_span_indices:
            span_index = agent_data.active_tool_call_span_indices[0]
            agent_data.tool_call_token_spans[span_index]["executed_ok"] = False
            agent_data.tool_call_token_spans[span_index]["status"] = "duplicate_call_limit"
        if trajectory_logger.is_enabled():
            tool_call = agent_data.tool_calls[0]
            agent_data.extra_fields.setdefault("trajectory", []).append(
                {
                    "turn": agent_data.assistant_turns,
                    "reasoning": getattr(agent_data, "last_assistant_text", None),
                    "tool_call": {"name": tool_call.name, "arguments": tool_call.arguments},
                    "tool_output_text": "Exact duplicate tool call blocked by the rollout loop guard.",
                    "num_output_images": 0,
                    "tool_reward": 0.0,
                    "executed_ok": False,
                    "status": "duplicate_call_limit",
                }
            )
        agent_data.extra_fields["duplicate_call_terminated"] = True
        agent_data.extra_fields["duplicate_call_key"] = {"name": key[0], "arguments": key[1]}
        agent_data.extra_fields["duplicate_call_guard"] = (
            "trajectory_occurrence" if occurrence_reached else "consecutive"
        )
        return True

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for idx, (tool_response, tool_reward, tool_metrics) in enumerate(responses):
            if idx < len(agent_data.active_tool_call_span_indices):
                span_index = agent_data.active_tool_call_span_indices[idx]
                agent_data.tool_call_token_spans[span_index]["executed_ok"] = (tool_metrics or {}).get(
                    "executed_ok"
                ) is True
                agent_data.tool_call_token_spans[span_index]["status"] = (tool_metrics or {}).get("status")
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            # Per-turn trajectory record (env-gated). Reasoning attached once per turn
            # (idx == 0) to avoid duplicating it across parallel tool calls.
            if trajectory_logger.is_enabled():
                tc = agent_data.tool_calls[idx] if idx < len(agent_data.tool_calls) else None
                agent_data.extra_fields.setdefault("trajectory", []).append(
                    {
                        "turn": agent_data.assistant_turns,
                        "reasoning": getattr(agent_data, "last_assistant_text", None) if idx == 0 else None,
                        "tool_call": (
                            {"name": tc.name, "arguments": tc.arguments}
                            if tc is not None
                            else {"name": tool_call_names[idx] if idx < len(tool_call_names) else None}
                        ),
                        "tool_output_text": (tool_response.text or "")[:4000],
                        "num_output_images": (len(tool_response.image) if tool_response.image else 0),
                        "tool_reward": tool_reward,
                        "executed_ok": (tool_metrics or {}).get("executed_ok"),
                        "status": (tool_metrics or {}).get("status"),
                        "tool_metrics": dict(tool_metrics or {}),
                    }
                )

        explicit_failures = [
            (tool_metrics or {}).get("executed_ok") is False for _, _, tool_metrics in responses
        ]
        if explicit_failures and all(explicit_failures):
            agent_data.consecutive_tool_failures += 1
        else:
            agent_data.consecutive_tool_failures = 0
        failure_limit = getattr(self, "max_consecutive_tool_failures", 0)
        if failure_limit > 0 and agent_data.consecutive_tool_failures > failure_limit:
            agent_data.extra_fields["termination_reason"] = "tool_failure_limit"
            agent_data.extra_fields["consecutive_tool_failures"] = agent_data.consecutive_tool_failures
            return AgentState.TERMINATED

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.extra_fields["termination_reason"] = "response_length_limit"
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            if agent_data.allowed_tools is not None and tool_name not in agent_data.allowed_tools:
                raise ValueError(f"tool {tool_name!r} is unavailable in condition {agent_data.factorial_condition!r}")
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {"error": str(e), "executed_ok": False},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        # Tools may report a handled execution failure without raising (syntax
        # error, timeout, etc.). Preserve that result instead of treating every
        # normal return as successful.
        res = dict(res) if res else {}
        res.setdefault("executed_ok", res.get("execution_successful", True))
        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
