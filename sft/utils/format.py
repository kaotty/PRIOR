"""Canonical assistant serialization and source-neutral prompt formatting."""

from __future__ import annotations

import copy
import re
from typing import Any, Sequence


BOXED_TOKEN = re.compile(r"\\boxed\b")
THINK_BLOCK = re.compile(r"\s*<think>([\s\S]*?)</think>\s*\Z")
ANSWER_TAG = re.compile(r"</?answer\b", re.IGNORECASE)


class CanonicalFormatError(ValueError):
    """A trajectory cannot be normalized without guessing its intended answer."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _reasoning(text: str) -> str:
    text = text.strip()
    has_think_tag = "<think>" in text or "</think>" in text
    if has_think_tag:
        match = THINK_BLOCK.fullmatch(text)
        if match is None or text.count("<think>") != 1 or text.count("</think>") != 1:
            raise CanonicalFormatError("malformed_think_block")
        text = match.group(1).strip()
    if not text:
        raise CanonicalFormatError("empty_reasoning")
    if "<think>" in text or "</think>" in text:
        raise CanonicalFormatError("nested_think_block")
    if ANSWER_TAG.search(text):
        raise CanonicalFormatError("answer_tag_in_reasoning")
    return text


def _boxed_answer(text: str) -> tuple[str, str]:
    """Return unwrapped reasoning and answer from one unambiguous final turn."""

    text = text.strip()
    if len(BOXED_TOKEN.findall(text)) != 1 or text.count(r"\boxed{") != 1:
        raise CanonicalFormatError("boxed_count_not_one")
    start = text.index(r"\boxed{")
    cursor = start + len(r"\boxed{")
    depth = 1
    while cursor < len(text) and depth:
        character = text[cursor]
        escaped = cursor > 0 and text[cursor - 1] == "\\"
        if character == "{" and not escaped:
            depth += 1
        elif character == "}" and not escaped:
            depth -= 1
        cursor += 1
    if depth:
        raise CanonicalFormatError("unclosed_boxed_answer")
    if text[cursor:].strip():
        raise CanonicalFormatError("text_after_boxed_answer")

    answer = text[start + len(r"\boxed{") : cursor - 1].strip()
    if not answer:
        raise CanonicalFormatError("empty_boxed_answer")
    if BOXED_TOKEN.search(answer) or "<think>" in answer or "</think>" in answer or ANSWER_TAG.search(answer):
        raise CanonicalFormatError("markup_inside_boxed_answer")
    return _reasoning(text[:start]), answer


def normalize_tool_turn(content: str) -> str:
    """Wrap tool-call reasoning in exactly one think block."""

    if BOXED_TOKEN.search(content):
        raise CanonicalFormatError("boxed_answer_in_tool_turn")
    return f"<think>{_reasoning(content)}</think>"


def normalize_final_turn(content: str) -> str:
    """Emit exactly ``<think>...</think>\n\\boxed{...}``."""

    reasoning, answer = _boxed_answer(content)
    return f"<think>{reasoning}</think>\n\\boxed{{{answer}}}"


def normalize_assistant_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize all assistant turns and require one terminal final answer."""

    normalized = copy.deepcopy(list(messages))
    terminal_indices: list[int] = []
    for index, message in enumerate(normalized):
        if message.get("role") != "assistant":
            continue
        tool_calls = list(message.get("tool_calls") or [])
        content = str(message.get("content") or "")
        if tool_calls:
            message["content"] = normalize_tool_turn(content)
        else:
            terminal_indices.append(index)
            message["content"] = normalize_final_turn(content)
    if len(terminal_indices) != 1:
        raise CanonicalFormatError("terminal_assistant_count_not_one")
    if terminal_indices[0] != len(normalized) - 1:
        raise CanonicalFormatError("terminal_assistant_not_last")
    return normalized


def validate_assistant_messages(messages: Sequence[dict[str, Any]]) -> None:
    """Require messages to already be in the exact canonical serialization."""

    normalized = normalize_assistant_messages(messages)
    for original, expected in zip(messages, normalized, strict=True):
        if original.get("role") == "assistant" and original.get("content") != expected.get("content"):
            raise CanonicalFormatError("assistant_content_not_canonical")


CANONICAL_SYSTEM_PROMPT = (
    "You are a careful multimodal reasoner. Solve the problem from the image and question. "
    "If tools are available, use them only when they are genuinely helpful. "
    "Use image_zoom_in_tool only for closer visual inspection. "
    "Use code_interpreter only for genuine numerical or symbolic computation over quantities and relations you "
    "have already extracted; it cannot inspect the original image."
)
CANONICAL_USER_SUFFIX = r"Put only the final answer in \boxed{}."

_SOURCE_INSTRUCTION = re.compile(
    r"\n\s*(?:"
    r"Answer the question using|"
    r"Answer with the option|"
    r"Answer with the single option|"
    r"Think in the mind first|"
    r"You must put your answer|"
    r"Please reason step by step\.\s*Use Python|"
    r"Format strictly as|"
    r"Guidelines\s*:|"
    r"Put ONLY the final answer|"
    r"Put only the option letter"
    r")",
    flags=re.IGNORECASE,
)
_FORBIDDEN_USER_TEXT = re.compile(
    r"</?(?:code|answer|tool_call)>|"
    r"Use Python code|"
    r"decide whether to call tools|"
    r"Format strictly as|"
    r"Guidelines\s*:",
    flags=re.IGNORECASE,
)


class UnifiedPromptError(ValueError):
    """A source prompt cannot be normalized without guessing its question."""


def extract_source_question(content: str) -> str:
    """Remove image placeholders and source-owned tool/answer instructions."""

    content = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    content, replacements = re.subn(r"^\s*<image>\s*", "", content, count=1, flags=re.IGNORECASE)
    if replacements != 1:
        raise UnifiedPromptError("initial_user_missing_single_image_placeholder")
    content = re.sub(r"^\s*Question\s*:\s*", "", content, count=1, flags=re.IGNORECASE)
    content = _SOURCE_INSTRUCTION.split(content, maxsplit=1)[0]
    lines = [re.sub(r"[ \t]+$", "", line) for line in content.splitlines()]
    content = "\n".join(lines).strip()
    content = re.sub(r"\n{3,}", "\n\n", content)
    if not content or re.search(r"[^\W_]", content, flags=re.UNICODE) is None:
        raise UnifiedPromptError("empty_question_after_prompt_cleanup")
    if "<image>" in content.lower():
        raise UnifiedPromptError("extra_image_placeholder_in_question")
    if _FORBIDDEN_USER_TEXT.search(content):
        raise UnifiedPromptError("source_instruction_survived_cleanup")
    return content


def canonical_user_message(question: str) -> str:
    """Render one image, the untouched question/options, and one answer contract."""

    question = str(question or "").strip()
    if not question:
        raise UnifiedPromptError("empty_question")
    return f"<image>\n{question}\n{CANONICAL_USER_SUFFIX}"


def normalize_prompt_messages(messages: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Normalize only system/user prompts while preserving the supervised trajectory."""

    normalized = copy.deepcopy(list(messages))
    system_indices = [index for index, message in enumerate(normalized) if message.get("role") == "system"]
    user_indices = [index for index, message in enumerate(normalized) if message.get("role") == "user"]
    if system_indices != [0]:
        raise UnifiedPromptError("requires_exactly_one_leading_system_message")
    if len(user_indices) != 1:
        raise UnifiedPromptError("requires_exactly_one_initial_user_message")
    user_index = user_indices[0]
    if user_index != 1:
        raise UnifiedPromptError("initial_user_message_must_follow_system")

    question = extract_source_question(str(normalized[user_index].get("content") or ""))
    normalized[0]["content"] = CANONICAL_SYSTEM_PROMPT
    normalized[user_index]["content"] = canonical_user_message(question)
    return normalized, question


def validate_unified_prompt(messages: Sequence[dict[str, Any]]) -> str:
    """Require a row to already use the exact unified prompt contract."""

    normalized = list(messages)
    if len(normalized) < 3 or normalized[0].get("role") != "system":
        raise UnifiedPromptError("invalid_message_sequence")
    if normalized[0].get("content") != CANONICAL_SYSTEM_PROMPT:
        raise UnifiedPromptError("system_prompt_not_canonical")
    users = [message for message in normalized if message.get("role") == "user"]
    if len(users) != 1:
        raise UnifiedPromptError("user_message_count_not_one")
    content = str(users[0].get("content") or "")
    if not content.startswith("<image>\n") or not content.endswith("\n" + CANONICAL_USER_SUFFIX):
        raise UnifiedPromptError("user_prompt_not_canonical")
    question = content[len("<image>\n") : -len("\n" + CANONICAL_USER_SUFFIX)]
    if not question.strip() or _FORBIDDEN_USER_TEXT.search(question):
        raise UnifiedPromptError("invalid_canonical_question")
    return question
