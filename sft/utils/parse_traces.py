"""Parse DeepEyesV2 messages and inline tool calls into structured trajectories.

DeepEyesV2 SFT records assistant tool invocations as inline tags inside the
``content`` string:

  * ``<code>``Python code``</code>``    — calls the code interpreter
  * ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``  — calls
    ``image_search`` (no args, reverse-image search on the current image)
    or ``search`` (single ``query`` / ``query_list`` arg, web search)

Tool *responses* are encoded as the next ``user`` turn (NOT a ``tool`` role) and
take the shape::

    Code execution result:
    stdout:
    ```
    ...
    ```

    stderr:
    ```
    ...
    ```

    Images:
    <image>

or for ``search`` (text-only)::

    <tool_response>
    ...search hits text...
    </tool_response>

or for ``image_search`` (mixed text + thumbnails)::

    <tool_response>
    A Google image search for the image found N results:
    ## Web Results
    1. <image>
    [Title]
    ...
    </tool_response>

This module:

  1. Splits assistant ``content`` into reasoning text + tool calls.
  2. Detects whether a ``<code>`` block is a *pure crop* (yields a
     ``image_zoom_in_tool`` call with ``bbox_2d``) or any other Python code
     (``code_interpreter``). When the caller knows the upcoming tool-response
     image count, we use it as the ground-truth signal: a single ``crop()``
     paired with exactly one response image stays as zoom; anything else
     downgrades to ``code_interpreter``.
  3. Re-shapes ``search`` calls into our ``search.query_list`` schema and
     surfaces ``image_search`` as a separate no-arg tool call.
  4. Strips the ``<code>``/``<tool_call>`` tags from the assistant content so
     the chat-template-rendered ``tool_calls`` field becomes the canonical
     source of the call.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


logger = logging.getLogger(__name__)


# Both ``image.crop(...)`` and ``image_1.crop(...)`` style calls are valid.
_CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.S)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


@dataclass
class ParsedAssistantTurn:
    """One assistant turn after parsing tool calls out of the raw content."""

    text_content: str  # plain reasoning text (and optional answer) with tool tags removed
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Each tool call is annotated with its detected category so the trajectory
    # filter can decide whether to keep it as zoom-only.
    call_categories: list[str] = field(default_factory=list)
    final_answer: Optional[str] = None  # extracted from <answer>...</answer> if present


def _strip_python_fences(code: str) -> str:
    """Remove ```python ... ``` fences if present."""

    code = code.strip()
    if code.startswith("```"):
        # remove leading ``` and optional language tag
        code = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", code)
        code = re.sub(r"```\s*$", "", code)
    return code.strip()


# crop calls we want to capture:
#   image_1.crop((100, 200, 300, 400))   # tuple
#   image_1.crop([100, 200, 300, 400])   # list
#   image_1.crop(box=(100,200,300,400))
#   crop = image.crop((..., ..., ..., ...))
_CROP_CALL_RE = re.compile(
    r"\.crop\s*\(\s*(?:box\s*=\s*)?[\[\(]\s*"
    r"([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*[\]\)]\s*\)",
    re.I,
)


def _try_extract_static_crop_bbox(code: str) -> Optional[list[int]]:
    """Pull a static ``[x1, y1, x2, y2]`` out of an ``image.crop(...)`` call.

    Returns ``None`` if the crop arguments aren't all integer/float literals,
    e.g. when the bbox is computed from variables. Such trajectories are not
    converted (we won't be able to recover the actual numbers without running
    the Python). They count as ``code_interpreter`` instead.
    """

    code = _strip_python_fences(code)

    # Fast-path regex pull. Falls back to AST when regex misses (e.g. mixed
    # whitespace or named-arg quoting we didn't anticipate).
    match = _CROP_CALL_RE.search(code)
    if match is not None:
        try:
            return [int(round(float(g))) for g in match.groups()]
        except ValueError:
            return None

    return _extract_crop_via_ast(code)


def _extract_crop_via_ast(code: str) -> Optional[list[int]]:
    """Best-effort AST traversal for the ``.crop()`` call."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "crop":
            continue
        # Single positional arg: a tuple/list of 4 numbers.
        bbox = None
        if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
            elts = node.args[0].elts
            if len(elts) == 4:
                bbox = elts
        if bbox is None:
            for kw in node.keywords:
                if kw.arg == "box" and isinstance(kw.value, (ast.Tuple, ast.List)):
                    if len(kw.value.elts) == 4:
                        bbox = kw.value.elts
                    break
        if bbox is None:
            continue
        try:
            return [int(round(float(ast.literal_eval(e)))) for e in bbox]
        except (ValueError, SyntaxError):
            return None
    return None


_CROP_RECEIVER_RE = re.compile(r"\.crop\s*\(", re.I)


def _is_pure_crop(code: str, expected_response_images: Optional[int] = None) -> bool:
    """A code block is a *pure crop* iff it does **exactly one** ``image_*.crop()``
    on the input image variable and produces exactly one response image.

    The response-image count (when known) is the most reliable signal: if the
    rendered tool response only contains a single image, that image must be
    the crop, regardless of how many ``plt.imshow`` calls precede it. If the
    response contains zero or multiple images then a single ``image_zoom_in_tool``
    call cannot honestly explain them and we demote to ``code_interpreter``.

    We also keep a small blacklist of image-processing primitives (Canny,
    threshold, contours, OCR, neural nets...) that, even when their output is
    a single image, are *not* faithful crops of the input frame and so don't
    match what ``image_zoom_in_tool`` produces at runtime.
    """

    code = _strip_python_fences(code)
    code_lower = code.lower()
    blacklist_tokens = (
        "cv2.canny",
        "cv2.threshold",
        "cv2.findcontours",
        "cv2.houghlines",
        "cv2.matchtemplate",
        "yolov",
        "torch.",
        "tensorflow",
        ".forward(",
        "ocr.",
        "pytesseract",
        "easyocr",
    )
    if any(tok in code_lower for tok in blacklist_tokens):
        return False

    if len(_CROP_RECEIVER_RE.findall(code)) != 1:
        return False

    if expected_response_images is not None and expected_response_images != 1:
        return False

    # Receiver must look like the input-image variable (image, image_1, ...).
    # Crops on derived variables would target a different reference frame.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "crop"
        ):
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and re.fullmatch(r"image(_\d+)?", receiver.id):
                return True
            return False
    return False


def _build_zoom_call(bbox_2d: list[int]) -> dict[str, Any]:
    """OpenAI-style call dict matching the rest of the codebase."""

    return {
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "arguments": json.dumps({"bbox_2d": bbox_2d}, ensure_ascii=False),
        },
    }


def _build_code_call(code: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "arguments": json.dumps({"code": _strip_python_fences(code)}, ensure_ascii=False),
        },
    }


def _build_search_call(payload: dict[str, Any]) -> Optional[tuple[dict[str, Any], str]]:
    """Map DeepEyesV2's two search variants onto our schemas.

    Returns ``(call_dict, category)`` where ``category`` is either
    ``"search"`` (web text search) or ``"image_search"`` (reverse-image
    search, no args). Returns ``None`` for malformed payloads (caller
    should drop the trajectory).
    """

    name = payload.get("name", "").strip().lower()
    args = payload.get("arguments", {}) or {}
    if name == "search":
        if "query_list" in args and isinstance(args["query_list"], list):
            queries = [str(q) for q in args["query_list"] if q]
        elif "query" in args:
            q = args["query"]
            queries = [q] if isinstance(q, str) else [str(x) for x in q]
        else:
            return None
        if not queries:
            return None
        return (
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query_list": queries}, ensure_ascii=False),
                },
            },
            "search",
        )
    if name == "image_search":
        # The source ``image_search`` is parameterless: it always operates on
        # the most recent image in the conversation. We surface the same
        # signature so SFT and RL stay in sync.
        return (
            {
                "type": "function",
                "function": {
                    "name": "image_search",
                    "arguments": json.dumps({}, ensure_ascii=False),
                },
            },
            "image_search",
        )
    return None


def parse_assistant_content(
    content: str,
    expected_response_images: Optional[int] = None,
) -> ParsedAssistantTurn:
    """Split a DeepEyesV2 assistant turn into (text, tool_calls, categories).

    ``expected_response_images`` is the count of ``<image>`` placeholders in
    the *next* user/tool turn (i.e., the response to this assistant's tool
    call). When provided, it disambiguates ``<code>`` blocks: a single
    ``crop()`` only counts as a zoom when the response contains exactly one
    image. Otherwise, the block stays as ``code_interpreter``.
    """

    tool_calls: list[dict[str, Any]] = []
    categories: list[str] = []

    # Iterate ``<code>`` and ``<tool_call>`` blocks in document order so the
    # resulting ``tool_calls`` list mirrors the original execution order.
    spans: list[tuple[int, int, str, str]] = []
    for m in _CODE_BLOCK_RE.finditer(content):
        spans.append((m.start(), m.end(), "code", m.group(1)))
    for m in _TOOL_CALL_RE.finditer(content):
        spans.append((m.start(), m.end(), "tool_call", m.group(1)))
    spans.sort()

    bad_call = False
    for _, _, kind, body in spans:
        if kind == "code":
            is_crop = _is_pure_crop(body, expected_response_images=expected_response_images)
            bbox = _try_extract_static_crop_bbox(body) if is_crop else None
            if bbox is not None:
                tool_calls.append(_build_zoom_call(bbox))
                categories.append("zoom")
            else:
                tool_calls.append(_build_code_call(body))
                categories.append("code")
        else:  # tool_call
            try:
                payload = json.loads(body.strip())
            except json.JSONDecodeError:
                bad_call = True
                continue
            converted = _build_search_call(payload) if isinstance(payload, dict) else None
            if converted is None:
                bad_call = True
                continue
            call_dict, category = converted
            tool_calls.append(call_dict)
            categories.append(category)

    # Strip the inline tool tags from the textual content. We keep the
    # surrounding <think>...</think> reasoning intact because the chat
    # template renders that as the assistant's text part.
    text = _CODE_BLOCK_RE.sub("", content)
    text = _TOOL_CALL_RE.sub("", text).strip()

    # Detect <answer>...</answer>. We don't strip it from text because
    # downstream we replace it with \boxed{...} in a separate pass.
    final_answer = None
    answer_match = _ANSWER_RE.search(text)
    if answer_match is not None:
        final_answer = answer_match.group(1).strip()

    if bad_call:
        # Signal the caller to discard the whole trajectory by clearing tool calls
        # categories — we don't want a partial conversion to corrupt training.
        categories.append("__bad__")

    return ParsedAssistantTurn(
        text_content=text,
        tool_calls=tool_calls,
        call_categories=categories,
        final_answer=final_answer,
    )


def normalize_final_answer(text: str) -> str:
    """Convert ``<answer>X</answer>`` mentions into ``\\boxed{X}``.

    Why the conversion? The RL reward in ``verl/utils/reward_score/geo3k.py``
    grades against ``\\boxed{}`` only. Keeping SFT and RL output formats
    aligned avoids "reward = 0 because format mismatch" issues immediately
    after RL kicks in.
    """

    def _replace(match: re.Match[str]) -> str:
        ans = match.group(1).strip()
        return r"\boxed{" + ans + "}"

    return _ANSWER_RE.sub(_replace, text)



_USER_IMAGE_PLACEHOLDER = re.compile(r"<image>")


_SEARCH_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)


@dataclass
class RawTurn:
    """One full assistant + tool-response pair, plus image bookkeeping."""

    assistant: ParsedAssistantTurn
    response_text: str = ""              # tool / observation message body (may be "")
    response_image_indices: list[int] = field(default_factory=list)  # indices into traj.image_paths
    is_terminal: bool = False


@dataclass
class RawTrajectory:
    """Intermediate representation: structured but still string-typed images."""

    question: str
    user_image_indices: list[int]        # indices into image_paths used in the user prompt
    image_paths: list[str]               # ALL images referenced anywhere in the conversation
    turns: list[RawTurn]
    categories: set[str]                 # subset of {"zoom", "code", "search", "image_search"}
    sample_id: str
    raw_index: int                       # which sample index in the source JSON
    source_file: str                     # JSON file basename
    valid: bool = True
    invalid_reason: Optional[str] = None

    @property
    def n_zoom_calls(self) -> int:
        return sum(1 for t in self.turns for c in t.assistant.call_categories if c == "zoom")

    @property
    def n_code_calls(self) -> int:
        return sum(1 for t in self.turns for c in t.assistant.call_categories if c == "code")

    @property
    def n_search_calls(self) -> int:
        return sum(1 for t in self.turns for c in t.assistant.call_categories if c == "search")

    @property
    def n_image_search_calls(self) -> int:
        return sum(
            1 for t in self.turns for c in t.assistant.call_categories if c == "image_search"
        )

    @property
    def n_non_zoom_calls(self) -> int:
        return self.n_code_calls + self.n_search_calls + self.n_image_search_calls

    @property
    def is_zoom_only(self) -> bool:
        """A zoom-only trajectory has at least one zoom call and zero non-zoom calls."""

        return self.n_zoom_calls > 0 and self.n_non_zoom_calls == 0

    @property
    def is_zoom_using(self) -> bool:
        """Has at least one zoom call (alone or mixed with other tools)."""

        return self.n_zoom_calls > 0


def _consume_images(
    content: str, image_cursor: int, total_images: int
) -> tuple[list[int], int]:
    """Pull image indices out of a content string in order of ``<image>`` tags."""

    placeholders = _USER_IMAGE_PLACEHOLDER.findall(content)
    indices: list[int] = []
    for _ in placeholders:
        if image_cursor >= total_images:
            # Source dataset is occasionally inconsistent; we record the issue
            # so the caller can drop the trajectory.
            return indices, -1
        indices.append(image_cursor)
        image_cursor += 1
    return indices, image_cursor


def _split_search_response(response_text: str) -> str:
    """Strip the ``<tool_response>`` wrappers around DeepEyesV2 search outputs."""

    matches = _SEARCH_RESPONSE_RE.findall(response_text)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return response_text.strip()


def parse_one(
    sample: dict, raw_index: int, source_file: str
) -> Optional[RawTrajectory]:
    """Parse a single DeepEyesV2 sample. Return ``None`` when malformed."""

    messages = sample.get("messages", [])
    image_paths = list(sample.get("images", []))
    if not messages or not image_paths:
        return None

    # Skip the leading system turn — we replace it with our own prompt later.
    if messages[0]["role"] == "system":
        messages = messages[1:]

    if len(messages) < 2 or messages[0]["role"] != "user":
        return None

    image_cursor = 0
    user_msg = messages[0]
    user_content = user_msg.get("content", "") or ""
    user_image_indices, image_cursor = _consume_images(
        user_content, image_cursor, len(image_paths)
    )
    if image_cursor < 0:
        return None
    # The first user turn ALWAYS has at least one image reference for our
    # purposes — questions without an attached image fall outside the
    # zoom-in / image-search experimental scope.
    if not user_image_indices:
        return None
    question_text = _USER_IMAGE_PLACEHOLDER.sub("", user_content).strip()

    turns: list[RawTurn] = []
    categories: set[str] = set()
    i = 1
    sample_id = sample.get("id") or f"{Path(source_file).stem}:{raw_index}"

    while i < len(messages):
        asst_msg = messages[i]
        if asst_msg.get("role") != "assistant":
            return None

        # Look ahead at the next user (tool-response) turn so we can pass its
        # ``<image>`` count down to the converter — that lets ``_is_pure_crop``
        # honestly distinguish single-crop blocks from multi-image plots.
        expected_resp_imgs: Optional[int] = None
        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        if next_msg is not None and next_msg.get("role") == "user":
            expected_resp_imgs = len(
                _USER_IMAGE_PLACEHOLDER.findall(next_msg.get("content") or "")
            )

        parsed = parse_assistant_content(
            asst_msg.get("content") or "",
            expected_response_images=expected_resp_imgs,
        )
        if "__bad__" in parsed.call_categories:
            return None
        if len(parsed.tool_calls) > 1:
            return None  # we filter to ≤1 tool call per turn

        cat = parsed.call_categories[0] if parsed.tool_calls else None
        if cat is not None:
            categories.add(cat)

        # Pair with the following user (tool response) turn iff the assistant
        # actually invoked a tool. The terminal answer turn has no follower.
        response_text = ""
        response_image_indices: list[int] = []
        if cat is not None:
            if next_msg is None or next_msg.get("role") != "user":
                return None  # tool-call without a response → broken
            resp_content = next_msg.get("content") or ""
            response_image_indices, image_cursor = _consume_images(
                resp_content, image_cursor, len(image_paths)
            )
            if image_cursor < 0:
                return None
            response_text = _USER_IMAGE_PLACEHOLDER.sub("", resp_content).strip()
            if cat in ("search", "image_search"):
                response_text = _split_search_response(response_text)

            # Sanity-check tool-response image counts.
            #   * zoom         → exactly 1 cropped image
            #   * search       → text-only, 0 images
            #   * image_search → at least 1 visual-search thumbnail
            #   * code         → variable; trust the source data
            n_resp_imgs = len(response_image_indices)
            if cat == "zoom" and n_resp_imgs != 1:
                return None
            if cat == "search" and n_resp_imgs != 0:
                return None
            if cat == "image_search" and n_resp_imgs == 0:
                return None
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

    # The final turn must be terminal (assistant answer with no tool call).
    if not turns or not turns[-1].is_terminal:
        return None
    # ... and we want a non-empty answer-bearing turn.
    if not (turns[-1].assistant.text_content or "").strip():
        return None

    return RawTrajectory(
        question=question_text,
        user_image_indices=user_image_indices,
        image_paths=image_paths,
        turns=turns,
        categories=categories,
        sample_id=sample_id,
        raw_index=raw_index,
        source_file=Path(source_file).name,
    )


def parse_json_file(
    path: str | Path, max_samples: Optional[int] = None
) -> Iterator[RawTrajectory]:
    """Stream-parse one ``sft_part_*.json`` file and yield every valid trajectory."""

    path = Path(path)
    logger.info("Parsing %s", path)
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


def parse_files(
    paths: Iterable[str | Path],
    max_samples_per_file: Optional[int] = None,
) -> Iterator[RawTrajectory]:
    for p in paths:
        yield from parse_json_file(p, max_samples=max_samples_per_file)
