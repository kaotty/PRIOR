#!/usr/bin/env python3
"""Fail closed unless structured SFT calls become trainable Hermes tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _extract_call_payloads(rendered: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = rendered.find(TOOL_CALL_START, cursor)
        if start < 0:
            break
        end = rendered.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
        if end < 0:
            raise ValueError("rendered tool call has no closing delimiter")
        raw = rendered[start + len(TOOL_CALL_START) : end].strip()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise ValueError("rendered tool call is not a name/arguments JSON object")
        if not isinstance(payload.get("arguments"), dict):
            raise ValueError("rendered tool-call arguments must be a JSON object, not a quoted JSON string")
        payloads.append(payload)
        cursor = end + len(TOOL_CALL_END)
    return payloads


def _expected_payload(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("structured call has no function name")
    if not isinstance(arguments, str):
        raise ValueError("canonical structured call arguments must be a JSON string")
    decoded = json.loads(arguments)
    if not isinstance(decoded, dict):
        raise ValueError("canonical structured call arguments must decode to a JSON object")
    return {"name": name, "arguments": decoded}


def _render_message(tokenizer: Any, message: dict[str, Any], tools: list[dict[str, Any]] | None) -> str:
    return tokenizer.apply_chat_template(
        [message],
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    )


def _assistant_loss_token_ids(tokenizer: Any, message: dict[str, Any]) -> list[int]:
    from verl.utils.chat_template import apply_chat_template, extract_system_prompt_and_generation
    from verl.utils.tokenizer import normalize_token_ids

    system_prompt, generation_prompt = extract_system_prompt_and_generation(tokenizer)
    token_ids = apply_chat_template(
        tokenizer,
        messages=[message],
        tools=None,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
    )
    token_ids = normalize_token_ids(token_ids)
    token_ids = token_ids[len(system_prompt) :]
    if token_ids[: len(generation_prompt)] != generation_prompt:
        raise ValueError("assistant message does not begin with the expected generation prompt")
    return token_ids[len(generation_prompt) :]


def _audit_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, int]:
    messages = list(row.get("messages") or [])
    tools = list(row.get("tools") or [])
    if not messages or messages[0].get("role") != "system" or len(tools) != 2:
        raise ValueError("row must contain a system message and exactly two tool schemas")

    rendered_system = _render_message(tokenizer, messages[0], tools)
    names = {str((tool.get("function") or {}).get("name")) for tool in tools}
    if "# Tools" not in rendered_system or names != {"image_zoom_in_tool", "code_interpreter"}:
        raise ValueError("system/tool schema is not the frozen two-tool interface")
    if any(name not in rendered_system for name in names):
        raise ValueError("rendered system prompt omits a tool schema")

    structured_calls = 0
    rendered_calls = 0
    tool_responses = 0
    assistant_loss_tokens = 0
    for message in messages:
        role = message.get("role")
        calls = list(message.get("tool_calls") or [])
        if role == "assistant":
            rendered = _render_message(tokenizer, message, None)
            actual = _extract_call_payloads(rendered)
            expected = [_expected_payload(call) for call in calls]
            if actual != expected:
                raise ValueError(f"rendered calls differ from structured calls: {actual!r} != {expected!r}")
            loss_ids = _assistant_loss_token_ids(tokenizer, message)
            assistant_loss_tokens += len(loss_ids)
            loss_text = tokenizer.decode(loss_ids, skip_special_tokens=False)
            if _extract_call_payloads(loss_text) != expected:
                raise ValueError("tool call is absent from assistant loss tokens")
            structured_calls += len(expected)
            rendered_calls += len(actual)
        elif calls:
            raise ValueError("only assistant messages may contain tool_calls")

        if role == "tool":
            rendered = _render_message(tokenizer, message, None)
            if rendered.count(TOOL_RESPONSE_START) != 1 or rendered.count(TOOL_RESPONSE_END) != 1:
                raise ValueError("tool observation is not wrapped in one tool_response block")
            tool_responses += 1

    if rendered_calls != structured_calls or tool_responses != structured_calls:
        raise ValueError(
            f"call/response count mismatch: structured={structured_calls}, rendered={rendered_calls}, "
            f"responses={tool_responses}"
        )
    if assistant_loss_tokens <= 0:
        raise ValueError("row has no trainable assistant tokens")
    return {
        "structured_calls": structured_calls,
        "rendered_calls": rendered_calls,
        "tool_responses": tool_responses,
        "assistant_loss_tokens": assistant_loss_tokens,
        "no_tool_rows": int(structured_calls == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = args.template.read_text(encoding="utf-8")

    totals = {
        "rows": 0,
        "structured_calls": 0,
        "rendered_calls": 0,
        "tool_responses": 0,
        "assistant_loss_tokens": 0,
        "no_tool_rows": 0,
    }
    files = []
    for raw_path in args.parquet:
        path = Path(raw_path)
        table = pq.ParquetFile(path).read(columns=["messages", "tools"])
        rows = table.to_pylist()
        if args.max_rows is not None:
            remaining = max(args.max_rows - totals["rows"], 0)
            rows = rows[:remaining]
        file_counts = {
            "structured_calls": 0,
            "rendered_calls": 0,
            "tool_responses": 0,
            "assistant_loss_tokens": 0,
            "no_tool_rows": 0,
        }
        for index, row in enumerate(rows):
            try:
                counts = _audit_row(tokenizer, row)
            except Exception as exc:
                raise SystemExit(f"FAIL {path}:row={index}: {exc}") from exc
            totals["rows"] += 1
            for key, value in counts.items():
                totals[key] += value
                file_counts[key] += value
        files.append({"path": str(path), "rows": len(rows), "sha256": _sha256(path), **file_counts})
        if args.max_rows is not None and totals["rows"] >= args.max_rows:
            break

    if totals["rows"] <= 0 or totals["structured_calls"] != totals["rendered_calls"]:
        raise SystemExit(f"FAIL totals={totals}")
    payload = {
        "status": "PASS",
        "model": str(args.model),
        "template": str(args.template),
        "template_sha256": _sha256(args.template),
        "files": files,
        **totals,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
