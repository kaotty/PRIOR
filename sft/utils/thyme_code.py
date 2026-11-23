#!/usr/bin/env python3
"""Convert Thyme's computation split to formal compute-only Code trajectories."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import io
import json
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .records import _truncate
from .records import _normalize_messages, _output_schema
from .tool_schema import FORMAL_TWO_TOOL_SCHEMAS
from .parse_traces import normalize_final_answer


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
OUTPUT_RE = re.compile(r"<sandbox_output>(.*?)</sandbox_output>", re.DOTALL | re.IGNORECASE)
QUESTION_RE = re.compile(r"^\s*<image>\s*(.*?)\s*###\s*User Image Path", re.DOTALL | re.IGNORECASE)
EXECUTION_ERROR_RE = re.compile(
    r"(?:traceback \(most recent call last\)|syntaxerror|nameerror|filenotfounderror|"
    r"modulenotfounderror|permissionerror|timed?\s*out|\berror:)",
    re.IGNORECASE,
)
FORBIDDEN_CODE_RE = re.compile(
    r"(?:\b(?:PIL|cv2|opencv|pytesseract|easyocr|imageio|skimage|matplotlib|seaborn)\b|"
    r"\b(?:plt|requests|urllib|socket|pathlib|glob|subprocess)\s*\.|"
    r"\b(?:user_image_path|input_image|image_path|processed_path|base64)\b|"
    r"\b(?:imread|imshow|crop|resize|cvtcolor)\s*\(|"
    r"\b(?:open|exec|eval|compile|input)\s*\(|"
    r"(?:https?|file)://|/mnt/data|\.(?:png|jpe?g|webp|gif)\b)",
    re.IGNORECASE,
)
FORBIDDEN_OUTPUT_RE = re.compile(
    r"(?:<image>|data:image/|(?:https?|file)://|/mnt/data|\.(?:png|jpe?g|webp|gif)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedResponse:
    calls: list[tuple[str, str, str]]
    tail_reasoning: str
    answer: str


@dataclass(frozen=True)
class Candidate:
    source_path: Path
    row_index: int
    fingerprint: str
    call_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-dir",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    return parser.parse_args()


def _density(call_count: int) -> str:
    if call_count == 1:
        return "low"
    if call_count <= 3:
        return "medium"
    return "high"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_fences(code: str) -> str:
    code = textwrap.dedent(code).strip()
    code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```$", "", code)
    return textwrap.dedent(code).strip()


def _clean_question(raw: str) -> str:
    match = QUESTION_RE.search(raw)
    if match is None:
        raise ValueError("unsupported_question_template")
    question = match.group(1).strip()
    if not question:
        raise ValueError("empty_question")
    return question


def _decode_original(images: Sequence[str]) -> bytes:
    if len(images) != 1 or not images[0]:
        raise ValueError("requires_exactly_one_original_image")
    try:
        data = base64.b64decode(images[0], validate=True)
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("invalid_base64_image") from exc
    return data


def _question_fingerprint(question: str, image_bytes: bytes) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L").resize((17, 16))
        pixels = gray.tobytes()
    bits = [pixels[y * 17 + x] > pixels[y * 17 + x + 1] for y in range(16) for x in range(16)]
    difference_hash = bytes(
        sum((1 << (7 - offset)) if bits[index + offset] else 0 for offset in range(8))
        for index in range(0, 256, 8)
    )
    return hashlib.sha256(normalized.encode() + b"\0" + difference_hash).hexdigest()


def _validate_code(code: str) -> None:
    if not code:
        raise ValueError("empty_code")
    if FORBIDDEN_CODE_RE.search(code):
        raise ValueError("perception_file_or_network_code")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError("invalid_python") from exc


def _parse_response(response: str) -> ParsedResponse:
    think_matches = THINK_RE.findall(response)
    answer_matches = ANSWER_RE.findall(response)
    if len(think_matches) != 1 or len(answer_matches) != 1 or not answer_matches[0].strip():
        raise ValueError("missing_or_ambiguous_think_answer")
    body = think_matches[0]
    calls: list[tuple[str, str, str]] = []
    cursor = 0
    seen_code: set[str] = set()
    seen_output: set[str] = set()
    for code_match in CODE_RE.finditer(body):
        output_match = OUTPUT_RE.search(body, code_match.end())
        next_code = CODE_RE.search(body, code_match.end())
        if output_match is None or (next_code is not None and next_code.start() < output_match.start()):
            raise ValueError("code_without_immediate_output")
        between = body[code_match.end() : output_match.start()]
        if re.sub(r"\s+", "", between):
            raise ValueError("text_between_code_and_output")
        reasoning = body[cursor : code_match.start()].strip()
        code = _strip_fences(code_match.group(1))
        output = output_match.group(1).strip()
        _validate_code(code)
        if not output:
            raise ValueError("empty_sandbox_output")
        if EXECUTION_ERROR_RE.search(output):
            raise ValueError("execution_error")
        if FORBIDDEN_OUTPUT_RE.search(output):
            raise ValueError("non_text_sandbox_output")
        normalized_code = re.sub(r"\s+", " ", code).strip()
        normalized_output = re.sub(r"\s+", " ", output).strip()
        if normalized_code in seen_code or normalized_output in seen_output:
            raise ValueError("duplicate_code_or_output")
        seen_code.add(normalized_code)
        seen_output.add(normalized_output)
        calls.append((reasoning, code, output))
        cursor = output_match.end()
    if not calls:
        raise ValueError("no_code_call")
    if len(CODE_RE.findall(body)) != len(calls) or len(OUTPUT_RE.findall(body)) != len(calls):
        raise ValueError("code_output_count_mismatch")
    tail = body[cursor:].strip()
    if not tail:
        raise ValueError("empty_final_reasoning")
    return ParsedResponse(calls=calls, tail_reasoning=tail, answer=answer_matches[0].strip())


def _inventory(paths: Sequence[Path]) -> tuple[list[Candidate], dict[str, Any]]:
    from sft.frozen import source_selection

    by_name = {path.name: path for path in paths}
    selected = []
    for record in source_selection("thyme_compute"):
        selected.append(Candidate(
            source_path=by_name[record["source_file"]], row_index=record["source_row_index"],
            fingerprint=record["question_image_fingerprint"], call_count=record["call_count"],
        ))
    return selected, {"unique_question_image_pairs": len(selected),
                      "selection_rule": "frozen raw record locators; independent of local filesystem path"}


def _extra_info(candidate: Candidate) -> dict[str, Any]:
    source = {
        "dataset": "Kwai-Keye/Thyme-SFT",
        "split": "computation",
        "license": "MIT",
        "question_image_fingerprint": candidate.fingerprint,
        "filter": "one image; computation-only Python; text output; no execution error",
    }
    return {
        "sample_id": f"thyme/{candidate.fingerprint}",
        "source_parquet": str(candidate.source_path),
        "source_row_index": candidate.row_index,
        "source_group": "thyme.compute_only",
        "domain_group": "math_and_chart",
        "pool_role": "call_density_candidate_core",
        "shared_anchor_type": "",
        "tool_type": "code",
        "difficulty": "",
        "density": _density(candidate.call_count),
        "no_tool_correct_count": None,
        "oracle_tool_correct_count": None,
        "no_tool_success_rate": None,
        "oracle_tool_success_rate": None,
        "oracle_tool_gain": None,
        "actual_tool_calls": candidate.call_count,
        "effective_tool_calls": None,
        "assistant_loss_tokens": None,
        "experimental_arm": "",
        "selection_seed": 42,
        "spec_sha256": "",
        "labels_sha256": "",
        "source_extra_json": json.dumps(source, sort_keys=True, ensure_ascii=False),
    }


def _convert(row: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    question = _clean_question(str(row.get("question") or ""))
    original_bytes = _decode_original(list(row.get("image") or []))
    parsed = _parse_response(str(row.get("response") or ""))
    if len(parsed.calls) != candidate.call_count:
        raise ValueError("inventory_conversion_call_mismatch")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"<image>\n{question}"},
    ]
    for reasoning, code, output in parsed.calls:
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{reasoning}</think>",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "code_interpreter",
                            "arguments": json.dumps({"code": code}, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "content": _truncate(output)})
    final = f"<think>{parsed.tail_reasoning}</think><answer>{parsed.answer}</answer>"
    messages.append({"role": "assistant", "content": normalize_final_answer(final).strip()})
    messages = _normalize_messages(messages)
    if sum(message["content"].count("<image>") for message in messages) != 1:
        raise ValueError("image_placeholder_mismatch")
    return {
        "messages": messages,
        "images": [{"bytes": original_bytes, "path": ""}],
        "tools": copy.deepcopy(FORMAL_TWO_TOOL_SCHEMAS),
        "data_source": "thyme.compute_only",
        "ability": "math_code",
        "extra_info": _extra_info(candidate),
    }


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output = Path(args.output).resolve()
    paths = sorted(input_dir.glob("computation-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Thyme computation shards under {input_dir}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    selected, report = _inventory(paths)
    wanted = {(candidate.source_path, candidate.row_index): candidate for candidate in selected}
    output.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = pq.ParquetWriter(output, _output_schema(), compression="zstd")
    converted_hist: Counter[str] = Counter()
    conversion_rejected: Counter[str] = Counter()
    written = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            offset = 0
            for batch in parquet.iter_batches(batch_size=16):
                output_rows = []
                for local_index, row in enumerate(batch.to_pylist()):
                    candidate = wanted.get((path, offset + local_index))
                    if candidate is None:
                        continue
                    try:
                        converted = _convert(row, candidate)
                    except (TypeError, ValueError) as exc:
                        conversion_rejected[str(exc)] += 1
                        continue
                    converted_hist[_density(candidate.call_count)] += 1
                    output_rows.append(converted)
                if output_rows:
                    writer.write_table(pa.Table.from_pylist(output_rows, schema=_output_schema()))
                    written += len(output_rows)
                offset += batch.num_rows
    finally:
        writer.close()

    report.update(
        {
            "converted_rows": written,
            "converted_density_counts": dict(converted_hist),
            "conversion_rejected": dict(conversion_rejected),
            "output": str(output),
            "output_sha256": _sha256(output),
        }
    )
    report_path = output.with_name("build_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
