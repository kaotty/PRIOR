#!/usr/bin/env python3
"""Build the frozen, prompt-free 16K multimodal RL dataset.

The output contains source questions, answers, and images only.  Tool schemas and
arm-specific instructions are deliberately added by a later preprocessing step.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps


SEED = 20260812
LOCAL_HELDOUT_ORDER = (
    "chartqa", "mathvision", "mathvista", "mmmu", "mmstar",
    "deepeyes_zoom", "mmk12", "geo3k",
)
EXPECTED_EVAL_INDEX_COUNTS = {
    "chartqa": 2500,
    "deepeyes_zoom": 500,
    "geo3k": 601,
    "hr_bench": 800,
    "mathvision": 3038,
    "mathvista": 997,
    "mmk12": 481,
    "mmmu": 895,
    "mmstar": 1495,
    "vstar_bench": 191,
}

SOURCES = {
    "vstar": {
        "repo": "ChenShawn/DeepEyes-Datasets-47k",
        "revision": "5546681e28fa2eda9f60a9ea9dd0cf291216ded3",
        "patterns": ["data_0.1.2_visual_toolbox_v2.parquet"],
    },
    "mmk12": {
        "repo": "FanqingM/MMK12",
        "revision": "372a609268ea79b5e78d90ab173e02c37b486163",
        "patterns": ["data/train-*.parquet"],
    },
    "chartqa": {
        "repo": "HuggingFaceM4/ChartQA",
        "revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "patterns": ["data/train-*.parquet"],
    },
    "geometry3k": {
        "repo": "hiyouga/geometry3k",
        "revision": "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7",
        "patterns": ["data/train-*.parquet"],
    },
}

EVAL_REPOS = {
    "vstar_bench": {
        "repo": "lmms-lab/VStar-Bench",
        "revision": "b44023b4dca749ed8a76b85eb576627d05a1c174",
        "patterns": ["data/test-*.parquet"],
    },
    # The 4K and 8K variants contain the same 800 questions.  The 4K images are
    # sufficient for perceptual-hash decontamination and avoid a 2.8 GB download.
    "hr_bench": {
        "repo": "DreamMr/HR-Bench",
        "revision": "83b9013d6293b85dc507e87199ca52517536939c",
        "patterns": ["hr_bench_4k.parquet"],
    },
}

# Exact, preregistered source/category counts.  16,384 examples equal 32 batches
# at train_batch_size=512.
QUOTAS = {
    "vstar": {"zoom_favored": 5376, "direct": 1024},
    "mmk12": {"zoom_favored": 384, "code_favored": 2816, "both_favored": 2432, "direct": 1024},
    "chartqa": {"zoom_favored": 128, "code_favored": 896, "both_favored": 512, "direct": 256},
    "geometry3k": {"zoom_favored": 256, "code_favored": 768, "both_favored": 128, "direct": 384},
}

EXPECTED_TOTAL = 16_384
assert sum(sum(groups.values()) for groups in QUOTAS.values()) == EXPECTED_TOTAL

COMPUTE_PATTERNS = {
    "average": re.compile(r"\b(?:average|mean)\b|\bmedian (?:value|of (?:all|the (?:bars|values|data)))\b", re.I),
    "sum": re.compile(r"\b(?:sum|total|combined|altogether|in all)\b", re.I),
    "difference": re.compile(
        r"\bdifference\b|\bhow (?:many|much) (?:more|less|higher|lower)\b|\b(?:more|less) than\b", re.I
    ),
    "ratio_or_percent": re.compile(r"\b(?:ratio|percent|percentage|rate|proportion|fraction)\b", re.I),
    "product_or_division": re.compile(r"\b(?:product|multiply|times|divided?|quotient|per each)\b", re.I),
    "geometry": re.compile(r"\b(?:area|perimeter|volume|circumference|radius|diameter|angle|length|distance)\b", re.I),
    "equation": re.compile(r"\b(?:solve|equation|value of [xy]|find [xy])\b", re.I),
}

SPATIAL_PATTERN = re.compile(
    r"\b(?:left|right|above|below|behind|front|between|next to|near|far|top|bottom|side|position)\b", re.I
)
TEXT_PATTERN = re.compile(r"\b(?:text|word|letter|number|sign|label|written|displayed|read|title|name)\b", re.I)
COUNT_PATTERN = re.compile(r"\b(?:how many|number of|count)\b", re.I)
FIGURE_PATTERN = re.compile(r"\b(?:figure|diagram|graph|chart|table|plot|shown|pictured)\b", re.I)
SUBJECTIVE_PATTERN = re.compile(
    r"\b(?:do you think|would you say|look(?:s|ing)?|appear(?:s|ing)?|seem(?:s|ing)?)\b.*"
    r"\b(?:small|large|big|tiny|young|old|beautiful|attractive|nice|good|bad|happy|sad)\b",
    re.I,
)
AMBIGUOUS_ANSWER_PATTERN = re.compile(
    r"\b(?:cannot tell|can't tell|can not tell|unclear|not visible|not possible to determine|unknown)\b", re.I
)
SUBJECTIVE_SIZE_PATTERN = re.compile(
    r"\bhow (?:large|small|old|young|beautiful|attractive)\b|"
    r"\b(?:is|are)\b.*\b(?:beautiful|attractive|nice|good|bad)\b",
    re.I,
)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?")
NUMERIC_ANSWER_PATTERN = re.compile(
    r"^[-+]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:/[-+]?\d+(?:\.\d+)?)?%?$"
)

MMK_UNGRADABLE = [
    ("system_or_matrix", re.compile(r"\\begin\{|\\left\s*\\l?brace|\\lbrace|\\cr\b")),
    ("relation", re.compile(r"[=<>]|\\leq|\\geq|\\neq|\\le\b|\\ge\b")),
    ("degrees", re.compile(r"\\circ|°|\\degree")),
    ("unit", re.compile(r"\d\s*(?:cm|mm|km|kg|m/s|meters?|centimeter)\b", re.I)),
    ("multi_value", re.compile(r"\d\s+\d")),
]


@dataclass(frozen=True)
class ImageInfo:
    sha256: str
    phash: int
    width: int
    height: int
    edge_density: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--heldout-manifest", type=Path, required=True,
        help="JSON mapping each frozen local evaluation source name to its parquet path.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="Require every Hub file to exist in cache.")
    return parser.parse_args()


def stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"<image\s*/?>", " ", text, flags=re.I)
    text = re.split(
        r"\n(?:Answer with the option|Put ONLY the final answer|You FIRST think|Reason carefully|The final answer MUST)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return re.sub(r"\s+", " ", text).strip()


def normalized_question(value: Any) -> str:
    text = clean_question(value).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def clean_math_answer(value: Any) -> str:
    answer = str(value or "").strip()
    for delimiter in ("$$", "$"):
        if len(answer) >= 2 * len(delimiter) and answer.startswith(delimiter) and answer.endswith(delimiter):
            answer = answer[len(delimiter) : -len(delimiter)].strip()
    return answer


def mmk_ungradable_reason(answer: str) -> str | None:
    if not answer:
        return "empty_answer"
    if len(answer) > 30:
        return "answer_too_long"
    for reason, pattern in MMK_UNGRADABLE:
        if pattern.search(answer):
            return reason
    return None


def image_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return base64.b64decode(value)
    if not isinstance(value, dict):
        raise ValueError("image is not a bytes/string/struct value")
    raw = value.get("bytes")
    if raw is not None:
        return bytes(raw)
    path = value.get("path")
    if path and Path(path).is_file():
        return Path(path).read_bytes()
    raise ValueError("image has neither embedded bytes nor a readable path")


def image_info(raw: bytes) -> ImageInfo:
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("L")
        width, height = image.size
        if width < 64 or height < 64:
            raise ValueError("image_too_small")
        resized = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)
        small = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    edges = cv2.Canny(resized, 80, 160)
    dct = cv2.dct(small)
    low = dct[:8, :8]
    median = float(np.median(low.flat[1:]))
    bits = (low > median).flatten()
    phash = sum(int(bit) << index for index, bit in enumerate(bits))
    return ImageInfo(digest, phash, width, height, float(np.count_nonzero(edges) / edges.size))


def phash_segments(value: int) -> Iterator[tuple[int, int]]:
    # Five segments guarantee a shared segment whenever Hamming distance <= 4.
    offset = 0
    for index, width in enumerate((13, 13, 13, 13, 12)):
        yield index, (value >> offset) & ((1 << width) - 1)
        offset += width


class EvalIndex:
    def __init__(self) -> None:
        self.image_sha256: set[str] = set()
        self.phashes: set[int] = set()
        self.phash_buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.phash_details: dict[int, list[ImageInfo]] = defaultdict(list)
        self.heldout_source_ids: dict[str, set[str]] = defaultdict(set)
        self.heldout_source_indices: dict[str, set[str]] = defaultdict(set)
        self.counts: Counter[str] = Counter()

    def add(self, name: str, raw: bytes, question: str) -> None:
        info = image_info(raw)
        self.image_sha256.add(info.sha256)
        if info.phash not in self.phashes:
            self.phashes.add(info.phash)
            for segment in phash_segments(info.phash):
                self.phash_buckets[segment].add(info.phash)
        self.phash_details[info.phash].append(info)
        self.counts[name] += 1

    def image_match(self, info: ImageInfo, *, allow_near: bool) -> str | None:
        if info.sha256 in self.image_sha256:
            return "eval_exact_image"
        if not allow_near:
            return None
        candidates: set[int] = set()
        for segment in phash_segments(info.phash):
            candidates.update(self.phash_buckets.get(segment, ()))
        aspect = info.width / info.height
        for other_hash in candidates:
            if (info.phash ^ other_hash).bit_count() > 2:
                continue
            for other in self.phash_details[other_hash]:
                other_aspect = other.width / other.height
                if abs(aspect - other_aspect) / max(aspect, other_aspect) > 0.01:
                    continue
                if abs(info.edge_density - other.edge_density) <= 0.02:
                    return "eval_near_image"
        return None


def hub_snapshot(spec: dict[str, Any], cache_dir: Path, no_download: bool) -> Path:
    return Path(
        snapshot_download(
            spec["repo"],
            repo_type="dataset",
            revision=spec["revision"],
            allow_patterns=spec["patterns"],
            cache_dir=cache_dir,
            local_files_only=no_download,
            token=False,
        )
    )


def parquet_rows(paths: Iterable[Path], columns: list[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    index = 0
    for path in sorted(paths):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                yield index, row
                index += 1


def local_eval_rows(path: Path) -> Iterator[tuple[bytes, str, str]]:
    parquet = pq.ParquetFile(path)
    columns = ["images", "extra_info.question", "extra_info.index"]
    for batch in parquet.iter_batches(batch_size=32, columns=columns):
        for row in batch.to_pylist():
            images = row.get("images") or []
            extra = row.get("extra_info") or {}
            question = extra.get("question", "")
            if len(images) == 1:
                yield image_bytes(images[0]), question, str(extra.get("index", ""))


def load_heldout_manifest(path: Path) -> dict[str, Path]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or set(entries) != set(LOCAL_HELDOUT_ORDER):
        raise ValueError(f"held-out manifest must contain exactly: {LOCAL_HELDOUT_ORDER}")
    result = {}
    for name in LOCAL_HELDOUT_ORDER:
        raw = entries[name]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"missing held-out parquet path for {name}")
        source = Path(raw).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        result[name] = source
    return result


def build_eval_index(
    cache_dir: Path, no_download: bool, rejects: Counter[str], heldout_files: dict[str, Path]
) -> tuple[EvalIndex, list[Path]]:
    index = EvalIndex()
    used_files: list[Path] = []
    for eval_name in LOCAL_HELDOUT_ORDER:
        path = heldout_files[eval_name]
        used_files.append(path)
        for raw, question, source_index in local_eval_rows(path):
            try:
                index.add(eval_name, raw, question)
                if eval_name == "mmk12":
                    index.heldout_source_indices["mmk12"].add(source_index)
                elif eval_name == "deepeyes_zoom":
                    index.heldout_source_ids["vstar"].add(source_index)
            except (OSError, ValueError):
                rejects["unreadable_eval_image"] += 1

    vstar_snapshot = hub_snapshot(EVAL_REPOS["vstar_bench"], cache_dir, no_download)
    vstar_paths = sorted((vstar_snapshot / "data").glob("test-*.parquet"))
    used_files.extend(vstar_paths)
    for _, row in parquet_rows(vstar_paths, ["image", "text"]):
        try:
            index.add("vstar_bench", image_bytes(row["image"]), row.get("text", ""))
        except (OSError, ValueError):
            rejects["unreadable_eval_image"] += 1

    hr_snapshot = hub_snapshot(EVAL_REPOS["hr_bench"], cache_dir, no_download)
    hr_paths = sorted(hr_snapshot.glob("hr_bench_4k.parquet"))
    used_files.extend(hr_paths)
    for _, row in parquet_rows(hr_paths, ["image", "question"]):
        try:
            index.add("hr_bench", image_bytes(row["image"]), row.get("question", ""))
        except (OSError, ValueError):
            rejects["unreadable_eval_image"] += 1
    return index, used_files


def compute_operation(question: str) -> str:
    for name, pattern in COMPUTE_PATTERNS.items():
        if pattern.search(question):
            return name
    return "none"


def task_family(source: str, question: str, operation: str) -> str:
    if source == "vstar":
        if TEXT_PATTERN.search(question):
            return "text_or_label"
        if SPATIAL_PATTERN.search(question):
            return "spatial_relation"
        if COUNT_PATTERN.search(question):
            return "counting"
        if re.search(r"\b(?:color|material|shape|wearing|holding)\b", question, re.I):
            return "visual_attribute"
        return "object_detail"
    if source == "chartqa":
        return operation if operation != "none" else "direct_chart_reading"
    if source == "geometry3k":
        if re.search(r"\barea\b", question, re.I):
            return "area"
        if re.search(r"\b(?:angle|degree)\b", question, re.I):
            return "angle"
        if re.search(r"\b(?:length|distance|perimeter|circumference)\b", question, re.I):
            return "length"
        return "solve_diagram"
    if re.search(r"\b(?:chart|graph|table|plot)\b", question, re.I):
        return "chart_or_table_math"
    if operation == "geometry" or re.search(r"\b(?:triangle|circle|rectangle|polygon)\b", question, re.I):
        return "geometry_math"
    if operation != "none":
        return operation
    return "other_math"


def utility_scores(source: str, question: str, answer: str, info: ImageInfo, operation: str) -> tuple[int, int]:
    area = info.width * info.height
    detail = int(area >= 300_000) + int(area >= 900_000) + int(max(info.width, info.height) >= 1200)
    detail += int(info.edge_density >= 0.10) + int(max(info.width, info.height) / min(info.width, info.height) >= 1.8)

    zoom = detail
    if source == "vstar":
        zoom += 2 + 2 * bool(TEXT_PATTERN.search(question)) + 2 * bool(SPATIAL_PATTERN.search(question))
        zoom += bool(COUNT_PATTERN.search(question))
    elif source == "chartqa":
        zoom += 1
    elif source == "geometry3k":
        zoom += 1
    zoom += 2 * bool(FIGURE_PATTERN.search(question))

    number_count = len(NUMBER_PATTERN.findall(question))
    code = min(number_count, 4)
    code += 3 * (operation != "none")
    code += int(bool(re.search(r"\b(?:calculate|compute|solve|find)\b", question, re.I)))
    code += int("\\frac" in answer or "/" in answer or "." in answer)
    if source == "geometry3k":
        code += 2
    if source == "vstar":
        code = 0
    return int(zoom), int(code)


def category_score(candidate: dict[str, Any], category: str) -> float:
    zoom = candidate["zoom_score"]
    code = candidate["code_score"]
    if category == "zoom_favored":
        return 2 * zoom - code
    if category == "code_favored":
        return 2 * code - zoom
    if category == "both_favored":
        return 2 * min(zoom, code) + zoom + code
    return -(zoom + code)


def eligible(candidate: dict[str, Any], category: str) -> bool:
    if category == "zoom_favored":
        return candidate["zoom_score"] >= 2
    if category == "code_favored":
        return candidate["code_score"] >= 2
    if category == "both_favored":
        return candidate["zoom_score"] >= 2 and candidate["code_score"] >= 2
    return True


def clean_source_row(source: str, index: int, row: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    if source == "vstar":
        if row.get("data_source") != "vstar":
            raise ValueError("wrong_source")
        extra = row.get("extra_info") or {}
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError("not_one_image")
        question = clean_question(extra.get("question"))
        answer = str(extra.get("answer") or "").strip()
        source_id = str(extra.get("index") or index)
        image = images[0]
    elif source == "mmk12":
        if row.get("subject") not in (None, "math"):
            raise ValueError("not_math")
        question = clean_question(row.get("question"))
        answer = clean_math_answer(row.get("answer"))
        reason = mmk_ungradable_reason(answer)
        if reason:
            raise ValueError(reason)
        source_id = str(row.get("id") or index)
        image = row.get("image")
    elif source == "chartqa":
        if int(row.get("human_or_machine", -1)) != 0:
            raise ValueError("machine_generated_question")
        question = clean_question(row.get("query"))
        labels = row.get("label") or []
        if len(labels) != 1:
            raise ValueError("not_one_answer")
        answer = str(labels[0]).strip()
        source_id = str(index)
        image = row.get("image")
    else:
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError("not_one_image")
        question = clean_question(row.get("problem"))
        answer = clean_math_answer(row.get("answer"))
        source_id = str(index)
        image = images[0]

    if not 4 <= len(question) <= 1000:
        raise ValueError("bad_question_length")
    answer_limit = 256 if source == "vstar" else 64
    if not answer or len(answer) > answer_limit:
        raise ValueError("bad_answer_length")
    if "\ufffd" in question or "\ufffd" in answer:
        raise ValueError("invalid_unicode")
    if source == "vstar" and SUBJECTIVE_PATTERN.search(question):
        raise ValueError("subjective_question")
    if source == "vstar" and SUBJECTIVE_SIZE_PATTERN.search(question):
        raise ValueError("subjective_question")
    if source == "vstar" and AMBIGUOUS_ANSWER_PATTERN.search(answer):
        raise ValueError("ambiguous_answer")
    return source_id, question, answer, image


def source_rows(source: str, paths: list[Path]) -> Iterator[tuple[int, dict[str, Any]]]:
    columns = {
        "vstar": ["data_source", "images", "extra_info"],
        "mmk12": ["id", "question", "answer", "subject", "image"],
        "chartqa": ["image", "query", "label", "human_or_machine"],
        "geometry3k": ["images", "problem", "answer"],
    }[source]
    yield from parquet_rows(paths, columns)


def collect_candidates(
    source: str,
    paths: list[Path],
    evaluation: EvalIndex,
    seed: int,
    rejects: Counter[str],
) -> list[dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    for index, row in source_rows(source, paths):
        try:
            source_id, question, answer, encoded_image = clean_source_row(source, index, row)
            if source_id in evaluation.heldout_source_ids[source]:
                raise ValueError("heldout_source_id")
            if str(index) in evaluation.heldout_source_indices[source]:
                raise ValueError("heldout_source_index")
            raw = image_bytes(encoded_image)
            info = image_info(raw)
        except (OSError, ValueError) as error:
            rejects[f"{source}:{error}"] += 1
            continue

        # Perceptual hashes work well for resized/compressed natural photos, but
        # collapse many distinct white-background charts and geometry diagrams.
        contamination = evaluation.image_match(info, allow_near=source == "vstar")
        if contamination:
            rejects[f"{source}:{contamination}"] += 1
            continue
        normalized = normalized_question(question)

        operation = compute_operation(question)
        zoom_score, code_score = utility_scores(source, question, answer, info, operation)
        family = task_family(source, question, operation)
        fingerprint = hashlib.sha256(
            f"{SOURCES[source]['revision']}\0{source_id}\0{normalized}\0{info.sha256}".encode()
        ).hexdigest()
        candidate = {
            "source": source,
            "source_id": source_id,
            "source_index": index,
            "question": question,
            "answer": answer,
            "image_sha256": info.sha256,
            "image_phash": f"{info.phash:016x}",
            "image_width": info.width,
            "image_height": info.height,
            "edge_density": round(info.edge_density, 6),
            "question_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "operation": operation,
            "task_family": family,
            "zoom_score": zoom_score,
            "code_score": code_score,
            "fingerprint": fingerprint,
        }
        old = by_image.get(info.sha256)
        if old is None or stable_rank(seed, source, fingerprint) < stable_rank(seed, source, old["fingerprint"]):
            if old is not None:
                rejects[f"{source}:duplicate_image"] += 1
            by_image[info.sha256] = candidate
        else:
            rejects[f"{source}:duplicate_image"] += 1
    return list(by_image.values())


def select_category(
    candidates: list[dict[str, Any]],
    used: set[str],
    source: str,
    category: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    ranked = [row for row in candidates if row["fingerprint"] not in used and eligible(row, category)]
    ranked.sort(
        key=lambda row: (
            -category_score(row, category),
            stable_rank(seed, f"{source}:{category}", row["fingerprint"]),
        )
    )
    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    family_cap = max(1, math.ceil(count * 0.50))
    for row in ranked:
        if category != "direct" and family_counts[row["task_family"]] >= family_cap:
            continue
        selected.append(row)
        family_counts[row["task_family"]] += 1
        if len(selected) == count:
            break
    if len(selected) < count:
        selected_ids = {row["fingerprint"] for row in selected}
        for row in ranked:
            if row["fingerprint"] in selected_ids:
                continue
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"{source}/{category}: need {count}, only {len(selected)} eligible rows")
    for row in selected:
        row["expected_tool_need"] = category
        row["selection_score"] = category_score(row, category)
        used.add(row["fingerprint"])
    return selected


def select_rows(candidates: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source, quotas in QUOTAS.items():
        used: set[str] = set()
        # Reserve the clearest direct cases and the strongest dual-utility cases first.
        order = [name for name in ("direct", "both_favored", "zoom_favored", "code_favored") if name in quotas]
        for category in order:
            selected.extend(select_category(candidates[source], used, source, category, quotas[category], seed))
    selected.sort(key=lambda row: stable_rank(seed, "final-order", row["fingerprint"]))
    if len(selected) != EXPECTED_TOTAL or len({row["fingerprint"] for row in selected}) != EXPECTED_TOTAL:
        raise RuntimeError("final selection is not exactly 16,384 unique rows")
    return selected


def materialize_selected(
    selected: list[dict[str, Any]], source_paths: dict[str, list[Path]], rejects: Counter[str]
) -> list[dict[str, Any]]:
    wanted = {(row["source"], row["source_index"]): row for row in selected}
    output: list[dict[str, Any]] = []
    for source, paths in source_paths.items():
        for index, source_row in source_rows(source, paths):
            metadata = wanted.get((source, index))
            if metadata is None:
                continue
            source_id, question, answer, encoded_image = clean_source_row(source, index, source_row)
            raw = image_bytes(encoded_image)
            if hashlib.sha256(raw).hexdigest() != metadata["image_sha256"]:
                raise RuntimeError(f"image changed while materializing {source}:{source_id}")
            item_id = f"{source}:{metadata['fingerprint'][:24]}"
            output.append(
                {
                    "item_id": item_id,
                    "source": source,
                    "source_repo": SOURCES[source]["repo"],
                    "source_revision": SOURCES[source]["revision"],
                    "source_split": "train",
                    "source_id": source_id,
                    "question": question,
                    "answer": answer,
                    "image": {"bytes": raw, "path": f"{item_id}{Path(str(encoded_image.get('path') or '.jpg')).suffix}"},
                    "expected_tool_need": metadata["expected_tool_need"],
                    "task_family": metadata["task_family"],
                    "operation": metadata["operation"],
                    "image_width": metadata["image_width"],
                    "image_height": metadata["image_height"],
                    "image_sha256": metadata["image_sha256"],
                    "image_phash": metadata["image_phash"],
                    "question_sha256": metadata["question_sha256"],
                    "selection_score": float(metadata["selection_score"]),
                    "zoom_score": metadata["zoom_score"],
                    "code_score": metadata["code_score"],
                }
            )
    if len(output) != EXPECTED_TOTAL:
        rejects["materialization_missing"] += EXPECTED_TOTAL - len(output)
        raise RuntimeError(f"materialized {len(output)} of {EXPECTED_TOTAL} selected rows")
    order = {row["fingerprint"][:24]: index for index, row in enumerate(selected)}
    output.sort(key=lambda row: order[row["item_id"].split(":", 1)[1]])
    return output


def validate_output(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"row count is {len(rows)}, expected {EXPECTED_TOTAL}")
    if len({row["item_id"] for row in rows}) != EXPECTED_TOTAL:
        raise RuntimeError("item_id is not unique")
    if len({row["image_sha256"] for row in rows}) != EXPECTED_TOTAL:
        raise RuntimeError("exact image duplicate survived selection")
    forbidden = re.compile(r"tool_call|image_zoom_in_tool|code_interpreter|</?tools>", re.I)
    for row in rows:
        if forbidden.search(row["question"]):
            raise RuntimeError(f"prompt injection survived in {row['item_id']}")
        if not row["image"]["bytes"]:
            raise RuntimeError(f"empty image in {row['item_id']}")
    actual = Counter((row["source"], row["expected_tool_need"]) for row in rows)
    expected = Counter({(source, group): count for source, groups in QUOTAS.items() for group, count in groups.items()})
    if actual != expected:
        raise RuntimeError(f"quota mismatch: actual={actual}, expected={expected}")


def json_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output_dir}; pass --overwrite explicitly")

    temp_dir = args.output_dir.with_name(f".{args.output_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    rejects: Counter[str] = Counter()
    print("[1/5] Resolving pinned source files")
    source_paths: dict[str, list[Path]] = {}
    for name, spec in SOURCES.items():
        snapshot = hub_snapshot(spec, args.cache_dir, args.no_download)
        source_paths[name] = sorted(path for pattern in spec["patterns"] for path in snapshot.glob(pattern))
        if not source_paths[name]:
            raise FileNotFoundError(f"no files matched for {name}: {spec['patterns']}")

    print("[2/5] Building evaluation decontamination index")
    heldout_files = load_heldout_manifest(args.heldout_manifest)
    evaluation, eval_files = build_eval_index(args.cache_dir, args.no_download, rejects, heldout_files)
    if dict(evaluation.counts) != EXPECTED_EVAL_INDEX_COUNTS:
        raise RuntimeError(
            f"held-out evaluation index differs from the frozen build: {dict(evaluation.counts)}"
        )
    print(f"      {len(evaluation.image_sha256)} unique evaluation images")

    print("[3/5] Filtering and scoring source candidates")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for source, paths in source_paths.items():
        candidates[source] = collect_candidates(source, paths, evaluation, args.seed, rejects)
        print(f"      {source}: {len(candidates[source])} clean unique candidates")

    print("[4/5] Applying fixed quotas and materializing images")
    selected = select_rows(candidates, args.seed)
    rows = materialize_selected(selected, source_paths, rejects)
    validate_output(rows)

    train_path = temp_dir / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train_path, compression="zstd", compression_level=3, row_group_size=128)

    samples = []
    sample_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (row["source"], row["expected_tool_need"])
        if sample_counts[key] >= 3:
            continue
        sample_counts[key] += 1
        samples.append({key: value for key, value in row.items() if key != "image"})
    with (temp_dir / "audit_samples.jsonl").open("w") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_counts = Counter(row["source"] for row in rows)
    need_counts = Counter(row["expected_tool_need"] for row in rows)
    family_counts = Counter(f"{row['source']}:{row['task_family']}" for row in rows)
    input_files = sorted({*sum(source_paths.values(), []), *eval_files})
    manifest = {
        "dataset_name": "clean_rl_16k_v1",
        "created_for": "frozen two-tool RL comparison",
        "row_count": len(rows),
        "batches_per_epoch_at_train_batch_size_512": len(rows) // 512,
        "seed": args.seed,
        "prompt_free": True,
        "image_count_per_row": 1,
        "source_counts": json_counter(source_counts),
        "expected_tool_need_counts": json_counter(need_counts),
        "source_by_expected_tool_need": {
            source: dict(groups) for source, groups in QUOTAS.items()
        },
        "task_family_counts": json_counter(family_counts),
        "candidate_counts_after_filtering": {source: len(rows_) for source, rows_ in candidates.items()},
        "rejection_counts": json_counter(rejects),
        "evaluation_index_counts": json_counter(evaluation.counts),
        "filtering": {
            "source_split": "official train only",
            "chartqa_questions": "human-authored only",
            "mmk12_answers": "short, single, rule-gradable answers",
            "vstar_questions": "drop subjective questions and ambiguous answers",
            "images": "decode successfully; both dimensions >=64",
            "internal_dedup": "one row per exact image SHA-256",
            "evaluation_decontamination": (
                "exact SHA-256 for every source; V* natural photos additionally use pHash Hamming distance <=2 "
                "with matching aspect ratio and edge density"
            ),
            "local_heldouts": "exclude MMK12 by original row index and V* by original source ID",
            "selection": "fixed source/category quotas; score by question operation and image detail; deterministic tie break",
        },
        "source_revisions": {source: spec["revision"] for source, spec in SOURCES.items()},
        "evaluation_revisions": {name: spec["revision"] for name, spec in EVAL_REPOS.items()},
        "input_files": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in input_files
        ],
        "train_parquet": {
            "path": "train.parquet",
            "size_bytes": train_path.stat().st_size,
            "sha256": sha256_file(train_path),
        },
        "ordered_item_id_sha256": hashlib.sha256("\n".join(row["item_id"] for row in rows).encode()).hexdigest(),
    }
    (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (temp_dir / "README.md").write_text(
        "# clean_rl_16k_v1\n\n"
        "16,384 frozen RL examples from V*, MMK12, ChartQA, and Geometry3K. "
        "`train.parquet` contains no system prompt, tool schema, or forced reasoning format. "
        "Use a later arm-specific preprocessing step to add prompts. `expected_tool_need` is "
        "selection metadata and must not be exposed to the model. Exact provenance, filters, "
        "counts, and hashes are in `manifest.json`.\n"
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    temp_dir.rename(args.output_dir)
    print("[5/5] Frozen dataset written")
    print(f"      {args.output_dir / 'train.parquet'}")
    print(f"      rows={len(rows)}, sha256={manifest['train_parquet']['sha256']}")
    print(f"      sources={dict(source_counts)}")
    print(f"      expected_tool_need={dict(need_counts)}")


if __name__ == "__main__":
    main()
