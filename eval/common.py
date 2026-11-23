"""Shared canonical benchmark records and streaming Parquet writer.

The final evaluation builders use these helpers; the obsolete six-benchmark
standalone builder is intentionally excluded.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from functools import lru_cache
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

MAX_MODEL_PIXELS = 3_000_000


MIN_SIDE = 28


SYSTEM_PROMPT = (
    "You are a careful multimodal reasoner. Solve the problem from the image and question. "
    "If tools are available, use them only when they are genuinely helpful. The code tool cannot inspect the "
    "original image; it can only compute from quantities and relations you provide."
)


def row_fingerprint(row: Mapping[str, Any]) -> str:
    """Hash semantic record content, ignoring Arrow null padding and row position."""
    def normalize(value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"bytes_sha256": hashlib.sha256(bytes(value)).hexdigest()}
        if isinstance(value, Mapping):
            return {key: normalize(item) for key, item in value.items() if item is not None}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    value = dict(row)
    value["extra_info"] = dict(value.get("extra_info") or {})
    value["extra_info"].pop("index", None)
    encoded = json.dumps(normalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@lru_cache(maxsize=1)
def frozen_suite() -> dict:
    """Question order and content hashes from the frozen five-benchmark suite."""
    with gzip.open(Path(__file__).with_name("frozen_suite.json.gz"), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_struct(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if raw is not None:
            return {"bytes": bytes(raw), "path": None}
        path = value.get("path")
        if path:
            return {"bytes": Path(str(path)).read_bytes(), "path": None}
    if isinstance(value, bytes):
        return {"bytes": value, "path": None}
    if isinstance(value, str):
        try:
            return {"bytes": base64.b64decode(value, validate=True), "path": None}
        except ValueError:
            return {"bytes": Path(value).read_bytes(), "path": None}
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        return {"bytes": buffer.getvalue(), "path": None}
    raise TypeError(f"unsupported image value: {type(value).__name__}")


def _model_image(original: Mapping[str, Any]) -> dict[str, Any]:
    image = Image.open(io.BytesIO(original["bytes"])).convert("RGB")
    width, height = image.size
    pixels = width * height
    changed = False
    if pixels > MAX_MODEL_PIXELS:
        scale = (MAX_MODEL_PIXELS / pixels) ** 0.5
        image = image.resize(
            (max(MIN_SIDE, round(width * scale)), max(MIN_SIDE, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        changed = True
    elif min(width, height) < MIN_SIDE:
        scale = MIN_SIDE / min(width, height)
        image = image.resize(
            (max(MIN_SIDE, round(width * scale)), max(MIN_SIDE, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        changed = True
    if not changed:
        return dict(original)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return {"bytes": buffer.getvalue(), "path": None}


def _image_size(image_struct: Mapping[str, Any]) -> list[int]:
    image = Image.open(io.BytesIO(image_struct["bytes"]))
    return [int(image.width), int(image.height)]


def _canonical_row(
    *,
    benchmark: str,
    data_source: str,
    source_index: str,
    question: str,
    answer: str,
    gt_text: str,
    original_image: Mapping[str, Any],
    model_image: Mapping[str, Any],
    category: str = "",
    cycle_category: int = -1,
) -> dict[str, Any]:
    item_id = f"{benchmark}:test:{source_index}"
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "images": [dict(model_image)],
        "ability": "vl",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "uid": item_id,
        "extra_info": {
            "split": "test",
            "index": -1,
            "item_id": item_id,
            "row_id": item_id,
            "benchmark": benchmark,
            "source_index": source_index,
            "category": category,
            "cycle_category": cycle_category,
            "question": question,
            "answer": answer,
            "gt_text": gt_text,
            "replicate_id": 0,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "image_zoom_in_tool": {
                    "create_kwargs": {
                        "image": dict(original_image),
                        "coordinate_image_size": _image_size(model_image),
                    }
                }
            },
        },
    }


class ShardWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.pending: list[dict[str, Any]] = []
        self.rows = 0

    def append(self, row: dict[str, Any]) -> None:
        self.pending.append(row)
        if len(self.pending) >= 16:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        table = pa.Table.from_pylist(self.pending, schema=self.writer.schema if self.writer else None)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows += len(self.pending)
        self.pending.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            raise ValueError(f"refusing to write empty shard {self.path}")
        self.writer.close()
