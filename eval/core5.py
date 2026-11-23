"""Rebuild V*, ChartQA and MathVista from their pinned public parquet releases."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import tempfile

import pyarrow.parquet as pq
from PIL import Image

from eval.common import (
    MAX_MODEL_PIXELS, MIN_SIDE, ShardWriter, _canonical_row, _image_struct,
    _model_image, _sha256, frozen_suite, row_fingerprint,
)

TOOL_OFF_PROMPT = (
    "You are a careful multimodal reasoner. Solve the problem from the image and question. "
    "Tools are unavailable in this setting. Answer directly without attempting or describing any tool call."
)
EXPECTED = {"vstar": 191, "chartqa": 2500, "mathvista": 1000}


def records(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        yield from batch.to_pylist()


def rgb(value):
    return Image.open(io.BytesIO(_image_struct(value)["bytes"])).convert("RGB")


def source_rows(benchmark: str, path: Path):
    for index, record in enumerate(records(path)):
        if benchmark == "vstar":
            original = _image_struct(record["image"])
            question = str(record["text"])
            if "\\boxed" not in question:
                question += "\nPut only the option letter in \\boxed{}."
            yield _canonical_row(
                benchmark=benchmark, data_source="vstar_mcq",
                source_index=str(record["question_id"]), question=f"<image>\n{question}",
                answer=str(record["label"]).strip().upper(), gt_text="",
                original_image=original, model_image=_model_image(original),
                category=str(record.get("category") or ""),
            )
        elif benchmark == "chartqa":
            # Match the original ChartQA preprocessing: cap the PIL image first,
            # then encode that same image for the model and Zoom environment.
            image = rgb(record["image"])
            width, height = image.size
            if width * height > MAX_MODEL_PIXELS:
                scale = (MAX_MODEL_PIXELS / (width * height)) ** 0.5
                image = image.resize((max(MIN_SIDE, round(width * scale)),
                                      max(MIN_SIDE, round(height * scale))), Image.Resampling.LANCZOS)
            elif min(width, height) < MIN_SIDE:
                scale = MIN_SIDE / min(width, height)
                image = image.resize((max(MIN_SIDE, round(width * scale)),
                                      max(MIN_SIDE, round(height * scale))), Image.Resampling.LANCZOS)
            original = _image_struct(image)
            answer = record.get("answer", record.get("label"))
            if isinstance(answer, (list, tuple)):
                answer = answer[0]
            question = str(record.get("question") or record["query"])
            yield _canonical_row(
                benchmark=benchmark, data_source="chartqa", source_index=str(index),
                question=f"<image>\n{question}\nPut ONLY the final answer in \\boxed{{}}.",
                answer=str(answer), gt_text="", original_image=original, model_image=original,
            )
        elif benchmark == "mathvista":
            original = _image_struct(rgb(record.get("decoded_image") or record["image"]))
            question = str(record["question"])
            answer = str(record["answer"]).strip()
            choices = record.get("choices") or []
            gt_text = ""
            data_source = "mathvista_free"
            if record.get("question_type") == "multi_choice" and choices:
                gt_text = answer
                answer = chr(ord("A") + [str(choice).strip() for choice in choices].index(answer))
                options = "\n".join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices))
                question += f"\n{options}\nPut only the option letter in \\boxed{{}}."
                data_source = "mathvista_mcq"
            else:
                question += "\nPut only the final answer in \\boxed{}."
            yield _canonical_row(
                benchmark=benchmark, data_source=data_source,
                source_index=str(record.get("pid") or index), question=f"<image>\n{question}",
                answer=answer, gt_text=gt_text, original_image=original,
                model_image=_model_image(original), category=str(record.get("task") or ""),
            )
        else:
            raise ValueError(f"unsupported core benchmark: {benchmark}")


def build(inputs: dict[str, Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frozen = frozen_suite()["benchmarks"]
    with tempfile.TemporaryDirectory(prefix=".core5-build-", dir=output) as temporary:
        staging = Path(temporary)
        manifests = {mode: {"status": "PASS", "counts": EXPECTED, "subsets": {}}
                     for mode in ("tool_on", "tool_off")}
        for benchmark, expected_count in EXPECTED.items():
            raw = {}
            for row in source_rows(benchmark, inputs[benchmark]):
                key = row["extra_info"]["source_index"]
                if key in raw:
                    raise ValueError(f"duplicate source ID for {benchmark}: {key}")
                raw[key] = row
            if len(raw) != expected_count:
                raise ValueError(f"{benchmark}: expected {expected_count} rows, found {len(raw)}")
            for mode in ("tool_on", "tool_off"):
                category = f"core5_{mode.replace('_', '')}_v1"
                path = staging / category / benchmark / mode / "shard_0/test.parquet"
                writer = ShardWriter(path)
                try:
                    for index, expected in enumerate(frozen[benchmark][mode]):
                        row = copy.deepcopy(raw[expected["source_index"]])
                        row["extra_info"]["index"] = index
                        if mode == "tool_off":
                            row["prompt"][0]["content"] = TOOL_OFF_PROMPT
                        if row_fingerprint(row) != expected["sha256"]:
                            raise ValueError(f"frozen {benchmark}/{mode} content mismatch: {expected['item_id']}")
                        writer.append(row)
                    writer.close()
                except BaseException:
                    if writer.writer is not None:
                        writer.writer.close()
                    raise
                relative = str(path.relative_to(staging / category))
                manifest = {"status": "PASS", "benchmark": benchmark, "condition": mode,
                            "total_rows": expected_count, "source_sha256": _sha256(inputs[benchmark]),
                            "shards": [{"shard": 0, "path": relative, "rows": expected_count,
                                        "sha256": _sha256(path)}]}
                (path.parent.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                manifests[mode]["subsets"][benchmark] = {"parquet": relative, "sha256": _sha256(path)}
            del raw
            print(f"{benchmark}: {expected_count} frozen records matched in both modes", flush=True)
        for mode, manifest in manifests.items():
            category = f"core5_{mode.replace('_', '')}_v1"
            (staging / category / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            target = output / category
            if target.exists():
                if not (target / "manifest.json").is_file():
                    raise FileExistsError(f"incomplete pre-existing output: {target}")
            else:
                (staging / category).rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build({name: getattr(args, name) for name in EXPECTED}, args.output)


if __name__ == "__main__":
    main()
