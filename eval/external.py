"""Build the paired WeMath or CharXiv evaluation records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import datasets
import pyarrow.parquet as pq

from eval.common import (
    SYSTEM_PROMPT as TOOL_ON_PROMPT,
    ShardWriter,
    _canonical_row,
    _image_struct,
    _model_image,
    _sha256,
    frozen_suite,
    row_fingerprint,
)

SYSTEM_PROMPT = TOOL_ON_PROMPT

TOOL_OFF_PROMPT = (
    "You are a careful multimodal reasoner. Solve the problem from the image and question. "
    "Tools are unavailable in this setting. Answer directly without attempting or describing any tool call."
)


TOOL_OFF = "You are a careful multimodal reasoner. Solve the problem from the image and question. Tools are unavailable in this setting. Answer directly without attempting or describing any tool call."


def condition(row, prompt):
    value = copy.deepcopy(row)
    value["prompt"][0]["content"] = prompt
    return value


def option_map(value):
    result = {}
    for part in re.split(r"\s*;\s*", str(value)):
        m = re.match(r"\s*([A-E])\s*[.)]\s*(.*)\s*$", part)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def wemath_rows(path):
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 1740
    for index, r in enumerate(rows):
        options = option_map(r["option"])
        answer = str(r["answer"]).strip().upper()
        assert answer in options, (index, answer, options)
        original = _image_struct(r["image_path"])
        rendered = "\n".join(f"{k}. {v}" for k, v in options.items())
        row = _canonical_row(
            benchmark="wemath",
            data_source="wemath_mcq",
            source_index=str(index),
            question=f"<image>\n{r['question']}\n{rendered}\nPut only the option letter in \\boxed{{}}.",
            answer=answer,
            gt_text=options[answer],
            original_image=original,
            model_image=_model_image(original),
            category=str(r["knowledge concept"]),
        )
        row["extra_info"].update(
            official_id=str(r["ID"]),
            official_key=str(r["key"]),
            question_number=int(r["question number"]),
            knowledge_concept=str(r["knowledge concept"]),
            knowledge_concept_description=str(r["knowledge concept description"]),
            official_index=index,
        )
        yield row


def verify_rows(name, rows):
    expected = frozen_suite()["benchmarks"][name]
    for mode, prompt in (("tool_on", TOOL_ON_PROMPT), ("tool_off", TOOL_OFF_PROMPT)):
        if len(rows) != len(expected[mode]):
            raise ValueError(f"{name}: expected {len(expected[mode])} rows, found {len(rows)}")
        for row, reference in zip(rows, expected[mode], strict=True):
            value = condition(row, prompt)
            if row_fingerprint(value) != reference["sha256"]:
                raise ValueError(f"frozen {name}/{mode} content mismatch: {reference['item_id']}")


def normalized_question(row):
    q = str(row["extra_info"]["question"]).lower().replace("<image>", " ")
    return re.sub(r"\s+", " ", q).strip()


def image_hash(row):
    return hashlib.sha256(
        row["extra_info"]["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image"]["bytes"]
    ).hexdigest()


def source_pairs(paths):
    q = set()
    im = set()
    pair = set()
    for path in paths:
        if not path.exists():
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                question = ""
                if row.get("extra_info") and row["extra_info"].get("question"):
                    question = str(row["extra_info"]["question"])
                elif row.get("messages"):
                    users = [m for m in row["messages"] if m.get("role") == "user"]
                    question = str(users[0].get("content") or "") if users else ""
                elif row.get("prompt"):
                    users = [m for m in row["prompt"] if m.get("role") == "user"]
                    question = str(users[0].get("content") or "") if users else ""
                question = re.sub(r"\s+", " ", question.lower().replace("<image>", " ")).strip()
                raw = None
                images = row.get("images") or []
                if images:
                    raw = (images[0] or {}).get("bytes")
                if raw:
                    h = hashlib.sha256(bytes(raw)).hexdigest()
                    im.add(h)
                    pair.add(hashlib.sha256((question + "|" + h).encode()).hexdigest())
                if question:
                    q.add(hashlib.sha256(question.encode()).hexdigest())
    return q, im, pair


def build_wemath(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    source_q, source_i, source_p = source_pairs(args.overlap_source)
    source_files = [args.wemath]
    top = {
        "status": "PASS",
        "schema_version": "tir-external-benchmarks-v1",
        "suites": {},
        "source_files": {str(x): _sha256(x) for x in source_files},
        "overlap_sources": {str(x): _sha256(x) for x in args.overlap_source},
    }
    suites = [("wemath", list(wemath_rows(args.wemath)))]
    for name, rows in suites:
        verify_rows(name, rows)
        q = {hashlib.sha256(normalized_question(x).encode()).hexdigest() for x in rows}
        ih = {image_hash(x) for x in rows}
        ph = {
            hashlib.sha256((normalized_question(x) + "|" + image_hash(x)).encode()).hexdigest()
            for x in rows
        }
        overlap = {
            "question_hash": len(q & source_q),
            "image_hash": len(ih & source_i),
            "question_image_pair_hash": len(ph & source_p),
        }
        if overlap["question_image_pair_hash"]:
            raise ValueError((name, overlap))
        selected = sorted(
            rows,
            key=lambda x: hashlib.sha256(
                ("external-gate-v1:" + x["extra_info"]["item_id"]).encode()
            ).hexdigest(),
        )[:64]
        for split, values in [("full", rows), ("gate64", selected)]:
            for cond, prompt in [("tool_on", TOOL_ON_PROMPT), ("tool_off", TOOL_OFF_PROMPT)]:
                path = args.output / name / split / cond / "shard_0" / "test.parquet"
                writer = ShardWriter(path)
                for index, row in enumerate(values):
                    value = condition(row, prompt)
                    value["extra_info"]["index"] = index
                    writer.append(value)
                writer.close()
                manifest = {
                    "suite": f"{name}_v1",
                    "benchmark": name,
                    "split": split,
                    "condition": cond,
                    "total_rows": len(values),
                    "system_prompt": prompt,
                    "shards": [
                        {
                            "shard": 0,
                            "path": str(path.relative_to(args.output)),
                            "rows": len(values),
                            "sha256": _sha256(path),
                        }
                    ],
                }
                (path.parent.parent / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
        top["suites"][name] = {
            "rows": len(rows),
            "gate_rows": 64,
            "overlap": overlap,
            "full_tool_on_sha256": _sha256(args.output / name / "full/tool_on/shard_0/test.parquet"),
            "full_tool_off_sha256": _sha256(args.output / name / "full/tool_off/shard_0/test.parquet"),
        }
    (args.output / "manifest.json").write_text(json.dumps(top, indent=2, sort_keys=True) + "\n")
    print(json.dumps(top, indent=2, sort_keys=True))



def charxiv_norm(q):
    return re.sub(r"\s+", " ", q.lower().replace("<image>", " ")).strip()


def charxiv_image_hash(r):
    return hashlib.sha256(
        r["extra_info"]["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image"]["bytes"]
    ).hexdigest()


def charxiv_source_pairs(paths):
    q = set()
    im = set()
    pair = set()
    for p in paths:
        if not p.exists():
            continue
        for batch in pq.ParquetFile(p).iter_batches(batch_size=16):
            for r in batch.to_pylist():
                e = r.get("extra_info") or {}
                question = str(e.get("question") or "")
                if not question and r.get("messages"):
                    u = [m for m in r["messages"] if m.get("role") == "user"]
                    question = str(u[0].get("content") or "") if u else ""
                if not question and r.get("prompt"):
                    u = [m for m in r["prompt"] if m.get("role") == "user"]
                    question = str(u[0].get("content") or "") if u else ""
                question = charxiv_norm(question)
                raw = None
                imgs = r.get("images") or []
                if imgs:
                    raw = (imgs[0] or {}).get("bytes")
                if raw:
                    h = hashlib.sha256(bytes(raw)).hexdigest()
                    im.add(h)
                    pair.add(hashlib.sha256((question + "|" + h).encode()).hexdigest())
                if question:
                    q.add(hashlib.sha256(question.encode()).hexdigest())
    return q, im, pair


def build_charxiv(args: argparse.Namespace) -> None:
    assert not args.output.exists()
    ds = datasets.Dataset.from_parquet(str(args.charxiv))
    rows = []
    for i, x in enumerate(ds):
        q = str(x.get("reasoning_q") or "").strip()
        ans = str(x.get("reasoning_a") or "").strip()
        img = x.get("image")
        if not q or not ans or img is None:
            continue
        original = _image_struct(img)
        r = _canonical_row(
            benchmark="charxiv",
            data_source="charxiv_reasoning_free",
            source_index=str(i),
            question=f"<image>\n{q}\nPut only the final short answer in \\boxed{{}}.",
            answer=ans,
            gt_text="",
            original_image=original,
            model_image=_model_image(original),
            category=str(x.get("category") or ""),
        )
        r["extra_info"].update(
            original_id=str(x.get("original_id") or ""),
            reasoning_answer_type=int(x.get("reasoning_a_type") or 0),
            num_subplots=int(x.get("num_subplots") or 0),
            official_index=i,
        )
        rows.append(r)
    verify_rows("charxiv", rows)
    sq, si, sp = charxiv_source_pairs(args.overlap_source)
    qhash = {hashlib.sha256(charxiv_norm(x["extra_info"]["question"]).encode()).hexdigest() for x in rows}
    ih = {charxiv_image_hash(x) for x in rows}
    ph = {
        hashlib.sha256(
            (charxiv_norm(x["extra_info"]["question"]) + "|" + charxiv_image_hash(x)).encode()
        ).hexdigest()
        for x in rows
    }
    overlap = {
        "question_hash": len(qhash & sq),
        "charxiv_image_hash": len(ih & si),
        "question_image_pair_hash": len(ph & sp),
    }
    selected = sorted(
        rows,
        key=lambda x: hashlib.sha256(
            ("charxiv-reasoning-gate128-seed42:" + x["extra_info"]["item_id"]).encode()
        ).hexdigest(),
    )[:128]
    for split, values in (("full", rows), ("gate128", selected)):
        for cond, prompt in (("tool_on", SYSTEM_PROMPT), ("tool_off", TOOL_OFF)):
            path = args.output / split / cond / "shard_0/test.parquet"
            w = ShardWriter(path)
            for idx, r in enumerate(values):
                v = copy.deepcopy(r)
                v["prompt"][0]["content"] = prompt
                v["extra_info"]["index"] = idx
                w.append(v)
            w.close()
            (path.parent.parent / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "suite": "charxiv_reasoning_v1",
                        "split": split,
                        "condition": cond,
                        "total_rows": len(values),
                        "overlap": overlap,
                        "shards": [
                            {
                                "shard": 0,
                                "path": str(path.relative_to(args.output)),
                                "rows": len(values),
                                "sha256": _sha256(path),
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
    m = {
        "status": "PASS",
        "schema_version": "tir-charxiv-reasoning-v1",
        "rows": len(rows),
        "gate_rows": 128,
        "overlap": overlap,
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "source_parquet_sha256": _sha256(args.charxiv),
        "overlap_sources": {str(x): _sha256(x) for x in args.overlap_source},
    }
    (args.output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wemath", "charxiv"), required=True)
    parser.add_argument("--wemath", type=Path)
    parser.add_argument("--charxiv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.benchmark == "wemath":
        if args.wemath is None:
            parser.error("--wemath is required for WeMath")
        build_wemath(args)
    else:
        if args.charxiv is None:
            parser.error("--charxiv is required for CharXiv")
        build_charxiv(args)


if __name__ == "__main__":
    main()
