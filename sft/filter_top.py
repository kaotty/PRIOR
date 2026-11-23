"""Rank High SFT demonstrations and select matched token-budget arms."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from verl.utils import hf_tokenizer
from verl.utils.chat_template import apply_chat_template, extract_system_prompt_and_generation
from verl.utils.tokenizer import normalize_token_ids

COUNTS = {"code": 375, "zoom": 375, "none": 250}


TOOL_NAME = {"code": "code_interpreter", "zoom": "image_zoom_in_tool"}


ARMS = {
    "top36_e2": {
        "counts": {"code": 270, "zoom": 270, "none": 180},
        "epochs": 2,
        "global_batch_size": 29,
        "loader_epochs": 3,
    },
    "top16_e4": {
        "counts": {"code": 120, "zoom": 120, "none": 80},
        "epochs": 4,
        "global_batch_size": 26,
        "loader_epochs": 5,
    },
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable(x, salt):
    return hashlib.sha256(f"{salt}:{x}".encode()).hexdigest()


def choose_direct(pool, count, target, seed):
    """Deterministically find a fixed-cardinality subset with exact token sum."""
    best = None
    for restart in range(100):
        rng = random.Random(seed + restart)
        selected = set(rng.sample(pool, count))
        total = sum(x[2] for x in selected)
        for _ in range(1000):
            outside = [x for x in pool if x not in selected]
            move = None
            gap = abs(total - target)
            for old in sorted(selected, key=lambda x: x[1]):
                wanted = target - (total - old[2])
                new = min(outside, key=lambda x: abs(x[2] - wanted))
                new_total = total - old[2] + new[2]
                new_gap = abs(new_total - target)
                if new_gap < gap:
                    gap = new_gap
                    move = (old, new, new_total)
            if move is None:
                break
            selected.remove(move[0])
            selected.add(move[1])
            total = move[2]
            if total == target:
                break
        rank = (abs(total - target), tuple(sorted(x[1] for x in selected)))
        if best is None or rank < best[0]:
            best = (rank, list(selected), total)
        if rank[0] == 0:
            break
    assert best is not None and best[0][0] == 0, (count, target, best[0][0])
    return [
        {
            "row_index": x[0],
            "sample_id": x[1],
            "tool": "none",
            "source": "direct",
            "calls": 0,
            "pre": [],
            "mean_pre": None,
            "assistant_tokens": x[2],
        }
        for x in sorted(best[1], key=lambda x: x[1])
    ]


def build_rank(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    tok = hf_tokenizer(str(args.tokenizer), local_files_only=True)
    tok.chat_template = args.template.read_text()
    sysids, genids = extract_system_prompt_and_generation(tok)
    rows = pq.read_table(args.source).to_pylist()
    records = []
    for ri, row in enumerate(rows):
        ex = row["extra_info"]
        tool = str(ex["tool_type"])
        vals = []
        for m in row["messages"]:
            if m["role"] != "assistant" or not m.get("tool_calls"):
                continue
            ids = normalize_token_ids(
                apply_chat_template(
                    tok,
                    messages=[m],
                    tools=None,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=False,
                )
            )
            ids = ids[len(sysids) + len(genids) :]
            render = tok.decode(ids, skip_special_tokens=False)
            mark = render.index("<tool_call>")
            vals.append(len(tok.encode(render[:mark], add_special_tokens=False)))
        expected = int(ex["actual_tool_calls"])
        assert len(vals) == expected
        if tool == "none":
            assert not vals
            mean = None
        else:
            assert tool in ("code", "zoom") and expected >= 4
            mean = statistics.fmean(vals)
        records.append(
            {
                "row_index": ri,
                "sample_id": ex["sample_id"],
                "tool": tool,
                "source": ex["source_group"],
                "calls": expected,
                "pre": vals,
                "mean_pre": mean,
                "assistant_tokens": int(ex["assistant_loss_tokens"]),
            }
        )
    by = {k: [x for x in records if x["tool"] == k] for k in COUNTS}
    assert {k: len(v) for k, v in by.items()} == {"code": 750, "zoom": 750, "none": 500}
    direct = sorted(by["none"], key=lambda x: stable(x["sample_id"], "shared-direct"))[:250]
    filtered = (
        direct
        + sorted(
            by["code"], key=lambda x: (-x["mean_pre"], stable(x["sample_id"], "filtered-code"))
        )[:375]
        + sorted(
            by["zoom"], key=lambda x: (-x["mean_pre"], stable(x["sample_id"], "filtered-zoom"))
        )[:375]
    )
    # Only this ranking is consumed by the current 25%/50% token-budget arms.
    # Their random controls are built separately with exact count/token matching.
    selected = sorted(filtered, key=lambda x: stable(x["sample_id"], "order-filtered"))
    destination = args.output / "filtered"
    destination.mkdir()
    selection = [{**record, "exposures": 2} for record in selected]
    (destination / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    manifest = {
        "status": "PASS", "schema_version": "tir-high-precall-ranking-v1",
        "source_sha256": sha(args.source), "rows": len(selection),
        "selection_sha256": sha(destination / "selection.json"),
        "ranking": "descending mean pre-call token count within Zoom/Code",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))



def build_budget(args: argparse.Namespace) -> None:
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    table = pq.read_table(args.source)
    rows = table.to_pylist()
    selection = json.load(open(args.top50_selection))
    by = collections.defaultdict(list)
    for x in selection:
        by[str(x["tool"])].append(x)
    assert {k: len(v) for k, v in by.items()} == {"none": 250, "code": 375, "zoom": 375}
    # The prior top-50 selection file is stored in shuffled training order.  Recover
    # its frozen scientific ordering explicitly from the measured pre-call value.
    for tool in ("code", "zoom"):
        by[tool] = sorted(by[tool], key=lambda x: (-float(x["mean_pre"]), str(x["sample_id"])))
    direct_pool = []
    for i, row in enumerate(rows):
        e = row.get("extra_info") or {}
        if str(e.get("tool_type")) == "none":
            direct_pool.append((i, str(e["sample_id"]), int(e["assistant_loss_tokens"])))
    assert len(direct_pool) == 500
    source_total = sum(int((r.get("extra_info") or {})["assistant_loss_tokens"]) for r in rows)
    summary = {}
    for name, spec in ARMS.items():
        selected = []
        for tool in ("code", "zoom"):
            selected += by[tool][: spec["counts"][tool]]
        target_per_epoch = round(source_total / spec["epochs"])
        direct_target = target_per_epoch - sum(int(x["assistant_tokens"]) for x in selected)
        selected += choose_direct(
            direct_pool, spec["counts"]["none"], direct_target, 42 if name == "top36_e2" else 4242
        )
        selected = sorted(
            selected,
            key=lambda x: hashlib.sha256((name + ":" + str(x["sample_id"])).encode()).hexdigest(),
        )
        out = []
        for x in selected:
            r = copy.deepcopy(rows[int(x["row_index"])])
            r["extra_info"]["experimental_arm"] = "high_precall_" + name
            out.append(r)
        d = args.output / name
        d.mkdir()
        parquet = d / "train.parquet"
        pq.write_table(pa.Table.from_pylist(out, schema=table.schema), parquet, compression="zstd")
        unique_tokens = sum(int(x["assistant_tokens"]) for x in selected)
        pre = [float(x["mean_pre"]) for x in selected if x["mean_pre"] is not None]
        batch = spec["global_batch_size"]
        steps_per_epoch = len(out) // batch
        assert len(out) == sum(spec["counts"].values())
        runtime = []
        for exposure in range(1, spec["epochs"] + 1):
            for x in selected:
                r = copy.deepcopy(rows[int(x["row_index"])])
                r["extra_info"]["sample_id"] = (
                    str(r["extra_info"]["sample_id"]) + f"/{name}_physical_exposure_{exposure}"
                )
                r["extra_info"]["experimental_arm"] = "high_precall_" + name
                runtime.append(r)
        extra = 50 * batch - len(runtime)
        assert extra >= 0
        extra_rows = sorted(
            selected, key=lambda x: (int(x["assistant_tokens"]), str(x["sample_id"]))
        )[:extra]
        for x in extra_rows:
            r = copy.deepcopy(rows[int(x["row_index"])])
            r["extra_info"]["sample_id"] = (
                str(r["extra_info"]["sample_id"]) + f"/{name}_physical_exposure_extra"
            )
            r["extra_info"]["experimental_arm"] = "high_precall_" + name
            runtime.append(r)
        assert len(runtime) == 50 * batch
        runtime_parquet = d / "train_runtime_50step.parquet"
        pq.write_table(
            pa.Table.from_pylist(runtime, schema=table.schema), runtime_parquet, compression="zstd"
        )
        extra_tokens = sum(int(x["assistant_tokens"]) for x in extra_rows)
        v = {
            "status": "READY",
            "schema_version": "tir-high-precall-token-control-v4",
            "arm": name,
            "source": str(args.source),
            "source_sha256": sha(args.source),
            "top50_selection": str(args.top50_selection),
            "top50_selection_sha256": sha(args.top50_selection),
            "selection": "Code/Zoom are exact prefixes of frozen descending pre-call ordering; Direct has no pre-call value and is selected deterministically to make the nominal 2x/4x assistant-token budget nearest-exact",
            "fraction_pct": 36 if name == "top36_e2" else 16,
            "unique_rows": len(out),
            "runtime_rows": len(runtime),
            "row_counts": spec["counts"],
            "requested_effective_epochs": spec["epochs"],
            "physical_full_exposures": spec["epochs"],
            "physical_extra_rows": extra,
            "loader_epochs": 1,
            "global_batch_size": batch,
            "total_steps": 50,
            "actual_example_exposures": 50 * batch / len(out),
            "assistant_tokens_per_nominal_epoch": unique_tokens,
            "assistant_tokens_nominal_training_total": unique_tokens * spec["epochs"],
            "assistant_tokens_runtime_total": unique_tokens * spec["epochs"] + extra_tokens,
            "original_high_assistant_tokens_one_epoch": source_total,
            "assistant_token_gap": unique_tokens * spec["epochs"] - source_total,
            "assistant_token_gap_pct": 100
            * (unique_tokens * spec["epochs"] - source_total)
            / source_total,
            "trajectory_mean_pre_call_tokens": sum(pre) / len(pre),
            "parquet": str(parquet),
            "parquet_sha256": sha(parquet),
            "runtime_parquet": str(runtime_parquet),
            "runtime_parquet_sha256": sha(runtime_parquet),
        }
        assert abs(v["assistant_token_gap"]) <= spec["epochs"] // 2
        (d / "manifest.json").write_text(json.dumps(v, indent=2) + "\n")
        summary[name] = v
    root = {
        "status": "PASS",
        "schema_version": "tir-high-precall-token-control-suite-v1",
        "created_unix": time.time(),
        "arms": summary,
    }
    (args.output / "manifest.json").write_text(json.dumps(root, indent=2) + "\n")
    print(json.dumps(root, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("rank", "budget"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--top50-selection", type=Path)
    args = parser.parse_args()
    if args.stage == "rank":
        if args.tokenizer is None or args.template is None:
            parser.error("--tokenizer and --template are required for ranking")
        build_rank(args)
    else:
        if args.top50_selection is None:
            parser.error("--top50-selection is required for token-budget selection")
        build_budget(args)


if __name__ == "__main__":
    main()
