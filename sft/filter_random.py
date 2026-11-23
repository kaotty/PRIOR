#!/usr/bin/env python3
"""Prepare fixed-seed random-token data; training is gated on explicit policy."""
import argparse, collections, copy, hashlib, json, math, random, statistics
from bisect import bisect_left
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from eval.common import ShardWriter

SOURCE: Path
REFERENCE: Path
OUTPUT: Path
MODEL: Path
TEMPLATE: Path
SEED = 42
RATIO = {"code": 3, "zoom": 3, "none": 2}
ARM = "random25_e4"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select(records, target_by_tool):
    selected = []
    solver = []
    for offset, tool in enumerate(RATIO):
        pool = [r for r in records if r["tool"] == tool]
        n, metadata_target, rendered_target = target_by_tool[tool]
        metadata_vals = np.asarray([r["assistant_tokens"] for r in pool], dtype=float)
        rendered_vals = np.asarray([r["selection_tokens"] for r in pool], dtype=float)
        A = np.vstack([np.ones(len(pool)), metadata_vals, rendered_vals])
        # A random linear objective chooses one feasible subset. The explicit count
        # and token equalities are the only selection constraints; pre-call fields
        # never enter the objective.
        rng = np.random.default_rng(SEED + offset)
        result = milp(
            rng.normal(size=len(pool)),
            integrality=np.ones(len(pool)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(
                A, [n, metadata_target, rendered_target], [n, metadata_target, rendered_target]
            ),
            options={"time_limit": 120},
        )
        if not result.success:
            raise RuntimeError(f"no exact random feasible subset for {tool}: {result.message}")
        chosen = [pool[i] for i, x in enumerate(result.x) if x > 0.5]
        assert (
            len(chosen) == n
            and sum(x["assistant_tokens"] for x in chosen) == metadata_target
            and sum(x["selection_tokens"] for x in chosen) == rendered_target
        )
        selected.extend(chosen)
        solver.append(
            {
                "tool": tool,
                "rows": n,
                "metadata_tokens": metadata_target,
                "rendered_tokens": rendered_target,
                "status": result.message,
            }
        )
    selected.sort(key=lambda x: hashlib.sha256((ARM + ":" + x["sample_id"]).encode()).hexdigest())
    return selected, {
        "seed": SEED,
        "stratification": RATIO,
        "solver": "scipy.milp_random_objective",
        "solver_constraints": "fixed per-tool row count and exact per-tool metadata and rendered assistant-loss subtotals",
        "solver_details": solver,
        "uses_pre_call_ranking": False,
        "description": "fixed-seed random-objective subset conditioned on the reference arm's per-tool counts and exact metadata/rendered token subtotals; not an unrestricted random sample",
    }


def main():
    global SOURCE, OUTPUT, REFERENCE, MODEL, TEMPLATE, SEED, ARM
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    SOURCE = args.source
    REFERENCE = args.reference
    OUTPUT = args.output
    MODEL = args.model
    TEMPLATE = args.template
    ARM = args.arm
    SEED = args.seed
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    ref = json.loads((REFERENCE / "manifest.json").read_text())
    assert sha(SOURCE) == ref["source_sha256"]
    rows = pq.read_table(SOURCE, columns=["extra_info", "messages", "tools"]).to_pylist()
    records = [
        {
            "row_index": i,
            "sample_id": r["extra_info"]["sample_id"],
            "tool": r["extra_info"]["tool_type"],
            "source": r["extra_info"]["source_group"],
            "calls": r["extra_info"]["actual_tool_calls"],
            "assistant_tokens": int(r["extra_info"]["assistant_loss_tokens"]),
        }
        for i, r in enumerate(rows)
    ]
    assert len(records) == 2000 and len({r["sample_id"] for r in records}) == 2000
    total = sum(r["assistant_tokens"] for r in records)
    metadata_target = ref["assistant_tokens_per_nominal_epoch"]
    assert total == 1450751 and metadata_target == ref["assistant_tokens_per_nominal_epoch"]
    from transformers import AutoTokenizer
    from sft.render_check import _audit_row, _assistant_loss_token_ids

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.chat_template = TEMPLATE.read_text()
    audit = {}
    for i, row in enumerate(rows):
        v = _audit_row(tok, row)
        pre = []
        for m in row["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                text = tok.decode(_assistant_loss_token_ids(tok, m), skip_special_tokens=False)
                pre.append(
                    len(tok.encode(text.split("<tool_call>", 1)[0], add_special_tokens=False))
                )
        audit[i] = {**v, "pre": pre}
        if i % 250 == 0:
            print("render audit", i, flush=True)
    for r in records:
        r["selection_tokens"] = audit[r["row_index"]]["assistant_loss_tokens"]
    reference_rows = pq.read_table(REFERENCE / "train.parquet", columns=["extra_info"]).to_pylist()
    reference_ids = {r["extra_info"]["sample_id"] for r in reference_rows}
    target_by_tool = {
        tool: (
            sum(r["extra_info"]["tool_type"] == tool for r in reference_rows),
            sum(
                int(r["extra_info"]["assistant_loss_tokens"])
                for r in reference_rows
                if r["extra_info"]["tool_type"] == tool
            ),
            sum(
                audit[x["row_index"]]["assistant_loss_tokens"]
                for x in records
                if x["sample_id"] in reference_ids and x["tool"] == tool
            ),
        )
        for tool in RATIO
    }
    metadata_target_from_reference = sum(x[1] for x in target_by_tool.values())
    rendered_target = sum(x[2] for x in target_by_tool.values())
    chosen, method = select(records, target_by_tool)
    for r in chosen:
        r["rendered_assistant_loss_tokens"] = audit[r["row_index"]]["assistant_loss_tokens"]
        r["pre"] = audit[r["row_index"]]["pre"]
        r["mean_pre"] = statistics.fmean(r["pre"]) if r["pre"] else None
    ref_ids = reference_ids
    rendered_source = sum(v["assistant_loss_tokens"] for v in audit.values())
    rendered_ref = sum(
        audit[r["row_index"]]["assistant_loss_tokens"] for r in records if r["sample_id"] in ref_ids
    )
    rendered_chosen = sum(r["rendered_assistant_loss_tokens"] for r in chosen)
    epochs = ref["requested_effective_epochs"]
    steps = ref["total_steps"]
    batch = math.ceil(epochs * len(chosen) / steps)
    extra_n = steps * batch - epochs * len(chosen)
    reference_runtime = json.loads((REFERENCE / "render_audit_runtime.json").read_text())[
        "assistant_loss_tokens"
    ]
    reference_runtime_metadata = ref["assistant_tokens_runtime_total"]
    extra_rendered_target = reference_runtime - epochs * rendered_chosen
    extra_metadata_target = reference_runtime_metadata - epochs * sum(
        r["assistant_tokens"] for r in chosen
    )
    extra_metadata_vals = np.asarray([r["assistant_tokens"] for r in chosen], dtype=float)
    extra_rendered_vals = np.asarray([r["selection_tokens"] for r in chosen], dtype=float)
    extra_result = milp(
        np.random.default_rng(SEED + 100).normal(size=len(chosen)),
        integrality=np.ones(len(chosen)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            np.vstack([np.ones(len(chosen)), extra_metadata_vals, extra_rendered_vals]),
            [extra_n, extra_metadata_target, extra_rendered_target],
            [extra_n, extra_metadata_target, extra_rendered_target],
        ),
        options={"time_limit": 120},
    )
    if not extra_result.success:
        raise RuntimeError(f"no exact random extra-row match: {extra_result.message}")
    extra = [chosen[i] for i, x in enumerate(extra_result.x) if x > 0.5]
    schedule = [(r, e) for e in range(1, epochs + 1) for r in chosen] + [
        (r, "extra") for r in extra
    ]
    OUTPUT.mkdir(parents=True)
    unique = OUTPUT / "train.parquet"
    runtime = OUTPUT / "train_runtime_50step.parquet"
    needed = {r["row_index"] for r in chosen}
    lookup = {}
    pos = 0
    for b in pq.ParquetFile(SOURCE).iter_batches(batch_size=16):
        for r in b.to_pylist():
            if pos in needed:
                lookup[pos] = r
            pos += 1
    for path, sched in [(unique, [(r, None) for r in chosen]), (runtime, schedule)]:
        w = ShardWriter(path)
        for r, e in sched:
            v = copy.deepcopy(lookup[r["row_index"]])
            v["extra_info"]["experimental_arm"] = "high_precall_" + ARM
            if e is not None:
                v["extra_info"]["sample_id"] += f"/{ARM}_physical_exposure_{e}"
            w.append(v)
        w.close()
        index = 0
        for b in pq.ParquetFile(path).iter_batches(batch_size=16):
            for r in b.to_pylist():
                orig = lookup[sched[index][0]["row_index"]]
                for key in ("messages", "images", "tools", "data_source", "ability"):
                    assert r[key] == orig[key], (path, index, key)
                index += 1
        assert index == len(sched)
    result = {
        "status": "READY",
        "schema_version": "tir-random-token-v2",
        "arm": ARM,
        "source": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "reference": str(REFERENCE / "manifest.json"),
        "reference_sha256": sha(REFERENCE / "manifest.json"),
        "selection": method,
        "unique_rows": len(chosen),
        "fraction_pct": 100 * len(chosen) / len(records),
        "row_counts": dict(collections.Counter(r["tool"] for r in chosen)),
        "source_counts": dict(collections.Counter(r["source"] for r in chosen)),
        "assistant_tokens_per_nominal_epoch": metadata_target_from_reference,
        "original_high_assistant_tokens_one_epoch": total,
        "unique_metadata_token_fraction": metadata_target_from_reference / total,
        "rendered_assistant_tokens_per_nominal_epoch": rendered_target,
        "rendered_token_fraction_of_source": rendered_target / rendered_source,
        "requested_effective_epochs": epochs,
        "physical_full_exposures": epochs,
        "physical_extra_rows": extra_n,
        "loader_epochs": 1,
        "global_batch_size": batch,
        "total_steps": steps,
        "runtime_rows": len(schedule),
        "actual_example_exposures": len(schedule) / len(chosen),
        "assistant_tokens_nominal_training_total": epochs * metadata_target_from_reference,
        "assistant_tokens_runtime_total": epochs * metadata_target_from_reference
        + sum(r["assistant_tokens"] for r in extra),
        "assistant_token_gap": epochs * metadata_target_from_reference - total,
        "metadata_runtime_reference": reference_runtime_metadata,
        "metadata_runtime_gap": epochs * sum(r["assistant_tokens"] for r in chosen)
        + sum(r["assistant_tokens"] for r in extra)
        - reference_runtime_metadata,
        "rendered_tokens": {
            "source_unique": rendered_source,
            "reference_unique": rendered_ref,
            "selected_unique": rendered_chosen,
            "selected_fraction_of_source": rendered_chosen / rendered_source,
            "reference_fraction_of_source": rendered_ref / rendered_source,
            "runtime": epochs * rendered_chosen
            + sum(r["rendered_assistant_loss_tokens"] for r in extra),
            "reference_runtime": reference_runtime,
            "runtime_gap": epochs * rendered_chosen
            + sum(r["rendered_assistant_loss_tokens"] for r in extra)
            - reference_runtime,
            "note": "selection and extra rows are matched to Filter25's metadata and actual rendered assistant-loss tokens",
        },
        "trajectory_mean_pre_call_tokens": statistics.fmean(
            r["mean_pre"] for r in chosen if r["mean_pre"] is not None
        ),
        "by_tool_pre_call_mean": {
            t: statistics.fmean(r["mean_pre"] for r in chosen if r["tool"] == t)
            for t in ("code", "zoom")
        },
        "structured_calls": sum(r["calls"] for r in chosen),
        "overlap_with_filter25": sum(r["sample_id"] in ref_ids for r in chosen),
        "content_unchanged": True,
        "full_source_render_audit_passed": True,
        "template_sha256": sha(TEMPLATE),
        "parquet": str(unique),
        "parquet_sha256": sha(unique),
        "runtime_parquet": str(runtime),
        "runtime_parquet_sha256": sha(runtime),
        "batch_policy": "token_matched_four_exposures",
        "training_setup_note": f"The constrained-random subset matches the reference {REFERENCE.name}: row count, tool composition, metadata and rendered assistant-loss token budgets, exposures, batch, and 50 steps. All optimizer, LR, seed, mask, sequence-parallel, model, save, and RL settings are unchanged.",
    }
    (OUTPUT / "selection.json").write_text(json.dumps(chosen, indent=2) + "\n")
    result["selection_sha256"] = sha(OUTPUT / "selection.json")
    (OUTPUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "selection"}, indent=2))


if __name__ == "__main__":
    main()
