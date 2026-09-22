"""Audit private native stage groups and export only aggregate comparison evidence."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


def require(ok, message):
    if not ok:
        raise DiagnosticError(message)


PAIRS = {
    "old": ("old_m1", "old_m8"),
    "fixed": ("fix1_m1", "fix1_m8"),
    "fix1_modes": ("fix1_eager_m8", "fix1_m8"),
    "final_modes": ("final_eager_m8", "final_m8"),
}


def expected_instances(stage):
    if stage == "Embedding + first input normalization":
        return {"0"}
    if stage == "Final normalization/layout":
        return {"final"}
    if stage == "Full BF16 target head":
        return {"head"}
    if stage == "Layer input residual/normalization":
        return {str(i) for i in range(1, 64)}
    if stage.startswith("MLP ") or stage == "Post-attention/GDN residual/normalization":
        return {str(i) for i in range(64)}
    if stage == "Attention Q/K normalization":
        return {f"{i}:{kind}" for i in range(3, 64, 4) for kind in ("q", "k")}
    if stage.startswith("GDN "):
        return {str(i) for i in range(64) if i % 4 != 3}
    if stage.startswith("Attention "):
        return {str(i) for i in range(3, 64, 4)}
    raise DiagnosticError("unadmitted native stage")


def instance_id(value):
    match = re.search(r"\.layers\.(\d+)\.", value)
    return match[1] if match else value


def aggregate_groups(groups):
    """A position passes a stage only if every admitted layer instance passes."""
    inventories = None
    totals = defaultdict(lambda: defaultdict(Counter))
    instance_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    local = defaultdict(lambda: defaultdict(Counter))
    for group in groups:
        authenticate(group)
        require(group["status"] == "REFERENCE_REPLAY_CHECKED", "unchecked reference replay")
        require(group["positions"] == 8, "wrong native group width")
        for key in (
            "reference_full_logits_exact",
            "injected_output_fault_detected",
            "injected_prefix_fault_detected",
            "authoritative_cache_restored",
        ):
            require(group[key] is True, "reference or negative control failed: " + key)
        require(group["cache_regions"] in (64, 80), "incomplete state adapter inventory")
        by_stage = defaultdict(dict)
        for item in group["stages"]:
            stage, instance = item["stage"], instance_id(item["instance"])
            require(instance not in by_stage[stage], "duplicate native stage instance")
            by_stage[stage][instance] = item
            for variant, receipt in item["variants"].items():
                require(
                    type(receipt["local_output_exact"]) is bool
                    and type(receipt["local_state_exact"]) is bool,
                    "invalid local equality receipt",
                )
                exact = receipt["local_output_exact"] and receipt["local_state_exact"]
                require(
                    receipt["suffix"]
                    == ("validated reference reuse" if exact else "native full suffix replay"),
                    "unequal stage bypassed the native vocabulary suffix",
                )
                local[stage][variant]["invocations"] += 1
                local[stage][variant]["local_output_exact"] += receipt["local_output_exact"]
                local[stage][variant]["local_state_exact"] += receipt["local_state_exact"]
                local[stage][variant]["native_suffix_replays"] += not exact
        inventory = {}
        for stage, instances in by_stage.items():
            require(
                set(instances) == expected_instances(stage), "incomplete layer coverage: " + stage
            )
            columns = set(next(iter(instances.values()))["comparisons"])
            require(columns and columns <= PAIRS.keys(), "invalid comparison columns")
            inventory[stage] = {"instances": sorted(instances), "columns": sorted(columns)}
            for column in columns:
                results = []
                for instance, item in instances.items():
                    require(set(item["comparisons"]) == columns, "uneven comparison coverage")
                    require(set(PAIRS[column]) <= item["variants"].keys(), "unbound comparison")
                    rows = item["comparisons"][column]
                    require(len(rows) == 8, "missing native comparison positions")
                    for row in rows:
                        instance_result = instance_totals[stage][instance][column]
                        instance_result["positions"] += 1
                        instance_result["full_logits_exact"] += row["full_logits_exact"]
                        for k in ("1", "10", "20"):
                            values = row[k]
                            require(
                                type(values["set_exact"]) is bool
                                and type(values["ranked_exact"]) is bool,
                                "non-boolean equality result",
                            )
                            require(
                                not values["ranked_exact"] or values["set_exact"],
                                "ordering equality without set equality",
                            )
                            require(0 <= values["overlap"] <= int(k), "invalid top-k overlap")
                            instance_result[f"top{k}_set_exact"] += values["set_exact"]
                            instance_result[f"top{k}_order_exact"] += values["ranked_exact"]
                    results.append(rows)
                target = totals[stage][column]
                for position in range(8):
                    target["positions"] += 1
                    target["full_logits_exact"] += all(
                        r[position]["full_logits_exact"] for r in results
                    )
                    for k in ("1", "10", "20"):
                        target[f"top{k}_set_exact"] += all(
                            r[position][k]["set_exact"] for r in results
                        )
                        target[f"top{k}_order_exact"] += all(
                            r[position][k]["ranked_exact"] for r in results
                        )
                target["top1_exact"] = target["top1_set_exact"]
        if inventories is None:
            inventories = inventory
        require(inventory == inventories, "stage/column coverage changed between position groups")
    require(bool(inventories), "empty native comparison domain")
    return {
        "stages": {
            stage: {column: dict(values) for column, values in columns.items()}
            for stage, columns in totals.items()
        },
        "inventory": inventories,
        "per_instance": {
            stage: {
                instance: {column: dict(values) for column, values in columns.items()}
                for instance, columns in instances.items()
            }
            for stage, instances in instance_totals.items()
        },
        "local_checks": {
            stage: {variant: dict(values) for variant, values in variants.items()}
            for stage, variants in local.items()
        },
    }


def analyze(public, private, release_public, release_private, sources):
    def read(path):
        value = private_json(path)
        authenticate(value)
        return value

    measurement = read(public / "measurement.json")
    lane = public / "fixed-bf16"
    runtime = read(lane / "actual-runtime.json")
    sample = read(lane / "pass-00.json")
    summary = read(lane / "summary.json")
    config = read(lane / "requested-config.json")
    rows = read(private / "fixed-bf16/pass-00/correctness/rows.json")
    release = read(release_private / "fixed-bf16/pass-00/correctness/rows.json")
    release_runtime = read(release_public / "fixed-bf16/actual-runtime.json")
    release_sample = read(release_public / "fixed-bf16/pass-00.json")
    require(
        measurement["execution_mode"] == "compiled-no-graphs" and measurement["correctness"],
        "native stage run is not an untimed compiled replay",
    )
    require(
        runtime["compilation_mode"] == 3
        and runtime["graph_mode"] == "NONE"
        and not runtime["enforce_eager"],
        "wrong native execution mode",
    )
    require(
        runtime["repair"] is not None and runtime["performance"] is not None,
        "reference repairs missing",
    )
    for key in ("effective_capacity", "diagnostic_sources"):
        require(runtime[key] == release_runtime[key], "reference identity changed: " + key)
    for key in ("packages", "flags", "compiler_settings", "kernel"):
        require(
            runtime["runtime"][key] == release_runtime["runtime"][key],
            "reference runtime changed: " + key,
        )
    require(summary["passes"] == [sample["sha256"]], "incomplete native run")
    require(
        sample["fixture"]
        == rows["continuation"]
        == release["continuation"]
        == measurement["fixture"],
        "different private fixtures",
    )
    require(sample["observation"]["forced"]["sha256"] == rows["sha256"], "unbound native rows")
    require(
        release_sample["observation"]["forced"]["sha256"] == release["sha256"],
        "unbound release rows",
    )
    require(len(rows["rows"]) == len(release["rows"]) == 320, "wrong reference domain")
    for index, (left, right) in enumerate(zip(rows["rows"], release["rows"], strict=True)):
        require(
            left["position"] == right["position"] == index
            and left["absolute_position"] == right["absolute_position"] == 60000 + index,
            "misaligned native positions",
        )
        require(
            left["logits"]["logits_sha256"] == right["logits"]["logits_sha256"],
            "native reference differs from compiled release",
        )
    require(rows["prefill"] == release["prefill"], "reference prefill differs")
    tape = sample["observation"]["native_tape"]
    require(tape["positions"] == 320 and len(tape["groups"]) == 40, "native sweep incomplete")

    def groups():
        identities = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        for index, digest in enumerate(tape["groups"]):
            group = read(private / f"fixed-bf16/pass-00/native-tape-group-{index:03d}.json")
            require(group["sha256"] == digest, "native group order or content changed")
            if "first_absolute_position" in group:
                require(
                    group["first_absolute_position"] == 60000 + 8 * index,
                    "native stage group is misaligned",
                )
            for name, actual in group.get("diagnostic_sources", {}).items():
                require(identities.get(name) == actual, "native diagnostic source differs")
            if "injected_early_projection_fault_detected" in group:
                require(
                    group["injected_early_projection_fault_detected"] is True,
                    "early native corruption was concealed",
                )
            yield group

    result = aggregate_groups(groups())
    return seal(
        {
            "schema": "qwen.native-isolated-stage-matrix.v1",
            "status": "SAMPLE_CHECKED",
            "positions": 320,
            "prefix_tokens": 60000,
            "fixture": measurement["fixture"],
            "scope": (
                "One stage/layer at a time on final-correct inputs, "
                "through the native full BF16 head"
            ),
            "aggregation": "A position passes only when all layer instances of that stage match",
            "release_full_vector_bridge": 320,
            "prefill_bridge_exact": True,
            "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "receipts": {
                "runtime": runtime["sha256"],
                "sample": sample["sha256"],
                "summary": summary["sha256"],
                "config": config["sha256"],
                "rows": rows["sha256"],
                "release_rows": release["sha256"],
                "measurement": measurement["sha256"],
                "groups": tape["groups"],
            },
            **result,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "private", "release-public", "release-private", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = analyze(
        args.public, args.private, args.release_public, args.release_private, args.source
    )
    write_private(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "positions": result["positions"],
                "stages": len(result["stages"]),
                "sha256": result["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
