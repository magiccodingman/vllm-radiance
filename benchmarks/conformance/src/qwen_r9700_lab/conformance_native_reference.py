"""CPU-only projection of an authenticated serial baseline onto a shorter replay.

This does not execute a model, qualify a native implementation, or select a
cache entry. The caller must bind native configuration and artifacts separately.
Every retained frame is independently copied and its actual bytes authenticated.
"""

from pathlib import Path

from qwen_r9700_lab.conformance_boundaries import validate_domain
from qwen_r9700_lab.conformance_replay import SCHEDULE_SCHEMA, observation_domain, scheduled_inputs
from qwen_r9700_lab.conformance_state import archive_frame, read_frame
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


def _require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def compatible_serial_plans(source, requested):
    for plan in (source, requested):
        authenticate(plan)
        _require("accepted_widths" not in plan, "reference projection requires serial M1 plans")
        for field in ("prefix", "forced_tokens"):
            values = plan.get(field)
            _require(
                isinstance(values, list)
                and bool(values)
                and all(type(v) is int and 0 <= v < 2**31 for v in values),
                "invalid serial token domain",
            )
        positions, _ = observation_domain(plan)
        _require(
            positions
            and all(type(p) is int for p in positions)
            and positions == sorted(set(positions))
            and positions[0] >= 0
            and positions[-1] < len(plan["prefix"]) + len(plan["forced_tokens"]) - 1,
            "invalid serial observation domain",
        )
    variable = {"sha256", "forced_tokens", "observation_positions"}
    _require(
        {k: v for k, v in source.items() if k not in variable}
        == {k: v for k, v in requested.items() if k not in variable},
        "reference contract, artifact or initial prefix changed",
    )
    forced = requested["forced_tokens"]
    _require(
        source["forced_tokens"][: len(forced)] == forced,
        "requested continuation is not the recorded serial prefix",
    )


def _schedule(plan, capture):
    schedule = private_json(capture / "schedule.json")
    authenticate(schedule)
    _require(
        schedule.get("schema") == SCHEDULE_SCHEMA
        and schedule.get("plan") == plan["sha256"]
        and schedule.get("contract") == plan["contract"]
        and schedule.get("initial_state") == "independent_zero_state"
        and schedule.get("published_to_session") is False,
        "serial baseline provenance is incomplete or incompatible",
    )
    expected = list(scheduled_inputs(plan))
    frames = schedule.get("frames", [])
    _require(len(frames) == len(expected), "serial baseline schedule is incomplete")
    for actual, wanted in zip(frames, expected, strict=True):
        _require(
            all(actual.get(k) == v for k, v in wanted.items()),
            "serial baseline did not follow its declared schedule",
        )
    return schedule


def project_serial_reference(
    source_plan, requested_plan, source_capture: Path, output: Path, *, reflink: bool = False
):
    """Retain exactly the requested prefix/domain, with explicit source lineage.

    A failed copy leaves diagnostic partial output without a completion receipt.
    It never alters the source. No physical allocation IDs are compared. Reflinks
    retain independent inodes and verified bytes; unsupported storage fails
    rather than silently copying the full baseline into a limited filesystem.
    """
    compatible_serial_plans(source_plan, requested_plan)
    source_capture, output = Path(source_capture), Path(output)
    schedule = _schedule(source_plan, source_capture)
    boundaries = private_json(source_capture / "boundaries/boundaries.json")
    authenticate(boundaries)
    positions, _ = observation_domain(source_plan)
    requested_positions, input_digests = observation_domain(requested_plan)
    stages = boundaries.get("layer_stages")
    validate_domain(positions, boundaries.get("layers"), stages)
    _require(
        boundaries.get("schema") == "urn:qwen:boundary-schedule:v2"
        and boundaries.get("positions") == positions,
        "serial baseline boundary provenance changed",
    )
    expected_names = [
        f"p{p:09d}-l{layer:03d}-{stage}"
        for p in positions
        for layer, layer_stages in enumerate(stages)
        for stage in layer_stages
    ]
    _require(
        [f["name"] for f in boundaries["frames"]] == expected_names
        and set(requested_positions) <= set(positions),
        "serial baseline is missing required boundary observations",
    )
    output.mkdir(mode=0o700)
    source_frames = {f["name"]: f for f in schedule["frames"]}
    projected_frames = []
    for wanted in scheduled_inputs(requested_plan):
        entry = source_frames[wanted["name"]]
        _require(all(entry[k] == v for k, v in wanted.items()), "unaligned serial prefix frame")
        original = read_frame(source_capture / entry["name"])
        _require(
            original["sha256"] == entry["sha256"]
            and original["contract"] == source_plan["contract"]
            and original["coverage"] == schedule["coverage"]
            and all(original[k] == v for k, v in wanted.items() if k != "name"),
            "serial baseline frame identity changed",
        )
        archive_frame(source_capture / entry["name"], output / entry["name"], reflink=reflink)
        projected_frames.append(entry)
    boundary_output = output / "boundaries"
    boundary_output.mkdir(mode=0o700)
    by_name = {f["name"]: f for f in boundaries["frames"]}
    projected_boundaries = []
    for position in requested_positions:
        for layer, layer_stages in enumerate(stages):
            for stage in layer_stages:
                name = f"p{position:09d}-l{layer:03d}-{stage}"
                entry = by_name[name]
                path = source_capture / "boundaries" / name
                frame = read_frame(path)
                _require(
                    frame["sha256"] == entry["sha256"]
                    and frame["contract"] == source_plan["contract"]
                    and frame["input_digest"] == input_digests[position]
                    and frame["consumed"] == position + 1
                    and frame["pending"] is None
                    and frame["phase"] == "operator"
                    and frame["logical"] == {"layer": layer, "stage": stage}
                    and frame["coverage"] == ["value"],
                    "boundary identity changed",
                )
                archive_frame(path, boundary_output / name, reflink=reflink)
                projected_boundaries.append(entry)
    lineage = {
        "source_plan": source_plan["sha256"],
        "requested_plan": requested_plan["sha256"],
        "source_schedule": schedule["sha256"],
        "source_boundaries": boundaries["sha256"],
    }
    projected_schedule = seal(
        {
            **{k: v for k, v in schedule.items() if k != "sha256"},
            "plan": requested_plan["sha256"],
            "frames": projected_frames,
            "reference_projection": lineage,
        }
    )
    projected_boundary_schedule = seal(
        {
            **{k: v for k, v in boundaries.items() if k != "sha256"},
            "positions": requested_positions,
            "frames": projected_boundaries,
            "reference_projection": lineage,
        }
    )
    write_private(output / "schedule.json", projected_schedule)
    write_private(boundary_output / "boundaries.json", projected_boundary_schedule)
    write_private(output / "plan.json", requested_plan)
    result = seal(
        {
            "schema": "urn:qwen:serial-reference-projection:v1",
            **lineage,
            "schedule": projected_schedule["sha256"],
            "boundaries": projected_boundary_schedule["sha256"],
            "states": len(projected_frames),
            "observations": len(projected_boundaries),
            "storage": (
                "independent verified reflinks" if reflink else "independent verified copies"
            ),
            "native_equivalence": "UNPROVED",
        }
    )
    write_private(output / "reference-projection.json", result)
    return result
