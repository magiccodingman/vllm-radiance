"""Portable, offline contracts for inference diagnostics.

An equal finite trace is evidence about the declared observations, not a proof
of model equivalence. Backend adapters own tensor/state extraction; this module
owns identities, coverage and comparison. It has no GPU or inference imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "urn:qwen:portable-diagnostic:v1"
SEMANTIC_FIELDS = frozenset(
    {
        "weights",
        "weight_quantization",
        "activation_quantization",
        "attention",
        "kv_representation",
        "recurrence",
        "position_encoding",
        "tokenizer",
        "chat_template",
        "sampler",
        "numerical_contract",
    }
)
ARTIFACT_GROUPS = frozenset(
    {
        "source",
        "generated_code",
        "gpu_binary",
        "compiler",
        "runtime",
        "hardware",
        "model",
        "tokenizer_template",
        "configuration",
        "adapter",
        "reference",
    }
)
SHA = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,159}\Z")


class DiagnosticError(ValueError):
    """Evidence is malformed, incomplete or not comparable."""


def digest(value: Any) -> str:
    # Same canonical JSON convention as the existing assurance trace writer.
    data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def require_sha(value: object) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise DiagnosticError("missing or invalid content identity")
    return value


def require_name(value: object) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise DiagnosticError("invalid diagnostic identifier")
    return value


def integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DiagnosticError("invalid diagnostic count or position")
    return value


def semantic_identity(specification: Mapping[str, Any]) -> str:
    """Changing an accepted approximation creates a different reference model."""
    if set(specification) != SEMANTIC_FIELDS:
        raise DiagnosticError("reference semantics must explicitly describe every model component")
    if any(not isinstance(v, dict) or not v for v in specification.values()):
        raise DiagnosticError("reference components require explicit nonempty specifications")
    return digest(specification)


def seal(document: Mapping[str, Any]) -> dict[str, Any]:
    if "sha256" in document:
        raise DiagnosticError("document is already sealed")
    return {**document, "sha256": digest(document)}


def authenticate(document: Mapping[str, Any]) -> None:
    unsigned = {k: v for k, v in document.items() if k != "sha256"}
    if require_sha(document.get("sha256")) != digest(unsigned):
        raise DiagnosticError("diagnostic document changed after publication")


def private_json(path: Path) -> dict[str, Any]:
    """Read one stable owner-only file without following a final symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) & 0o077
        ):
            raise DiagnosticError("diagnostic input must be an owned private regular file")
        data = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise DiagnosticError("diagnostic input changed during reading")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise DiagnosticError("diagnostic input must be an object")
    return value


def write_private(path: Path, document: Mapping[str, Any]) -> None:
    """Create once; incomplete writes never replace previous evidence."""
    data = (json.dumps(document, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execution_manifest(
    semantics: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Path]],
    unavailable: Mapping[str, str],
) -> dict[str, Any]:
    """Hash build artifacts once; requests refer to the resulting manifest hash.

    This records files supplied by the operator. A loaded-process attestation is
    separate evidence; hashing a source checkout does not attest loaded kernels.
    """
    from qwen_r9700_lab.manifests import sha256_file

    if set(files) | set(unavailable) != ARTIFACT_GROUPS or set(files) & set(unavailable):
        raise DiagnosticError("every artifact group must be present or explicitly unavailable")
    if any(not isinstance(reason, str) or not reason.strip() for reason in unavailable.values()):
        raise DiagnosticError("unavailable artifacts need a reason")
    artifacts = []
    for group, members in sorted(files.items()):
        if not members:
            raise DiagnosticError("an empty artifact group does not establish coverage")
        for name, path in sorted(members.items()):
            require_name(name)
            path = Path(path)
            before = path.stat()
            if not stat.S_ISREG(before.st_mode):
                raise DiagnosticError("artifact is not a regular file")
            identity = sha256_file(path)
            after = path.stat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, key) != getattr(after, key) for key in fields):
                raise DiagnosticError("artifact changed during hashing")
            artifacts.append(
                {"group": group, "name": name, "sha256": identity, "bytes": after.st_size}
            )
    return seal(
        {
            "schema": SCHEMA + "/execution",
            "semantics_sha256": semantic_identity(semantics),
            "artifacts": artifacts,
            "unavailable": dict(unavailable),
            "artifact_inventory_complete": not unavailable,
            "loaded_process_attested": False,
            "formal_equivalence_proven": False,
        }
    )


def verify_source_bindings(root: Path, expected: Mapping[str, str]) -> None:
    """Authenticate all native bindings before an adapter may install any hook."""
    from qwen_r9700_lab.manifests import sha256_file

    if not expected:
        raise DiagnosticError("native adapter has no source bindings")
    root = root.resolve(strict=True)
    for relative, expected_hash in expected.items():
        path = (root / relative).resolve(strict=True)
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise DiagnosticError("native binding escapes its declared runtime")
        if sha256_file(path) != require_sha(expected_hash):
            raise DiagnosticError("native source binding changed; adapter needs requalification")


@dataclass(frozen=True, order=True)
class Boundary:
    layer: int
    name: str
    kind: str = "tensor"

    def __post_init__(self) -> None:
        integer(self.layer)
        require_name(self.name)
        if self.kind not in {"tensor", "logical_state"}:
            raise DiagnosticError("unknown boundary value kind")

    def document(self) -> dict[str, Any]:
        return {"layer": self.layer, "name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class Observation:
    position: int
    boundary: Boundary
    value_sha256: str
    nonfinite: int | None = None

    def document(self) -> dict[str, Any]:
        return {
            "position": integer(self.position),
            **self.boundary.document(),
            "value_sha256": require_sha(self.value_sha256),
            "nonfinite": None if self.nonfinite is None else integer(self.nonfinite),
        }


def trace(
    *,
    execution_sha256: str,
    semantics_sha256: str,
    adapter_sha256: str,
    input_sha256: str,
    mode: str,
    positions: list[int],
    boundaries: list[Boundary],
    observations: Iterable[Observation],
) -> dict[str, Any]:
    if mode not in {"forced_tokens", "prefill", "snapshot_roundtrip"}:
        raise DiagnosticError("free generation is not forced-token differential execution")
    if not positions or positions != sorted(set(positions)):
        raise DiagnosticError("capture positions must be nonempty, unique and ordered")
    for position in positions:
        integer(position)
    if not boundaries or len(set(boundaries)) != len(boundaries):
        raise DiagnosticError("capture boundaries must be nonempty and unique")
    rows = [observation.document() for observation in observations]
    expected = {(p, b.layer, b.name, b.kind) for p in positions for b in boundaries}
    keys = [(r["position"], r["layer"], r["name"], r["kind"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise DiagnosticError("capture has missing, duplicate or undeclared observations")
    order = {(b.layer, b.name, b.kind): i for i, b in enumerate(boundaries)}
    rows.sort(key=lambda r: (r["position"], order[r["layer"], r["name"], r["kind"]]))
    return seal(
        {
            "schema": SCHEMA + "/trace",
            "execution_sha256": require_sha(execution_sha256),
            "semantics_sha256": require_sha(semantics_sha256),
            "adapter_sha256": require_sha(adapter_sha256),
            "input_sha256": require_sha(input_sha256),
            "mode": mode,
            "positions": positions,
            "boundaries": [b.document() for b in boundaries],
            "observations": rows,
        }
    )


def validate_trace(value: Mapping[str, Any]) -> None:
    authenticate(value)
    if value.get("schema") != SCHEMA + "/trace":
        raise DiagnosticError("unsupported trace schema")
    rebuilt = trace(
        **{
            key: value[key]
            for key in (
                "execution_sha256",
                "semantics_sha256",
                "adapter_sha256",
                "input_sha256",
                "mode",
                "positions",
            )
        },
        boundaries=[Boundary(**b) for b in value["boundaries"]],
        observations=[
            Observation(
                r["position"],
                Boundary(r["layer"], r["name"], r["kind"]),
                r["value_sha256"],
                r["nonfinite"],
            )
            for r in value["observations"]
        ],
    )
    if rebuilt != value:
        raise DiagnosticError("trace contains unknown fields or noncanonical observations")


def compare_traces(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare declared boundaries, not backend-specific addresses or row slots."""
    for value in (left, right):
        validate_trace(value)
    for key in ("semantics_sha256", "input_sha256", "mode", "positions", "boundaries"):
        if left[key] != right[key]:
            raise DiagnosticError("traces differ in reference semantics, input or coverage")
    differences = []
    for a, b in zip(left["observations"], right["observations"], strict=True):
        if a["value_sha256"] != b["value_sha256"] or a["nonfinite"] or b["nonfinite"]:
            differences.append({key: a[key] for key in ("position", "layer", "name", "kind")})
    return seal(
        {
            "schema": SCHEMA + "/comparison",
            "left_sha256": left["sha256"],
            "right_sha256": right["sha256"],
            "observations": len(left["observations"]),
            "equal_observed_boundaries": not differences,
            "differing_boundaries": len(differences),
            "first_difference": differences[0] if differences else None,
            "claim": "finite_declared_boundary_comparison",
            "finite_values_verified": all(
                r["nonfinite"] == 0 for value in (left, right) for r in value["observations"]
            ),
            "numerical_difference_is_automatically_a_bug": False,
            "sampling_distribution_verified": False,
            "formal_equivalence_proven": False,
        }
    )


def logical_cache_state(
    allocations: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    free: list[int],
) -> list[dict[str, Any]]:
    """Validate each allocator locally, compare logical ownership across backends.

    Shared immutable prefix blocks and different physical block numbering are
    legal. Adapters must export the same logical granularity and include all
    references, including pins and other references not held by a sequence.
    """
    physical = {}
    for block in allocations:
        if set(block) != {"id", "refcount", "external_references"}:
            raise DiagnosticError("allocation descriptor is incomplete")
        key = integer(block["id"])
        if key in physical:
            raise DiagnosticError("duplicate physical allocation")
        physical[key] = block
    free_set = {integer(value) for value in free}
    if len(free_set) != len(free) or free_set & physical.keys():
        raise DiagnosticError("free and allocated blocks overlap or repeat")
    references: Counter[int] = Counter()
    logical = []
    identities = set()
    for row in mappings:
        if set(row) != {"sequence", "component", "layer", "start", "end", "block", "sha256"}:
            raise DiagnosticError("logical cache mapping is incomplete")
        key = integer(row["block"])
        if key not in physical:
            raise DiagnosticError("live sequence reaches an unallocated or free block")
        integer(row["layer"])
        start, end = integer(row["start"]), integer(row["end"])
        if end <= start:
            raise DiagnosticError("cache mapping has an empty or reversed token span")
        identity = (
            require_name(row["sequence"]),
            require_name(row["component"]),
            row["layer"],
            start,
            end,
        )
        if identity in identities:
            raise DiagnosticError("duplicate logical cache ownership")
        identities.add(identity)
        references[key] += 1
        logical.append(
            {
                **{k: row[k] for k in ("sequence", "component", "layer", "start", "end")},
                "sha256": require_sha(row["sha256"]),
            }
        )
    for key, block in physical.items():
        count = integer(block["refcount"], minimum=1)
        if count != references[key] + integer(block["external_references"]):
            raise DiagnosticError("allocator reference count differs from live references")
    ordered = sorted(logical, key=lambda r: (r["sequence"], r["component"], r["layer"], r["start"]))
    previous = {}
    for row in ordered:
        group = (row["sequence"], row["component"], row["layer"])
        if row["start"] < previous.get(group, 0):
            raise DiagnosticError("logical cache ownership overlaps within one sequence")
        previous[group] = row["end"]
    return ordered


def remaining_materialized(before, old_pending, emitted, new_pending):
    """Shared exact-integer accounting, also executed symbolically by Z3."""
    return before + old_pending + emitted - new_pending


def validate_speculative_commit(
    *,
    before_materialized: int,
    before_pending: int,
    drafted: int,
    accepted: int,
    emitted: int,
    after_materialized: int,
    after_pending: int,
    component_versions: Mapping[str, int],
    required_components: set[str],
) -> None:
    """Check one successful publication using explicit pending-token accounting.

    Each version is the materialized prefix length represented by that state,
    not an implementation's physical slot/generation number. Rejected suffix
    independence additionally requires comparing the actual state values against
    serial replay; these integer invariants alone do not establish it.
    """
    for value in (
        before_materialized,
        before_pending,
        drafted,
        accepted,
        emitted,
        after_materialized,
        after_pending,
    ):
        integer(value)
    if before_pending > 1 or after_pending > 1 or accepted > drafted or emitted != accepted + 1:
        raise DiagnosticError("invalid speculative acceptance or pending-token accounting")
    if after_materialized != remaining_materialized(
        before_materialized, before_pending, emitted, after_pending
    ):
        raise DiagnosticError("materialized and emitted token counts disagree")
    if not required_components or set(component_versions) != required_components:
        raise DiagnosticError("persistent state component coverage is incomplete")
    for name, version in component_versions.items():
        require_name(name)
        if integer(version) != after_materialized:
            raise DiagnosticError("persistent state version includes rejected or stale tokens")


class ForcedTokenAdapter(Protocol):
    """A native adapter must consume the supplied token; never its own argmax.

    reset/step return only after the observed state transition is complete.
    step's position is the token *consumed*, not the sampled pending bonus token.
    A backend requiring another materialization convention needs an explicit
    adapter, not an off-by-one exception in the common comparator.
    """

    execution_sha256: str
    semantics_sha256: str
    adapter_sha256: str

    def reset(self, prefix: tuple[int, ...]) -> None: ...
    def step(self, token: int, position: int) -> Iterable[Observation]: ...
    def close(self) -> None: ...


def forced_token_replay(
    adapter: ForcedTokenAdapter,
    prefix: tuple[int, ...],
    suffix: tuple[int, ...],
    boundaries: list[Boundary],
) -> dict[str, Any]:
    """Reusable driver; private tokens never appear in its evidence document."""
    if not prefix or not suffix:
        raise DiagnosticError("forced replay requires a prefix and a nonempty suffix")
    for token in (*prefix, *suffix):
        integer(token)
    observations = []
    try:
        adapter.reset(prefix)
        for position, token in enumerate(suffix, start=len(prefix)):
            rows = list(adapter.step(token, position))
            if any(row.position != position for row in rows):
                raise DiagnosticError("adapter returned state for a different consumed token")
            observations.extend(rows)
    finally:
        adapter.close()
    return trace(
        execution_sha256=adapter.execution_sha256,
        semantics_sha256=adapter.semantics_sha256,
        adapter_sha256=adapter.adapter_sha256,
        input_sha256=digest({"prefix": prefix, "forced_suffix": suffix}),
        mode="forced_tokens",
        positions=list(range(len(prefix), len(prefix) + len(suffix))),
        boundaries=boundaries,
        observations=observations,
    )
