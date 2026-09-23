"""Pinned stock-M1 arithmetic identity for future GDN repair comparisons.

This CPU-only audit checks retained sources and compiler artifacts. It does not
load a GPU library, attest a dispatched kernel, change a replay plan or qualify a
candidate. Existing NumPy and R4D references keep their own semantic identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    integer,
    seal,
    write_private,
)

CONTRACT_ID = "stock-packed-gdn-m1-gfx1201-v1"
CONTRACT_SHA256 = "d8149b00210c94acfca4f1f66030de302dcaf8155894de6a23d3f4b76da61e00"


def read_contract(path: Path) -> dict:
    result = json.loads(path.read_text())
    authenticate(result)
    if result.get("id") != CONTRACT_ID or result["sha256"] != CONTRACT_SHA256:
        raise DiagnosticError("unreviewed GDN arithmetic contract; use a distinct revision")
    return result


def _check_file(path: Path, expected: str) -> dict:
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise DiagnosticError(f"GDN arithmetic evidence changed: {path.name}")
    return {"sha256": actual, "bytes": path.stat().st_size}


def audit_artifacts(contract: dict, compiled_root: Path, sources: dict[str, Path]) -> dict:
    authenticate(contract)
    if contract.get("id") != CONTRACT_ID or contract["sha256"] != CONTRACT_SHA256:
        raise DiagnosticError("unreviewed GDN arithmetic contract; use a distinct revision")
    authority = contract["arithmetic_authority"]
    if set(sources) != set(authority["source"]):
        raise DiagnosticError("incomplete GDN source evidence")
    checked_sources = {
        name: _check_file(sources[name], expected) for name, expected in authority["source"].items()
    }
    checked_artifacts = {
        name: _check_file(compiled_root / name, expected)
        for name, expected in authority["files"].items()
    }
    # The full metadata file is already hash-bound; these explicit comparisons
    # also keep the human-readable contract consistent with the retained build.
    metadata = json.loads((compiled_root / (authority["kernel"] + ".json")).read_text())
    if metadata["target"] != authority["target"] or any(
        metadata.get(key) != value for key, value in authority["compiler"].items()
    ):
        raise DiagnosticError("GDN compiler metadata contradicts the declared arithmetic")
    return seal(
        {
            "schema": "urn:qwen:gdn-arithmetic-artifact-audit:v1",
            "contract": contract["sha256"],
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sources": checked_sources,
            "artifacts": checked_artifacts,
            "artifact_identity": "TESTED",
            "runtime_dispatch": "UNPROVED",
            "candidate_equivalence": "UNPROVED",
            "full_model_equivalence": "UNPROVED",
            "gpu_used": False,
            "installed_backend_changed": False,
            "existing_matrix_contract_changed": False,
        }
    )


def committed_gdn_row(accepted_proposals: int) -> int:
    """Index into eight after-row states, with row zero consuming pending input.

    The correction/bonus token emitted by this round is still pending. This is
    an index into *after-row* states, not a count or an initial-inclusive frame.
    """
    integer(accepted_proposals)
    if accepted_proposals > 7:
        raise DiagnosticError("D7 admits zero through seven accepted proposals")
    return accepted_proposals


def required_processed_rows(accepted_proposals: int) -> tuple[int, ...]:
    return tuple(range(committed_gdn_row(accepted_proposals) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--stock-source", type=Path, required=True)
    parser.add_argument("--op-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_artifacts(
        read_contract(args.contract),
        args.compiled_root,
        {"fused_recurrent.py": args.stock_source, "fla_op.py": args.op_source},
    )
    write_private(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
