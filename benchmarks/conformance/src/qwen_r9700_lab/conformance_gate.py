"""Experimental exact output/state publication gate.

This is an in-process authority for isolated conformance runs, not a production
Pi proxy. Native workers must be independently isolated and supply complete
canonical state bytes. Their isolation, reference arithmetic and serialization
are separate obligations; this gate cannot infer them from matching hashes.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    digest,
    integer,
    require_name,
    require_sha,
    seal,
)


def publication_allowed(output_equal, state_equal, identity_equal, complete):
    """Pure Boolean predicate also executed symbolically by the proof runner."""
    return output_equal & state_equal & identity_equal & complete


def copy_accepted_prefix(current: tuple, proposed: tuple, count: int) -> tuple:
    """Small actual reference helper; never keep the rejected proposed suffix."""
    if type(count) is not int or len(current) != len(proposed) or not 0 <= count <= len(current):
        raise DiagnosticError("invalid prefix-copy domain")
    return proposed[:count] + current[count:]


@dataclass(frozen=True)
class TentativeTransition:
    base_revision: int
    semantics_sha256: str
    input_sha256: str
    consumed_tokens: int
    pending_token: bytes | None
    state: Mapping[str, bytes]
    output_events: tuple[bytes, ...]
    stop_reason: str | None

    def __post_init__(self):
        integer(self.base_revision)
        integer(self.consumed_tokens)
        require_sha(self.semantics_sha256)
        require_sha(self.input_sha256)
        if self.pending_token is not None and type(self.pending_token) is not bytes:
            raise DiagnosticError("pending token must be immutable canonical bytes")
        if self.stop_reason not in {None, "eos", "tool_call", "complete"}:
            raise DiagnosticError("errors and truncation are not successful stop events")
        components = {}
        for name, payload in self.state.items():
            require_name(name)
            if type(payload) is not bytes:
                raise DiagnosticError("state components must be immutable canonical bytes")
            components[name] = payload
        if type(self.output_events) is not tuple or any(
            type(v) is not bytes for v in self.output_events
        ):
            raise DiagnosticError("output events must be immutable canonical bytes")
        object.__setattr__(self, "state", MappingProxyType(components))


class ConformanceMismatchError(DiagnosticError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("candidate did not match the reference; no transition published")
        self.report = report


class CheckedAuthority:
    """Commit state/events together after exact comparison and revision checking.

    No callback receives candidate events while they are tentative. Publication
    is the return value after the authority's revision is advanced. Durability,
    an external Pi delivery channel and worker isolation are not implemented here.
    """

    def __init__(self, semantics_sha256: str, required_components: set[str], *, sampler="greedy"):
        self.semantics_sha256 = require_sha(semantics_sha256)
        if sampler != "greedy":
            raise DiagnosticError("sampled execution needs a separate distribution contract")
        if not required_components:
            raise DiagnosticError("state gate cannot admit empty component coverage")
        self.required_components = frozenset(require_name(v) for v in required_components)
        self._lock = threading.Lock()
        self._revision = 0
        self._transition = None

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def compare_and_commit(
        self,
        reference: TentativeTransition,
        candidate: TentativeTransition,
    ) -> tuple[bytes, ...]:
        with self._lock:
            identity_equal = bool(
                reference.base_revision == candidate.base_revision == self._revision
                and reference.semantics_sha256
                == candidate.semantics_sha256
                == self.semantics_sha256
                and reference.input_sha256 == candidate.input_sha256
            )
            complete = bool(
                set(reference.state) == set(candidate.state) == self.required_components
            )
            # Compare actual bytes, not digests. A hash collision cannot admit
            # an incorrect transition through this gate.
            output_equal = bool(
                reference.output_events == candidate.output_events
                and reference.stop_reason == candidate.stop_reason
            )
            state_equal = bool(
                reference.state == candidate.state
                and reference.consumed_tokens == candidate.consumed_tokens
                and reference.pending_token == candidate.pending_token
            )
            if not publication_allowed(output_equal, state_equal, identity_equal, complete):
                differing = [
                    name
                    for name in sorted(self.required_components)
                    if reference.state.get(name) != candidate.state.get(name)
                ]
                report = seal(
                    {
                        "schema": "urn:qwen:conformance-gate-mismatch:v1",
                        "revision": self._revision,
                        "identity_equal": identity_equal,
                        "complete_state_coverage": complete,
                        "output_equal": output_equal,
                        "state_equal": state_equal,
                        "first_different_component": differing[0] if differing else None,
                        "reference_input_sha256": reference.input_sha256,
                        "candidate_input_sha256": candidate.input_sha256,
                        "published": False,
                        "scope": "isolated_in_process_authority",
                    }
                )
                raise ConformanceMismatchError(report)
            # Retain the trusted reference state; the candidate cannot substitute
            # a later mutable buffer after its bytes have been checked.
            self._transition = reference
            self._revision += 1
            return candidate.output_events

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            return {
                "revision": self._revision,
                "semantics_sha256": self.semantics_sha256,
                "components_sha256": digest(sorted(self.required_components)),
                "publication_gate": "TESTED",
                "native_reference_adapter": "UNPROVED",
                "native_state_completeness": "UNPROVED",
                "worker_isolation": "ASSUMED",
                "python_runtime_and_hardware": "ASSUMED",
                "production_pi_integration": "UNPROVED",
                "formal_backend_equivalence": "UNPROVED",
            }
