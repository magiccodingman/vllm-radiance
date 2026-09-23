"""Explicit proof obligations and implementation coverage, not a certificate.

Each row names an actual instrumentation/checking path and its remaining gap.
An observed execution or a helper theorem cannot discharge a backend theorem.
The inventory is scoped to this text-only, single-GPU Radiance deployment.
"""

from pathlib import Path

from qwen_r9700_lab.conformance_artifacts import file_identity
from qwen_r9700_lab.conformance_reference import reference_contract
from qwen_r9700_lab.diagnostic_contract import seal

# id, obligation, available implementation, regression suite, unresolved link.
COMPONENTS = (
    (
        "reference",
        "R_C implements the declared finite-precision model on every admitted input",
        "conformance_reference.py",
        "test_conformance_reference.py",
        (
            "NumPy transcendental/rounding behavior and the stock model "
            "implementation are not proved equivalent"
        ),
    ),
    (
        "prompt",
        "Tokens_B(u) = Tokens_R(u), including positions, masks and template",
        "conformance_replay.py",
        "test_conformance_replay.py",
        (
            "Replay begins with explicit token IDs; live tokenizer, template and "
            "request rendering are outside it"
        ),
    ),
    (
        "weights",
        "Decode_B(Q(W), i) = Decode_R(Q(W), i) for every admitted packed code/index",
        "conformance_model.py",
        "test_conformance_reference.py",
        (
            "Nibble helpers are checked; native permutation, E8M0 scale application "
            "and all native shapes need comparison"
        ),
    ),
    (
        "activations",
        "Quant_B(x) = Quant_C(x); no undeclared activation quantizer",
        "conformance_reference.py",
        "test_conformance_profiles.py",
        (
            "FP8 activations intentionally change the weight-only reference; actual "
            "dispatch needs validation"
        ),
    ),
    (
        "normalization",
        "Norm_B(x) = Norm_R(x), including reductions and rounding points",
        "conformance_boundaries.py",
        "test_conformance_boundaries.py",
        (
            "Common native module outputs are captured; fused internal operations are"
            " not independently qualified"
        ),
    ),
    (
        "linear",
        "Linear_B(x,Q(W)) = Linear_R(x,Q(W))",
        "conformance_instrumentation.py",
        "test_conformance_instrumentation.py",
        (
            "Python calls and tensors are observable; HIP/ISA arithmetic and all "
            "dispatch regimes remain unproved"
        ),
    ),
    (
        "rope",
        "RoPE_B(x,p) = RoPE_R(x,p) for every admitted position",
        "conformance_reference.py",
        "test_conformance_reference.py",
        "Native rotation/trigonometric implementations need same-input operator replay",
    ),
    (
        "attention",
        "Attention_B(q,K,V,mask) = Attention_R(q,K,V,mask)",
        "conformance_radiance.py",
        "test_conformance_radiance.py",
        (
            "Native R4D entrypoints can be traced; complete causality, selector, "
            "reduction and scratch-lifetime proofs are missing"
        ),
    ),
    (
        "kv",
        "alpha(KV_B)[p] = KV_R[p], including declared dtype/scales",
        "conformance_state.py",
        "test_conformance_profiles.py",
        (
            "Byte comparison rejects missing values; BF16 native extraction and GPU "
            "write completion remain unvalidated"
        ),
    ),
    (
        "convolution",
        "(y,H')_B = (y,H')_R; rejected suffix cannot change H'",
        "conformance_radiance.py",
        "test_conformance_radiance.py",
        (
            "Selected history offsets are checked; the real temporal bank mapping and"
            " convolution kernel need native replay"
        ),
    ),
    (
        "gdn",
        "(y,S')_B = (y,S')_R, including every committed recurrent-state element",
        "conformance_model.py",
        "test_conformance_replay.py",
        (
            "Independent recurrence and glue captures exist; native "
            "prefill/verify/rollback equivalence is not proved"
        ),
    ),
    (
        "prefill",
        "alpha(Prefill_B(t_0..t_n)) = Fold(R_C, t_0..t_n)",
        "conformance_replay.py",
        "test_conformance_boundaries.py",
        (
            "Every materialized token can be observed; native intermediate tensors, "
            "all chunk sizes and actual schedules need qualification"
        ),
    ),
    (
        "head",
        "argmax_full(z_B) = argmax_full(z_R); excluded candidates cannot win",
        "conformance_invariants.py",
        "test_conformance_control.py",
        (
            "Interval certificate assumes sound bounds; INT2 shortlist exact "
            "reranking alone is insufficient"
        ),
    ),
    (
        "speculation",
        "Target_B(P,d_<j) = Target_R(P,d_<j); future d_>=j has no effect",
        "conformance_radiance.py",
        "test_conformance_replay.py",
        (
            "Forced D7 widths check selected state; native acceptance algorithm, "
            "proposer and all dynamic widths need separate evidence"
        ),
    ),
    (
        "commit",
        "alpha(Commit_B(P,d,k)) = R_C on exactly the processed accepted prefix",
        "conformance_gate.py",
        "test_conformance_proofs.py",
        (
            "Eight-slot copy/pending-token helpers have scoped proofs; these are not "
            "proofs of Radiance's implementation"
        ),
    ),
    (
        "ownership",
        "No writable alias of shared state; unrelated logical sessions are unchanged",
        "conformance_control.py",
        "test_conformance_control.py",
        (
            "Control trace invariants are checked; producer mapping, device races and"
            " physical ownership need native validation"
        ),
    ),
    (
        "scheduler",
        "Interleavings preserve each session's trace and state; stale epochs cannot publish",
        "conformance_control.py",
        "test_radiance_priority.py",
        (
            "CPU scheduling cases exist; production concurrency and priority "
            "preemption have no native conformance proof"
        ),
    ),
    (
        "snapshot",
        "Load_C(Save_C(S)) = S and every continued suffix has equal observables",
        "conformance_lifecycle.py",
        "test_conformance_lifecycle.py",
        (
            "Real storage transport plus diagnostic frames is tested; the actual "
            "native snapshot mapping and failure schedules are unqualified"
        ),
    ),
    (
        "sampling",
        "Greedy_B(z)=Greedy_R(z); sampled modes need identical conditional distributions",
        "conformance_invariants.py",
        "test_conformance_control.py",
        (
            "Replay forces tokens; it does not certify production penalties, RNG, "
            "rejection sampling, grammar or stopping"
        ),
    ),
    (
        "protocol",
        "Tokens, reasoning/content, tool events, finish status and pending state match R_C",
        "conformance_session.py",
        "test_conformance_session.py",
        (
            "Prototype checks tokens/stops; live tool parser, streaming transport and"
            " Pi publication are outside the gate"
        ),
    ),
    (
        "gate",
        (
            "Publish => equal output and next state, complete coverage, completed "
            "writes and current epoch"
        ),
        "conformance_invariants.py",
        "test_conformance_session.py",
        (
            "Prototype gate is tested and helper expressions proved; the production "
            "backend is not connected to it"
        ),
    ),
    (
        "dispatch",
        "Every executed implementation satisfies C and its declared preconditions",
        "conformance_dispatch.py",
        "test_conformance_dispatch.py",
        (
            "Reviewed library exports/explicit aliases can be recorded; hidden C++ "
            "launches and graph replay are outside this observer"
        ),
    ),
    (
        "graphs",
        (
            "Graph replay refines the serial transition with no stale pointers, races"
            " or lifetime violations"
        ),
        "conformance_instrumentation.py",
        "test_conformance_instrumentation.py",
        (
            "Tensor observation is serialized/eager; it can hide races in the "
            "production asynchronous schedule"
        ),
    ),
    (
        "compiler",
        "Compiled(K) refines the proved source under declared ISA semantics",
        "conformance_artifacts.py",
        "test_conformance_artifacts.py",
        (
            "Mapped library hashes are not actual HSACO dispatch identity or "
            "compiler/ISA translation validation"
        ),
    ),
    (
        "machine",
        "Execution follows the declared memory, arithmetic and isolation model",
        "conformance_artifacts.py",
        "test_conformance_artifacts.py",
        (
            "GPU hardware, driver, firmware, host runtime, reference isolation and "
            "fault-free execution remain assumptions"
        ),
    ),
)


def proof_obligations(profile="weight-only-bf16"):
    package = Path(__file__).parent
    root = package.parent.parent
    rows = []
    for identifier, formula, implementation, regression, gap in COMPONENTS:
        paths = (package / implementation, root / "tests" / regression)
        rows.append(
            {
                "id": identifier,
                "formula": formula,
                "backend_status": "UNPROVED",
                "available_checks": {
                    str(path.relative_to(root)): file_identity(path)
                    if path.is_file()
                    else "UNAVAILABLE"
                    for path in paths
                },
                "remaining_obligation": gap,
            }
        )
    return seal(
        {
            "schema": "urn:qwen:backend-proof-obligations:v1",
            "reference": reference_contract(profile),
            "domain": (
                "declared text-only dense Qwen3.5 architecture, one GPU, greedy "
                "reference; admitted finite inputs and reachable states"
            ),
            "logical_state": [
                "ordered KV and quantizers",
                "GDN state",
                "convolution history",
                "consumed positions",
                "emitted but pending token",
                "ownership and versions",
                "sampler/stop/parser state where admitted",
            ],
            "theorem": (
                "forall (S,P,u) in D: alpha(P)=S => alpha(B_C(P,u).state)=R_C(S,u).state "
                "AND Obs(B_C(P,u))=Obs(R_C(S,u))"
            ),
            "composition": {
                "initialization": "Init_B and Init_R satisfy alpha(P_0)=S_0",
                "step": "Every admitted transition preserves alpha and exactly matches observables",
                "frame": "A transition cannot modify another session or a shared immutable prefix",
                "closure": "Every result remains within the next transition's proved preconditions",
                "induction": (
                    "Initialization + step + frame + closure imply equal finite published traces"
                ),
                "availability": (
                    "Equal published prefixes alone do not prove termination or service "
                    "availability"
                ),
            },
            "obligations": rows,
            "undischarged": [row["id"] for row in rows],
            "universal_equivalence": "UNPROVED",
            "stock_quantized_equivalence": "UNPROVED",
            "instrumentation_complete": False,
            "production_gate_installed": False,
            "evidence_rule": (
                "TESTED and RUNTIME-CHECKED are scoped observations, never universal "
                "proof; helper proofs do not discharge native obligations"
            ),
            "gpu_used": False,
        }
    )
