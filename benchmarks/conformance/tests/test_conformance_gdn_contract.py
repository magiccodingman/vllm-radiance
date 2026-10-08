from copy import deepcopy
from pathlib import Path

import pytest

from qwen_r9700_lab import conformance_gdn_contract as contract
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/profiles/gdn-stock-m1-arithmetic-v1.json"


def test_reference_choice_cannot_be_silently_relabelled(tmp_path):
    import json

    original = contract.read_contract(PATH)
    altered = deepcopy(original)
    altered.pop("sha256")
    altered["precision"]["beta"] = "FP32 with no BF16 rounding"
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(seal(altered)))
    with pytest.raises(DiagnosticError, match="unreviewed"):
        contract.read_contract(changed)


def test_a_resealed_contract_does_not_admit_a_different_compiler(tmp_path):
    original = contract.read_contract(PATH)
    altered = deepcopy(original)
    altered.pop("sha256")
    altered["arithmetic_authority"]["compiler"]["enable_fp_fusion"] = False
    with pytest.raises(DiagnosticError, match="unreviewed"):
        contract.audit_artifacts(seal(altered), tmp_path, {})


def test_evidence_audit_requires_both_the_kernel_and_its_math_helpers(tmp_path):
    with pytest.raises(DiagnosticError, match="incomplete"):
        contract.audit_artifacts(contract.read_contract(PATH), tmp_path, {})


def test_edited_source_cannot_pass_as_the_pinned_stock_reference(tmp_path):
    source = tmp_path / "fused_recurrent.py"
    source.write_text("# modified beta or normalization arithmetic\n")
    with pytest.raises(DiagnosticError, match="evidence changed"):
        contract.audit_artifacts(
            contract.read_contract(PATH),
            tmp_path,
            {"fused_recurrent.py": source, "fla_op.py": tmp_path / "not-read.py"},
        )


@pytest.mark.parametrize("accepted", range(8))
def test_acceptance_counts_processed_pending_input_but_not_new_bonus(accepted):
    rows = ("old_pending", *(f"proposal_{i}" for i in range(1, 8)))
    processed = tuple(rows[i] for i in contract.required_processed_rows(accepted))
    assert processed == rows[: accepted + 1]
    assert rows[contract.committed_gdn_row(accepted)] == processed[-1]
    assert len(processed) == accepted + 1


@pytest.mark.parametrize("accepted", [-1, 8, True, 1.0, "1", None])
def test_invalid_rejection_widths_fail_closed(accepted):
    with pytest.raises(DiagnosticError):
        contract.required_processed_rows(accepted)


def test_different_rounding_and_contraction_are_observable():
    import numpy as np

    from qwen_r9700_lab.conformance_reference import bf16

    halfway = np.array([0x3F808000], dtype=np.uint32)
    assert bf16(halfway.view(np.float32)).view(np.uint32)[0] == 0x3F800000
    assert ((halfway + np.uint32(0x8000)) & np.uint32(0xFFFF0000))[0] == 0x3F810000
    # These particular operands have an exact FP64 product and sum, making this
    # a valid finite counterexample, not a general FP64 emulation of FP32 FMA.
    a, b = np.float32(1 + 2**-23), np.float32(1 - 2**-23)
    separate = np.float32(np.float32(a * b) - np.float32(1))
    fused_for_these_operands = np.float32(np.float64(a) * np.float64(b) - 1)
    assert separate == 0
    assert fused_for_these_operands == -(2**-46)
