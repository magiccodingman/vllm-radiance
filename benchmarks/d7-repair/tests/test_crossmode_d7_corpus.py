"""CPU negative controls for the identities behind the report's cross-mode table."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from qwen_r9700_lab.conformance_topk import aggregate, summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, seal, write_private

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "benchmark_crossmode_d7_corpus", SOURCE / "benchmark_crossmode_d7_corpus.py"
)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
audit_spec = importlib.util.spec_from_file_location(
    "audit_crossmode_d7_corpus_test", SOURCE / "audit_crossmode_d7_corpus.py"
)
auditor = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(auditor)


def runtime_pair(arm="m8", revision="final"):
    eager, final = arm == "m1", revision == "final"
    measurement = {"repair_manifest": "repair", "performance_manifest": "performance"}
    config = {"enforce_eager": eager}
    if not eager:
        config["speculative_config"] = {"num_speculative_tokens": 7}
    runtime = {
        "enforce_eager": eager,
        "compilation_mode": 0 if eager else 3,
        "graph_mode": "NONE" if eager else "PIECEWISE",
        "capture_sizes": [] if eager else [1, 2, 4, 8],
        "repair": {"bundle": "repair"} if final else None,
        "performance": {"manifest": "performance"} if final else None,
        "runtime": seal(
            {
                "compiler_settings": {"emulate_precision_casts": final},
                "flags": {
                    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": str(int(final)),
                    "RADIANCE_VERIFY_HEAD": "0",
                },
                "packages": {},
                "kernel": "fixture-kernel",
            }
        ),
        "diagnostic_sources": {"fixture.py": "pinned"},
        "effective_capacity": {
            "block_size": 1568 if eager else 1648,
            "num_gpu_blocks": 194 if eager else 370,
            "max_model_len": 253792,
        },
    }
    if final and eager:
        runtime["rotary_intervention"] = {"identity": seal({"installed_before_load": True})}
    return config, runtime, measurement


def test_audit_admits_only_declared_revision_and_mode_differences():
    comparisons = []
    for revision in ("before", "final"):
        for arm in ("m1", "m8"):
            config, runtime, measurement = runtime_pair(arm, revision)
            comparisons.append(
                auditor.validate_mode(
                    config, runtime, arm=arm, revision=revision, measurement=measurement
                )
            )
    assert all(item == comparisons[0] for item in comparisons)


@pytest.mark.parametrize(
    "fault", ["eager", "no_graph", "no_fix1", "no_fix2", "wrong_head", "width", "no_rotary"]
)
def test_audit_rejects_relabelled_modes_or_missing_repairs(fault):
    arm = "m1" if fault == "no_rotary" else "m8"
    config, runtime, measurement = runtime_pair(arm)
    if fault == "eager":
        runtime["enforce_eager"] = True
    elif fault == "no_graph":
        runtime["graph_mode"] = "NONE"
    elif fault == "no_fix1":
        runtime["repair"] = None
    elif fault == "no_fix2":
        rt = runtime["runtime"]
        rt["compiler_settings"]["emulate_precision_casts"] = False
        runtime["runtime"] = seal({k: v for k, v in rt.items() if k != "sha256"})
    elif fault == "wrong_head":
        rt = runtime["runtime"]
        rt["flags"]["RADIANCE_VERIFY_HEAD"] = "1"
        runtime["runtime"] = seal({k: v for k, v in rt.items() if k != "sha256"})
    elif fault == "width":
        config["speculative_config"]["num_speculative_tokens"] = 3
    else:
        runtime.pop("rotary_intervention")
    with pytest.raises(DiagnosticError):
        auditor.validate_mode(config, runtime, arm=arm, revision="final", measurement=measurement)


def fixture(root):
    prefix, output = list(range(24)), list(range(9))
    data = seal({"prefix": prefix, "output": output})
    name = "continuation-000.json"
    item = {
        "name": name,
        "sha256": data["sha256"],
        "prefix_tokens": len(prefix),
        "evaluate_positions": 8,
        "prefix_sha256": digest(prefix),
        "output_sha256": digest(output),
    }
    manifest = seal({"positions": 8, "continuations": [item]})
    write_private(root / name, data)
    write_private(root / "manifest.json", manifest)
    return manifest, item


def rows(item, width, values):
    logits = summarize_logits(np.asarray(values, dtype=np.float32))
    return seal(
        {
            "schema": "urn:qwen:d7-equivalence-private-rows:v1",
            "continuation": item["sha256"],
            "prefill": logits,
            "rows": [
                {
                    "position": i,
                    "absolute_position": item["prefix_tokens"] + i,
                    "target_rows": width,
                    "logits": logits,
                }
                for i in range(item["evaluate_positions"])
            ],
        }
    )


def test_pair_distinguishes_rank_order_from_membership(tmp_path):
    manifest, item = fixture(tmp_path)
    assert driver.load_corpus(tmp_path, manifest["sha256"], 8) == manifest
    values = np.arange(64, dtype=np.float32)
    changed = values.copy()
    changed[60], changed[61] = changed[61], changed[60]
    compared, initial = driver.compare_pair(rows(item, 1, values), rows(item, 8, changed), item)
    metrics = aggregate(compared)
    assert metrics["positions"] == 8
    assert metrics["1"]["ranked_exact"] == 8
    assert metrics["20"]["set_exact"] == 8
    assert metrics["20"]["ranked_exact"] == 0
    assert metrics["20"]["mean_overlap_tokens"] == 20
    assert not initial["full_logits_exact"]


@pytest.mark.parametrize(
    "fault", ["width", "position", "shift_both", "history", "missing", "unauthenticated"]
)
def test_pair_rejects_evidence_from_the_wrong_comparison(tmp_path, fault):
    _, item = fixture(tmp_path)
    reference, candidate = rows(item, 1, np.arange(64)), rows(item, 8, np.arange(64))
    reference, candidate = copy.deepcopy(reference), copy.deepcopy(candidate)
    if fault == "width":
        candidate["rows"][0]["target_rows"] = 1
    elif fault == "position":
        candidate["rows"][0]["position"] = 1
    elif fault == "shift_both":
        reference["rows"][0]["absolute_position"] += 1
        candidate["rows"][0]["absolute_position"] += 1
    elif fault == "history":
        candidate["continuation"] = "f" * 64
    elif fault == "missing":
        candidate["rows"].pop()
    else:
        candidate["sha256"] = "f" * 64
    if fault != "unauthenticated":
        reference = seal({k: v for k, v in reference.items() if k != "sha256"})
        candidate = seal({k: v for k, v in candidate.items() if k != "sha256"})
    with pytest.raises(DiagnosticError):
        driver.compare_pair(reference, candidate, item)


def test_corpus_rejects_missing_and_changed_fixture(tmp_path):
    manifest, item = fixture(tmp_path)
    with pytest.raises(DiagnosticError):
        driver.load_corpus(tmp_path, manifest["sha256"], 16)
    path = tmp_path / item["name"]
    path.write_text("{}\n")
    with pytest.raises(DiagnosticError):
        driver.load_corpus(tmp_path, manifest["sha256"], 8)
