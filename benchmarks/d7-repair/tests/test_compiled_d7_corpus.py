import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from qwen_r9700_lab.conformance_topk import aggregate, summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, seal, write_private


@pytest.fixture
def driver():
    path = Path(__file__).parents[1] / "benchmark_compiled_d7_corpus.py"
    spec = importlib.util.spec_from_file_location("compiled_corpus_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus(tmp_path, count=8):
    fixture = seal({"prefix": [1, 2, 3], "output": list(range(count + 1))})
    row = {
        "name": "continuation-000.json",
        "sha256": fixture["sha256"],
        "evaluate_positions": count,
        "prefix_tokens": 3,
        "prefix_sha256": digest(fixture["prefix"]),
        "output_sha256": digest(fixture["output"]),
    }
    manifest = seal({"positions": count, "continuations": [row]})
    write_private(tmp_path / row["name"], fixture)
    write_private(tmp_path / "manifest.json", manifest)
    return manifest, row


def evidence(row, width, changed=False):
    values = np.arange(64, dtype=np.float32)
    logits = summarize_logits(values)
    rows = [
        {"position": n, "absolute_position": n + 3, "target_rows": width, "logits": logits}
        for n in range(row["evaluate_positions"])
    ]
    if changed:
        # Leave the top-20 identical but change a lower score: full-vector checks must catch it.
        values[0] = 0.5
        rows[3]["logits"] = summarize_logits(values)
    return seal(
        {
            "schema": "urn:qwen:d7-equivalence-private-rows:v1",
            "continuation": row["sha256"],
            "prefill": logits,
            "rows": rows,
        }
    )


def test_frozen_corpus_and_exact_budget(driver, tmp_path):
    manifest, _ = corpus(tmp_path)
    assert driver.load_corpus(tmp_path, manifest["sha256"], 8) == manifest
    with pytest.raises(DiagnosticError, match="budget"):
        driver.load_corpus(tmp_path, manifest["sha256"], 16)


def test_changed_fixture_rejected(driver, tmp_path):
    manifest, _ = corpus(tmp_path)
    (tmp_path / "continuation-000.json").write_text(
        json.dumps(seal({"prefix": [4], "output": [5]}))
    )
    with pytest.raises(DiagnosticError, match="changed"):
        driver.load_corpus(tmp_path, manifest["sha256"], 8)


def test_partial_m8_group_rejected(driver, tmp_path):
    manifest, _ = corpus(tmp_path, 7)
    with pytest.raises(DiagnosticError, match="M8 group"):
        driver.load_corpus(tmp_path, manifest["sha256"], 7)


def test_complete_rows_and_full_logit_negative_control(driver, tmp_path):
    _, row = corpus(tmp_path)
    a, b = evidence(row, 1), evidence(row, 8, changed=True)
    pairs, prefill = driver.compare_pair(a, b, row)
    result = aggregate(pairs)
    assert prefill["full_logits_exact"]
    assert result["positions"] == 8 and result["full_logits_exact"] == 7
    for k in ("1", "10", "20"):
        assert result[k]["set_exact"] == result[k]["ranked_exact"] == 8


@pytest.mark.parametrize("failure", ("width", "position", "truncated", "fixture"))
def test_incomparable_evidence_rejected(driver, tmp_path, failure):
    _, row = corpus(tmp_path)
    a, b = evidence(row, 1), evidence(row, 8)
    b.pop("sha256")
    if failure == "width":
        b["rows"][0]["target_rows"] = 1
    elif failure == "position":
        b["rows"][0]["absolute_position"] += 1
    elif failure == "truncated":
        b["rows"].pop()
    else:
        b["continuation"] = "different"
    with pytest.raises(DiagnosticError):
        driver.compare_pair(a, seal(b), row)
