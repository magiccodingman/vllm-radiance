"""Compare saved eager RoPE intervention with eager and precision-cast controls."""

import argparse
import json
from pathlib import Path

from compare_execution_modes_d7 import load

from qwen_r9700_lab.conformance_rotary_intervention import (
    compare_common_rounding,
    compare_rotary_intervention,
)
from qwen_r9700_lab.diagnostic_contract import private_json, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference",
        "candidate",
        "compiled",
        "reference-rows",
        "candidate-rows",
        "compiled-rows",
        "binding",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    reference, candidate, compiled = [
        load(x) for x in (args.reference, args.candidate, args.compiled)
    ]
    ref_rows, candidate_rows, compiled_rows = [
        private_json(x) for x in (args.reference_rows, args.candidate_rows, args.compiled_rows)
    ]
    binding = private_json(args.binding)
    args.output.mkdir(mode=0o700)
    a = compare_rotary_intervention(reference, candidate, ref_rows, candidate_rows, binding)
    b = compare_common_rounding(
        reference, candidate, compiled, candidate_rows, compiled_rows, binding
    )
    for name, report in (("eager-change", a), ("common-rounding", b)):
        write_private(args.output / (name + ".json"), report)
        print(
            json.dumps({"comparison": name, "sha256": report["sha256"], "decode": report["decode"]})
        )


if __name__ == "__main__":
    main()
