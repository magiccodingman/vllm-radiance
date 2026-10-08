"""Align observed eager/compiled boundaries after both declared rounding interventions."""

import argparse
import json
from pathlib import Path

from compare_execution_modes_d7 import load
from compare_mode_boundaries_d7 import run

from qwen_r9700_lab.conformance_rotary_intervention import admit_common_rounding
from qwen_r9700_lab.diagnostic_contract import private_json, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-run",
        "left-run",
        "right-run",
        "left",
        "right",
        "left-bridge",
        "right-bridge",
        "binding",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    reference = load(args.reference_run)
    binding = private_json(args.binding)
    args.output.mkdir(mode=0o700)

    def admit(left, right):
        return admit_common_rounding(reference, left, right, binding)

    admission = admit(load(args.left_run), load(args.right_run))
    write_private(args.output / "admission.json", admission)
    for phase in ("prefill", "decode"):
        result = run(
            args.left,
            args.right,
            phase,
            [args.left_run, args.right_run],
            [private_json(args.left_bridge), private_json(args.right_bridge)],
            admit=admit,
        )
        write_private(args.output / (phase + ".json"), result)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "sha256": result["sha256"],
                    "boundaries": len(result["boundaries"]),
                    "first_difference": result["first_observed_different_boundary"],
                }
            )
        )


if __name__ == "__main__":
    main()
