"""CPU-only comparison of two controlled 320-position D7 execution-mode runs."""

import argparse
import json
from pathlib import Path

from qwen_r9700_lab.conformance_execution_modes import compare_pair
from qwen_r9700_lab.diagnostic_contract import private_json, write_private


def load(root):
    lane = root / "fixed-bf16"
    return {
        "measurement": private_json(root / "measurement.json"),
        "config": private_json(lane / "requested-config.json"),
        "runtime": private_json(lane / "actual-runtime.json"),
        "pass": private_json(lane / "pass-00.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "left-rows", "right-rows", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("m1", "m8"), default="m8")
    args = parser.parse_args()
    result = compare_pair(
        load(args.left),
        load(args.right),
        private_json(args.left_rows),
        private_json(args.right_rows),
        arm=args.arm,
    )
    write_private(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "decode", "prefill", "stage_localization")}))


if __name__ == "__main__":
    main()
