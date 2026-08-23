#!/usr/bin/env python3
"""Client-only entry point for vLLM's online serving benchmark.

The installed top-level ``vllm`` CLI initializes server platform defaults even
for client benchmarks.  On this host that fails before argument parsing.  This
small wrapper invokes the upstream serving benchmark parser directly and does
not change benchmark behavior.
"""

import argparse

from vllm.benchmarks.serve import add_cli_args, main


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible online serving endpoint"
    )
    add_cli_args(parser)
    main(parser.parse_args())


if __name__ == "__main__":
    run()
