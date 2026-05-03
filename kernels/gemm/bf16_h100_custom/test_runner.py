#!/usr/bin/env python3
"""
Thin wrapper around harness.py for backward compatibility.

All correctness testing logic has been consolidated into harness.py. This file
delegates so the Makefile `make test` target continues to work unchanged.

Usage (identical to before):
    python3 test_runner.py                     # run all tests, print summary
    python3 test_runner.py --report REPORT.md  # generate Markdown report file
    python3 test_runner.py gelu linear fused full  # run specific tests
"""
import argparse
import sys
import torch

from tk_bench import input_group_count
import harness

DEFAULT_M = 4096
DEFAULT_K = 4096
DEFAULT_N = 4096

FEATURES = dict(harness.Registry.list_correctness())


def main():
    parser = argparse.ArgumentParser(description="Correctness test runner")
    parser.add_argument(
        "features", nargs="*",
        default=[],
        help=f"Tests to run: {list(FEATURES.keys()) + ['all']}. Default: all.",
    )
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=str, default=None,
                        help="Write Markdown report to file")
    args = parser.parse_args()

    raw_features = args.features if args.features else ["all"]
    if "all" in raw_features:
        target = None
    else:
        target = raw_features

    M, K, N = args.M, args.K, args.N
    seed = args.seed

    groups = [harness.CUDAWorkspace(M, K, N).create()
              for _ in range(input_group_count(8 * M * N * 2))]

    report = harness.Registry.run_correctness(names=target, M=M, K=K, N=N, seed=seed, groups=groups)

    table = report.console_table()
    print(table)

    if args.report:
        report.write_markdown(args.report, all_pairwise=False)
        print(f"\nReport -> {args.report}")


if __name__ == "__main__":
    main()
