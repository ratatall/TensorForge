#!/usr/bin/env python3
"""Reproduce nine workloads; requires a Release build and Python 3, no packages."""
import argparse
import csv
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, default=Path("build/tensorforge"))
parser.add_argument("--output", type=Path, default=Path("results"))
parser.add_argument("--iterations", type=int, default=101)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = []
for size in (256, 16384, 262144):
    for name, expression in (
        ("single", "A + B"),
        ("chain", "relu((A + B) * B + A)"),
        ("broadcast", "relu(A * 2.0 + B)"),
    ):
        source = args.output / f"{name}_{size}.tf"
        source.write_text(f"input A: tensor<{size}>;\ninput B: tensor<{size}>;\nreturn {expression};\n")
        result = source.with_suffix(".csv")
        subprocess.run([str(args.binary.resolve()), "benchmark", str(source),
                        "--iterations", str(args.iterations), "--seed", str(args.seed),
                        "--csv", str(result)], check=True)
        with result.open(newline="") as stream:
            measured = list(csv.DictReader(stream))
            if len(measured) != 5 or any(row['verified'] != 'true' for row in measured):
                raise RuntimeError(f'Incomplete or unverified benchmark: {result}')
            rows.extend(measured)
with (args.output / "results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {args.output / 'results.csv'}")
