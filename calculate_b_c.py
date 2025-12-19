#!/usr/bin/env python3
"""Compute McNemar contingency table (a/b/c/d) for two graded result files.

This is a minimal variant that assumes both inputs are single `.json` files with
aligned question ordering.

For a more flexible version (supports `.txt` lists of files and CSV output), see
`calculate_mcnemar.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import chi2


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute McNemar test values for two result JSON files")
    parser.add_argument("file1", type=Path)
    parser.add_argument("file2", type=Path)
    return parser


def load_results(filepath: Path) -> list[dict]:
    raw = json.loads(filepath.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected a list in {filepath}")
    return raw


def calculate_mcnemar_values(file1: Path, file2: Path) -> dict[str, int | str]:
    results1 = load_results(file1)
    results2 = load_results(file2)
    if len(results1) != len(results2):
        raise ValueError(f"result length mismatch: {len(results1)} vs {len(results2)}")

    a = b = c = d = 0
    for q1, q2 in zip(results1, results2):
        if q1.get("question_num") != q2.get("question_num"):
            raise ValueError("question_num mismatch; inputs are not aligned")

        m1 = str(q1.get("grade", "")).strip().upper() == "T"
        m2 = str(q2.get("grade", "")).strip().upper() == "T"

        if m1 and m2:
            a += 1
        elif m1 and not m2:
            b += 1
        elif (not m1) and m2:
            c += 1
        else:
            d += 1

    return {"model1": file1.stem, "model2": file2.stem, "a": a, "b": b, "c": c, "d": d, "total": a + b + c + d}


def calculate_mcnemar_statistic(b: int, c: int) -> float:
    if b + c == 0:
        return 0.0
    return ((abs(b - c) - 1) ** 2) / (b + c)


def main() -> int:
    args = build_argparser().parse_args()
    if not args.file1.exists():
        raise SystemExit(f"can't find {args.file1}")
    if not args.file2.exists():
        raise SystemExit(f"can't find {args.file2}")

    results = calculate_mcnemar_values(args.file1, args.file2)
    chi_sq = calculate_mcnemar_statistic(int(results["b"]), int(results["c"]))
    p_value = 1 - chi2.cdf(chi_sq, df=1)

    print(f"\ncomparing {results['model1']} vs {results['model2']}")
    print(f"both correct: {results['a']}")
    print(f"m1 right, m2 wrong: {results['b']}")
    print(f"m1 wrong, m2 right: {results['c']}")
    print(f"both wrong: {results['d']}")
    print(f"\nchi-squared: {chi_sq:.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        winner = results["model1"] if int(results["b"]) > int(results["c"]) else results["model2"]
        print(f"statistically significant (p<0.05) - {winner} is better")
    else:
        print("not statistically significant, basically the same")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
