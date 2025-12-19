#!/usr/bin/env python3
"""Backfill `num_gpus` into result `.log` JSON files.

Some downstream analyses (e.g., GPU-hours Pareto plots) assume each `.log` file in
`results/` has a `num_gpus` field. Older runs may not include it.

This script updates `results/*.log` in-place, adding `num_gpus` when missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Heuristic GPU assignments for known large models.
GPU_COUNTS: dict[str, int] = {
    "deepseek-r1:671b": 3,
    "llama3.1:405b": 4,
    "qwen2.5:230b": 2,
    "qwen3:235b": 2,
    "gpt-oss:120b": 2,
}


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def update_log_files(results_dir: Path, *, default_num_gpus: int = 1, verbose: bool = False) -> int:
    """Update `.log` files under `results_dir`.

    Returns the number of files updated.
    """

    updated = 0
    for log_file in sorted(results_dir.glob("*.log")):
        try:
            raw = log_file.read_text(encoding="utf-8")
            if not raw.strip():
                if verbose:
                    print(f"skip empty file: {log_file.name}")
                continue

            data = json.loads(raw)
            if not isinstance(data, dict):
                if verbose:
                    print(f"skip non-object json: {log_file.name}")
                continue

            if "num_gpus" in data:
                continue

            # Extract model name (strip trailing "_en"/"_no" from stem).
            model_base = log_file.stem.rsplit("_", 1)[0]
            num_gpus = int(GPU_COUNTS.get(model_base, default_num_gpus))
            data["num_gpus"] = num_gpus

            _atomic_write_json(log_file, data)
            updated += 1

            if verbose or num_gpus > 1:
                print(f"updated {log_file.name}: num_gpus={num_gpus}")
        except Exception as exc:
            print(f"error processing {log_file.name}: {exc}")

    return updated


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill num_gpus into results/*.log JSON files")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing .log files")
    parser.add_argument("--default-num-gpus", type=int, default=1, help="Fallback GPU count for unknown models")
    parser.add_argument("--verbose", action="store_true", help="Print per-file decisions")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if not args.results_dir.exists():
        raise SystemExit(f"results dir not found: {args.results_dir}")

    updated = update_log_files(
        args.results_dir,
        default_num_gpus=args.default_num_gpus,
        verbose=bool(args.verbose),
    )
    print(f"updated {updated} log files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
