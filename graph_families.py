"""Generate per-family accuracy charts from `results/*.log` files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def split_model_lang(stem: str) -> tuple[str, str] | None:
    """Split a log filename stem into (model, lang) for multiple conventions."""

    for lang in ("en", "no"):
        for sep in ("_", "-", ":"):
            suffix = f"{sep}{lang}"
            if stem.endswith(suffix) and len(stem) > len(suffix):
                return stem[: -len(suffix)], lang
    return None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a graph for each model family")
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("OUT_DIR", "results")))
    parser.add_argument("--graph-dir", type=Path, default=Path("family_graphs"))
    return parser


def parse_model_size(model_name: str) -> tuple[float, str]:
    """Return (numeric_size_in_B, label) extracted from model name."""
    if "llama" in model_name and "instruct" in model_name:
        return 71.0, "70B-instruct"
    if "mimir" in model_name:
        if "scratch" in model_name:
            return 7.0, "7B-core-scratch-instruct"
        return 8.0, "7B-core-instruct"

    match = re.search(r"(\d+\.?\d*)([bBmM])", model_name)
    if not match:
        return 0.0, model_name
    value_str, unit = match.groups()
    value = float(value_str)
    label = f"{value_str}{unit.upper()}"
    if unit.lower() == "m":
        return value / 1000.0, label
    return value, label


def main() -> int:
    args = build_argparser().parse_args()
    results_dir: Path = args.out_dir
    graph_dir: Path = args.graph_dir
    graph_dir.mkdir(parents=True, exist_ok=True)
    print(f"Graphs will be saved in '{graph_dir}/'")

    log_files = sorted(results_dir.glob("*.log"))
    families: dict[str, dict[str, list[tuple[float, float, str]]]] = defaultdict(lambda: defaultdict(list))
    print(f"Found {len(log_files)} log files to process...")

    for log_path in log_files:
        try:
            stem = log_path.stem
            parsed = split_model_lang(stem)
            if parsed is None:
                raise ValueError("invalid name for log file")
            model_name, lang = parsed

            family_match = re.match(r"([a-zA-Z0-9\-\.]+)", model_name)
            if not family_match:
                continue
            family_name = family_match.group(1)

            numeric_size, size_label = parse_model_size(model_name)
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
            accuracy = float(log_data.get("accuracy", 0.0))
            families[family_name][lang].append((numeric_size, accuracy, size_label))
        except Exception as exc:
            print(f"Warning: could not process file {log_path}: {exc}", file=sys.stderr)

    if not families:
        print("No model families found. Exiting.", file=sys.stderr)
        return 1

    for family_name, lang_data in families.items():
        print(f"Generating graph for family: {family_name}")

        all_models: dict[float, str] = {}
        for data_list in lang_data.values():
            for numeric_size, _, size_label in data_list:
                all_models.setdefault(numeric_size, size_label)

        sorted_models = sorted(all_models.items())
        if not sorted_models:
            print(f"  - Skipping {family_name}, no data to plot.")
            continue

        labels = [item[1] for item in sorted_models]
        en_scores_map = {size: acc for size, acc, _ in lang_data.get("en", [])}
        no_scores_map = {size: acc for size, acc, _ in lang_data.get("no", [])}
        en_scores = [en_scores_map.get(size, 0.0) * 100 for size, _ in sorted_models]
        no_scores = [no_scores_map.get(size, 0.0) * 100 for size, _ in sorted_models]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        rects1 = ax.bar(x - width / 2, en_scores, width, label="English Prompt", color="royalblue")
        rects2 = ax.bar(x + width / 2, no_scores, width, label="Norwegian Prompt", color="darkorange")

        ax.plot(x - width / 2, en_scores, marker="o", color="navy", linestyle="--")
        ax.plot(x + width / 2, no_scores, marker="o", color="saddlebrown", linestyle="--")

        ax.set_ylabel("Accuracy (%)", fontsize=15)
        ax.set_xlabel("Model Size (Parameters)", fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=19)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        ax.bar_label(rects1, padding=3, fmt="%.1f", fontsize=14)
        ax.bar_label(rects2, padding=3, fmt="%.1f", fontsize=14)

        graph_filename = graph_dir / f"{family_name}.png"
        plt.savefig(graph_filename)
        plt.close(fig)
        print(f"  - Saved graph to {graph_filename}")

    print("\nAll family graphs generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())